"""
Deterministic parser for Supernova Google Doc proven ad scripts.

Pure Python + regex + deep_translator (with offline/stdlib fallbacks).
Zero Gemini API calls.

Responsibilities:
1. Separate visual blueprint (`visual_blueprint`) cleanly from spoken dialogue,
   stripping out `Reference link : ...` and `Visual Frame...` footer noise.
2. Extract `native_script` and `english_script` (translating Indic scripts via
   `deep_translator.GoogleTranslator` when only regional text is present).
3. Partition `english_script` into the 4 canonical Supernova `story_blocks`:
   - `block_1_hook`
   - `block_2_pain_and_old_methods`
   - `block_3_live_fix_and_usps`
   - `block_4_desire_and_cta`
4. Compute `metadata_tags` (persona, setting, hook_angle, hook_word_count,
   avg_words_per_line, total_turns, ai_teacher_entry_turn, has_first_half_usps).
"""

from __future__ import annotations

import json
import math
import re
import ssl
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

try:
    from deep_translator import GoogleTranslator  # type: ignore
except ImportError:
    GoogleTranslator = None  # type: ignore

try:
    import certifi  # type: ignore
except ImportError:
    certifi = None  # type: ignore


# ---------------------------------------------------------------------------
# Unicode ranges for Indic scripts
# ---------------------------------------------------------------------------
INDIC_SCRIPT_RANGES: Dict[str, Tuple[int, int]] = {
    "Hindi": (0x0900, 0x097F),      # Devanagari (Hindi & Marathi)
    "Bengali": (0x0980, 0x09FF),    # Bengali & Assamese
    "Punjabi": (0x0A00, 0x0A7F),    # Gurmukhi
    "Gujarati": (0x0A80, 0x0AFF),   # Gujarati
    "Odia": (0x0B00, 0x0B7F),       # Odia
    "Tamil": (0x0B80, 0x0BFF),      # Tamil
    "Telugu": (0x0C00, 0x0C7F),     # Telugu
    "Kannada": (0x0C80, 0x0CFF),    # Kannada
    "Malayalam": (0x0D00, 0x0D7F),  # Malayalam
}

INDIC_CHAR_RE = re.compile(r"[\u0900-\u0D7F]")

# ---------------------------------------------------------------------------
# Noise & Header Regexes
# ---------------------------------------------------------------------------
NOISE_LINE_RE = re.compile(
    r"^\s*\*?\*?(?:"
    r"(?:visual\s+)?ref(?:erence)?(?:\s*links?)?\s*(?:[:\-–—].*)?$|"
    r"for\s+ref(?:erence)?\s*(?:[:\-–—].*)?$|"
    r"og(?:\s+ad)?\s*(?:[:\-–—].*)?$|"
    r"(?:youtube|notion)\s*links?\s*(?:[:\-–—].*)?$|"
    r"https?://\S+\s*$|"
    r"visual\s*frames?\b.*|"
    r"[-_=*]{3,}\s*$"
    r")",
    re.IGNORECASE,
)

INLINE_URL_OR_REF_RE = re.compile(
    r"(?:"
    r"\(\s*(?:(?:visual\s+)?ref(?:erence)?|for\s+reference|og(?:\s+ad)?)\s*[:\-–—]?\s*https?://[^)]+\)|"
    r"\[\s*(?:(?:visual\s+)?ref(?:erence)?|for\s+reference|og(?:\s+ad)?)\s*[:\-–—]?\s*https?://[^\]]+\]|"
    r"\b(?:(?:visual\s+)?ref(?:erence)?(?:\s*links?)?|for\s+reference|og\s+ad|og)\s*[:\-–—]\s*https?://\S+|"
    r"https?://(?:www\.)?(?:youtube\.com|youtu\.be|notion\.so|notion\.site)\S*|"
    r"https?://\S+"
    r")",
    re.IGNORECASE,
)

TRAILING_EMPTY_REF_LABEL_RE = re.compile(
    r"(?:^|\s+)(?:(?:visual\s+)?ref(?:erence)?(?:\s*links?)?|for\s+reference|og\s+ad|og)\s*[:\-–—]?\s*$",
    re.IGNORECASE,
)

VISUAL_TOP_HEADER_RE = re.compile(
    r"^\s*\*?\*?(?:visual\s*description|visual\s*brief|visual\s*notes?|visual\s*setup|visuals?)"
    r"(?:\s*\([^)]*\))?\*?\*?(?:\s*[:\-–—]\s*(.*)|\s*)$",
    re.IGNORECASE,
)

SPLIT_SCREEN_HEADER_RE = re.compile(
    r"^\s*\*?\*?(?:split[\s\-]*screen(?:\s*(?:&|and)\s*characters?)?|screen\s*layout)"
    r"(?:\s*\([^)]*\))?\*?\*?(?:\s*[:\-–—]\s*(.*)|\s*)$",
    re.IGNORECASE,
)

HOOK_VISUALS_HEADER_RE = re.compile(
    r"^\s*\*?\*?(?:hook\s*visuals?|opening\s*visuals?)"
    r"(?:\s*\([^)]*\))?\*?\*?(?:\s*[:\-–—]\s*(.*)|\s*)$",
    re.IGNORECASE,
)

BODY_VISUALS_HEADER_RE = re.compile(
    r"^\s*\*?\*?(?:body\s*visuals?|main\s*visuals?)"
    r"(?:\s*\([^)]*\))?\*?\*?(?:\s*[:\-–—]\s*(.*)|\s*)$",
    re.IGNORECASE,
)

VIDEO_FLOW_HEADER_RE = re.compile(
    r"^\s*\*?\*?(?:video\s*flow|visual\s*flow|flow\s*of\s*video)"
    r"(?:\s*\([^)]*\))?\*?\*?(?:\s*[:\-–—]\s*(.*)|\s*)$",
    re.IGNORECASE,
)

CHARACTERS_HEADER_RE = re.compile(
    r"^\s*\*?\*?characters?(?:\s*description)?"
    r"(?:\s*\([^)]*\))?\*?\*?(?:\s*[:\-–—]\s*(.*)|\s*)$",
    re.IGNORECASE,
)

SCENE_HEADER_RE = re.compile(
    r"^\s*\*?\*?\[?\(?(?:scene\s*(?:\d+|one|two|three|four|five|"
    r"changes?|shifts?|transition|cuts?)\b[^\n]*?)\]?\)?\*?\*?\s*$",
    re.IGNORECASE,
)

SCRIPT_SECTION_HEADER_RE = re.compile(
    r"^\s*\*?\*?\[?\(?(?:"
    r"(?:english|tamil|telugu|malayalam|kannada|hindi|bengali|assamese|marathi|gujarati|punjabi|odia|oriya|regional|native|vernacular|spoken|audio|final|full)\s+"
    r"(?:script|dialogue|translation|version|voiceover|lines?)|"
    r"script\s*(?:in\s+(?:english|tamil|telugu|malayalam|kannada|hindi|bengali|assamese|marathi|gujarati|punjabi|odia|oriya))?|"
    r"dialogue|spoken\s*lines?|voice\s*over"
    r")\]?\)?\*?\*?\s*:?\s*$",
    re.IGNORECASE,
)

CHARACTER_BULLET_RE = re.compile(
    r"^\s*(?:[•\*\-\u2022\u25aa\u25cf\u00b7\u2023\u2043]|\d+[\.\)])\s*"
    r"([^:\-\u2013\u2014\n]{1,60}?)\s*(?:--|—|–| - |:\s+)\s*(.+)$"
)

CHARACTER_DOUBLE_DASH_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9\s/\(\)\.\-]{1,50}?)\s*(?:--|—|–)\s*(.+)$"
)

SPEAKER_LINE_RE = re.compile(
    r"^\s*(?:[•\*\-\u2022]\s*)?"
    r"\*?\*?\[?([A-Za-z\u0900-\u0D7F]"
    r"[^:\n\]]{0,55}?)\]?\*?\*?\s*:\s*(.*)$"
)

DASH_SPEAKER_LINE_RE = re.compile(
    r"^\s*\*?\*?([A-Za-z\u0900-\u0D7F][^:\n]{0,45}?)\*?\*?"
    r"(?:\s*(?:--|—|–)\s*|\s+-\s+)(.+)$"
)

KNOWN_DASH_SPEAKER_RE = re.compile(
    r"^(?:"
    r"(?:(?:young|old|middle[\s\-]*aged|indian|same|other|another|long[\s\-]*haired|short[\s\-]*haired|"
    r"first|second|third|1st|2nd|3rd|corporate|office|college|school|working|homemaker|housewife)\s+)*"
    r"(?:"
    r"lady|woman|women|girl|boy|man|men|guy|person|people|interviewer|interviewee|anchor|host|presenter|"
    r"reporter|air\s*hostess|flight\s*attendant|waiter|waitress|server|doctor|nurse|patient|"
    r"mother|mom|mummy|amma|father|dad|papa|appa|parent|son|daughter|kid|child|"
    r"student|learner|user|speaker|human|customer|client|passenger|stranger|friend|colleague|"
    r"boss|manager|hr|recruiter|employee|candidate|wife|husband|"
    r"female\s+humanoid\s+robot|female\s+robot|humanoid\s+robot|robot\s+ai|ai\s+robot|robot|"
    r"miss\s+nova|ms\.?\s*nova|nova|supernova\s+ai|supernova\s+teacher|supernova|"
    r"ai\s+teacher|ai\s+tutor|ai\s+coach|ai\s+bot|ai|bot"
    r")(?:\s*#?\d+|\s+[a-z])?(?:\s*\([^)]*\))?"
    r"|"
    r"[\u0900-\u0D7F][\u0900-\u0D7F\s\.\d]{0,30}"
    r")$",
    re.IGNORECASE,
)

NON_SPEAKER_LABEL_RE = re.compile(
    r"^\s*(?:"
    r"visual(?:s|\s+description|\s+brief|\s+notes?|\s+setup|\s+ref(?:erence)?|\s+flow|\s+frames?)?|"
    r"split[\s\-]*screen(?:\s*(?:&|and)\s*characters?)?|screen\s*layout|"
    r"hook\s*visuals?|opening\s*visuals?|body\s*visuals?|main\s*visuals?|"
    r"video\s*flow|flow\s*of\s*video|characters?(?:\s*description)?|"
    r"scene(?:\s*\d+|\s+one|\s+two|\s+three|\s+four|\s+changes?|\s+shifts?|\s+transition)?|"
    r"stage\s*directions?|actions?|camera|shots?|settings?|locations?|backgrounds?|"
    r"sfx|music|bgm|sound\s*effects?|(?:on[\s\-]*screen\s+)?text(?:\s+on\s+screen)?|supers?|"
    r"ref(?:erence)?(?:\s*links?)?|for\s+ref(?:erence)?|og(?:\s+ad)?|(?:youtube|notion)\s*links?|"
    r"on\s+the\s+(?:left|right|top|bottom)|left\s+side|right\s+side|top\s+half|bottom\s+half"
    r")\s*$",
    re.IGNORECASE,
)

AI_SPEAKER_RE = re.compile(
    r"\b(?:"
    r"female\s+humanoid\s+robot|female\s+robot|humanoid\s+robot|robot\s+ai|ai\s+robot|robot|"
    r"miss\s+nova|ms\.?\s*nova|nova|supernova\s+ai|supernova\s+teacher|"
    r"ai\s+teacher|ai\s+tutor|ai\s+coach|ai\s+bot|ai\b|bot\b"
    r")|"
    r"(?:ரோபோ|மிஸ்\s*நோவா|நோவா|रोबोट|मिस\s*नोवा|రోబో|మిస్\s*నోవా|റോബോട്ട്|ರೋಬೋಟ್|"
    r"রোবট|মিস\s*নোভা|ਰੋਬੋਟ|રોબોટ|ରୋବୋଟ୍)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Supernova Core USP & Story Block Patterns
# ---------------------------------------------------------------------------
USP_PATTERNS: Dict[str, re.Pattern[str]] = {
    "daily_15_20_mins": re.compile(
        r"\b(?:15\s*(?:to|\-|–|or)\s*20\s*(?:minutes?|mins?)|"
        r"(?:15|20)\s*(?:minutes?|mins?)\s*(?:a\s*day|daily|every\s*day|practice)|"
        r"just\s*(?:15|20)\s*(?:minutes?|mins?))\b",
        re.IGNORECASE,
    ),
    "within_30_days": re.compile(
        r"\b(?:within\s*30\s*days|in\s*(?:just\s*)?30\s*days|30\s*days|"
        r"within\s*(?:a|one|1)\s*month)\b",
        re.IGNORECASE,
    ),
    "improve_spoken_english": re.compile(
        r"\b(?:improve\s+(?:your\s+|their\s+)?spoken\s+english|"
        r"speak\s+(?:fluent\s+)?english\s+(?:fluently|confidently|without\s+fear|easily)|"
        r"fluent\s+spoken\s+english|spoken\s+english\s+fluency|"
        r"learn\s+spoken\s+english)\b",
        re.IGNORECASE,
    ),
    "social_proof_1_crore": re.compile(
        r"\b(?:1\s*crore|one\s*crore|10\s*million|crore\s*\+?\s*indians?|"
        r"crores?\s+of\s+(?:indians?|people|learners?|users?)|"
        r"lakhs?\s+of\s+(?:indians?|people|learners?|mothers?|students?))\b",
        re.IGNORECASE,
    ),
    "through_native_language": re.compile(
        r"\b(?:through\s+(?:tamil|telugu|malayalam|kannada|hindi|bengali|assamese|marathi|gujarati|punjabi|odia|oriya|your\s+(?:own\s+)?(?:mother\s+tongue|native\s+language|language))(?:\s+itself)?|"
        r"in\s+(?:tamil|telugu|malayalam|kannada|hindi|bengali|assamese|marathi|gujarati|punjabi|odia|oriya)\s+itself|"
        r"(?:tamil|telugu|malayalam|kannada|hindi|bengali|assamese|marathi|gujarati|punjabi|odia|oriya)\s+to\s+english|"
        r"via\s+(?:tamil|telugu|malayalam|kannada|hindi|bengali|assamese|marathi|gujarati|punjabi|odia|oriya)|"
        r"using\s+(?:tamil|telugu|malayalam|kannada|hindi|bengali|assamese|marathi|gujarati|punjabi|odia|oriya))\b",
        re.IGNORECASE,
    ),
}

SUPERNOVA_MENTION_RE = re.compile(
    r"\b(?:supernova|miss\s+nova|ai\s+english\s+teacher|personal\s+ai\s+teacher|"
    r"24\s*/\s*7|anytime\s+anywhere)\b",
    re.IGNORECASE,
)

LIVE_PRACTICE_CUES_RE = re.compile(
    r"\b(?:repeat\s+after\s+me|say\s+it\s+with\s+me|try\s+saying|now\s+you\s+say|"
    r"say\s+this\s+now|say\s*:\s*|practice\s+with\s+me|let'?s\s+practice)\b",
    re.IGNORECASE,
)

OLD_METHODS_RE = re.compile(
    r"\b(?:youtube\s+videos?|grammar\s+books?|grammar\s+notes?|coaching\s+class(?:es)?|"
    r"spoken\s+english\s+class(?:es)?|bought\s+books?)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Language Detection & Translation Helpers
# ---------------------------------------------------------------------------
_SSL_CONTEXT: Optional[ssl.SSLContext] = None


def _get_ssl_context() -> ssl.SSLContext:
    """Return a cached SSLContext using certifi CA bundle first, falling back safely."""
    global _SSL_CONTEXT
    if _SSL_CONTEXT is not None:
        return _SSL_CONTEXT
    if certifi is not None:
        try:
            _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
            return _SSL_CONTEXT
        except Exception:
            pass
    try:
        _SSL_CONTEXT = ssl.create_default_context()
        return _SSL_CONTEXT
    except Exception:
        _SSL_CONTEXT = ssl._create_unverified_context()
        return _SSL_CONTEXT


def contains_indic_script(text: str) -> bool:
    """Return True if `text` contains any Indian script Unicode characters (\u0900-\u0D7F)."""
    return bool(INDIC_CHAR_RE.search(text or ""))


def detect_indic_language(text: str, fallback: str = "Tamil") -> Tuple[bool, str]:
    """
    Detect whether `text` contains Indic Unicode characters and return
    `(has_indic, detected_language_name)`.
    """
    if not text:
        return False, fallback

    counts: Dict[str, int] = {lang: 0 for lang in INDIC_SCRIPT_RANGES}
    for ch in text:
        code = ord(ch)
        for lang, (low, high) in INDIC_SCRIPT_RANGES.items():
            if low <= code <= high:
                counts[lang] += 1

    best_lang, best_count = max(counts.items(), key=lambda kv: kv[1])
    if best_count > 0:
        fb_clean = (fallback or "").strip().lower()
        if best_lang == "Hindi" and fb_clean == "marathi":
            return True, "Marathi"
        if best_lang == "Bengali" and fb_clean == "assamese":
            return True, "Assamese"
        return True, best_lang
    return False, fallback


def _fetch_url_bytes_with_ssl(req: urllib.request.Request, timeout: int = 8) -> bytes:
    """Fetch URL bytes using certifi SSL context first, falling back to unverified SSL context."""
    try:
        ctx = _get_ssl_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return resp.read()
    except urllib.error.HTTPError:
        raise
    except Exception:
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return resp.read()


def _stdlib_google_translate(text: str, source_lang: str = "auto", target_lang: str = "en") -> Optional[str]:
    """
    Lightweight stdlib Google Translate helper (zero API keys) with multi-endpoint
    failover (`client=at`, `clients5 dict-chrome-ex`, `client=gtx`) and certifi SSL context.
    """
    if not text.strip():
        return text

    single_endpoints = [
        (
            "https://translate.google.com/translate_a/single",
            "at",
            "GoogleTranslate/6.29.59279 (Linux; U; Android 11; Pixel 5)",
        ),
        (
            "https://translate.googleapis.com/translate_a/single",
            "gtx",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        ),
    ]
    for base_url, client_id, ua in single_endpoints:
        try:
            params = urllib.parse.urlencode(
                {
                    "client": client_id,
                    "sl": source_lang,
                    "tl": target_lang,
                    "dt": "t",
                    "q": text,
                }
            )
            req = urllib.request.Request(f"{base_url}?{params}", headers={"User-Agent": ua})
            raw_bytes = _fetch_url_bytes_with_ssl(req, timeout=8)
            payload = json.loads(raw_bytes.decode("utf-8", errors="replace"))
            if isinstance(payload, list) and payload and isinstance(payload[0], list):
                translated_chunks = [
                    seg[0] for seg in payload[0] if isinstance(seg, list) and seg and seg[0]
                ]
                if translated_chunks:
                    joined = "".join(translated_chunks).strip()
                    if joined:
                        return joined
        except Exception:
            continue

    # Fallback 3: clients5 dict-chrome-ex endpoint
    try:
        params = urllib.parse.urlencode(
            {
                "client": "dict-chrome-ex",
                "sl": source_lang,
                "tl": target_lang,
                "q": text,
            }
        )
        req = urllib.request.Request(
            f"https://clients5.google.com/translate_a/t?{params}",
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
        )
        raw_bytes = _fetch_url_bytes_with_ssl(req, timeout=8)
        payload = json.loads(raw_bytes.decode("utf-8", errors="replace"))
        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, list) and first and isinstance(first[0], str):
                return first[0].strip()
            if isinstance(first, str):
                return first.strip()
    except Exception:
        pass

    return None


def translate_to_english(text: str, source_lang: str = "auto") -> str:
    """
    Translate non-English text to English using fast certifi-backed stdlib endpoints
    and `deep_translator.GoogleTranslator` with offline fallback. Zero Gemini API calls.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    if not contains_indic_script(cleaned):
        return cleaned

    # 1. Primary: fast stdlib Google Translate endpoints (certifi SSL + multi-client failover)
    fallback_translated = _stdlib_google_translate(cleaned, source_lang=source_lang, target_lang="en")
    if fallback_translated and not contains_indic_script(fallback_translated):
        return fallback_translated

    # 2. Secondary: deep_translator.GoogleTranslator
    if GoogleTranslator is not None:
        try:
            translated = GoogleTranslator(source=source_lang, target="en").translate(cleaned)
            if (
                translated
                and isinstance(translated, str)
                and translated.strip()
                and not contains_indic_script(translated)
            ):
                return translated.strip()
        except Exception:
            pass

    if fallback_translated:
        return fallback_translated

    # 3. Offline fallback: return original text safely without crashing
    return cleaned


def _batch_translate_turns(turns: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Translate a list of dialogue turns (some or all in Indic script) into English
    while preserving speaker names and turn boundaries. Pure-ASCII turns are kept
    as-is; Indic turns are batch-translated first for speed, falling back per-turn.
    """
    if not turns:
        return []

    indic_indices = [
        idx for idx, t in enumerate(turns) if contains_indic_script(t.get("text") or "")
    ]
    translated_by_idx: Dict[int, str] = {}

    if indic_indices:
        join_sep = "\n|||\n"
        indic_texts = [turns[idx]["text"] for idx in indic_indices]
        combined = join_sep.join(indic_texts)

        if len(combined) < 4200:
            batch_out = translate_to_english(combined, source_lang="auto")
            parts = [p.strip() for p in re.split(r"\s*\|\|\|\s*", batch_out)]
            if len(parts) == len(indic_indices) and all(parts) and not contains_indic_script(batch_out):
                for idx, trans in zip(indic_indices, parts):
                    translated_by_idx[idx] = trans

        if len(translated_by_idx) != len(indic_indices):
            for idx in indic_indices:
                translated_by_idx[idx] = translate_to_english(turns[idx]["text"], source_lang="auto")

    result: List[Dict[str, str]] = []
    for idx, orig in enumerate(turns):
        speaker = orig["speaker"]
        if contains_indic_script(speaker):
            speaker = translate_to_english(speaker, source_lang="auto")
        eng_text = translated_by_idx.get(idx, orig["text"])
        result.append(
            {
                "speaker": speaker.strip() or "Speaker",
                "text": eng_text.strip(),
                "native_text": orig["text"].strip(),
            }
        )
    return result


# ---------------------------------------------------------------------------
# Step 1: Visual Blueprint Extraction
# ---------------------------------------------------------------------------
def _strip_links_and_ref_tags(text: str) -> str:
    """
    Strip all YouTube, youtu.be, Notion, and reference links/prefixes
    (`Ref`, `Ref:`, `Visual Ref -`, `For reference:`, `Visual reference:`, `OG AD:`, `OG:`)
    from any visual or dialogue line.
    """
    if not text:
        return ""
    cleaned = INLINE_URL_OR_REF_RE.sub("", text)
    cleaned = TRAILING_EMPTY_REF_LABEL_RE.sub("", cleaned)
    cleaned = re.sub(r"\(\s*\)|\[\s*\]", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def _is_stage_direction_line(stripped: str) -> bool:
    """
    Return True if `stripped` is a parenthetical or bracketed stage direction line, e.g.:
    - `(The long-haired girl will flip her hair...)`
    - `(Vertical Split screen. On the left - Female Robot. On the right- same girl at home.)`
    - `[The Air Hostess takes him to her private crew cabin.]`
    """
    core = stripped.strip("*").strip()
    if len(core) < 4:
        return False
    if core.startswith("(") and core.endswith(")"):
        return not bool(SCRIPT_SECTION_HEADER_RE.match(core))
    if core.startswith("[") and core.endswith("]") and "]:" not in core:
        return not bool(SCRIPT_SECTION_HEADER_RE.match(core))
    return False


def _mentions_split_screen(text: str) -> bool:
    """Return True if `text` describes a split-screen layout."""
    return bool(
        re.search(
            r"split[\s\-]*screen|on\s+the\s+left\b.*\bon\s+the\s+right\b|"
            r"top[\s/\-&]*bottom|side[\s\-]*by[\s\-]*side",
            text,
            re.IGNORECASE,
        )
    )


def _match_speaker_line(stripped: str) -> Optional[Tuple[str, str]]:
    """
    Match either:
    1. Standard `Speaker: dialogue` line (`SPEAKER_LINE_RE`), or
    2. Vox-pop / dash-separated `Speaker - dialogue` / `Speaker -- dialogue` line
       (`DASH_SPEAKER_LINE_RE`) when the left side is a known speaker word.
    Returns `(speaker, spoken_text)` or `None`.
    """
    m_colon = SPEAKER_LINE_RE.match(stripped)
    if m_colon:
        cand_speaker = m_colon.group(1).strip().strip("*[]").strip()
        cand_text = m_colon.group(2).strip()
        if (
            cand_speaker
            and len(cand_speaker.split()) <= 6
            and not NON_SPEAKER_LABEL_RE.match(cand_speaker)
            and not cand_speaker.lower().startswith(("http", "www.", "scene ", "visual"))
        ):
            return cand_speaker, cand_text

    if not stripped.startswith(("•", "*", "\u2022", "(", "[")):
        m_dash = DASH_SPEAKER_LINE_RE.match(stripped)
        if m_dash:
            cand_speaker = m_dash.group(1).strip().strip("*[]").strip()
            cand_text = m_dash.group(2).strip()
            if (
                cand_speaker
                and KNOWN_DASH_SPEAKER_RE.match(cand_speaker)
                and not NON_SPEAKER_LABEL_RE.match(cand_speaker)
            ):
                return cand_speaker, cand_text

    return None


def _looks_like_character_description(
    speaker: str, rest: str, in_characters_block: bool, dialogue_started: bool
) -> bool:
    """
    Heuristic to distinguish a character description line from spoken dialogue
    when a colon or dash is used before dialogue begins.
    """
    if dialogue_started:
        return False
    lower = rest.lower()
    if rest.strip().startswith(('"', "“", "'")):
        return False
    desc_markers = (
        "year-old", "years old", "yr old", "yo ", "homemaker", "housewife",
        "mother", "father", "professional", "student", "wearing", "sitting",
        "standing", "dressed", "robot", "ai teacher", "studio", "background",
        "looking", "middle-aged", "young", "indian", "friendly", "futuristic",
    )
    if in_characters_block and any(m in lower for m in desc_markers):
        return True
    strong_bio_markers = (
        "year-old", "years old", "yr old", "wearing a", "dressed in",
        "humanoid robot", "futuristic studio", "homemaker mother",
    )
    if any(m in lower for m in strong_bio_markers):
        return True
    return False


def extract_visual_blueprint(raw_text: str) -> Tuple[Dict[str, str], List[Tuple[Optional[str], str]]]:
    """
    Step 1: Separate `visual_blueprint` from spoken dialogue lines.

    Strips `Reference link : ...`, `Visual Ref - ...`, `For reference:`, `OG AD:`,
    YouTube/Notion URLs, and `Visual Frame...` footer labels.
    Routes `Visual:`, `Visuals -`, `Scene 1`, `Scene 2 — Split-Screen...`,
    `Scene changes - ...`, and parenthetical/bracketed stage directions `( ... )` / `[ ... ]`
    to `visual_blueprint` rather than spoken dialogue turns.
    """
    normalized = (raw_text or "").replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = normalized.split("\n")

    split_screen_parts: List[str] = []
    hook_visual_parts: List[str] = []
    body_visual_parts: List[str] = []
    video_flow_parts: List[str] = []
    character_parts: List[str] = []
    raw_visual_lines: List[str] = []

    remaining_lines: List[Tuple[Optional[str], str]] = []

    current_visual_bucket: Optional[str] = None
    in_characters_block = False
    dialogue_started = False
    current_script_section: Optional[str] = None

    for line in raw_lines:
        raw_stripped = line.strip()
        if not raw_stripped:
            if dialogue_started:
                current_visual_bucket = None
            continue

        # 1. Strip Reference / YouTube / Notion links & Visual Frame footers completely
        if NOISE_LINE_RE.match(raw_stripped):
            continue
        stripped = _strip_links_and_ref_tags(raw_stripped)
        if not stripped or NOISE_LINE_RE.match(stripped):
            continue

        # 2. Check explicit Script/Language section headers
        if SCRIPT_SECTION_HEADER_RE.match(stripped):
            lower_hdr = stripped.lower()
            if "english" in lower_hdr:
                current_script_section = "english"
            elif any(
                lang in lower_hdr
                for lang in (
                    "tamil", "telugu", "malayalam", "kannada", "hindi",
                    "bengali", "assamese", "marathi", "gujarati", "punjabi", "odia", "oriya",
                    "regional", "native", "vernacular",
                )
            ):
                current_script_section = "native"
            current_visual_bucket = None
            in_characters_block = False
            dialogue_started = True
            continue

        # 3. Check `Scene ...` headers and scene transition lines (before OR during dialogue)
        if SCENE_HEADER_RE.match(stripped):
            raw_visual_lines.append(stripped)
            video_flow_parts.append(stripped)
            if _mentions_split_screen(stripped):
                split_screen_parts.append(stripped)
            if not dialogue_started or re.search(r"\bscene\s*(?:1|one)\b", stripped, re.I):
                if re.search(r"[:\-–—]", stripped):
                    hook_visual_parts.append(stripped)
                if not dialogue_started:
                    current_visual_bucket = "video_flow"
            else:
                body_visual_parts.append(stripped)
            in_characters_block = False
            continue

        # 4. Check parenthetical / bracketed stage directions `( ... )` or `[ ... ]`
        if _is_stage_direction_line(stripped):
            clean_stage = stripped.strip("*").strip()
            if (clean_stage.startswith("(") and clean_stage.endswith(")")) or (
                clean_stage.startswith("[") and clean_stage.endswith("]")
            ):
                inner_stage = clean_stage[1:-1].strip()
            else:
                inner_stage = clean_stage
            raw_visual_lines.append(stripped)
            video_flow_parts.append(inner_stage)
            if _mentions_split_screen(inner_stage):
                split_screen_parts.append(inner_stage)
            if not dialogue_started:
                hook_visual_parts.append(inner_stage)
            else:
                body_visual_parts.append(inner_stage)
            continue

        # 5. Check Visual Description top-level header (`Visual:`, `Visuals -`, `Visual Description:`)
        m_top = VISUAL_TOP_HEADER_RE.match(stripped)
        if m_top:
            raw_visual_lines.append(stripped)
            inline_val = (m_top.group(1) or "").strip()
            current_visual_bucket = "body_visuals" if dialogue_started else "general_visual"
            in_characters_block = False
            if inline_val:
                if _mentions_split_screen(inline_val) or not dialogue_started:
                    split_screen_parts.append(inline_val)
                if not dialogue_started:
                    hook_visual_parts.append(inline_val)
                else:
                    body_visual_parts.append(inline_val)
                video_flow_parts.append(inline_val)
            continue

        # 6. Check Split Screen & Characters header
        m_split = SPLIT_SCREEN_HEADER_RE.match(stripped)
        if m_split:
            raw_visual_lines.append(stripped)
            inline_val = (m_split.group(1) or "").strip()
            current_visual_bucket = "split_screen"
            in_characters_block = "character" in stripped.lower()
            if inline_val:
                split_screen_parts.append(inline_val)
            continue

        # 7. Check Hook visuals header
        m_hook = HOOK_VISUALS_HEADER_RE.match(stripped)
        if m_hook:
            raw_visual_lines.append(stripped)
            inline_val = (m_hook.group(1) or "").strip()
            current_visual_bucket = "hook_visuals"
            in_characters_block = False
            if inline_val:
                hook_visual_parts.append(inline_val)
            continue

        # 8. Check Body visuals header
        m_body = BODY_VISUALS_HEADER_RE.match(stripped)
        if m_body:
            raw_visual_lines.append(stripped)
            inline_val = (m_body.group(1) or "").strip()
            current_visual_bucket = "body_visuals"
            in_characters_block = False
            if inline_val:
                body_visual_parts.append(inline_val)
            continue

        # 9. Check Video flow header
        m_flow = VIDEO_FLOW_HEADER_RE.match(stripped)
        if m_flow:
            raw_visual_lines.append(stripped)
            inline_val = (m_flow.group(1) or "").strip()
            current_visual_bucket = "video_flow"
            in_characters_block = False
            if inline_val:
                video_flow_parts.append(inline_val)
            continue

        # 10. Check Characters header
        m_chars = CHARACTERS_HEADER_RE.match(stripped)
        if m_chars and not dialogue_started:
            raw_visual_lines.append(stripped)
            inline_val = (m_chars.group(1) or "").strip()
            current_visual_bucket = "characters"
            in_characters_block = True
            if inline_val:
                character_parts.append(inline_val)
            continue

        # 11. Check if this line is a `Speaker: dialogue` or `Speaker - dialogue` line
        matched_speaker = _match_speaker_line(stripped)
        if matched_speaker is not None:
            cand_speaker, cand_text = matched_speaker
            if _looks_like_character_description(
                cand_speaker, cand_text, in_characters_block, dialogue_started
            ):
                character_parts.append(stripped)
                raw_visual_lines.append(stripped)
                continue
            dialogue_started = True
            current_visual_bucket = None
            in_characters_block = False
            remaining_lines.append((current_script_section, stripped))
            continue

        # 12. Check Character bullet lines (`• Name -- description`) before dialogue starts
        if not dialogue_started:
            m_char_double = CHARACTER_DOUBLE_DASH_RE.match(stripped)
            m_char_bullet = CHARACTER_BULLET_RE.match(stripped)
            if m_char_double or (
                m_char_bullet
                and (
                    "--" in stripped
                    or "—" in stripped
                    or "–" in stripped
                    or in_characters_block
                    or current_visual_bucket in ("characters", "split_screen")
                )
            ):
                character_parts.append(stripped)
                raw_visual_lines.append(stripped)
                continue

        # 13. If we are inside an active visual bucket, decide whether this non-speaker line
        # continues the visual description or starts dialogue
        if current_visual_bucket is not None:
            if contains_indic_script(stripped) or stripped.startswith(('"', "“", "‘", "'")):
                dialogue_started = True
                current_visual_bucket = None
                in_characters_block = False
                remaining_lines.append((current_script_section, stripped))
                continue

            if current_visual_bucket == "split_screen":
                if stripped.startswith(("•", "*", "-", "\u2022")):
                    character_parts.append(stripped)
                else:
                    split_screen_parts.append(stripped)
                raw_visual_lines.append(stripped)
                continue
            elif current_visual_bucket == "hook_visuals":
                hook_visual_parts.append(stripped)
                raw_visual_lines.append(stripped)
                continue
            elif current_visual_bucket == "body_visuals":
                body_visual_parts.append(stripped)
                if _mentions_split_screen(stripped):
                    split_screen_parts.append(stripped)
                raw_visual_lines.append(stripped)
                continue
            elif current_visual_bucket == "video_flow":
                video_flow_parts.append(stripped)
                if _mentions_split_screen(stripped):
                    split_screen_parts.append(stripped)
                raw_visual_lines.append(stripped)
                continue
            elif current_visual_bucket == "characters":
                if stripped.startswith(("•", "*", "-", "\u2022")) or "--" in stripped or "—" in stripped:
                    character_parts.append(stripped)
                    raw_visual_lines.append(stripped)
                    continue
                else:
                    dialogue_started = True
                    current_visual_bucket = None
                    in_characters_block = False
                    remaining_lines.append((current_script_section, stripped))
                    continue
            elif current_visual_bucket == "general_visual":
                split_screen_parts.append(stripped)
                if not hook_visual_parts:
                    hook_visual_parts.append(stripped)
                video_flow_parts.append(stripped)
                raw_visual_lines.append(stripped)
                continue

        # Otherwise it's a spoken dialogue line
        dialogue_started = True
        remaining_lines.append((current_script_section, stripped))

    split_screen_type = " ".join(split_screen_parts).strip()
    if not split_screen_type and raw_visual_lines:
        joined_vis = " ".join(raw_visual_lines)
        if re.search(r"top[\s/\-&]*bottom|vertical\s+split|upper[\s/\-&]*lower", joined_vis, re.I):
            split_screen_type = "Top-Bottom Split Screen"
        elif re.search(r"left[\s/\-&]*right|side[\s\-]*by[\s\-]*side", joined_vis, re.I):
            split_screen_type = "Side-by-Side Split Screen"
        elif re.search(r"split[\s\-]*screen", joined_vis, re.I):
            split_screen_type = "Split Screen"

    visual_blueprint = {
        "split_screen_type": split_screen_type,
        "hook_visuals": " ".join(hook_visual_parts).strip(),
        "body_visuals": " ".join(body_visual_parts).strip(),
        "video_flow": " ".join(video_flow_parts).strip(),
        "characters_description": "\n".join(character_parts).strip(),
        "raw_visual_text": "\n".join(raw_visual_lines).strip(),
    }
    return visual_blueprint, remaining_lines


# ---------------------------------------------------------------------------
# Step 2: Dialogue Turn Extraction & Bilingual / Indic Handling
# ---------------------------------------------------------------------------
def _clean_dialogue_text(text: str) -> str:
    """Clean wrapping quotes, inline reference URLs, and redundant whitespace from a spoken turn."""
    cleaned = _strip_links_and_ref_tags(text or "").strip()
    if len(cleaned) >= 2 and (
        (cleaned[0] == '"' and cleaned[-1] == '"')
        or (cleaned[0] == "“" and cleaned[-1] == "”")
    ):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _parse_dialogue_lines(
    section_lines: List[Tuple[Optional[str], str]]
) -> List[Dict[str, Optional[str]]]:
    """
    Convert raw post-visual script lines into structured turn dicts:
    `{"section": section_hint, "speaker": speaker, "text": text}`.
    Supports `Speaker: text`, `Speaker - text` / `Speaker -- text`,
    multi-line speaker blocks, and paragraph-based scripts.
    """
    turns: List[Dict[str, Optional[str]]] = []
    pending_speaker: Optional[str] = None
    pending_section: Optional[str] = None

    for sec_hint, line in section_lines:
        raw_stripped = line.strip()
        if not raw_stripped or NOISE_LINE_RE.match(raw_stripped):
            continue
        stripped = _strip_links_and_ref_tags(raw_stripped)
        if not stripped or NOISE_LINE_RE.match(stripped):
            continue
        if SCENE_HEADER_RE.match(stripped) or _is_stage_direction_line(stripped):
            continue

        matched_speaker = _match_speaker_line(stripped)
        if matched_speaker is not None:
            speaker, raw_spoken = matched_speaker
            spoken = _clean_dialogue_text(raw_spoken)
            if len(speaker.split()) <= 6:
                if spoken:
                    turns.append(
                        {
                            "section": sec_hint,
                            "speaker": speaker,
                            "text": spoken,
                        }
                    )
                    pending_speaker = None
                else:
                    pending_speaker = speaker
                    pending_section = sec_hint
                continue

        if pending_speaker is not None:
            cleaned_line = _clean_dialogue_text(stripped)
            if cleaned_line:
                turns.append(
                    {
                        "section": pending_section or sec_hint,
                        "speaker": pending_speaker,
                        "text": cleaned_line,
                    }
                )
            pending_speaker = None
            pending_section = None
        else:
            cleaned_line = _clean_dialogue_text(stripped)
            if cleaned_line:
                turns.append(
                    {
                        "section": sec_hint,
                        "speaker": "",
                        "text": cleaned_line,
                    }
                )

    # Assign inferred speakers if any turns had empty speaker labels
    for idx, t in enumerate(turns):
        if not t["speaker"]:
            text_val = t["text"] or ""
            if re.search(
                r"\b(?:stop|hold\s+on|wait|don'?t\s+say|with\s+supernova|supernova\s+ai|"
                r"install\s+now|15\s+to\s+20\s+minutes|within\s+30\s+days|repeat\s+after\s+me)\b",
                text_val,
                re.I,
            ):
                t["speaker"] = "Female Robot"
            else:
                t["speaker"] = "Female Robot" if idx % 2 == 1 else "Speaker"

    return turns


def _format_turns_as_script(turns: List[Dict[str, Any]], use_native: bool = False) -> str:
    """Format a list of turns into `Speaker: text` lines joined by newline."""
    lines: List[str] = []
    for t in turns:
        speaker = (t.get("speaker") or "Speaker").strip()
        text = (t.get("native_text") if use_native and t.get("native_text") else t.get("text") or "").strip()
        if not text:
            continue
        if speaker:
            lines.append(f"{speaker}: {text}")
        else:
            lines.append(text)
    return "\n".join(lines)


def _is_bilingual_duplicate_blocks(raw_turns: List[Dict[str, Optional[str]]]) -> bool:
    """
    Return True ONLY when `raw_turns` consists of two separate contiguous full-script blocks
    (an all-English block followed by an all-Indic block, or vice versa) rather than
    a single conversation that mixes English practice lines and Indic dialogue turns.
    """
    if len(raw_turns) < 6:
        return False
    flags = [
        bool(
            contains_indic_script(t.get("text") or "")
            or contains_indic_script(t.get("speaker") or "")
        )
        for t in raw_turns
    ]
    n_indic = sum(1 for f in flags if f)
    n_ascii = len(flags) - n_indic
    if n_indic < 3 or n_ascii < 3:
        return False
    if abs(n_indic - n_ascii) > 2:
        return False
    transitions = sum(1 for i in range(1, len(flags)) if flags[i] != flags[i - 1])
    return transitions <= 2


def extract_and_translate_dialogue(
    section_lines: List[Tuple[Optional[str], str]],
    language_hint: str = "Tamil",
) -> Tuple[str, str, List[Dict[str, Any]], str]:
    """
    Step 2: Identify dialogue turns, detect Indic Unicode scripts, and produce:
    `(native_script, english_script, structured_english_turns, detected_language)`.
    """
    raw_turns = _parse_dialogue_lines(section_lines)
    if not raw_turns:
        return "", "", [], language_hint

    all_dialogue_text = "\n".join((t["text"] or "") for t in raw_turns)
    has_any_indic, detected_lang = detect_indic_language(all_dialogue_text, fallback=language_hint)

    explicit_eng = [t for t in raw_turns if t.get("section") == "english"]
    explicit_nat = [t for t in raw_turns if t.get("section") == "native"]

    indic_turns = [
        {"speaker": t["speaker"] or "Speaker", "text": t["text"] or ""}
        for t in raw_turns
        if contains_indic_script(t["text"] or "") or contains_indic_script(t["speaker"] or "")
    ]
    ascii_turns = [
        {"speaker": t["speaker"] or "Speaker", "text": t["text"] or ""}
        for t in raw_turns
        if not (contains_indic_script(t["text"] or "") or contains_indic_script(t["speaker"] or ""))
    ]

    # Case A: Explicit English and Native sections exist
    if explicit_eng and explicit_nat:
        eng_list = [
            {"speaker": t["speaker"] or "Speaker", "text": t["text"] or ""}
            for t in explicit_eng
        ]
        nat_list = [
            {"speaker": t["speaker"] or "Speaker", "text": t["text"] or ""}
            for t in explicit_nat
        ]
    # Case B: Both an English script block and a Regional script block exist as two contiguous blocks
    elif has_any_indic and _is_bilingual_duplicate_blocks(raw_turns):
        eng_list = ascii_turns
        nat_list = indic_turns
    # Case C: Dialogue is in an Indic language (or code-mixed Indic + English turns in one script)
    elif has_any_indic:
        nat_list = [
            {"speaker": t["speaker"] or "Speaker", "text": t["text"] or ""}
            for t in raw_turns
        ]
        eng_list = _batch_translate_turns(nat_list)
    # Case D: Dialogue is already in English / Latin script
    else:
        eng_list = [
            {"speaker": t["speaker"] or "Speaker", "text": t["text"] or ""}
            for t in raw_turns
        ]
        nat_list = list(eng_list)

    structured_turns: List[Dict[str, Any]] = []
    for idx, t in enumerate(eng_list):
        speaker = (t.get("speaker") or "Speaker").strip()
        text = (t.get("text") or "").strip()
        native_text = (
            t.get("native_text")
            or (nat_list[idx]["text"] if idx < len(nat_list) else text)
        )
        words = re.findall(r"[\w']+", text)
        is_ai = bool(AI_SPEAKER_RE.search(speaker))
        structured_turns.append(
            {
                "turn_index": idx + 1,
                "speaker": speaker,
                "text": text,
                "native_text": native_text,
                "is_ai_teacher": is_ai,
                "word_count": len(words),
            }
        )

    native_script = _format_turns_as_script(nat_list, use_native=False)
    english_script = _format_turns_as_script(structured_turns, use_native=False)

    return native_script, english_script, structured_turns, detected_lang


# ---------------------------------------------------------------------------
# Step 3: Split `english_script` into the 4 Story Blocks
# ---------------------------------------------------------------------------
def _turn_has_usp_or_live_fix(text: str) -> bool:
    """Return True if a dialogue turn contains a Supernova USP or live practice cue."""
    if not text:
        return False
    for pat in USP_PATTERNS.values():
        if pat.search(text):
            return True
    if SUPERNOVA_MENTION_RE.search(text) or LIVE_PRACTICE_CUES_RE.search(text):
        return True
    return False


def split_into_story_blocks(turns: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Step 3: Partition `english_script` turns into the 4 Supernova story blocks:
    - `block_1_hook`: First 2 dialogue turns (opening hook + immediate AI teacher interruption).
    - `block_2_pain_and_old_methods`: Turns from Line 3 up to before the first Supernova USP mention (or middle third).
    - `block_3_live_fix_and_usps`: Turns where the AI teacher does on-the-spot practice and delivers Supernova USPs.
    - `block_4_desire_and_cta`: Final 2 turns (human's active desire line + AI teacher's Install Now CTA).
    """
    n = len(turns)
    if n == 0:
        return {
            "block_1_hook": "",
            "block_2_pain_and_old_methods": "",
            "block_3_live_fix_and_usps": "",
            "block_4_desire_and_cta": "",
        }

    if n <= 2:
        return {
            "block_1_hook": _format_turns_as_script(turns),
            "block_2_pain_and_old_methods": "",
            "block_3_live_fix_and_usps": "",
            "block_4_desire_and_cta": _format_turns_as_script(turns[-1:]),
        }

    if n == 3:
        return {
            "block_1_hook": _format_turns_as_script(turns[0:2]),
            "block_2_pain_and_old_methods": _format_turns_as_script(turns[1:2]),
            "block_3_live_fix_and_usps": _format_turns_as_script(turns[2:3]),
            "block_4_desire_and_cta": _format_turns_as_script(turns[2:3]),
        }

    if n == 4:
        return {
            "block_1_hook": _format_turns_as_script(turns[0:2]),
            "block_2_pain_and_old_methods": _format_turns_as_script(turns[2:3]),
            "block_3_live_fix_and_usps": _format_turns_as_script(turns[2:3]),
            "block_4_desire_and_cta": _format_turns_as_script(turns[3:4]),
        }

    if n == 5:
        return {
            "block_1_hook": _format_turns_as_script(turns[0:2]),
            "block_2_pain_and_old_methods": _format_turns_as_script(turns[2:3]),
            "block_3_live_fix_and_usps": _format_turns_as_script(turns[3:4]),
            "block_4_desire_and_cta": _format_turns_as_script(turns[3:5]),
        }

    # Standard script with n >= 6 turns
    block_1 = turns[0:2]
    block_4 = turns[n - 2 : n]
    middle_end = n - 2  # exclusive index

    first_usp_idx: Optional[int] = None
    for idx in range(2, middle_end):
        t_text = turns[idx].get("text", "")
        if any(pat.search(t_text) for pat in USP_PATTERNS.values()):
            first_usp_idx = idx
            break

    if first_usp_idx is None:
        for idx in range(2, middle_end):
            if _turn_has_usp_or_live_fix(turns[idx].get("text", "")):
                first_usp_idx = idx
                break

    # Also check if an explicit failed-old-methods turn (e.g., YouTube videos, grammar books,
    # coaching classes) appears in the early middle turns (indices 2 .. min(5, middle_end - 1))
    last_old_methods_idx: Optional[int] = None
    for idx in range(2, min(6, middle_end - 1)):
        if OLD_METHODS_RE.search(turns[idx].get("text", "")):
            last_old_methods_idx = idx

    if first_usp_idx is None and last_old_methods_idx is None:
        split_idx = 2 + max(1, (middle_end - 2) // 2)
    elif last_old_methods_idx is not None and first_usp_idx is not None:
        split_idx = max(last_old_methods_idx + 1, first_usp_idx)
    elif last_old_methods_idx is not None:
        split_idx = last_old_methods_idx + 1
    else:
        split_idx = first_usp_idx  # type: ignore[assignment]

    split_idx = max(3, min(int(split_idx), middle_end - 1))

    block_2 = turns[2:split_idx]
    block_3 = turns[split_idx:middle_end]

    return {
        "block_1_hook": _format_turns_as_script(block_1),
        "block_2_pain_and_old_methods": _format_turns_as_script(block_2),
        "block_3_live_fix_and_usps": _format_turns_as_script(block_3),
        "block_4_desire_and_cta": _format_turns_as_script(block_4),
    }


# ---------------------------------------------------------------------------
# Step 4: Metadata Tags & Pacing Metrics
# ---------------------------------------------------------------------------
def _kw_matches(text: str, kw: str) -> bool:
    """Word-boundary aware keyword match so short words ('amma', 'mom', 'hr') never match substrings."""
    pattern = r"\b" + re.escape(kw.strip()) + r"\b"
    return bool(re.search(pattern, text, re.IGNORECASE))


def _infer_persona(
    visual_blueprint: Dict[str, str],
    turns: List[Dict[str, Any]],
    title: str = "",
) -> str:
    """
    Infer canonical persona:
    - "Homemaker Mother"
    - "Working Professional"
    - "Job Seeker / Interviewee"
    - "School Parent / Father"
    - "College Student / General"
    """
    char_text = (
        f"{title} {visual_blueprint.get('characters_description', '')} "
        f"{visual_blueprint.get('raw_visual_text', '')}"
    ).lower()
    dialogue_text = " ".join(t.get("text", "") for t in turns).lower()

    def score_keywords(keywords: List[str]) -> int:
        score = 0
        for kw in keywords:
            if _kw_matches(char_text, kw):
                score += 3
            if _kw_matches(dialogue_text, kw):
                score += 1
        return score

    scores = {
        "Homemaker Mother": score_keywords(
            [
                "homemaker", "housewife", "house wife", "mother", "mom", "amma",
                "mummy", "saree", "sari", "kitchen", "cooking", "her son",
                "her daughter", "her kid", "her child",
            ]
        ),
        "Working Professional": score_keywords(
            [
                "working professional", "software engineer", "it employee", "employee",
                "office", "manager", "boss", "client", "team meeting", "colleague",
                "promotion", "appraisal", "corporate", "presentation", "workplace",
                "developer", "executive", "client call", "zoom call",
            ]
        ),
        "Job Seeker / Interviewee": score_keywords(
            [
                "job seeker", "interviewee", "job interview", "interview", "hr",
                "recruiter", "resume", "cv", "fresher", "hiring", "job offer",
                "walk-in",
            ]
        ),
        "School Parent / Father": score_keywords(
            [
                "father", "dad", "appa", "papa", "school parent", "ptm",
                "parent-teacher", "parent teacher", "principal", "class teacher",
                "school admission", "his son", "his daughter",
            ]
        ),
        "College Student / General": score_keywords(
            [
                "college", "student", "campus", "professor", "lecturer", "hostel",
                "semester", "placement", "youth", "classmate", "daily situations",
                "general life", "modern society", "communication skills",
            ]
        ),
    }

    best_persona, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score == 0:
        return "College Student / General"
    return best_persona


def _infer_setting(
    visual_blueprint: Dict[str, str],
    turns: List[Dict[str, Any]],
    title: str = "",
) -> str:
    """
    Infer canonical setting:
    - "Home / Study Table"
    - "Office / Team Meeting"
    - "Job Interview"
    - "School / PTM"
    - "Public / Restaurant / Doctor"
    """
    vis_text = (
        f"{title} {visual_blueprint.get('raw_visual_text', '')} "
        f"{visual_blueprint.get('characters_description', '')}"
    ).lower()
    dialogue_text = " ".join(t.get("text", "") for t in turns).lower()

    def score_keywords(keywords: List[str]) -> int:
        score = 0
        for kw in keywords:
            if _kw_matches(vis_text, kw):
                score += 3
            if _kw_matches(dialogue_text, kw):
                score += 1
        return score

    scores = {
        "Job Interview": score_keywords(
            [
                "job interview", "interview room", "interview panel", "interview desk",
                "hr cabin", "interviewee", "recruiter", "resume", "interview",
            ]
        ),
        "Office / Team Meeting": score_keywords(
            [
                "office", "team meeting", "conference room", "boardroom", "cubicle",
                "workplace", "client call", "manager", "presentation", "zoom call",
                "zoom meeting", "office desk", "colleague", "laptop",
            ]
        ),
        "School / PTM": score_keywords(
            [
                "ptm", "parent-teacher", "parent teacher", "school", "classroom",
                "principal", "report card", "class teacher", "school diary",
            ]
        ),
        "Public / Restaurant / Doctor": score_keywords(
            [
                "restaurant", "cafe", "coffee shop", "waiter", "menu", "doctor",
                "hospital", "clinic", "prescription", "airport", "flight", "hotel",
                "reception", "mall", "supermarket", "shop", "store", "bank", "public",
            ]
        ),
        "Home / Study Table": score_keywords(
            [
                "home", "house", "study table", "living room", "bedroom", "kitchen",
                "sofa", "couch", "dining table", "homework", "study room",
            ]
        ),
    }

    best_setting, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score == 0:
        return "Home / Study Table"
    return best_setting


def _infer_hook_angle(
    visual_blueprint: Dict[str, str],
    turns: List[Dict[str, Any]],
    title: str = "",
) -> str:
    """
    Infer canonical hook_angle:
    - "Spoken Mistake Correction"
    - "Social / Public Embarrassment"
    - "Child / Family Mirror"
    - "Career / Salary Shock"
    - "Direct Question / Challenge"
    """
    hook_turns_text = " ".join(t.get("text", "") for t in turns[:2]).lower()
    hook_vis_text = f"{title} {visual_blueprint.get('hook_visuals', '')}".lower()
    full_text = " ".join(t.get("text", "") for t in turns).lower()

    def score_keywords(keywords: List[str]) -> int:
        score = 0
        for kw in keywords:
            if _kw_matches(hook_turns_text, kw):
                score += 4
            if _kw_matches(hook_vis_text, kw):
                score += 3
            if _kw_matches(full_text, kw):
                score += 1
        return score

    scores = {
        "Spoken Mistake Correction": score_keywords(
            [
                "don't say", "do not say", "never say", "stop saying", "stop right there",
                "instead of saying", "say it like this", "myself", "didn't knew",
                "didn't went", "return back", "revert back", "cousin brother",
                "cousin sister", "today morning", "yesterday night", "asking doubts",
                "asking me doubts", "homework doing daily", "already sending",
                "stomach paining", "tablet eating", "broken english", "spoken mistake",
            ]
        ),
        "Social / Public Embarrassment": score_keywords(
            [
                "laugh", "laughed", "mock", "embarrassed", "embarrassing", "humiliated",
                "insult", "ashamed", "shame", "made fun", "froze", "nervous",
                "everyone looked", "afraid to speak", "hesitant", "shy",
            ]
        ),
        "Child / Family Mirror": score_keywords(
            [
                "my son", "my daughter", "my boy", "my girl", "my child", "my kid",
                "homework", "report card", "school diary", "mummy", "amma", "appa",
                "children speak", "teach my child", "help my child", "parent-teacher",
                "ptm",
            ]
        ),
        "Career / Salary Shock": score_keywords(
            [
                "salary", "promotion", "increment", "hike", "package", "rejected",
                "juniors speak fluent", "career", "job offer", "appraisal",
                "client call", "boss", "manager",
            ]
        ),
        "Direct Question / Challenge": score_keywords(
            [
                "have you ever noticed", "still struggling", "why can't you",
                "want to speak", "can you translate", "how do you say",
                "do you also", "challenge", "tell me in english", "are you still",
                "communication is very important",
            ]
        ),
    }

    if turns and "?" in (turns[0].get("text") or ""):
        scores["Direct Question / Challenge"] += 3

    best_angle, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score == 0:
        return "Direct Question / Challenge"
    return best_angle


def extract_metadata_tags(
    turns: List[Dict[str, Any]],
    visual_blueprint: Dict[str, str],
    title: str = "",
    language: str = "Tamil",
) -> Dict[str, Any]:
    """
    Step 4: Compute `metadata_tags` including persona, setting, hook_angle,
    and exact pacing / USP metrics.
    """
    total_turns = len(turns)
    hook_word_count = turns[0]["word_count"] if total_turns > 0 else 0
    total_words = sum(t.get("word_count", 0) for t in turns)
    avg_words_per_line = round(total_words / total_turns, 1) if total_turns > 0 else 0.0

    # Find 1-based turn index where AI teacher first speaks
    ai_teacher_entry_turn = 0
    for t in turns:
        if t.get("is_ai_teacher"):
            ai_teacher_entry_turn = int(t["turn_index"])
            break
    if ai_teacher_entry_turn == 0 and total_turns >= 2:
        for t in turns:
            if re.search(
                r"\b(?:stop|hold\s+on|wait|don'?t\s+say|supernova|miss\s+nova|install\s+now)\b",
                t.get("text", ""),
                re.I,
            ):
                ai_teacher_entry_turn = int(t["turn_index"])
                break
        if ai_teacher_entry_turn == 0:
            ai_teacher_entry_turn = 2
    elif ai_teacher_entry_turn == 0 and total_turns == 1:
        ai_teacher_entry_turn = 1

    # Check whether at least 2 core Supernova USPs appear in the first 55% of turns
    first_half_cutoff = max(1, math.ceil(total_turns * 0.55)) if total_turns > 0 else 0
    first_half_text = " ".join(t.get("text", "") for t in turns[:first_half_cutoff])
    matched_first_half_usps = [
        usp_name for usp_name, pat in USP_PATTERNS.items() if pat.search(first_half_text)
    ]
    has_first_half_usps = len(matched_first_half_usps) >= 2

    persona = _infer_persona(visual_blueprint, turns, title=title)
    setting = _infer_setting(visual_blueprint, turns, title=title)
    hook_angle = _infer_hook_angle(visual_blueprint, turns, title=title)

    return {
        "persona": persona,
        "setting": setting,
        "hook_angle": hook_angle,
        "hook_word_count": int(hook_word_count),
        "avg_words_per_line": float(avg_words_per_line),
        "total_turns": int(total_turns),
        "ai_teacher_entry_turn": int(ai_teacher_entry_turn),
        "has_first_half_usps": bool(has_first_half_usps),
        "language": language,
    }


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def parse_proven_doc_text(
    raw_text: str,
    language_hint: str = "Tamil",
    title: str = "",
) -> Dict[str, Any]:
    """
    Parse a raw Supernova Google Doc script into structured components:
    - `visual_blueprint`
    - `native_script`
    - `english_script`
    - `story_blocks`
    - `metadata_tags`
    - `dialogue_turns`

    Zero Gemini API calls. Uses pure Python + regex + `deep_translator.GoogleTranslator`.
    """
    visual_blueprint, remaining_lines = extract_visual_blueprint(raw_text or "")
    native_script, english_script, structured_turns, detected_lang = (
        extract_and_translate_dialogue(remaining_lines, language_hint=language_hint)
    )
    story_blocks = split_into_story_blocks(structured_turns)
    metadata_tags = extract_metadata_tags(
        structured_turns,
        visual_blueprint=visual_blueprint,
        title=title,
        language=detected_lang,
    )

    return {
        "title": title,
        "language": detected_lang,
        "visual_blueprint": visual_blueprint,
        "native_script": native_script,
        "english_script": english_script,
        "story_blocks": story_blocks,
        "metadata_tags": metadata_tags,
        "dialogue_turns": structured_turns,
    }


# Convenience alias
parse_script_text = parse_proven_doc_text
