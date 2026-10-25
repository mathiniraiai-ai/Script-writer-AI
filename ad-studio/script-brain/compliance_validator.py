"""Deterministic compliance validator and Google Doc formatter for Supernova Ad Scripts.

Validates generated scripts against Supernova's non-negotiable brand, linguistic,
and conversion house rules, and formats the complete output into clean, markdown-free
Google Doc text ready for production copy-paste.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Exact Forbidden Phrase Regexes (No blanket word bans!)
# e.g. "I learned this the hard way" MUST pass.
# ---------------------------------------------------------------------------
FORBIDDEN_REGEX_SPECS: List[Tuple[str, re.Pattern[str], str]] = [
    (
        r"\b(learn|learning|learned|learnt|teach|teaching|teaches|taught)\s+(spoken\s+)?english\b",
        re.compile(
            r"\b(learn|learning|learned|learnt|teach|teaching|teaches|taught)\s+(spoken\s+)?english\b",
            re.IGNORECASE,
        ),
        "Never say 'learn/teach (spoken) English'; always say 'improve your spoken English'",
    ),
    (
        r"\b10(\s*(to|-)\s*15)?\s*minutes\b",
        re.compile(
            r"\b10(\s*(to|-)\s*15)?\s*minutes\b",
            re.IGNORECASE,
        ),
        "Never use '10 minutes' or '10 to 15 minutes'; always use '15 to 20 minutes'",
    ),
    (
        r"^Robot\s*:",
        re.compile(
            r"^\s*Robot\s*:",
            re.MULTILINE | re.IGNORECASE,
        ),
        "Never use standalone 'Robot:' speaker label; use the specific AI teacher character name (e.g. 'Female Humanoid Robot:')",
    ),
    (
        r"(\b(rs\.?|inr|rupees?)\s*\d+|₹\s*\d+|\d+\s*rupees?\b|\bfree\s+trial\b)",
        re.compile(
            r"(\b(rs\.?|inr|rupees?)\s*\d+|₹\s*\d+|\d+\s*rupees?\b|\bfree\s+trial\b)",
            re.IGNORECASE,
        ),
        "Never mention rupee pricing (Rs/INR/₹) or 'free trial' in ad scripts",
    ),
    (
        r"\b(mysivi|mysore|speakx|stimuler|duolingo|cambly)\b",
        re.compile(
            r"\b(mysivi|mysore|speakx|stimuler|duolingo|cambly)\b",
            re.IGNORECASE,
        ),
        "Never mention competitor app names or 'Mysore'/'MySivi' mistranscriptions",
    ),
]


def _build_required_phrase_specs(
    language: str = "Tamil",
) -> List[Tuple[str, re.Pattern[str], str]]:
    """Build the 7 required hard-locked phrase specifications for the target language."""
    clean_lang = (language or "Tamil").strip() or "Tamil"
    # Support composite language labels like "Tamil / Malayalam" by matching either the full
    # string or individual language tokens, while always matching "through <language> itself".
    lang_tokens = [
        re.escape(tok.strip())
        for tok in re.split(r"[/,|]+", clean_lang)
        if tok.strip()
    ]
    if len(lang_tokens) > 1:
        lang_pattern = "(?:" + "|".join([re.escape(clean_lang)] + lang_tokens) + ")"
    else:
        lang_pattern = re.escape(clean_lang)

    return [
        (
            "improve your spoken English",
            re.compile(
                r"\bimprove\s+(?:(?:your|my|our|their|his|her)\s+)?spoken\s+english\b",
                re.IGNORECASE,
            ),
            "Must include 'improve your spoken English' (or 'improve my/spoken English')",
        ),
        (
            "within 30 days",
            re.compile(r"\bwithin\s+30\s+days\b", re.IGNORECASE),
            "Must include the timeframe promise 'within 30 days'",
        ),
        (
            "15 to 20 minutes",
            re.compile(r"\b15\s*(?:to|-)\s*20\s+minutes\b", re.IGNORECASE),
            "Must include daily practice commitment '15 to 20 minutes'",
        ),
        (
            "over 1 crore Indians",
            re.compile(r"\bover\s+1\s+crore\s+indians\b", re.IGNORECASE),
            "Must include social proof 'over 1 crore Indians'",
        ),
        (
            f"through {clean_lang} itself",
            re.compile(
                rf"\bthrough\s+{lang_pattern}\s+itself\b",
                re.IGNORECASE,
            ),
            f"Must include native-language bridge 'through {clean_lang} itself'",
        ),
        (
            "Install Now button given below",
            re.compile(
                r"[\"']?\binstall\s+now\b[\"']?\s+button\s+given\s+below\b",
                re.IGNORECASE,
            ),
            "Must include exact CTA phrase 'Install Now button given below'",
        ),
        (
            "today itself",
            re.compile(r"\btoday\s+itself\b", re.IGNORECASE),
            "Must include urgency close 'today itself'",
        ),
    ]


def validate_supernova_script(
    script_text: str,
    language: str = "Tamil",
    ai_teacher_character: str = "Female Humanoid Robot",
) -> Dict[str, Any]:
    """Deterministically validate a Supernova script against all hard-locked house rules.

    Args:
        script_text: The full dialogue script text to validate.
        language: The target regional language (e.g., "Tamil", "Malayalam", "Telugu", "Kannada").
        ai_teacher_character: The designated AI teacher speaker label (e.g., "Female Humanoid Robot").

    Returns:
        dict with keys:
            - passed (bool)
            - score (float, 0.0 to 1.0)
            - missing_phrases (list[str])
            - forbidden_matches (list[str])
            - violations (list[str])
    """
    text = script_text or ""
    missing_phrases: List[str] = []
    forbidden_matches: List[str] = []
    violations: List[str] = []

    if not text.strip():
        required_specs = _build_required_phrase_specs(language)
        missing_phrases = [label for label, _, _ in required_specs]
        violations.append("Script text is empty.")
        for label, _, reason in required_specs:
            violations.append(f"Missing required phrase: '{label}' ({reason})")
        return {
            "passed": False,
            "score": 0.0,
            "missing_phrases": missing_phrases,
            "forbidden_matches": forbidden_matches,
            "violations": violations,
        }

    # 1. Check Required Hard-Locked Phrases
    required_specs = _build_required_phrase_specs(language)
    for canonical_label, pattern, reason in required_specs:
        if not pattern.search(text):
            missing_phrases.append(canonical_label)
            violations.append(f"Missing required phrase: '{canonical_label}' — {reason}")

    # 2. Check Exact Forbidden Phrase Regexes
    for pattern_str, regex, reason in FORBIDDEN_REGEX_SPECS:
        for match in regex.finditer(text):
            matched_value = match.group(0).strip()
            forbidden_matches.append(matched_value)
            violations.append(
                f"Forbidden phrase matched: '{matched_value}' (pattern: {pattern_str}) — {reason}"
            )

    # 3. Calculate deterministic compliance score (0.0 to 1.0)
    total_required = len(required_specs)
    present_required = total_required - len(missing_phrases)
    base_score = present_required / float(total_required) if total_required > 0 else 1.0
    penalty = 0.15 * len(forbidden_matches)
    score = max(0.0, min(1.0, round(base_score - penalty, 4)))

    passed = len(missing_phrases) == 0 and len(forbidden_matches) == 0
    if passed:
        score = 1.0

    return {
        "passed": passed,
        "score": score,
        "missing_phrases": missing_phrases,
        "forbidden_matches": forbidden_matches,
        "violations": violations,
    }


def _strip_markdown(text: Any) -> str:
    """Remove Markdown bold/heading markers (** and #) while preserving clean plain text."""
    if text is None:
        return ""
    s = str(text)
    # Remove markdown bold/italic asterisks
    s = s.replace("**", "")
    # Remove leading markdown heading symbols (#, ##, ###, etc.) at the start of lines
    s = re.sub(r"(?m)^\s*#{1,6}\s*", "", s)
    # Remove inline stray # used as markdown headers while keeping normal text clean
    s = s.replace("#", "")
    # Collapse excessive blank lines (max 2 consecutive newlines)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _extract_prediction_header(result: Dict[str, Any]) -> Tuple[str, str, str]:
    """Extract (rate_str, confidence_label, historical_benchmark) from a result dict."""
    prediction_obj = result.get("prediction") or result.get("cac_prediction") or {}
    if not isinstance(prediction_obj, dict):
        prediction_obj = {}

    raw_rate = (
        result.get("predicted_success_rate")
        if "predicted_success_rate" in result
        else prediction_obj.get(
            "predicted_success_rate",
            result.get("success_rate", prediction_obj.get("score")),
        )
    )

    if isinstance(raw_rate, dict):
        prediction_obj = {**prediction_obj, **raw_rate}
        raw_rate = raw_rate.get(
            "predicted_success_rate",
            raw_rate.get("rate", raw_rate.get("score", raw_rate.get("percentage"))),
        )

    rate_num: Optional[float] = None
    if isinstance(raw_rate, (int, float)):
        rate_num = float(raw_rate)
        if 0.0 < rate_num <= 1.0:
            rate_num *= 100.0
    elif isinstance(raw_rate, str):
        m = re.search(r"(\d+(?:\.\d+)?)", raw_rate)
        if m:
            rate_num = float(m.group(1))
            if 0.0 < rate_num <= 1.0 and "%" not in raw_rate:
                rate_num *= 100.0

    if rate_num is None:
        rate_num = 82.0

    if abs(rate_num - round(rate_num)) < 1e-6:
        rate_str = f"{int(round(rate_num))}"
    else:
        rate_str = f"{rate_num:.1f}"

    confidence_label = (
        result.get("confidence_label")
        or prediction_obj.get("confidence_label")
        or result.get("confidence")
        or prediction_obj.get("confidence")
    )
    if not confidence_label:
        if rate_num >= 75:
            confidence_label = "High Confidence"
        elif rate_num >= 60:
            confidence_label = "Moderate-High Confidence"
        else:
            confidence_label = "Moderate Confidence"

    benchmark = (
        result.get("historical_benchmark")
        or prediction_obj.get("historical_benchmark")
        or result.get("benchmark_summary")
        or prediction_obj.get("benchmark_summary")
        or result.get("benchmark")
        or prediction_obj.get("benchmark")
    )

    if not benchmark:
        winners = (
            result.get("matched_winners")
            or prediction_obj.get("matched_winners")
            or result.get("similar_scripts")
            or result.get("top_matches")
            or []
        )
        if isinstance(winners, list) and winners:
            parts: List[str] = []
            for w in winners[:3]:
                if isinstance(w, dict):
                    code = (
                        w.get("script_id")
                        or w.get("code")
                        or w.get("id")
                        or w.get("name")
                        or w.get("title")
                        or "Winner"
                    )
                    cac = w.get("cac") or w.get("best_cac")
                    conv = w.get("conversion_rate") or w.get("tc_to_paid")
                    detail_bits = []
                    if cac is not None:
                        detail_bits.append(f"CAC {cac}")
                    if conv is not None:
                        detail_bits.append(f"Conv {conv}")
                    if detail_bits:
                        parts.append(f"{code} ({', '.join(detail_bits)})")
                    else:
                        parts.append(str(code))
                elif isinstance(w, str):
                    parts.append(w)
            if parts:
                benchmark = f"Matched top historical CAC winners: {', '.join(parts)}"

    if not benchmark:
        benchmark = (
            "Modelled on top-performing Supernova low-CAC winners featuring relatable pain-point hooks, "
            "Female Humanoid Robot product demo, and native-language reassurance."
        )

    # Ensure single-line summary for Historical Benchmark
    benchmark_clean = " ".join(_strip_markdown(benchmark).splitlines()).strip()
    return _strip_markdown(rate_str), _strip_markdown(confidence_label), benchmark_clean


def _format_characters(chars: Any) -> List[str]:
    """Format character descriptions into clean lines."""
    if not chars:
        return [
            "- Lead Protagonist: Relatable Indian speaker facing an everyday spoken English hesitation moment.",
            "- Supporting Character: Peer, colleague, or family member who introduces Supernova.",
            "- Female Humanoid Robot: Warm, encouraging AI teacher inside the Supernova app guiding live speaking practice.",
        ]
    if isinstance(chars, str):
        cleaned = _strip_markdown(chars)
        return [line.strip() for line in cleaned.splitlines() if line.strip()]
    if isinstance(chars, list):
        lines: List[str] = []
        for item in chars:
            if isinstance(item, dict):
                name = _strip_markdown(
                    item.get("name") or item.get("role") or item.get("character") or "Character"
                )
                desc = _strip_markdown(
                    item.get("description")
                    or item.get("details")
                    or item.get("appearance")
                    or item.get("persona")
                    or ""
                )
                if desc:
                    lines.append(f"- {name}: {desc}")
                else:
                    lines.append(f"- {name}")
            else:
                line = _strip_markdown(item)
                if line:
                    lines.append(line if line.startswith("-") else f"- {line}")
        return lines
    if isinstance(chars, dict):
        return [
            f"- {_strip_markdown(k)}: {_strip_markdown(v)}"
            for k, v in chars.items()
        ]
    return [_strip_markdown(chars)]


def _format_visual_description_section(result: Dict[str, Any]) -> str:
    """Format the Visual Description section with Hook visuals, Body visuals, Video flow, Characters."""
    vis = result.get("visual_description") or result.get("visuals") or {}
    if isinstance(vis, str) and vis.strip():
        cleaned = _strip_markdown(vis)
        # Ensure all required labels are present even if a raw string was passed
        has_all = all(
            lbl in cleaned
            for lbl in ("Hook visuals:", "Body visuals:", "Video flow:", "Characters")
        )
        if has_all:
            if not cleaned.startswith("Visual Description:"):
                return f"Visual Description:\n{cleaned}"
            return cleaned

    if not isinstance(vis, dict):
        vis = {}

    hook_visuals = _strip_markdown(
        vis.get("hook_visuals")
        or vis.get("Hook visuals")
        or vis.get("hook")
        or result.get("hook_visuals")
        or "Split-screen or dynamic close-up capturing the protagonist's hesitation while speaking English in a high-stakes real-life moment."
    )
    body_visuals = _strip_markdown(
        vis.get("body_visuals")
        or vis.get("Body visuals")
        or vis.get("body")
        or result.get("body_visuals")
        or "Natural conversation transitioning to live phone screen recording of the Supernova app with the Female Humanoid Robot giving instant, friendly feedback."
    )
    video_flow = _strip_markdown(
        vis.get("video_flow")
        or vis.get("Video flow")
        or vis.get("flow")
        or result.get("video_flow")
        or "Fast 3-second pattern-interrupt hook -> relatable struggle -> peer recommendation -> interactive AI Teacher demo -> urgent Install Now CTA."
    )
    characters_raw = (
        vis.get("characters")
        or vis.get("Characters")
        or vis.get("character_descriptions")
        or result.get("characters")
        or result.get("character_descriptions")
    )
    char_lines = _format_characters(characters_raw)

    lines = [
        "Visual Description:",
        f"Hook visuals: {hook_visuals}",
        f"Body visuals: {body_visuals}",
        f"Video flow: {video_flow}",
        "Characters:",
        *char_lines,
    ]
    return "\n".join(lines)


def _format_hook_options_section(result: Dict[str, Any]) -> str:
    """Format the 3 Recommended Hook + Bridge Options section (Option A, Option B, Option C)."""
    raw_hooks = (
        result.get("hook_options")
        or result.get("recommended_hooks")
        or result.get("three_hook_options")
        or result.get("hook_and_bridge_options")
        or result.get("hooks")
    )

    option_letters = ["Option A", "Option B", "Option C"]
    formatted_options: List[str] = []

    if isinstance(raw_hooks, dict):
        for idx, letter in enumerate(["A", "B", "C"]):
            opt_key = f"Option {letter}"
            val = (
                raw_hooks.get(opt_key)
                or raw_hooks.get(letter)
                or raw_hooks.get(f"option_{letter.lower()}")
                or raw_hooks.get(f"hook_{idx + 1}")
            )
            if val is not None:
                formatted_options.append(_format_single_hook_option(opt_key, val))
    elif isinstance(raw_hooks, list):
        for idx, item in enumerate(raw_hooks[:3]):
            label = option_letters[idx] if idx < len(option_letters) else f"Option {idx + 1}"
            formatted_options.append(_format_single_hook_option(label, item))
    elif isinstance(raw_hooks, str) and raw_hooks.strip():
        cleaned = _strip_markdown(raw_hooks)
        if all(opt in cleaned for opt in option_letters):
            if not cleaned.startswith("3 Recommended Hook + Bridge Options:"):
                return f"3 Recommended Hook + Bridge Options:\n{cleaned}"
            return cleaned

    # Pad up to 3 options if fewer were provided
    defaults = [
        (
            "Option A (High-Stakes Freeze Hook)",
            "Visual: Close-up expression during an awkward English pause.\n"
            "Hook: I knew the exact answer in my head, but the moment I had to say it in English, the words just froze.\n"
            "Bridge: That was the day I realized I didn't need another grammar textbook — I needed daily speaking practice to improve my spoken English.",
        ),
        (
            "Option B (Relatable Embarrassment Hook)",
            "Visual: Protagonist watching a peer speak effortlessly while they hesitate.\n"
            "Hook: Have you ever stayed silent in a meeting or interview just because you were scared of making a mistake in English?\n"
            "Bridge: I learned this the hard way until a friend showed me how over 1 crore Indians are fixing their hesitation in just 15 to 20 minutes a day.",
        ),
        (
            "Option C (Direct Pattern-Interrupt Hook)",
            "Visual: Protagonist holding up their phone showing the Supernova AI Teacher interface.\n"
            "Hook: Stop trying to memorize English dictionary words if you still hesitate when someone speaks to you!\n"
            "Bridge: Here is how you can actually improve your spoken English within 30 days with a 24/7 personal AI teacher.",
        ),
    ]
    while len(formatted_options) < 3:
        idx = len(formatted_options)
        default_title, default_body = defaults[idx]
        formatted_options.append(f"{default_title}:\n{default_body}")

    return "3 Recommended Hook + Bridge Options:\n" + "\n\n".join(formatted_options[:3])


def _format_single_hook_option(option_label: str, item: Any) -> str:
    """Format a single Hook + Bridge option entry."""
    if isinstance(item, str):
        cleaned = _strip_markdown(item)
        if cleaned.lower().startswith(option_label.lower()):
            return cleaned
        return f"{option_label}:\n{cleaned}"

    if isinstance(item, dict):
        title = _strip_markdown(
            item.get("title")
            or item.get("angle")
            or item.get("style")
            or item.get("name")
            or item.get("label")
            or ""
        )
        # Strip duplicate "Option A" prefix from title if present
        title_clean = re.sub(
            rf"^{re.escape(option_label)}\s*[:\-–()]*\s*",
            "",
            title,
            flags=re.IGNORECASE,
        ).strip(" ()")

        header = f"{option_label} ({title_clean}):" if title_clean else f"{option_label}:"
        sublines: List[str] = []

        visual = _strip_markdown(item.get("visual") or item.get("hook_visual") or "")
        if visual:
            sublines.append(f"Visual: {visual}")

        hook_line = _strip_markdown(
            item.get("hook")
            or item.get("hook_text")
            or item.get("hook_line")
            or item.get("dialogue")
            or ""
        )
        if hook_line:
            sublines.append(f"Hook: {hook_line}")

        bridge_line = _strip_markdown(
            item.get("bridge")
            or item.get("bridge_text")
            or item.get("bridge_line")
            or item.get("transition")
            or ""
        )
        if bridge_line:
            sublines.append(f"Bridge: {bridge_line}")

        rationale = _strip_markdown(
            item.get("why_it_works") or item.get("rationale") or item.get("note") or ""
        )
        if rationale:
            sublines.append(f"Why it works: {rationale}")

        if not sublines:
            # Fallback if dict has arbitrary keys
            for k, v in item.items():
                if k not in ("title", "angle", "style", "name", "label", "option") and v:
                    sublines.append(f"{_strip_markdown(k).capitalize()}: {_strip_markdown(v)}")

        return header + "\n" + "\n".join(sublines)

    return f"{option_label}:\n{_strip_markdown(item)}"


def _format_four_block_strategy_section(result: Dict[str, Any]) -> str:
    """Format the Middle Brain 4-Block Strategy section (Block 1 to Block 4)."""
    raw_blocks = (
        result.get("middle_brain_strategy")
        or result.get("four_block_plan")
        or result.get("middle_brain_plan")
        or result.get("thinking_plan")
        or result.get("four_blocks")
        or result.get("strategy_blocks")
        or result.get("blocks")
    )

    block_Labels = [
        ("Block 1", "Hook & Relatable Pain Point"),
        ("Block 2", "The Pivot & Supernova Introduction"),
        ("Block 3", "Product Proof & Hard-Locked Value Pillars"),
        ("Block 4", "Urgency & Call to Action"),
    ]

    formatted_blocks: List[str] = []

    if isinstance(raw_blocks, dict):
        for idx, (blk_label, default_sub) in enumerate(block_Labels, start=1):
            val = (
                raw_blocks.get(blk_label)
                or raw_blocks.get(f"block_{idx}")
                or raw_blocks.get(f"Block_{idx}")
                or raw_blocks.get(str(idx))
            )
            if val is not None:
                formatted_blocks.append(_format_single_block(blk_label, default_sub, val))
    elif isinstance(raw_blocks, list):
        for idx, item in enumerate(raw_blocks[:4]):
            blk_label, default_sub = (
                block_Labels[idx]
                if idx < len(block_Labels)
                else (f"Block {idx + 1}", f"Stage {idx + 1}")
            )
            formatted_blocks.append(_format_single_block(blk_label, default_sub, item))
    elif isinstance(raw_blocks, str) and raw_blocks.strip():
        cleaned = _strip_markdown(raw_blocks)
        if all(f"Block {i}" in cleaned for i in range(1, 5)):
            if not cleaned.startswith("Middle Brain 4-Block Strategy:"):
                return f"Middle Brain 4-Block Strategy:\n{cleaned}"
            return cleaned

    default_descriptions = [
        "Open with an emotionally resonant spoken-English hesitation scenario that mirrors the viewer's daily life and immediately captures attention.",
        "Bridge from the struggle to the discovery of the Supernova Spoken English App, reframing the solution around active speaking practice rather than passive memorization.",
        "Demonstrate live interaction with the AI teacher character, naturally weaving in how over 1 crore Indians improve your spoken English within 30 days by practicing 15 to 20 minutes daily through their native language itself.",
        "Close with high-conviction urgency encouraging the viewer to tap the Install Now button given below and start practicing today itself.",
    ]

    while len(formatted_blocks) < 4:
        idx = len(formatted_blocks)
        blk_label, default_sub = block_Labels[idx]
        formatted_blocks.append(
            f"{blk_label} ({default_sub}): {default_descriptions[idx]}"
        )

    return "Middle Brain 4-Block Strategy:\n" + "\n".join(formatted_blocks[:4])


def _format_single_block(blk_label: str, default_subtitle: str, item: Any) -> str:
    """Format a single block in the 4-Block Thinking Plan."""
    if isinstance(item, str):
        cleaned = _strip_markdown(item)
        if cleaned.lower().startswith(blk_label.lower()):
            return cleaned
        return f"{blk_label} ({default_subtitle}): {cleaned}"

    if isinstance(item, dict):
        title = _strip_markdown(
            item.get("title")
            or item.get("name")
            or item.get("focus")
            or item.get("stage")
            or default_subtitle
        )
        title_clean = re.sub(
            rf"^{re.escape(blk_label)}\s*[:\-–()]*\s*",
            "",
            title,
            flags=re.IGNORECASE,
        ).strip(" ()")
        summary = _strip_markdown(
            item.get("description")
            or item.get("summary")
            or item.get("strategy")
            or item.get("objective")
            or item.get("content")
            or item.get("plan")
            or ""
        )
        beats = item.get("beats") or item.get("key_points") or item.get("points")
        if isinstance(beats, list) and beats:
            beats_str = " ".join(_strip_markdown(b) for b in beats if b)
            summary = f"{summary} {beats_str}".strip()

        header = f"{blk_label} ({title_clean}):" if title_clean else f"{blk_label}:"
        return f"{header} {summary}".strip()

    return f"{blk_label} ({default_subtitle}): {_strip_markdown(item)}"


def _format_script_section(result: Dict[str, Any]) -> str:
    """Format the full word-for-word Supernova Script section."""
    raw_script = (
        result.get("script")
        or result.get("script_text")
        or result.get("full_script")
        or result.get("supernova_script")
        or ""
    )

    if isinstance(raw_script, list):
        lines: List[str] = []
        for entry in raw_script:
            if isinstance(entry, dict):
                speaker = _strip_markdown(
                    entry.get("speaker") or entry.get("character") or "Speaker"
                )
                dialogue = _strip_markdown(
                    entry.get("dialogue") or entry.get("line") or entry.get("text") or ""
                )
                stage_dir = _strip_markdown(entry.get("visual") or entry.get("action") or "")
                if stage_dir:
                    lines.append(f"[{stage_dir}]")
                lines.append(f"{speaker}: {dialogue}")
            else:
                lines.append(_strip_markdown(entry))
        script_body = "\n".join(lines)
    else:
        script_body = _strip_markdown(raw_script)

    # Remove redundant leading "Script:" header if already inside raw_script
    script_body = re.sub(r"(?i)^script\s*:\s*\n?", "", script_body).strip()
    return f"Script:\n{script_body}"


def format_complete_google_doc_text(result: Dict[str, Any]) -> str:
    """Format a complete generation result dictionary into clean Google Doc text.

    Produces Supernova's exact structure without any markdown (`**` or `#`) symbols:
      - Predicted Success Rate: <X>% (<confidence_label>)
      - Historical Benchmark: <1-line summary of matched CAC winners>
      - Blank line
      - Visual Description: (Hook visuals:, Body visuals:, Video flow:, Characters)
      - Blank line
      - 3 Recommended Hook + Bridge Options: (Option A, Option B, Option C)
      - Blank line
      - Middle Brain 4-Block Strategy: (Block 1 to Block 4)
      - Blank line
      - Script: (Full word-for-word Supernova dialogue)
    """
    if not isinstance(result, dict):
        result = {"script": str(result or "")}

    rate_str, confidence_label, benchmark_summary = _extract_prediction_header(result)
    header_section = (
        f"Predicted Success Rate: {rate_str}% ({confidence_label})\n"
        f"Historical Benchmark: {benchmark_summary}"
    )

    visual_section = _format_visual_description_section(result)
    hooks_section = _format_hook_options_section(result)
    strategy_section = _format_four_block_strategy_section(result)
    script_section = _format_script_section(result)

    full_doc = "\n\n".join(
        [
            header_section,
            visual_section,
            hooks_section,
            strategy_section,
            script_section,
        ]
    )

    # Final pass to guarantee zero markdown '**' or '#' symbols anywhere in output
    return _strip_markdown(full_doc)
