"""Stanford DSPy (`stanfordnlp/dspy`) Supernova Middle Brain (`dspy_brain.py`).

Implements:
1. `ConceptBriefToSupernovaPackage(dspy.Signature)`:
   - Inputs:
     - `concept_brief`
     - `language`
     - `ai_teacher_character`
     - `matched_winner_scripts`
     - `matched_winner_visuals`
     - `contrastive_playbook_rules`
   - Outputs:
     - `four_block_thinking_plan`
     - `visual_description_block`
     - `three_hook_options`
     - `full_supernova_script`
2. `SupernovaMiddleBrain(dspy.Module)` powered by `dspy.ChainOfThought(ConceptBriefToSupernovaPackage)`
   with graceful fallback to `google.genai` (`gemini-3.1-flash-lite-preview` / `gemini-2.5-flash`)
   and offline context-aware synthesis when no live API key is available.
3. `compile_dspy_brain(records: list[dict]) -> str`:
   - Builds `dspy.Example` demonstrations from `TIER_S` and `TIER_A` CAC winners and saves
     `ad-studio/script-brain/data/compiled_dspy_brain.json` (0 Gemini API calls by default,
     with optional `BootstrapFewShot` live compilation).
4. `generate_with_middle_brain(...) -> dict`:
   - Loads compiled demonstrations from `data/compiled_dspy_brain.json`.
   - Runs `SupernovaMiddleBrain` and validates with `compliance_validator.validate_supernova_script`.
   - If any required phrase is missing or forbidden pattern is matched, runs at most 1 retry
     with explicit violation feedback, followed by deterministic compliance enforcement.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & Environment Setup
# ---------------------------------------------------------------------------
SCRIPT_BRAIN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_BRAIN_DIR.parent.parent
DATA_DIR = SCRIPT_BRAIN_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
COMPILED_BRAIN_PATH = DATA_DIR / "compiled_dspy_brain.json"
CONTRASTIVE_PLAYBOOK_PATH = DATA_DIR / "contrastive_playbook.json"


def _load_project_env() -> None:
    """Load environment variables from project root `.env` (`../../.env`)."""
    env_path = PROJECT_ROOT / ".env"
    try:
        from dotenv import load_dotenv

        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
            return
    except ImportError:
        pass

    # Fallback manual parser if python-dotenv is not installed yet
    if env_path.exists():
        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
        except Exception:
            pass


_load_project_env()

# ---------------------------------------------------------------------------
# Import Stanford DSPy (with graceful fallback shim if package is still installing)
# ---------------------------------------------------------------------------
try:
    import dspy  # type: ignore

    HAS_DSPY = True
except Exception:
    HAS_DSPY = False

    class _FallbackField:
        def __init__(self, desc: str = "", prefix: str = "", **kwargs: Any) -> None:
            self.desc = desc
            self.prefix = prefix
            self.kwargs = kwargs

    class _FallbackSignatureMeta(type):
        pass

    class _FallbackSignature(metaclass=_FallbackSignatureMeta):
        """Lightweight DSPy Signature compatibility class when `dspy` is not installed."""

    class _FallbackExample(dict):
        """Lightweight `dspy.Example` compatibility class."""

        def __init__(self, base: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
            super().__init__(base or {}, **kwargs)
            self._input_keys: set[str] = set()

        def __getattr__(self, item: str) -> Any:
            try:
                return self[item]
            except KeyError as exc:
                raise AttributeError(item) from exc

        def __setattr__(self, key: str, value: Any) -> None:
            if key.startswith("_"):
                super().__setattr__(key, value)
            else:
                self[key] = value

        def with_inputs(self, *keys: str) -> "_FallbackExample":
            clone = _FallbackExample(dict(self))
            clone._input_keys = set(keys)
            return clone

        def inputs(self) -> Dict[str, Any]:
            return {k: v for k, v in self.items() if k in self._input_keys}

        def labels(self) -> Dict[str, Any]:
            return {k: v for k, v in self.items() if k not in self._input_keys}

        def toDict(self) -> Dict[str, Any]:
            return dict(self)

    class _FallbackPrediction(_FallbackExample):
        """Lightweight `dspy.Prediction` compatibility class."""

    class _FallbackModule:
        """Lightweight `dspy.Module` compatibility class."""

        def __init__(self) -> None:
            self.demos: List[Any] = []

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.forward(*args, **kwargs)

    class _FallbackChainOfThought:
        """Lightweight `dspy.ChainOfThought` compatibility wrapper."""

        def __init__(self, signature: Any, **kwargs: Any) -> None:
            self.signature = signature
            self.demos: List[Any] = []

        def __call__(self, **kwargs: Any) -> Any:
            raise RuntimeError("Native DSPy LM not active; use fallback execution path.")

    class _FallbackNamespace:
        Signature = _FallbackSignature
        InputField = _FallbackField
        OutputField = _FallbackField
        Example = _FallbackExample
        Prediction = _FallbackPrediction
        Module = _FallbackModule
        ChainOfThought = _FallbackChainOfThought
        settings = type("settings", (), {"lm": None})()

        @staticmethod
        def configure(**kwargs: Any) -> None:
            if "lm" in kwargs:
                _FallbackNamespace.settings.lm = kwargs["lm"]

    dspy = _FallbackNamespace()  # type: ignore

# ---------------------------------------------------------------------------
# Import Google GenAI SDK (`google.genai`) as direct fallback
# ---------------------------------------------------------------------------
try:
    from google import genai as google_genai  # type: ignore
    from google.genai import types as genai_types  # type: ignore

    HAS_GOOGLE_GENAI = True
except Exception:
    google_genai = None  # type: ignore
    genai_types = None  # type: ignore
    HAS_GOOGLE_GENAI = False

# ---------------------------------------------------------------------------
# Constants & Validations
# ---------------------------------------------------------------------------
VALID_LANGUAGES = ("Tamil", "Telugu", "Malayalam", "Kannada", "Hindi")
SOUTH_LANGUAGES = ("Tamil", "Telugu", "Malayalam", "Kannada")
VALID_AI_CHARACTERS = ("Female Humanoid Robot", "Female Robot", "Miss Nova")

DEFAULT_DSPY_MODELS = (
    "gemini/gemini-3.5-flash-lite",
    "gemini/gemini-3.5-flash",
    "gemini/gemini-3.8-flash",
)
DEFAULT_GENAI_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
)


def _get_gemini_api_key(explicit_key: Optional[str] = None) -> str:
    """Resolve `GEMINI_API_KEY` from argument or environment."""
    _load_project_env()
    key = (
        explicit_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or ""
    ).strip()
    if key in ("", "your_gemini_api_key_here", "YOUR_GEMINI_API_KEY"):
        return ""
    return key


def normalize_language(language: Optional[str]) -> str:
    """Normalize target language to one of Tamil, Telugu, Malayalam, Kannada, Hindi."""
    raw = (language or "Tamil").strip()
    for valid in VALID_LANGUAGES:
        if raw.lower() == valid.lower() or valid.lower() in raw.lower():
            return valid
    return raw if raw else "Tamil"


def normalize_ai_character(character: Optional[str]) -> str:
    """Ensure AI teacher speaker label is one of the 3 allowed Supernova character names."""
    raw = (character or "Female Humanoid Robot").strip()
    low = raw.lower()
    if "miss nova" in low or "ms nova" in low or "nova" in low:
        return "Miss Nova"
    if "humanoid" in low:
        return "Female Humanoid Robot"
    if "female robot" in low or low == "robot":
        return "Female Robot"
    if raw in VALID_AI_CHARACTERS:
        return raw
    return "Female Humanoid Robot"


def configure_dspy_gemini(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Configure Stanford DSPy with Gemini (`gemini/gemini-3.1-flash-lite-preview` or `gemini/gemini-2.5-flash`)."""
    resolved_key = _get_gemini_api_key(api_key)
    preferred_model = (
        model_name
        or os.environ.get("DSPY_GEMINI_MODEL")
        or os.environ.get("GEMINI_MODEL")
        or DEFAULT_DSPY_MODELS[0]
    ).strip()
    if not preferred_model.startswith("gemini/"):
        dspy_model_id = f"gemini/{preferred_model}"
    else:
        dspy_model_id = preferred_model

    if not resolved_key:
        return {
            "configured": False,
            "has_dspy": HAS_DSPY,
            "has_google_genai": HAS_GOOGLE_GENAI,
            "model": dspy_model_id,
            "reason": "GEMINI_API_KEY not set or placeholder",
        }

    # Ensure environment variables expected by LiteLLM / DSPy / Google GenAI are set
    os.environ["GEMINI_API_KEY"] = resolved_key

    if HAS_DSPY and hasattr(dspy, "LM"):
        candidate_models = [dspy_model_id] + [
            m for m in DEFAULT_DSPY_MODELS if m != dspy_model_id
        ]
        for candidate in candidate_models:
            try:
                lm = dspy.LM(
                    candidate,
                    api_key=resolved_key,
                    temperature=0.4,
                    max_tokens=4096,
                )
                dspy.configure(lm=lm)
                return {
                    "configured": True,
                    "engine": "dspy",
                    "has_dspy": True,
                    "has_google_genai": HAS_GOOGLE_GENAI,
                    "model": candidate,
                }
            except Exception as exc:
                logger.debug("DSPy LM init failed for %s: %s", candidate, exc)

    return {
        "configured": bool(HAS_GOOGLE_GENAI and resolved_key),
        "engine": "google.genai" if HAS_GOOGLE_GENAI else "deterministic_fallback",
        "has_dspy": HAS_DSPY,
        "has_google_genai": HAS_GOOGLE_GENAI,
        "model": preferred_model.replace("gemini/", ""),
    }


# ---------------------------------------------------------------------------
# 2. DSPy Signature: ConceptBriefToSupernovaPackage
# ---------------------------------------------------------------------------
class ConceptBriefToSupernovaPackage(dspy.Signature):
    """Act as Supernova's Principal Creative Strategist & Low-CAC Ad Script Architect.

    Expand a short 1-3 line `concept_brief` into a complete, production-ready 45-second
    Supernova Spoken English App performance ad package modeled on the Top 3 proven
    CAC-winning scripts (`matched_winner_scripts`) and their visual setups (`matched_winner_visuals`),
    while obeying all `contrastive_playbook_rules`.

    NON-NEGOTIABLE SUPERNOVA HOUSE RULES:
    1. AI TEACHER & HUMAN SPEAKER LABELS:
       - Always use the exact `ai_teacher_character` label (`Female Humanoid Robot`, `Female Robot`, or `Miss Nova`).
       - NEVER use standalone `Robot:` as a speaker label, and never have the human character address the AI teacher as "Female Humanoid Robot" out loud in spoken dialogue (call her "Ma'am" or speak naturally).
       - Always give the human speaker a natural Indian first name (e.g., `Swapna`, `Lakshmi`, `Priya`, `Karthik`, `Rahul`, `Arjun`, `Divya`) as their speaker label in both `Characters` and `full_supernova_script` (NEVER use generic role labels like `Homemaker Mother:`, `Young Man:`, or `Young Woman:` as the speaker name).
    2. VISUAL DESCRIPTION HEADERS:
       - In `visual_description_block`, use clean headers (`Visual Description:`, `Split Screen & Characters:`, `Hook visuals:`, `Body visuals:`, `Video flow:`, `Characters`).
       - NEVER mention "Left", "Right", or "Stitched Frame" in the visual headers.
    3. MANDATORY HARD-LOCKED PHRASES (ALL 7 MUST APPEAR VERBATIM IN `full_supernova_script`):
       - "improve your spoken English" (or "improve my spoken English" / "improve spoken English")
       - "within 30 days"
       - "15 to 20 minutes"
       - "over 1 crore Indians"
       - "through <language> itself" (e.g., "through Tamil itself", "through Telugu itself", "through Hindi itself")
       - "Install Now button given below"
       - "today itself"
    4. STRICTLY FORBIDDEN PATTERNS (ZERO TOLERANCE):
       - NEVER say "learn English", "learning English", "learned English", "teach English", or "teaches spoken English" (always say "improve your spoken English").
       - NEVER say "10 minutes" or "10 to 15 minutes" (always say "15 to 20 minutes").
       - NEVER mention rupee prices (Rs, INR, ₹, rupees) or "free trial".
       - NEVER mention competitor apps (MySivi, Mysore, SpeakX, Stimuler, Duolingo, Cambly).
    5. 4-BLOCK STORY PACING:
       - Block 1: Hook & Mistake [0-5s] (Turns 1-2: Relatable spoken English mistake or high-stakes freeze + immediate AI teacher interruption).
       - Block 2: Real-Life Pain & Why Old Methods Failed [5-15s] (Turns 3-5: Real-world embarrassment + why grammar books/YouTube videos failed because there was no speaking partner).
       - Block 3: Live On-The-Spot Fix & 4 Supernova USPs [15-35s] (Turns 6-9: AI teacher makes the human practice the corrected sentence live on the spot + delivers the 4 Supernova USPs in the first half/middle).
       - Block 4: Active Desire & CTA [35-45s] (Turns 10-12: Human asks how to start right now + AI teacher delivers the "Install Now button given below" + "today itself" close).
    6. NO MARKDOWN:
       - Output clean plain text without `**` or `#` symbols so it can be pasted directly into a Google Doc.
    """

    concept_brief: str = dspy.InputField(
        desc="The user's short 1-3 line brief idea describing the protagonist, setting, and English struggle."
    )
    language: str = dspy.InputField(
        desc="Target regional language: Tamil, Telugu, Malayalam, Kannada, or Hindi."
    )
    ai_teacher_character: str = dspy.InputField(
        desc="Must be 'Female Humanoid Robot', 'Female Robot', or 'Miss Nova'."
    )
    matched_winner_scripts: str = dspy.InputField(
        desc="Top 3 proven CAC-winning scripts (story_blocks + full script + CAC win stats)."
    )
    matched_winner_visuals: str = dspy.InputField(
        desc="The visual_blueprint (Hook visuals, Body visuals, Split Screen, Characters) from those Top 3 CAC winners so the model writes a proven-style Visual Description for the new script."
    )
    contrastive_playbook_rules: str = dspy.InputField(
        desc="Winner-vs-Loser insights from contrastive_trainer.py + high-CAC loser patterns to avoid."
    )

    four_block_thinking_plan: str = dspy.OutputField(
        desc=(
            "Detailed 4-part Supernova plan: "
            "Block 1: Hook & Mistake [0-5s], "
            "Block 2: Real-Life Pain & Why Old Methods Failed [5-15s], "
            "Block 3: Live On-The-Spot Fix & 4 Supernova USPs [15-35s], "
            "Block 4: Active Desire & CTA [35-45s]."
        )
    )
    visual_description_block: str = dspy.OutputField(
        desc=(
            "Complete production-ready Visual Description for the new script modeled on the winners "
            "(Visual Description:, Hook visuals:, Body visuals:, Video flow:, Characters bullet list)."
        )
    )
    three_hook_options: list[str] = dspy.OutputField(
        desc="List of 3 coupled Hook + Bridge (Line 1 + Line 2) variations that all converge cleanly onto Line 3 of the script."
    )
    full_supernova_script: str = dspy.OutputField(
        desc=(
            "Complete word-for-word Supernova spoken dialogue (Speaker: dialogue) containing all mandatory "
            "Supernova phrases ('improve your spoken English', 'within 30 days', '15 to 20 minutes', "
            "'over 1 crore Indians', 'through <Language> itself', 'Install Now button given below', 'today itself')."
        )
    )


# ---------------------------------------------------------------------------
# Formatting Helpers for Structured Inputs
# ---------------------------------------------------------------------------
def _clean_plain_text(text: Any) -> str:
    """Strip markdown `**` and `#` symbols while preserving clean line breaks."""
    if text is None:
        return ""
    s = str(text).replace("**", "")
    s = re.sub(r"(?m)^\s*#{1,6}\s*", "", s)
    s = s.replace("#", "")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def format_winner_scripts_for_prompt(winners: Any) -> str:
    """Format `matched_winner_scripts` (list of dicts or string) into a rich prompt string."""
    if isinstance(winners, str):
        return winners.strip() or "No historical winner scripts provided; use standard Supernova Tier-S structure."
    if not isinstance(winners, list) or not winners:
        return "No historical winner scripts provided; use standard Supernova Tier-S structure."

    blocks: List[str] = []
    for idx, item in enumerate(winners[:3], start=1):
        if not isinstance(item, dict):
            blocks.append(f"Winner #{idx}: {str(item)}")
            continue
        rec = item.get("record", item) if isinstance(item.get("record"), dict) else item
        ad_code = item.get("ad_code") or rec.get("ad_code") or f"WINNER_{idx}"
        title = item.get("title") or rec.get("title") or "Proven Supernova Script"
        tier = item.get("cac_tier") or rec.get("cac_tier") or "TIER_S"
        lang = item.get("language") or rec.get("language") or "Tamil"
        cac_metrics = item.get("cac_metrics") or rec.get("cac_metrics") or {}
        vs_avg = cac_metrics.get("cac_vs_avg_pct")
        first_cac = cac_metrics.get("first_pay_cac")
        second_cac = cac_metrics.get("second_pay_cac_7d")
        story_blocks = item.get("story_blocks") or rec.get("story_blocks") or {}
        english_script = (
            item.get("english_script")
            or rec.get("english_script")
            or item.get("script")
            or rec.get("script")
            or ""
        )

        cac_summary_parts = [f"Tier={tier}", f"Language={lang}"]
        if isinstance(vs_avg, (int, float)):
            cac_summary_parts.append(f"CAC vs Avg={vs_avg:+.1f}%")
        if isinstance(first_cac, (int, float)):
            cac_summary_parts.append(f"1st-Pay CAC=Rs.{first_cac}")
        if isinstance(second_cac, (int, float)):
            cac_summary_parts.append(f"7d 2nd-Pay CAC=Rs.{second_cac}")

        has_story_blocks = isinstance(story_blocks, dict) and any(
            str(story_blocks.get(k) or "").strip()
            for k in (
                "block_1_hook",
                "block_2_pain_and_old_methods",
                "block_3_live_fix_and_usps",
                "block_4_desire_and_cta",
            )
        )
        if has_story_blocks:
            b1_txt = str(story_blocks.get("block_1_hook") or "").strip()[:220]
            b2_txt = str(story_blocks.get("block_2_pain_and_old_methods") or "").strip()[:280]
            b3_txt = str(story_blocks.get("block_3_live_fix_and_usps") or "").strip()[:280]
            b4_txt = str(story_blocks.get("block_4_desire_and_cta") or "").strip()[:200]
            sb_lines = [
                f"  - Block 1 (Hook): {b1_txt}",
                f"  - Block 2 (Pain & Old Methods): {b2_txt}",
                f"  - Block 3 (Live Fix & USPs): {b3_txt}",
                f"  - Block 4 (Desire & CTA): {b4_txt}",
            ]
            sb_text = "\n".join(sb_lines)
            entry_text = (
                f"[Winner {idx}: {ad_code} — {title[:80]} ({', '.join(cac_summary_parts)})]\n"
                f"Story Blocks:\n{sb_text}"
            )
        else:
            entry_text = (
                f"[Winner {idx}: {ad_code} — {title[:80]} ({', '.join(cac_summary_parts)})]\n"
                f"Proven Dialogue:\n{str(english_script).strip()[:700]}"
            )
        blocks.append(entry_text.strip())

    return "\n\n---\n\n".join(blocks)


def format_winner_visuals_for_prompt(visuals: Any, winners_fallback: Any = None) -> str:
    """Format `matched_winner_visuals` (list of dicts or string) into a clean blueprint reference."""
    if isinstance(visuals, str) and visuals.strip():
        return visuals.strip()

    source_list = visuals if isinstance(visuals, list) and visuals else (
        winners_fallback if isinstance(winners_fallback, list) else []
    )
    if not source_list:
        return (
            "Proven Supernova Visual Pattern:\n"
            "- Split Screen: Horizontal Split Screen (Protagonist in real-life setting on top panel, "
            "AI Teacher in futuristic studio on bottom panel).\n"
            "- Hook visuals: Protagonist hesitating mid-sentence in English with visible embarrassment while AI Teacher pops in immediately.\n"
            "- Body visuals: Interactive live voice correction on phone screen with green speech-confidence bars.\n"
            "- Characters: Relatable Indian protagonist in authentic everyday attire + AI Teacher character."
        )

    out: List[str] = []
    for idx, item in enumerate(source_list[:3], start=1):
        if not isinstance(item, dict):
            out.append(f"Winner #{idx} Visuals: {str(item)[:220]}")
            continue
        ad_code = item.get("ad_code") or f"WINNER_{idx}"
        tier = item.get("cac_tier") or "TIER_S"
        vb = item.get("visual_blueprint") if isinstance(item.get("visual_blueprint"), dict) else item
        split_type = str(vb.get("split_screen_type") or "Horizontal Split Screen").strip()[:180]
        hook_v = str(vb.get("hook_visuals") or "").strip()[:200]
        body_v = str(vb.get("body_visuals") or "").strip()[:200]
        chars = str(vb.get("characters_description") or "").strip()[:160]

        lines = [
            f"[Winner {idx} ({ad_code}, {tier}) Visual Blueprint]",
            f"  Split Screen: {split_type}",
        ]
        if hook_v and hook_v != split_type:
            lines.append(f"  Hook visuals: {hook_v}")
        if body_v and body_v != hook_v:
            lines.append(f"  Body visuals: {body_v}")
        if chars:
            lines.append(f"  Characters: {chars}")
        out.append("\n".join(lines))

    return "\n\n".join(out)


def format_contrastive_rules_for_prompt(
    contrastive_input: Any,
    losers_to_avoid: Any = None,
    violation_feedback: Optional[str] = None,
) -> str:
    """Format `contrastive_playbook_rules` + `losers_to_avoid` + optional retry violation feedback."""
    sections: List[str] = []

    if isinstance(contrastive_input, str) and contrastive_input.strip():
        sections.append(contrastive_input.strip()[:900])
    elif isinstance(contrastive_input, dict):
        pb = contrastive_input.get("contrastive_playbook", contrastive_input)
        if isinstance(pb, dict):
            rules = (
                pb.get("rules")
                or pb.get("statistical_rules")
                or pb.get("thinking_rules")
                or pb.get("winner_vs_loser_rules")
                or pb.get("key_insights")
                or []
            )
            if isinstance(rules, list) and rules:
                sections.append(
                    "Contrastive Winner-vs-Loser Rules:\n"
                    + "\n".join(f"- {str(r)[:180]}" for r in rules[:5])
                )
            elif isinstance(rules, str) and rules.strip():
                sections.append(f"Contrastive Winner-vs-Loser Rules:\n{rules.strip()[:600]}")

            anti = pb.get("anti_patterns")
            if isinstance(anti, list) and anti:
                sections.append(
                    "Loser Anti-Patterns to Avoid:\n"
                    + "\n".join(f"- {str(a)[:160]}" for a in anti[:3])
                )

        embedded_losers = contrastive_input.get("losers_to_avoid")
        if embedded_losers and not losers_to_avoid:
            losers_to_avoid = embedded_losers

    if isinstance(losers_to_avoid, list) and losers_to_avoid:
        loser_lines: List[str] = ["High-CAC Loser Patterns to Avoid (TIER_B):"]
        for idx, l_item in enumerate(losers_to_avoid[:2], start=1):
            if isinstance(l_item, dict):
                code = l_item.get("ad_code") or f"LOSER_{idx}"
                tags = l_item.get("metadata_tags") or {}
                sb = l_item.get("story_blocks") or {}
                hook_txt = sb.get("block_1_hook") or (l_item.get("english_script") or "")[:160]
                loser_lines.append(
                    f"- {code} (Hook Words={tags.get('hook_word_count', 'high')}, "
                    f"AI Entry Turn={tags.get('ai_teacher_entry_turn', 'late')}): "
                    f"Avoid slow/generic opening like: {hook_txt}"
                )
            else:
                loser_lines.append(f"- {str(l_item)}")
        sections.append("\n".join(loser_lines))

    if not sections:
        sections.append(
            "Contrastive Winner-vs-Loser Rules:\n"
            "- Open immediately mid-mistake or high-stakes freeze in Turn 1 (6-14 words max).\n"
            "- Have the AI Teacher interrupt in Turn 2 to correct the exact spoken mistake.\n"
            "- Show live on-the-spot speaking practice before pitching app features.\n"
            "- Weave all 4 Supernova USPs naturally in Block 3 and end with the urgent Install Now CTA."
        )

    if violation_feedback:
        sections.append(
            "CRITICAL COMPLIANCE RETRY FEEDBACK (MUST FIX ALL ITEMS BELOW IN THIS ATTEMPT):\n"
            f"{violation_feedback}"
        )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Scenario Analysis & Deterministic Supernova Synthesis (Fallback & Repair)
# ---------------------------------------------------------------------------
def _infer_scenario_context(concept_brief: str, language: str, ai_teacher_character: str) -> Dict[str, str]:
    """Extract protagonist name, role, setting, and specific English mistake from `concept_brief`."""
    brief = (concept_brief or "").strip()
    low = brief.lower()

    # Check if a specific name is mentioned in the brief
    name_match = re.search(
        r"\b(Rahul|Karthik|Arun|Suresh|Ramesh|Priya|Lakshmi|Anitha|Divya|Meena|Kavitha|Ravi|Deepak|Sanjay|Pooja|Sneha|Vikram|Neha|Rohan|Anjali)\b",
        brief,
        re.IGNORECASE,
    )

    if any(k in low for k in ("standup", "office", "manager", "meeting", "client", "project", "it ", "software", "colleague", "promotion", "appraisal")):
        protagonist = name_match.group(1).capitalize() if name_match else "Karthik"
        role_desc = f"{protagonist} (27-year-old IT software engineer sitting at an office desk with laptop open during a live team call)"
        setting = "Modern IT office desk during a daily standup / client video call"
        broken_line = "Yesterday my task is completed... today I am doing bug fixing only, sir."
        corrected_line = "Yesterday I completed my module, and today I am resolving the priority bugs."
        pain_line = (
            "Every single standup call, I know the technical work inside out. "
            "But when my manager asks me to explain in English, I freeze up while my juniors speak fluently and get promoted."
        )
        old_methods_line = (
            "I watched so many grammar YouTube channels at night and memorized word lists, "
            "but the moment someone speaks to me in English on a live call, my mind goes completely blank!"
        )
        hook_opt_b_l1 = (
            f"{protagonist}: Sir, regarding the deployment... actual problem is... I am checking the server logs only."
        )
        hook_opt_b_l2 = (
            f"{ai_teacher_character}: Pause right there, {protagonist}! Never say 'I am checking only' on a client call — let's fix how you present your update right now."
        )
        hook_opt_c_l1 = (
            f"{protagonist}: Why do my juniors with less coding experience speak so confidently in office meetings while I stay on mute?"
        )
        hook_opt_c_l2 = (
            f"{ai_teacher_character}: Because fluency comes from live speaking practice, {protagonist} — not silently memorizing grammar rules!"
        )
    elif any(k in low for k in ("ptm", "school", "teacher", "mother", "mom", "kid", "child", "son", "daughter", "homework", "diary", "parent")):
        protagonist = name_match.group(1).capitalize() if name_match else "Lakshmi"
        role_desc = f"{protagonist} (31-year-old homemaker mother in a neat cotton saree holding her child's school diary)"
        setting = "Warm Indian living room / school PTM table with a school notebook open"
        broken_line = "Ma'am, my son is doing all homework nicely, but in class he is very shy only."
        corrected_line = "Ma'am, my son completes his homework regularly, though he feels a little shy speaking up in class."
        pain_line = (
            "At every parent-teacher meeting, other mothers talk so smoothly in English. "
            "My own child looks at me waiting for me to speak, and I feel so embarrassed stumbling over basic words."
        )
        old_methods_line = (
            "I bought spoken English books and watched recorded videos while cooking, "
            "but without anyone to talk to me at home, my hesitation never went away!"
        )
        hook_opt_b_l1 = (
            f"{protagonist}: Amma, tomorrow is my school PTM... please don't stay quiet when my class teacher speaks in English!"
        )
        hook_opt_b_l2 = (
            f"{ai_teacher_character}: Don't worry, {protagonist}! Let's practice the exact sentences you need for tomorrow's PTM right now."
        )
        hook_opt_c_l1 = (
            f"{protagonist}: I understood everything my child's teacher asked in English, but my tongue froze when I tried to reply."
        )
        hook_opt_c_l2 = (
            f"{ai_teacher_character}: That freeze happens when you translate in your head, {protagonist} — let me help you speak out loud with confidence!"
        )
    elif any(k in low for k in ("interview", "hr", "job", "resume", "fresher", "Hiring", "recruiter", "salary", "career")):
        protagonist = name_match.group(1).capitalize() if name_match else "Arun"
        role_desc = f"{protagonist} (24-year-old job seeker in a crisp formal shirt sitting across a laptop interview screen)"
        setting = "Study table setup during a live HR / technical job interview"
        broken_line = "Myself Arun, I am passed out from mechanical engineering in 2023 batch, ma'am."
        corrected_line = "Hi, I'm Arun, and I graduated in Mechanical Engineering in 2023."
        pain_line = (
            "I cleared every technical test, but in the final HR round I always get rejected "
            "because I fumble in the first two minutes of speaking English."
        )
        old_methods_line = (
            "I attended offline coaching classes and mugged up grammar rules, "
            "but nobody practiced real interview conversations with me one-on-one!"
        )
        hook_opt_b_l1 = (
            f"{protagonist}: Tell me about yourself... Every time an HR interviewer asks me those four words, my heart starts racing!"
        )
        hook_opt_b_l2 = (
            f"{ai_teacher_character}: Take a breath, {protagonist}! Let's replace that nervous intro with a confident 10-second opening right now."
        )
        hook_opt_c_l1 = (
            f"{protagonist}: Sir, I have all the technical skills for this role, I just struggle to explain my projects in fluent English."
        )
        hook_opt_c_l2 = (
            f"{ai_teacher_character}: Your skills deserve to be heard, {protagonist} — let's fix how you explain your project in one quick practice turn!"
        )
    else:
        protagonist = name_match.group(1).capitalize() if name_match else "Rahul"
        role_desc = f"{protagonist} (28-year-old relatable young Indian adult in everyday casual wear holding a smartphone)"
        setting = f"Everyday real-life setting matching brief: {brief[:90]}"
        broken_line = "Actually I am knowing the answer in my mind, but when speaking English it is coming very slowly only."
        corrected_line = "I know the exact answer in my head, and now I can express it clearly in English."
        pain_line = (
            f"Regarding '{brief[:80]}', I always know what I want to say in {language}, "
            "but the moment I have to speak in English in front of people, I hesitate and lose confidence."
        )
        old_methods_line = (
            "I tried watching English videos and reading grammar tips on my phone, "
            "but without a real speaking partner who corrects me without judging, I kept getting stuck!"
        )
        hook_opt_b_l1 = (
            f"{protagonist}: Have you ever stayed quiet in an important moment just because you were scared of making an English mistake?"
        )
        hook_opt_b_l2 = (
            f"{ai_teacher_character}: Stop holding back, {protagonist}! Let's practice that exact conversation together right now."
        )
        hook_opt_c_l1 = (
            f"{protagonist}: Every time someone suddenly speaks to me in English, I start translating word by word in my head and freeze."
        )
        hook_opt_c_l2 = (
            f"{ai_teacher_character}: Let's stop the mental translation, {protagonist} — repeat this natural English sentence with me right now!"
        )

    return {
        "protagonist": protagonist,
        "role_desc": role_desc,
        "setting": setting,
        "broken_line": broken_line,
        "corrected_line": corrected_line,
        "pain_line": pain_line,
        "old_methods_line": old_methods_line,
        "hook_opt_b_l1": hook_opt_b_l1,
        "hook_opt_b_l2": hook_opt_b_l2,
        "hook_opt_c_l1": hook_opt_c_l1,
        "hook_opt_c_l2": hook_opt_c_l2,
    }


def _synthesize_supernova_package_deterministic(
    concept_brief: str,
    language: str,
    ai_teacher_character: str,
    matched_winner_scripts: Any = None,
    matched_winner_visuals: Any = None,
    contrastive_playbook_rules: Any = None,
) -> Dict[str, Any]:
    """Build a complete, 100% rule-compliant Supernova script package modeled on the matched winners."""
    lang = normalize_language(language)
    ai_char = normalize_ai_character(ai_teacher_character)
    ctx = _infer_scenario_context(concept_brief, lang, ai_char)
    p = ctx["protagonist"]

    four_block_plan = (
        f"Block 1: Hook & Mistake [0-5s]\n"
        f"Open mid-situation in ({ctx['setting']}) where {p} makes the relatable spoken-English mistake: "
        f"\"{ctx['broken_line']}\" {ai_char} immediately interrupts in Turn 2 with a warm, zero-judgment correction.\n\n"
        f"Block 2: Real-Life Pain & Why Old Methods Failed [5-15s]\n"
        f"{p} vents the real-world frustration from the brief ({concept_brief.strip()}) and explains why passive "
        f"grammar videos and books failed — there was no judgment-free partner for live speaking practice.\n\n"
        f"Block 3: Live On-The-Spot Fix & 4 Supernova USPs [15-35s]\n"
        f"{ai_char} conducts a live 1-sentence speaking drill where {p} speaks the corrected sentence confidently, "
        f"then delivers all 4 Supernova USPs: personalized level test, practicing 15 to 20 minutes daily through {lang} itself, "
        f"trusted by over 1 crore Indians to improve your spoken English within 30 days.\n\n"
        f"Block 4: Active Desire & CTA [35-45s]\n"
        f"{p} expresses immediate eagerness to start practicing today, and {ai_char} closes with the hard-locked CTA: "
        f"click the Install Now button given below and start your journey today itself."
    )

    visual_description = (
        f"Visual Description:\n"
        f"Hook visuals: Horizontal split screen. Top panel shows {p} in {ctx['setting']}, hesitating visibly "
        f"while speaking broken English. Bottom panel shows {ai_char} appearing with an attentive, encouraging expression.\n"
        f"Body visuals: Split screen continues with live Supernova app interface overlay showing real-time voice wave "
        f"feedback as {p} repeats the corrected English sentence confidently, followed by the level-test and daily practice badges.\n"
        f"Video flow: 3-second spoken-mistake pattern interrupt -> immediate AI Teacher correction -> relatable pain & why old methods failed -> live 1-sentence speaking demo + 4 Supernova USPs -> urgent Install Now CTA.\n"
        f"Characters:\n"
        f"- {ctx['role_desc']}\n"
        f"- {ai_char}: Warm, expressive 3D AI English teacher inside the Supernova Spoken English app guiding {p} through {lang} itself."
    )

    option_a = (
        f"Option A (Spoken Mistake Correction Hook — Converges onto Line 3):\n"
        f"Line 1 — {p}: \"{ctx['broken_line']}\"\n"
        f"Line 2 — {ai_char}: \"Hold on, {p}! Instead of saying that, say: '{ctx['corrected_line']}' — it sounds so much more natural!\""
    )
    option_b = (
        f"Option B (High-Stakes Freeze Hook — Converges onto Line 3):\n"
        f"Line 1 — {ctx['hook_opt_b_l1']}\n"
        f"Line 2 — {ctx['hook_opt_b_l2']}"
    )
    option_c = (
        f"Option C (Relatable Pain Question Hook — Converges onto Line 3):\n"
        f"Line 1 — {ctx['hook_opt_c_l1']}\n"
        f"Line 2 — {ctx['hook_opt_c_l2']}"
    )
    three_hooks = [option_a, option_b, option_c]

    script_lines = [
        f"{p}: {ctx['broken_line']}",
        f"{ai_char}: Hold on, {p}! Say: \"{ctx['corrected_line']}\"",
        f"{p}: {ctx['pain_line']}",
        f"{ai_char}: Stop watching grammar videos! Repeat that sentence out loud with me right now.",
        f"{p}: \"{ctx['corrected_line']}\" — wow, speaking it with you felt so easy!",
        f"{ai_char}: With Supernova, practice 15 to 20 minutes daily through {lang} itself to improve your spoken English!",
        f"{p}: Really? You explain my mistakes through {lang} itself without judging me?",
        f"{ai_char}: Yes! Over 1 crore Indians use Supernova to speak confidently within 30 days.",
        f"{p}: This daily speaking partner is exactly what I needed! How do I start?",
        f"{ai_char}: Tap the Install Now button given below and start practicing today itself!",
    ]
    full_script = "\n".join(script_lines)

    return {
        "reasoning": (
            f"Analyzed brief ('{concept_brief}') for {lang} with {ai_char}. "
            "Selected immediate Turn-1 spoken mistake hook + Turn-2 AI Teacher interruption modeled on Tier-S CAC winners, "
            "bridged through real-life hesitation and failure of passive methods into a live speaking correction drill "
            "and all 7 mandatory Supernova phrases."
        ),
        "four_block_thinking_plan": four_block_plan,
        "visual_description_block": visual_description,
        "three_hook_options": three_hooks,
        "full_supernova_script": full_script,
    }


def _enforce_supernova_compliance(
    script_text: str,
    language: str,
    ai_teacher_character: str,
) -> str:
    """Deterministic post-processor that repairs any forbidden regex or missing phrase in `script_text`."""
    lang = normalize_language(language)
    ai_char = normalize_ai_character(ai_teacher_character)
    text = _clean_plain_text(script_text)

    if not text:
        return _synthesize_supernova_package_deterministic(
            "Relatable spoken English struggle in daily life",
            lang,
            ai_char,
        )["full_supernova_script"]

    # 1. Repair standalone "Robot:" speaker label
    text = re.sub(r"(?m)^\s*Robot\s*:", f"{ai_char}:", text, flags=re.IGNORECASE)

    # 2. Repair forbidden "learn/teach (spoken) English"
    text = re.sub(
        r"\b(learn|learning|learned|learnt|teach|teaching|teaches|taught)\s+(?:spoken\s+)?english\b",
        "improve your spoken English",
        text,
        flags=re.IGNORECASE,
    )

    # 3. Repair forbidden "10 minutes" or "10 to 15 minutes"
    text = re.sub(
        r"\b10(?:\s*(?:to|-)\s*15)?\s*minutes\b",
        "15 to 20 minutes",
        text,
        flags=re.IGNORECASE,
    )

    # 4. Repair forbidden pricing or free trial mentions
    text = re.sub(
        r"(\b(?:rs\.?|inr|rupees?)\s*\d+|₹\s*\d+|\d+\s*rupees?\b|\bfree\s+trial\b)",
        "a simple daily session",
        text,
        flags=re.IGNORECASE,
    )

    # 5. Repair forbidden competitor names
    text = re.sub(
        r"\b(mysivi|mysore|speakx|stimuler|duolingo|cambly)\b",
        "Supernova",
        text,
        flags=re.IGNORECASE,
    )

    # 6. Check all 7 required phrases and weave any missing ones into the AI Teacher's USP/CTA lines
    from compliance_validator import validate_supernova_script

    check = validate_supernova_script(text, language=lang, ai_teacher_character=ai_char)
    if check.get("passed"):
        return text

    missing = set(check.get("missing_phrases", []))
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    usp_missing_bits: List[str] = []
    if "15 to 20 minutes" in missing:
        usp_missing_bits.append("practice for just 15 to 20 minutes daily")
    if f"through {lang} itself" in missing or any(m.startswith("through ") for m in missing):
        usp_missing_bits.append(f"guided step by step through {lang} itself")
    if "over 1 crore Indians" in missing:
        usp_missing_bits.append("trusted by over 1 crore Indians")
    if "improve your spoken English" in missing:
        usp_missing_bits.append("to improve your spoken English")
    if "within 30 days" in missing:
        usp_missing_bits.append("within 30 days")

    if usp_missing_bits:
        usp_line = (
            f"{ai_char}: With Supernova, you {', '.join(usp_missing_bits)} "
            f"so you never freeze in real conversations!"
        )
        # Insert right before the last 2 lines (before final CTA) if possible
        insert_pos = max(0, len(lines) - 2)
        lines.insert(insert_pos, usp_line)

    # Re-check CTA phrases ("Install Now button given below" and "today itself")
    joined_now = "\n".join(lines)
    recheck = validate_supernova_script(joined_now, language=lang, ai_teacher_character=ai_char)
    still_missing = set(recheck.get("missing_phrases", []))

    if still_missing:
        cta_line = (
            f"{ai_char}: Tap the Install Now button given below to practice for 15 to 20 minutes a day "
            f"through {lang} itself, join over 1 crore Indians, and improve your spoken English within 30 days starting today itself!"
        )
        if lines and lines[-1].lower().startswith(ai_char.lower()):
            lines[-1] = cta_line
        else:
            lines.append(cta_line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. SupernovaMiddleBrain(dspy.Module)
# ---------------------------------------------------------------------------
class SupernovaMiddleBrain(dspy.Module):
    """Stanford DSPy Chain-of-Thought Middle Brain for Supernova Ad Script Generation."""

    def __init__(self, compiled_path: Optional[Union[str, Path]] = None) -> None:
        super().__init__()
        self.generate = dspy.ChainOfThought(ConceptBriefToSupernovaPackage)
        self.compiled_path = Path(compiled_path) if compiled_path else COMPILED_BRAIN_PATH
        self.compiled_demos: List[Dict[str, Any]] = []
        self._load_compiled_demos()

    def _load_compiled_demos(self) -> None:
        """Load compiled few-shot demonstrations from `data/compiled_dspy_brain.json` into `self.generate.demos`."""
        if not self.compiled_path.exists():
            return
        try:
            payload = json.loads(self.compiled_path.read_text(encoding="utf-8"))
            raw_demos = (
                payload.get("demos")
                or payload.get("generate", {}).get("demos")
                or payload.get("examples")
                or []
            )
            if not isinstance(raw_demos, list):
                return

            self.compiled_demos = raw_demos[:1]
            dspy_examples: List[Any] = []
            for d in raw_demos[:1]:
                if isinstance(d, dict):
                    ex = dspy.Example(**d).with_inputs(
                        "concept_brief",
                        "language",
                        "ai_teacher_character",
                        "matched_winner_scripts",
                        "matched_winner_visuals",
                        "contrastive_playbook_rules",
                    )
                    dspy_examples.append(ex)
            if hasattr(self.generate, "demos"):
                self.generate.demos = dspy_examples
        except Exception as exc:
            logger.debug("Could not load compiled DSPy demos from %s: %s", self.compiled_path, exc)

    def forward(
        self,
        concept_brief: str,
        language: str = "Tamil",
        ai_teacher_character: str = "Female Humanoid Robot",
        matched_winner_scripts: Any = "",
        matched_winner_visuals: Any = "",
        contrastive_playbook_rules: Any = "",
    ) -> Any:
        """Execute the DSPy `ChainOfThought(ConceptBriefToSupernovaPackage)` forward pass."""
        norm_lang = normalize_language(language)
        norm_char = normalize_ai_character(ai_teacher_character)
        formatted_winners = format_winner_scripts_for_prompt(matched_winner_scripts)
        formatted_visuals = format_winner_visuals_for_prompt(
            matched_winner_visuals, winners_fallback=matched_winner_scripts
        )
        formatted_rules = format_contrastive_rules_for_prompt(contrastive_playbook_rules)

        # Tier 1: Native Stanford DSPy ChainOfThought forward pass
        if HAS_DSPY and getattr(getattr(dspy, "settings", None), "lm", None) is not None:
            try:
                pred = self.generate(
                    concept_brief=concept_brief,
                    language=norm_lang,
                    ai_teacher_character=norm_char,
                    matched_winner_scripts=formatted_winners,
                    matched_winner_visuals=formatted_visuals,
                    contrastive_playbook_rules=formatted_rules,
                )
                hooks_raw = getattr(pred, "three_hook_options", [])
                if isinstance(hooks_raw, str):
                    hooks_list = [
                        b.strip()
                        for b in re.split(r"\n\s*(?=Option\s+[ABC123]\b)", hooks_raw)
                        if b.strip()
                    ]
                    if len(hooks_list) < 3:
                        hooks_list = [ln.strip() for ln in hooks_raw.splitlines() if ln.strip()][:3]
                elif isinstance(hooks_raw, list):
                    hooks_list = [str(x).strip() for x in hooks_raw if str(x).strip()]
                else:
                    hooks_list = []

                setattr(pred, "three_hook_options", hooks_list)
                setattr(pred, "_engine_used", "dspy")
                return pred
            except Exception as dspy_exc:
                logger.warning("DSPy ChainOfThought call failed, falling back to google.genai: %s", dspy_exc)

        # Tier 2: Direct `google.genai` fallback with compiled DSPy demonstrations
        genai_result = _call_google_genai_fallback(
            concept_brief=concept_brief,
            language=norm_lang,
            ai_teacher_character=norm_char,
            matched_winner_scripts=formatted_winners,
            matched_winner_visuals=formatted_visuals,
            contrastive_playbook_rules=formatted_rules,
            compiled_demos=self.compiled_demos,
        )
        if genai_result is not None:
            pred = dspy.Prediction(**genai_result)
            setattr(pred, "_engine_used", "google.genai")
            return pred

        # Tier 3: Deterministic context-aware Supernova synthesis (offline / no API key)
        synth = _synthesize_supernova_package_deterministic(
            concept_brief=concept_brief,
            language=norm_lang,
            ai_teacher_character=norm_char,
            matched_winner_scripts=matched_winner_scripts,
            matched_winner_visuals=matched_winner_visuals,
            contrastive_playbook_rules=contrastive_playbook_rules,
        )
        pred = dspy.Prediction(**synth)
        setattr(pred, "_engine_used", "deterministic_fallback")
        return pred


def _call_google_genai_fallback(
    concept_brief: str,
    language: str,
    ai_teacher_character: str,
    matched_winner_scripts: str,
    matched_winner_visuals: str,
    contrastive_playbook_rules: str,
    compiled_demos: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Direct `google.genai` fallback using the DSPy signature structure and compiled few-shot demos."""
    api_key = _get_gemini_api_key()
    if not HAS_GOOGLE_GENAI or not api_key or google_genai is None:
        return None

    demos_text = ""
    if compiled_demos:
        demo_blocks: List[str] = []
        for idx, d in enumerate(compiled_demos[:2], start=1):
            demo_blocks.append(
                f"--- COMPILED DSPy DEMONSTRATION #{idx} ---\n"
                f"Concept Brief: {d.get('concept_brief', '')}\n"
                f"Language: {d.get('language', language)}\n"
                f"AI Teacher Character: {d.get('ai_teacher_character', ai_teacher_character)}\n"
                f"4-Block Plan:\n{d.get('four_block_thinking_plan', '')}\n"
                f"Visual Description:\n{d.get('visual_description_block', '')}\n"
                f"Full Supernova Script:\n{d.get('full_supernova_script', '')}"
            )
        demos_text = "\n\n".join(demo_blocks)

    prompt = f"""You are the Stanford DSPy SupernovaMiddleBrain (`ConceptBriefToSupernovaPackage`).
Return a valid JSON object with the exact keys:
- "reasoning": string (your step-by-step Chain-of-Thought strategy)
- "four_block_thinking_plan": string (Block 1: Hook & Mistake [0-5s], Block 2: Real-Life Pain & Why Old Methods Failed [5-15s], Block 3: Live On-The-Spot Fix & 4 Supernova USPs [15-35s], Block 4: Active Desire & CTA [35-45s])
- "visual_description_block": string (Complete production-ready Visual Description starting with 'Visual Description:' and including 'Hook visuals:', 'Body visuals:', 'Video flow:', and 'Characters:')
- "three_hook_options": array of 3 strings (Option A, Option B, Option C — each containing coupled Line 1 + Line 2 that converge cleanly onto Line 3)
- "full_supernova_script": string (Complete 10-14 turn dialogue `Speaker: line` using `{ai_teacher_character}` as the AI speaker label)

MANDATORY HARD-LOCKED PHRASES IN `full_supernova_script`:
1. "improve your spoken English"
2. "within 30 days"
3. "15 to 20 minutes"
4. "over 1 crore Indians"
5. "through {language} itself"
6. "Install Now button given below"
7. "today itself"

FORBIDDEN PATTERNS (NEVER USE):
- Never say "learn English", "learned English", "learning English", "teach English", or "taught English".
- Never say "10 minutes" or "10 to 15 minutes".
- Never use standalone "Robot:" as a speaker label (always use "{ai_teacher_character}:").
- Never mention rupee prices (Rs, INR, ₹, rupees) or "free trial".
- Never mention competitor apps (MySivi, Mysore, SpeakX, Stimuler, Duolingo, Cambly).
- Never use markdown `**` or `#` symbols.

{demos_text}

--- CURRENT INPUT ---
Concept Brief: {concept_brief}
Target Language: {language}
AI Teacher Character: {ai_teacher_character}

Matched Winner Scripts:
{matched_winner_scripts}

Matched Winner Visual Blueprints:
{matched_winner_visuals}

Contrastive Playbook Rules & Losers to Avoid:
{contrastive_playbook_rules}
"""

    client = google_genai.Client(api_key=api_key)
    for model_id in DEFAULT_GENAI_MODELS:
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.4,
                    response_mime_type="application/json",
                ),
            )
            raw_text = (response.text or "").strip()
            if not raw_text:
                continue
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict) and parsed.get("full_supernova_script"):
                parsed["_model_used"] = model_id
                return parsed
        except Exception as exc:
            logger.debug("google.genai model %s failed: %s", model_id, exc)
            continue

    return None


# ---------------------------------------------------------------------------
# 4. DSPy Compilation (`compile_dspy_brain`)
# ---------------------------------------------------------------------------
def _record_to_dspy_example_dict(
    rec: Dict[str, Any],
    all_winners: List[Dict[str, Any]],
    playbook_rules_text: str,
) -> Dict[str, Any]:
    """Convert a proven `TIER_S` or `TIER_A` record into a complete `ConceptBriefToSupernovaPackage` demonstration."""
    ad_code = str(rec.get("ad_code") or "SN_WINNER")
    title = str(rec.get("title") or "Proven Supernova Winner")
    lang = normalize_language(rec.get("language") or "Tamil")
    ai_char = normalize_ai_character(rec.get("character_type") or "Female Humanoid Robot")
    tags = rec.get("metadata_tags") if isinstance(rec.get("metadata_tags"), dict) else {}
    persona = tags.get("persona") or "Relatable Indian Speaker"
    setting = tags.get("setting") or "Everyday real-life conversation"
    hook_angle = tags.get("hook_angle") or "Spoken Mistake Correction"

    story_blocks = rec.get("story_blocks") if isinstance(rec.get("story_blocks"), dict) else {}
    b1 = _clean_plain_text(story_blocks.get("block_1_hook") or "")[:200]
    b2 = _clean_plain_text(story_blocks.get("block_2_pain_and_old_methods") or "")[:260]
    b3 = _clean_plain_text(story_blocks.get("block_3_live_fix_and_usps") or "")[:260]
    b4 = _clean_plain_text(story_blocks.get("block_4_desire_and_cta") or "")[:200]

    concept_brief = (
        rec.get("concept_brief")
        or f"{persona} in {setting} ({hook_angle}): {title}. {b1[:140]}"
    ).strip()

    # Format 4-block thinking plan from the record's story blocks
    four_block_plan = (
        f"Block 1: Hook & Mistake [0-5s]\n{b1 or 'Open with a relatable spoken English mistake and immediate AI Teacher interruption.'}\n\n"
        f"Block 2: Real-Life Pain & Why Old Methods Failed [5-15s]\n{b2 or 'Highlight the social embarrassment and why passive videos/books failed.'}\n\n"
        f"Block 3: Live On-The-Spot Fix & 4 Supernova USPs [15-35s]\n{b3 or f'Conduct live 1-sentence correction and deliver all 4 USPs through {lang} itself.'}\n\n"
        f"Block 4: Active Desire & CTA [35-45s]\n{b4 or 'Protagonist asks to start now; AI Teacher delivers the Install Now button given below + today itself CTA.'}"
    )

    vb = rec.get("visual_blueprint") if isinstance(rec.get("visual_blueprint"), dict) else {}
    hook_vis = _clean_plain_text(
        vb.get("hook_visuals")
        or f"{vb.get('split_screen_type', 'Horizontal Split Screen')} showing {persona} in {setting} hesitating in English while {ai_char} interrupts."
    )
    body_vis = _clean_plain_text(
        vb.get("body_visuals")
        or f"Interactive live speaking drill with {ai_char} correcting pronunciation and displaying Supernova progress UI."
    )
    video_flow = _clean_plain_text(
        vb.get("video_flow")
        or f"Spoken mistake hook [0-5s] -> Real-life embarrassment [5-15s] -> Live voice correction & 4 Supernova USPs [15-35s] -> Urgent Install Now CTA [35-45s]."
    )
    chars_desc = _clean_plain_text(
        vb.get("characters_description")
        or f"- Protagonist ({persona}): Authentic attire in {setting}.\n- {ai_char}: Friendly 3D AI English teacher inside the Supernova app."
    )
    visual_description_block = (
        f"Visual Description:\n"
        f"Hook visuals: {hook_vis}\n"
        f"Body visuals: {body_vis}\n"
        f"Video flow: {video_flow}\n"
        f"Characters:\n{chars_desc}"
    )

    raw_script = _clean_plain_text(rec.get("english_script") or rec.get("native_script") or "")
    compliant_script = _enforce_supernova_compliance(raw_script, lang, ai_char)
    script_turns = [ln.strip() for ln in compliant_script.splitlines() if ln.strip()]
    l1 = script_turns[0] if len(script_turns) > 0 else f"Protagonist: I hesitated while speaking English in {setting}."
    l2 = script_turns[1] if len(script_turns) > 1 else f"{ai_char}: Stop worrying! Let's fix that sentence right now."

    three_hooks = [
        f"Option A (Proven {ad_code} Spoken Mistake Hook — Converges onto Line 3):\nLine 1 — {l1}\nLine 2 — {l2}",
        f"Option B (High-Stakes Situation Freeze Hook — Converges onto Line 3):\nLine 1 — Protagonist: Everyone in {setting} was waiting for my reply in English, and my mind went blank!\nLine 2 — {ai_char}: Don't freeze up! Repeat this simple English sentence with me right now.",
        f"Option C (Direct Question Pattern-Interrupt Hook — Converges onto Line 3):\nLine 1 — Protagonist: Why do I know the answer in {lang} but still fumble every time I speak English?\nLine 2 — {ai_char}: Because you need live speaking practice, not grammar books — let's try one sentence together!",
    ]

    cac_metrics = rec.get("cac_metrics") if isinstance(rec.get("cac_metrics"), dict) else {}
    vs_avg = cac_metrics.get("cac_vs_avg_pct")
    cac_str = f"CAC vs Avg {vs_avg:+.1f}%" if isinstance(vs_avg, (int, float)) else "Proven Low-CAC Winner"
    split_screen = vb.get("split_screen_type") or "Horizontal Split Screen"

    return {
        "ad_code": ad_code,
        "cac_tier": rec.get("cac_tier", "TIER_S"),
        "concept_brief": concept_brief,
        "language": lang,
        "ai_teacher_character": ai_char,
        "matched_winner_scripts": f"[Matched Winner: {ad_code} ({rec.get('cac_tier', 'TIER_S')}, {lang}, {cac_str})]",
        "matched_winner_visuals": f"[Split Screen: {split_screen} | Hook: {hook_vis[:120]}]",
        "contrastive_playbook_rules": "[Tier-S Rules: Hook <= 12 words, AI Teacher enters by Turn 2, all 7 Supernova phrases verbatim]",
        "reasoning": (
            f"Modeled on proven {rec.get('cac_tier', 'TIER_S')} winner {ad_code} ({title}) in {lang}. "
            f"Opens in {setting} with a {hook_angle} hook ({tags.get('hook_word_count', 10)} words), "
            f"brings {ai_char} in at Turn {tags.get('ai_teacher_entry_turn', 2)}, transitions from real pain "
            f"into on-the-spot speaking practice, and locks all 7 required Supernova phrases."
        ),
        "four_block_thinking_plan": four_block_plan,
        "visual_description_block": visual_description_block,
        "three_hook_options": three_hooks,
        "full_supernova_script": compliant_script,
    }


def supernova_dspy_metric(example: Any, pred: Any, trace: Any = None) -> float:
    """DSPy evaluation metric combining deterministic Supernova compliance and 4-block completeness."""
    from compliance_validator import validate_supernova_script

    script = getattr(pred, "full_supernova_script", "") or ""
    lang = getattr(example, "language", "Tamil") or "Tamil"
    char = getattr(example, "ai_teacher_character", "Female Humanoid Robot") or "Female Humanoid Robot"

    comp = validate_supernova_script(script, language=lang, ai_teacher_character=char)
    compliance_score = float(comp.get("score", 0.0))

    plan = getattr(pred, "four_block_thinking_plan", "") or ""
    vis = getattr(pred, "visual_description_block", "") or ""
    hooks = getattr(pred, "three_hook_options", []) or []

    structure_bonus = 0.0
    if all(f"Block {i}" in str(plan) for i in range(1, 5)):
        structure_bonus += 0.1
    if "Hook visuals:" in str(vis) and "Body visuals:" in str(vis):
        structure_bonus += 0.05
    if isinstance(hooks, list) and len(hooks) >= 3:
        structure_bonus += 0.05

    total = min(1.0, round(0.8 * compliance_score + structure_bonus, 4))
    if trace is not None:
        return total >= 0.95
    return total


def compile_dspy_brain(
    records: List[Dict[str, Any]],
    playbook: Optional[Dict[str, Any]] = None,
    use_live_bootstrap: bool = False,
) -> str:
    """Build `dspy.Example` training examples from `TIER_S` and `TIER_A` winners and compile/save `data/compiled_dspy_brain.json`.

    Runs with 0 Gemini API calls by default (`use_live_bootstrap=False`) so offline compilation
    always succeeds deterministically, while also supporting live `dspy.teleprompt.BootstrapFewShot`
    compilation when `use_live_bootstrap=True`.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Filter TIER_S and TIER_A records, sorting TIER_S first and lowest cac_vs_avg_pct first
    winners: List[Dict[str, Any]] = []
    for r in records or []:
        if not isinstance(r, dict):
            continue
        tier = str(r.get("cac_tier") or "TIER_A").upper()
        if tier in ("TIER_S", "TIER_A"):
            winners.append(r)

    if not winners and records:
        winners = [r for r in records if isinstance(r, dict)]

    def _winner_sort_key(rec: Dict[str, Any]) -> Tuple[int, int, float]:
        tier_rank = 0 if str(rec.get("cac_tier", "")).upper() == "TIER_S" else 1
        tags = rec.get("metadata_tags") if isinstance(rec.get("metadata_tags"), dict) else {}
        turns = int(tags.get("total_turns") or 12)
        concise_rank = 0 if turns <= 20 else 1
        cac_metrics = rec.get("cac_metrics") if isinstance(rec.get("cac_metrics"), dict) else {}
        vs_avg = cac_metrics.get("cac_vs_avg_pct")
        return (tier_rank, concise_rank, float(vs_avg) if isinstance(vs_avg, (int, float)) else 0.0)

    winners.sort(key=_winner_sort_key)

    if playbook is None and CONTRASTIVE_PLAYBOOK_PATH.exists():
        try:
            playbook = json.loads(CONTRASTIVE_PLAYBOOK_PATH.read_text(encoding="utf-8"))
        except Exception:
            playbook = {}

    playbook_rules_text = format_contrastive_rules_for_prompt(playbook or {})

    # Build DSPy Examples
    demo_dicts: List[Dict[str, Any]] = []
    dspy_trainset: List[Any] = []
    for rec in winners[:8]:
        ex_dict = _record_to_dspy_example_dict(rec, winners, playbook_rules_text)
        demo_dicts.append(ex_dict)
        dspy_ex = dspy.Example(**ex_dict).with_inputs(
            "concept_brief",
            "language",
            "ai_teacher_character",
            "matched_winner_scripts",
            "matched_winner_visuals",
            "contrastive_playbook_rules",
        )
        dspy_trainset.append(dspy_ex)

    optimizer_used = "BootstrapFewShot (Offline Labeled Demonstrations — 0 API Calls)"
    if use_live_bootstrap and HAS_DSPY and dspy_trainset:
        cfg = configure_dspy_gemini()
        if cfg.get("configured") and cfg.get("engine") == "dspy":
            try:
                from dspy.teleprompt import BootstrapFewShot  # type: ignore

                student = SupernovaMiddleBrain()
                teleprompter = BootstrapFewShot(
                    metric=supernova_dspy_metric,
                    max_bootstrapped_demos=2,
                    max_labeled_demos=4,
                )
                compiled_module = teleprompter.compile(student, trainset=dspy_trainset)
                compiled_demos = getattr(compiled_module.generate, "demos", [])
                if compiled_demos:
                    demo_dicts = [
                        d.toDict() if hasattr(d, "toDict") else dict(d)
                        for d in compiled_demos
                    ]
                    optimizer_used = "dspy.teleprompt.BootstrapFewShot (Live Gemini)"
            except Exception as boot_exc:
                logger.warning("Live BootstrapFewShot fallback to labeled demos: %s", boot_exc)

    compiled_payload = {
        "module": "SupernovaMiddleBrain",
        "signature": "ConceptBriefToSupernovaPackage",
        "dspy_installed": HAS_DSPY,
        "optimizer": optimizer_used,
        "compiled_at": datetime.now(timezone.utc).isoformat(),
        "total_winner_examples": len(dspy_trainset),
        "selected_few_shot_demos": len(demo_dicts[:1]),
        "demos": demo_dicts[:1],
        "generate": {
            "signature": "ConceptBriefToSupernovaPackage",
            "demos": demo_dicts[:1],
        },
    }

    COMPILED_BRAIN_PATH.write_text(
        json.dumps(compiled_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return str(COMPILED_BRAIN_PATH)


# ---------------------------------------------------------------------------
# 5. Runtime Generation (`generate_with_middle_brain`)
# ---------------------------------------------------------------------------
def _normalize_hook_options(raw_hooks: Any, fallback_hooks: List[str]) -> List[str]:
    """Ensure `three_hook_options` is a clean list of 3 strings."""
    if isinstance(raw_hooks, list):
        cleaned = [_clean_plain_text(h) for h in raw_hooks if str(h).strip()]
        if len(cleaned) >= 3:
            return cleaned[:3]
        while len(cleaned) < 3:
            cleaned.append(fallback_hooks[len(cleaned)])
        return cleaned
    if isinstance(raw_hooks, str) and raw_hooks.strip():
        parts = [
            _clean_plain_text(p)
            for p in re.split(r"\n\s*(?=Option\s+[ABC123]\b)", raw_hooks)
            if p.strip()
        ]
        if len(parts) >= 3:
            return parts[:3]
    return fallback_hooks[:3]


def generate_with_middle_brain(
    concept_brief: str = "",
    language: str = "Tamil",
    ai_teacher_character: str = "Female Humanoid Robot",
    matched_winner_scripts: Any = None,
    matched_winner_visuals: Any = None,
    contrastive_playbook_rules: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run the compiled `SupernovaMiddleBrain` to generate a complete Supernova script package.

    - Loads compiled demonstrations from `data/compiled_dspy_brain.json`.
    - Configures Gemini (`gemini/gemini-3.1-flash-lite-preview` or `gemini/gemini-2.5-flash`).
    - Runs `SupernovaMiddleBrain` (`dspy.ChainOfThought(ConceptBriefToSupernovaPackage)`).
    - Validates the script with `compliance_validator.validate_supernova_script`.
    - If any required phrase is missing or a forbidden regex matches, runs at most 1 retry
      with the specific violation feedback before applying final compliance enforcement.
    """
    from compliance_validator import validate_supernova_script

    brief = (
        concept_brief
        or kwargs.get("brief")
        or kwargs.get("concept")
        or "A relatable Indian speaker hesitates while speaking English in an important daily moment."
    ).strip()
    lang = normalize_language(language or kwargs.get("lang"))
    ai_char = normalize_ai_character(
        ai_teacher_character or kwargs.get("character") or kwargs.get("character_type")
    )

    winners_input = (
        matched_winner_scripts
        if matched_winner_scripts is not None
        else (kwargs.get("matched_winners") or kwargs.get("top_winners") or [])
    )
    visuals_input = (
        matched_winner_visuals
        if matched_winner_visuals is not None
        else (kwargs.get("winner_visuals") or kwargs.get("visual_blueprints") or winners_input)
    )
    losers_input = kwargs.get("losers_to_avoid") or []

    if contrastive_playbook_rules is None:
        contrastive_playbook_rules = kwargs.get("contrastive_playbook") or kwargs.get("playbook")
        if contrastive_playbook_rules is None and CONTRASTIVE_PLAYBOOK_PATH.exists():
            try:
                contrastive_playbook_rules = json.loads(
                    CONTRASTIVE_PLAYBOOK_PATH.read_text(encoding="utf-8")
                )
            except Exception:
                contrastive_playbook_rules = {}

    # Configure DSPy / Gemini
    lm_cfg = configure_dspy_gemini()
    brain = SupernovaMiddleBrain(compiled_path=COMPILED_BRAIN_PATH)

    formatted_winners = format_winner_scripts_for_prompt(winners_input)
    formatted_visuals = format_winner_visuals_for_prompt(visuals_input, winners_fallback=winners_input)
    formatted_rules = format_contrastive_rules_for_prompt(
        contrastive_playbook_rules, losers_to_avoid=losers_input
    )

    active_lm = getattr(getattr(dspy, "settings", None), "lm", None)
    hist_start = len(getattr(active_lm, "history", []) or []) if active_lm is not None else 0

    # Attempt 1: Forward pass
    pred = brain(
        concept_brief=brief,
        language=lang,
        ai_teacher_character=ai_char,
        matched_winner_scripts=formatted_winners,
        matched_winner_visuals=formatted_visuals,
        contrastive_playbook_rules=formatted_rules,
    )

    raw_script = _clean_plain_text(getattr(pred, "full_supernova_script", "") or "")
    compliance = validate_supernova_script(
        raw_script,
        language=lang,
        ai_teacher_character=ai_char,
    )

    retries = 0
    engine_used = getattr(pred, "_engine_used", lm_cfg.get("engine", "dspy"))

    # At most 1 retry with violation feedback if compliance validator reports any violation
    if not compliance.get("passed", False) and engine_used in ("dspy", "google.genai"):
        retries = 1
        violations_list = compliance.get("violations", [])
        missing_list = compliance.get("missing_phrases", [])
        forbidden_list = compliance.get("forbidden_matches", [])
        feedback_lines = [
            f"- Missing required phrases that MUST be included verbatim: {missing_list}",
            f"- Forbidden matches that MUST be removed immediately: {forbidden_list}",
            f"- Detailed violations: {'; '.join(violations_list)}",
        ]
        retry_rules = format_contrastive_rules_for_prompt(
            contrastive_playbook_rules,
            losers_to_avoid=losers_input,
            violation_feedback="\n".join(feedback_lines),
        )
        pred_retry = brain(
            concept_brief=brief,
            language=lang,
            ai_teacher_character=ai_char,
            matched_winner_scripts=formatted_winners,
            matched_winner_visuals=formatted_visuals,
            contrastive_playbook_rules=retry_rules,
        )
        retry_script = _clean_plain_text(getattr(pred_retry, "full_supernova_script", "") or "")
        if retry_script:
            pred = pred_retry
            raw_script = retry_script
            compliance = validate_supernova_script(
                raw_script,
                language=lang,
                ai_teacher_character=ai_char,
            )

    # Extract exact Gemini token usage from DSPy LM history (or LiteLLM token counter on cache hit)
    total_in_tokens = 0
    total_out_tokens = 0
    was_cached = False
    new_calls = (getattr(active_lm, "history", []) or [])[hist_start:] if active_lm is not None else []
    for call in new_calls:
        if not isinstance(call, dict):
            continue
        call_in = 0
        call_out = 0
        resp_obj = call.get("response")
        usage_obj = getattr(resp_obj, "usage", None) if resp_obj is not None else None
        if usage_obj is not None:
            call_in = int(
                getattr(usage_obj, "prompt_tokens", 0)
                or getattr(usage_obj, "input_tokens", 0)
                or 0
            )
            call_out = int(
                getattr(usage_obj, "completion_tokens", 0)
                or getattr(usage_obj, "output_tokens", 0)
                or 0
            )
        if call_in == 0 and isinstance(call.get("usage"), dict):
            u_dict = call["usage"]
            call_in = int(u_dict.get("prompt_tokens") or u_dict.get("input_tokens") or 0)
            call_out = int(u_dict.get("completion_tokens") or u_dict.get("output_tokens") or 0)
        if call_in == 0:
            was_cached = True
            try:
                import litellm  # type: ignore

                msgs = call.get("messages") or []
                outs = call.get("outputs") or []
                if msgs:
                    call_in = int(litellm.token_counter(model="gemini/gemini-2.5-flash", messages=msgs))
                if outs:
                    out_str = "\n".join(str(o) for o in outs)
                    call_out = int(litellm.token_counter(model="gemini/gemini-2.5-flash", text=out_str))
            except Exception:
                pass
        total_in_tokens += call_in
        total_out_tokens += call_out

    token_usage = {
        "input_tokens": int(total_in_tokens),
        "output_tokens": int(total_out_tokens),
        "total_tokens": int(total_in_tokens + total_out_tokens),
        "llm_calls": len(new_calls) or 1,
        "cached": bool(was_cached),
        "model": str(lm_cfg.get("model", "gemini/gemini-3.5-flash-lite")).replace("gemini/", ""),
    }

    # Deterministic safeguard so the final script always passes 100% of Supernova house rules
    if not compliance.get("passed", False):
        raw_script = _enforce_supernova_compliance(raw_script, lang, ai_char)
        compliance = validate_supernova_script(
            raw_script,
            language=lang,
            ai_teacher_character=ai_char,
        )

    fallback_pkg = _synthesize_supernova_package_deterministic(
        concept_brief=brief,
        language=lang,
        ai_teacher_character=ai_char,
        matched_winner_scripts=winners_input,
        matched_winner_visuals=visuals_input,
        contrastive_playbook_rules=contrastive_playbook_rules,
    )

    four_block_plan = _clean_plain_text(
        getattr(pred, "four_block_thinking_plan", "") or fallback_pkg["four_block_thinking_plan"]
    )
    visual_desc = _clean_plain_text(
        getattr(pred, "visual_description_block", "") or fallback_pkg["visual_description_block"]
    )
    if not visual_desc.startswith("Visual Description:"):
        visual_desc = f"Visual Description:\n{visual_desc}"

    three_hooks = _normalize_hook_options(
        getattr(pred, "three_hook_options", None),
        fallback_pkg["three_hook_options"],
    )
    reasoning = _clean_plain_text(
        getattr(pred, "reasoning", "") or fallback_pkg["reasoning"]
    )

    return {
        "four_block_thinking_plan": four_block_plan,
        "middle_brain_strategy": four_block_plan,
        "four_block_plan": four_block_plan,
        "visual_description_block": visual_desc,
        "visual_description": visual_desc,
        "three_hook_options": three_hooks,
        "three_hooks": three_hooks,
        "hook_options": three_hooks,
        "full_supernova_script": raw_script,
        "script": raw_script,
        "supernova_script": raw_script,
        "reasoning": reasoning,
        "compliance": compliance,
        "retries": retries,
        "retried": retries > 0,
        "used_dspy": HAS_DSPY,
        "engine_used": engine_used,
        "model": str(lm_cfg.get("model", "gemini/gemini-3.1-flash-lite-preview")),
        "compiled_demos_loaded": len(brain.compiled_demos),
        "language": lang,
        "ai_teacher_character": ai_char,
        "token_usage": token_usage,
    }
