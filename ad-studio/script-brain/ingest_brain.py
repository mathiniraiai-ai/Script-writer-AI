"""
Google Docs & Sheets + CAC Metrics Ingestion Pipeline for Supernova Script Brain.

Ingests 50-100 proven Supernova scripts and their CAC metrics (from a Google Sheet,
JSON/CSV file, or direct list of Google Docs / inline scripts) into the Vector + SQL
Brain with ZERO Gemini API calls.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

# Ensure sibling modules in ad-studio/script-brain are importable regardless of cwd
SCRIPT_BRAIN_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_BRAIN_DIR.parents[1]
if str(SCRIPT_BRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_BRAIN_DIR))

# Load .env from project root (../../.env) and fallback to current working directory
load_dotenv(REPO_ROOT / ".env")
load_dotenv()

logger = logging.getLogger("supernova.ingest_brain")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/documents.readonly",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

DEFAULT_REPLICATION_SHEET_ID = "13sQlJMgAZ_nFxatPEp5XfZ9swE4et00qEBpwNS2a82I"
DEFAULT_REPLICATION_TAB = "REPLICATE - LANGUAGE"

# Default language average CAC benchmarks (INR) used only when lang_avg_cac is omitted
DEFAULT_LANGUAGE_AVG_CAC: dict[str, float] = {
    "Tamil": 280.0,
    "Telugu": 295.0,
    "Malayalam": 270.0,
    "Kannada": 290.0,
    "Hindi": 320.0,
    "English": 310.0,
    "Marathi": 268.0,
    "Bengali": 282.0,
    "Gujarati": 275.0,
    "Punjabi": 285.0,
    "Odia": 260.0,
    "Assamese": 265.0,
}

CANONICAL_LANGUAGES: dict[str, str] = {
    "tam": "Tamil",
    "tamil": "Tamil",
    "tel": "Telugu",
    "telugu": "Telugu",
    "mal": "Malayalam",
    "malayalam": "Malayalam",
    "kan": "Kannada",
    "kannada": "Kannada",
    "hin": "Hindi",
    "hindi": "Hindi",
    "eng": "English",
    "english": "English",
    "mar": "Marathi",
    "marathi": "Marathi",
    "ben": "Bengali",
    "bengali": "Bengali",
    "guj": "Gujarati",
    "gujarati": "Gujarati",
    "pun": "Punjabi",
    "punjabi": "Punjabi",
    "odi": "Odia",
    "odia": "Odia",
    "asm": "Assamese",
    "assamese": "Assamese",
    "assammese": "Assamese",
}

CANONICAL_CHARACTERS: dict[str, str] = {
    "female humanoid robot": "Female Humanoid Robot",
    "humanoid robot": "Female Humanoid Robot",
    "female robot": "Female Robot",
    "robot": "Female Humanoid Robot",
    "miss nova": "Miss Nova",
    "ms nova": "Miss Nova",
    "ms. nova": "Miss Nova",
    "nova": "Miss Nova",
}


# ---------------------------------------------------------------------------
# 1. Google Docs & Sheets Authentication (via .env Service Account)
# ---------------------------------------------------------------------------

def _load_env_fallback_vars() -> None:
    """Manual fallback parser for ../../.env if python-dotenv is not installed."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            if len(val) >= 2 and ((val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")):
                val = val[1:-1]
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        pass


def get_google_credentials(scopes: list[str] | None = None) -> Any:
    """
    Builds google.oauth2.service_account.Credentials from:
    1. GOOGLE_SERVICE_ACCOUNT_EMAIL + GOOGLE_PRIVATE_KEY in .env (primary, matching TypeScript services)
    2. GOOGLE_SERVICE_ACCOUNT_JSON in .env
    3. GOOGLE_APPLICATION_CREDENTIALS or task-assigner/credentials.json file
    """
    _load_env_fallback_vars()
    target_scopes = scopes or GOOGLE_SCOPES

    try:
        from google.oauth2 import service_account
    except ImportError as exc:
        raise RuntimeError(
            "google-auth / google-api-python-client is not installed in the active environment."
        ) from exc

    sa_email = (os.environ.get("GOOGLE_SERVICE_ACCOUNT_EMAIL") or "").strip()
    sa_private_key = (os.environ.get("GOOGLE_PRIVATE_KEY") or "").strip()

    if sa_email and sa_private_key:
        normalized_key = sa_private_key.replace("\\n", "\n")
        info = {
            "type": "service_account",
            "project_id": os.environ.get("GOOGLE_PROJECT_ID", "marketing-automation-511412"),
            "private_key_id": os.environ.get("GOOGLE_PRIVATE_KEY_ID", ""),
            "private_key": normalized_key,
            "client_email": sa_email,
            "client_id": os.environ.get("GOOGLE_CLIENT_ID", ""),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        }
        return service_account.Credentials.from_service_account_info(info, scopes=target_scopes)

    sa_json = (os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON") or "").strip()
    if sa_json:
        info = json.loads(sa_json)
        if "private_key" in info and isinstance(info["private_key"], str):
            info["private_key"] = info["private_key"].replace("\\n", "\n")
        return service_account.Credentials.from_service_account_info(info, scopes=target_scopes)

    cred_path_str = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    candidate_paths = []
    if cred_path_str:
        candidate_paths.append(Path(cred_path_str))
        candidate_paths.append(REPO_ROOT / cred_path_str)
    candidate_paths.append(REPO_ROOT / "task-assigner" / "credentials.json")

    for path in candidate_paths:
        if path.exists() and path.is_file():
            return service_account.Credentials.from_service_account_file(
                str(path), scopes=target_scopes
            )

    raise RuntimeError(
        "Google Service Account credentials not found. Set GOOGLE_SERVICE_ACCOUNT_EMAIL and "
        "GOOGLE_PRIVATE_KEY in .env (or GOOGLE_SERVICE_ACCOUNT_JSON / GOOGLE_APPLICATION_CREDENTIALS)."
    )


def get_docs_service(credentials: Any = None) -> Any:
    """Returns an authenticated Google Docs v1 service client."""
    from googleapiclient.discovery import build

    creds = credentials or get_google_credentials()
    return build("docs", "v1", credentials=creds, cache_discovery=False)


def get_sheets_service(credentials: Any = None) -> Any:
    """Returns an authenticated Google Sheets v4 service client."""
    from googleapiclient.discovery import build

    creds = credentials or get_google_credentials()
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


# ---------------------------------------------------------------------------
# 2. Google Doc Reading (All Tabs with `includeTabsContent=True`)
# ---------------------------------------------------------------------------

def extract_google_doc_id(doc_url_or_id: str) -> str | None:
    """
    Extracts the Google Doc / Drive file ID from a URL or raw ID string.
    Supports:
      - https://docs.google.com/document/d/<ID>/edit
      - https://drive.google.com/file/d/<ID>/view
      - https://drive.google.com/open?id=<ID>
      - Raw ID strings
    """
    if not doc_url_or_id:
        return None
    raw = str(doc_url_or_id).strip()
    patterns = [
        r"/document/(?:u/\d+/)?d/([a-zA-Z0-9_-]{15,})",
        r"/file/(?:u/\d+/)?d/([a-zA-Z0-9_-]{15,})",
        r"[?&]id=([a-zA-Z0-9_-]{15,})",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            return match.group(1)

    if re.fullmatch(r"[a-zA-Z0-9_-]{20,}", raw):
        return raw
    return None


def _extract_structural_elements_text(elements: list[dict[str, Any]] | None) -> str:
    """Recursively extracts plain text from Google Docs body.content structural elements."""
    if not elements:
        return ""
    chunks: list[str] = []
    for elem in elements:
        if not isinstance(elem, dict):
            continue
        if "paragraph" in elem:
            para = elem.get("paragraph") or {}
            for run in para.get("elements") or []:
                text_run = run.get("textRun")
                if text_run and "content" in text_run:
                    chunks.append(str(text_run["content"]))
        elif "table" in elem:
            table = elem.get("table") or {}
            for row in table.get("tableRows") or []:
                row_cells: list[str] = []
                for cell in row.get("tableCells") or []:
                    cell_text = _extract_structural_elements_text(cell.get("content")).strip()
                    if cell_text:
                        row_cells.append(cell_text)
                if row_cells:
                    chunks.append(" | ".join(row_cells) + "\n")
        elif "tableOfContents" in elem:
            toc = elem.get("tableOfContents") or {}
            chunks.append(_extract_structural_elements_text(toc.get("content")))
    return "".join(chunks)


def _flatten_doc_tabs(tabs: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    """Flattens top-level and nested child tabs into [{'tab_id', 'title', 'text'}]."""
    if not tabs:
        return []
    collected: list[dict[str, str]] = []
    for idx, tab in enumerate(tabs):
        if not isinstance(tab, dict):
            continue
        props = tab.get("tabProperties") or {}
        tab_id = str(props.get("tabId") or f"t.{idx}")
        title = str(props.get("title") or f"Tab {idx + 1}").strip()
        doc_tab = tab.get("documentTab") or {}
        body_content = (doc_tab.get("body") or {}).get("content") or []
        text = _extract_structural_elements_text(body_content).strip()
        collected.append({"tab_id": tab_id, "title": title, "text": text})

        child_tabs = tab.get("childTabs") or []
        if child_tabs:
            collected.extend(_flatten_doc_tabs(child_tabs))
    return collected


def _strip_forbidden_links(text: str) -> str:
    """Strips any YouTube or Notion URLs from script/visual text."""
    if not text:
        return ""
    cleaned = re.sub(
        r"https?://(?:www\.)?(?:youtube\.com|youtu\.be|notion\.so|app\.notion\.com)[^\s)\]]*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Remove empty 'Ref' lines left behind after stripping a YouTube URL
    cleaned = re.sub(r"(?im)^[ \t]*ref(?:erence)?(?:\s+link)?\s*:?[ \t]*\n?", "", cleaned)
    return cleaned.strip()


def fetch_google_doc_payload(
    doc_url_or_id: str,
    preferred_language: str = "Hindi",
    preferred_tab_title: str | None = None,
    docs_service: Any = None,
) -> dict[str, Any]:
    """
    Fetches a Google Doc with `includeTabsContent=True`.
    When a Google Doc has multiple language tabs (e.g., Hindi, Marathi, Kannada, Bengali,
    Punjabi, Gujarati, Assammese, Odia, Telugu, Malayalam), it NEVER concatenates all tabs.
    Instead, it selects the single tab matching `preferred_language` (or `preferred_tab_title`
    or `"Supernova Script"`), falling back to the first non-empty tab, and returns
    `available_tabs: list[str]` with all tab titles in the document.
    """
    doc_id = extract_google_doc_id(doc_url_or_id)
    if not doc_id:
        raise ValueError(f"Could not extract a valid Google Doc ID from: {doc_url_or_id!r}")

    service = docs_service or get_docs_service()
    try:
        doc = (
            service.documents()
            .get(documentId=doc_id, includeTabsContent=True)
            .execute()
        )
    except TypeError:
        # Fallback in case an older google-api-python-client discovery doc is cached
        doc = service.documents().get(documentId=doc_id).execute()

    doc_title = str(doc.get("title") or "").strip()
    tabs = _flatten_doc_tabs(doc.get("tabs"))
    available_tabs = [t["title"] for t in tabs if t.get("title")]
    non_empty_tabs = [t for t in tabs if t.get("text")]

    selected_tab_title = "Body"
    selected_text = ""

    if non_empty_tabs:
        candidate_prefs: list[str] = []
        if preferred_tab_title and preferred_tab_title.strip():
            candidate_prefs.append(preferred_tab_title.strip().lower())
        if preferred_language and preferred_language.strip():
            candidate_prefs.append(preferred_language.strip().lower())
        if "supernova script" not in candidate_prefs:
            candidate_prefs.append("supernova script")

        chosen: dict[str, str] | None = None

        # 1. Exact case-insensitive match on any candidate preference
        for pref in candidate_prefs:
            chosen = next(
                (t for t in non_empty_tabs if t["title"].strip().lower() == pref),
                None,
            )
            if chosen:
                break

        # 2. Substring match on any candidate preference or "supernova"
        if not chosen:
            for pref in candidate_prefs:
                chosen = next(
                    (
                        t
                        for t in non_empty_tabs
                        if pref in t["title"].strip().lower()
                    ),
                    None,
                )
                if chosen:
                    break

        if not chosen:
            chosen = next(
                (
                    t
                    for t in non_empty_tabs
                    if "supernova" in t["title"].strip().lower()
                ),
                None,
            )

        # 3. If 2+ tabs exist and Tab 1 is "Competitor Transcript", prefer first non-competitor tab
        if not chosen and len(non_empty_tabs) > 1:
            chosen = next(
                (
                    t
                    for t in non_empty_tabs
                    if "competitor" not in t["title"].strip().lower()
                ),
                None,
            )

        if not chosen:
            chosen = non_empty_tabs[0]

        selected_tab_title = chosen["title"]
        selected_text = chosen["text"]
    else:
        body_content = (doc.get("body") or {}).get("content") or []
        selected_text = _extract_structural_elements_text(body_content).strip()

    selected_text = _strip_forbidden_links(selected_text)

    return {
        "doc_id": doc_id,
        "title": doc_title,
        "selected_tab_title": selected_tab_title,
        "text": selected_text,
        "available_tabs": available_tabs,
        "tabs": tabs,
    }


def fetch_google_doc_with_tabs(
    doc_url_or_id: str,
    preferred_language: str = "Hindi",
    docs_service: Any = None,
) -> dict[str, Any]:
    """
    Fetches a single preferred language tab's text along with `available_tabs: list[str]`
    from a multi-tab Supernova Google Doc.
    """
    return fetch_google_doc_payload(
        doc_url_or_id=doc_url_or_id,
        preferred_language=preferred_language,
        docs_service=docs_service,
    )


def fetch_google_doc_text(
    doc_url_or_id: str,
    preferred_language: str = "Hindi",
    preferred_tab_title: str | None = None,
    docs_service: Any = None,
) -> str:
    """
    Returns the plain text of a single matching tab in a Google Doc (preferring
    `preferred_language` or `"Supernova Script"` when multiple tabs exist).
    """
    payload = fetch_google_doc_payload(
        doc_url_or_id=doc_url_or_id,
        preferred_language=preferred_language,
        preferred_tab_title=preferred_tab_title,
        docs_service=docs_service,
    )
    return payload["text"]


# ---------------------------------------------------------------------------
# 3. CAC Tier Classification, Performance Cell Parser & Normalization
# ---------------------------------------------------------------------------

def _to_float(val: Any) -> float | None:
    """Safely converts numbers, currency strings ('₹245.50'), or percentages ('-18.4%') to float."""
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s or s.lower() in {"na", "n/a", "null", "none", "-", "--", ""}:
        return None
    cleaned = re.sub(r"[₹$,%\s,]", "", s)
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_performance_cell(perf_text: str | None) -> dict[str, Any]:
    """
    Parses the multi-line `Performance` column (Column AL / index 37) from the
    Supernova Replication Queue sheet into structured `cac_metrics`.

    Example input:
      Spends: ₹558,734   (lifetime, spend 6 Oct–now)
      CAC ₹176 vs Hindi avg ₹209 (-16.2%)
      Post-1Hr Cancellation CAC ₹238 vs Hindi avg ₹275 (-13.4%)
      second pay 4d CAC ₹551 (6–17 Oct) vs Hindi avg ₹664 (-17.0%)
      second pay 7d CAC ₹425 (6–14 Oct) vs Hindi avg ₹576 (-26.1%)
      Cancelled 1-day 37.6% vs Hindi avg 34.2% (+3.4pp)
      Cancelled 3-day 56.1% vs Hindi avg 52.5% (+3.6pp)
    """
    if not perf_text or not str(perf_text).strip():
        return {}

    text = str(perf_text).strip()
    metrics: dict[str, Any] = {}

    # 1. Spends & spend window
    spend_match = re.search(
        r"Spends\s*:\s*₹?\s*([\d,]+(?:\.\d+)?)\s*(?:\(([^)]*)\))?",
        text,
        flags=re.IGNORECASE,
    )
    if spend_match:
        metrics["spend_inr"] = _to_float(spend_match.group(1))
        if spend_match.group(2):
            metrics["spend_window"] = spend_match.group(2).strip()

    # 2. First-pay CAC ("CAC ₹176 vs Hindi avg ₹209 (-16.2%)")
    first_pay_match = re.search(
        r"(?m)^\s*CAC\s*₹?\s*([\d,]+(?:\.\d+)?)\s+vs\s+.*?avg\s*₹?\s*([\d,]+(?:\.\d+)?)\s*\(\s*([+-]?\d+(?:\.\d+)?)\s*%\s*\)",
        text,
        flags=re.IGNORECASE,
    )
    if first_pay_match:
        metrics["first_pay_cac"] = _to_float(first_pay_match.group(1))
        metrics["first_pay_lang_avg"] = _to_float(first_pay_match.group(2))
        metrics["first_pay_vs_avg_pct"] = _to_float(first_pay_match.group(3))

    # 3. Post-1Hr Cancellation CAC
    post_1hr_match = re.search(
        r"Post-1Hr\s+Cancellation\s+CAC\s*₹?\s*([\d,]+(?:\.\d+)?)\s+vs\s+.*?avg\s*₹?\s*([\d,]+(?:\.\d+)?)\s*\(\s*([+-]?\d+(?:\.\d+)?)\s*%\s*\)",
        text,
        flags=re.IGNORECASE,
    )
    if post_1hr_match:
        metrics["post_1hr_cancel_cac"] = _to_float(post_1hr_match.group(1))
        metrics["post_1hr_lang_avg"] = _to_float(post_1hr_match.group(2))
        metrics["post_1hr_vs_avg_pct"] = _to_float(post_1hr_match.group(3))

    # 4. Second pay 4d CAC
    sp_4d_match = re.search(
        r"second\s+pay\s+4d\s+CAC\s*₹?\s*([\d,]+(?:\.\d+)?)(?:\s*\([^)]*\))?\s+vs\s+.*?avg\s*₹?\s*([\d,]+(?:\.\d+)?)\s*\(\s*([+-]?\d+(?:\.\d+)?)\s*%\s*\)",
        text,
        flags=re.IGNORECASE,
    )
    if sp_4d_match:
        metrics["second_pay_cac_4d"] = _to_float(sp_4d_match.group(1))
        metrics["second_pay_4d_lang_avg"] = _to_float(sp_4d_match.group(2))
        metrics["second_pay_4d_vs_avg_pct"] = _to_float(sp_4d_match.group(3))

    # 5. Second pay 7d CAC
    sp_7d_match = re.search(
        r"second\s+pay\s+7d\s+CAC\s*₹?\s*([\d,]+(?:\.\d+)?)(?:\s*\([^)]*\))?\s+vs\s+.*?avg\s*₹?\s*([\d,]+(?:\.\d+)?)\s*\(\s*([+-]?\d+(?:\.\d+)?)\s*%\s*\)",
        text,
        flags=re.IGNORECASE,
    )
    if sp_7d_match:
        metrics["second_pay_cac_7d"] = _to_float(sp_7d_match.group(1))
        metrics["second_pay_7d_lang_avg"] = _to_float(sp_7d_match.group(2))
        metrics["second_pay_7d_vs_avg_pct"] = _to_float(sp_7d_match.group(3))

    # 6. Cancelled 1-day & 3-day percentages
    cancel_1d_match = re.search(
        r"Cancelled\s+1-day\s+([\d,]+(?:\.\d+)?)\s*%",
        text,
        flags=re.IGNORECASE,
    )
    if cancel_1d_match:
        c1d = _to_float(cancel_1d_match.group(1))
        metrics["cancel_1d_pct"] = c1d
        metrics["cancel_pct"] = c1d

    cancel_3d_match = re.search(
        r"Cancelled\s+3-day\s+([\d,]+(?:\.\d+)?)\s*%",
        text,
        flags=re.IGNORECASE,
    )
    if cancel_3d_match:
        metrics["cancel_3d_pct"] = _to_float(cancel_3d_match.group(1))

    # 7. Language average CAC (prefer second_pay_7d_lang_avg, then first_pay_lang_avg)
    lang_avg_cac = (
        metrics.get("second_pay_7d_lang_avg")
        or metrics.get("first_pay_lang_avg")
        or metrics.get("second_pay_4d_lang_avg")
        or metrics.get("post_1hr_lang_avg")
    )
    if lang_avg_cac is not None:
        metrics["lang_avg_cac"] = lang_avg_cac

    # 8. Best / most negative CAC vs language average % (Supernova Rule Q2/A4)
    pct_candidates = [
        v
        for v in (
            metrics.get("second_pay_7d_vs_avg_pct"),
            metrics.get("second_pay_4d_vs_avg_pct"),
            metrics.get("first_pay_vs_avg_pct"),
        )
        if v is not None
    ]
    if pct_candidates:
        metrics["cac_vs_avg_pct"] = round(min(pct_candidates), 2)
    elif metrics.get("post_1hr_vs_avg_pct") is not None:
        metrics["cac_vs_avg_pct"] = round(float(metrics["post_1hr_vs_avg_pct"]), 2)

    return metrics


def normalize_language(language: str | None, fallback_text: str = "") -> str:
    """Normalizes language code/name to canonical language name (e.g. Hindi, Tamil, Telugu)."""
    if language:
        cleaned = language.strip().lower()
        if cleaned in CANONICAL_LANGUAGES:
            return CANONICAL_LANGUAGES[cleaned]
        for key, canonical in CANONICAL_LANGUAGES.items():
            if re.search(rf"\b{re.escape(key)}\b", cleaned):
                return canonical

    if fallback_text:
        lower_fb = fallback_text.lower()
        for key, canonical in CANONICAL_LANGUAGES.items():
            if re.search(rf"\b{re.escape(key)}\b", lower_fb):
                return canonical

    return "Tamil"


def normalize_character_type(character_type: str | None, fallback_text: str = "") -> str:
    """Normalizes AI teacher character type to one of the 3 canonical Supernova teachers."""
    for candidate in (character_type, fallback_text):
        if not candidate:
            continue
        cleaned = candidate.strip().lower()
        if cleaned in CANONICAL_CHARACTERS:
            return CANONICAL_CHARACTERS[cleaned]
        if "miss nova" in cleaned or "ms nova" in cleaned or "ms. nova" in cleaned:
            return "Miss Nova"
        if "female humanoid robot" in cleaned or "humanoid" in cleaned:
            return "Female Humanoid Robot"
        if "female robot" in cleaned:
            return "Female Robot"
    return "Female Humanoid Robot"


def classify_cac_tier(cac_metrics: dict[str, Any] | None) -> str:
    """
    Classifies a script's CAC performance into TIER_S, TIER_A, or TIER_B:
      - If `cac_vs_avg_pct <= -12.0` (or `first_pay_vs_avg_pct <= -15.0` or
        `second_pay_7d_vs_avg_pct <= -12.0` or `second_pay_4d_vs_avg_pct <= -12.0`): "TIER_S".
      - Else if `cac_vs_avg_pct < 0.0` or any of `second_pay_7d_vs_avg_pct < 0.0`,
        `second_pay_4d_vs_avg_pct < 0.0`, `first_pay_vs_avg_pct < 0.0`: "TIER_A".
      - Otherwise: "TIER_B".
    """
    if not isinstance(cac_metrics, dict):
        return "TIER_B"

    cac_vs_avg_pct = _to_float(cac_metrics.get("cac_vs_avg_pct"))
    first_pay_vs_avg_pct = _to_float(cac_metrics.get("first_pay_vs_avg_pct"))
    second_pay_4d_vs_avg_pct = _to_float(cac_metrics.get("second_pay_4d_vs_avg_pct"))
    second_pay_7d_vs_avg_pct = _to_float(cac_metrics.get("second_pay_7d_vs_avg_pct"))

    if cac_vs_avg_pct is None:
        pct_candidates = [
            v
            for v in (second_pay_7d_vs_avg_pct, second_pay_4d_vs_avg_pct, first_pay_vs_avg_pct)
            if v is not None
        ]
        if pct_candidates:
            cac_vs_avg_pct = min(pct_candidates)
            cac_metrics["cac_vs_avg_pct"] = round(cac_vs_avg_pct, 2)
        else:
            lang_avg_cac = _to_float(cac_metrics.get("lang_avg_cac"))
            primary_cac = (
                _to_float(cac_metrics.get("second_pay_cac_7d"))
                or _to_float(cac_metrics.get("second_pay_cac_4d"))
                or _to_float(cac_metrics.get("post_1hr_cancel_cac"))
                or _to_float(cac_metrics.get("first_pay_cac"))
                or _to_float(cac_metrics.get("cac"))
            )
            if primary_cac is not None and lang_avg_cac is not None and lang_avg_cac > 0:
                cac_vs_avg_pct = ((primary_cac - lang_avg_cac) / lang_avg_cac) * 100.0
                cac_metrics["cac_vs_avg_pct"] = round(cac_vs_avg_pct, 2)

    has_any_pct = any(
        v is not None
        for v in (
            cac_vs_avg_pct,
            first_pay_vs_avg_pct,
            second_pay_4d_vs_avg_pct,
            second_pay_7d_vs_avg_pct,
        )
    )

    if has_any_pct:
        if cac_vs_avg_pct is not None:
            cac_metrics["cac_vs_avg_pct"] = round(cac_vs_avg_pct, 2)
        if (
            (cac_vs_avg_pct is not None and cac_vs_avg_pct <= -12.0)
            or (first_pay_vs_avg_pct is not None and first_pay_vs_avg_pct <= -15.0)
            or (second_pay_7d_vs_avg_pct is not None and second_pay_7d_vs_avg_pct <= -12.0)
            or (second_pay_4d_vs_avg_pct is not None and second_pay_4d_vs_avg_pct <= -12.0)
        ):
            return "TIER_S"
        if (
            (cac_vs_avg_pct is not None and cac_vs_avg_pct < 0.0)
            or (second_pay_7d_vs_avg_pct is not None and second_pay_7d_vs_avg_pct < 0.0)
            or (second_pay_4d_vs_avg_pct is not None and second_pay_4d_vs_avg_pct < 0.0)
            or (first_pay_vs_avg_pct is not None and first_pay_vs_avg_pct < 0.0)
        ):
            return "TIER_A"
        return "TIER_B"

    # Fallback: check explicit `tier` or `cac_tier` string if no numeric CAC comparison was possible
    explicit_tier = str(cac_metrics.get("tier") or cac_metrics.get("cac_tier") or "").strip().upper()
    if explicit_tier in {"TIER_S", "S", "TIER-S", "WINNER_S"}:
        cac_metrics.setdefault("cac_vs_avg_pct", -20.0)
        return "TIER_S"
    if explicit_tier in {"TIER_A", "A", "TIER-A", "WINNER_A", "WINNER"}:
        cac_metrics.setdefault("cac_vs_avg_pct", -8.0)
        return "TIER_A"
    if explicit_tier in {"TIER_B", "B", "TIER-B", "LOSER"}:
        cac_metrics.setdefault("cac_vs_avg_pct", 12.0)
        return "TIER_B"

    cac_metrics.setdefault("cac_vs_avg_pct", 0.0)
    return "TIER_B"


def normalize_cac_metrics(
    raw_metrics: dict[str, Any] | None,
    language: str = "Tamil",
    explicit_tier: str | None = None,
) -> tuple[dict[str, Any], str]:
    """
    Normalizes a `cac_metrics` dictionary so all standard keys expected by `db_store.py`
    and `success_predictor.py` (plus extended Replication Queue CAC fields) are populated,
    and returns `(normalized_metrics, cac_tier)`.
    """
    src = dict(raw_metrics or {})
    if isinstance(src.get("performance_text"), str) and src["performance_text"].strip():
        parsed_perf = parse_performance_cell(src.pop("performance_text"))
        for k, v in parsed_perf.items():
            if src.get(k) is None:
                src[k] = v

    if explicit_tier and "tier" not in src and "cac_tier" not in src:
        src["tier"] = explicit_tier

    first_pay_cac = _to_float(src.get("first_pay_cac") or src.get("first_pay"))
    post_1hr_cancel_cac = _to_float(
        src.get("post_1hr_cancel_cac") or src.get("post_1hr_cac") or src.get("effective_cac")
    )
    second_pay_cac_4d = _to_float(src.get("second_pay_cac_4d") or src.get("cac_4d"))
    second_pay_cac_7d = _to_float(
        src.get("second_pay_cac_7d") or src.get("cac_7d") or src.get("cac")
    )
    cancel_pct = _to_float(
        src.get("cancel_pct") or src.get("cancel_1d_pct") or src.get("cancellation_pct")
    )
    lang_avg_cac = _to_float(
        src.get("lang_avg_cac")
        or src.get("second_pay_7d_lang_avg")
        or src.get("first_pay_lang_avg")
        or src.get("language_avg_cac")
    )

    if lang_avg_cac is None and any(
        x is not None for x in (second_pay_cac_7d, second_pay_cac_4d, post_1hr_cancel_cac, first_pay_cac)
    ):
        lang_avg_cac = DEFAULT_LANGUAGE_AVG_CAC.get(language, 290.0)

    normalized: dict[str, Any] = {
        "first_pay_cac": first_pay_cac,
        "post_1hr_cancel_cac": post_1hr_cancel_cac,
        "second_pay_cac_4d": second_pay_cac_4d,
        "second_pay_cac_7d": second_pay_cac_7d,
        "cancel_pct": cancel_pct,
        "lang_avg_cac": lang_avg_cac,
        "cac_vs_avg_pct": _to_float(src.get("cac_vs_avg_pct")),
    }

    # Preserve extended Replication Queue CAC fields if present
    extra_float_keys = (
        "spend_inr",
        "first_pay_lang_avg",
        "first_pay_vs_avg_pct",
        "post_1hr_lang_avg",
        "post_1hr_vs_avg_pct",
        "second_pay_4d_lang_avg",
        "second_pay_4d_vs_avg_pct",
        "second_pay_7d_lang_avg",
        "second_pay_7d_vs_avg_pct",
        "cancel_1d_pct",
        "cancel_3d_pct",
    )
    for k in extra_float_keys:
        if k in src and src[k] is not None:
            normalized[k] = _to_float(src[k])

    if "spend_window" in src and src["spend_window"]:
        normalized["spend_window"] = str(src["spend_window"]).strip()

    if "tier" in src:
        normalized["tier"] = src["tier"]
    if "cac_tier" in src:
        normalized["cac_tier"] = src["cac_tier"]

    tier = classify_cac_tier(normalized)
    # Remove temporary tier hint keys to keep cac_metrics clean
    normalized.pop("tier", None)
    normalized.pop("cac_tier", None)
    return normalized, tier


# ---------------------------------------------------------------------------
# 4. Loading Proven Script Items from JSON / CSV / Google Sheets
# ---------------------------------------------------------------------------

_CONCEPT_BOILERPLATE_WORDS = {
    "never",
    "mean",
    "say",
    "this",
    "mp4",
    "mov",
    "not",
    "knowing",
    "english",
    "recently",
    "in",
    "the",
    "a",
    "an",
    "for",
    "to",
}


def _derive_replication_ad_code(
    actual_ad_name: str,
    week_col: str,
    concept_cleaned: str,
    idea_name: str,
    ads_manager_link: str,
    index: int = 1,
) -> str:
    """
    Derives a clean unique `ad_code` from a Replication Queue sheet row, e.g.:
      - `W40_LONG_HAIRED_INSULTS`
      - `W37_STICK_HONEYMOON`
      - `W33_SUDDENLY_IMMEDIATELY`
      - `W34_BOYS_PARTY`
      - `W36_LEANING_CAMERA`
    """
    week_match = re.search(r"\b(W\d{1,2})\b", actual_ad_name or "", flags=re.IGNORECASE)
    if not week_match:
        week_match = re.search(r"\b(W\d{1,2})\b", week_col or "", flags=re.IGNORECASE)
    week_prefix = week_match.group(1).upper() if week_match else f"SN{index:02d}"

    raw_concept = (concept_cleaned or idea_name or actual_ad_name or "").strip().lower()
    words = [w for w in re.findall(r"[a-z0-9]+", raw_concept) if w]
    filtered = [w for w in words if w not in _CONCEPT_BOILERPLATE_WORDS]
    slug_words = filtered[:3] if len(filtered) >= 2 else words[:3]

    if slug_words:
        slug = "_".join(w.upper() for w in slug_words)
        return f"{week_prefix}_{slug}"

    ad_id_match = re.search(r"selected_ad_ids=(\d+)", ads_manager_link or "")
    if ad_id_match:
        return f"{week_prefix}_{ad_id_match.group(1)[-6:]}"

    return f"{week_prefix}_SCRIPT_{index:02d}"


def _coerce_row_to_item(row: dict[str, Any], index: int = 1) -> dict[str, Any]:
    """
    Converts a flexible JSON object or CSV/Sheet row dict into a canonical ingest item.
    Strictly ignores `YouTube link`, `Notion page`, and `Notion Link` columns.
    """
    # Filter out forbidden YouTube / Notion keys immediately
    forbidden_keys = {"youtube link", "youtube_link", "notion page", "notion_page", "notion link", "notion_link"}
    lower_map = {
        str(k).strip().lower(): v
        for k, v in row.items()
        if k is not None and str(k).strip().lower() not in forbidden_keys
    }

    def pick(*keys: str) -> Any:
        for k in keys:
            val = lower_map.get(k.lower())
            if val is not None and str(val).strip() != "":
                return val
        return None

    actual_ad_name = str(pick("actual ad name (live source ad)", "actual_ad_name") or "").strip()
    idea_name = str(pick("idea / ad name", "idea_name") or "").strip()
    ad_name_with_code = str(pick("ad name with code") or "").strip()
    concept_cleaned = str(pick("concept (cleaned)", "concept_cleaned") or "").strip()
    week_col = str(pick("week") or "").strip()
    ads_manager_link = str(pick("ads manager link", "ads_manager_link") or "").strip()

    explicit_ad_code = pick("ad_code", "adcode", "code", "id", "script_code")
    if explicit_ad_code:
        ad_code = str(explicit_ad_code).strip()
    elif actual_ad_name or concept_cleaned:
        ad_code = _derive_replication_ad_code(
            actual_ad_name=actual_ad_name,
            week_col=week_col,
            concept_cleaned=concept_cleaned,
            idea_name=idea_name,
            ads_manager_link=ads_manager_link,
            index=index,
        )
    elif ad_name_with_code:
        ad_code = ad_name_with_code
    else:
        ad_code = f"SN_INGEST_{index:03d}"

    title = str(
        pick(
            "title",
            "actual ad name (live source ad)",
            "idea / ad name",
            "ad name with code",
            "ad_name",
            "ad name",
            "name",
        )
        or ad_code
    ).strip()

    candidate_doc_url = str(
        pick("doc_url", "script doc", "script link", "docurl", "script_url", "google_doc_url", "url", "link")
        or ""
    ).strip()
    # Never allow YouTube or Notion links as doc_url
    if re.search(r"youtube\.com|youtu\.be|notion\.so|app\.notion\.com", candidate_doc_url, re.IGNORECASE):
        candidate_doc_url = ""
    doc_url = candidate_doc_url

    raw_script_text = _strip_forbidden_links(
        str(
            pick(
                "raw_script_text",
                "script_text",
                "raw_text",
                "script",
                "supernova_script",
                "english_script",
            )
            or ""
        ).strip()
    )

    source_language_raw = pick("source language", "source_language")
    target_language_raw = pick("language", "lang")
    language = normalize_language(
        str(source_language_raw or target_language_raw or ""),
        fallback_text=f"{actual_ad_name} {title} {ad_code}",
    )

    character_type = normalize_character_type(
        pick("character_type", "character", "ai_teacher", "ai_teacher_character"),
        fallback_text=f"{actual_ad_name} {title} {raw_script_text[:300]}",
    )

    perf_cell_text = str(pick("performance", "performance_text") or "").strip()
    parsed_perf_metrics = parse_performance_cell(perf_cell_text) if perf_cell_text else {}

    nested_metrics = row.get("cac_metrics") if isinstance(row.get("cac_metrics"), dict) else {}
    flat_metrics = {
        "first_pay_cac": pick("first_pay_cac", "first pay cac", "1st pay cac"),
        "post_1hr_cancel_cac": pick(
            "post_1hr_cancel_cac",
            "post 1hr cancel cac",
            "post-1hr cancel cac",
            "effective cac",
        ),
        "second_pay_cac_4d": pick("second_pay_cac_4d", "2nd pay cac (4d)", "second pay cac 4d"),
        "second_pay_cac_7d": pick(
            "second_pay_cac_7d",
            "2nd pay cac (7d)",
            "second pay cac 7d",
            "cac_7d",
            "cac",
        ),
        "cancel_pct": pick("cancel_pct", "cancel %", "cancellation %", "1hr cancel %"),
        "lang_avg_cac": pick("lang_avg_cac", "language avg cac", "avg cac", "benchmark cac"),
        "cac_vs_avg_pct": pick("cac_vs_avg_pct", "cac vs avg %", "cac diff %", "cac delta %"),
        "tier": pick("cac_tier", "tier"),
    }
    merged_metrics = {
        **parsed_perf_metrics,
        **{k: v for k, v in flat_metrics.items() if v is not None},
        **nested_metrics,
    }

    existing_sheet_context = (
        row.get("sheet_context")
        if isinstance(row.get("sheet_context"), dict)
        else (row.get("metadata_tags") or {}).get("sheet_context")
        if isinstance(row.get("metadata_tags"), dict)
        else None
    )

    sheet_context: dict[str, Any] = dict(existing_sheet_context or {})
    if any(
        k in lower_map
        for k in (
            "actual ad name (live source ad)",
            "source language",
            "card verdict",
            "concept (cleaned)",
            "script doc",
        )
    ):
        sheet_context.update(
            {
                "target_replication_language": str(target_language_raw or "").strip(),
                "source_language": str(source_language_raw or language).strip(),
                "type": str(pick("type") or "").strip(),
                "idea_name": idea_name,
                "category": str(pick("category") or "").strip(),
                "priority": str(pick("priority") or "").strip(),
                "confidence": str(pick("confidence") or "").strip(),
                "duration": str(pick("duration") or "").strip(),
                "tools": str(pick("tools") or "").strip(),
                "edit_pattern": str(pick("edit pattern", "edit_pattern") or "").strip(),
                "sheet_character": str(pick("character") or "").strip(),
                "actual_ad_name": actual_ad_name,
                "lane": str(pick("lane") or "").strip(),
                "card_verdict": str(pick("card verdict", "card_verdict") or "").strip(),
                "concept_cleaned": concept_cleaned,
                "already_live_in": str(pick("already live in", "already_live_in") or "").strip(),
                "why_this_row": str(pick("why this row", "why_this_row") or "").strip(),
                "ads_manager_link": ads_manager_link,
                "available_language_tabs": list(sheet_context.get("available_language_tabs") or []),
            }
        )

    item_out: dict[str, Any] = {
        "ad_code": ad_code,
        "title": title,
        "doc_url": doc_url,
        "raw_script_text": raw_script_text,
        "language": language,
        "character_type": character_type,
        "cac_metrics": merged_metrics,
    }
    if sheet_context:
        item_out["sheet_context"] = sheet_context
    if isinstance(row.get("available_language_tabs"), list):
        item_out["available_language_tabs"] = row["available_language_tabs"]
    return item_out


def load_items_from_file(file_path: str | Path) -> list[dict[str, Any]]:
    """Loads proven script items from a `.json`, `.jsonl`, or `.csv` file."""
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Ingestion file not found: {path}")

    suffix = path.suffix.lower()
    raw_rows: list[dict[str, Any]] = []

    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_rows.append(dict(row))
    elif suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                raw_rows.append(json.loads(line))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            raw_rows = data
        elif isinstance(data, dict):
            for key in ("items", "scripts", "records", "data"):
                if isinstance(data.get(key), list):
                    raw_rows = data[key]
                    break
            else:
                raw_rows = [data]

    return [_coerce_row_to_item(r, idx + 1) for idx, r in enumerate(raw_rows)]


def fetch_items_from_google_sheet(
    spreadsheet_id: str | None = DEFAULT_REPLICATION_SHEET_ID,
    sheet_name: str | None = DEFAULT_REPLICATION_TAB,
    limit: int | None = 100,
    sheets_service: Any = None,
) -> list[dict[str, Any]]:
    """
    Reads proven script rows + Google Doc links (including Smart Chips & hyperlinks)
    and CAC metrics directly from the Supernova Replication Queue Google Sheet.
    - Strictly ignores `YouTube link`, `Notion page`, and `Notion Link` columns.
    - Skips rows without a valid Google Doc link in `Script doc` or `Script Link`.
    - Deduplicates by Google Doc `doc_id` so the same source script is not ingested twice.
    """
    _load_env_fallback_vars()
    target_spreadsheet_id = (
        spreadsheet_id
        or os.environ.get("GOOGLE_BRAIN_SPREADSHEET_ID")
        or os.environ.get("GOOGLE_SPREADSHEET_ID")
        or DEFAULT_REPLICATION_SHEET_ID
    ).strip()
    if not target_spreadsheet_id:
        raise ValueError("No spreadsheet_id provided and GOOGLE_SPREADSHEET_ID is not set.")

    target_sheet_name = (
        sheet_name
        or os.environ.get("GOOGLE_BRAIN_SHEET_NAME")
        or DEFAULT_REPLICATION_TAB
    ).strip()

    service = sheets_service or get_sheets_service()
    quoted_sheet = "'" + target_sheet_name.replace("'", "''") + "'"

    val_res = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=target_spreadsheet_id, range=quoted_sheet)
        .execute()
    )
    rows: list[list[str]] = val_res.get("values") or []
    if len(rows) <= 1:
        return []

    headers = [re.sub(r"\s+", " ", str(h or "")).strip() for h in rows[0]]
    forbidden_headers = {"youtube link", "youtube_link", "notion page", "notion_page", "notion link", "notion_link"}

    # Locate Script doc (Col AV / index 47) and Script Link (Col N / index 13) columns
    script_doc_col_indices = [
        i
        for i, h in enumerate(headers)
        if re.search(r"^script\s*doc$|^script\s*link$|doc\s*url|google\s*doc", h, re.IGNORECASE)
    ]
    # Prefer `Script doc` first, then `Script Link`
    script_doc_col_indices.sort(
        key=lambda idx: 0 if headers[idx].lower() == "script doc" else 1
    )

    def _col_letter(col_idx: int) -> str:
        n = col_idx + 1
        letters = ""
        while n > 0:
            rem = (n - 1) % 26
            letters = chr(65 + rem) + letters
            n = (n - 1) // 26
        return letters

    link_metadata_by_col: dict[int, list[dict[str, Any]]] = {}
    if script_doc_col_indices:
        ranges = [
            f"{quoted_sheet}!{_col_letter(c_idx)}1:{_col_letter(c_idx)}{len(rows)}"
            for c_idx in script_doc_col_indices
        ]
        try:
            meta_res = (
                service.spreadsheets()
                .get(
                    spreadsheetId=target_spreadsheet_id,
                    ranges=ranges,
                    fields=(
                        "sheets.data.rowData.values("
                        "formattedValue,hyperlink,"
                        "textFormatRuns(format/link/uri),"
                        "chipRuns(chip/richLinkProperties/uri)"
                        ")"
                    ),
                )
                .execute()
            )
            data_blocks = ((meta_res.get("sheets") or [{}])[0].get("data")) or []
            for idx_pos, c_idx in enumerate(script_doc_col_indices):
                if idx_pos < len(data_blocks):
                    link_metadata_by_col[c_idx] = data_blocks[idx_pos].get("rowData") or []
        except Exception as exc:
            logger.warning("Could not read rich link metadata from Sheet: %s", exc)

    items: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()
    seen_ad_codes: set[str] = set()

    for row_idx in range(1, len(rows)):
        raw_row = rows[row_idx]
        row_dict: dict[str, Any] = {}
        for col_idx, header in enumerate(headers):
            if not header or header.lower() in forbidden_headers:
                continue
            if col_idx < len(raw_row):
                row_dict[header] = raw_row[col_idx]

        # Resolve Google Doc URL from Script doc or Script Link (plain value or Smart Chip / hyperlink)
        resolved_doc_url = ""
        for c_idx in script_doc_col_indices:
            col_header = headers[c_idx]
            raw_cell_val = str(row_dict.get(col_header) or "").strip()
            if extract_google_doc_id(raw_cell_val):
                resolved_doc_url = raw_cell_val
                break
            col_meta_rows = link_metadata_by_col.get(c_idx) or []
            if row_idx < len(col_meta_rows):
                cell_values = col_meta_rows[row_idx].get("values") or []
                cell_meta = cell_values[0] if cell_values else {}
                candidate_uri = cell_meta.get("hyperlink")
                if not candidate_uri:
                    for chip in cell_meta.get("chipRuns") or []:
                        uri = ((chip.get("chip") or {}).get("richLinkProperties") or {}).get("uri")
                        if uri:
                            candidate_uri = uri
                            break
                if not candidate_uri:
                    for run in cell_meta.get("textFormatRuns") or []:
                        uri = ((run.get("format") or {}).get("link") or {}).get("uri")
                        if uri:
                            candidate_uri = uri
                            break
                if candidate_uri and extract_google_doc_id(candidate_uri):
                    resolved_doc_url = candidate_uri
                    break

        if resolved_doc_url:
            row_dict["doc_url"] = resolved_doc_url

        item = _coerce_row_to_item(row_dict, index=row_idx)
        doc_id = extract_google_doc_id(item.get("doc_url") or "")
        if not doc_id and not item.get("raw_script_text"):
            continue
        if doc_id:
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)

        base_code = item["ad_code"]
        if base_code in seen_ad_codes:
            item["ad_code"] = f"{base_code}_{len(items) + 1}"
        seen_ad_codes.add(item["ad_code"])

        items.append(item)
        if limit and limit > 0 and len(items) >= limit:
            break

    return items


# ---------------------------------------------------------------------------
# 5. Core Ingestion Pipeline (`ingest_proven_scripts` & `ingest_from_google_sheet`)
# ---------------------------------------------------------------------------

def _fallback_parse_proven_doc_text(
    raw_text: str,
    language_hint: str = "Tamil",
    title: str = "",
) -> dict[str, Any]:
    """
    Lightweight fallback parser used only if `script_parser.py` is not yet available.
    """
    lines = [ln.strip() for ln in (raw_text or "").splitlines() if ln.strip()]
    dialogue_lines = [
        ln for ln in lines if re.match(r"^[A-Za-z][A-Za-z0-9 ()/_-]{1,40}:\s+.+", ln)
    ]
    if not dialogue_lines:
        dialogue_lines = lines[-8:] if len(lines) >= 8 else lines

    n = len(dialogue_lines)
    b1 = "\n".join(dialogue_lines[:2])
    b2 = "\n".join(dialogue_lines[2 : max(3, n // 2)])
    b3 = "\n".join(dialogue_lines[max(3, n // 2) : max(4, n - 2)])
    b4 = "\n".join(dialogue_lines[max(0, n - 2) :])

    first_turn_words = len(dialogue_lines[0].split(":", 1)[-1].split()) if dialogue_lines else 0
    total_words = sum(len(ln.split(":", 1)[-1].split()) for ln in dialogue_lines)
    avg_words = round(total_words / max(1, n), 1)

    return {
        "english_script": "\n".join(dialogue_lines),
        "native_script": "\n".join(dialogue_lines),
        "story_blocks": {
            "block_1_hook": b1,
            "block_2_pain_and_old_methods": b2,
            "block_3_live_fix_and_usps": b3,
            "block_4_desire_and_cta": b4,
        },
        "visual_blueprint": {
            "split_screen_type": "Vertical 2-panel split screen (Top: AI Teacher | Bottom: Human Learner)",
            "hook_visuals": "",
            "body_visuals": "",
            "video_flow": "",
            "characters_description": "",
            "raw_visual_text": "",
        },
        "metadata_tags": {
            "persona": "Working Professional",
            "setting": "Home / Study Table",
            "hook_angle": "Spoken Mistake Correction",
            "hook_word_count": first_turn_words,
            "avg_words_per_line": avg_words,
            "total_turns": n,
            "ai_teacher_entry_turn": 2,
            "has_first_half_usps": True,
        },
    }


def _clear_brain_db_store(db_store: Any) -> int:
    """Clears all existing records in `BrainDBStore` (`brain_store.json` and PostgreSQL if active)."""
    existing = db_store.get_all_scripts() if hasattr(db_store, "get_all_scripts") else []
    removed_count = len(existing)

    if hasattr(db_store, "_open_pg_connection"):
        conn = db_store._open_pg_connection()
        if conn is not None:
            try:
                with conn:
                    with conn.cursor() as cur:
                        table_name = getattr(db_store, "TABLE_NAME", "supernova_script_brain")
                        cur.execute(f"DELETE FROM {table_name};")
            except Exception:
                pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

    if hasattr(db_store, "_write_local_records"):
        db_store._write_local_records({})

    return removed_count


def ingest_proven_scripts(
    items: list[dict[str, Any]] | str | Path,
    db_store: Any = None,
    vector_engine: Any = None,
    parser_fn: Callable[..., dict[str, Any]] | None = None,
    docs_service: Any = None,
    reset_existing: bool = False,
) -> dict[str, Any]:
    """
    Ingests a batch of proven Supernova scripts + CAC metrics into `BrainDBStore`
    and `VectorMemoryEngine` with ZERO Gemini API calls.

    For each item (`ad_code`, `doc_url` or `raw_script_text`, `language`, `character_type`, `cac_metrics`):
      1. Fetches the Google Doc text if `doc_url` is provided and `raw_script_text` is not inline,
         selecting the single tab matching `language` (or `"Supernova Script"`) rather than
         concatenating all language tabs.
      2. Calls `parse_proven_doc_text(...)` from `script_parser.py` (separating `visual_blueprint`,
         `story_blocks`, `english_script`, `native_script`, and `metadata_tags`).
      3. Calls `VectorMemoryEngine.embed_script_record(...)` from `vector_memory.py` (0 Gemini calls).
      4. Upserts into `BrainDBStore` (`db_store.py`) and indexes in `VectorMemoryEngine`.
    """
    if isinstance(items, (str, Path)):
        resolved_items = load_items_from_file(items)
    elif isinstance(items, dict):
        resolved_items = items.get("items") or items.get("scripts") or [items]
    else:
        resolved_items = list(items or [])

    if db_store is None:
        from db_store import BrainDBStore

        db_store = BrainDBStore()
    if hasattr(db_store, "init_schema"):
        db_store.init_schema()

    if reset_existing:
        _clear_brain_db_store(db_store)

    if vector_engine is None:
        from vector_memory import VectorMemoryEngine

        vector_engine = VectorMemoryEngine()

    if parser_fn is None:
        try:
            from script_parser import parse_proven_doc_text as _parse_fn

            parser_fn = _parse_fn
        except ImportError:
            parser_fn = _fallback_parse_proven_doc_text

    ingested_records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    tier_counts: dict[str, int] = {"TIER_S": 0, "TIER_A": 0, "TIER_B": 0}
    language_counts: dict[str, int] = {}

    lazy_docs_service = docs_service

    for idx, raw_item in enumerate(resolved_items):
        coerced = _coerce_row_to_item(raw_item, index=idx + 1)
        ad_code = coerced["ad_code"]
        title = coerced["title"]
        doc_url = coerced["doc_url"]
        raw_script_text = coerced["raw_script_text"]
        language = coerced["language"]
        character_type = coerced["character_type"]
        sheet_context = dict(coerced.get("sheet_context") or {})
        available_tabs: list[str] = list(
            coerced.get("available_language_tabs")
            or sheet_context.get("available_language_tabs")
            or []
        )

        try:
            # Step 1: Fetch Google Doc text if `doc_url` is provided and `raw_script_text` is not inline
            if not raw_script_text and doc_url:
                if lazy_docs_service is None:
                    lazy_docs_service = get_docs_service()
                doc_payload = fetch_google_doc_payload(
                    doc_url,
                    preferred_language=language,
                    docs_service=lazy_docs_service,
                )
                raw_script_text = doc_payload.get("text") or ""
                available_tabs = list(doc_payload.get("available_tabs") or [])
                if (not title or title == ad_code) and doc_payload.get("title"):
                    title = doc_payload["title"]

            if not raw_script_text or not raw_script_text.strip():
                raise ValueError(
                    f"No inline `raw_script_text` or readable `doc_url` content for ad_code={ad_code}"
                )

            # Re-verify character_type with the actual script text if it defaulted
            if not raw_item.get("character_type"):
                character_type = normalize_character_type(
                    sheet_context.get("sheet_character") or None,
                    fallback_text=f"{title} {raw_script_text}",
                )

            # Classify CAC tier and normalize metrics
            explicit_tier = raw_item.get("cac_tier") or raw_item.get("tier")
            cac_metrics, cac_tier = normalize_cac_metrics(
                coerced.get("cac_metrics"),
                language=language,
                explicit_tier=explicit_tier,
            )

            # Step 2: Parse Visual Blueprint, Story Blocks, English/Native Script, and Metadata Tags
            try:
                parsed = parser_fn(
                    raw_script_text,
                    language_hint=language,
                    title=title,
                )
            except TypeError:
                # Support parser signatures taking (raw_text, language_hint) or (raw_text,)
                try:
                    parsed = parser_fn(raw_script_text, language)
                except TypeError:
                    parsed = parser_fn(raw_script_text)

            english_script = str(parsed.get("english_script") or raw_script_text).strip()
            native_script = str(parsed.get("native_script") or raw_script_text).strip()
            story_blocks = parsed.get("story_blocks") or {}
            visual_blueprint = parsed.get("visual_blueprint") or {}
            metadata_tags = dict(parsed.get("metadata_tags") or {})

            if available_tabs:
                metadata_tags["available_language_tabs"] = available_tabs
            if sheet_context:
                sheet_context["available_language_tabs"] = available_tabs
                metadata_tags["sheet_context"] = sheet_context

            now_iso = dt.datetime.now(dt.timezone.utc).isoformat()

            record: dict[str, Any] = {
                "ad_code": ad_code,
                "title": title,
                "doc_url": doc_url,
                "language": language,
                "character_type": character_type,
                "cac_tier": cac_tier,
                "cac_metrics": cac_metrics,
                "english_script": english_script,
                "native_script": native_script,
                "story_blocks": story_blocks,
                "visual_blueprint": visual_blueprint,
                "metadata_tags": metadata_tags,
                "hook_vector": [],
                "script_vector": [],
                "updated_at": now_iso,
            }

            # Step 3: Local vector embedding via VectorMemoryEngine (ZERO Gemini API calls)
            emb_result = vector_engine.embed_script_record(record)
            if isinstance(emb_result, tuple) and len(emb_result) == 2:
                hook_vec, script_vec = emb_result
            elif isinstance(emb_result, dict):
                hook_vec = emb_result.get("hook_vector") or []
                script_vec = emb_result.get("script_vector") or []
            else:
                hook_vec, script_vec = [], []

            record["hook_vector"] = list(hook_vec) if hook_vec is not None else []
            record["script_vector"] = list(script_vec) if script_vec is not None else []

            # Step 4: Upsert into BrainDBStore
            db_store.upsert_script(record)
            ingested_records.append(record)
            tier_counts[cac_tier] = tier_counts.get(cac_tier, 0) + 1
            language_counts[language] = language_counts.get(language, 0) + 1

        except Exception as exc:
            logger.warning("Failed to ingest script %s: %s", ad_code, exc)
            errors.append({"ad_code": ad_code, "doc_url": doc_url, "error": str(exc)})

    # Index all stored records into VectorMemoryEngine (ChromaDB / local memory)
    all_records = (
        db_store.get_all_scripts()
        if hasattr(db_store, "get_all_scripts")
        else ingested_records
    )
    indexed_count = 0
    if all_records and hasattr(vector_engine, "index_scripts"):
        indexed_count = vector_engine.index_scripts(all_records)

    db_stats = db_store.get_stats() if hasattr(db_store, "get_stats") else {}

    return {
        "success": len(errors) == 0 or len(ingested_records) > 0,
        "ingested_count": len(ingested_records),
        "failed_count": len(errors),
        "tier_counts": tier_counts,
        "language_counts": language_counts,
        "indexed_vector_count": indexed_count,
        "ad_codes": [r["ad_code"] for r in ingested_records],
        "ingested_scripts": [
            {
                "ad_code": r["ad_code"],
                "title": r["title"],
                "doc_url": r["doc_url"],
                "language": r["language"],
                "character_type": r["character_type"],
                "cac_tier": r["cac_tier"],
                "cac_metrics": r["cac_metrics"],
                "metadata_tags": r["metadata_tags"],
            }
            for r in ingested_records
        ],
        "errors": errors,
        "db_stats": db_stats,
    }


def ingest_from_google_sheet(
    spreadsheet_id: str | None = DEFAULT_REPLICATION_SHEET_ID,
    sheet_name: str | None = DEFAULT_REPLICATION_TAB,
    limit: int | None = None,
    reset_existing: bool = False,
    db_store: Any = None,
    vector_engine: Any = None,
    parser_fn: Callable[..., dict[str, Any]] | None = None,
    docs_service: Any = None,
    sheets_service: Any = None,
) -> dict[str, Any]:
    """
    Fetches proven script rows + CAC performance metrics from the Supernova Replication Queue
    Google Sheet (`DEFAULT_REPLICATION_SHEET_ID`, tab `DEFAULT_REPLICATION_TAB`) and ingests
    them into `BrainDBStore` and `VectorMemoryEngine` with ZERO Gemini API calls.

    - Strictly ignores `YouTube link`, `Notion page`, and `Notion Link` columns.
    - Deduplicates by Google Doc `doc_id` and skips rows without a valid Google Doc link.
    - Selects the single tab matching `Source Language` (e.g., `"Hindi"`) from multi-tab Google Docs.
    - When `reset_existing=True`, clears all existing records in `BrainDBStore` before upserting.
    """
    resolved_sheet_id = (spreadsheet_id or DEFAULT_REPLICATION_SHEET_ID).strip()
    resolved_sheet_name = (sheet_name or DEFAULT_REPLICATION_TAB).strip()

    if db_store is None:
        from db_store import BrainDBStore

        db_store = BrainDBStore()
    if hasattr(db_store, "init_schema"):
        db_store.init_schema()

    removed_before_ingest = 0
    if reset_existing:
        removed_before_ingest = _clear_brain_db_store(db_store)

    items = fetch_items_from_google_sheet(
        spreadsheet_id=resolved_sheet_id,
        sheet_name=resolved_sheet_name,
        limit=limit,
        sheets_service=sheets_service,
    )

    result = ingest_proven_scripts(
        items=items,
        db_store=db_store,
        vector_engine=vector_engine,
        parser_fn=parser_fn,
        docs_service=docs_service,
        reset_existing=False,
    )
    result["spreadsheet_id"] = resolved_sheet_id
    result["sheet_name"] = resolved_sheet_name
    result["reset_existing"] = bool(reset_existing)
    result["removed_before_ingest"] = removed_before_ingest
    return result


# ---------------------------------------------------------------------------
# 6. Realistic Supernova Demo Benchmark Scripts (`seed_demo_winners_if_empty`)
# ---------------------------------------------------------------------------

DEMO_BENCHMARK_SCRIPTS: list[dict[str, Any]] = [
    # 1. TAMIL — TIER_S WINNER (-28.6% vs Tamil average CAC)
    {
        "ad_code": "SN3195_TAM",
        "title": "SN3195 - Myself Introduction Mistake in Job Interview (Tamil)",
        "doc_url": "https://docs.google.com/document/d/1SN3195TamilInterviewTierSWinner/edit",
        "language": "Tamil",
        "character_type": "Female Humanoid Robot",
        "cac_metrics": {
            "first_pay_cac": 185.0,
            "post_1hr_cancel_cac": 194.0,
            "second_pay_cac_4d": 205.0,
            "second_pay_cac_7d": 200.0,
            "cancel_pct": 14.2,
            "lang_avg_cac": 280.0,
            "cac_vs_avg_pct": -28.57,
        },
        "raw_script_text": """Visual Description:
Split Screen & Characters: Horizontal 2-panel split screen. Above is the Female Humanoid Robot in a bright studio; below is Karthik, a young man in his mid-20s wearing an Indian formal shirt at an interview desk.
Hook visuals: Karthik nervously clasping his hands at an office interview desk while introducing himself with "Myself Karthik".
Body visuals: Female Humanoid Robot gesturing warmly on the top split screen while correcting his sentence on the spot with live text callouts.
Video flow: A young job seeker starts his interview with "Myself Karthik", gets immediately corrected by the AI teacher on the spot, and practices confident interview answers through Tamil.

Reference link : https://pub-demo.r2.dev/SN3195_TAM.mp4

Characters
• Karthik -- Man, mid-20s, Indian formal shirt and trousers, sitting nervously at a job interview desk.
• Female Humanoid Robot -- The AI spoken English teacher. Sleek white-and-blue humanoid tutor in a modern studio split screen.

Script:
Karthik: Hello sir, myself Karthik, I having four years experience in accounts.
Female Humanoid Robot: Stop right there! Starting your job interview with "Myself Karthik" is the fastest way to lose confidence in front of HR.
Karthik: Everyone in my office says "myself" only! In every job interview, the moment they ask a follow-up question in English, my mind goes blank.
Female Humanoid Robot: That happens because you only memorized rules. With me, by practicing just 15 to 20 minutes a day, you can improve your spoken English within 30 days.
Karthik: Really? I already tried watching YouTube videos, bought grammar books, and even joined coaching classes, but I still get stuck while speaking.
Female Humanoid Robot: Because those don't talk back to you! As your AI spoken English teacher, I explain your mistakes through Tamil itself, and over 1 crore Indians are already practicing with me.
Karthik: Will you know where I get stuck during office meetings and interviews?
Female Humanoid Robot: Yes! First, I take a quick level test to see exactly where you hesitate, and then I create personalized conversation activities just for you. Say this now: "Hi, I am Karthik, and I have four years of experience in accounting."
Karthik: Hi, I am Karthik, and I have four years of experience in accounting. Wow, you corrected me on the spot without judging me!
Female Humanoid Robot: With me, you can make as many mistakes as you want without any fear of being laughed at, and I correct you on the spot.
Karthik: Super! I want to start practicing right now before my next interview. Where do I start?
Female Humanoid Robot: Within 30 days, you will improve your spoken English! Click the Install Now button given below, download Supernova AI, and start practicing with me today itself.""",
    },
    # 2. TELUGU — TIER_S WINNER (-23.7% vs Telugu average CAC)
    {
        "ad_code": "SN3218_TEL",
        "title": "SN3218 - Homemaker Mother PTM School Teacher Panic (Telugu)",
        "doc_url": "https://docs.google.com/document/d/1SN3218TeluguPTMTierSWinner/edit",
        "language": "Telugu",
        "character_type": "Miss Nova",
        "cac_metrics": {
            "first_pay_cac": 210.0,
            "post_1hr_cancel_cac": 218.0,
            "second_pay_cac_4d": 222.0,
            "second_pay_cac_7d": 225.0,
            "cancel_pct": 15.0,
            "lang_avg_cac": 295.0,
            "cac_vs_avg_pct": -23.73,
        },
        "raw_script_text": """Visual Description:
Split Screen & Characters: Horizontal split-screen layout. Above is Miss Nova (3D animated female AI teacher with glasses); below is Swapna, a young woman in her late 20s wearing a traditional saree at her dining table with her child's school diary.
Hook visuals: Swapna holding her son's report card looking anxious and speaking broken English.
Body visuals: Miss Nova smiling encouragingly and guiding the mother through a real parent-teacher meeting conversation.
Video flow: A homemaker mother struggles to speak to her child's class teacher at PTM, gets corrected gently by Miss Nova, and practices real PTM conversations through Telugu.

Reference link : https://pub-demo.r2.dev/SN3218_TEL.mp4

Characters
• Swapna -- Woman, late 20s, Traditional cotton saree, sitting at home with her child's school notebook.
• Miss Nova -- The AI spoken English teacher. Warm 3D animated teacher in a friendly classroom interface.

Script:
Swapna: Ma'am, my son homework doing daily, why marks less coming in exam?
Miss Nova: Hold on! Instead of saying "homework doing daily", say "Ma'am, my son does his homework every day, why are his marks low?"
Swapna: At every parent-teacher meeting in my son's school, when the teacher speaks fast English, I feel so embarrassed in front of other parents and my own child.
Miss Nova: Don't worry at all! As your AI teacher, if you practice with me for 15 to 20 minutes daily, you will improve your spoken English within 30 days.
Swapna: I tried watching YouTube videos while cooking and even kept grammar notes in a diary, but when I stand in front of the school teacher, no words come out.
Miss Nova: Watching videos won't give you speaking confidence. I guide you step by step through Telugu itself, which is why over 1 crore Indians trust me for daily speaking practice.
Swapna: Can I actually talk to you like I talk to my child's school teacher?
Miss Nova: Absolutely! First, I check your current speaking level and where you get stuck, then build personalized roleplay activities for PTM, doctor visits, and family functions. You can speak freely without fear of anyone judging you, and I correct every mistake on the spot.
Swapna: That feels so comforting! I want to start practicing right now before Saturday's PTM. How do I join?
Miss Nova: Within 30 days, you will improve your spoken English! Click the Install Now button given below, download Supernova AI, and start practicing with me today itself.""",
    },
    # 3. HINDI — TIER_S WINNER (-21.9% vs Hindi average CAC)
    {
        "ad_code": "SN3244_HIN",
        "title": "SN3244 - Office Team Meeting Presentation Freeze (Hindi)",
        "doc_url": "https://docs.google.com/document/d/1SN3244HindiOfficeTierSWinner/edit",
        "language": "Hindi",
        "character_type": "Female Robot",
        "cac_metrics": {
            "first_pay_cac": 232.0,
            "post_1hr_cancel_cac": 240.0,
            "second_pay_cac_4d": 246.0,
            "second_pay_cac_7d": 250.0,
            "cancel_pct": 16.1,
            "lang_avg_cac": 320.0,
            "cac_vs_avg_pct": -21.88,
        },
        "raw_script_text": """Visual Description:
Split Screen & Characters: Horizontal 2-panel split screen. Above is Female Robot; below is Rahul, a man in his early 30s wearing Indian formal office attire in front of an open laptop.
Hook visuals: Rahul looking stressed at his laptop screen during an office Zoom call, stammering a broken sentence.
Body visuals: Female Robot correcting his sentence structure live while he repeats the polished version confidently.
Video flow: A working professional stumbles over broken English in a client status update, gets an instant correction from the AI teacher, and practices workplace English through Hindi.

Reference link : https://pub-demo.r2.dev/SN3244_HIN.mp4

Characters
• Rahul -- Man, early 30s, Indian formal shirt, sitting at an office desk with a laptop.
• Female Robot -- The AI spoken English teacher. Friendly robotic tutor in a clean split-screen studio.

Script:
Rahul: Sir, yesterday report I already sending on mail, please checking once.
Female Robot: Wait! Don't say "yesterday report I already sending" to your manager—say, "Sir, I have already sent yesterday's report by email."
Rahul: I know the work better than anyone in my team, but in Monday team meetings and client calls, my juniors speak fluent English and get the promotion.
Female Robot: You don't have to stay silent anymore! Just practice speaking with me for 15 to 20 minutes a day and improve your spoken English within 30 days.
Rahul: I wasted months on YouTube videos, grammar books, and expensive coaching classes, yet I still freeze when my boss asks a question in English.
Female Robot: Because you need real conversation practice! As your AI spoken English teacher, I explain everything clearly through Hindi itself, and over 1 crore Indians are already improving their confidence with me.
Rahul: How will you help me with my exact office meetings and presentations?
Female Robot: I first understand your current English level and where you hesitate while speaking, then design personalized office and interview conversation practice for you. Even if you make ten mistakes, nobody laughs at you—I correct you on the spot.
Rahul: This is exactly what I needed for my career. I want to start practicing right now—where can I get it?
Female Robot: Within 30 days, you will improve your spoken English! Click the Install Now button given below, download Supernova AI, and start practicing with me today itself.""",
    },
    # 4. TAMIL — TIER_A WINNER (-10.0% vs Tamil average CAC)
    {
        "ad_code": "SN3270_TAM",
        "title": "SN3270 - Doctor Hospital Appointment Conversation (Tamil)",
        "doc_url": "https://docs.google.com/document/d/1SN3270TamilDoctorTierAWinner/edit",
        "language": "Tamil",
        "character_type": "Female Humanoid Robot",
        "cac_metrics": {
            "first_pay_cac": 238.0,
            "post_1hr_cancel_cac": 245.0,
            "second_pay_cac_4d": 249.0,
            "second_pay_cac_7d": 252.0,
            "cancel_pct": 17.5,
            "lang_avg_cac": 280.0,
            "cac_vs_avg_pct": -10.0,
        },
        "raw_script_text": """Visual Description:
Split Screen & Characters: Horizontal split screen. Above is Female Humanoid Robot; below is Priya, a young woman in casual home wear holding a medical prescription file.
Hook visuals: Priya trying to explain stomach pain in broken English at a clinic desk.
Body visuals: Female Humanoid Robot coaching her through natural English sentences with clear Tamil explanations.
Video flow: A young woman struggles to explain symptoms to a city doctor in English, gets corrected on the spot by the AI teacher, and builds fluency through Tamil.

Reference link : https://pub-demo.r2.dev/SN3270_TAM.mp4

Characters
• Priya -- Woman, late 20s, Casual kurti, holding a prescription folder at home.
• Female Humanoid Robot -- The AI spoken English teacher. Modern humanoid AI teacher on the top split screen.

Script:
Priya: Doctor sir, since morning my stomach paining too much, what tablet eating?
Female Humanoid Robot: Careful! Don't say "stomach paining, what tablet eating"—say, "Doctor, I have had severe stomach pain since morning, which medicine should I take?"
Priya: Whenever I visit the hospital, bank, or my child's school, I understand what they say in English, but when I try to reply, I get stuck.
Female Humanoid Robot: That happens to so many people! Practice with me for just 15 to 20 minutes every day, and you can improve your spoken English within 30 days.
Priya: I tried watching YouTube videos and reading grammar books at night, but I had nobody at home to speak English with.
Female Humanoid Robot: Now you have me as your AI teacher! I explain every sentence through Tamil itself, and over 1 crore Indians already practice speaking with me every day.
Priya: What if I make silly mistakes while talking to you?
Female Humanoid Robot: Make as many mistakes as you want! I never judge you. First, I check your current English level and where you get stuck, then give you personalized daily conversation practice and correct you on the spot.
Priya: That sounds so easy! I want to start practicing today. How do I begin?
Female Humanoid Robot: Within 30 days, you will improve your spoken English! Click the Install Now button given below, download Supernova AI, and start practicing with me today itself.""",
    },
    # 5. TELUGU — TIER_B UNDERPERFORMER (+18.6% higher CAC than Telugu average)
    {
        "ad_code": "SN3110_TEL_B",
        "title": "SN3110 - Generic English Study Monologue (Telugu Tier-B)",
        "doc_url": "https://docs.google.com/document/d/1SN3110TeluguGenericTierBLoser/edit",
        "language": "Telugu",
        "character_type": "Female Robot",
        "cac_metrics": {
            "first_pay_cac": 330.0,
            "post_1hr_cancel_cac": 342.0,
            "second_pay_cac_4d": 346.0,
            "second_pay_cac_7d": 350.0,
            "cancel_pct": 26.4,
            "lang_avg_cac": 295.0,
            "cac_vs_avg_pct": 18.64,
        },
        "raw_script_text": """Visual Description:
Split Screen & Characters: Single full-screen talking head for the first 15 seconds before switching to a static app screen.
Hook visuals: Young Man sitting on a sofa talking slowly about how important communication skills are in modern life.
Body visuals: Generic app screen slide showing text bullets.
Video flow: A man talks at length about wanting better communication skills in daily life before the AI teacher appears late in the video.

Characters
• Young Man -- Man, late 20s, Casual t-shirt, sitting on a couch at home.
• Female Robot -- The AI spoken English teacher.

Script:
Young Man: Friends, nowadays in our modern society English communication is very important for everyone in all kinds of daily situations and general life.
Young Man: Many times I have thought that if my English vocabulary and grammar were stronger, I would feel much more confident everywhere I go.
Young Man: I tried watching some YouTube videos and coaching classes in the past, but nothing really changed for me over the years.
Female Robot: Hello! I am your AI teacher and I can help you practice English conversations whenever you have free time at home.
Young Man: How does this work for someone like me who wants to practice?
Female Robot: First I check your level and give you personalized speaking practice, and I correct your mistakes on the spot without any judgement.
Young Man: How much time does it take to see results?
Female Robot: Just practice for 15 to 20 minutes daily with over 1 crore Indians, and through Telugu itself you can improve your spoken English within 30 days.
Young Man: Okay, I want to try this right now. Where do I find it?
Female Robot: Within 30 days, you will improve your spoken English! Click the Install Now button given below, download Supernova AI, and start practicing with me today itself.""",
    },
    # 6. HINDI — TIER_B UNDERPERFORMER (+21.9% higher CAC than Hindi average)
    {
        "ad_code": "SN3142_HIN_B",
        "title": "SN3142 - Long Rambling Intro Without Spoken Mistake Hook (Hindi Tier-B)",
        "doc_url": "https://docs.google.com/document/d/1SN3142HindiRamblingTierBLoser/edit",
        "language": "Hindi",
        "character_type": "Female Humanoid Robot",
        "cac_metrics": {
            "first_pay_cac": 365.0,
            "post_1hr_cancel_cac": 378.0,
            "second_pay_cac_4d": 384.0,
            "second_pay_cac_7d": 390.0,
            "cancel_pct": 28.0,
            "lang_avg_cac": 320.0,
            "cac_vs_avg_pct": 21.88,
        },
        "raw_script_text": """Visual Description:
Split Screen & Characters: Single frame of a woman walking in a corridor talking to camera, with late AI teacher appearance.
Hook visuals: Woman explaining her general thoughts about spoken English for 12 seconds without making any concrete spoken mistake.
Body visuals: Woman looking at her phone screen quietly.
Video flow: A long monologue about wanting fluency in daily life with late AI tutor entry and back-loaded USPs.

Characters
• Young Woman -- Woman, early 30s, Casual salwar suit, standing in a hallway.
• Female Humanoid Robot -- The AI spoken English teacher.

Script:
Young Woman: Have you ever noticed that whenever we go out in daily situations and meet different people, everybody expects us to speak fluent English?
Young Woman: For so many years I kept thinking that one day I will also speak fluently, so I watched YouTube videos and read grammar books.
Young Woman: Still, whenever someone talks to me in English, I feel hesitant and nervous because I do not have a speaking partner at home.
Female Humanoid Robot: Hi there! I am your AI spoken English teacher, and you can practice speaking with me without any fear of judgement.
Young Woman: How do you help me practice my conversations?
Female Robot: I check your level first, build personalized practice activities, and correct you on the spot whenever you make a mistake.
Young Woman: Can you explain things in my own language too?
Female Humanoid Robot: Yes, I explain through Hindi itself! Over 1 crore Indians practice 15 to 20 minutes daily with me to improve your spoken English within 30 days.
Young Woman: Great, I want to start practicing right now. Where can I download it?
Female Humanoid Robot: Within 30 days, you will improve your spoken English! Click the Install Now button given below, download Supernova AI, and start practicing with me today itself.""",
    },
]


def seed_demo_winners_if_empty(
    db_store: Any = None,
    vector_engine: Any = None,
    force: bool = False,
) -> dict[str, Any]:
    """
    Seeds 6 realistic Supernova benchmark scripts (`TIER_S`, `TIER_A`, `TIER_B`
    across Tamil, Telugu, and Hindi) if `BrainDBStore` currently has 0 scripts
    (or when `force=True`), with ZERO Gemini API calls.
    """
    if db_store is None:
        from db_store import BrainDBStore

        db_store = BrainDBStore()
    if hasattr(db_store, "init_schema"):
        db_store.init_schema()

    existing = db_store.get_all_scripts() if hasattr(db_store, "get_all_scripts") else []
    if existing and not force:
        return {
            "seeded": False,
            "reason": "Brain store already contains proven scripts.",
            "existing_count": len(existing),
            "db_stats": db_store.get_stats() if hasattr(db_store, "get_stats") else {},
        }

    ingest_res = ingest_proven_scripts(
        items=DEMO_BENCHMARK_SCRIPTS,
        db_store=db_store,
        vector_engine=vector_engine,
    )
    return {
        "seeded": True,
        "seeded_count": ingest_res.get("ingested_count", 0),
        **ingest_res,
    }


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest proven Supernova scripts + CAC metrics into the Vector + SQL Brain (0 Gemini API calls)."
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to JSON, JSONL, or CSV file containing proven scripts and CAC metrics.",
    )
    parser.add_argument(
        "--from-sheet",
        action="store_true",
        help="Ingest proven scripts from the Supernova Replication Queue Google Sheet.",
    )
    parser.add_argument(
        "--sheet-id",
        type=str,
        default=None,
        help=f"Google Spreadsheet ID to ingest proven scripts from (default: {DEFAULT_REPLICATION_SHEET_ID}).",
    )
    parser.add_argument(
        "--sheet-name",
        "--tab",
        dest="sheet_name",
        type=str,
        default=None,
        help=f"Google Sheet tab name (default: {DEFAULT_REPLICATION_TAB!r}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max rows to ingest when pulling from a Google Sheet (default: 100).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing records in BrainDBStore before ingesting.",
    )
    parser.add_argument(
        "--seed-demo",
        action="store_true",
        help="Seed the 6 realistic Supernova benchmark scripts if the brain is empty.",
    )
    parser.add_argument(
        "--force-seed",
        action="store_true",
        help="Force re-seeding the 6 realistic Supernova benchmark scripts.",
    )

    args = parser.parse_args(argv)

    if args.seed_demo or args.force_seed:
        result = seed_demo_winners_if_empty(force=args.force_seed)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if args.file:
        items = load_items_from_file(args.file)
        result = ingest_proven_scripts(items, reset_existing=args.reset)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("success") else 1

    if args.from_sheet or args.sheet_id:
        result = ingest_from_google_sheet(
            spreadsheet_id=args.sheet_id or DEFAULT_REPLICATION_SHEET_ID,
            sheet_name=args.sheet_name or DEFAULT_REPLICATION_TAB,
            limit=args.limit,
            reset_existing=args.reset,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("success") else 1

    # Default action when invoked without flags: seed demo winners if empty
    result = seed_demo_winners_if_empty(force=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
