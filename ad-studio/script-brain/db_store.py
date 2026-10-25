"""
PostgreSQL + Local JSON Fallback Store for the Supernova Script Brain.

Manages the `supernova_script_brain` table in PostgreSQL (`DATABASE_URL` loaded
from the project root `.env`) while maintaining a synchronized local mirror at
`ad-studio/script-brain/data/brain_store.json` for zero-latency local reads and
transparent offline fallback.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import urlparse

logger = logging.getLogger("supernova.script_brain.db_store")

# ---------------------------------------------------------------------------
# 1. Environment & Paths Configuration
# ---------------------------------------------------------------------------

SCRIPT_BRAIN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_BRAIN_DIR.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_JSON_STORE_PATH = SCRIPT_BRAIN_DIR / "data" / "brain_store.json"
TABLE_NAME = "supernova_script_brain"


def _load_project_env(env_path: Path = ENV_PATH) -> None:
    """Load environment variables from `../../.env` relative to this file."""
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]

        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
            return
    except ImportError:
        pass

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
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                val = val[1:-1]
            if key and key not in os.environ:
                os.environ[key] = val
    except OSError as exc:
        logger.debug("Could not read .env at %s: %s", env_path, exc)


_load_project_env()

# ---------------------------------------------------------------------------
# 2. Data Structures & Schema Definitions
# ---------------------------------------------------------------------------

VALID_LANGUAGES: tuple[str, ...] = (
    "Tamil",
    "Telugu",
    "Malayalam",
    "Kannada",
    "Hindi",
    "English",
)

VALID_CHARACTER_TYPES: tuple[str, ...] = (
    "Female Humanoid Robot",
    "Female Robot",
    "Miss Nova",
)

VALID_CAC_TIERS: tuple[str, ...] = (
    "TIER_S",
    "TIER_A",
    "TIER_B",
)

TIER_SORT_ORDER: dict[str, int] = {
    "TIER_S": 0,
    "TIER_A": 1,
    "TIER_B": 2,
}


class CACMetricsDict(TypedDict, total=False):
    first_pay_cac: float | None
    post_1hr_cancel_cac: float | None
    second_pay_cac_4d: float | None
    second_pay_cac_7d: float | None
    cancel_pct: float | None
    lang_avg_cac: float | None
    cac_vs_avg_pct: float | None


class StoryBlocksDict(TypedDict, total=False):
    block_1_hook: str
    block_2_pain_and_old_methods: str
    block_3_live_fix_and_usps: str
    block_4_desire_and_cta: str


class VisualBlueprintDict(TypedDict, total=False):
    split_screen_type: str
    hook_visuals: str
    body_visuals: str
    video_flow: str
    characters_description: str
    raw_visual_text: str


class MetadataTagsDict(TypedDict, total=False):
    persona: str
    setting: str
    hook_angle: str
    hook_word_count: int
    avg_words_per_line: float
    total_turns: int
    ai_teacher_entry_turn: int
    has_first_half_usps: bool


class ScriptRecordDict(TypedDict, total=False):
    ad_code: str
    title: str
    doc_url: str
    language: str
    character_type: str
    cac_tier: str
    cac_metrics: CACMetricsDict
    english_script: str
    native_script: str
    story_blocks: StoryBlocksDict
    visual_blueprint: VisualBlueprintDict
    metadata_tags: MetadataTagsDict
    hook_vector: list[float]
    script_vector: list[float]
    updated_at: str


DEFAULT_CAC_METRICS: CACMetricsDict = {
    "first_pay_cac": None,
    "post_1hr_cancel_cac": None,
    "second_pay_cac_4d": None,
    "second_pay_cac_7d": None,
    "cancel_pct": None,
    "lang_avg_cac": None,
    "cac_vs_avg_pct": None,
}

DEFAULT_STORY_BLOCKS: StoryBlocksDict = {
    "block_1_hook": "",
    "block_2_pain_and_old_methods": "",
    "block_3_live_fix_and_usps": "",
    "block_4_desire_and_cta": "",
}

DEFAULT_VISUAL_BLUEPRINT: VisualBlueprintDict = {
    "split_screen_type": "",
    "hook_visuals": "",
    "body_visuals": "",
    "video_flow": "",
    "characters_description": "",
    "raw_visual_text": "",
}

DEFAULT_METADATA_TAGS: MetadataTagsDict = {
    "persona": "",
    "setting": "",
    "hook_angle": "",
    "hook_word_count": 0,
    "avg_words_per_line": 0.0,
    "total_turns": 0,
    "ai_teacher_entry_turn": 0,
    "has_first_half_usps": False,
}


@dataclass
class ProvenScriptRecord:
    """Dataclass representation of a single proven Supernova script record."""

    ad_code: str
    title: str = ""
    doc_url: str = ""
    language: str = "English"
    character_type: str = "Miss Nova"
    cac_tier: str = "TIER_B"
    cac_metrics: dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_CAC_METRICS)
    )
    english_script: str = ""
    native_script: str = ""
    story_blocks: dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_STORY_BLOCKS)
    )
    visual_blueprint: dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_VISUAL_BLUEPRINT)
    )
    metadata_tags: dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_METADATA_TAGS)
    )
    hook_vector: list[float] = field(default_factory=list)
    script_vector: list[float] = field(default_factory=list)
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return normalize_script_record(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProvenScriptRecord":
        norm = normalize_script_record(data)
        return cls(
            ad_code=norm["ad_code"],
            title=norm["title"],
            doc_url=norm["doc_url"],
            language=norm["language"],
            character_type=norm["character_type"],
            cac_tier=norm["cac_tier"],
            cac_metrics=norm["cac_metrics"],
            english_script=norm["english_script"],
            native_script=norm["native_script"],
            story_blocks=norm["story_blocks"],
            visual_blueprint=norm["visual_blueprint"],
            metadata_tags=norm["metadata_tags"],
            hook_vector=norm["hook_vector"],
            script_vector=norm["script_vector"],
            updated_at=norm["updated_at"],
        )


CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    ad_code VARCHAR(64) PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    doc_url TEXT NOT NULL DEFAULT '',
    language VARCHAR(50) NOT NULL DEFAULT 'English',
    character_type VARCHAR(100) NOT NULL DEFAULT 'Miss Nova',
    cac_tier VARCHAR(20) NOT NULL DEFAULT 'TIER_B',
    cac_metrics JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    english_script TEXT NOT NULL DEFAULT '',
    native_script TEXT NOT NULL DEFAULT '',
    story_blocks JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    visual_blueprint JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    metadata_tags JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    hook_vector DOUBLE PRECISION[] NOT NULL DEFAULT '{{}}',
    script_vector DOUBLE PRECISION[] NOT NULL DEFAULT '{{}}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_lang_tier
    ON {TABLE_NAME} (language, cac_tier);

CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_updated_at
    ON {TABLE_NAME} (updated_at DESC);
"""

UPSERT_SCRIPT_SQL = f"""
INSERT INTO {TABLE_NAME} (
    ad_code,
    title,
    doc_url,
    language,
    character_type,
    cac_tier,
    cac_metrics,
    english_script,
    native_script,
    story_blocks,
    visual_blueprint,
    metadata_tags,
    hook_vector,
    script_vector,
    updated_at
) VALUES (
    %s, %s, %s, %s, %s, %s,
    %s::jsonb, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb,
    %s::double precision[], %s::double precision[], %s::timestamptz
)
ON CONFLICT (ad_code) DO UPDATE SET
    title = EXCLUDED.title,
    doc_url = EXCLUDED.doc_url,
    language = EXCLUDED.language,
    character_type = EXCLUDED.character_type,
    cac_tier = EXCLUDED.cac_tier,
    cac_metrics = EXCLUDED.cac_metrics,
    english_script = EXCLUDED.english_script,
    native_script = EXCLUDED.native_script,
    story_blocks = EXCLUDED.story_blocks,
    visual_blueprint = EXCLUDED.visual_blueprint,
    metadata_tags = EXCLUDED.metadata_tags,
    hook_vector = EXCLUDED.hook_vector,
    script_vector = EXCLUDED.script_vector,
    updated_at = EXCLUDED.updated_at;
"""


def _coerce_float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_vector(value: Any) -> list[float]:
    if not value:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                value = parsed
        except (ValueError, TypeError):
            return []
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for item in value:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                continue
        return out
    return []


def _coerce_json_dict(raw: Any, defaults: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(defaults)
    if isinstance(raw, str) and raw.strip():
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            raw = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if v is not None or k not in merged:
                merged[k] = v
    return merged


def _normalize_language(language: Any) -> str:
    if not language or not isinstance(language, str):
        return "English"
    cleaned = language.strip()
    for valid in VALID_LANGUAGES:
        if cleaned.lower() == valid.lower():
            return valid
    return cleaned


def _normalize_character_type(character_type: Any) -> str:
    if not character_type or not isinstance(character_type, str):
        return "Miss Nova"
    cleaned = character_type.strip()
    for valid in VALID_CHARACTER_TYPES:
        if cleaned.lower() == valid.lower():
            return valid
    return cleaned


def _normalize_cac_tier(cac_tier: Any) -> str:
    if not cac_tier or not isinstance(cac_tier, str):
        return "TIER_B"
    cleaned = cac_tier.strip().upper().replace(" ", "_").replace("-", "_")
    if cleaned in ("S", "TIERS"):
        return "TIER_S"
    if cleaned in ("A", "TIERA"):
        return "TIER_A"
    if cleaned in ("B", "TIERB"):
        return "TIER_B"
    if cleaned in VALID_CAC_TIERS:
        return cleaned
    return cleaned


def normalize_script_record(
    record: dict[str, Any] | ProvenScriptRecord,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Normalize a raw dict or ProvenScriptRecord into the canonical schema structure.
    If `existing` is provided, merges incoming fields over `existing`.
    """
    if dataclasses.is_dataclass(record) and not isinstance(record, type):
        raw_dict = dataclasses.asdict(record)
    elif hasattr(record, "model_dump") and callable(record.model_dump):
        raw_dict = record.model_dump()
    elif hasattr(record, "dict") and callable(record.dict):
        raw_dict = record.dict()
    elif isinstance(record, dict):
        raw_dict = dict(record)
    else:
        raise TypeError(f"Unsupported script record type: {type(record)!r}")

    base: dict[str, Any] = copy.deepcopy(existing) if existing else {}
    for key, val in raw_dict.items():
        if key in ("cac_metrics", "story_blocks", "visual_blueprint", "metadata_tags"):
            if isinstance(val, dict) and isinstance(base.get(key), dict):
                merged_sub = dict(base[key])
                merged_sub.update(val)
                base[key] = merged_sub
            elif val is not None:
                base[key] = val
        elif val is not None or key not in base:
            base[key] = val

    ad_code = str(base.get("ad_code") or "").strip().upper()
    if not ad_code:
        raise ValueError("Script record must contain a non-empty 'ad_code'.")

    raw_cac = _coerce_json_dict(base.get("cac_metrics"), dict(DEFAULT_CAC_METRICS))
    cac_metrics: dict[str, Any] = {}
    for k, v in raw_cac.items():
        if k in DEFAULT_CAC_METRICS:
            cac_metrics[k] = _coerce_float_or_none(v)
        else:
            cac_metrics[k] = v

    story_blocks = _coerce_json_dict(
        base.get("story_blocks"), dict(DEFAULT_STORY_BLOCKS)
    )
    for k in DEFAULT_STORY_BLOCKS:
        story_blocks[k] = str(story_blocks.get(k) or "")

    visual_blueprint = _coerce_json_dict(
        base.get("visual_blueprint"), dict(DEFAULT_VISUAL_BLUEPRINT)
    )
    for k in DEFAULT_VISUAL_BLUEPRINT:
        visual_blueprint[k] = str(visual_blueprint.get(k) or "")

    metadata_tags = _coerce_json_dict(
        base.get("metadata_tags"), dict(DEFAULT_METADATA_TAGS)
    )
    metadata_tags["persona"] = str(metadata_tags.get("persona") or "")
    metadata_tags["setting"] = str(metadata_tags.get("setting") or "")
    metadata_tags["hook_angle"] = str(metadata_tags.get("hook_angle") or "")
    try:
        metadata_tags["hook_word_count"] = int(metadata_tags.get("hook_word_count") or 0)
    except (TypeError, ValueError):
        metadata_tags["hook_word_count"] = 0
    try:
        metadata_tags["avg_words_per_line"] = float(
            metadata_tags.get("avg_words_per_line") or 0.0
        )
    except (TypeError, ValueError):
        metadata_tags["avg_words_per_line"] = 0.0
    try:
        metadata_tags["total_turns"] = int(metadata_tags.get("total_turns") or 0)
    except (TypeError, ValueError):
        metadata_tags["total_turns"] = 0
    try:
        metadata_tags["ai_teacher_entry_turn"] = int(
            metadata_tags.get("ai_teacher_entry_turn") or 0
        )
    except (TypeError, ValueError):
        metadata_tags["ai_teacher_entry_turn"] = 0
    metadata_tags["has_first_half_usps"] = bool(
        metadata_tags.get("has_first_half_usps", False)
    )

    updated_at_raw = base.get("updated_at")
    if isinstance(updated_at_raw, datetime):
        updated_at = updated_at_raw.astimezone(timezone.utc).isoformat()
    elif isinstance(updated_at_raw, str) and updated_at_raw.strip():
        updated_at = updated_at_raw.strip()
    else:
        updated_at = datetime.now(timezone.utc).isoformat()

    normalized: dict[str, Any] = {
        "ad_code": ad_code,
        "title": str(base.get("title") or ""),
        "doc_url": str(base.get("doc_url") or ""),
        "language": _normalize_language(base.get("language")),
        "character_type": _normalize_character_type(base.get("character_type")),
        "cac_tier": _normalize_cac_tier(base.get("cac_tier")),
        "cac_metrics": cac_metrics,
        "english_script": str(base.get("english_script") or ""),
        "native_script": str(base.get("native_script") or ""),
        "story_blocks": story_blocks,
        "visual_blueprint": visual_blueprint,
        "metadata_tags": metadata_tags,
        "hook_vector": _coerce_vector(base.get("hook_vector")),
        "script_vector": _coerce_vector(base.get("script_vector")),
        "updated_at": updated_at,
    }

    # Preserve any extra top-level metadata keys callers may attach
    for extra_key, extra_val in base.items():
        if extra_key not in normalized:
            normalized[extra_key] = extra_val

    return normalized


# ---------------------------------------------------------------------------
# 3. BrainDBStore Implementation
# ---------------------------------------------------------------------------


def _is_placeholder_database_url(url: str | None) -> bool:
    """Return True if `DATABASE_URL` is unset or uses a template placeholder."""
    if not url or not url.strip():
        return True
    cleaned = url.strip()
    placeholders = (
        "roundhouse.proxy.rlwy.net:12345",
        "postgres:password@",
        "your_database_url",
        "<password>",
    )
    if any(token in cleaned for token in placeholders):
        return True
    parsed = urlparse(cleaned)
    return parsed.scheme not in ("postgres", "postgresql") or not parsed.hostname


class BrainDBStore:
    """
    Dual-mode persistence store for `supernova_script_brain`.

    - Connects to PostgreSQL (`DATABASE_URL` from `.env`) when configured and
      reachable.
    - Automatically maintains and falls back to
      `ad-studio/script-brain/data/brain_store.json` so fast local reads and
      offline development always work seamlessly.
    """

    TABLE_NAME = TABLE_NAME

    def __init__(
        self,
        database_url: str | None = None,
        json_path: str | Path | None = None,
        connect_timeout: int = 3,
    ) -> None:
        _load_project_env()
        self.database_url: str = (
            database_url
            if database_url is not None
            else os.environ.get("DATABASE_URL", "")
        ).strip()
        self.json_path: Path = (
            Path(json_path).resolve()
            if json_path is not None
            else DEFAULT_JSON_STORE_PATH
        )
        self.connect_timeout = connect_timeout
        self.pg_active: bool = False
        self._pg_driver: str | None = None
        self._schema_initialized: bool = False

        # Always make sure the local JSON store directory & file exist
        self._ensure_local_store_file()

    # ------------------------------------------------------------------
    # Internal PostgreSQL Connection Helpers
    # ------------------------------------------------------------------

    def _open_pg_connection(self) -> Any | None:
        """
        Attempt to open a PostgreSQL connection using psycopg2 or psycopg (v3).
        Returns None and sets `self.pg_active = False` if unavailable/unreachable.
        """
        if _is_placeholder_database_url(self.database_url):
            self.pg_active = False
            return None

        # Try psycopg2 first
        try:
            import psycopg2  # type: ignore[import-not-found]

            conn = psycopg2.connect(
                self.database_url,
                connect_timeout=self.connect_timeout,
            )
            self._pg_driver = "psycopg2"
            self.pg_active = True
            return conn
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("PostgreSQL (psycopg2) connection unavailable: %s", exc)
            self.pg_active = False
            return None

        # Try psycopg v3
        try:
            import psycopg  # type: ignore[import-not-found]

            conn = psycopg.connect(
                self.database_url,
                connect_timeout=self.connect_timeout,
            )
            self._pg_driver = "psycopg"
            self.pg_active = True
            return conn
        except ImportError:
            logger.debug(
                "Neither psycopg2 nor psycopg is installed; using local JSON store."
            )
            self.pg_active = False
            return None
        except Exception as exc:
            logger.debug("PostgreSQL (psycopg) connection unavailable: %s", exc)
            self.pg_active = False
            return None

    def _row_to_record(
        self, columns: list[str], row: tuple[Any, ...] | dict[str, Any]
    ) -> dict[str, Any]:
        raw = dict(row) if isinstance(row, dict) else dict(zip(columns, row))
        return normalize_script_record(raw)

    def _record_to_sql_params(self, record: dict[str, Any]) -> tuple[Any, ...]:
        return (
            record["ad_code"],
            record["title"],
            record["doc_url"],
            record["language"],
            record["character_type"],
            record["cac_tier"],
            json.dumps(record["cac_metrics"], ensure_ascii=False),
            record["english_script"],
            record["native_script"],
            json.dumps(record["story_blocks"], ensure_ascii=False),
            json.dumps(record["visual_blueprint"], ensure_ascii=False),
            json.dumps(record["metadata_tags"], ensure_ascii=False),
            [float(x) for x in record.get("hook_vector", [])],
            [float(x) for x in record.get("script_vector", [])],
            record["updated_at"],
        )

    # ------------------------------------------------------------------
    # Internal Local JSON Mirror Helpers
    # ------------------------------------------------------------------

    def _ensure_local_store_file(self) -> None:
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.json_path.exists():
            self._write_local_records({})

    def _read_local_records(self) -> dict[str, dict[str, Any]]:
        """Read all script records from `data/brain_store.json`, keyed by `ad_code`."""
        self._ensure_local_store_file()
        try:
            raw_text = self.json_path.read_text(encoding="utf-8").strip()
            if not raw_text:
                return {}
            payload = json.loads(raw_text)
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Failed to parse %s (%s); resetting mirror.", self.json_path, exc)
            return {}

        records_by_code: dict[str, dict[str, Any]] = {}
        raw_items: list[Any] = []

        if isinstance(payload, list):
            raw_items = payload
        elif isinstance(payload, dict):
            if "scripts" in payload and isinstance(payload["scripts"], dict):
                for code, item in payload["scripts"].items():
                    if isinstance(item, dict):
                        item_copy = dict(item)
                        item_copy.setdefault("ad_code", code)
                        raw_items.append(item_copy)
            elif "scripts" in payload and isinstance(payload["scripts"], list):
                raw_items = payload["scripts"]
            else:
                for code, item in payload.items():
                    if isinstance(item, dict) and ("ad_code" in item or "title" in item or "english_script" in item):
                        item_copy = dict(item)
                        item_copy.setdefault("ad_code", code)
                        raw_items.append(item_copy)

        for item in raw_items:
            if not isinstance(item, dict):
                continue
            try:
                norm = normalize_script_record(item)
                records_by_code[norm["ad_code"]] = norm
            except ValueError:
                continue

        return records_by_code

    def _write_local_records(self, records_by_code: dict[str, dict[str, Any]]) -> None:
        """Atomically persist all script records to `data/brain_store.json`."""
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        ordered_codes = sorted(
            records_by_code.keys(),
            key=lambda code: (
                TIER_SORT_ORDER.get(records_by_code[code].get("cac_tier", "TIER_B"), 99),
                code,
            ),
        )
        ordered_scripts = {code: records_by_code[code] for code in ordered_codes}
        payload = {
            "table": self.TABLE_NAME,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_count": len(ordered_scripts),
            "scripts": ordered_scripts,
        }
        serialized = json.dumps(payload, indent=2, ensure_ascii=False)

        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.json_path.parent),
            prefix=".brain_store_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                tmp_file.write(serialized)
                tmp_file.write("\n")
            os.replace(tmp_path, self.json_path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init_schema(self) -> bool:
        """
        Create `supernova_script_brain` table in PostgreSQL if `DATABASE_URL` is
        set and reachable, and sync `ad-studio/script-brain/data/brain_store.json`
        as a local mirror so fast local reads always work without network latency.
        """
        self._ensure_local_store_file()
        local_records = self._read_local_records()

        conn = self._open_pg_connection()
        if conn is None:
            self._schema_initialized = True
            return True

        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(CREATE_TABLE_SQL)
                    cur.execute(
                        f"""
                        SELECT
                            ad_code, title, doc_url, language, character_type,
                            cac_tier, cac_metrics, english_script, native_script,
                            story_blocks, visual_blueprint, metadata_tags,
                            hook_vector, script_vector, updated_at
                        FROM {self.TABLE_NAME}
                        ORDER BY ad_code ASC;
                        """
                    )
                    columns = [desc[0] for desc in cur.description] if cur.description else []
                    rows = cur.fetchall()

                    pg_records: dict[str, dict[str, Any]] = {}
                    for row in rows:
                        rec = self._row_to_record(columns, row)
                        pg_records[rec["ad_code"]] = rec

                    # Bidirectional sync between PostgreSQL and local mirror
                    merged_records = dict(local_records)
                    for code, pg_rec in pg_records.items():
                        if code not in merged_records:
                            merged_records[code] = pg_rec
                        else:
                            local_ts = str(merged_records[code].get("updated_at") or "")
                            pg_ts = str(pg_rec.get("updated_at") or "")
                            if pg_ts >= local_ts:
                                merged_records[code] = pg_rec

                    # Push any local-only or newer local records into PostgreSQL
                    for code, local_rec in merged_records.items():
                        if code not in pg_records or merged_records[code] is local_records.get(code):
                            cur.execute(
                                UPSERT_SCRIPT_SQL,
                                self._record_to_sql_params(local_rec),
                            )

            self._write_local_records(merged_records)
            self._schema_initialized = True
            return True
        except Exception as exc:
            logger.warning(
                "PostgreSQL schema init/sync failed (%s); falling back to local JSON.",
                exc,
            )
            self.pg_active = False
            self._write_local_records(local_records)
            self._schema_initialized = True
            return True
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def upsert_script(self, record: dict[str, Any] | ProvenScriptRecord) -> bool:
        """
        Upsert a single proven script record into PostgreSQL (if connected)
        and update `data/brain_store.json`.
        """
        if not self._schema_initialized:
            self.init_schema()

        local_records = self._read_local_records()
        raw_code = (
            record.ad_code
            if isinstance(record, ProvenScriptRecord)
            else str((record or {}).get("ad_code") or "")
        ).strip().upper()
        if not raw_code:
            return False

        existing = local_records.get(raw_code)
        incoming_dict = (
            dataclasses.asdict(record)
            if dataclasses.is_dataclass(record) and not isinstance(record, type)
            else dict(record)
        )
        if "updated_at" not in incoming_dict or not incoming_dict["updated_at"]:
            incoming_dict["updated_at"] = datetime.now(timezone.utc).isoformat()

        normalized = normalize_script_record(incoming_dict, existing=existing)
        local_records[normalized["ad_code"]] = normalized
        self._write_local_records(local_records)

        conn = self._open_pg_connection()
        if conn is not None:
            try:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            UPSERT_SCRIPT_SQL,
                            self._record_to_sql_params(normalized),
                        )
            except Exception as exc:
                logger.warning(
                    "PostgreSQL upsert failed for %s (%s); saved to local JSON mirror.",
                    normalized["ad_code"],
                    exc,
                )
                self.pg_active = False
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        return True

    def bulk_upsert_scripts(
        self, records: list[dict[str, Any] | ProvenScriptRecord]
    ) -> int:
        """
        Bulk upsert multiple proven script records into PostgreSQL (if connected)
        and update `data/brain_store.json` in a single atomic write.

        Returns the count of upserted script records.
        """
        if not self._schema_initialized:
            self.init_schema()

        if not records:
            return 0

        local_records = self._read_local_records()
        normalized_batch: list[dict[str, Any]] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for item in records:
            try:
                raw_dict = (
                    dataclasses.asdict(item)
                    if dataclasses.is_dataclass(item) and not isinstance(item, type)
                    else dict(item)
                )
                ad_code = str(raw_dict.get("ad_code") or "").strip().upper()
                if not ad_code:
                    continue
                if "updated_at" not in raw_dict or not raw_dict["updated_at"]:
                    raw_dict["updated_at"] = now_iso
                norm = normalize_script_record(
                    raw_dict,
                    existing=local_records.get(ad_code),
                )
                local_records[norm["ad_code"]] = norm
                normalized_batch.append(norm)
            except (ValueError, TypeError) as exc:
                logger.warning("Skipping invalid script record in bulk upsert: %s", exc)

        if not normalized_batch:
            return 0

        self._write_local_records(local_records)

        conn = self._open_pg_connection()
        if conn is not None:
            try:
                with conn:
                    with conn.cursor() as cur:
                        for norm in normalized_batch:
                            cur.execute(
                                UPSERT_SCRIPT_SQL,
                                self._record_to_sql_params(norm),
                            )
            except Exception as exc:
                logger.warning(
                    "PostgreSQL bulk upsert failed (%s); %d records saved to local JSON mirror.",
                    exc,
                    len(normalized_batch),
                )
                self.pg_active = False
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        return len(normalized_batch)

    def get_all_scripts(
        self,
        language: str | None = None,
        cac_tiers: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return all stored scripts filtered optionally by `language` and `cac_tiers`.
        Reads from the synchronized local mirror for fast zero-latency access.
        """
        if not self._schema_initialized:
            self.init_schema()

        records_by_code = self._read_local_records()
        scripts = list(records_by_code.values())

        if language and language.strip() and language.strip().upper() != "ALL":
            target_lang = language.strip().lower()
            scripts = [
                s
                for s in scripts
                if str(s.get("language") or "").strip().lower() == target_lang
            ]

        if cac_tiers:
            if isinstance(cac_tiers, str):
                cac_tiers = [cac_tiers]
            normalized_tiers = {_normalize_cac_tier(t) for t in cac_tiers if t}
            if normalized_tiers:
                scripts = [
                    s
                    for s in scripts
                    if _normalize_cac_tier(s.get("cac_tier")) in normalized_tiers
                ]

        scripts.sort(
            key=lambda s: (
                TIER_SORT_ORDER.get(str(s.get("cac_tier") or "TIER_B"), 99),
                (
                    float(s.get("cac_metrics", {}).get("cac_vs_avg_pct"))
                    if isinstance(s.get("cac_metrics"), dict)
                    and s["cac_metrics"].get("cac_vs_avg_pct") is not None
                    else 9999.0
                ),
                str(s.get("ad_code") or ""),
            )
        )
        return scripts

    def get_script_by_code(self, ad_code: str) -> dict[str, Any] | None:
        """Retrieve a single proven script record by `ad_code` (e.g., `'SN3195'`)."""
        if not ad_code or not str(ad_code).strip():
            return None

        if not self._schema_initialized:
            self.init_schema()

        target_code = str(ad_code).strip().upper()
        records_by_code = self._read_local_records()
        if target_code in records_by_code:
            return records_by_code[target_code]

        for code, record in records_by_code.items():
            if code.upper() == target_code:
                return record

        return None

    def get_stats(self) -> dict[str, Any]:
        """
        Return total script count, count by `cac_tier`, count by `language`,
        and whether PostgreSQL is currently active.
        """
        if not self._schema_initialized:
            self.init_schema()

        scripts = list(self._read_local_records().values())
        total_count = len(scripts)

        by_cac_tier: dict[str, int] = {tier: 0 for tier in VALID_CAC_TIERS}
        for s in scripts:
            tier = _normalize_cac_tier(s.get("cac_tier"))
            by_cac_tier[tier] = by_cac_tier.get(tier, 0) + 1

        by_language: dict[str, int] = {}
        for s in scripts:
            lang = _normalize_language(s.get("language"))
            by_language[lang] = by_language.get(lang, 0) + 1

        by_character_type: dict[str, int] = {}
        for s in scripts:
            char_type = _normalize_character_type(s.get("character_type"))
            by_character_type[char_type] = by_character_type.get(char_type, 0) + 1

        return {
            "total_count": total_count,
            "total": total_count,
            "by_cac_tier": by_cac_tier,
            "count_by_cac_tier": by_cac_tier,
            "by_language": by_language,
            "count_by_language": by_language,
            "by_character_type": by_character_type,
            "postgres_active": self.pg_active,
            "pg_active": self.pg_active,
            "is_postgres_active": self.pg_active,
            "storage_backend": (
                "postgresql+json_mirror" if self.pg_active else "local_json_fallback"
            ),
            "json_store_path": str(self.json_path),
            "table_name": self.TABLE_NAME,
        }


__all__ = [
    "BrainDBStore",
    "ProvenScriptRecord",
    "CACMetricsDict",
    "StoryBlocksDict",
    "VisualBlueprintDict",
    "MetadataTagsDict",
    "ScriptRecordDict",
    "VALID_LANGUAGES",
    "VALID_CHARACTER_TYPES",
    "VALID_CAC_TIERS",
    "CREATE_TABLE_SQL",
    "UPSERT_SCRIPT_SQL",
    "normalize_script_record",
]
