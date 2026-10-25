#!/usr/bin/env python3
"""
Supernova Script Brain — Main Orchestrator & CLI Entrypoint (`run_brain.py`)

Connects:
- `db_store.py` (`BrainDBStore`)
- `vector_memory.py` (`VectorMemoryEngine`)
- `script_parser.py` (`parse_proven_doc_text`)
- `ingest_brain.py` (`ingest_proven_scripts`, `seed_demo_winners_if_empty`)
- `contrastive_trainer.py` (`build_contrastive_playbook`, `load_contrastive_playbook`)
- `dspy_brain.py` (`compile_dspy_brain`, `generate_with_middle_brain`)
- `success_predictor.py` (`train_success_model`, `predict_success_rate`)
- `compliance_validator.py` (`validate_supernova_script`, `format_complete_google_doc_text`)

CLI Subcommands (all emit clean JSON to stdout):
  python run_brain.py status [--seed-if-empty]
  python run_brain.py ingest --file <json_path>
  python run_brain.py ingest --from-sheet [--sheet-id <id>] [--tab "REPLICATE - LANGUAGE"] [--limit 5] [--reset]
  python run_brain.py train [--use-llm-textgrad]
  python run_brain.py generate --brief "<text>" --language "<lang>" --character "<char>"
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata
import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

# Ensure local directory is on sys.path regardless of CWD
SCRIPT_BRAIN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_BRAIN_DIR.parent.parent
DATA_DIR = SCRIPT_BRAIN_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SHEET_ID = "13sQlJMgAZ_nFxatPEp5XfZ9swE4et00qEBpwNS2a82I"
DEFAULT_SHEET_TAB = "REPLICATE - LANGUAGE"

if str(SCRIPT_BRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_BRAIN_DIR))

# Load environment variables from root .env
try:
    from dotenv import load_dotenv

    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Flexible Function Invocation Helper
# ---------------------------------------------------------------------------


def _invoke_flexible(fn: Callable[..., Any], *pos_args: Any, **candidate_kwargs: Any) -> Any:
    """
    Calls `fn` by matching `candidate_kwargs` against its signature if needed,
    preserving compatibility across slight parameter naming differences.
    """
    try:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)

        if has_var_kw:
            return fn(*pos_args, **candidate_kwargs)

        accepted_names = {
            p.name
            for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        # Remove names already satisfied by positional args
        pos_param_names = [
            p.name
            for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        for idx in range(min(len(pos_args), len(pos_param_names))):
            accepted_names.discard(pos_param_names[idx])

        filtered_kwargs = {k: v for k, v in candidate_kwargs.items() if k in accepted_names}
        return fn(*pos_args, **filtered_kwargs)
    except TypeError:
        return fn(*pos_args)


# ---------------------------------------------------------------------------
# Package Version Check
# ---------------------------------------------------------------------------

REQUIRED_PACKAGES: list[tuple[str, str]] = [
    ("dspy", "dspy"),
    ("chromadb", "chromadb"),
    ("sentence_transformers", "sentence-transformers"),
    ("textgrad", "textgrad"),
    ("xgboost", "xgboost"),
    ("sklearn", "scikit-learn"),
    ("deep_translator", "deep-translator"),
]


def check_installed_packages() -> dict[str, dict[str, Any]]:
    """Checks installed versions of all core Middle Brain Python libraries."""
    report: dict[str, dict[str, Any]] = {}
    for import_name, dist_name in REQUIRED_PACKAGES:
        version: str | None = None
        installed = False
        err_msg: str | None = None

        # First try importlib.metadata (fast, avoids heavy module import side-effects)
        for candidate_dist in (dist_name, import_name, f"{dist_name}-ai"):
            try:
                version = importlib.metadata.version(candidate_dist)
                installed = True
                break
            except importlib.metadata.PackageNotFoundError:
                continue

        # Verify actual importability
        try:
            with contextlib.redirect_stdout(sys.stderr):
                mod = importlib.import_module(import_name)
            installed = True
            if not version:
                version = getattr(mod, "__version__", "installed")
        except Exception as exc:
            if not installed:
                err_msg = str(exc)

        entry: dict[str, Any] = {
            "installed": installed,
            "version": version,
        }
        if err_msg:
            entry["error"] = err_msg
        report[import_name] = entry

    return report


# ---------------------------------------------------------------------------
# Store Reset & Script Summary Helpers
# ---------------------------------------------------------------------------


def _reset_brain_db_store(db: Any) -> int:
    """
    Wipes all existing/demo script records from `BrainDBStore` (both PostgreSQL
    if connected and local `data/brain_store.json`) and clears vector memory.
    Returns the number of records removed.
    """
    existing = db.get_all_scripts() if hasattr(db, "get_all_scripts") else []
    removed_count = len(existing)

    for method_name in ("clear_all_scripts", "reset_store", "delete_all_scripts", "clear"):
        fn = getattr(db, method_name, None)
        if callable(fn):
            try:
                fn()
                break
            except Exception:
                pass

    # Ensure PostgreSQL table is wiped if connected so init_schema() does not re-merge old rows
    if hasattr(db, "_open_pg_connection"):
        conn = db._open_pg_connection()
        if conn is not None:
            try:
                with conn:
                    with conn.cursor() as cur:
                        table_name = getattr(db, "TABLE_NAME", "supernova_script_brain")
                        cur.execute(f"DELETE FROM {table_name};")
            except Exception:
                pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

    # Ensure local JSON mirror is empty
    if hasattr(db, "_write_local_records"):
        try:
            db._write_local_records({})
        except Exception:
            pass

    return removed_count


def _summarize_script_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Builds a clean summary list of script records containing `ad_code`, `title`,
    `language`, `cac_tier`, `cac_metrics`, and `metadata_tags`.
    """
    summary: list[dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        summary.append(
            {
                "ad_code": rec.get("ad_code", ""),
                "title": rec.get("title", ""),
                "doc_url": rec.get("doc_url", ""),
                "language": rec.get("language", ""),
                "character_type": rec.get("character_type", ""),
                "cac_tier": rec.get("cac_tier", ""),
                "cac_metrics": rec.get("cac_metrics") or {},
                "metadata_tags": rec.get("metadata_tags") or {},
            }
        )
    return summary


# ---------------------------------------------------------------------------
# Subcommand 1: status
# ---------------------------------------------------------------------------


def run_status(seed_if_empty: bool = False) -> dict[str, Any]:
    """
    Checks installed package versions, initializes DB schema, seeds demo winners
    ONLY if the DB has 0 records AND `seed_if_empty=True` was explicitly passed,
    and returns a status dictionary.
    """
    with contextlib.redirect_stdout(sys.stderr):
        packages = check_installed_packages()

        from db_store import BrainDBStore

        db = BrainDBStore()
        if hasattr(db, "init_schema"):
            db.init_schema()

        existing_records = db.get_all_scripts() if hasattr(db, "get_all_scripts") else []
        total_records = len(existing_records)

        # Only seed demo winners if the database has 0 records AND --seed-if-empty was explicitly passed
        seeded_result: Any = None
        if bool(seed_if_empty) and total_records == 0:
            from ingest_brain import seed_demo_winners_if_empty

            seeded_result = _invoke_flexible(
                seed_demo_winners_if_empty,
                db_store=db,
                store=db,
                force=False,
            )

        after_records = db.get_all_scripts() if hasattr(db, "get_all_scripts") else []
        stats_after = db.get_stats() if hasattr(db, "get_stats") else {}
        total_after = len(after_records)

        # If demo winners were just seeded (or models are missing while records exist),
        # ensure the zero-API-call artifacts exist on disk without seeding demo records.
        playbook_path = DATA_DIR / "contrastive_playbook.json"
        dspy_path = DATA_DIR / "compiled_dspy_brain.json"
        xgb_path = DATA_DIR / "xgb_success_model.pkl"

        auto_trained = False
        if total_after > 0 and (
            seeded_result is not None
            or not (playbook_path.exists() and dspy_path.exists() and xgb_path.exists())
        ):
            try:
                run_train(use_llm_textgrad=False, seed_if_empty=False)
                auto_trained = True
            except Exception as train_exc:
                sys.stderr.write(f"[run_brain] Auto-train warning during status: {train_exc}\n")

        artifacts = {
            "brain_store_json": {
                "path": str(DATA_DIR / "brain_store.json"),
                "exists": (DATA_DIR / "brain_store.json").exists(),
            },
            "contrastive_playbook_json": {
                "path": str(playbook_path),
                "exists": playbook_path.exists(),
            },
            "compiled_dspy_brain_json": {
                "path": str(dspy_path),
                "exists": dspy_path.exists(),
            },
            "xgb_success_model_pkl": {
                "path": str(xgb_path),
                "exists": xgb_path.exists(),
            },
        }

        return {
            "status": "ok",
            "command": "status",
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
            "gemini_configured": bool(os.environ.get("GEMINI_API_KEY")),
            "packages": packages,
            "db_stats": stats_after,
            "total_records": total_after,
            "scripts": _summarize_script_records(after_records),
            "seeded": bool(seeded_result and seeded_result.get("seeded", True)),
            "seed_details": seeded_result,
            "auto_trained": auto_trained,
            "artifacts": artifacts,
        }


def handle_status(args: argparse.Namespace | None = None) -> dict[str, Any]:
    """CLI handler for `status` subcommand."""
    seed_flag = bool(getattr(args, "seed_if_empty", False)) if args is not None else False
    return run_status(seed_if_empty=seed_flag)


# ---------------------------------------------------------------------------
# Subcommand 2: train
# ---------------------------------------------------------------------------


def run_train(
    use_llm_textgrad: bool = False,
    seed_if_empty: bool = False,
) -> dict[str, Any]:
    """
    Loads all records from `BrainDBStore`, indexes vectors in `VectorMemoryEngine`,
    builds `contrastive_playbook.json`, compiles `compiled_dspy_brain.json`,
    and trains `xgb_success_model.pkl` (0 Gemini API calls by default).
    """
    t0 = time.time()
    with contextlib.redirect_stdout(sys.stderr):
        from contrastive_trainer import build_contrastive_playbook
        from db_store import BrainDBStore
        from dspy_brain import compile_dspy_brain
        from ingest_brain import seed_demo_winners_if_empty
        from success_predictor import train_success_model
        from vector_memory import VectorMemoryEngine

        db = BrainDBStore()
        if hasattr(db, "init_schema"):
            db.init_schema()

        records = db.get_all_scripts()
        seeded = False
        if not records and seed_if_empty:
            _invoke_flexible(seed_demo_winners_if_empty, db_store=db, store=db, force=False)
            records = db.get_all_scripts()
            seeded = bool(records)

        vector_engine = VectorMemoryEngine()

        # Ensure every record has hook_vector and script_vector populated locally
        updated_vectors = 0
        for rec in records:
            hook_vec = rec.get("hook_vector")
            script_vec = rec.get("script_vector")
            if not hook_vec or not script_vec:
                h_vec, s_vec = vector_engine.embed_script_record(rec)
                rec["hook_vector"] = h_vec
                rec["script_vector"] = s_vec
                db.upsert_script(rec)
                updated_vectors += 1

        # 1. Index in VectorMemoryEngine
        indexed_count = vector_engine.index_scripts(records) if records else 0

        # 2. Build Contrastive Playbook (Winner vs Loser DNA)
        playbook = _invoke_flexible(
            build_contrastive_playbook,
            records,
            use_llm_textgrad=use_llm_textgrad,
        )

        # 3. Compile DSPy Brain (BootstrapFewShot / Winning Demonstrations)
        compiled_dspy_result = _invoke_flexible(
            compile_dspy_brain,
            records,
            playbook=playbook,
            contrastive_playbook=playbook,
        )

        # 4. Train XGBoost / Scikit-Learn Success Rate Model
        xgb_training_metrics = _invoke_flexible(
            train_success_model,
            records,
        )

        elapsed_sec = round(time.time() - t0, 3)

        return {
            "status": "ok",
            "command": "train",
            "elapsed_seconds": elapsed_sec,
            "gemini_api_calls": 0 if not use_llm_textgrad else 1,
            "total_records": len(records),
            "seeded_demo_if_empty": seeded,
            "backfilled_vectors": updated_vectors,
            "indexed_vectors": indexed_count,
            "contrastive_playbook": {
                "path": str(DATA_DIR / "contrastive_playbook.json"),
                "summary": (
                    playbook.get("summary")
                    or playbook.get("rules")
                    or playbook.get("statistical_rules")
                    if isinstance(playbook, dict)
                    else str(playbook)
                ),
            },
            "dspy_compilation": (
                compiled_dspy_result
                if isinstance(compiled_dspy_result, dict)
                else {"compiled_path": str(compiled_dspy_result)}
            ),
            "success_model": xgb_training_metrics,
        }


def handle_train(args: argparse.Namespace | None = None) -> dict[str, Any]:
    """CLI handler for `train` subcommand."""
    use_llm = bool(getattr(args, "use_llm_textgrad", False)) if args is not None else False
    return run_train(use_llm_textgrad=use_llm, seed_if_empty=False)


# ---------------------------------------------------------------------------
# Subcommand 3: ingest
# ---------------------------------------------------------------------------


def run_ingest(
    file_path: str | None = None,
    auto_train: bool = True,
    sheet_id: str | None = None,
    from_sheet: bool = False,
    tab: str = DEFAULT_SHEET_TAB,
    limit: int | None = None,
    reset: bool = False,
) -> dict[str, Any]:
    """
    Ingests proven scripts + CAC metrics either from a Google Sheet (`--from-sheet` / `--sheet-id`)
    or from a local JSON/CSV file (`--file`), optionally wiping old/demo records first (`--reset`),
    and automatically runs `handle_train` (`build_contrastive_playbook`, `compile_dspy_brain`,
    `train_success_model`) on the newly ingested scripts.
    """
    t0 = time.time()
    with contextlib.redirect_stdout(sys.stderr):
        from db_store import BrainDBStore
        import ingest_brain

        db = BrainDBStore()
        if hasattr(db, "init_schema"):
            db.init_schema()

        use_sheet = bool(sheet_id or from_sheet or not file_path)

        if use_sheet:
            resolved_sheet_id = (sheet_id or DEFAULT_SHEET_ID).strip()
            resolved_tab = (tab or DEFAULT_SHEET_TAB).strip()

            removed_count = 0
            if reset:
                removed_count = _reset_brain_db_store(db)

            if hasattr(ingest_brain, "ingest_from_google_sheet"):
                ingest_result = _invoke_flexible(
                    ingest_brain.ingest_from_google_sheet,
                    spreadsheet_id=resolved_sheet_id,
                    sheet_id=resolved_sheet_id,
                    sheet_name=resolved_tab,
                    tab=resolved_tab,
                    limit=limit,
                    reset_existing=reset,
                    reset=reset,
                    db_store=db,
                )
            else:
                items = ingest_brain.fetch_items_from_google_sheet(
                    spreadsheet_id=resolved_sheet_id,
                    sheet_name=resolved_tab,
                    limit=limit,
                )
                ingest_result = ingest_brain.ingest_proven_scripts(items, db_store=db)

            # Refresh DB store view after ingestion
            db_after = BrainDBStore()
            if hasattr(db_after, "init_schema"):
                db_after.init_schema()
            all_stored_scripts = db_after.get_all_scripts()

            # Build the detailed list of ingested scripts
            ingested_codes: list[str] = []
            if isinstance(ingest_result, dict):
                raw_codes = ingest_result.get("ad_codes") or [
                    s.get("ad_code")
                    for s in (
                        ingest_result.get("ingested_scripts")
                        or ingest_result.get("scripts")
                        or []
                    )
                    if isinstance(s, dict) and s.get("ad_code")
                ]
                ingested_codes = [str(c).strip().upper() for c in raw_codes if c]

            if ingested_codes:
                by_code = {str(r.get("ad_code", "")).strip().upper(): r for r in all_stored_scripts}
                matched_records = [by_code[c] for c in ingested_codes if c in by_code]
                if not matched_records:
                    matched_records = all_stored_scripts
            else:
                matched_records = all_stored_scripts

            ingested_scripts_summary = _summarize_script_records(matched_records)

            if isinstance(ingest_result, dict) and "ingested_scripts" not in ingest_result:
                ingest_result["ingested_scripts"] = ingested_scripts_summary

            train_result: dict[str, Any] | None = None
            if auto_train:
                train_result = handle_train()

            elapsed_sec = round(time.time() - t0, 3)

            return {
                "status": "ok",
                "command": "ingest",
                "source": "google_sheet",
                "sheet_id": resolved_sheet_id,
                "spreadsheet_id": resolved_sheet_id,
                "tab": resolved_tab,
                "sheet_name": resolved_tab,
                "limit": limit,
                "reset": bool(reset),
                "removed_before_ingest": removed_count,
                "ingested_count": len(ingested_scripts_summary),
                "ingested_scripts": ingested_scripts_summary,
                "scripts": ingested_scripts_summary,
                "gemini_api_calls": 0,
                "elapsed_seconds": elapsed_sec,
                "ingest_result": ingest_result,
                "training": train_result,
            }

        # File-based ingestion (--file <json_path>)
        resolved_path = Path(file_path).expanduser().resolve()  # type: ignore[arg-type]
        if not resolved_path.exists():
            raise FileNotFoundError(f"Ingest JSON file not found: {resolved_path}")

        removed_count = 0
        if reset:
            removed_count = _reset_brain_db_store(db)

        raw_content = resolved_path.read_text(encoding="utf-8")
        payload = json.loads(raw_content)

        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            items = (
                payload.get("items")
                or payload.get("scripts")
                or payload.get("records")
                or [payload]
            )
        else:
            raise ValueError("Ingest JSON must contain a list of script objects or a dict with 'items'.")

        if limit is not None and limit > 0:
            items = items[:limit]

        ingest_result = _invoke_flexible(
            ingest_brain.ingest_proven_scripts,
            items,
            db_store=db,
        )

        db_after = BrainDBStore()
        if hasattr(db_after, "init_schema"):
            db_after.init_schema()
        all_stored_scripts = db_after.get_all_scripts()

        ingested_codes = [
            str(c).strip().upper()
            for c in (ingest_result.get("ad_codes") or [])
            if c
        ] if isinstance(ingest_result, dict) else []

        if ingested_codes:
            by_code = {str(r.get("ad_code", "")).strip().upper(): r for r in all_stored_scripts}
            matched_records = [by_code[c] for c in ingested_codes if c in by_code]
            if not matched_records:
                matched_records = all_stored_scripts
        else:
            matched_records = all_stored_scripts

        ingested_scripts_summary = _summarize_script_records(matched_records)

        train_result = None
        if auto_train:
            train_result = handle_train()

        elapsed_sec = round(time.time() - t0, 3)

        return {
            "status": "ok",
            "command": "ingest",
            "source": "file",
            "file": str(resolved_path),
            "reset": bool(reset),
            "removed_before_ingest": removed_count,
            "items_submitted": len(items),
            "ingested_count": len(ingested_scripts_summary),
            "ingested_scripts": ingested_scripts_summary,
            "scripts": ingested_scripts_summary,
            "gemini_api_calls": 0,
            "elapsed_seconds": elapsed_sec,
            "ingest_result": ingest_result,
            "training": train_result,
        }


def handle_ingest(args: argparse.Namespace) -> dict[str, Any]:
    """
    CLI handler for `ingest` subcommand.
    Supports both `--file <json_path>` and Google Sheet flags (`--from-sheet`,
    `--sheet-id`, `--tab`, `--limit`, `--reset`).
    """
    return run_ingest(
        file_path=getattr(args, "file", None),
        auto_train=not bool(getattr(args, "no_train", False)),
        sheet_id=getattr(args, "sheet_id", None),
        from_sheet=bool(getattr(args, "from_sheet", False)),
        tab=getattr(args, "tab", DEFAULT_SHEET_TAB) or DEFAULT_SHEET_TAB,
        limit=getattr(args, "limit", None),
        reset=bool(getattr(args, "reset", False)),
    )


# ---------------------------------------------------------------------------
# Subcommand 4: generate
# ---------------------------------------------------------------------------


def _build_historical_benchmark_summary(top_winners: list[dict[str, Any]]) -> str:
    """Formats a concise 1-line summary of the matched CAC winners for the Google Doc header."""
    if not top_winners:
        return "Matched against Supernova historical low-CAC benchmark scripts."
    parts: list[str] = []
    for w in top_winners[:3]:
        rec = w.get("record", w) if isinstance(w, dict) else {}
        code = w.get("ad_code") or rec.get("ad_code") or "SN-WIN"
        tier = w.get("cac_tier") or rec.get("cac_tier") or "TIER_S"
        lang = w.get("language") or rec.get("language") or ""
        cac_metrics = w.get("cac_metrics") or rec.get("cac_metrics") or {}
        vs_avg = cac_metrics.get("cac_vs_avg_pct")
        sim = w.get("cosine_similarity") or w.get("similarity")
        detail = f"{code} ({tier}"
        if lang:
            detail += f", {lang}"
        if isinstance(vs_avg, (int, float)):
            detail += f", {vs_avg:+.1f}% vs avg CAC"
        if isinstance(sim, (int, float)):
            detail += f", {int(round(sim * 100))}% match"
        detail += ")"
        parts.append(detail)
    return "Top Matched CAC Winners: " + " | ".join(parts)


def run_generate(
    brief: str,
    language: str = "Tamil",
    character: str = "Female Humanoid Robot",
) -> dict[str, Any]:
    """
    Full Middle Brain generation pipeline:
    1. Searches `VectorMemoryEngine` for Top 3 CAC Winners + Losers to avoid.
    2. Extracts the winners' `visual_blueprint` and `story_blocks` from SQL/store.
    3. Runs `generate_with_middle_brain` (`dspy_brain.py`) + `validate_supernova_script` (`compliance_validator.py`).
    4. Parses the new script with `parse_proven_doc_text` and computes Approximate Success Rate % via `predict_success_rate`.
    5. Formats the complete Google Doc output via `format_complete_google_doc_text` and returns the full JSON response.
    """
    t0 = time.time()
    with contextlib.redirect_stdout(sys.stderr):
        from compliance_validator import (
            format_complete_google_doc_text,
            validate_supernova_script,
        )
        from contrastive_trainer import load_contrastive_playbook
        from db_store import BrainDBStore
        from dspy_brain import generate_with_middle_brain
        from ingest_brain import seed_demo_winners_if_empty
        from script_parser import parse_proven_doc_text
        from success_predictor import predict_success_rate
        from vector_memory import VectorMemoryEngine

        # Normalize inputs
        brief = (brief or "").strip()
        if not brief:
            raise ValueError("A non-empty --brief is required for script generation.")
        language = (language or "Tamil").strip()
        character = (character or "Female Humanoid Robot").strip()

        # 0. Initialize DB & ensure records + trained artifacts exist
        db = BrainDBStore()
        if hasattr(db, "init_schema"):
            db.init_schema()

        records = db.get_all_scripts()
        if not records:
            _invoke_flexible(seed_demo_winners_if_empty, db_store=db, store=db)
            records = db.get_all_scripts()
            run_train(use_llm_textgrad=False)

        # 1. Search VectorMemoryEngine for Top 3 CAC Winners + Losers to avoid
        vector_engine = VectorMemoryEngine()
        retrieval_stats = vector_engine.search_similar_scripts(
            concept_brief=brief,
            language=language,
            records=records,
            top_k_winners=3,
            top_k_losers=1,
        )

        raw_top_winners = retrieval_stats.get("top_winners", [])
        raw_losers_to_avoid = retrieval_stats.get("losers_to_avoid", [])

        # 2. Extract winners' `visual_blueprint` and `story_blocks` from SQL/store
        enriched_winners: list[dict[str, Any]] = []
        winner_visual_blueprints: list[dict[str, Any]] = []
        winner_story_blocks_list: list[dict[str, Any]] = []

        for item in raw_top_winners:
            rec = item.get("record", item) if isinstance(item, dict) else {}
            ad_code = item.get("ad_code") or rec.get("ad_code")
            sql_record = db.get_script_by_code(ad_code) if ad_code and hasattr(db, "get_script_by_code") else None
            source_rec = sql_record or rec or item

            visual_blueprint = source_rec.get("visual_blueprint") or {}
            story_blocks = source_rec.get("story_blocks") or {}
            cac_metrics = source_rec.get("cac_metrics") or {}
            metadata_tags = source_rec.get("metadata_tags") or {}

            winner_entry = {
                "ad_code": ad_code or source_rec.get("ad_code", "UNKNOWN"),
                "title": source_rec.get("title", ""),
                "language": source_rec.get("language", language),
                "character_type": source_rec.get("character_type", character),
                "cac_tier": source_rec.get("cac_tier", "TIER_S"),
                "cac_metrics": cac_metrics,
                "cosine_similarity": item.get("cosine_similarity", item.get("similarity", 0.0)),
                "ranking_score": item.get("ranking_score", item.get("score", 0.0)),
                "story_blocks": story_blocks,
                "visual_blueprint": visual_blueprint,
                "metadata_tags": metadata_tags,
                "english_script": source_rec.get("english_script", ""),
            }
            enriched_winners.append(winner_entry)
            winner_visual_blueprints.append(
                {
                    "ad_code": winner_entry["ad_code"],
                    "cac_tier": winner_entry["cac_tier"],
                    "visual_blueprint": visual_blueprint,
                }
            )
            winner_story_blocks_list.append(
                {
                    "ad_code": winner_entry["ad_code"],
                    "cac_tier": winner_entry["cac_tier"],
                    "cac_metrics": cac_metrics,
                    "story_blocks": story_blocks,
                    "english_script": winner_entry["english_script"],
                }
            )

        enriched_losers: list[dict[str, Any]] = []
        for item in raw_losers_to_avoid:
            rec = item.get("record", item) if isinstance(item, dict) else {}
            ad_code = item.get("ad_code") or rec.get("ad_code")
            sql_record = db.get_script_by_code(ad_code) if ad_code and hasattr(db, "get_script_by_code") else None
            source_rec = sql_record or rec or item
            enriched_losers.append(
                {
                    "ad_code": ad_code or source_rec.get("ad_code", "UNKNOWN"),
                    "title": source_rec.get("title", ""),
                    "language": source_rec.get("language", language),
                    "cac_tier": source_rec.get("cac_tier", "TIER_B"),
                    "cac_metrics": source_rec.get("cac_metrics") or {},
                    "cosine_similarity": item.get("cosine_similarity", item.get("similarity", 0.0)),
                    "story_blocks": source_rec.get("story_blocks") or {},
                    "metadata_tags": source_rec.get("metadata_tags") or {},
                    "english_script": source_rec.get("english_script", ""),
                }
            )

        # Load Contrastive Playbook Rules
        playbook = load_contrastive_playbook()
        playbook_with_losers = {
            "contrastive_playbook": playbook,
            "losers_to_avoid": enriched_losers,
        }

        # 3. Run `generate_with_middle_brain` (`dspy_brain.py`)
        dspy_output = _invoke_flexible(
            generate_with_middle_brain,
            concept_brief=brief,
            brief=brief,
            language=language,
            ai_teacher_character=character,
            character=character,
            character_type=character,
            matched_winner_scripts=enriched_winners,
            top_winners=enriched_winners,
            matched_winners=enriched_winners,
            matched_winner_visuals=winner_visual_blueprints,
            winner_visuals=winner_visual_blueprints,
            visual_blueprints=winner_visual_blueprints,
            contrastive_playbook_rules=playbook_with_losers,
            contrastive_playbook=playbook,
            playbook=playbook,
            losers_to_avoid=enriched_losers,
            records=records,
        )

        if not isinstance(dspy_output, dict):
            dspy_output = {"full_supernova_script": str(dspy_output)}

        full_supernova_script: str = (
            dspy_output.get("full_supernova_script")
            or dspy_output.get("script")
            or dspy_output.get("script_text")
            or ""
        )
        visual_description_block: Any = (
            dspy_output.get("visual_description_block")
            or dspy_output.get("visual_description")
            or dspy_output.get("visual_blueprint")
            or ""
        )
        three_hook_options: Any = (
            dspy_output.get("three_hook_options")
            or dspy_output.get("three_hooks")
            or dspy_output.get("hook_options")
            or []
        )
        four_block_thinking_plan: Any = (
            dspy_output.get("four_block_thinking_plan")
            or dspy_output.get("thinking_plan")
            or dspy_output.get("story_blocks")
            or ""
        )

        # Validate generated script with `validate_supernova_script` (`compliance_validator.py`)
        compliance_result = _invoke_flexible(
            validate_supernova_script,
            full_supernova_script,
            script_text=full_supernova_script,
            language=language,
            ai_teacher_character=character,
            character=character,
        )

        # 4. Parse the new script with `parse_proven_doc_text` and compute Approximate Success Rate %
        combined_doc_for_parse = (
            f"{visual_description_block}\n\nScript:\n{full_supernova_script}"
            if isinstance(visual_description_block, str) and visual_description_block.strip()
            else full_supernova_script
        )
        parsed_new_script = _invoke_flexible(
            parse_proven_doc_text,
            combined_doc_for_parse,
            raw_text=combined_doc_for_parse,
            language_hint=language,
            title=brief[:80],
        )

        # Ensure retrieval_stats carries the enriched winners so `predict_success_rate` can inspect them
        retrieval_stats_for_pred = dict(retrieval_stats)
        retrieval_stats_for_pred["top_winners"] = enriched_winners
        retrieval_stats_for_pred["losers_to_avoid"] = enriched_losers

        prediction_result = _invoke_flexible(
            predict_success_rate,
            parsed_new_script,
            retrieval_stats_for_pred,
            compliance_result,
            parsed_new_script=parsed_new_script,
            parsed_script=parsed_new_script,
            retrieval_stats=retrieval_stats_for_pred,
            compliance_result=compliance_result,
        )

        predicted_success_rate_pct = int(
            prediction_result.get("predicted_success_rate_pct", 75)
        )
        confidence_label = str(
            prediction_result.get("confidence_label", "Moderate-High Confidence")
        )
        historical_benchmark = _build_historical_benchmark_summary(enriched_winners)

        # 5. Assemble unified result dictionary & format complete Google Doc output
        complete_package: dict[str, Any] = {
            "status": "ok",
            "command": "generate",
            "brief": brief,
            "concept_brief": brief,
            "language": language,
            "character": character,
            "ai_teacher_character": character,
            "predicted_success_rate_pct": predicted_success_rate_pct,
            "predicted_success_rate": predicted_success_rate_pct,
            "success_rate": predicted_success_rate_pct,
            "confidence_label": confidence_label,
            "historical_benchmark": historical_benchmark,
            "historical_cluster_win_rate_pct": prediction_result.get(
                "historical_cluster_win_rate_pct", 0
            ),
            "pacing_alignment_score_pct": prediction_result.get(
                "pacing_alignment_score_pct", 0
            ),
            "compliance_score_pct": prediction_result.get(
                "compliance_score_pct",
                int(round(float(compliance_result.get("score", 1.0)) * 100)),
            ),
            "matched_winner_codes": prediction_result.get(
                "matched_winner_codes",
                [w["ad_code"] for w in enriched_winners],
            ),
            "explanation_bullets": prediction_result.get("explanation_bullets", []),
            "visual_description": visual_description_block,
            "visual_description_block": visual_description_block,
            "three_hooks": three_hook_options,
            "three_hook_options": three_hook_options,
            "hook_options": three_hook_options,
            "four_block_thinking_plan": four_block_thinking_plan,
            "four_block_plan": four_block_thinking_plan,
            "thinking_plan": four_block_thinking_plan,
            "middle_brain_strategy": four_block_thinking_plan,
            "full_supernova_script": full_supernova_script,
            "script": full_supernova_script,
            "matched_winners": enriched_winners,
            "losers_to_avoid": enriched_losers,
            "compliance": compliance_result,
            "compliance_result": compliance_result,
            "success_prediction": prediction_result,
            "prediction": prediction_result,
            "parsed_script": {
                "visual_blueprint": parsed_new_script.get("visual_blueprint"),
                "story_blocks": parsed_new_script.get("story_blocks"),
                "metadata_tags": parsed_new_script.get("metadata_tags"),
            },
            "token_usage": dspy_output.get(
                "token_usage",
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "llm_calls": 1},
            ),
            "generation_meta": {
                "used_dspy": dspy_output.get("used_dspy", True),
                "model": dspy_output.get("model", os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")),
                "retries": dspy_output.get("retries", 0),
                "elapsed_seconds": round(time.time() - t0, 3),
            },
        }

        formatted_doc_text = _invoke_flexible(
            format_complete_google_doc_text,
            complete_package,
            result=complete_package,
        )
        complete_package["formatted_google_doc_text"] = formatted_doc_text

        return complete_package


def handle_generate(args: argparse.Namespace) -> dict[str, Any]:
    """CLI handler for `generate` subcommand."""
    return run_generate(
        brief=args.brief,
        language=args.language,
        character=args.character,
    )


# ---------------------------------------------------------------------------
# CLI Argument Parser & Entrypoint
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_brain.py",
        description="Supernova Script Brain Orchestrator & CLI Entrypoint",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # 1. status
    p_status = subparsers.add_parser(
        "status",
        help="Check installed package versions, initialize DB schema, optionally seed demo winners, and print JSON status.",
    )
    p_status.add_argument(
        "--seed-if-empty",
        action="store_true",
        default=False,
        help="Seed demo CAC winners ONLY if the database currently has 0 records.",
    )

    # 2. ingest
    p_ingest = subparsers.add_parser(
        "ingest",
        help="Ingest proven scripts + CAC metrics from a JSON file or Google Sheet (0 Gemini API calls) and train models.",
    )
    p_ingest.add_argument(
        "--file",
        "-f",
        default=None,
        help="Path to JSON file containing proven scripts and CAC metrics.",
    )
    p_ingest.add_argument(
        "--from-sheet",
        action="store_true",
        default=False,
        help=f"Ingest proven scripts directly from Google Sheet (defaults to sheet '{DEFAULT_SHEET_ID}').",
    )
    p_ingest.add_argument(
        "--sheet-id",
        nargs="?",
        const=DEFAULT_SHEET_ID,
        default=None,
        help=f"Google Spreadsheet ID (defaults to '{DEFAULT_SHEET_ID}' when --from-sheet or --sheet-id is passed).",
    )
    p_ingest.add_argument(
        "--tab",
        "--sheet-name",
        dest="tab",
        default=DEFAULT_SHEET_TAB,
        help=f"Google Sheet tab name (defaults to '{DEFAULT_SHEET_TAB}').",
    )
    p_ingest.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of unique proven scripts with Google Docs to ingest (e.g. --limit 5).",
    )
    p_ingest.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help="Wipe any old/demo records from BrainDBStore before ingesting real scripts from the sheet.",
    )
    p_ingest.add_argument(
        "--no-train",
        action="store_true",
        default=False,
        help="Skip automatic model training after ingestion.",
    )

    # 3. train
    p_train = subparsers.add_parser(
        "train",
        help="Index vectors, build contrastive_playbook.json, compile compiled_dspy_brain.json, and train xgb_success_model.pkl.",
    )
    p_train.add_argument(
        "--use-llm-textgrad",
        action="store_true",
        default=False,
        help="Run optional LLM TextGrad textual backpropagation pass in addition to statistical extraction.",
    )

    # 4. generate
    p_gen = subparsers.add_parser(
        "generate",
        help="Generate a complete Supernova script package from a Concept Brief using the Middle Brain.",
    )
    p_gen.add_argument(
        "--brief",
        "-b",
        required=True,
        help="Short 1-3 line concept brief in English.",
    )
    p_gen.add_argument(
        "--language",
        "-l",
        default="Tamil",
        help="Target language (Tamil, Telugu, Malayalam, Kannada, Hindi).",
    )
    p_gen.add_argument(
        "--character",
        "-c",
        default="Female Humanoid Robot",
        help="AI teacher character ('Female Humanoid Robot', 'Female Robot', or 'Miss Nova').",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        if args.subcommand == "status":
            result = handle_status(args)
        elif args.subcommand == "ingest":
            result = handle_ingest(args)
        elif args.subcommand == "train":
            result = handle_train(args)
        elif args.subcommand == "generate":
            result = handle_generate(args)
        else:
            raise ValueError(f"Unknown subcommand: {args.subcommand}")

        sys.__stdout__.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        sys.__stdout__.flush()
        return 0

    except Exception as exc:
        err_payload = {
            "status": "error",
            "command": getattr(args, "subcommand", None),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        sys.__stdout__.write(json.dumps(err_payload, indent=2, ensure_ascii=False) + "\n")
        sys.__stdout__.flush()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
