"""Local vector embedding generation and hybrid vector + CAC-tier search engine.

Operates with ZERO external or Gemini API calls:
- Primary embedding backend: `sentence_transformers` (`SentenceTransformer("all-MiniLM-L6-v2")`)
- Fallback embedding backend: `scikit-learn` `TfidfVectorizer` + deterministic semantic/n-gram
  projection and `numpy` cosine similarity when the transformer model has not been downloaded
  or is unavailable on the host Python version.
- Primary vector store: `chromadb` persistent client at `ad-studio/script-brain/data/.chromadb`
- Fallback vector store: Seamless in-memory `numpy` cosine similarity index when `chromadb`
  is unavailable on a given Python version (e.g., Python 3.14).

CRITICAL INVARIANT:
`visual_blueprint` (camera angles, outfit colors, split-screen layout, visual direction)
is NEVER included in any embedded text (`hook_text` or `script_text`). It is preserved
exclusively as payload metadata on the retrieved records.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np

    HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]
    HAS_NUMPY = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer

    HAS_SKLEARN = True
except Exception:  # pragma: no cover
    TfidfVectorizer = None  # type: ignore[assignment]
    HAS_SKLEARN = False

try:
    import chromadb

    HAS_CHROMADB = True
except Exception:  # pragma: no cover
    chromadb = None  # type: ignore[assignment]
    HAS_CHROMADB = False


EMBEDDING_DIM: int = 384
SOUTH_LANGUAGE_CLUSTER = {"tamil", "telugu", "malayalam", "kannada"}

# Canonical domain concepts for Supernova spoken-English ad scripts so that the
# deterministic TF-IDF + concept fallback captures both lexical and semantic similarity.
_SEMANTIC_CONCEPT_CLUSTERS: List[Tuple[str, Tuple[str, ...]]] = [
    (
        "concept_parent_child_school",
        (
            "mother",
            "mom",
            "father",
            "dad",
            "parent",
            "parents",
            "child",
            "kid",
            "kids",
            "son",
            "daughter",
            "school",
            "ptm",
            "teacher",
            "homework",
            "principal",
            "admission",
            "classmate",
            "housewife",
            "homemaker",
        ),
    ),
    (
        "concept_office_career_interview",
        (
            "office",
            "job",
            "interview",
            "manager",
            "boss",
            "colleague",
            "colleagues",
            "promotion",
            "meeting",
            "client",
            "workplace",
            "corporate",
            "resume",
            "salary",
            "career",
            "hr",
            "presentation",
            "it",
            "professional",
            "engineer",
        ),
    ),
    (
        "concept_speaking_hesitation_pain",
        (
            "hesitate",
            "hesitation",
            "stutter",
            "stammer",
            "nervous",
            "fear",
            "afraid",
            "scared",
            "shy",
            "embarrassed",
            "embarrassment",
            "laugh",
            "laughed",
            "mock",
            "mocked",
            "judged",
            "shame",
            "stuck",
            "blank",
            "freeze",
            "froze",
            "struggle",
            "struggling",
            "broken",
            "weak",
            "poor",
            "mistake",
            "mistakes",
            "wrong",
            "confidence",
            "underconfident",
        ),
    ),
    (
        "concept_old_methods_failure",
        (
            "youtube",
            "video",
            "videos",
            "grammar",
            "book",
            "books",
            "coaching",
            "class",
            "classes",
            "institute",
            "course",
            "courses",
            "memorize",
            "memorizing",
            "translation",
            "translating",
            "newspaper",
            "dictionary",
            "partner",
            "nobody",
            "environment",
            "tense",
            "tenses",
            "rules",
        ),
    ),
    (
        "concept_ai_tutor_live_fix",
        (
            "supernova",
            "ai",
            "robot",
            "humanoid",
            "female",
            "tutor",
            "teacher",
            "mentor",
            "app",
            "practice",
            "speaking",
            "spoken",
            "live",
            "correction",
            "corrects",
            "instant",
            "feedback",
            "judgment",
            "judge",
            "24x7",
            "anytime",
            "daily",
            "minutes",
            "15",
            "20",
            "30",
            "days",
            "crore",
            "indians",
        ),
    ),
    (
        "concept_native_language_bridge",
        (
            "tamil",
            "telugu",
            "malayalam",
            "kannada",
            "hindi",
            "mother",
            "tongue",
            "native",
            "regional",
            "explain",
            "explains",
            "through",
            "itself",
            "understand",
            "simple",
            "easy",
        ),
    ),
    (
        "concept_social_public_setting",
        (
            "restaurant",
            "cafe",
            "mall",
            "airport",
            "flight",
            "hotel",
            "reception",
            "bank",
            "hospital",
            "doctor",
            "wedding",
            "party",
            "relatives",
            "neighbor",
            "neighbours",
            "friends",
            "public",
            "waiter",
            "counter",
            "shopping",
        ),
    ),
    (
        "concept_college_student_freshers",
        (
            "college",
            "student",
            "students",
            "campus",
            "placement",
            "fresher",
            "freshers",
            "university",
            "hostel",
            "exam",
            "viva",
            "seminar",
            "internship",
            "graduate",
            "degree",
        ),
    ),
    (
        "concept_desire_fluency_cta",
        (
            "fluent",
            "fluently",
            "fluency",
            "confident",
            "confidently",
            "improve",
            "improved",
            "transform",
            "proud",
            "respect",
            "impress",
            "impressed",
            "install",
            "download",
            "button",
            "below",
            "today",
            "start",
            "journey",
        ),
    ),
]

_SEED_CORPUS_SENTENCES: List[str] = [
    "homemaker mother struggles to speak English at school PTM meeting with teacher and principal",
    "software engineer IT professional freezes during client call and manager interview in office",
    "college fresher student nervous during campus placement interview due to spoken English hesitation",
    "watching YouTube grammar videos and reading books failed because there is no speaking practice partner",
    "Supernova AI Female Humanoid Robot teacher corrects mistakes live and explains through Tamil itself",
    "improve your spoken English within 30 days by practicing 15 to 20 minutes daily with over 1 crore Indians",
    "click the Install Now button given below and start speaking fluent English confidently today itself",
    "social embarrassment at restaurant cafe airport hotel reception when ordering or talking in English",
    "colleagues and relatives laughed at broken English grammar mistakes and pronunciation hesitation",
    "Explain in Malayalam Telugu Kannada Hindi native language without fear or judgment 24x7 practice",
]


def _to_dict(record: Any) -> Dict[str, Any]:
    """Convert a record (dict, Pydantic model, or object) into a plain dictionary."""
    if isinstance(record, dict):
        return record
    if hasattr(record, "model_dump") and callable(record.model_dump):
        return record.model_dump()
    if hasattr(record, "dict") and callable(record.dict):
        return record.dict()
    if hasattr(record, "__dict__"):
        return dict(record.__dict__)
    return {}


def _extract_block_english_text(block_val: Any) -> str:
    """Extract only spoken/narrative English text from a story block, never visual notes."""
    if block_val is None:
        return ""
    if isinstance(block_val, str):
        return block_val.strip()
    if isinstance(block_val, dict):
        # Strictly pick text/dialogue fields, never visual_blueprint or camera fields
        for key in (
            "english",
            "english_text",
            "text",
            "dialogue",
            "content",
            "script",
            "summary",
            "narrative",
        ):
            val = block_val.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        # Fallback: join non-visual string values
        parts: List[str] = []
        for k, v in block_val.items():
            k_lower = str(k).lower()
            if any(
                vis_term in k_lower
                for vis_term in (
                    "visual",
                    "camera",
                    "outfit",
                    "split",
                    "layout",
                    "b_roll",
                    "broll",
                    "lighting",
                    "color",
                    "wardrobe",
                    "prop",
                )
            ):
                continue
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
        return " ".join(parts)
    if isinstance(block_val, list):
        return " ".join(_extract_block_english_text(item) for item in block_val if item)
    return str(block_val).strip()


def build_hook_text(record: Dict[str, Any]) -> str:
    """Build `hook_text` for embedding.

    CRITICAL RULE:
    - Never includes `visual_blueprint` (camera angles, outfit colors, split-screen layout).
    - Combines `metadata_tags.persona`, `metadata_tags.setting`, `metadata_tags.hook_angle`,
      and `story_blocks.block_1_hook` + `story_blocks.block_2_pain_and_old_methods`.
    """
    rec = _to_dict(record)
    metadata_tags = _to_dict(rec.get("metadata_tags") or {})
    story_blocks = _to_dict(rec.get("story_blocks") or {})

    persona = str(
        metadata_tags.get("persona") or rec.get("persona") or ""
    ).strip()
    setting = str(
        metadata_tags.get("setting") or rec.get("setting") or ""
    ).strip()
    hook_angle = str(
        metadata_tags.get("hook_angle") or rec.get("hook_angle") or ""
    ).strip()

    block_1 = _extract_block_english_text(
        story_blocks.get("block_1_hook")
        if "block_1_hook" in story_blocks
        else rec.get("block_1_hook")
    )
    block_2 = _extract_block_english_text(
        story_blocks.get("block_2_pain_and_old_methods")
        if "block_2_pain_and_old_methods" in story_blocks
        else rec.get("block_2_pain_and_old_methods")
    )

    segments: List[str] = []
    if persona:
        segments.append(f"Persona: {persona}")
    if setting:
        segments.append(f"Setting: {setting}")
    if hook_angle:
        segments.append(f"Hook Angle: {hook_angle}")
    if block_1:
        segments.append(block_1)
    if block_2:
        segments.append(block_2)

    return " | ".join(segments).strip()


def build_script_text(record: Dict[str, Any]) -> str:
    """Build `script_text` for embedding.

    CRITICAL RULE:
    - Never includes `visual_blueprint` (camera angles, outfit colors, split-screen layout).
    - Combines all 4 `story_blocks` (`block_1_hook`, `block_2_pain_and_old_methods`,
      `block_3_live_fix_and_usps`, `block_4_desire_and_cta`) in English.
    """
    rec = _to_dict(record)
    story_blocks = _to_dict(rec.get("story_blocks") or {})

    ordered_keys = (
        "block_1_hook",
        "block_2_pain_and_old_methods",
        "block_3_live_fix_and_usps",
        "block_4_desire_and_cta",
    )
    segments: List[str] = []
    for key in ordered_keys:
        raw_val = story_blocks.get(key) if key in story_blocks else rec.get(key)
        text_val = _extract_block_english_text(raw_val)
        if text_val:
            segments.append(text_val)

    if not segments and isinstance(rec.get("english_script"), str):
        return rec["english_script"].strip()

    return " ".join(segments).strip()


def _normalize_word(token: str) -> str:
    """Lightweight deterministic stemmer for English ad vocabulary."""
    w = token.lower().strip()
    if len(w) <= 3:
        return w
    for suffix in ("ingly", "edly", "ment", "ness", "tion", "sion", "able", "ible"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[: -len(suffix)]
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ing") and len(w) > 5:
        base = w[:-3]
        if len(base) >= 3 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if w.endswith("ed") and len(w) > 4:
        base = w[:-2]
        if len(base) >= 3 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if w.endswith("es") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "had",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
    "you",
    "your",
}


def _tokenize_and_enrich(text: str) -> Tuple[List[str], Dict[str, int]]:
    """Tokenize text into normalized stems and detect domain concept cluster hits."""
    clean = re.sub(r"[^a-zA-Z0-9\s]", " ", (text or "").lower())
    raw_tokens = [t for t in clean.split() if t and t not in _STOPWORDS]
    stems = [_normalize_word(t) for t in raw_tokens]

    token_set = set(raw_tokens) | set(stems)
    concept_hits: Dict[str, int] = {}
    for concept_name, keywords in _SEMANTIC_CONCEPT_CLUSTERS:
        hits = sum(
            1
            for kw in keywords
            if kw in token_set or _normalize_word(kw) in token_set
        )
        if hits > 0:
            concept_hits[concept_name] = hits

    return stems, concept_hits


def _cosine_similarity_vecs(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two equal-length vectors using numpy (or math fallback)."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    if HAS_NUMPY and np is not None:
        arr_a = np.asarray(vec_a, dtype=np.float64)
        arr_b = np.asarray(vec_b, dtype=np.float64)
        norm_a = float(np.linalg.norm(arr_a))
        norm_b = float(np.linalg.norm(arr_b))
        if norm_a <= 1e-12 or norm_b <= 1e-12:
            return 0.0
        sim = float(np.dot(arr_a, arr_b) / (norm_a * norm_b))
    else:
        dot = sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(float(a) * float(a) for a in vec_a))
        norm_b = math.sqrt(sum(float(b) * float(b) for b in vec_b))
        if norm_a <= 1e-12 or norm_b <= 1e-12:
            return 0.0
        sim = dot / (norm_a * norm_b)

    return max(0.0, min(1.0, sim))


def _normalize_cac_tier(record: Dict[str, Any]) -> str:
    """Extract and normalize CAC tier ('TIER_S', 'TIER_A', 'TIER_B') from a script record."""
    rec = _to_dict(record)
    perf = _to_dict(rec.get("performance_metrics") or {})
    cac_metrics = _to_dict(rec.get("cac_metrics") or {})
    meta = _to_dict(rec.get("metadata_tags") or {})

    raw_tier = (
        rec.get("cac_tier")
        or cac_metrics.get("cac_tier")
        or perf.get("cac_tier")
        or meta.get("cac_tier")
        or rec.get("tier")
        or cac_metrics.get("tier")
        or perf.get("tier")
        or ""
    )
    if hasattr(raw_tier, "value"):
        raw_tier = getattr(raw_tier, "value")
    tier_str = str(raw_tier).strip().upper().replace("-", "_").replace(" ", "_")

    if tier_str in ("TIER_S", "S"):
        return "TIER_S"
    if tier_str in ("TIER_A", "A"):
        return "TIER_A"
    if tier_str in ("TIER_B", "B", "TIER_C", "C", "LOSER"):
        return "TIER_B"
    return tier_str or "TIER_A"


def _extract_language(record: Dict[str, Any]) -> str:
    """Extract primary language string from a script record."""
    rec = _to_dict(record)
    meta = _to_dict(rec.get("metadata_tags") or {})
    lang = (
        rec.get("language")
        or meta.get("language")
        or rec.get("primary_language")
        or ""
    )
    if hasattr(lang, "value"):
        lang = getattr(lang, "value")
    return str(lang).strip()


def _compute_language_bonus(requested_language: str, record_language: str) -> float:
    """Apply South-cluster and Hindi language bonus rules.

    Rules:
    - South cluster (`Tamil`, `Telugu`, `Malayalam`, `Kannada`):
      share freely across South languages (`+0.00`), with `+0.05` bonus for exact language match.
    - `Hindi` prefers `Hindi` with `+0.08` bonus.
    """
    req = (requested_language or "").strip().lower()
    rec = (record_language or "").strip().lower()
    if not req or not rec:
        return 0.0

    # Support composite language strings like "Tamil / Malayalam"
    rec_tokens = {
        tok.strip() for tok in re.split(r"[/,|]+", rec) if tok.strip()
    } | {rec}
    req_tokens = {
        tok.strip() for tok in re.split(r"[/,|]+", req) if tok.strip()
    } | {req}

    exact_match = bool(req_tokens & rec_tokens)

    if req in SOUTH_LANGUAGE_CLUSTER or bool(req_tokens & SOUTH_LANGUAGE_CLUSTER):
        if exact_match:
            return 0.05
        if bool(rec_tokens & SOUTH_LANGUAGE_CLUSTER):
            # Shares freely within South cluster
            return 0.0
        return -0.03

    if "hindi" in req_tokens:
        if "hindi" in rec_tokens:
            return 0.08
        return 0.0

    return 0.05 if exact_match else 0.0


def _compute_cac_boost(cac_tier: str) -> float:
    """Return CAC tier ranking boost (`TIER_S` -> +0.12, `TIER_A` -> +0.05)."""
    tier = (cac_tier or "").upper()
    if tier == "TIER_S":
        return 0.12
    if tier == "TIER_A":
        return 0.05
    return 0.0


class VectorMemoryEngine:
    """Local vector embedding and hybrid CAC-tier retrieval engine for Supernova scripts."""

    def __init__(
        self,
        persist_directory: Optional[str | Path] = None,
        model_name: str = "all-MiniLM-L6-v2",
        allow_model_download: bool = False,
    ) -> None:
        self.model_name = model_name
        base_dir = Path(__file__).resolve().parent
        self.persist_directory = (
            Path(persist_directory)
            if persist_directory is not None
            else base_dir / "data" / ".chromadb"
        )

        self._st_model: Any = None
        self.embedding_backend: str = "tfidf_numpy_fallback"
        self._init_embedding_model(allow_model_download=allow_model_download)

        self._tfidf_vectorizer: Any = None
        self._tfidf_projection: Any = None
        self._init_tfidf_fallback()

        self._chroma_client: Any = None
        self._chroma_hooks_collection: Any = None
        self._chroma_scripts_collection: Any = None
        self.storage_backend: str = "numpy_in_memory"
        self._memory_store: Dict[str, Dict[str, Any]] = {}
        self._init_vector_store()

    def _init_embedding_model(self, allow_model_download: bool = False) -> None:
        """Initialize SentenceTransformer('all-MiniLM-L6-v2') if locally cached or allowed."""
        try:
            from sentence_transformers import SentenceTransformer

            should_download = allow_model_download or (
                os.environ.get("SCRIPT_BRAIN_DOWNLOAD_MODEL", "0") == "1"
            )
            if should_download:
                self._st_model = SentenceTransformer(self.model_name)
            else:
                self._st_model = SentenceTransformer(
                    self.model_name, local_files_only=True
                )
            self.embedding_backend = f"sentence_transformers:{self.model_name}"
        except Exception:
            self._st_model = None
            self.embedding_backend = (
                "sklearn_tfidf_numpy" if HAS_SKLEARN else "deterministic_tfidf_numpy"
            )

    def _init_tfidf_fallback(self) -> None:
        """Initialize deterministic scikit-learn TfidfVectorizer + 384-d projection fallback."""
        if not HAS_SKLEARN or TfidfVectorizer is None:
            return
        try:
            corpus: List[str] = list(_SEED_CORPUS_SENTENCES)
            for concept_name, keywords in _SEMANTIC_CONCEPT_CLUSTERS:
                corpus.append(f"{concept_name} " + " ".join(keywords))

            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=256,
                sublinear_tf=True,
                lowercase=True,
            )
            vectorizer.fit(corpus)
            self._tfidf_vectorizer = vectorizer
        except Exception:
            self._tfidf_vectorizer = None

    def _init_vector_store(self) -> None:
        """Initialize ChromaDB persistent client at `data/.chromadb` with in-memory fallback."""
        if not HAS_CHROMADB or chromadb is None:
            self.storage_backend = "numpy_in_memory"
            return

        try:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(self.persist_directory))
            self._chroma_hooks_collection = client.get_or_create_collection(
                name="script_brain_hooks",
                metadata={"hnsw:space": "cosine"},
            )
            self._chroma_scripts_collection = client.get_or_create_collection(
                name="script_brain_scripts",
                metadata={"hnsw:space": "cosine"},
            )
            self._chroma_client = client
            self.storage_backend = "chromadb_persistent"
        except Exception:
            self._chroma_client = None
            self._chroma_hooks_collection = None
            self._chroma_scripts_collection = None
            self.storage_backend = "numpy_in_memory"

    def _embed_with_fallback(self, text: str) -> List[float]:
        """Produce a deterministic 384-dim L2-normalized embedding using TF-IDF + feature hashing."""
        clean_text = (text or "").strip()
        if not clean_text:
            return [0.0] * EMBEDDING_DIM

        stems, concept_hits = _tokenize_and_enrich(clean_text)
        vec = [0.0] * EMBEDDING_DIM

        # 1. Domain semantic concept slots (slots 0..35, 4 slots per concept cluster)
        for idx, (concept_name, _) in enumerate(_SEMANTIC_CONCEPT_CLUSTERS):
            hits = concept_hits.get(concept_name, 0)
            if hits > 0:
                weight = min(2.2, 0.85 * (1.0 + math.log1p(hits)))
                base_slot = (idx * 4) % 36
                vec[base_slot] += weight
                vec[base_slot + 1] += weight * 0.75

        # 2. scikit-learn TfidfVectorizer features (slots 36..227, 192 slots)
        if self._tfidf_vectorizer is not None:
            try:
                enriched_str = " ".join(stems) + " " + " ".join(concept_hits.keys())
                tfidf_row = self._tfidf_vectorizer.transform([enriched_str]).toarray()[0]
                for i, val in enumerate(tfidf_row):
                    if val > 0.0:
                        slot = 36 + (i % 192)
                        vec[slot] += float(val) * 1.35
            except Exception:
                pass

        # 3. Deterministic signed feature hashing of stems, bigrams, and char 3-grams (slots 36..383)
        # Ensures any shared vocabulary (even outside the seed TF-IDF vocab) aligns in cosine space.
        features: List[Tuple[str, float]] = []
        for stem in stems:
            features.append((f"w:{stem}", 1.0))
        for i in range(len(stems) - 1):
            features.append((f"bi:{stems[i]}_{stems[i + 1]}", 0.85))
        for cname, count in concept_hits.items():
            features.append((f"c:{cname}", 1.25 * (1.0 + 0.25 * min(count, 4))))
        # Character 4-grams on normalized stems for morphological robustness
        joined_stems = "_".join(stems[:60])
        for i in range(0, max(0, len(joined_stems) - 4), 2):
            features.append((f"ch:{joined_stems[i : i + 4]}", 0.2))

        hash_span = EMBEDDING_DIM - 36
        for feat_key, feat_weight in features:
            digest = hashlib.md5(feat_key.encode("utf-8")).digest()
            bucket = 36 + (int.from_bytes(digest[:4], "little") % hash_span)
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[bucket] += sign * feat_weight

        # L2 normalize to unit length
        if HAS_NUMPY and np is not None:
            arr = np.asarray(vec, dtype=np.float64)
            norm = float(np.linalg.norm(arr))
            if norm > 1e-12:
                arr = arr / norm
            return [round(float(x), 6) for x in arr.tolist()]

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 1e-12:
            vec = [round(v / norm, 6) for v in vec]
        return vec

    def embed_text(self, text: str) -> List[float]:
        """Generate a local vector embedding for `text` with zero external API calls."""
        clean_text = (text or "").strip()
        if not clean_text:
            return [0.0] * EMBEDDING_DIM

        if self._st_model is not None:
            try:
                encoded = self._st_model.encode(
                    clean_text,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                if hasattr(encoded, "tolist"):
                    encoded = encoded.tolist()
                return [round(float(x), 6) for x in encoded]
            except Exception:
                pass

        return self._embed_with_fallback(clean_text)

    def embed_script_record(
        self, record: Dict[str, Any]
    ) -> Tuple[List[float], List[float]]:
        """Embed `hook_text` and `script_text` for a script record without external API calls.

        CRITICAL RULE:
        Never includes `visual_blueprint` inside `hook_text` or `script_text`.

        Returns:
            `(hook_vector, script_vector)` as a tuple of `list[float]`.
        """
        hook_text = build_hook_text(record)
        script_text = build_script_text(record)
        hook_vector = self.embed_text(hook_text)
        script_vector = self.embed_text(script_text)
        return hook_vector, script_vector

    def index_scripts(self, records: List[Dict[str, Any]]) -> int:
        """Index script records into ChromaDB (if available) and the in-memory store.

        Returns:
            Number of indexed records (`int`).
        """
        if not records:
            return 0

        indexed_count = 0
        ids: List[str] = []
        hook_embeddings: List[List[float]] = []
        script_embeddings: List[List[float]] = []
        hook_docs: List[str] = []
        script_docs: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for idx, raw_rec in enumerate(records):
            rec = dict(_to_dict(raw_rec))
            script_id = str(
                rec.get("ad_code")
                or rec.get("script_id")
                or rec.get("id")
                or f"script_{idx + 1:03d}"
            )

            hook_vec = rec.get("hook_vector")
            script_vec = rec.get("script_vector")
            if (
                not isinstance(hook_vec, list)
                or not isinstance(script_vec, list)
                or len(hook_vec) != EMBEDDING_DIM
                or len(script_vec) != EMBEDDING_DIM
            ):
                hook_vec, script_vec = self.embed_script_record(rec)
                rec["hook_vector"] = hook_vec
                rec["script_vector"] = script_vec

            hook_text = build_hook_text(rec)
            script_text = build_script_text(rec)

            self._memory_store[script_id] = rec
            indexed_count += 1

            meta_tags = _to_dict(rec.get("metadata_tags") or {})
            ids.append(script_id)
            hook_embeddings.append([float(x) for x in hook_vec])
            script_embeddings.append([float(x) for x in script_vec])
            hook_docs.append(hook_text)
            script_docs.append(script_text)
            metadatas.append(
                {
                    "ad_code": script_id,
                    "language": _extract_language(rec),
                    "cac_tier": _normalize_cac_tier(rec),
                    "persona": str(meta_tags.get("persona") or ""),
                    "setting": str(meta_tags.get("setting") or ""),
                    "hook_angle": str(meta_tags.get("hook_angle") or ""),
                }
            )

        if (
            self._chroma_hooks_collection is not None
            and self._chroma_scripts_collection is not None
            and ids
        ):
            try:
                self._chroma_hooks_collection.upsert(
                    ids=ids,
                    embeddings=hook_embeddings,
                    documents=hook_docs,
                    metadatas=metadatas,
                )
                self._chroma_scripts_collection.upsert(
                    ids=ids,
                    embeddings=script_embeddings,
                    documents=script_docs,
                    metadatas=metadatas,
                )
            except Exception:
                # Fallback remains in self._memory_store
                pass

        return indexed_count

    def search_similar_scripts(
        self,
        concept_brief: str,
        language: str,
        records: Optional[List[Dict[str, Any]]] = None,
        top_k_winners: int = 3,
        top_k_losers: int = 1,
    ) -> Dict[str, Any]:
        """Perform hybrid vector + language-cluster + CAC-tier search over historical scripts.

        - Embeds `concept_brief` locally.
        - Computes cosine similarity against both `hook_vector` (0.6 weight) and
          `script_vector` (0.4 weight) of all candidate records.
        - Applies South-cluster language rule (`Tamil`, `Telugu`, `Malayalam`, `Kannada`
          share freely with `+0.05` bonus for exact language match; `Hindi` prefers
          `Hindi` with `+0.08` bonus).
        - Weights winners by CAC performance (`TIER_S` gets `+0.12` ranking boost,
          `TIER_A` gets `+0.05` boost) while preserving raw `cosine_similarity` in the
          returned dictionary.
        """
        brief_vec = self.embed_text(concept_brief)

        candidate_records: List[Dict[str, Any]] = []
        if records:
            candidate_records = [_to_dict(r) for r in records]
        elif self._memory_store:
            candidate_records = list(self._memory_store.values())

        if not candidate_records:
            return {
                "brief_vector": brief_vec,
                "top_winners": [],
                "losers_to_avoid": [],
                "max_winner_similarity": 0.0,
                "mean_winner_similarity": 0.0,
                "mean_loser_similarity": 0.0,
                "cluster_historical_win_rate": 0.0,
            }

        scored_candidates: List[Dict[str, Any]] = []
        for idx, raw_rec in enumerate(candidate_records):
            rec = dict(_to_dict(raw_rec))
            hook_vec = rec.get("hook_vector")
            script_vec = rec.get("script_vector")

            if (
                not isinstance(hook_vec, list)
                or not isinstance(script_vec, list)
                or len(hook_vec) != len(brief_vec)
                or len(script_vec) != len(brief_vec)
                or not any(hook_vec)
            ):
                hook_vec, script_vec = self.embed_script_record(rec)

            hook_sim = _cosine_similarity_vecs(brief_vec, hook_vec)
            script_sim = _cosine_similarity_vecs(brief_vec, script_vec)
            raw_cosine_sim = (0.6 * hook_sim) + (0.4 * script_sim)

            rec_lang = _extract_language(rec)
            cac_tier = _normalize_cac_tier(rec)
            lang_bonus = _compute_language_bonus(language, rec_lang)
            cac_boost = _compute_cac_boost(cac_tier)

            ranking_score = raw_cosine_sim + lang_bonus + cac_boost
            topic_relevance_score = raw_cosine_sim + lang_bonus

            # Preserve full record (including visual_blueprint & story_blocks) + similarity fields
            enriched_entry: Dict[str, Any] = {
                **rec,
                "ad_code": rec.get("ad_code")
                or rec.get("script_id")
                or rec.get("id")
                or f"script_{idx + 1:03d}",
                "language": rec_lang,
                "cac_tier": cac_tier,
                "cosine_similarity": round(float(raw_cosine_sim), 4),
                "similarity": round(float(raw_cosine_sim), 4),
                "hook_similarity": round(float(hook_sim), 4),
                "script_similarity": round(float(script_sim), 4),
                "language_bonus": round(float(lang_bonus), 4),
                "cac_boost": round(float(cac_boost), 4),
                "ranking_score": round(float(ranking_score), 4),
                "hybrid_score": round(float(ranking_score), 4),
                "_topic_relevance_score": float(topic_relevance_score),
                "record": rec,
            }
            scored_candidates.append(enriched_entry)

        # Partition into winners (TIER_S, TIER_A) and losers (TIER_B)
        winner_pool = [
            c for c in scored_candidates if c["cac_tier"] in ("TIER_S", "TIER_A")
        ]
        loser_pool = [c for c in scored_candidates if c["cac_tier"] == "TIER_B"]

        # Sort winners by hybrid ranking_score (cosine_similarity + lang_bonus + cac_boost)
        winner_pool.sort(
            key=lambda x: (x["ranking_score"], x["cosine_similarity"]),
            reverse=True,
        )
        # Sort losers by topic relevance (cosine_similarity + lang_bonus) so DSPy sees relevant failures
        loser_pool.sort(
            key=lambda x: (x["_topic_relevance_score"], x["cosine_similarity"]),
            reverse=True,
        )

        top_winners: List[Dict[str, Any]] = []
        for item in winner_pool[: max(0, top_k_winners)]:
            clean_item = {
                k: v for k, v in item.items() if not k.startswith("_")
            }
            top_winners.append(clean_item)

        losers_to_avoid: List[Dict[str, Any]] = []
        for item in loser_pool[: max(0, top_k_losers)]:
            clean_item = {
                k: v for k, v in item.items() if not k.startswith("_")
            }
            losers_to_avoid.append(clean_item)

        # Compute similarity summary metrics using raw cosine_similarity
        if top_winners:
            winner_sims = [float(w["cosine_similarity"]) for w in top_winners]
            max_winner_similarity = round(max(winner_sims), 4)
            mean_winner_similarity = round(
                sum(winner_sims) / len(winner_sims), 4
            )
        elif winner_pool:
            winner_sims = [float(w["cosine_similarity"]) for w in winner_pool]
            max_winner_similarity = round(max(winner_sims), 4)
            mean_winner_similarity = round(
                sum(winner_sims) / len(winner_sims), 4
            )
        else:
            max_winner_similarity = 0.0
            mean_winner_similarity = 0.0

        if losers_to_avoid:
            loser_sims = [float(l["cosine_similarity"]) for l in losers_to_avoid]
            mean_loser_similarity = round(sum(loser_sims) / len(loser_sims), 4)
        elif loser_pool:
            loser_sims = [float(l["cosine_similarity"]) for l in loser_pool]
            mean_loser_similarity = round(sum(loser_sims) / len(loser_sims), 4)
        else:
            mean_loser_similarity = 0.0

        # Share of top-10 most similar scripts (by cosine_similarity + language_bonus) that are TIER_S or TIER_A
        top_10_similar = sorted(
            scored_candidates,
            key=lambda x: (x["_topic_relevance_score"], x["cosine_similarity"]),
            reverse=True,
        )[:10]
        if top_10_similar:
            winners_in_top_10 = sum(
                1
                for c in top_10_similar
                if c["cac_tier"] in ("TIER_S", "TIER_A")
            )
            cluster_historical_win_rate = round(
                winners_in_top_10 / float(len(top_10_similar)), 4
            )
        else:
            cluster_historical_win_rate = 0.0

        return {
            "brief_vector": brief_vec,
            "top_winners": top_winners,
            "losers_to_avoid": losers_to_avoid,
            "max_winner_similarity": float(max_winner_similarity),
            "mean_winner_similarity": float(mean_winner_similarity),
            "mean_loser_similarity": float(mean_loser_similarity),
            "cluster_historical_win_rate": float(cluster_historical_win_rate),
        }
