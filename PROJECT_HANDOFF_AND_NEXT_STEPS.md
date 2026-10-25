# Script writer-AI — Complete Project Context, Architecture & Next Steps Build Plan

> **Purpose of this file:** When opening `/Users/mathiniraiselvan/Documents/Office project/Script writer-AI` as a standalone workspace, read this file first. It contains **all context from previous sessions**, **what is already built and working**, **how the 12-Layer Supernova AI Brain works**, and the **exact step-by-step checklist of what needs to be built next**.

---

## 1. Project Location, GitHub & Environment

- **Local Project Directory**: `/Users/mathiniraiselvan/Documents/Office project/Script writer-AI`
- **GitHub Repository**: `https://github.com/mathiniraiai-ai/Script-writer-AI` (`main` branch)
- **Python Virtual Environment**: `ad-studio/script-brain/.venv/bin/python` (Python 3.14 virtualenv with `dspy==3.4.1`, `chromadb`, `sentence-transformers`, `textgrad`, `xgboost`, `scikit-learn`, `deep-translator`, `optuna`, `google-api-python-client` pre-installed)
- **Environment Variables (`.env`)**:
  - `GEMINI_API_KEY` — Used **only** at final script generation time (`gemini-3.1-flash-lite-preview` / `gemini-2.5-flash`)
  - `GOOGLE_SERVICE_ACCOUNT_EMAIL` & `GOOGLE_PRIVATE_KEY` — Used to read the Supernova Replication Queue Google Sheet and fetch multi-tab Google Docs with **0 Gemini tokens**
- **How to Run Locally**:
  - Start Web UI + Express API (`http://localhost:3001`): `npm run dev`
  - Check Python Brain CLI status: `ad-studio/script-brain/.venv/bin/python ad-studio/script-brain/run_brain.py status`

---

## 2. What Is Already Built & Working (Current 5-Layer Baseline)

All core files are in place and tested end-to-end:

| File Path | Role in Current Baseline |
| :--- | :--- |
| `ad-studio/script-brain/script_parser.py` | **Zero-Token Multi-Language Script Parser**: Separates `visual_blueprint` (`Hook visuals`, `Body visuals`, `Split Screen`, `Video flow`, `Characters`) from spoken dialogue, strips all YouTube/Notion/Reference links, translates Indic scripts (`Hindi`, `Tamil`, `Telugu`, `Malayalam`, `Kannada`, `Marathi`, `Bengali`, `Punjabi`, `Gujarati`, `Odia`) to English via Google Translate (`0` Gemini tokens), splits into 4 Supernova Story Blocks, and extracts pacing metadata (`hook_word_count`, `avg_words_per_line`, `ai_teacher_entry_turn`, `has_first_half_usps`). |
| `ad-studio/script-brain/ingest_brain.py` | **Google Sheets & Multi-Tab Google Docs Ingester**: Connects to the Replication Queue Google Sheet (`13sQlJMgAZ_nFxatPEp5XfZ9swE4et00qEBpwNS2a82I`), parses the multi-line `Performance` column (`Spends`, `1st-Pay CAC`, `Post-1Hr Cancel CAC`, `2nd-Pay 4d CAC`, `2nd-Pay 7d CAC`, `1d/3d Cancel %`), selects the matching language tab from multi-tab Google Docs without concatenating 8 languages together, and strictly ignores YouTube/Notion links. |
| `ad-studio/script-brain/vector_memory.py` | **Layer 1 (Baseline) — Local Vector Memory (`ChromaDB` + `all-MiniLM-L6-v2`)**: Embeds every script locally on CPU (`0` API tokens) and retrieves Top-K CAC Winners + Losers to avoid. |
| `ad-studio/script-brain/contrastive_trainer.py` | **Layer 2 (Baseline) — Contrastive Playbook (`TextGrad` + Statistical DNA)**: Compares `TIER_S`/`TIER_A` winners against `TIER_B` losers to build `data/contrastive_playbook.json`. |
| `ad-studio/script-brain/dspy_brain.py` | **Layer 3 (Baseline) — Stanford DSPy Middle Brain (`SupernovaMiddleBrain`)**: Uses `dspy.ChainOfThought(ConceptBriefToSupernovaPackage)` + compiled few-shot demonstrations (`data/compiled_dspy_brain.json`) to turn a short 1–3 line brief into: (1) `Visual Description`, (2) `3 Hook + Bridge Options`, (3) `4-Block Thinking Plan`, and (4) `Full Supernova Script`. Tracks exact Gemini `input_tokens`, `output_tokens`, and `total_tokens`. |
| `ad-studio/script-brain/success_predictor.py` | **Layer 4 (Baseline) — Local `XGBoost` + `Scikit-Learn` Predictor**: Trains `data/xgb_success_model.pkl` on 12 structural/CAC features (`0` API tokens) and outputs `predicted_success_rate_pct` (`0–100%`). |
| `ad-studio/script-brain/compliance_validator.py` | **Layer 5 (Baseline) — Deterministic Supernova Compliance Validator**: Enforces all 7 hard-locked Supernova phrases (`"improve your spoken English"`, `"within 30 days"`, `"15 to 20 minutes"`, `"over 1 crore Indians"`, `"through <Language> itself"`, `"Install Now button given below"`, `"today itself"`) and blocks forbidden phrases via context-aware regexes. |
| `ad-studio/script-brain/run_brain.py` & `bridge.ts` | **CLI Orchestrator & Node-to-Python Bridge**: Connects `server.ts` and `src/pages/script-brain/ScriptBrain.tsx` to the Python Brain. |
| `src/pages/script-brain/ScriptBrain.tsx` | **React Frontend UI**: Lets the user sync from the Replication Queue Google Sheet, inspect trained scripts & CAC metrics, enter a brief, generate a full script package, and view the Predicted Success Rate % + Gemini token usage badge. |

### Currently Ingested Test Data (`ad-studio/script-brain/data/brain_store.json`)
Currently holds the **first 5 real scripts** synced from `REPLICATE - LANGUAGE`:
1. `W40_LONG_HAIRED_INSULTS` (`TIER_S`, ₹5.58L spend, 7d 2nd-Pay CAC `-26.1%`)
2. `W37_STICK_HONEYMOON` (`TIER_S`, ₹8.16L spend, 7d 2nd-Pay CAC `-16.2%`)
3. `W33_SUDDENLY_IMMEDIATELY` (`TIER_S`, ₹5.08L spend, 1st-Pay CAC `-15.7%`)
4. `W34_BOYS_PARTY` (`TIER_A`, ₹4.88L spend, 1st-Pay CAC `-8.7%`)
5. `W36_LEANING_CAMERA` (`TIER_S`, ₹3.95L spend, 1st-Pay CAC `-17.0%`)

---

## 3. Plain-English Summary: How We Upgrade This into a Custom 12-Layer Supernova AI Brain

- **Do we build a heavy neural network from scratch?** **No.** We do not train a 7B–70B LLM from scratch (zero GPU needed).
- **How do we make it a specialized Supernova model?** We combine **Gemini (`gemini-3.5-flash-lite` / `gemini-2.5-flash`)** as the final writer with **local open-source ML/AI layers** (`Sentence-Transformers`, `CrossEncoder`, `NetworkX`, `XGBoost`, `DSPy`) that learn directly from your Google Sheet's CAC and 7-day retention numbers.
- **How many layers total?** **12 Layers total** (**5 already built** + **7 new layers to build next**), using only **`~260 MB` of RAM** on CPU.

### The 12 Layers at a Glance
| # | Layer Name | Status | Engine / Library | Gemini Token Cost |
| :---: | :--- | :---: | :--- | :---: |
| **1** | Local Vector Memory | ✅ Built | `chromadb` + `all-MiniLM-L6-v2` | `0` tokens |
| **2** | Contrastive Winner-vs-Loser Playbook | ✅ Built | `textgrad` + Statistical DNA | `0` tokens |
| **3** | DSPy Structured 4-Block Generator | ✅ Built | `dspy.ChainOfThought` + Gemini | `~3,800` tokens/gen |
| **4** | Single-Score CAC Predictor | ✅ Built | `xgboost` + `scikit-learn` | `0` tokens |
| **5** | Deterministic House-Rule Validator | ✅ Built | Python Regex Gate (`compliance_validator.py`) | `0` tokens |
| **6** | **Contrastive MiniLM + Cross-Encoder Reranker** | 🔨 **Build Next** | `sentence-transformers` (`SetFit` + `CrossEncoder`) | `0` tokens |
| **7** | **Creative Knowledge Graph + Bandit Explorer** | 🔨 **Build Next** | `networkx` + Thompson Sampling (`scipy`/`numpy`) | `0` tokens |
| **8** | **4-Head Funnel Predictor (`1st-Pay` vs `7d 2nd-Pay` vs `Cancel Risk`)** | 🔨 **Build Next** | `xgboost` (`Multi-Head Calibrated` + `XGBRanker`) | `0` tokens |
| **9** | **Spoken Cadence, Breath-Group & Lip-Sync Scorer** | 🔨 **Build Next** | Deterministic Indic Akshara + English Syllable Analyzer | `0` tokens |
| **10** | **4-Block Tree-of-Thoughts (ToT) + `dspy.BestOfN` / `dspy.Refine`** | 🔨 **Build Next** | `dspy.BestOfN` + Local Reward Function | `+0` to `+1,800` tokens |
| **11** | **Episodic Reflexion Memory (`Weak 2nd Pay` Guardrails)** | 🔨 **Build Next** | `reflexion_memory.py` (`data/reflexion_memory.json`) | `0` tokens |
| **12** | **Gemini Explicit Context Caching + `MIPROv2` Compiler** | 🔨 **Build Next** | `google-genai` `caches.create` + `dspy.MIPROv2` | **Saves `-70%` to `-75%` input tokens** |

---

## 4. What Needs to Be Built Next (Step-by-Step Execution Checklist)

### STEP 1: Sync All Tabs from the Replication Queue Sheet & Fix `WEAK 2ND PAY` Labeling
**Files to update:** `ad-studio/script-brain/ingest_brain.py`, `ad-studio/script-brain/run_brain.py`

1. **Critical Bug Fix in `classify_cac_tier()` (`ingest_brain.py`)**:
   - In the Google Sheet (`13sQlJMgAZ_nFxatPEp5XfZ9swE4et00qEBpwNS2a82I`), rows on the `REPLICATE - LANGUAGE - WEAK 2ND PAY` tab (such as `W32_AUTODRIVER_LEARNING`, `W27_VERY_TIRED_TODAY`, `W43_HELPING_SPEAK`) have cheap 1st-Pay CAC (`-3.3%` to `-4.3%`) but terrible 7-Day 2nd-Pay CAC (`+5.5%` to `+26.0%`) and high 3-day cancellation (`49.6%` to `54.7%`).
   - Update `parse_performance_cell()` and `classify_cac_tier()` so:
     - `cac_vs_avg_pct` prioritizes `second_pay_7d_vs_avg_pct` (falling back to `second_pay_4d_vs_avg_pct`, then `first_pay_vs_avg_pct`) instead of blindly taking `min(...)`.
     - If a row comes from a `*WEAK 2ND PAY*` tab, or has `first_pay_vs_avg_pct < 0` while `second_pay_7d_vs_avg_pct > 5.0` (or `cancel_3d_pct >= 49.0`), classify its `cac_tier` as `"TIER_B"` and set `funnel_verdict = "WEAK_2ND_PAY_HIGH_CHURN"`.
     - If a row comes from `NOT PROPOSED - LANGUAGE`, classify as `"TIER_B"` (`funnel_verdict = "NOT_PROPOSED"`).
2. **Multi-Tab Batch Ingestion Support**:
   - Add `--all-tabs` flag to `run_brain.py ingest --from-sheet` (and a button in `ScriptBrain.tsx`) to automatically ingest all unique Google Doc scripts across:
     - `REPLICATE - LANGUAGE`
     - `REPLICATE - CHARACTER`
     - `REPLICATE - LANGUAGE - WEAK 2ND PAY`
     - `REPLICATE - CHARACTER - WEAK 2ND PAY`
     - `NOT PROPOSED - LANGUAGE`

---

### STEP 2: Build Phase 1 — The 4 Local Zero-Token Intelligence Layers

#### 2A. Upgrade `ad-studio/script-brain/success_predictor.py` (4-Head Multi-Objective Funnel Predictor)
Instead of predicting a single `predicted_success_rate_pct`, train **3 calibrated heads + 1 pairwise ranker (`XGBRanker`)** on expanded 22-feature vectors:
1. **Head 1 — `first_pay_hook_score_pct` (`0–100%`)**: Probability of beating language-average 1st-Pay CAC (`first_pay_vs_avg_pct < 0`). Driven by Block 1 hook length, opening mistake/interruption, and visual hook.
2. **Head 2 — `second_pay_7d_retention_score_pct` (`0–100%`)**: Probability of beating 7-Day 2nd-Pay CAC (`second_pay_7d_vs_avg_pct <= -10%`). Driven by Block 3 interactive speaking drill (`repeat_after_me`), early USP density, and active learner desire in Block 4.
3. **Head 3 — `weak_2nd_pay_cancel_risk_pct` (`0–100%`)**: Probability of suffering the `"Weak 2nd Pay"` trap (cheap trial but `>48%` 3-day cancellation). Flags passive skits or clickbait hooks that don't qualify real English learners.
4. **Head 4 — `XGBRanker(objective="rank:pairwise")`**: Ranks candidate scripts across Grade `3` (`TIER_S`), Grade `2` (`TIER_A`), Grade `1` (`WEAK_2ND_PAY`), and Grade `0` (`TIER_B`).

#### 2B. Create `ad-studio/script-brain/prosody_analyzer.py` (Spoken Cadence, Breath-Group & Lip-Sync Layer)
Create a zero-API deterministic phonetic and cadence analyzer for Supernova AI video avatars (`Female Humanoid Robot`, `Miss Nova`, human characters):
- Count Indic **aksharas** (grapheme clusters across Devanagari, Tamil, Telugu, Kannada, Malayalam, Bengali, Punjabi, Gujarati, Odia) and English syllables.
- Check **Speaking Rate (`aksharas_per_sec` / `syllables_per_sec`)** against target `4.2–5.4 syllables/sec`.
- Check **Breath-Group Length** (words between punctuation pauses `, . ? ! —`) so AI avatar lip-sync never exceeds `12 words` without a natural breath pause.
- Check **3-Stage Emotional Arc**:
  - Block 1–2: Hesitation / embarrassment markers
  - Block 3: Interactive repetition drill (`"Say:"`, `"Repeat:"`, `"Try saying:"`)
  - Block 4: High-agency confidence (`"Install Now button given below"`, `"today itself"`)
- Return `prosody_score_pct` (`0–100%`) and actionable warnings.

#### 2C. Upgrade `ad-studio/script-brain/vector_memory.py` (Contrastive MiniLM + Local `CrossEncoder` Reranker)
- Add **Stage-1 Contrastive Fine-Tuning** option on `(Winner, Winner)` vs. `(Winner, Weak-2nd-Pay/Loser)` pairs (`data/supernova-minilm-cac-v1`).
- Add **Stage-2 Local Cross-Encoder Reranker** (`sentence_transformers.CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`) to rerank the Top-15 ChromaDB matches down to the Top-3 true structural CAC winners in `~13 ms` on CPU (`0` Gemini tokens).

#### 2D. Create `ad-studio/script-brain/creative_graph.py` (`NetworkX` Creative Knowledge Graph + Thompson Sampling Bandit)
- Build a local heterogeneous graph (`data/creative_graph.json`) from ingested sheet rows:
  `(Persona) -> (Setting) -> (Hook Angle) -> (Split-Screen Visual Pattern) -> (Language CAC Win Rate)`.
- Implement a **Contextual Beta-Bernoulli Thompson Sampling Bandit** (`data/bandit_posteriors.json`):
  - **80% Exploitation**: Picks proven `TIER_S` structural paths for the requested brief/language.
  - **20% Exploration**: Suggests high-upside cross-language bridges (e.g., bringing a proven Hindi `TIER_S` hook/visual structure into Tamil/Telugu where it is not yet in `already_live_in`).

---

### STEP 3: Build Phase 2 — The 3 Self-Thinking & Token-Saving Layers

#### 3A. Create `ad-studio/script-brain/reflexion_memory.py` (Episodic Verbal Reinforcement Memory)
- Automatically analyze every `WEAK_2ND_PAY` and `TIER_B` script in `brain_store.json` against the closest `TIER_S` winner in the same persona/language.
- Store structured negative lessons in `data/reflexion_memory.json` (e.g., *"Avoid passive grammar lectures longer than 60s without a live speaking drill in Block 3; this caused +26% 7d 2nd-Pay CAC and 53.7% 3-day cancellation in W32_AUTODRIVER_LEARNING"*).
- Retrieve the top 2 relevant failure guardrails at generation time and pass them into `dspy_brain.py`.

#### 3B. Upgrade `ad-studio/script-brain/dspy_brain.py` (`dspy.BestOfN` + 4-Block Tree-of-Thoughts + Context Caching)
1. **4-Block Tree-of-Thoughts (`SupernovaBlockToTPlanner`) / `dspy.BestOfN`**:
   - Branch 3 candidate Hook + Bridge pairs (`Block 1 [0–5s]`).
   - Score all 3 hooks locally in `< 4 ms` (`0` LLM judge tokens) using `success_predictor.py` + `prosody_analyzer.py` + `compliance_validator.py`.
   - Prune the 2 weaker hooks and expand only the highest-scoring hook into `Blocks 2–4` (or use `dspy.Refine` if `second_pay_7d_retention_score_pct < 75%` or `weak_2nd_pay_cancel_risk_pct >= 55%`).
2. **Gemini Explicit Context Caching**:
   - Cache the static Supernova house rules + `contrastive_playbook.json` + compiled few-shot examples via `google-genai` Context Caching (`ttl="3600s"`), reducing per-request input token cost by **~70–75%**.

---

### STEP 4: Update Frontend UI (`src/pages/script-brain/ScriptBrain.tsx`) & Backend (`server.ts`)
- Add **"Sync All 5 Sheet Tabs"** button in `ScriptBrain.tsx`.
- Upgrade the output scorecard to display all 3 funnel metrics:
  - **1st-Pay Hook Score (`%`)**
  - **7-Day 2nd-Pay Retention Score (`%`)**
  - **Weak-2nd-Pay Cancellation Risk (`%` + Low/High badge)**
  - **Spoken Cadence & Lip-Sync Score (`%` + syllables/sec)**
- Show the **Episodic Reflexion Guardrails Applied** and **Creative Graph Bandit Mode (`Exploit Tier-S` vs `Explore Cross-Language Angle`)**.

---

## 5. Deep Technical Reference
For full mathematical formulas, feature tables, and subagent research benchmarks, see:
- [`BRAIN_NOTE.md`](./BRAIN_NOTE.md)
- [`docs/SUPERNOVA_CUSTOM_AI_LAYERS_REPORT.md`](./docs/SUPERNOVA_CUSTOM_AI_LAYERS_REPORT.md)
- [`docs/SCRIPT_BRAIN_ARCHITECTURE.md`](./docs/SCRIPT_BRAIN_ARCHITECTURE.md)
