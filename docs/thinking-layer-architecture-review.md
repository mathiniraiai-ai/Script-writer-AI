# Architecture Review: Supernova "Thin Thinking Layer"

**Reviewed:** SUPERNOVA_THINKING_LAYER_ARCHITECTURE.md
**Date:** 22 October 2026
**Mode:** Brutal / adversarial review

---

## Verdict

**Not confirmed.**

The engineering in Phase 1 is solid. Phase 2 is a well-dressed solution to a problem that has not been diagnosed, built on a feedback loop that will actively make the creative worse.

Five structural problems follow, then answers to the three review questions, then a revised build order.

---

## 1. There is no metric, so none of Section 4 can work

DSPy, TextGrad and Gemini SFT are all optimizers. An optimizer is a function of its loss.

The stated rubric is "hook punchiness, cultural specificity, first-half USP placement, South format compliance." Two of those are mechanically checkable. The other two are LLM-judge vibes. MIPROv2 will happily climb that gradient and return a compiled prompt that maximizes a judge model's taste for punchiness, which has no established relationship to anything that matters.

**The part that should sting:** the real metric already exists. First-pay CAC, post-1hr cancellation CAC, 2nd-pay CAC 4d, 2nd-pay CAC 7d, cancellation %, each measured against the same-language average. That is a performance signal most creative teams never get. This architecture does not reference it once.

Consequence: the "Winning Script Vector Bank" is selected on **human approval**, not on CAC. The system would encode reviewer taste and label it winning. If reviewer taste were reliably predictive, there would be no need for a thinking layer at all.

**Fix:** label the bank by outcome, not approval. A script counts as winning only if its ad beat the language average on the five metrics. If that yields 12 scripts instead of 50, use 12. Twelve CAC-validated examples beat fifty approved ones.

---

## 2. Component C (Diff-to-Rule) is the worst idea in the document

The diff loop teaches the model to predict what the reviewer will not change. That is a different objective from winning. It distills the bottleneck into the system instead of removing it.

Four concrete failure modes:

| Failure | Detail |
| :--- | :--- |
| **Diffs are unlabelled** | An edit may be a compliance fix, a factual fix, a pacing improvement, or one writer's pet peeve. The evaluator prompt cannot tell these apart and files all of them as "thinking preferences." |
| **n=1 generalization** | "Cut introductory pleasantries; start directly with the interviewer's question" is either a real pattern or a one-script coincidence. There is no mechanism to distinguish them, and it gets promoted to a general rule either way. |
| **No retirement mechanism** | Section 6 Q2 flags prompt bloat and then proposes nothing for it. Rules accumulate monotonically, begin contradicting each other within weeks, and nobody deletes one because nobody knows which rule caused which outcome. |
| **Mem0 per-writer memory compounds it** | Per-writer preference storage entrenches individual idiosyncrasies as retrievable doctrine. |

---

## 3. The design optimizes for creative monoculture

Trace the loop as specified:

1. Retrieve top-3 most similar winners
2. Generate something resembling them
3. Get it approved because it resembles what gets approved
4. Add it to the bank
5. Return to step 1 with a narrower retrieval space

For paid social, creative **diversity is the asset**. Fatigue is what kills CAC. This architecture builds a machine that converges on the current winning pattern faster, which exhausts that pattern faster, with nothing queued behind it. There is no exploration term anywhere in the design.

**Related, and unasked:** the hard-locked phrase list plus "all 4 USPs in the first half" already forces heavy convergence across every shipped script. Has first-half versus second-half USP placement ever been tested against CAC, or is it house lore that got written down?

---

## 4. The RAG layer is unnecessary at n=50

Cosine similarity on `gemini-embedding-001` over a 1 to 3 sentence brief retrieves by **surface topic**: office, interview, school, PTM. It does not retrieve by the thing that actually transfers, which is hook mechanic, tension structure and pacing.

Two briefs about job interviews can require completely different scripts. A PTM brief and an appraisal brief might want an identical beat structure. Embedding similarity gets both cases backwards.

**At 50 items, do not use vector search.** Use four hand-assigned tags and a dictionary lookup:

- Hook type
- Setting
- Protagonist
- Tension source

This retrieves better, is debuggable, and has no embedding drift. Add embeddings at 500 examples, if that point is ever reached.

---

## 5. Effort allocation is backwards

Frame-stitching was optimized down to 255 image tokens. Good craft, on a cost line that is almost certainly rounding error against media spend.

Meanwhile there is:

- No eval harness
- No error taxonomy
- No held-out set

And the single highest-leverage element in the whole problem is the hook. It is one line long and the cheapest thing in the pipeline to generate. Ten hook variants with a human picking one in twenty seconds is half a day of work and will probably outperform the entire thinking layer. It does not appear in the document.

---

## Answers to the three review questions

### Q1. Two-call vs single-call chain-of-thought

**Single call.** Blueprint emitted as a structured block before the script.

Reasoning:

- A separate blueprint call creates a JSON hand-off. A dialogue writer handed a JSON spec writes to the spec, which produces exactly the box-ticking infomercial tone the project is trying to escape.
- Keeping blueprint and script in one context lets the writer model see the reasoning itself, not a lossy summary of it.

Two caveats:

1. **Do not make the blueprint rigid JSON.** Prose bullets preserve creative reasoning that schema fields flatten.
2. **Write the blueprint into Tab 2 of the Doc as a visible section.** The blueprint is a far better human intervention point than the finished script.

Move to two calls only when blueprint review becomes an actual workflow step with a human in it.

### Q2. Turning Google Doc diffs into updated rules

**Do not convert diffs to rules. Convert them to examples.**

The value in a diff is the human's final script, not a generalization about it. Append the approved final plus its brief to the example bank and let retrieval carry it.

Examples versus rules:

- Examples encode pacing, rhythm and voice that no rule can express
- Examples self-retire when dropped
- Examples can be weighted by CAC
- Rules are lossy, mutually conflicting and effectively permanent

**If leadership insists on rules anyway, constrain them hard:**

- A pattern needs N independent occurrences across M different writers before it becomes a rule
- Cap the rule list at roughly 15 entries
- Date-stamp every entry
- Review monthly; delete anything nobody will defend out loud
- Restrict the extractor to **structural** rules only (where things go, how long turns run). The moment it starts writing content rules, drift becomes unrecoverable.

**Before any of this:** have writers tag each edit with a reason code from a fixed six-item list.

1. Compliance
2. Factual
3. Pacing
4. Weak hook
5. Unnatural language
6. Preference

Three weeks of that yields the actual error distribution. The feedback loop is currently being designed without knowing what it is correcting for. It is entirely possible the answer is "70% of edits are compliance fixes," in which case the thinking layer solves nothing and a deterministic validator solves everything.

### Q3. Hybrid inputs (competitor video + custom brief twist)

**The question is framed wrong.** "How should the layer weight competitor beats versus the brief's scenario" assumes a blend dial. A blend dial produces mush.

The two inputs are orthogonal. Compose them, do not weight them:

| Input | Supplies |
| :--- | :--- |
| **Reference video** | **Structure.** Beat timing, shot grammar, split-screen rhythm, the exact second the pivot lands. |
| **Concept brief** | **Content.** Who, where, what the tension is, which mistake gets made. |

Put both in the blueprint schema as separate fields. Force the model to lay out the reference's beat timing explicitly before writing a single line of the new scenario.

**The only real conflict case** is when the new scenario physically cannot carry the reference's structure (reference needs two people in frame, brief has a protagonist alone). Make the model flag that and escalate rather than silently compromising. Silent compromise is how a script ends up being neither thing.

---

## Revised build order

| # | Item | Effort | Why |
| :--- | :--- | :--- | :--- |
| 1 | **Error taxonomy + eval harness.** Reason-coded edits, three weeks of data, a held-out set of 10 briefs never trained on. | 3 days | Unblocks every decision below. Nothing else should start first. |
| 2 | **Single-call blueprint step.** Prose not JSON, blueprint visible in the Doc. | 1 day | Core of the actual idea, and the only part that clearly earns its place. |
| 3 | **10 to 15 gold examples selected by CAC**, hand-tagged by archetype, retrieved by tag. | 2 days | Replaces the vector bank at current data volume. |
| 4 | **Hook variant generator.** Ten hooks per brief, human picks one. | 0.5 day | Likely the best ROI in the entire project. |
| 5 | **Approved finals append to the bank** behind a manual curation gate, with CAC backfilled once the ad runs. | Ongoing | Replaces Component C entirely. |
| 6 | **Everything in Section 4: not yet.** | n/a | See below. |

### On Section 4 specifically

- **DSPy:** revisit at 200+ CAC-labelled examples and a metric that correlates with performance. Not before.
- **TextGrad:** drop. It optimizes against the same unvalidated judge signal, with more machinery.
- **Mem0 / LangMem:** drop. Per-writer memory entrenches idiosyncrasy.
- **Gemini SFT:** at 50 to 100 pairs it will learn the format, which the prompt already handles, and will bake in a style precisely when the winning style is shifting.

---

## Closing note on the document itself

The brief is written to be approved, not to be interrogated. Mermaid diagram, repo handles, "cognitive middleware," a four-row tooling table where three rows do not survive contact with the actual data volume.

That presentation will get it signed off on the strength of the diagram rather than the reasoning, and three weeks will go into building a RAG layer for fifty documents.

Strip it to the two components that earn their place, the blueprint step and the example bank, and defend those.

---

## Open questions to answer before building

1. What is the actual distribution of human edits by reason code?
2. Which existing scripts beat their language average on the five CAC metrics, and how many are there?
3. Has first-half versus second-half USP placement ever been A/B tested against CAC?
4. What share of reviewer edits are deterministic compliance fixes that a validator could catch instead?
5. What is the exploration budget, meaning what fraction of weekly scripts are deliberately unlike the current winners?
