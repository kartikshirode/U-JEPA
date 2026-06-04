# Pipeline 1 — Build Plan
## Probe-Gated Latent-Memory Continual Learner (a JEPA-based continual-learning architecture)

**What we're building.** A frozen small model (the stable "world-model core") that never moves, with new facts living in an external memory you can write to and delete cleanly. Two separate gates decide what gets in — a *plausibility* gate (does this cohere?) and a *truth* gate (is this real?) — and a *probe* watches every merge so that a bad one is caught and **rolled back automatically**. SEAL is this loop without the safety; our whole contribution is making it safe and making it run on free compute.

**The budget reality.** The real atomic unit is not the week, it's the **9-hour session**. 30 h/week ≈ **3 good sessions/week**. Every phase below is sized in sessions, and every phase must save its state so the next session can resume. Calendar weeks are a guide; the *gates* are what actually decide when you move on.

---

### Legend — how uncertainty is marked

- **[SOLID]** — known recipe, low risk. Just execute.
- **[EXPERIMENTAL]** — we have an approach but the *outcome* is genuinely uncertain. The deliverable is a *measurement*, and the measurement might say "this doesn't work" — which is still a result.
- **[OPEN FORK]** — we don't yet know the right approach. **Resolve with a timeboxed spike before building anything on top of it.**

---

### Standing constraints (apply to every single session)

1. **Checkpoint or die.** The 9 h cap is hard. Phase 0 builds a save-state-to-Kaggle-Dataset / reload-next-session pattern *once*; never break it. Model weights are frozen so they don't need saving, but the **memory store, codebook, probe sets, thresholds, and results logs** all must persist.
2. **fp16, not bf16.** The T4 is Turing — no native bf16. Use NF4/QLoRA for anything you fine-tune, `batch_size=1` + gradient accumulation, gradient checkpointing.
3. **The core stays frozen.** If you ever feel tempted to train the core, stop — that's the entire point of the architecture, and it won't fit the T4 anyway.
4. **Keep a "bad-merge" injection set from day one.** You cannot measure the contribution without deliberately broken facts to catch. Start collecting/writing them in Phase 0.

---

### The one result that is the paper (put this on a sticky note)

> Of **K** fact-merges (a mix of good and deliberately-bad ones), what fraction of the **bad** merges did the probe catch and the system roll back — and what was the **false-abort rate** on the good ones? Plus the ROC curve as you slide the threshold.

Everything below is scaffolding for *that table* and its ablations. If a task doesn't eventually feed this table or an ablation of it, it's a distraction.

---

## Open forks to resolve EARLY (before they block you)

These are the parts we are honestly not sure about. Each has a cheap way to settle it. Do these spikes inside Phases 0–1; don't let them lurk.

| Fork | Options | How to resolve | When |
|---|---|---|---|
| **F1. Editing substrate** | EasyEdit-GRACE vs hand-rolled codebook vs WISE | 1-day spike: run EasyEdit's GRACE on GPT-2-XL. If it fights you, hand-roll the codebook — it's ~100 lines (cached activation = key, learned vector = value, ε deferral radius). | Phase 0 |
| **F2. How facts are injected** | text-RAG (paste retrieved text into the prompt — reliable) vs **latent injection** (inject KV/soft-prompt vectors — novel, JEPA-flavored, but possibly hollow per the Coconut finding) | **Decision already made:** build text-RAG first as the working baseline. Treat latent injection as a *gated upgrade* (Phase 1b) that you keep **only if its own probe shows it adds signal.** | Phase 1 |
| **F3. Plausibility scorer** | off-the-shelf NLI alone vs NLI + a JEPA latent scorer | Start NLI-only (cheap, interpretable, outputs entail/neutral/contradict directly). Add the JEPA scorer only if NLI's discrimination is weak. | Phase 2 |
| **F4. Core model** | GPT-2-XL (1.5B — fits EasyEdit baselines on T4) vs Qwen2.5-1.5B-Instruct (better core, Apache-2.0) | Use **GPT-2-XL for baseline reproduction** (comparability with the editing literature) and **Qwen2.5-1.5B for the actual system.** Report both if time allows. | Phase 0–1 |

---

## Phase 0 — Foundations & baseline reproduction
**Weeks ~1 · ~3 sessions · [SOLID]**

**Goal.** A reliable Kaggle notebook, the checkpoint/resume pattern, and a reproduced editing baseline so you have a sanity anchor and a number to beat.

**Build.**
- Kaggle env: load GPT-2-XL and Qwen2.5-1.5B in fp16/NF4; confirm VRAM headroom.
- The persistence pattern (save state → Kaggle Dataset → reload). Test it by killing and resuming a run.
- Reproduce a **GRACE** baseline on **zsRE** and **CounterFact**: N sequential edits, report edit-success / locality / held-out perplexity. (Resolves **F1**.)
- Stand up the **EasyEdit** harness *or* your hand-rolled codebook.

**Acceptance gate.** GRACE's edit-success and locality land roughly in the paper's range, and your perplexity probe runs. Resume-after-kill works.

**If it fails.** EasyEdit too heavy → hand-roll the codebook. Numbers far off → you have a harness bug, not a research problem; fix before continuing.

---

## Phase 1 — Core + memory loop (no gate yet)
**Weeks ~2–3 · ~5–6 sessions**

### 1a — Text-RAG store/retrieve/answer · [SOLID]
**Goal.** The skeleton: frozen Qwen core + a vector memory + retrieval + **text** injection. "Store a fact → retrieve it → answer correctly," with *no* gate, probe, or rollback yet.

**Build.** Fact representation (start with natural-language sentences; triples optional later — this is a minor **[OPEN FORK]**, default to sentences). Embed + store + retrieve. Inject retrieved text into the prompt. Measure recall on a small fact set.

**Gate.** Stored facts are reliably retrieved and answered; un-stored facts are unaffected (locality intact).

### 1b — Latent-injection upgrade · [EXPERIMENTAL] (was OPEN FORK F2)
**Goal.** Try injecting facts as latent vectors (KV / soft-prompt) instead of text. This is the part that earns the "JEPA / latent" framing — but it might not beat text-RAG.

**Build.** A small latent-injection path; a mini-probe comparing latent-injection answer quality vs text-RAG on the same facts.

**Gate / decision.** Keep latent injection **only if** it matches or beats text-RAG *and* survives the Phase 2 probe. Otherwise: text-RAG is your system, and "latent injection didn't add signal" is an honest ablation in the paper. **Do not let latent injection block the project** — that's exactly the "relocate the hard part into an assumed component" trap.

---

## Phase 2 — The probe / error signal (the heart)
**Weeks ~3–4 · ~4 sessions · [EXPERIMENTAL]**

**Goal.** Build the held-out probe suite that detects when a merge has degraded the model. This is the core novelty; it is genuinely empirical, so the plan is *build all three signals, measure each, keep what discriminates.*

**Build the three signals.**
1. **Perplexity probe** — held-out WikiText; perplexity must not rise past threshold τ.
2. **Locality/retention probe** — a fixed QA set covering previously-known facts/skills; count answer flips.
3. **Consistency probe** — NLI/entailment AUROC on a battery of (premise → consequence) pairs the core should rate plausible/implausible. (Resolves **F3**.)

Then inject your **deliberately-bad facts** and measure whether each signal moves.

**Acceptance gate (a real fork).** Combined **probe AUROC for "did this merge degrade the model?" ≥ ~0.7.**
- **≥ 0.7:** proceed to Phase 3.
- **< 0.7:** the probe is your bottleneck. **Stop and invest here** — add behavioral tests, calibrate the consistency scorer, weight the signals — before adding anything else. Do not paper over a weak probe by moving on.

**If it fails after real effort.** Fall back to perplexity-only gating and report the loop as an *evaluation harness* even if gating is imperfect. Still publishable, weaker claim.

---

## Phase 3 — Merge + auto-rollback (produces the table)
**Weeks ~4–5 · ~3 sessions · [SOLID mechanism, EXPERIMENTAL thresholds]**

**Goal.** Close the loop. Merge = memory write. After each merge, run the probe; if it trips thresholds, **auto-rollback** (delete the entry, or revert to the pre-merge snapshot).

**Build.** The merge→probe→decide→rollback controller. Snapshot-before-merge for O(1) revert. Run K merges (mixed good/bad) end-to-end **inside a single 9 h session**.

**Acceptance gate.** The system catches and reverts a clear majority of bad merges at a tolerable false-abort rate, and you can draw **the ROC curve** by sliding the threshold. **This is the paper's headline table** — get it clean.

**If it fails.** High false-abort → threshold/probe tuning. Misses bad merges → back to Phase 2's signals. Loop too slow for 9 h → cut K or use GPT-2-XL; checkpoint mid-run.

---

## Phase 4 — Truth gate + novelty-vs-plausibility (the second result)
**Weeks ~5–7 · ~5–6 sessions · [EXPERIMENTAL]**

**Goal.** Add the *second* gate and prove the plausibility/truth separation matters. The plausibility gate alone is fooled by good hoaxes and rejects genuine-but-weird truths; the source-trust gate rescues it.

**Build.**
- Intake gate: source whitelist + a FEVER-style verification step (retrieve evidence → NLI against it).
- Two curated test sets — **(a) genuine-but-implausible true facts** and **(b) plausible hoaxes.** (Constructing these is a small **[OPEN FORK]**: plan for hand-curation of a few dozen each; that's fine for a workshop paper.)
- The key experiment: plausibility-only vs plausibility+truth on both sets → the **novelty-vs-plausibility trade-off curve.**

**Acceptance gate.** Adding the truth gate measurably improves admission of (a) and rejection of (b) versus plausibility-only.

**If it fails.** Report it honestly — "plausibility gate alone behaves like X; the tension is irreducible at this scale." The research already told us this tension *cannot* be fully solved; measuring it is the contribution, not eliminating it.

---

## Phase 5 — Consolidation / "sleep" (stretch; reuses your O-LoRA)
**Weeks ~7–8 · ~3 sessions · [STRETCH / EXPERIMENTAL]**

**Goal.** Address the accumulation problem: periodically fold high-trust, stable facts out of memory into an **orthogonal-LoRA skills adapter** (your existing zero-forgetting code), so the memory store doesn't grow without bound.

**Build.** A consolidation pass that selects stable/high-trust facts, trains an O-LoRA adapter on them, verifies no forgetting via the Phase 2 probe, and prunes those entries from memory.

**Gate.** Post-consolidation, retention on the locality set stays within tolerance (your Phase 1 zero-forgetting result should carry over).

**If it fails / runs out of time.** Drop it — the core paper stands without consolidation. If it *does* forget, that's still a reportable finding and it directly extends your prior work. This phase is the bridge between this project and your existing N-LoRA result; include it if the calendar allows.

---

## Phase 6 — Full eval, ablations, seeds, writeup
**Weeks ~8–10 · ~6 sessions · [SOLID process]**

**Goal.** The full matrix + the paper.

**Build.**
- Benchmarks: zsRE, CounterFact, **MQuAKE** (ripple/multi-hop — your weakest-expected area, so measure it honestly), KnowEdit, FEVER, WikiText.
- **Ablations** (each should hurt a specific metric — this *is* the argument):
  - *no gate* (admit everything) → SEAL-like → degradation accumulates (reproduces the documented sequential-edit collapse).
  - *no probe* → can't catch bad merges.
  - *no rollback* → catches but can't recover.
  - *plausibility-only* (no truth gate) → fails on plausible hoaxes.
  - *latent vs text injection* (from Phase 1b).
- **Two seeds** for the final matrix (the research flagged single-seed as a real weakness; report mean ± SD). This doubles eval time — budget it.
- Draft the workshop paper around the loop, citing LLM-JEPA, WISE, GRACE, SEAL, and the model-editing-at-scale collapse result as the immediate context.

**Acceptance gate.** A coherent results section where removing each organ degrades a named metric, with variance bars. → submit.

---

## Timeline reality check (honest)

- These estimates are **soft**. ~8–10 weeks of focused part-time work to a draft is realistic, but expect at least one phase to slip — most likely **Phase 2** (the probe) or **Phase 4** (curating the hoax/implausible sets).
- **Gates beat calendar.** If a gate fails, the plan tells you whether to *push* (Phase 2 < 0.7 AUROC → invest more) or *pivot* (Phase 4 fails → report the tension). Don't march past a failed gate to stay on schedule.
- The **minimum publishable core** is Phases 0–3 + the ablations from Phase 6 that touch them. Phases 4–5 are strengthening, not load-bearing. If time collapses, ship the gated-rollback loop alone — it's already novel.

---

## Where it can go wrong → what you do (pivot map)

| Phase | Failure | Fallback |
|---|---|---|
| 0 | EasyEdit too heavy | hand-roll GRACE codebook |
| 1b | latent injection ≤ text-RAG | text-RAG is the system; latent becomes an honest ablation |
| 2 | probe AUROC < 0.7 | stop and improve the probe before anything else; worst case, perplexity-only gating |
| 3 | high false-abort / misses | threshold + probe tuning; cut K for the 9 h cap |
| 4 | truth gate doesn't help | report the irreducible novelty-vs-plausibility tension (a result) |
| 5 | consolidation forgets | drop it; report the finding; core paper unaffected |
| 6 | numbers thin | lead with the gated-rollback table; it stands alone |
