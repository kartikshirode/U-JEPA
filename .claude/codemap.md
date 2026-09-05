<!-- codemap-format: v1 -->

# Codemap: U-JEPA

## Overview

JEPA-based continual-learning research repo. Three generations live side by side.

- `v3/` is the **newest** line, started 2026-08-11 for a ground-up redesign targeting ICML 2027. Stages 0 (harness), 1 (RQ1) and 2 (the gate) are built and green on CPU; stage 2 is a tested mechanism carrying no numbers, since which signals survive is what RQ1 and RQ2 answer. RQ1 asks whether a poisoned entry in an automated maintenance feed survives the upstream correction that is supposed to remove it. `v3/spikes/q1_volatility/` asked whether facts sort into an invariant layer and a volatile one using time-stamped Wikidata. The answer was partial: the metric it measures is the composition of observed change, not volatility, so the two-layer design became a candidate gate feature rather than an architectural layer.
- `v2/` is the **built-out** project: a frozen small LLM core (GPT-2-XL, Qwen2.5-1.5B) plus an external editable memory, two intake gates (plausibility via NLI, truth via FEVER-style verification), and a probe that runs after every fact-merge and auto-rolls-back bad ones. Its headline claim (that a gated loop survives where ungated sequential editing collapses) was undercut in May 2026 by UltraEdit, which sustains 1M sequential edits ungated. Code and tests still green; the framing is what broke.
- `legacy/v1/` is **frozen** (2026-06-05, do not modify): orthogonal-LoRA continual learning + LLM-JEPA aux losses + SIGReg on a frozen NF4 Qwen3-14B. Phase 0 and 1 passed, Phase 2 failed both gates. Kept as reference and for reusable organs (the LoRA bank, the CL metrics).
- `results/` holds v1 phase result JSONs; `docs/` holds the cross-generation audit and the one ADR.

Stack: Python 3.10+ (v1 pinned 3.12), PyTorch, HuggingFace transformers. No PEFT in v1's bank (hand-rolled). v1 and v2 heavy runs go to Kaggle free-tier T4. v3 runs on the Baramati Slurm cluster, where the H200s are cut into 18 GB MIG slices; see `v3/docs/cluster-baramati.md`. The local RTX 4060 laptop runs unit tests only.

Two independent packages, each with its own `pyproject.toml` and test suite:

```
v2/     package u-jepa-v2, src/u_jepa_v2, tests/ (27 tests, 1 network-gated skip)
legacy/v1/  package u-jepa,  src/u_jepa,     tests/ (122 tests, network + lejepa skips)
```

Build and test:

```bash
cd v2 && pip install -e .[dev] && pytest         # active suite
cd legacy/v1 && pytest                           # frozen suite, still green
U_JEPA_V2_RUN_NETWORK=1 U_JEPA_V2_RUN_SLOW=1 pytest tests/test_models_core.py   # HF download tests
```

Kaggle notebooks are **generated** from Python builders (`v2/kaggle/build_*.py`), never hand-edited; a hand-written ipynb corrupted into invalid JSON cost v1 a push. Edit the builder, re-run it, then `kaggle kernels push` from the notebook folder.

Repo-wide gotchas:

- **fp16 everywhere, never bf16.** Kaggle T4 is Turing with no native bf16, and the code locks fp16 for parity. `v2/src/u_jepa_v2/env.preferred_dtype_str()` hardcodes this.
- **The core stays frozen.** Both generations train only adapters or write to external memory. Anything that unfreezes base weights is a design violation, not an optimization.
- **9 hour Kaggle session cap** is the real unit of work. Anything approaching it must persist resumable state through `v2/src/u_jepa_v2/persistence.py` and reload from `/kaggle/input/<slug>`.
- HF cache must not live in `/kaggle/working` (20 GB quota); notebooks force `/tmp/hf_cache`.
- `.gitignore` blocks `.claude/`, so this map is force-added. Result JSONs are ignored except the per-phase headline files whitelisted by name.
- Kaggle account drift: v1 phase0/phase1 kernels point at `chinmayishirode`, everything newer at `kartikshirode`. Check `~/.kaggle/kaggle.json` before pushing.

## Root

### README.md
Entry point: what U-JEPA is, the v1-to-v2 pivot, where to start reading, hardware contract, license (MIT code, upstream licenses for vendored/data).

### Research.md
Original May 2026 ideation doc that led to v1. Ranks two implementation paths for an 8 GB consumer GPU, recommends Approach 1 (fork LatentMAS, JEPA-regularize, N-LoRA adapters, V-JEPA vision bridge). Historical: the 8 GB framing was dropped by the Kaggle ADR and the whole architecture was replaced by v2.

### .gitignore
Ignores caches, venvs, credentials (kaggle.json, hf_token.txt, *.token), checkpoints, model weights, `.claude/`, and `results/*.json` except the explicitly whitelisted per-phase headline JSONs.

## docs/

### docs/external_audit_context.md
Full project snapshot at 2026-06-03 written for an external auditor: the research bet, phase plan with status, what actually ran in Phases 0-2, and a 14-point logic audit of why Phase 2 failed. Section 8 is the load-bearing part; 8.1 (JEPA target collapsed by design on Q/A pairs), 8.2 (SIGReg applied to the wrong axis), 8.7 (Spider without schema is the wrong benchmark) and 8.8 (n=200 underpowered) are the critical findings that motivated the v2 redesign.
Gotcha: path references inside are pre-archive (`src/u_jepa/...`); those files now live under `legacy/v1/`. The header notes this but the body was not rewritten.

### docs/superpowers/specs/2026-09-05-u-jepa-v3-design.md
The current v3 design. Starts from one named deployment, automated knowledge maintenance driven by a public feed, where the attacker can alter a share of upstream entries but not the gate, editor or model. RQ1 asks whether poison survives that pipeline and the upstream revert that corrects it, which is open because editing suppresses rather than erases. 5 research questions, staging with kill switches, non-goals, 9 rules traced to v1 audit failures.
Gotcha: supersedes the 2026-08-11 spec and `v2/docs/pipeline-1-build-plan.md`. Section 13 is the reviewer-attack list and 14 the open items; both are deliberate. No path bypasses verification, and the 78/20 accretion split is a WikiBigEdit statistic rather than a property of knowledge.

### docs/superpowers/specs/2026-08-11-u-jepa-v3-design.md
Superseded first draft of the v3 design, kept for history. Framed RQ1 as whether editors admit adversarial knowledge as readily as benign, which external review found tautological, and built a threat model from four papers that each assume a different attack surface.
Gotcha: carries a superseded banner. Do not implement from it. Disposition table in `docs/reviews/2026-09-05-external-review-v3.md`.

### docs/superpowers/plans/2026-09-05-u-jepa-v3-harness-and-rq1.md
Current TDD plan for v3 stages 0-1: 14 tasks, 71 steps, every failing test written out. Builds `v3/src/u_jepa_v3/` (env, schema, corpora, a feed simulator carrying poison and its upstream corrections, editor protocol over EasyEdit, efficacy, elicitation and downstream-harm probes, atomic cell state, shard worker, RQ1 driver and analysis).
Gotcha: cells are atomic on purpose, since weights and editor normalization state are never checkpointed and a mid-cell resume would continue from the wrong model. Editors own `responder()` so a probe cannot read an unedited model. Everything stays CPU-testable through `StubEditor`; network and GPU tests gate on `U_JEPA_V3_RUN_NETWORK=1` and `U_JEPA_V3_RUN_GPU=1`. Probe sets load from `U_JEPA_V3_PROBE_DIR` and raise a named error until built.

### docs/superpowers/plans/2026-08-11-u-jepa-v3-harness-and-rq1.md
Superseded first implementation plan, kept for history. Its EasyEdit adapter discarded the model `edit()` returns, its resume continued from the wrong model, its worker had no run path, and its 3 attack families ran identical code.
Gotcha: carries a superseded banner. Do not implement from it.

### docs/reviews/2026-09-05-external-review-v3.md
Disposition record for the external review that reshaped v3. Holds the 3 framing challenges (tautological RQ1, overstated SSGM relationship, 78/20 treated as a property of knowledge), 10 findings with what each one changed, the 5 defects verified against the files, and where the fix was scoped narrower than recommended.
Gotcha: records the recurring pattern that both collapsed framings were derived from literature rather than a deployment. Read before questioning why the v3 spec is shaped the way it is.

### docs/decisions/2026-05-26-kaggle-pivot.md
The only ADR. Moves heavy compute from the RTX 4060 laptop to Kaggle GPUs, drops the 8 GB VRAM ceiling, bumps the base model to Qwen3-14B. Reasons: vLLM is Linux-only, the 8 GB cap forced a weaker 4B baseline, Kaggle gives free Linux T4s at 30 h/week. Introduces the 9-hour-session and checkpoint-or-die constraints that both generations still live under.

## v3/ (newest)

### v3/README.md
Entry point for v3: what RQ1 and RQ2 ask, install and test, the first hour on the cluster, how to run a grid, and the four design choices that look like bugs and are not (atomic cells, editor-owned responders, gate input redaction, attacker cover traffic).

### v3/pyproject.toml
Package `u-jepa-v3`, src layout, Python 3.12+. `[dev]` adds pytest, `[edit]` adds easyeditor, `[probes]` adds datasets for the login-node probe builder. Testpaths point at `tests/`.

### v3/scripts/00_smoke_gpu.py
First thing to run on the cluster. Reports the Slurm context, the visible devices and whether they look like MIG slices, the derived dtype, the memory arithmetic for each planned arm, peer access, and whether easyeditor and the probe sets are there.
Exports: PLANNED (the model, method arms the pilot intends); main() -> int
Used by: v3/slurm/00_smoke.slurm
Gotcha: exits 2 with no CUDA and 1 when easyeditor or the probe sets are missing, so a job script can gate on it. The fit table is what tells you an 8B arm cannot run here.

### v3/scripts/01_build_probes.py
Builds the 5 pinned probe sets from HuggingFace datasets plus WikiBigEdit's own loc/loc_ans columns. Login node only, since compute nodes have no network.
Exports: SOURCES; BUILDERS; build_sst/build_mmlu/build_mrpc/build_nli(rows); build_locality(n, seed); sample(pairs, n, seed); write_set(out_dir, name, pairs); main(argv)
Gotcha: writes a manifest with a sha256 per set. Build once and never regenerate, because a probe set that shifts between cells makes every cross-cell comparison meaningless and does it silently.

### v3/scripts/02_power.py
CLI over `power.py`: how many corrected poison items an arm needs, the power curve at other sizes, and the grid parameters that deliver it.
Exports: main(argv)
Gotcha: every input is an assumption, and it says so. Re-run it with the pilot's own discordant proportion before any number is reported.

### v3/scripts/03_prefetch.py
Pulls model weights, tokenizers and the WikiBigEdit files into a shared HF cache. Login node only.
Exports: prefetch_model(name); prefetch_dataset(); main(argv)
Gotcha: warns when HF_HOME points somewhere node-local. Compute nodes run with HF_HUB_OFFLINE=1, so anything missed here fails several minutes into a queue slot.

### v3/scripts/04_check_hparams.py
Validates the hparams YAML against the installed EasyEdit and the real model config: the HyperParams class has to build, layer indices have to be in range, module templates have to name modules that exist.
Exports: TEMPLATE_KEYS; load_yaml(path); check_builds(path, alg); check_layers(raw, n_layers); check_modules(raw, model_name); check_file(path, skip_model); main(argv)
Gotcha: the files it checks were written from published templates on a laptop with no EasyEdit installed. Until this passes, they are a guess.

### v3/grids/rq1_pilot.json
RQ1 pilot: Llama-3.2-3B, 3 editors, all 3 attack families, 3 seeds, 1000 benign and 160 poison at a 0.25 base rate.
Gotcha: sized by `scripts/02_power.py` for 120 corrected poison items per arm at a 0.15 gap. No `hparams` key, because the path is derived from the editor and model; listing it as a dimension would pair every editor with every file.

### v3/grids/rq1_scale.json
The 10K-edit version: 2 model families, 4 editors, 3 attack families, 5 seeds, 2 revert lags.
Gotcha: needs the Qwen hparams validated too. 480 cells, so check the shard split before submitting.

### v3/grids/rq2_pilot.json
The gated arm. Same shape as the RQ1 pilot plus `calibrate_on`, which names the attack families the thresholds are fitted on.
Gotcha: `arm` is `rq2`, which is what routes the cell to the gate driver. Cells whose evaluation family appears in `calibrate_on` are the ceiling, not the transfer result; `GateSummary.held_out` separates them.

### v3/src/u_jepa_v3/env.py
Device, dtype and run-directory resolution. Dtype is derived from compute capability rather than hardcoded, which is what v1 and v2 both got wrong for the Kaggle T4.
Exports: VALID_DTYPES; has_native_bf16(capability); preferred_dtype_str(capability=None); device_capability(index=0); run_root(); EnvSummary; summarize()
Used by: v3/src/u_jepa_v3/experiments/rq1_survival.py, v3/scripts/00_smoke_gpu.py
Gotcha: `U_JEPA_V3_DTYPE` overrides and raises on an unknown value. No capability (CPU) means fp32, not fp16.

### v3/src/u_jepa_v3/cluster.py
What the target cluster offers and whether an arm fits it. MIG slice budgets, model specs, per-method editor memory, Slurm context and a device report.
Exports: BYTES_PER_PARAM; SLICE_GB; USABLE_FRACTION; CUDA_CONTEXT_GB; ModelSpec; KNOWN_MODELS; MODEL_SLUG; hparams_slug(model); EDIT_LAYERS; COVARIANCE_MULTIPLIER; FitReport; weights_gb(); editor_overhead_gb(); slice_budget_gb(); plan_fit(); SlurmContext; slurm_context(environ=None); device_report()
Used by: v3/src/u_jepa_v3/runs/worker.py, v3/src/u_jepa_v3/experiments/rq1_survival.py, v3/scripts/00_smoke_gpu.py, v3/scripts/03_prefetch.py
Gotcha: the slice is the schedulable unit, so the per-job cap is 18 GB and not 141. MEMIT and AlphaEdit hold an intermediate-squared fp32 covariance per edited layer, which is what puts an 8B banded arm out of reach here. Every number is an estimate with a stated basis until a real arm measures it.

### v3/src/u_jepa_v3/power.py
Sample sizes and intervals. McNemar for the paired survival gap, two-proportion for unpaired arms, precision at a deployment base rate, Wilson intervals, and the grid parameters that deliver a required item count.
Exports: z(p); PairedPlan; mcnemar_sample_size(); mcnemar_power(); two_proportion_sample_size(); precision_at_base_rate(); wilson_interval(); FeedPlan; feed_plan()
Used by: v3/src/u_jepa_v3/gate/combiner.py, v3/src/u_jepa_v3/experiments/rq2_gate.py, v3/scripts/02_power.py
Gotcha: the survival gap is two rates on the same items, so power is driven by the discordant proportion and not by the rates. No scipy; quantiles come from statistics.NormalDist.

### v3/src/u_jepa_v3/schema.py
The shared vocabulary every corpus normalises into, plus the feed entry that wraps a candidate with its position and its relationship to other entries.
Exports: EditKind (ACCRETION, REVISION); Decision (ADMIT, REFUSE, QUARANTINE); EditCandidate frozen dataclass with `.key`; FeedEntry frozen dataclass; ApplyResult
Used by: v3/src/u_jepa_v3/data/*, v3/src/u_jepa_v3/editors/*, v3/src/u_jepa_v3/probes/*, v3/src/u_jepa_v3/gate/*, v3/src/u_jepa_v3/experiments/rq1_survival.py, v3/src/u_jepa_v3/experiments/rq2_gate.py
Gotcha: validation is in `__post_init__` and is load-bearing. Adversarial candidates need a risk_category and benign ones must not carry one; poison entries need an attack_family and cannot also be a revert.

### v3/src/u_jepa_v3/data/wikibigedit.py
Benign corpus from 8 Wikidata snapshot diffs (2024-02-01 to 2024-07-01), normalised into EditCandidate.
Exports: REPO_ID; TIMESTEP_FILES; TAG_TO_KIND; load_raw(); to_candidates(frame); sample_candidates(candidates, n, seed); load_candidates(n=None, seed=0)
Used by: v3/src/u_jepa_v3/experiments/rq1_survival.py, v3/src/u_jepa_v3/experiments/rq2_gate.py, v3/scripts/01_build_probes.py, v3/scripts/03_prefetch.py
Gotcha: TIMESTEP_FILES order defines the timestep index, so never sort it. Sampling is seeded and uniform, never a sorted prefix, because a prefix biases toward low Q-ids and makes every seed identical.

### v3/src/u_jepa_v3/data/relation_prior.py
Per-relation `update_share` and concentration, offered to the gate as candidate features.
Exports: DEFAULT_THRESHOLD (0.1); DEFAULT_MIN_SUPPORT (200); RelationStats; RelationPrior with .from_candidates/.update_share/.is_low/.stats/.coverage
Used by: v3/src/u_jepa_v3/gate/base.py, v3/src/u_jepa_v3/experiments/rq2_gate.py
Gotcha: update_share is the composition of observed change, NOT volatility and NOT the probability a fact changes, because the denominator holds only rows that already changed. Whether it helps a decision is RQ3 and the answer may be no.

### v3/src/u_jepa_v3/data/adversarial.py
Poisoned entries in 3 families that differ by mechanism, each returning matched (original, poisoned) pairs.
Exports: RISK_CATEGORIES; AttackFamily (OBJECT_SWAP, TYPE_CONSISTENT, TEMPORAL_STALE); poison_object_swap(); poison_type_consistent(); build_history(); poison_temporal_stale(); load_editrisk(path)
Used by: v3/src/u_jepa_v3/experiments/rq1_survival.py, v3/src/u_jepa_v3/experiments/rq2_gate.py
Gotcha: matched pairs are the point, since comparing real benign additions against synthetic malicious revisions would measure dataset difficulty rather than security. TEMPORAL_STALE needs a slot that changed twice (913 of 99,404 in Q1) and raises rather than padding.

### v3/src/u_jepa_v3/data/feed.py
The maintenance feed: benign stream, poison at a configurable base rate, and the upstream correction that follows each poison at a lag.
Exports: build_feed(benign, poison_pairs, base_rate, revert_lag, seed); poison_entries(feed); reverted_by(feed); poison_state(feed, upto)
Used by: v3/src/u_jepa_v3/experiments/rq1_survival.py, v3/src/u_jepa_v3/experiments/rq2_gate.py
Gotcha: revert_lag counts benign entries, so the gap in feed positions is lag + 1 because the poison occupies a position of its own. `poison_state` splits on whether the correction has been reached, which is the RQ1 measurement.

### v3/src/u_jepa_v3/editors/base.py
The editor and responder protocols. Editors expose `responder()` rather than accepting one.
Exports: Responder Protocol (answer); Editor Protocol (name, apply, responder)
Used by: v3/src/u_jepa_v3/editors/stub.py, v3/src/u_jepa_v3/editors/easyedit_adapter.py, v3/src/u_jepa_v3/probes/*, v3/src/u_jepa_v3/gate/rollback.py, v3/src/u_jepa_v3/experiments/rq1_survival.py
Gotcha: the ownership direction is deliberate. An earlier design let a caller hold a responder never bound to the edited model, so probes read the untouched base for whole runs and every test passed.

### v3/src/u_jepa_v3/editors/stub.py
Editor that records instead of editing, so the whole harness tests on CPU.
Exports: UNEDITED ("<unedited>"); StubEditor(fail_keys=None) with .applied, .apply, .responder
Used by: every v3 test module
Gotcha: its responder answers from the edits it accepted, on purpose. A stub whose responder ignored edits would reproduce the exact bug this design exists to prevent, invisibly.

### v3/src/u_jepa_v3/editors/easyedit_adapter.py
Wraps EasyEdit BaseEditor so every method looks identical from above, and keeps the model `edit()` returns.
Exports: SUPPORTED_METHODS (ultraedit, alphaedit, rome, memit, wise, grace); HFResponder(model, tokenizer, max_new_tokens); EasyEditAdapter(method, hparams_path, sequential=True) with .edited_model, .apply, .responder, .to_easyedit_payload
Gotcha: keeping `edited_model` is the whole job; dropping it means probes read the base and sequential edits restart from it. `responder()` raises before the first successful edit rather than falling back to the base. The easyeditor import is deferred so payload tests run without CUDA.

### v3/src/u_jepa_v3/editors/registry.py
name -> Editor factory, so grids can name editors as plain strings.
Exports: register(name, factory); available(); build(name, **kwargs); register_defaults()
Used by: v3/src/u_jepa_v3/experiments/rq1_survival.py, v3/src/u_jepa_v3/experiments/rq2_gate.py

### v3/src/u_jepa_v3/probes/efficacy.py
Did the edit take, and did unrelated neighbours survive.
Exports: normalize_answer(text); efficacy(responder, candidates); locality(responder, pairs)
Used by: v3/src/u_jepa_v3/probes/general_ability.py, v3/src/u_jepa_v3/probes/elicitation.py, v3/src/u_jepa_v3/probes/downstream.py, v3/src/u_jepa_v3/gate/signals.py, v3/src/u_jepa_v3/gate/rollback.py, v3/src/u_jepa_v3/experiments/rq1_survival.py
Gotcha: answers are normalised (case, punctuation, articles) before comparison, because exact match on raw generation measures formatting rather than knowledge.

### v3/src/u_jepa_v3/probes/general_ability.py
SST, MMLU, MRPC and NLI, matching UltraEdit's own evaluation set so numbers sit beside theirs.
Exports: REQUIRED_SUITES; GeneralAbility dataclass with .mean; general_ability(responder, suites)
Used by: v3/src/u_jepa_v3/experiments/rq1_survival.py, v3/src/u_jepa_v3/experiments/rq2_gate.py
Gotcha: raises when any of the four suites is missing rather than averaging over what it has. This is the stealth detector: flat here plus corrupted knowledge is the dangerous case.

### v3/src/u_jepa_v3/probes/elicitation.py
Is a corrected fact gone, or only hidden. Three pressure levels, reported separately.
Exports: ELICITATION_MODES (direct, paraphrase, leading); paraphrases(candidate); leading_contexts(candidate); elicitation_rate(responder, poisoned, mode)
Used by: v3/src/u_jepa_v3/gate/rollback.py, v3/src/u_jepa_v3/experiments/rq1_survival.py
Gotcha: the gap between direct and leading is the RQ1 result, not any single mode. A candidate counts as elicited on a hit from any probe in the mode.

### v3/src/u_jepa_v3/probes/downstream.py
Does surviving poison move answers that depend on it. Locality asks about unrelated facts; this asks about dependent ones.
Exports: DownstreamHarm dataclass (n_questions, corrupted, poisoned_answer); downstream_harm(responder, hop_questions)
Used by: v3/src/u_jepa_v3/experiments/rq1_survival.py
Gotcha: takes (prompt, true_answer, poison_implied_answer) triples. `corrupted` is damage and `poisoned_answer` is targeted control; reporting only one hides the gap EditRisk-Bench found between single-hop and multi-hop success.

### v3/src/u_jepa_v3/runs/state.py
Per-cell state, written atomically via temp file plus rename.
Exports: RunState (cell_id, checkpoints, finished, meta); save(state, path); load(path); is_finished(path)
Used by: v3/src/u_jepa_v3/runs/grid.py, v3/src/u_jepa_v3/runs/worker.py, v3/src/u_jepa_v3/experiments/rq1_survival.py, v3/src/u_jepa_v3/experiments/rq2_gate.py
Gotcha: carries no resume counter on purpose. Weights, editor normalization state and RNG state are never checkpointed, so a mid-cell resume would continue from the wrong model. Serialisation happens before the temp file opens, so an unserialisable payload cannot destroy the previous good state.

### v3/src/u_jepa_v3/runs/grid.py
Cartesian expansion, a stable content-hashed cell id, interleaved sharding and pending detection.
Exports: Cell frozen dataclass with .cell_id; expand(grid); shard(cells, node, of); pending(cells, out_dir)
Used by: v3/src/u_jepa_v3/runs/worker.py
Gotcha: sharding is `index % of`, not contiguous blocks, so an unbalanced grid spreads evenly. cell_id is order-independent because params are canonicalised with sorted keys.

### v3/src/u_jepa_v3/runs/worker.py
CLI running one shard of a grid, skipping finished cells. Under a Slurm array the shard coordinates come from the array task.
Exports: ARMS (rq1, rq2); run_cell(cell, out_dir, runner); resolve_shard(node, of, ctx); main(argv)
Used by: v3/slurm/worker_array.slurm
Gotcha: it leaves CUDA_VISIBLE_DEVICES alone under Slurm, because the allocation has already narrowed it to the granted MIG slice and overwriting it with the shard index makes the process see no device at all. The `arm` key routes to the rq1 or rq2 driver and an unknown value raises. `run_cell` never raises; a dead cell is written unfinished with its error and reruns from zero next pass.

### v3/src/u_jepa_v3/gate/__init__.py
Re-exports the stage 2 surface. Imports provenance before base, because base depends on it.

### v3/src/u_jepa_v3/gate/base.py
What the gate is allowed to see, plus the signal protocol and the decision types.
Exports: GateInput frozen dataclass with .from_entry(entry, source) and .key; GateContext (prior, trust, belief, window, object_vocab, slot_values, slot_writes) with .prime/.observe/.recent_from_source/.recent_for_subject; Signal Protocol; GateScore; GateDecision; is_revision(gate_input, ctx)
Used by: v3/src/u_jepa_v3/gate/signals.py, v3/src/u_jepa_v3/gate/combiner.py, v3/src/u_jepa_v3/experiments/rq2_gate.py
Gotcha: `from_entry` is a redaction, not a wrapper. It drops is_poison and attack_family and overwrites candidate.source, because the attack generators write the family name into source and a signal reading it would score perfectly and mean nothing. `observe` must be called on admitted entries only, or a refused attacker teaches the vocabulary that poison is normal. `is_revision` uses what the gate has seen, falling back to the corpus tag only for unseen slots.

### v3/src/u_jepa_v3/gate/provenance.py
Who submitted the claim and how much that account has earned, plus the simulation behind it.
Exports: DEFAULT_PRIOR_TRUST; DEFAULT_PRIOR_WEIGHT; SourceRecord; SourceTrust(prior_trust, prior_weight) with .observe/.trust/.record/.known; TrustTracker(trust, lag) with .submitted/.reverted/.advance/.pending; simulate_sources(feed, seed, n_sources, n_attacker_sources, cover_rate); attacker_sources()
Used by: v3/src/u_jepa_v3/gate/base.py, v3/src/u_jepa_v3/experiments/rq2_gate.py
Gotcha: cover_rate is load-bearing. At 0 the attacker account becomes a perfect label and any gate scores perfectly on an artefact of the simulation. TrustTracker credits an entry only after it survives `lag` positions, so an attacker cannot earn trust from the very entries being measured.

### v3/src/u_jepa_v3/gate/signals.py
The six suspicion scores, each in [0, 1] and each batched.
Exports: ABSTAIN (0.5); UNSEEN_OBJECT (0.15); TypeViolationSignal; PriorMismatchSignal; SourceTrustSignal; BurstSignal(cap=25); SlotChurnSignal(cap=3); BeliefContradictionSignal; default_signals(with_belief=False)
Used by: v3/src/u_jepa_v3/experiments/rq2_gate.py
Gotcha: none of these is a fact checker and every one fires on legitimate edits. The module docstring records the prediction before the numbers exist: object swap should be reachable by type, type-consistent needs the belief and stream signals, and temporal stale asserts a value the slot genuinely held so nothing here can call it false. BeliefContradictionSignal raises without a model rather than scoring zero.

### v3/src/u_jepa_v3/gate/combiner.py
Weighted mean of the signals, the three-way decision, and threshold fitting against a deployment base rate.
Exports: DEFAULT_REFUSE_AT; DEFAULT_QUARANTINE_AT; Thresholds; LinearCombiner(signals, weights=None, thresholds=None) with .score/.decide/.with_thresholds; OperatingPoint; Calibration; sweep(scores, labels, base_rate); calibrate(scores, labels, base_rate, target_precision, quarantine_recall)
Used by: v3/src/u_jepa_v3/experiments/rq2_gate.py
Gotcha: precision is computed at the deployment prevalence, not at the calibration sample's. A gate reported by balanced AUROC can look excellent and refuse mostly good edits in production. A target that cannot be met comes back with `met_target` False and the best available point, never a threshold that quietly misses it.

### v3/src/u_jepa_v3/gate/rollback.py
The ledger of admitted edits, and what replaying it without the bad ones costs.
Exports: DEFAULT_BATCH; LedgerEntry; ShadowLedger with .record/.replay_plan/.position_of/.cost_of_dropping; RollbackAudit with .seconds_per_edit; audit_rollback(editor_factory, ledger, drop_ids, poisoned, benign_sample, batch_size)
Gotcha: takes a factory, not an editor, because auditing the model that already carries the poison measures nothing. Replay starts from the base, so the cost is the whole ledger less the drops rather than everything after the poison. Residual elicitation should be zero and is measured anyway, as a check that the replay reproduces a clean model.

### v3/src/u_jepa_v3/experiments/rq1_survival.py
The RQ1 driver. Streams a feed through one editor and probes at intervals, recording uncorrected poison, corrected poison at three elicitation pressures, downstream harm, locality and general ability.
Exports: PROBE_DIR_ENV; HPARAMS_DIR_ENV; Rq1Config; checkpoint_metrics(..., responder=None); run_arm(editor, feed, suites, locality_pairs, base_responder, config, hop_questions=None); resolve_hparams(params); check_fits(params); run_cell_from_params(params); _load_base, _load_suites, _load_locality
Used by: v3/src/u_jepa_v3/runs/worker.py, v3/src/u_jepa_v3/experiments/rq2_gate.py
Gotcha: the untouched-base arm is measured once through `base_responder` before anything is applied and stored on `meta["baseline_general"]`. Probe sets load from `U_JEPA_V3_PROBE_DIR` and raise a named error until built. `resolve_hparams` derives the path from editor and model rather than taking it as a grid dimension, which would pair every editor with every file. `check_fits` refuses an arm too large for the device before anything loads.

### v3/src/u_jepa_v3/experiments/rq1_analysis.py
Collapses per-seed cell states into arm summaries and the reported numbers.
Exports: GROUP_KEYS (model, editor, attack_family, base_rate, at); ArmSummary; summarize(states); survival_gap(summary); is_stealthy(summary, tolerance=0.02); curve(summaries, model, editor, family)
Used by: v3/src/u_jepa_v3/experiments/rq2_analysis.py
Gotcha: every experimental dimension stays a grouping key and only seeds collapse, into mean and SAMPLE standard deviation. An earlier version grouped by editor and corpus alone, which made the 1K/10K/100K curves unobtainable from its own output. `survival_gap` is leading minus direct elicitation and is the headline.

### v3/src/u_jepa_v3/experiments/rq2_gate.py
The RQ2 driver. Same feed, editor and probes as RQ1, with entries passing the gate first, so the two arms subtract.
Exports: Rq2Config (adds trust_lag, calibrated_on, deployment_prevalence); GateCounts; collect_calibration(combiner, feed, sources, ctx); run_gated_arm(...); run_cell_from_params(params)
Used by: v3/src/u_jepa_v3/runs/worker.py
Gotcha: poison rates stay over every poison entry the feed reached, not over the admitted ones, because an operator cares how many attacks landed and not how many of the ones they let through worked. Quarantine blocks, so it counts as caught for the model and as cost for the review queue. deployment_prevalence is not base_rate: base_rate is how much of the attack pool this run injects, prevalence is what a real feed carries. The belief signal is repointed at the edited model each batch, since a gate asking the base model is asking a stale snapshot.

### v3/src/u_jepa_v3/experiments/rq2_analysis.py
Collapses gated cells into what the gate blocked and what it destroyed, then subtracts the matching undefended arm.
Exports: GROUP_KEYS (model, editor, attack_family, calibrated_on, at); GateSummary with .held_out; summarize_gated(states); NetBenefit with .worth_it; net_benefit(gated, ungated); pair_arms(gated, ungated)
Gotcha: calibrated_on is a comma separated list, so `held_out` tests membership rather than string equality; comparing it whole would mark every multi-family calibration as transfer including the ones containing the evaluation family. Unmatched gated arms are dropped rather than compared against a default, since an arm with no control is not a weaker result but not a result.

### v3/tests/test_schema.py
EditCandidate and FeedEntry validation: blank required fields, the adversarial and risk_category pairing, and that a poison entry cannot also be a revert.

### v3/tests/test_wikibigedit.py
Normalisation into EditCandidate, dropping untagged and unkeyable rows, and that seeded sampling differs between seeds instead of returning a sorted prefix.

### v3/tests/test_relation_prior.py
update_share and concentration over a synthetic corpus, the min_support cut, and coverage.

### v3/tests/test_adversarial.py
The three attack families as matched pairs, that object swap crosses relations while type consistent does not, and that temporal stale raises rather than padding when too few slots changed twice.

### v3/tests/test_feed.py
Revert placement at the configured lag, the position gap being lag + 1, base rate injection, and the uncorrected versus corrected split at a checkpoint.

### v3/tests/test_editors.py
The Editor protocol, the registry, and that the stub responder answers from the edits it accepted.

### v3/tests/test_easyedit_adapter.py
Payload construction and that responder() raises before the first successful edit. The GPU path is the suite's one skip.

### v3/tests/test_probes.py
Answer normalisation, efficacy, locality, and general_ability raising when a suite is missing.

### v3/tests/test_elicitation.py
The three modes, that paraphrases never equal the original prompt, and that a hit from any probe in a mode counts.

### v3/tests/test_downstream.py
Corrupted versus poisoned answers over hop triples, and that poisoned is never above corrupted.

### v3/tests/test_env.py
Dtype derivation from compute capability, the override, and that no capability means fp32.

### v3/tests/test_grid.py
Cartesian expansion, cell id stability under key order, interleaved sharding, and pending detection.

### v3/tests/test_state.py
Atomic write, that an unserialisable payload leaves the previous state intact, and that a corrupt file reads as unfinished rather than raising.

### v3/tests/test_rq1_survival.py
Checkpoint intervals, the baseline general-ability measurement, and config validation.

### v3/tests/test_rq1_analysis.py
Grouping across every dimension, sample standard deviation over seeds, and skipping cells with no checkpoints.

### v3/tests/test_end_to_end.py
Two CPU passes through the whole chain. The ungated one runs corpus to attack families to feed to editor to probes to cell state to analysis; the gated one calibrates on two attack families and evaluates on the third. Also checks shard coverage and that an unfinished cell leaves no partial state.
Gotcha: this is the test that would have caught the superseded plan's central defect, where the adapter dropped the edited model and every unit test still passed.

### v3/tests/test_cluster.py
The MIG memory arithmetic, including the case that reshaped the pilot grid: an 8B model with a banded method does not fit an 18 GB slice. Also Slurm context parsing.

### v3/tests/test_power.py
Sample sizes, the guards against impossible inputs, and precision collapsing as prevalence falls. Includes the fact that the superseded pilot grid would have produced 2 poisoned facts per cell.

### v3/tests/test_gate_base.py
The redaction. Asserts the gate cannot reach the attack family through candidate.source, the adversarial flags or the entry itself, and that the claim survives redaction unchanged.

### v3/tests/test_gate_provenance.py
Trust smoothing, the account simulation including the cover traffic that stops the account being a label, and that outcomes are credited only when the operator would learn them.

### v3/tests/test_gate_signals.py
Each signal alone, including where it abstains and where it is honestly useless.

### v3/tests/test_gate_combiner.py
Weighting, the three-way split, and calibration. The fixture deliberately overlaps the classes so no threshold reaches a zero false positive rate, which is what makes the precision collapse visible.

### v3/tests/test_gate_rollback.py
Ledger order, replay plans, and that replaying without the poison removes it while replaying with it does not.

### v3/tests/test_rq2_gate.py
The gated arm: what it blocks, that refused entries never reach the trusted vocabulary, and that a checkpoint still happens when nothing has been applied.

### v3/tests/test_rq2_analysis.py
Grouping gated cells, the held-out flag over comma separated calibration sets, and refusing to subtract arms that differ in setup.

### v3/tests/test_worker.py
Shard resolution under a Slurm array, including a one-based array range, and that a dry run never touches CUDA_VISIBLE_DEVICES.

### v3/docs/cluster-baramati.md
The cluster this runs on: node table, MIG slices, the fit table that killed the 8B arm, the traps from earlier projects on the same machine, and the order to do things in.
Gotcha: access details and credentials stay out of the repo. The table comes from the cluster user guide and earlier session notes, not from a run of this code, and nothing is settled until 00_smoke.slurm has run once.

### v3/slurm/00_smoke.slurm
One slice, 20 minutes, runs the smoke script and exits with its code.
Gotcha: LF endings are mandatory; sbatch refuses a CRLF script. `.gitattributes` pins it.

### v3/slurm/worker_array.slurm
One array task per MIG slice, each running its own shard. Takes a grid path and an output directory, so RQ1 and RQ2 share it.
Gotcha: sets `--cpus-per-task=8` because the default of 1 silently single-threads the job, and refuses to start when the probe sets are missing rather than failing per cell. Compute nodes are offline, so it sets HF_HUB_OFFLINE=1.

### v3/hparams/
Eight EasyEdit hparams files, 4 methods by 2 models, named `<method>_<slug>.yaml` to match what `resolve_hparams` derives.
Gotcha: written from published templates on a laptop with no EasyEdit installed, so every file says NOT VALIDATED at the top. `scripts/04_check_hparams.py` is what settles them, and the UltraEdit pair is the least certain.

### v3/spikes/q1_volatility/FINDINGS.md
Verdict on the Q1 question, does knowledge split into invariant and volatile layers. Revised 2026-09-05 after review: the metric measures the composition of observed change rather than volatility, so the headline was downgraded from "volatility is predictable" to "the revision-to-addition mix is predictable" (split-half Spearman 0.695 across 278 relations). The distribution is one hump with a long tail, so any split is a threshold rather than a boundary.
Gotcha: leads with the correction, so read the first section before quoting any number. The 78/20 accretion split is demoted to a WikiBigEdit statistic with 4 reasons it does not generalise, and "adding cannot contradict" is explicitly withdrawn. A real rate needs Wikidata statement counts from the query service.

### v3/spikes/q1_volatility/load_wikibigedit.py
Downloads the 8 WikiBigEdit Wikidata snapshot diffs (2024-02-01 to 2024-07-01) and flattens them into one dataframe carrying a timestep index.
Exports: REPO_ID; TIMESTEP_FILES (ordered oldest first); KEEP_COLUMNS; LoadReport dataclass; load_all() -> (DataFrame, LoadReport)
Used by: v3/spikes/q1_volatility/analyze_volatility.py, v3/spikes/q1_volatility/analyze_stability.py
Gotcha: TIMESTEP_FILES order defines the timestep index, so never sort it. Drops 11,038 rows carrying a null subject_id or relation_id. First run pulls roughly 190 MB into the HF cache; later runs are offline.

### v3/spikes/q1_volatility/analyze_volatility.py
Measures per-relation update_share (share of a relation's rows tagged update rather than new) and recurrence (share of its updated pairs revised in more than one timestep), then tests the distribution shape.
Exports: MIN_SUPPORT (200); tag_breakdown(); recurrence_table(); per_relation_stats(); bimodality(); main()
Gotcha: update_share is the composition of observed change, NOT volatility and NOT the probability a fact changes, because the denominator holds only rows that already changed. The module docstring says so at length; do not reintroduce the name churn. Writes results.json and per_relation.csv beside itself. The bimodality coefficient reads 0.755 against a 0.556 reference but that is right skew rather than two modes, so read the histogram.

### v3/spikes/q1_volatility/analyze_stability.py
Tests whether update_share is a stable trait, via split-half Spearman between early and late timesteps, plus per-relation concentration of updates in a single timestep.
Exports: MIN_SUPPORT; MIN_UPDATES_PER_HALF (20); EARLY, LATE timestep tuples; concentration(); split_half(); main()
Gotcha: needs scipy, which the other two do not. Writes stability.json. The split-half Spearman of 0.695 shows the revision-to-addition MIX is stable, not that facts are volatile at a predictable rate. Concentration flags a relation for inspection and does not classify it: elections are lumpy real change and bot passes can spread evenly.

### v3/spikes/q1_volatility/results.json
Generated by analyze_volatility.py: load counts, tag shares, per-pair recurrence histogram, update_share and recurrence distribution shape, highest and lowest update_share relations.

### v3/spikes/q1_volatility/per_relation.csv
Generated by analyze_volatility.py: one row per relation with n_rows, n_updates, update_share, recurrence and pair counts. All 941 relations, unfiltered by support.

### v3/spikes/q1_volatility/stability.json
Generated by analyze_stability.py: split-half correlation, concentration summary, and the lumpiest and most evenly spread relations.

## v2/ (built out, framing superseded)

### v2/README.md
Active-project readme: architecture in one paragraph, directory layout, where future modules slot in (memory/, probe/, gates/, editing/, controller/, consolidation/), quick start, hardware contract, reproducibility habits (pip freeze per phase), phase status.

### v2/pyproject.toml
Package `u-jepa-v2`, src layout, requires-python >=3.10. Core deps torch/transformers/accelerate/safetensors/numpy/tqdm; optional extras `nf4` (bitsandbytes), `memory` (faiss-cpu, sentence-transformers), `probes` (datasets), `dev` (pytest, pytest-xdist). Declares pytest markers network/cuda/slow.
Gotcha: pins are floor-only, so a fresh install resolves different versions each time. `env.summary()` records torch and transformers versions into every results JSON so numbers stay traceable; the README asks for a `pip freeze` per phase.

### v2/docs/pipeline-1-build-plan.md
The active build plan. Six phases (0 foundations + GRACE baseline, 1 memory loop, 2 the probe, 3 merge/auto-rollback, 4 truth gate, 5 consolidation, 6 eval and writeup), each sized in 9h sessions with an explicit acceptance gate and a fallback. Also holds the four open forks (F1 editing substrate, F2 text-RAG vs latent injection, F3 plausibility scorer, F4 core model) and the pivot map.
Gotcha: gates beat calendar. Phase 2's probe AUROC >= 0.7 is a hard fork; below it the plan says stop and invest in the probe rather than move on.

### v2/docs/compass_artifact_wf-be604ab5-5612-4004-94bd-86c26e8da84c_text_markdown.md
Literature research report backing the v2 plan. Covers JEPA family (I-JEPA, V-JEPA 2, LLM-JEPA, LeJEPA/SIGReg), world models vs knowledge bases, the editing literature (ROME/MEMIT ripple failures, WISE's impossible triangle, GRACE's rollback-friendly codebook), SEAL and model collapse, and CLS theory as the citation lineage for consolidation.

### v2/src/u_jepa_v2/__init__.py
Package docstring plus `__version__ = "0.0.1"`. Points at the build plan.

### v2/src/u_jepa_v2/env.py
Runtime/device/dtype detection shared by every script and results log. Lazily imports torch so it works in a torch-less env.
Exports: detect_runtime() -> "kaggle"|"colab"|"local"; detect_device() -> "cuda"|"cpu"; gpu_name(); gpu_vram_gb(); has_native_bf16(); preferred_dtype_str(); summary() -> EnvSummary; dataclass EnvSummary (runtime, device, gpu_name, gpu_vram_gb, native_bf16, python, platform, torch_version, transformers_version) with .as_dict()
Used by: v2/scripts/00_smoke_env.py, v2/scripts/01_smoke_load_cores.py, v2/scripts/02_spike_f1_grace.py, v2/tests/test_env.py
Gotcha: `preferred_dtype_str()` returns the constant "fp16" regardless of hardware; it encodes the plan's T4-parity decision, not a capability check. `has_native_bf16()` is the actual capability probe (compute capability major >= 8).

### v2/src/u_jepa_v2/persistence.py
Atomic save/load for everything that must survive a killed 9h session: memory store, codebook, probe sets, thresholds, results, RunState. Writes via temp file plus fsync plus os.replace so a crash leaves the previous file intact.
Exports: atomic_write_text(path, text); save_json(path, data, *, indent=2) -> Path; load_json(path); save_torch(path, state) -> Path; load_torch(path) -> dict; sha256_of_file(path, chunk=1MiB); dataclass RunState(phase, step, seed, elapsed_seconds, last_checkpoint, extras); kaggle_working_dir() -> Path; kaggle_input_path(slug) -> Path | None; class CheckpointTimer (context manager, .elapsed accumulates)
Used by: v2/scripts/01_smoke_load_cores.py, v2/scripts/02_spike_f1_grace.py, v2/tests/test_persistence.py, v2/tests/test_persistence_torch.py
Gotcha: `load_torch` uses `weights_only=True`, so any payload needing full pickle execution raises. `save_json` raises TypeError on unknown types rather than coercing (numpy scalars and Path are handled). `kaggle_working_dir()` resolution order is `U_JEPA_V2_CKPT_DIR` env, then `/kaggle/working`, then the first ancestor dir with .git or pyproject.toml plus `/checkpoints`.

### v2/src/u_jepa_v2/models/__init__.py
Barrel re-exporting CORE_MODELS, CoreSpec, freeze_, load_core, resolve_core from `core`.

### v2/src/u_jepa_v2/models/core.py
Frozen-core loader. Registry of the two cores the plan's fork F4 picked, plus an HF load path that returns the model in eval() with every param requires_grad=False.
Exports: dataclass CoreSpec(short_name, hf_id, family, params_b, notes); CORE_MODELS dict keyed "gpt2-xl" and "qwen2.5-1.5b"; resolve_core(name) -> CoreSpec (accepts short name or hf_id, case-insensitive, raises KeyError); freeze_(model) -> model; load_core(name, *, dtype="fp16", nf4=False, device="auto", cache_dir=None, trust_remote_code=False) -> (model, tokenizer)
Used by: v2/scripts/01_smoke_load_cores.py, v2/scripts/02_spike_f1_grace.py, v2/tests/test_models_core.py
Gotcha: `device="auto"` hands off to HF `device_map="auto"`; any other value pins the whole model via `device_map={"": device}`. `torch_dtype` is set in both the plain and nf4 branches on purpose, since under nf4 it controls the dtype of the modules bitsandbytes leaves unquantized. Pad token falls back to eos when the tokenizer ships none.

### v2/src/u_jepa_v2/eval/__init__.py
Barrel re-exporting Edit, match_prefix, mean_heldout_perplexity, normalize, perplexity_from_logits, score_edits, sequence_perplexity from `edit_metrics`.

### v2/src/u_jepa_v2/eval/edit_metrics.py
Edit-success, locality, and perplexity measurement for knowledge editing. The F1 spike uses all of it; locality and perplexity are two of the three Phase 2 probe signals, which is why they live here rather than in the spike script.
Exports: normalize(s) -> str; match_prefix(generation, target) -> bool; perplexity_from_logits(logits, targets) -> float; dataclass Edit(prompt, target_new, ground_truth, subject="", paraphrases=[], neighborhood=[]); greedy_generate(model, tok, prompt, max_new_tokens=8) -> str; score_edits(model, tok, edits, max_new_tokens=8) -> dict with edit_success / locality_preserved / paraphrase_success / n_* / details; sequence_perplexity(model, tok, text, max_len=512) -> float; mean_heldout_perplexity(model, tok, texts) -> float
Used by: v2/scripts/02_spike_f1_grace.py, v2/tests/test_edit_metrics.py
Gotcha: an empty target never matches, so a model that emits nothing cannot score a free hit. `sequence_perplexity` returns NaN on inputs under 2 tokens and `mean_heldout_perplexity` drops those NaNs silently. locality_preserved and paraphrase_success are None (not 0.0) when no probes were supplied.

### v2/scripts/00_smoke_env.py
Phase 0 step 1. Prints `env.summary()` as JSON and asserts the plan's assumptions. Run first on every fresh Kaggle session.
Gotcha: exits 2 (not 0) when the runtime is Kaggle and the device is CPU, so a forgotten accelerator fails fast instead of burning session quota. Also exits 2 if a T4 claims native bf16. Low VRAM is a warning, not a failure. Bootstraps `../src` onto sys.path so it runs without an install.

### v2/scripts/01_smoke_load_cores.py
Phase 0 step 2. Loads each core one at a time (never together, to keep peak VRAM honest), runs a 12-token greedy generation, reports load time / generate time / peak VRAM per model, optionally writes a results JSON.
Gotcha: flags are `--only <core>`, `--nf4`, `--dtype {fp16,bf16,fp32}`, `--cache-dir`, `--out`. Between models it does del + gc.collect() + empty_cache and resets peak-memory stats, so the peak numbers are per-model and not cumulative.

### v2/scripts/02_spike_f1_grace.py
The timeboxed spike resolving open fork F1: can EasyEdit's GRACE edit GPT-2-XL cleanly on a T4? Scores a 6-item inline CounterFact-style edit set before editing, tries GRACE, re-scores after, and writes a verdict JSON (KEEP_EASYEDIT / HANDROLL / INVESTIGATE) with all numbers and any captured traceback.
Exports: EDITS (list[Edit]); HELDOUT (list[str]); run_easyedit_grace(edits, hparams_path) -> (edited_model, native_metrics); main() -> int
Gotcha: no step is allowed to hard-crash. A missing hparams yaml or an EasyEdit import failure is captured and turned into a HANDROLL verdict, because "EasyEdit fights us" is itself the answer the spike is looking for. Verdict thresholds are edit-success lift >= 0.5, locality >= 0.6, perplexity ratio <= 1.5. Defaults `--hparams` to the notebook's clone path under `/kaggle/working/EasyEdit`.

### v2/kaggle/README.md
How v2 runs on Kaggle: account warning (kernel metadata points at `kartikshirode` while the repo-root kaggle.json is a different account), per-spike folder layout, the standard six-cell notebook shape, push/pull commands, session limits. Notes that a single T4 is enough for v2's 1.5B cores, unlike v1's dual-T4 Qwen3-14B setup.

### v2/kaggle/build_f1_notebook.py
Generator for `f1_spike/f1_spike.ipynb`. Builds the nbformat-4 cell list in Python and dumps it as JSON so the output always parses. Re-run after editing any cell.
Exports: md(*lines), code(*lines), CELLS, NB
Gotcha: this is the only supported way to change the notebook. Hand-editing the ipynb is what broke a v1 Kaggle push.

### v2/kaggle/f1_spike/f1_spike.ipynb
Generated notebook for the F1 spike. Six cells: move HF cache to /tmp, clone U-JEPA + EasyEdit and pip install both, set HF env/secret, run the env gate, run the spike script, print the verdict. Do not edit directly; regenerate from `build_f1_notebook.py`.

### v2/kaggle/f1_spike/kernel-metadata.json
Kernel config for `kaggle kernels push`: id `kartikshirode/u-jepa-v2-f1-spike-grace`, private, GPU on, internet on, single NvidiaTeslaT4.

### v2/tests/__init__.py
Empty package marker.

### v2/tests/conftest.py
Puts `v2/src` on sys.path so the suite runs without an editable install, and implements the marker-based skipping: `network` skips unless U_JEPA_V2_RUN_NETWORK is truthy, `slow` unless U_JEPA_V2_RUN_SLOW, `cuda` unless torch reports a CUDA device.
Gotcha: truthy means one of 1/true/yes/on, case-insensitive.

### v2/tests/test_env.py
Contract tests for `env`: runtime and device return known values, preferred dtype is fp16, summary round-trips to a dict with all nine keys, bf16 flag is False on CPU. No GPU or network needed.

### v2/tests/test_models_core.py
Registry tests for `models/core`: both cores present, resolve by short name / hf_id / mixed case, unknown raises KeyError, freeze_ clears requires_grad. Plus one network+slow GPT-2-XL load test that also asserts `device="cpu"` really pins placement.

### v2/tests/test_edit_metrics.py
Tests for `eval/edit_metrics`. Pure string scoring and logit math tested directly (uniform logits over V give perplexity ~V, peaked logits give ~1). `score_edits` and `sequence_perplexity` run against a toy whitespace tokenizer and a stub nn.Module with a canned `generate`, so no real model loads.

### v2/tests/test_persistence.py
JSON persistence tests: round-trip, atomicity under a monkeypatched crashing `os.replace` (old file survives, no temp leftovers), parent-dir creation, dataclass and numpy-scalar and Path serialization, TypeError on unknown types, checkpoint-dir resolution and env override, a known sha256, and the timer accumulating.

### v2/tests/test_persistence_torch.py
Tensor save/load round-trip, atomicity of `save_torch` under a failing replace, and proof that `load_torch`'s weights_only=True rejects a payload needing full pickle execution.

## legacy/v1/ (frozen 2026-06-05, do not modify)

### legacy/v1/README.md
Archive notice and layout for the frozen v1 tree. Explains what the v1 architecture was, that Phases 0 and 1 shipped and Phase 2 revealed structural design errors, and points at the audit doc.

### legacy/v1/pyproject.toml
Package `u-jepa` 0.1.0, src layout, requires-python >=3.11,<3.13. pytest markers slow and needs_gpu, `--strict-markers`.

### legacy/v1/requirements.txt
Shared deps with loose pins so Kaggle's preinstalled torch/transformers do not conflict: torch 2.4-2.8, transformers 4.46+, peft, bitsandbytes, accelerate, datasets, plus the usual scientific stack.

### legacy/v1/requirements-kaggle.txt
Kaggle-only extras on top of requirements.txt: vllm>=0.6.6 and autoawq>=0.2.7. flash-attn deliberately omitted because it builds from source and reliably fails on Kaggle.

### legacy/v1/.python-version
3.12.10

### legacy/v1/src/u_jepa/__init__.py
`__version__ = "0.1.0"`. No re-exports.

### legacy/v1/src/u_jepa/config.py
Central frozen dataclasses every v1 script reads its config from.
Exports: HardwareConfig(device, vram_ceiling_mb=7800, grad_accum=8, micro_batch=1, max_seq_len=512, bf16_compute); QwenConfig(model_id="Qwen/Qwen3-14B", quant_4bit, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="bfloat16", hidden_size=5120, n_layers=40); QwenBaselineConfig (LatentMAS vLLM repro settings, tensor_parallel_size=2); VJEPAConfig; LoraConfig(rank=16, alpha=32, dropout, target_modules=("q_proj","v_proj")); Paths (root/results/checkpoints/hf_cache)
Used by: legacy/v1/src/u_jepa/models/qwen_base.py, legacy/v1/scripts/02_train_continual_phase1.py, legacy/v1/scripts/03_train_jepa_aux_phase2.py
Gotcha: there is no "Qwen3-14B-Instruct" repo on HuggingFace; `Qwen/Qwen3-14B` IS the post-trained chat model and `-Base` is the pretrained one. Docstrings elsewhere in v1 still say "Instruct".

### legacy/v1/src/u_jepa/util/env.py
v1's environment detection and path resolution, one layer richer than v2's (it also resolves results/checkpoint/hf-cache dirs and a can_run_vllm flag).
Exports: frozen dataclass Env(name, is_kaggle, repo_root, results_dir, checkpoint_dir, hf_cache_dir, can_run_vllm); detect() -> Env; prepare(env=None) -> Env
Used by: legacy/v1/scripts/01_repro_latentmas_gsm8k.py, legacy/v1/scripts/02_train_continual_phase1.py, legacy/v1/scripts/03_train_jepa_aux_phase2.py, legacy/v1/tests/test_env_detection.py, legacy/v1/tests/test_env_prepare.py
Gotcha: `prepare()` has side effects. It mkdirs all three directories and setdefaults HF_HOME / HF_HUB_CACHE / TRANSFORMERS_CACHE. On Kaggle the HF cache is forced to `/tmp/hf_cache` because `/kaggle/working` has a 20 GB quota and Qwen3-14B alone is ~28 GB. `can_run_vllm` is False on Windows.

### legacy/v1/src/u_jepa/util/prompting.py
Chat-template formatting shared by v1's train and eval paths, so both sides see identical formatting.
Exports: format_chat_prompt(tokenizer, prompt) -> (formatted_prompt, used_template)
Used by: legacy/v1/src/u_jepa/train/continual_loop.py, legacy/v1/src/u_jepa/train/jepa_aux_loop.py, legacy/v1/src/u_jepa/eval/continual.py, legacy/v1/src/u_jepa/eval/spider_em.py, legacy/v1/tests/test_prompting.py
Gotcha: the returned bool drives `add_special_tokens=not used_template` at every call site; getting it wrong double-adds BOS. It tries `enable_thinking=False` first and falls back for older tokenizers, because Qwen3 on greedy decode can emit a `<think>` preamble that pushes the real answer outside a short eval window.

### legacy/v1/src/u_jepa/util/__init__.py
Empty package marker.

### legacy/v1/src/u_jepa/models/qwen_base.py
NF4 loader for Qwen3-14B plus VRAM and module-discovery helpers, used from Phase 1 onward where v1 owns the loader (Phase 0 loaded through vendored LatentMAS/vLLM instead).
Exports: load_qwen_nf4(cfg=None, device_map=None) -> (model, tokenizer); model_input_device(model) -> torch.device; qwen_vram_usage_mb(model) -> int; target_module_names(model, target_short_names=("q_proj","v_proj")) -> list[str]
Used by: legacy/v1/scripts/02_train_continual_phase1.py, legacy/v1/scripts/03_train_jepa_aux_phase2.py, legacy/v1/tests/test_qwen_helpers_and_metrics_edges.py
Gotcha: default device_map pins everything to cuda:0 (`{"": 0}`), NOT `auto`. Sharding across two T4s crashes the training loop, which forces inputs onto one device. Pass `device_map="auto"` explicitly to opt in. BitsAndBytesConfig is imported lazily so the module still imports on Windows without bitsandbytes.

### legacy/v1/src/u_jepa/models/__init__.py
Empty package marker.

### legacy/v1/src/u_jepa/continual/orthogonal_lora.py
The hot-swappable LoRA bank: per-task adapters as plain nn.Parameters on a frozen base, activated by flipping one pointer, with forward hooks adding the active adapter's delta to each target module's output. Written instead of using peft so the A and B matrices stay directly reachable for the N-LoRA penalties and so activation is O(1).
Exports: class OrthogonalLoRABank(base_model, rank=16, target_modules=("q_proj","v_proj"), alpha=32.0) with add_task(task_id), activate(task_id), property active, adapter_matrices(task_id, module_name) -> (A, B), forward_target(x, module_name), install_hooks() -> list of handles; class _Adapter; _safe_key(module_name)
Used by: legacy/v1/src/u_jepa/train/continual_loop.py, legacy/v1/src/u_jepa/train/jepa_aux_loop.py, legacy/v1/src/u_jepa/eval/continual.py, legacy/v1/src/u_jepa/eval/spider_em.py, and 6 test modules (grep to enumerate)
Gotcha: `install_hooks()` returns handles the caller MUST remove in a finally block or the bank leaks callbacks into the base model. Adapter params are held in fp32 for stable AdamW updates and the delta is cast back to the activation dtype in `forward_target`. Adapters are placed on the first target module's device, so a multi-GPU device_map can mismatch. B initializes to zeros so delta_W is zero at step 0. Callers reach into `bank._active` directly (the Phase 2 view-B bug was exactly that); there is no suspend() context manager.

### legacy/v1/src/u_jepa/continual/n_lora_loss.py
The O-LoRA orthogonality term plus the N-LoRA non-collision term over the A matrices of sequential adapters. This is the penalty that produced Phase 1's zero forgetting.
Exports: n_lora_penalty(A_curr, A_prevs, collision_weight=0.01) -> scalar tensor; collect_a_matrices(bank, task_ids, module_name) -> list; n_lora_penalty_over_bank(bank, current_task, prev_tasks, collision_weight=0.01) -> scalar tensor
Used by: legacy/v1/src/u_jepa/train/continual_loop.py, legacy/v1/tests/test_n_lora_loss.py, legacy/v1/tests/test_n_lora_penalty_correctness.py
Gotcha: previous-task A matrices are always detached, so gradient only ever flows into the current adapter. With no previous tasks the penalty is an explicit zero built on a current-task parameter's device and dtype, which keeps `.backward()` type-safe. Cost is O(num_tasks) per step, untested past two tasks.

### legacy/v1/src/u_jepa/continual/__init__.py
Empty package marker.

### legacy/v1/src/u_jepa/losses/llm_jepa.py
LLM-JEPA auxiliary loss: predict view-B's pooled hidden state from view-A's, under cosine or MSE.
Exports: class TiedPredictor(hidden, k_tokens=3) (one shared Linear applied k times); llm_jepa_loss(predictor, h_a, h_b, metric="cosine") -> scalar
Used by: legacy/v1/src/u_jepa/train/jepa_aux_loop.py, legacy/v1/tests/test_llm_jepa_loss.py
Gotcha: h_b is detached inside the loss, which is what stops the trivial both-sides-constant collapse. The tied predictor is mathematically W^k, and the audit (doc section 8.5) flags k=3 tied as unstable versus the paper's stack of independent linears; it also concluded (8.1) that Q/A pairs are not "views" in the JEPA sense at all.

### legacy/v1/src/u_jepa/losses/sigreg.py
Sliced Epps-Pulley normality statistic over embeddings, used as an anti-collapse regularizer. Prefers the vendored lejepa implementation and falls back to a self-contained one.
Exports: sigreg_loss(embeddings, num_slices=128) -> scalar; using_lejepa() -> bool; _fallback_sliced_epps_pulley(x, num_slices, n_points=17)
Used by: legacy/v1/src/u_jepa/train/jepa_aux_loop.py, legacy/v1/tests/test_sigreg.py
Gotcha: import has side effects. It prints the active backend to stderr and mutates sys.path if `vendored/lejepa` exists. That subtree was removed in the 2026-06-03 cleanup, so the fallback path is now always live; re-add it with `git subtree add --prefix=vendored/lejepa https://github.com/rbalestr-lab/lejepa main --squash`. Input is upcast to fp32 for the lejepa path (a fp16 mismatch crashed a Kaggle run) and the result divided by batch size, because lejepa returns an N-scaled statistic while the fallback returns a mean; the two differed ~500x before that fix.

### legacy/v1/src/u_jepa/losses/__init__.py
One-line docstring, no re-exports.

### legacy/v1/src/u_jepa/data/trace.py
TRACE benchmark loaders for the two Phase 1 tasks, emitting `{prompt, target}` dicts ready for PromptTargetDataset.
Exports: load_fomc(split="train", n=None) -> list[dict]; load_scienceqa_text(split="train", n=None) -> list[dict]; TASK_LOADERS dict; load_trace_task(name, split, n) -> list[dict]
Used by: legacy/v1/scripts/02_train_continual_phase1.py, legacy/v1/tests/test_trace_loader.py
Gotcha: `_load_dataset_resilient` tries several HF config names and falls back across splits (validation -> test -> train) because FOMC is config-gated and ships no validation split. ScienceQA rows with an image attached are skipped entirely (vision was deferred to Phase 3). Unrecognized FOMC labels silently become "neutral".

### legacy/v1/src/u_jepa/data/spider.py
Spider text-to-SQL loader producing the NL/SQL paired views Phase 2's JEPA loss consumed.
Exports: SPIDER_REPOS tuple; load_spider_pairs(split="train", n=None) -> list[dict] with keys prompt/target/view_a/view_b
Used by: legacy/v1/scripts/03_train_jepa_aux_phase2.py, legacy/v1/tests/test_spider_loader.py
Gotcha: it deliberately does NOT fall back from validation to train (an earlier version did, which silently turned eval into train-on-train). test falls back to validation only, since Spider publishes no public test split. The prompt carries no database schema, which the audit (8.7) identifies as the reason Phase 2's EM ceiling sat at 5-7%.

### legacy/v1/src/u_jepa/data/__init__.py
Empty package marker.

### legacy/v1/src/u_jepa/train/continual_loop.py
Phase 1's training loop: one fresh adapter per task, CE on masked prompt/target pairs plus the N-LoRA penalty against every previous adapter.
Exports: class PromptTargetDataset(items, tokenizer, max_len=512, pad_to_max=False); train_task(bank, tokenizer, task_id, items, prev_task_ids=(), epochs=2, lr=3e-4, orth_weight=0.5, collision_weight=0.01, grad_accum=8, max_len=512, device="cuda:0", log_every=25) -> dict; _resolve_input_device(model, fallback)
Used by: legacy/v1/scripts/02_train_continual_phase1.py, legacy/v1/src/u_jepa/train/jepa_aux_loop.py, legacy/v1/tests/test_continual_helpers.py, legacy/v1/tests/test_phase1_fixes.py, legacy/v1/tests/test_prompt_target_masking.py
Gotcha: the dataset guards hard against all-masked label rows, which make CE return NaN and poison every gradient in the accumulation window. It drops rows whose target tokenizes to nothing, always reserves at least one target token on truncation (dropping the head of the prompt instead), and raises if a row still ends up fully masked. Prompt and target are tokenized separately then concatenated to stop a BPE merge across the boundary from shifting prompt_len. batch_size is hardcoded to 1; the end-of-epoch optimizer step fires only when a partial accumulation is actually pending. Hooks are removed in a finally block.

### legacy/v1/src/u_jepa/train/jepa_aux_loop.py
Phase 2's training loop: CE plus LLM-JEPA cosine plus SIGReg, on top of the same bank and hooks as Phase 1.
Exports: class SpiderJEPADataset(items, tokenizer, max_len=512); train_with_jepa_aux(bank, tokenizer, task_id, items, predictor, epochs=2, lr=3e-4, lambda_jepa=0.5, lambda_sigreg=0.1, grad_accum=8, max_len=512, device="cuda:0", sigreg_slices=64, log_every=25) -> dict; _spider_collate(batch); _pool_view_b(...); _precompute_view_b_cache(...)
Used by: legacy/v1/scripts/03_train_jepa_aux_phase2.py, legacy/v1/tests/test_jepa_aux_loop.py
Gotcha: view-B embeddings are pooled ONCE up front with `bank._active` forced to None. Before that fix the JEPA target was computed through the very adapter being trained, so the loss chased a moving target. The hidden-state tuple is deleted immediately after the needed layer is captured, because a 41-layer stack otherwise stays resident across the SIGReg and JEPA forwards. `_spider_collate` hard-raises on any batch size other than 1. SIGReg runs over per-token vectors of a single sequence, which the audit (8.2) calls the wrong axis; it needs a rolling buffer of pooled embeddings across micro-batches instead.

### legacy/v1/src/u_jepa/train/__init__.py
Empty package marker.

### legacy/v1/src/u_jepa/eval/continual.py
Per-task eval harness for sequential continual learning: greedy decode, match against the target, assemble one row of the accuracy matrix.
Exports: eval_task(bank, tokenizer, task_id, items, device="cuda:0", max_new_tokens=8, log_every=50) -> float; build_accuracy_matrix(bank, tokenizer, eval_sets, seen_tasks, device, task_order=None) -> list[float]; _match_generation(generated, target) -> bool; _resolve_input_device(model, fallback)
Used by: legacy/v1/scripts/02_train_continual_phase1.py, legacy/v1/src/u_jepa/eval/spider_em.py, legacy/v1/tests/test_accuracy_matrix_alignment.py, legacy/v1/tests/test_continual_helpers.py, legacy/v1/tests/test_phase1_fixes.py
Gotcha: single-character targets (ScienceQA's A..H) require an exact first-char match AND a non-letter after it, so a generation like "absolute zero" does not count as label "A". Multi-char targets use a case-insensitive prefix match. `eval_task` activates the task's OWN adapter before scoring, so Phase 1's zero-forgetting number is hot-swap evaluation, not joint inference (audit note 5.1). Unseen tasks get 0.0 by convention. Pass `task_order` explicitly; relying on eval_sets dict order is how a column silently misaligns against the forgetting metric.

### legacy/v1/src/u_jepa/eval/metrics.py
Continual-learning metrics on an accuracy matrix, following Lopez-Paz and Ranzato (2017). A[i][j] = accuracy on task j after training through task i.
Exports: backward_transfer(A) -> float; average_forgetting(A) -> float; average_accuracy(A) -> float; forward_transfer(A, b) -> float
Used by: legacy/v1/scripts/02_train_continual_phase1.py, legacy/v1/tests/test_metrics.py, legacy/v1/tests/test_metrics_handbuilt.py, legacy/v1/tests/test_qwen_helpers_and_metrics_edges.py
Gotcha: every metric returns 0.0 rather than raising on a degenerate matrix (T < 2, or empty). The upper-triangular-ish convention is assumed but not enforced, so a caller that fills unseen entries with something other than 0 gets quietly wrong numbers.

### legacy/v1/src/u_jepa/eval/spider_em.py
Phase 2's two eval gates: a string-level Spider exact-match proxy, and the representation-collapse probe.
Exports: spider_em(bank, tokenizer, items, task_id=None, device="cuda:0", max_new_tokens=128, log_every=25, max_len=1024) -> float; hidden_state_cond_number(bank, tokenizer, prompts, task_id=None, device="cuda:0", max_len=512, log_every=16) -> float; _norm(s)
Used by: legacy/v1/scripts/03_train_jepa_aux_phase2.py, legacy/v1/tests/test_spider_em.py
Gotcha: this is NOT real Spider EM (which parses the SQL AST); it is a normalized prefix match, justified because the gate is a delta between two arms scored the same way. `hidden_state_cond_number` clamps the smallest singular value at 1e-12 before dividing, so a fully collapsed representation reports a huge finite number rather than inf. It returns NaN with fewer than 2 prompts. Both functions install hooks and remove them in a finally block.

### legacy/v1/src/u_jepa/eval/__init__.py
Empty package marker.

### legacy/v1/scripts/00_smoke_env.py
v1 day-1 smoke: CUDA present, >= 7500 MB VRAM, and a real bitsandbytes NF4 quantize of a 4096x4096 tensor. Exits non-zero on any missing prereq.

### legacy/v1/scripts/01_repro_latentmas_gsm8k.py
Phase 0 entry: reproduce LatentMAS GSM8K accuracy on Qwen3-14B-AWQ via vLLM, calling vendored LatentMASMethod directly with a hand-built argparse.Namespace to bypass run.py's hardcoded model choices. Shipped 60.8%.
Gotcha: HISTORICAL and currently unrunnable. The `vendored/LatentMAS` subtree was removed in the 2026-06-03 cleanup; restore it with `git subtree add --prefix=vendored/LatentMAS https://github.com/Gen-Verse/LatentMAS main --squash` first. It applies monkey-patches at module import time (an autoawq/transformers activations shim among them), which is why its tests import it via importlib.

### legacy/v1/scripts/02_train_continual_phase1.py
Phase 1 entry: FOMC then ScienceQA-text sequentially on a frozen NF4 Qwen3-14B, 1500 train / 300 eval per task. Gate was average_forgetting < 0.05 and backward_transfer >= -0.02; it passed at 0.0 forgetting.
Gotcha: no-ops with a message on local Windows, since bitsandbytes NF4 is effectively Linux-only. Pivot trigger baked into the plan: forgetting > 0.10 meant swapping the base to Phi-3.5-mini-Q4 and rerunning.

### legacy/v1/scripts/03_train_jepa_aux_phase2.py
Phase 2 entry: two independent LoRA arms on the same frozen Qwen3-14B, arm A CE-only and arm B CE + LLM-JEPA + SIGReg, both scored on the same Spider validation slice. 800 train / 200 eval per arm, sized to fit both arms in one 9h session.
Gotcha: both gates failed (delta EM -0.005 against a +0.02 bar, condition number 4.1e13 against a <100 bar). See `docs/external_audit_context.md` section 8 before reusing any of this design. Local Windows is a no-op.

### legacy/v1/kaggle/README.md
v1's Kaggle workflow: one-time account and HF-token setup, per-phase notebook layout, the standard cell shape, push/pull commands, session limits. Superseded for new work by `v2/kaggle/README.md`.

### legacy/v1/kaggle/phase0/kernel-metadata.json
Kernel config, id `chinmayishirode/u-jepa-phase0-baseline`, single T4, GPU and internet on.

### legacy/v1/kaggle/phase0/phase0_baseline.ipynb
Phase 0 notebook: clone, install, HF auth, run the LatentMAS GSM8K reproduction, print results. Took 14 kernel versions to get green.

### legacy/v1/kaggle/phase0/phase0_baseline.py
Script form of the Phase 0 notebook, kept alongside it.

### legacy/v1/kaggle/phase1/kernel-metadata.json
Kernel config, id `chinmayishirode/u-jepa-phase1-continual`, single T4.

### legacy/v1/kaggle/phase1/phase1_continual.ipynb
Phase 1 notebook: the sequential FOMC then ScienceQA continual run that produced zero forgetting.

### legacy/v1/kaggle/phase2/kernel-metadata.json
Kernel config, id `kartikshirode/u-jepa-phase2-jepa-aux`, single T4, keywords gpu/qwen3/lora/jepa/spider.

### legacy/v1/kaggle/phase2/phase2_jepa_aux.ipynb
Phase 2 notebook: the two-arm Spider comparison. Rewritten as real nbformat JSON during the swarm audit after the original pseudo-XML version broke a push.

### legacy/v1/docs/manual_verification_phase0.md
Cell-by-cell manual smoke checklist for the Phase 0 Kaggle run, with the exact output patterns each cell must produce and what to do when one is missing. Covers what unit tests cannot: that the whole pipeline runs together on real hardware.

### legacy/v1/docs/manual_verification_phase1.md
Post-run verification checklist for Phase 1: the expected shape of `results/phase1_continual.json`, how to read the accuracy matrix, and live decode spot-checks that must run in the same kernel while the model and bank are still resident.

### legacy/v1/docs/superpowers/plans/2026-05-26-ujepa-prototype.md
The v1 master plan, 12-14 weeks, six phases, full intended repo layout and per-task checkboxes. Superseded twice over (first by the Kaggle ADR for hardware, then wholesale by the v2 pipeline plan) but it is the record of what each v1 phase was supposed to deliver.

### legacy/v1/docs/superpowers/specs/2026-06-03-u-jepa-v2-architecture.md
The intermediate "central JEPA brain + domain sub-agents" design that was drafted after Phase 2 failed and then abandoned in favor of the frozen-core-plus-memory pipeline. History only; the active design is `v2/docs/pipeline-1-build-plan.md`.

### legacy/v1/tests/__init__.py
Empty package marker.

### legacy/v1/tests/conftest.py
Puts `legacy/v1/src` on sys.path so the frozen suite runs without an editable install. No marker logic (unlike v2's conftest).

### legacy/v1/tests/test_orthogonal_lora.py
Core bank behavior on a CPU stub: adapter creation, per-task activation, zero delta at init, base params staying frozen.

### legacy/v1/tests/test_orthogonal_lora_hooks.py
Forward-hook integration: the active adapter's delta actually appears at the base module's output on a two-linear stub block.

### legacy/v1/tests/test_orthogonal_lora_errors.py
Error paths: no matching target modules, duplicate add_task, activating an unknown task, and multi-module hook delta correctness.

### legacy/v1/tests/test_bank_gradient_flow.py
End-to-end gradient flow on a CPU stub with q_proj, v_proj and a non-target o_proj: backprop updates the active task's A and B, never the frozen base, and never another task's adapter. The most load-bearing Phase 1 property.

### legacy/v1/tests/test_n_lora_loss.py
Single-module `n_lora_penalty`: zero with no previous tasks, larger for overlapping A matrices than orthogonal ones.

### legacy/v1/tests/test_n_lora_penalty_correctness.py
`n_lora_penalty_over_bank` end to end on a stub bank, which is the function the training loop actually calls. Also confirms previous-task A is detached so the penalty only moves the current task.

### legacy/v1/tests/test_continual_helpers.py
Pure-Python coverage of PromptTargetDataset masking and build_accuracy_matrix, using a toy whitespace tokenizer and a fake bank so no model or tokenizer downloads.

### legacy/v1/tests/test_prompt_target_masking.py
The two masking failure modes that must never ship: an all -100 label row (undefined CE, poisons the loss) and prompt tokens leaking into the loss. Asserts the exact target ids survive unmasked.

### legacy/v1/tests/test_accuracy_matrix_alignment.py
Column-alignment contract for build_accuracy_matrix: row length equals n, unseen tasks are 0.0, column j always maps to task_order[j] regardless of how eval_sets was built. A misaligned column produces a plausible-looking but wrong forgetting number.

### legacy/v1/tests/test_metrics.py
BWT, forgetting, average accuracy and forward transfer on small matrices, including the degenerate single-task cases.

### legacy/v1/tests/test_metrics_handbuilt.py
Hand-computed BWT and forgetting for the exact Phase 1 scenario (learn task 0, then task 1, task 0 drops), so a refactor of the formulas gets caught.

### legacy/v1/tests/test_qwen_helpers_and_metrics_edges.py
qwen_base helpers (VRAM counter, target-module-name finder) plus metric edge cases on empty matrices and single-task forward transfer.

### legacy/v1/tests/test_llm_jepa_loss.py
TiedPredictor shape preservation, loss going to zero on aligned views and rising on divergent ones, and gradient reaching the predictor.

### legacy/v1/tests/test_sigreg.py
SIGReg shape contract, correct ordering between normal and degenerate embeddings, and the fallback path running without the vendored lejepa package.

### legacy/v1/tests/test_jepa_aux_loop.py
Phase 2 loop wiring on a tiny CPU stub: dataset construction, predictor gradients, and a no-crash smoke run in a few seconds.

### legacy/v1/tests/test_spider_em.py
The `_norm` matcher and `hidden_state_cond_number` against CPU stubs, including the too-few-samples NaN path.

### legacy/v1/tests/test_spider_loader.py
Spider prompt construction and loader smoke tests. The real HF fetch is opt-in via SPIDER_NETWORK=1.

### legacy/v1/tests/test_trace_loader.py
TRACE loader registry and unknown-task KeyError. The download tests are marked slow.

### legacy/v1/tests/test_env_detection.py
`env.detect()` returns a known platform name with a repo_root that exists and contains pyproject.toml.

### legacy/v1/tests/test_env_prepare.py
`env.prepare()` side effects: it creates every directory it advertises and setdefaults the three HF cache env vars.

### legacy/v1/tests/test_phase0_patches.py
The three import-time monkey-patch helpers and the namespace builder in `scripts/01_repro_latentmas_gsm8k.py`, loaded via importlib. Runs on Windows with no GPU, no vllm, no autoawq.

### legacy/v1/tests/test_phase1_fixes.py
Regression suite for the swarm-audit fixes: dataset boundary alignment and overflow guard, grad-accumulation flush, adapter dtype cast, adapter device placement, single-char label matching, and explicit task_order. CPU-only, no downloads.

## results/

### results/phase0_baseline.json
Phase 0 headline: LatentMAS GSM8K on Qwen3-14B-AWQ, 60.8% (152/250), 2x T4, 6383 s. Notes record why it sits under the paper's ~78%: AWQ quantization, an 8192 context cap, and eager mode.

### results/phase1_continual.json
Phase 1 headline: accuracy matrix [[0.753, 0.0], [0.753, 0.95]], average_forgetting 0.0, backward_transfer 0.0, average accuracy 0.852. The FOMC score was byte-identical before and after training ScienceQA.

### results/phase2_jepa_aux.json
Phase 2 headline: baseline EM 0.065, treatment EM 0.060, delta -0.005 against a +0.02 gate, hidden covariance condition number 4.12e13 against a <100 gate. Both gates failed; the second by eleven orders of magnitude.

### results/.gitkeep
Keeps the directory tracked while `.gitignore` blocks unlisted result JSONs.
