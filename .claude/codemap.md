<!-- codemap-format: v1 -->

# Codemap: U-JEPA

## Overview

JEPA-based continual-learning research repo. Three generations live side by side.

- `v3/` is the **newest** line, started 2026-08-11 for a ground-up redesign targeting ICML 2027. It currently holds feasibility spikes only, no system code. `v3/spikes/q1_volatility/` tests whether facts sort into an invariant layer and a volatile one using time-stamped Wikidata, since the proposed architecture rests on that split.
- `v2/` is the **built-out** project: a frozen small LLM core (GPT-2-XL, Qwen2.5-1.5B) plus an external editable memory, two intake gates (plausibility via NLI, truth via FEVER-style verification), and a probe that runs after every fact-merge and auto-rolls-back bad ones. Its headline claim (that a gated loop survives where ungated sequential editing collapses) was undercut in May 2026 by UltraEdit, which sustains 1M sequential edits ungated. Code and tests still green; the framing is what broke.
- `legacy/v1/` is **frozen** (2026-06-05, do not modify): orthogonal-LoRA continual learning + LLM-JEPA aux losses + SIGReg on a frozen NF4 Qwen3-14B. Phase 0 and 1 passed, Phase 2 failed both gates. Kept as reference and for reusable organs (the LoRA bank, the CL metrics).
- `results/` holds v1 phase result JSONs; `docs/` holds the cross-generation audit and the one ADR.

Stack: Python 3.10+ (v1 pinned 3.12), PyTorch, HuggingFace transformers. No PEFT in v1's bank (hand-rolled). Heavy runs go to Kaggle free-tier T4; the local RTX 4060 laptop runs unit tests only.

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

### docs/superpowers/specs/2026-08-11-u-jepa-v3-design.md
The v3 design, written to be read cold by an external reviewer. Reframes the project from "gated editing survives collapse" (dead: UltraEdit sustains 1M ungated edits) to admission control, since modern editors verify nothing. Carries the Q1 result, 5 research questions, the staging with kill switches, an explicit non-goals list and 9 rules traced to v1 audit failures.
Gotcha: supersedes `v2/docs/pipeline-1-build-plan.md`. Section 12 is the reviewer-attack list and section 13 the open items; both are deliberate, not unfinished sections. The first implementation plan covers stages 0-1 only.

### docs/superpowers/plans/2026-08-11-u-jepa-v3-harness-and-rq1.md
TDD implementation plan for v3 stages 0-1: 12 tasks, 60 steps, each with the failing test written out in full. Builds `v3/src/u_jepa_v3/` (env, schema, data loaders, editor protocol over EasyEdit, probes, resumable run state, shard worker, RQ1 driver and analysis).
Gotcha: every task must stay CPU-testable through `StubEditor` and a fake responder, so development runs on the laptop and only real edits go to the H200s. Network and GPU tests gate on `U_JEPA_V3_RUN_NETWORK=1` and `U_JEPA_V3_RUN_GPU=1`. RLEdit is named in the spec but is not confirmed present in EasyEdit; task 7 ships the confirmed methods and documents the one-line addition.

### docs/decisions/2026-05-26-kaggle-pivot.md
The only ADR. Moves heavy compute from the RTX 4060 laptop to Kaggle GPUs, drops the 8 GB VRAM ceiling, bumps the base model to Qwen3-14B. Reasons: vLLM is Linux-only, the 8 GB cap forced a weaker 4B baseline, Kaggle gives free Linux T4s at 30 h/week. Introduces the 9-hour-session and checkpoint-or-die constraints that both generations still live under.

## v3/ (newest, spikes only)

### v3/spikes/q1_volatility/FINDINGS.md
Verdict on the Q1 question, does knowledge split into invariant and volatile layers. Answer: volatility is real, large and predictable (split-half Spearman 0.695), but it is a continuum rather than a binary, so the two-layer design has to become a threshold with a measured error rate. Records the unasked finding that 78% of Wikidata change is new facts rather than revisions, which means the expensive coherence gate only needs to run on the 20% that overwrite.
Gotcha: the 5-month window has low power for recurrence, so low recurrence means absence of evidence and not proof of invariance. Wikidata diffs also conflate world-change with database curation.

### v3/spikes/q1_volatility/load_wikibigedit.py
Downloads the 8 WikiBigEdit Wikidata snapshot diffs (2024-02-01 to 2024-07-01) and flattens them into one dataframe carrying a timestep index.
Exports: REPO_ID; TIMESTEP_FILES (ordered oldest first); KEEP_COLUMNS; LoadReport dataclass; load_all() -> (DataFrame, LoadReport)
Used by: v3/spikes/q1_volatility/analyze_volatility.py, v3/spikes/q1_volatility/analyze_stability.py
Gotcha: TIMESTEP_FILES order defines the timestep index, so never sort it. Drops 11,038 rows carrying a null subject_id or relation_id. First run pulls roughly 190 MB into the HF cache; later runs are offline.

### v3/spikes/q1_volatility/analyze_volatility.py
Measures per-relation churn (share of a relation's rows tagged update rather than new) and recurrence (share of its updated pairs revised in more than one timestep), then tests the distribution shape.
Exports: MIN_SUPPORT (200); tag_breakdown(); recurrence_table(); per_relation_stats(); bimodality(); main()
Gotcha: writes results.json and per_relation.csv beside itself. The bimodality coefficient reads 0.755 against a 0.556 reference but that is right skew rather than two modes, so read the histogram instead.

### v3/spikes/q1_volatility/analyze_stability.py
Separates real churn from one-off Wikidata bot passes, via split-half Spearman between early and late timesteps plus per-relation concentration of updates in a single timestep.
Exports: MIN_SUPPORT; MIN_UPDATES_PER_HALF (20); EARLY, LATE timestep tuples; concentration(); split_half(); main()
Gotcha: needs scipy, which the other two do not. Writes stability.json. The split-half Spearman of 0.695 is the single number the layer-assignment design rests on.

### v3/spikes/q1_volatility/results.json
Generated by analyze_volatility.py: load counts, tag shares, per-pair recurrence histogram, churn and recurrence distribution shape, most and least churning relations.

### v3/spikes/q1_volatility/per_relation.csv
Generated by analyze_volatility.py: one row per relation with n_rows, n_updates, churn, recurrence and pair counts. All 941 relations, unfiltered by support.

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
