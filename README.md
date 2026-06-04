# U-JEPA

JEPA-based continual-learning research project. The architecture pivoted in June 2026 from "central JEPA brain + sub-agents" to "frozen core + external memory + probe-gated rollback". The new direction lives in `U-JEPA v2/`; the original work is frozen under `legacy/v1/`.

## Where to start

- **`U-JEPA v2/pipeline-1-build-plan.md`**: the current build plan. Phases sized in 9h Kaggle sessions. Headline result is the ROC curve of bad-merges-caught vs false-aborts on good merges.
- **`U-JEPA v2/compass_artifact_*.md`**: the research backing the v2 plan (V-JEPA + world models + editing literature + CLS theory + FEVER + model collapse).
- **`Research.md`**: original ideation that led to all of this.
- **`docs/external_audit_context.md`**: snapshot of project state at 2026-06-03 covering the v1 phases as they shipped. Useful context for anyone reading the repo cold.
- **`legacy/v1/`**: Phase 0 / Phase 1 / Phase 2 code, tests, scripts, kaggle notebooks, v1 plans and specs. Frozen, not maintained, useful as reference. See `legacy/v1/README.md`.
- **`results/`**: experimental numbers from the v1 phases (Phase 0 GSM8K 60.8%, Phase 1 zero forgetting on 2 tasks, Phase 2 gates failed with full diagnosis).

## What v2 changes

The v1 plan tried to compose orthogonal-LoRA continual learning + LLM-JEPA auxiliary losses + V-JEPA vision + a router into one stack on a big LLM (Qwen3-14B). Phase 1 worked (zero forgetting). Phase 2 failed both gates and the post-mortem identified four structural design errors. See `docs/external_audit_context.md` section 8 for the full audit.

v2 ships a different architecture. Small frozen core (Qwen2.5-1.5B + GPT-2-XL baseline). Facts live OUTSIDE the weights in a vector memory plus a GRACE-style codebook. Two gates control what gets in (plausibility via NLI, truth via FEVER-style verification). A probe runs after every merge and auto-rolls back bad ones. The novel claim is the safe-intake loop itself; no editing paper combines automatic abort, dual gating, and rollback.

## Hardware

Kaggle free-tier T4 (15 GB, 9h sessions, 30h/week) for all training. Local RTX 4060 laptop for unit tests, design exploration, paper writing.

## License

Code under MIT. Datasets and any vendored upstream code retain their original licenses.
