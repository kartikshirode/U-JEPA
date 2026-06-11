# U-JEPA v2: Probe-Gated Latent-Memory Continual Learner

The active U-JEPA project. A frozen small LLM core plus an external editable memory, with two intake gates (plausibility, truth) and an auto-rollback probe after every merge. v1 is frozen under `../legacy/v1/`.

The one result that is the paper: of K fact-merges (mix of good and deliberately-bad), what fraction of bad merges did the probe catch and the system roll back, and what was the false-abort rate on good ones, plus the ROC curve as the threshold slides.

## Layout

```
v2/
  pyproject.toml          package definition (u-jepa-v2)
  README.md               this file
  docs/
    pipeline-1-build-plan.md     active build plan (6 phases, gates per phase)
    compass_artifact_*.md        research backing the plan
  src/u_jepa_v2/
    __init__.py
    env.py                runtime + device + dtype detection
    persistence.py        atomic JSON / torch save, RunState, kaggle paths
    models/
      __init__.py
      core.py             frozen-core loader (GPT-2-XL, Qwen2.5-1.5B)
  scripts/
    00_smoke_env.py       Phase 0 step 1: env sanity check
    01_smoke_load_cores.py  Phase 0 step 2: load both cores, report VRAM
  tests/                  pytest suite (26 passing, 1 network-gated skip)
```

Future modules slot in alongside `models/` as phases land: `memory/` (Phase 1), `probe/` (Phase 2), `gates/` (Phase 2, 4), `editing/` (Phase 0-3 substrate), `controller/` (Phase 3 merge-probe-rollback loop), `consolidation/` (Phase 5 O-LoRA sleep).

## Quick start

```bash
cd v2
pip install -e .            # core deps
pip install -e .[dev]       # plus pytest

pytest                      # 26 pass, 1 skip (network-gated)
python scripts/00_smoke_env.py
python scripts/01_smoke_load_cores.py --only gpt2-xl --dtype fp32
```

Enable the network-gated test (downloads GPT-2-XL) with:

```bash
U_JEPA_V2_RUN_NETWORK=1 U_JEPA_V2_RUN_SLOW=1 pytest tests/test_models_core.py -v
```

## Hardware contract

Kaggle T4 (Turing, no native bf16) is the primary training surface, so the code defaults to fp16 everywhere. The local RTX 4060 laptop (Ampere, 8 GB) runs unit tests and small smoke checks but cannot fit either core at full precision; use `--dtype fp32` only for the smallest models or rely on Kaggle for real runs.

Every phase must save its resumable state to a path that survives a kernel restart. `persistence.kaggle_working_dir()` resolves to `/kaggle/working` on Kaggle and `<project root>/checkpoints` locally (the root is found by walking up from the package to the first dir with `.git` or `pyproject.toml`). Set `U_JEPA_V2_CKPT_DIR` to override either. Pair with `persistence.kaggle_input_path(slug)` to pick up a prior session's snapshot.

## Reproducibility

Dependency pins are floor-only, so each fresh install can resolve different library versions. Two habits keep the paper's numbers traceable:

- Run `pip freeze > freeze-phase<N>.txt` at the start of every phase and keep the file next to that phase's results.
- Results JSON written through `env.summary()` records `torch_version` and `transformers_version`, so any number can be matched to the libraries that produced it.

## Phase status

- **Phase 0** (foundations and GRACE baseline): scaffold + env + persistence + core loader done. GRACE baseline + EasyEdit harness still to build.
- **Phase 1** (memory loop): not started.
- **Phase 2-6**: not started.

See `docs/pipeline-1-build-plan.md` for gates, fallbacks, and the open-forks resolution table.
