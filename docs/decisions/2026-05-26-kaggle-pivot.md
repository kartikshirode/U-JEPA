# ADR: Move heavy compute to Kaggle, drop the 8 GB laptop ceiling

**Date:** 2026-05-26
**Status:** Accepted
**Context:** Phase 0 of the implementation plan, before any training has run.

## Decision

Switch the U-JEPA prototype from a strict "fits on RTX 4060 8 GB laptop" target to a "develop locally, train and eval on Kaggle GPUs" workflow. Drop the laptop VRAM ceiling as a hard constraint. Default base LLM moves from Qwen3-4B to **Qwen3-14B-Instruct** (the model LatentMAS was actually demonstrated on). Vision encoder stays V-JEPA 2 ViT-L for now, with ViT-g as a stretch option on dual-T4 sessions.

## Why

Three forcing factors hit at once during Phase 0 setup:

1. **vLLM is Linux-only.** LatentMAS imports `from vllm import SamplingParams` at the top of its method file. vLLM has no Windows wheels. Running the vendored LatentMAS code as-is on Windows is not possible; a HF-only rewrite would cost 1-2 days of Phase 0 budget and would diverge from the validated implementation.

2. **The 8 GB ceiling was forcing us onto Qwen3-4B.** LatentMAS was demonstrated on Qwen3-14B, and the smallest model tested in the paper was 4B. Running on the smaller model meant carrying through a constant "we expect weaker baselines" caveat in the paper.

3. **Kaggle gives free Linux GPUs.** Tesla P100 16 GB, T4 16 GB, and dual T4 (2x 16 GB = 32 GB) are available at no cost, with 30 hours per week of GPU credit. A 16 GB T4 fits Qwen3-14B-NF4 (~8 GB) plus V-JEPA 2 ViT-L plus adapters plus KV cache comfortably.

## Consequences

### Architecture upgrades
- Base LLM: Qwen3-4B → **Qwen3-14B-Instruct** (with Qwen3-32B-Instruct as a stretch on 2xT4)
- Vision: V-JEPA 2 ViT-L stays as default, ViT-g/16 (~1B params) available as a stretch
- Router: Phi-3.5-mini-Q4 unchanged
- LoRA rank: can go higher (32 or 64) with more VRAM available

### Workflow changes
- Local laptop: code editing, unit tests on CPU or small CUDA tests, paper writing
- Push to GitHub main
- Kaggle notebook clones repo, installs deps, runs training/eval, saves outputs
- Outputs come back as Kaggle Datasets that local can pull via the kaggle CLI

### Session-management constraints (new)
- Kaggle has a **9 hour session limit**. Every training script must support checkpoint and resume.
- Each session must `pip install` and warm the HuggingFace cache. We pre-stage Qwen3-14B and V-JEPA 2 as Kaggle Datasets to skip re-download.
- 30 hours per week budget is the new resource ceiling. Schedule training runs against it.

### Headline-claim change
The "fits on a single 8 GB consumer GPU" line in the theoretical paper draft no longer holds for training. Inference on the trained adapters may still fit on the 4060 (LoRA stacks are tiny), so the narrative becomes "we train on free cloud GPUs and deploy on consumer hardware." Update U-JEPA_Paper_Draft.md accordingly when results land.

### Things this does NOT change
- Approach 1 (fork LatentMAS + JEPA + N-LoRA) is still the chosen architecture
- All the phase gates and metrics in the implementation plan still apply
- AUTO-COMMIT after every task, vendored read-only, frozen-base rules all still apply

## Alternatives considered

| Option | Reason rejected |
|--------|-----------------|
| Stay on Windows, write HF-only LatentMAS shim | Costs 1-2 days, diverges from validated impl, keeps us on the weaker 4B baseline. |
| Move dev to WSL2 with GPU passthrough | Adds a permanent layer to the workflow, GPU sharing across Win and WSL2 has driver quirks, still capped at 8 GB. |
| Rent A100 spot instances | $200-500 budget overhead. Worth doing later for the final benchmark runs but not for Phase 0 iteration. |

## Action items unblocked by this decision

1. Bump `requirements.txt` to include `vllm`, drop the strict CUDA 12.4 wheel index (Kaggle handles drivers).
2. Add a `kaggle/` directory with notebook templates and a README documenting the cycle.
3. Add `src/u_jepa/util/kaggle.py` with environment detection so the same scripts work locally and on Kaggle.
4. Update Task 0.4 and 0.5 in the implementation plan to target Qwen3-14B and the Kaggle notebook runner.
5. Pre-stage Qwen3-14B-Instruct and V-JEPA 2 ViT-L as Kaggle Datasets (one-time manual step for the user).
