# legacy/v1: frozen archive of the original U-JEPA architecture

Frozen on 2026-06-05. Do not modify. Anything new goes under the v2 plan at `v2/docs/pipeline-1-build-plan.md`.

## What this is

The first U-JEPA architecture: "central JEPA brain + domain-specialist sub-agents + V-JEPA vision bridge + N-LoRA continual learning + LLM-JEPA auxiliary losses on a frozen Qwen3-14B". Three phases were attempted, two shipped successfully, the third revealed structural design errors in the auxiliary-loss composition. Full audit is at `../../docs/external_audit_context.md`.

## Layout

```
legacy/v1/
  src/u_jepa/                    the package: bank, losses, eval, train, util, etc.
  tests/                          122 tests (1 network-gated, 3 lejepa-path skips)
  scripts/                        phase entry scripts
    00_smoke_env.py
    01_repro_latentmas_gsm8k.py   (needs vendored/LatentMAS re-cloned to run)
    02_train_continual_phase1.py
    03_train_jepa_aux_phase2.py
  kaggle/                         per-phase Kaggle notebooks + metadata
    phase0/
    phase1/
    phase2/
  pyproject.toml                  package definition (name: u-jepa)
  requirements.txt
  requirements-kaggle.txt
  .python-version
  docs/
    manual_verification_phase0.md
    manual_verification_phase1.md
    superpowers/
      plans/2026-05-26-ujepa-prototype.md         the v1 master plan
      specs/2026-06-03-u-jepa-v2-architecture.md  the brainstorm spec that
                                                  was itself superseded by
                                                  the v2 pipeline plan
```

## How to use this archive

Read `../../docs/external_audit_context.md` first for the project narrative including what worked, what didn't, and why we pivoted. Then dive into specific code if you need to.

To actually run any of this code:

```bash
cd legacy/v1
pip install -e .
pip install -r requirements.txt
pytest tests/                                  # 122 pass, 4 skip
```

To re-run Phase 0 (LatentMAS reproduction) you also need to re-vendor LatentMAS:

```bash
git subtree add --prefix=legacy/v1/vendored/LatentMAS \
    https://github.com/Gen-Verse/LatentMAS main --squash
```

(The cleanup commit `4580a85` deleted the original `vendored/` tree.)

## Reusable bits the v2 plan might still want

- `src/u_jepa/continual/orthogonal_lora.py` and `src/u_jepa/continual/n_lora_loss.py`: the orthogonal LoRA bank from Phase 1, which scored zero forgetting. v2's Phase 5 (consolidation/sleep) is the natural reuse spot.
- `src/u_jepa/models/qwen_base.py`: NF4 loader. Trivial to retarget at Qwen2.5-1.5B or GPT-2-XL for v2's Phase 0 baselines.
- `src/u_jepa/util/env.py` and `src/u_jepa/util/prompting.py`: env detection and chat-template wrapping. Generic enough to lift.
- Kaggle notebook patterns under `kaggle/`: the cleanup/setup/auth/gpu-smoke/run/show-results cell layout shipped reliably across 14 versions of Phase 0 and one good Phase 1 run. Worth copying for v2 notebooks.

Everything else (Phase 2 loss modules, Spider loader, JEPA-aux training loop) is documented as a negative-result reference in the audit, not for v2 to reuse.
