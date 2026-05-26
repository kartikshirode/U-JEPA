# Vendored upstream repos

Pinned via `git subtree add --squash` so the code lives locally but upstream history stays referenceable. To bump: `git subtree pull --prefix=vendored/<name> <url> main --squash`.

LFS smudge was skipped during vendoring (`GIT_LFS_SKIP_SMUDGE=1`). Pointer files for LFS-tracked artifacts (e.g. `vendored/llm-jepa/datasets/*.jsonl`, `vendored/llm-jepa/spider_data.zip`) are present as pointers, not actual content. We do not need them; this project loads datasets via the HuggingFace `datasets` library.

| Path | Upstream | SHA at vendor time | Purpose |
|------|----------|--------------------|---------|
| vendored/LatentMAS | Gen-Verse/LatentMAS | bf8174b | Multi-agent latent communication substrate. Entry points: `methods/latent_mas.py`, `models.py`, `run.py`. |
| vendored/llm-jepa | rbalestr-lab/llm-jepa | ea0017c | LLM-JEPA auxiliary loss reference impl. |
| vendored/lejepa | rbalestr-lab/lejepa | c293d29 | SIGReg collapse prevention; `lejepa/multivariate` and `lejepa/univariate`. |
| vendored/N-LoRA | PKU-YuanGroup/N-LoRA | 7301afe | Orthogonal + non-collision LoRA penalty. |
| vendored/Online-LoRA | Christina200/Online-LoRA-official | 96e7de0 | Loss-spike task-shift detector. |

## Rule: read-only

Do not edit anything under `vendored/`. If you need to change behavior, write a shim in `src/u_jepa/` that imports the upstream module and overrides what you need. This keeps `git subtree pull` painless.
