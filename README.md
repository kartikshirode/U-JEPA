# U-JEPA

Prototype implementation of a unified JEPA-based architecture for continual multimodal learning on a single 8 GB consumer GPU.

## What this is

U-JEPA pairs a frozen V-JEPA 2 visual encoder with a frozen Qwen3-4B language backbone, aligns them in a shared latent space, and adapts to new domains via orthogonal LoRA adapters governed by a lightweight Phi-3.5-mini router. Cross-agent communication piggybacks on the LatentMAS KV-cache mechanism, and continual learning happens in embedding space through N-LoRA orthogonality plus LLM-JEPA auxiliary losses.

Headline claim: zero-retraining adaptation to new domains on an RTX 4060 8 GB laptop, with bounded forgetting and JEPA-style latent reasoning.

## Status

Phase 0: bootstrap. Reproducing the LatentMAS baseline on Qwen3-4B-Q4 before any architectural changes.

## Docs

- Theoretical paper draft: U-JEPA_Paper_Draft.md (formerly compass_artifact_*.md)
- Prototype design spec: Research.md
- Implementation plan: docs/superpowers/plans/2026-05-26-ujepa-prototype.md
- Original brainstorm: JEPA_Research_Context.md

## Hardware

Built for an NVIDIA RTX 4060 Laptop GPU (8188 MiB VRAM). Every design choice is constrained by that ceiling.

## License

Code under MIT. Vendored upstream code retains its original licenses. V-JEPA 2 weights are CC-BY-NC 4.0; this repo is for research use only.
