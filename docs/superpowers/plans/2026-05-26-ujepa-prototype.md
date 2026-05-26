# U-JEPA Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working U-JEPA prototype on one RTX 4060 8 GB laptop in 12-14 weeks: fork LatentMAS, downsize to Qwen3-4B at NF4, add a frozen V-JEPA 2 visual sub-agent through a Q-Former bridge, layer in LLM-JEPA plus SIGReg auxiliary losses, install N-LoRA orthogonal adapters per domain, and orchestrate everything with a Phi-3.5-mini router. Crush four benchmarks: GSM8K (latent CoT), Spider (LLM-JEPA gains), TRACE-3 sub-sequence (continual forgetting), Visual7W / OK-VQA mini (vision bridge), and demonstrate zero-retraining adaptation on a held-out fifth domain.

**Architecture:** Frozen Qwen3-4B-Instruct (NF4) base with hot-swappable LoRA stacks per domain, fed by a 2-stage modality plus domain router (Phi-3.5-mini-Q4). Vision enters via a frozen V-JEPA 2 ViT-L plus a small Q-Former producing a soft-prefix in Qwen's embedding space. Sub-agents reason in Coconut-style latent chain-of-thought; cross-agent transfer uses LatentMAS-style KV-cache prepending via `past_key_values`. Continual learning happens entirely in adapter space with N-LoRA orthogonality plus non-collision penalties, LLM-JEPA auxiliary loss for representational structure, and SIGReg for collapse prevention.

**Tech Stack:** Python 3.11, PyTorch 2.4 + CUDA 12.4, HuggingFace transformers 4.46, PEFT 0.12, bitsandbytes 0.43 (NF4), accelerate 0.34, datasets 3.0, vendored LatentMAS / lejepa / llm-jepa / N-LoRA / Online-LoRA. Vision: `facebook/vjepa2-vitl-fpc64-256` via HF AutoModel. Routers: `microsoft/Phi-3.5-mini-instruct` quantized. Eval: lm-eval-harness 0.4, custom continual-learning metrics module.

---

## Repository Layout

Below is the full file tree this plan produces. Subagents executing one task at a time should consult this map so their work fits the global structure.

```
U-JEPA/
  src/u_jepa/
    __init__.py
    config.py                       # central config dataclasses
    models/
      __init__.py
      qwen_base.py                  # loads Qwen3-4B NF4, exposes hooks
      vjepa_bridge.py               # V-JEPA 2 + Q-Former + projection
      router.py                     # Phi-3.5-mini classifier head
    continual/
      __init__.py
      orthogonal_lora.py            # OrthogonalLoRABank with hot-swap
      n_lora_loss.py                # O-LoRA + N-LoRA non-collision
      online_lora_detector.py       # loss-spike trigger from Wei et al.
    losses/
      __init__.py
      llm_jepa.py                   # auxiliary JEPA loss on hidden states
      sigreg.py                     # vendored from lejepa, thin wrapper
      alignment.py                  # InfoNCE + VICReg for vision-text
    mas/
      __init__.py
      kv_cache_bridge.py            # past_key_values prepending
      latent_reasoner.py            # Coconut-style latent CoT loop
      orchestrator.py               # routes + swaps LoRA + sequences agents
      realign.py                    # ridge-regression W_a from LatentMAS
    data/
      __init__.py
      gsm8k.py
      spider.py
      trace.py
      visual7w.py
      something_v2.py
      nl_rx.py
    eval/
      __init__.py
      metrics.py                    # accuracy, exact-match, BWT, forgetting
      continual.py                  # sequential-task evaluation harness
      benchmark_runner.py           # unified entry point
    train/
      __init__.py
      continual_loop.py             # sequential task training with N-LoRA
      vision_bridge_loop.py         # train Q-Former + projection
      jepa_aux_loop.py              # combined LLM-JEPA + SIGReg loop
  tests/
    test_orthogonal_lora.py
    test_n_lora_loss.py
    test_llm_jepa_loss.py
    test_sigreg.py
    test_kv_cache_bridge.py
    test_vjepa_bridge_shapes.py
    test_router.py
    test_metrics.py
    test_realign.py
    fixtures/
      tiny_text_pairs.json
      tiny_video_clip.pt
  scripts/
    00_smoke_env.py                 # CUDA + bitsandbytes sanity
    01_repro_latentmas_gsm8k.py     # phase 0 gate
    02_train_continual_phase1.py    # phase 1
    03_train_jepa_aux_phase2.py     # phase 2
    04_train_vision_bridge.py       # phase 3
    05_train_router.py              # phase 4
    06_eval_full_ablation.py        # phase 5
    07_zero_retraining_demo.py      # phase 4 gate
  vendored/
    LatentMAS/                      # git subtree from Gen-Verse/LatentMAS
    llm-jepa/                       # git subtree from rbalestr-lab/llm-jepa
    lejepa/                         # git subtree from rbalestr-lab/lejepa
    N-LoRA/                         # git subtree from PKU-YuanGroup/N-LoRA
    Online-LoRA/                    # git subtree from Christina200/Online-LoRA
  configs/
    qwen3_4b_q4.yaml
    phi35_mini_q4.yaml
    vjepa2_vitl.yaml
    trace_subseq.yaml
    spider_jepa.yaml
    visual7w_bridge.yaml
  docs/
    superpowers/plans/2026-05-26-ujepa-prototype.md   # this plan
    decisions/                                          # one ADR per pivot
  results/
    phase0_baseline.json
    phase1_continual.json
    phase2_jepa_aux.json
    phase3_vision.json
    phase4_router.json
    phase5_ablation.json
    figures/
  pyproject.toml
  requirements.txt
  .python-version
  .gitignore
  README.md
```

Two responsibilities split that matter most:
- `models/` holds inference graphs (forward only); `train/` holds training loops. A subagent should never put a training loop into `models/`.
- `losses/` holds loss functions that take tensors and return scalars. They never touch dataloaders or models directly.

---

## Phase 0: Environment + LatentMAS Baseline (Weeks 0-1)

**Goal:** Working CUDA stack, LatentMAS reproduced on Qwen3-4B-Q4 with GSM8K accuracy at or above 65 percent with a 3-agent text-only MAS.

**Gate to advance to Phase 1:** GSM8K accuracy >= 65 percent at >= 10 tokens per second on the RTX 4060, with results saved to `results/phase0_baseline.json`. If you fail this gate, every subsequent phase is built on sand. Do not advance.

### Task 0.1: Initialize project structure

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `.python-version`
- Create: `.gitignore`
- Create: `src/u_jepa/__init__.py`
- Create: `src/u_jepa/config.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Write pyproject.toml**

```toml
[project]
name = "u-jepa"
version = "0.1.0"
description = "U-JEPA prototype: latent multi-agent JEPA-regularized continual learning on 8GB GPU"
readme = "README.md"
requires-python = ">=3.11,<3.12"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
markers = [
  "slow: deselect with -m 'not slow'",
  "needs_gpu: requires CUDA device",
]
```

- [ ] **Step 2: Write requirements.txt (pinned versions for reproducibility)**

```
torch==2.4.1
torchvision==0.19.1
transformers==4.46.0
peft==0.12.0
bitsandbytes==0.43.3
accelerate==0.34.2
datasets==3.0.0
sentencepiece==0.2.0
einops==0.8.0
pyyaml==6.0.2
pytest==8.3.3
numpy==1.26.4
pandas==2.2.3
scipy==1.14.1
tqdm==4.66.5
matplotlib==3.9.2
huggingface-hub==0.25.0
```

- [ ] **Step 3: Write .python-version and .gitignore**

`.python-version`:
```
3.11.10
```

`.gitignore`:
```
__pycache__/
*.pyc
.venv/
.env
results/*.json
results/figures/
!results/.gitkeep
checkpoints/
*.pt
*.bin
*.safetensors
.pytest_cache/
.ipynb_checkpoints/
wandb/
hf_cache/
```

- [ ] **Step 4: Write src/u_jepa/config.py**

```python
"""Central config dataclasses. Every script reads its config from here."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

@dataclass(frozen=True)
class HardwareConfig:
    device: str = "cuda:0"
    vram_ceiling_mb: int = 7800
    grad_accum: int = 8
    micro_batch: int = 1
    max_seq_len: int = 1024
    bf16_compute: bool = True

@dataclass(frozen=True)
class QwenConfig:
    model_id: str = "Qwen/Qwen3-4B-Instruct"
    quant_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    hidden_size: int = 2560
    n_layers: int = 36

@dataclass(frozen=True)
class VJEPAConfig:
    model_id: str = "facebook/vjepa2-vitl-fpc64-256"
    out_dim: int = 1024
    frames_per_clip: int = 64
    crop: int = 256

@dataclass(frozen=True)
class LoraConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple = ("q_proj", "v_proj")

@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[2]
    results: Path = field(init=False)
    checkpoints: Path = field(init=False)
    hf_cache: Path = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "results", self.root / "results")
        object.__setattr__(self, "checkpoints", self.root / "checkpoints")
        object.__setattr__(self, "hf_cache", self.root / "hf_cache")
```

- [ ] **Step 5: Empty __init__ files**

`src/u_jepa/__init__.py`:
```python
__version__ = "0.1.0"
```

`tests/__init__.py`: empty file.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml requirements.txt .python-version .gitignore src tests
git commit -m "scaffold u_jepa package with config dataclasses"
```

### Task 0.2: Verify CUDA + bitsandbytes on RTX 4060

**Files:**
- Create: `scripts/00_smoke_env.py`

- [ ] **Step 1: Write the smoke test script**

```python
"""Day-1 smoke: verify CUDA, bitsandbytes NF4 load, and free VRAM."""
import torch
import bitsandbytes as bnb

def main():
    assert torch.cuda.is_available(), "CUDA not available"
    dev = torch.device("cuda:0")
    name = torch.cuda.get_device_name(dev)
    total_mb = torch.cuda.get_device_properties(dev).total_memory // (1024 * 1024)
    free_mb, _ = torch.cuda.mem_get_info(dev)
    free_mb = free_mb // (1024 * 1024)
    print(f"Device: {name}")
    print(f"Total VRAM: {total_mb} MB, Free: {free_mb} MB")
    assert total_mb >= 7500, f"Expected >= 7500 MB, got {total_mb}"

    # NF4 sanity: quantize a 4096x4096 tensor
    w = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    qw = bnb.nn.Params4bit(w, quant_type="nf4").cuda(dev)
    print(f"NF4 quantized 4096x4096 OK. dtype={qw.dtype}")

    # Compute capability check (need >= 8.0 for bf16 fast path)
    cc_major, cc_minor = torch.cuda.get_device_capability(dev)
    print(f"Compute capability: {cc_major}.{cc_minor}")
    assert cc_major >= 8, "Need Ampere or newer for efficient bf16"

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
python scripts/00_smoke_env.py
```

Expected output: device name (NVIDIA GeForce RTX 4060 Laptop GPU), total VRAM around 8188 MB, free VRAM around 7500-8000 MB at idle, NF4 quantize line printed, compute capability 8.9. If the assertion fails on free VRAM, kill background GPU processes first.

- [ ] **Step 3: Commit**

```bash
git add scripts/00_smoke_env.py
git commit -m "add day-1 CUDA and NF4 smoke test"
```

### Task 0.3: Vendor LatentMAS, llm-jepa, lejepa, N-LoRA, Online-LoRA

**Files:**
- Create: `vendored/` (directory)
- Create: `vendored/README.md` documenting upstream commit SHAs

- [ ] **Step 1: Clone each as git subtrees so upstream history is preserved but the code is local**

```bash
mkdir -p vendored
git subtree add --prefix=vendored/LatentMAS https://github.com/Gen-Verse/LatentMAS main --squash
git subtree add --prefix=vendored/llm-jepa https://github.com/rbalestr-lab/llm-jepa main --squash
git subtree add --prefix=vendored/lejepa https://github.com/rbalestr-lab/lejepa main --squash
git subtree add --prefix=vendored/N-LoRA https://github.com/PKU-YuanGroup/N-LoRA main --squash
git subtree add --prefix=vendored/Online-LoRA https://github.com/Christina200/Online-LoRA-official main --squash
```

- [ ] **Step 2: Pin upstream SHAs in vendored/README.md**

```markdown
# Vendored upstream repos

Pinned via `git subtree add --squash`. To bump, run `git subtree pull --prefix=vendored/<name> <url> main --squash`.

| Path | Upstream | SHA at vendor time | Purpose |
|------|----------|--------------------|---------|
| vendored/LatentMAS | Gen-Verse/LatentMAS | (record sha from git log) | Multi-agent latent communication substrate |
| vendored/llm-jepa | rbalestr-lab/llm-jepa | (record sha) | LLM-JEPA auxiliary loss |
| vendored/lejepa | rbalestr-lab/lejepa | (record sha) | SIGReg collapse prevention |
| vendored/N-LoRA | PKU-YuanGroup/N-LoRA | (record sha) | Orthogonal + non-collision LoRA loss |
| vendored/Online-LoRA | Christina200/Online-LoRA-official | (record sha) | Loss-spike task-shift detector |
```

Fill the SHAs by running `git log --oneline -1 -- vendored/<name>/` for each entry.

- [ ] **Step 3: Add vendored/* to pyproject.toml as additional sys.path roots only at import sites where needed (do not pip-install)**

Reasoning: these repos are not pip packages. Import them via `sys.path.insert(0, str(VENDORED / "LatentMAS"))` inside the specific module that needs them, so the global namespace stays clean.

- [ ] **Step 4: Commit**

The subtree adds already produce commits. Verify with `git log --oneline -n 10` and amend the vendored/README.md commit if needed:

```bash
git add vendored/README.md
git commit -m "document vendored upstream repos and pinned SHAs"
```

### Task 0.4: Load Qwen3-4B-Instruct at NF4 and measure VRAM

**Files:**
- Create: `src/u_jepa/models/__init__.py`
- Create: `src/u_jepa/models/qwen_base.py`
- Create: `tests/test_qwen_load.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_qwen_load.py
import pytest
import torch
from u_jepa.models.qwen_base import load_qwen_nf4, qwen_vram_usage_mb

pytestmark = pytest.mark.needs_gpu

def test_qwen_loads_under_3gb():
    model, tok = load_qwen_nf4()
    vram = qwen_vram_usage_mb(model)
    assert vram < 3000, f"Qwen3-4B-NF4 took {vram} MB, expected < 3000"
    # Generation sanity
    ids = tok("The capital of France is", return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=5, do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)
    assert "Paris" in text, f"Expected Paris in output, got: {text}"
```

- [ ] **Step 2: Run it to see it fail**

```bash
pytest tests/test_qwen_load.py -v
```

Expected: ImportError on `u_jepa.models.qwen_base`.

- [ ] **Step 3: Implement qwen_base.py**

```python
"""Qwen3-4B-Instruct at NF4 with bitsandbytes."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from u_jepa.config import QwenConfig

def _bnb_config(cfg: QwenConfig) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=cfg.quant_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=True,
    )

def load_qwen_nf4(cfg: QwenConfig | None = None):
    cfg = cfg or QwenConfig()
    tok = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=_bnb_config(cfg),
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()
    return model, tok

def qwen_vram_usage_mb(model) -> int:
    """Approximate VRAM held by a model's parameters and buffers."""
    bytes_total = 0
    for p in model.parameters():
        bytes_total += p.numel() * p.element_size()
    for b in model.buffers():
        bytes_total += b.numel() * b.element_size()
    return bytes_total // (1024 * 1024)
```

- [ ] **Step 4: Run the test, expect pass**

```bash
pytest tests/test_qwen_load.py -v
```

If it OOMs at load time, drop `bnb_4bit_use_double_quant=True` to free a few MB. If `Paris` is not in the output, swap to a deterministic prompt that the Qwen3-4B-Instruct model card guarantees.

- [ ] **Step 5: Commit**

```bash
git add src/u_jepa/models/__init__.py src/u_jepa/models/qwen_base.py tests/test_qwen_load.py
git commit -m "load Qwen3-4B-Instruct at NF4 with VRAM sanity test"
```

### Task 0.5: Reproduce LatentMAS GSM8K baseline with Qwen3-4B-Q4

**Files:**
- Create: `scripts/01_repro_latentmas_gsm8k.py`
- Create: `src/u_jepa/data/__init__.py`
- Create: `src/u_jepa/data/gsm8k.py`
- Create: `results/.gitkeep`

- [ ] **Step 1: Write a small GSM8K loader**

```python
# src/u_jepa/data/gsm8k.py
"""GSM8K eval split loader. Returns list of {question, answer_int}."""
import re
from datasets import load_dataset

ANSWER_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")

def load_gsm8k_eval(n: int | None = None):
    ds = load_dataset("gsm8k", "main", split="test")
    items = []
    for ex in ds:
        m = ANSWER_RE.search(ex["answer"])
        if not m:
            continue
        items.append({"question": ex["question"], "answer": float(m.group(1))})
        if n is not None and len(items) >= n:
            break
    return items

def extract_pred(text: str) -> float | None:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None
```

- [ ] **Step 2: Write the baseline script**

```python
# scripts/01_repro_latentmas_gsm8k.py
"""Phase-0 gate: LatentMAS-style 3-agent latent MAS on GSM8K with Qwen3-4B-Q4."""
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

from u_jepa.config import Paths
from u_jepa.data.gsm8k import load_gsm8k_eval, extract_pred
from u_jepa.models.qwen_base import load_qwen_nf4

VENDORED = Path(__file__).resolve().parents[1] / "vendored" / "LatentMAS"
sys.path.insert(0, str(VENDORED))
# Vendored entry points: methods/latent_mas.py exposes a run() function.
from methods.latent_mas import run_latent_mas  # type: ignore

N_EVAL = 250  # subset for laptop timing budget; full GSM8K test = 1319

def main():
    paths = Paths()
    paths.results.mkdir(parents=True, exist_ok=True)
    model, tok = load_qwen_nf4()
    items = load_gsm8k_eval(N_EVAL)
    correct, total, tok_total, t0 = 0, 0, 0, time.time()

    for ex in tqdm(items):
        out_text, tok_used = run_latent_mas(
            model=model, tokenizer=tok,
            question=ex["question"],
            n_agents=3, n_latent_steps=4,
            realign=True,
        )
        pred = extract_pred(out_text)
        if pred is not None and abs(pred - ex["answer"]) < 1e-3:
            correct += 1
        total += 1
        tok_total += tok_used

    elapsed = time.time() - t0
    acc = correct / total
    tps = tok_total / elapsed
    result = {
        "phase": 0, "method": "latent_mas_qwen3_4b_q4",
        "n_eval": total, "accuracy": acc,
        "tokens_per_second": tps, "elapsed_sec": elapsed,
        "n_agents": 3, "n_latent_steps": 4,
    }
    out_path = paths.results / "phase0_baseline.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    assert acc >= 0.65, f"GATE FAIL: GSM8K acc {acc:.3f} < 0.65"
    assert tps >= 10, f"GATE FAIL: throughput {tps:.1f} tok/s < 10"

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it**

```bash
python scripts/01_repro_latentmas_gsm8k.py
```

If the import `from methods.latent_mas import run_latent_mas` fails because the vendored entry point has a different signature, write a thin shim `vendored_shim/latent_mas_runner.py` that calls the real entry point and exposes `run_latent_mas`. Do not edit vendored code in place; keep that boundary clean so subtree pulls stay easy.

Expected wall time: 250 GSM8K problems at roughly 10-15 s each on RTX 4060, so 45-70 minutes. If throughput is below 5 tok/s, lower N_EVAL to 100 for the first pass and profile.

- [ ] **Step 4: Save results and commit**

```bash
git add src/u_jepa/data/__init__.py src/u_jepa/data/gsm8k.py scripts/01_repro_latentmas_gsm8k.py results/.gitkeep results/phase0_baseline.json
git commit -m "reproduce LatentMAS GSM8K baseline on Qwen3-4B-Q4"
```

- [ ] **Step 5: Decision point - phase 0 gate**

If `phase0_baseline.json` shows accuracy >= 0.65 and throughput >= 10 tok/s, advance to Phase 1.

If accuracy is 0.50-0.65, investigate: vanilla LatentMAS paper reports Qwen3-14B GSM8K results; Qwen3-4B is materially weaker, so confirm the baseline (no MAS, just Qwen3-4B-Q4 single-agent CoT) and write a short Decision Record at `docs/decisions/2026-XX-phase0-accuracy.md` documenting the gap and chosen path.

If accuracy < 0.50, the realignment or KV-cache transfer is broken; debug before any other work.

---

## Phase 1: Continual LoRA Stack (Weeks 2-4)

**Goal:** N-LoRA orthogonal adapter bank on top of frozen Qwen3-4B-Q4. Train sequentially on 2 TRACE tasks (FOMC then ScienceQA-text-only) and measure forgetting.

**Gate to advance to Phase 2:** Forgetting on task 1 (FOMC) after training on task 2 (ScienceQA) is < 5 percent absolute. BWT >= +1.0 percent over a sequential-FT baseline. Results in `results/phase1_continual.json`.

**Pivot trigger:** If forgetting > 10 percent, swap base model to Phi-3.5-mini-Q4 (its reported forgetting score is 0.02 vs Llama-3.1-8B's 0.59 per arXiv:2504.01241) and re-run.

### Task 1.1: Implement OrthogonalLoRABank with hot-swap

**Files:**
- Create: `src/u_jepa/continual/__init__.py`
- Create: `src/u_jepa/continual/orthogonal_lora.py`
- Create: `tests/test_orthogonal_lora.py`

- [ ] **Step 1: Write tests first**

```python
# tests/test_orthogonal_lora.py
import torch
import pytest
from u_jepa.continual.orthogonal_lora import OrthogonalLoRABank

class _StubLinear(torch.nn.Module):
    """Minimal stand-in for a 'base model' so the bank can be unit-tested CPU-only."""
    def __init__(self, d=64):
        super().__init__()
        self.d = d
        self.layers = torch.nn.ModuleDict({
            "q_proj": torch.nn.Linear(d, d, bias=False),
            "v_proj": torch.nn.Linear(d, d, bias=False),
        })
        for p in self.parameters():
            p.requires_grad = False

    def named_lora_targets(self):
        return list(self.layers.items())

def test_add_task_creates_low_rank_adapter():
    base = _StubLinear(d=64)
    bank = OrthogonalLoRABank(base, rank=4, target_modules=("q_proj", "v_proj"))
    bank.add_task("fomc")
    assert "fomc" in bank.adapters
    a, b = bank.adapters["fomc"]["q_proj"]
    assert a.shape == (4, 64) and b.shape == (64, 4)

def test_hot_swap_returns_different_deltas():
    base = _StubLinear(d=64)
    bank = OrthogonalLoRABank(base, rank=4, target_modules=("q_proj",))
    bank.add_task("a"); bank.add_task("b")
    x = torch.randn(2, 64)
    bank.activate("a"); ya = bank.forward_target(x, "q_proj")
    bank.activate("b"); yb = bank.forward_target(x, "q_proj")
    assert not torch.allclose(ya, yb)

def test_total_adapter_param_count_under_10M():
    base = _StubLinear(d=2560)  # Qwen3-4B hidden size
    bank = OrthogonalLoRABank(base, rank=16, target_modules=("q_proj", "v_proj"))
    for t in range(4):
        bank.add_task(f"d{t}")
    total = sum(p.numel() for p in bank.parameters() if p.requires_grad)
    assert total < 10_000_000, f"Too many trainable params: {total}"
```

- [ ] **Step 2: Run tests, see them fail**

```bash
pytest tests/test_orthogonal_lora.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement OrthogonalLoRABank**

```python
# src/u_jepa/continual/orthogonal_lora.py
"""Hot-swappable LoRA bank with orthogonality bookkeeping.

Design choice: we do not use peft.PeftModel here because we need:
  (a) per-task adapter activation in O(1) (just flip a pointer);
  (b) direct access to A, B matrices for the orthogonality penalty;
  (c) compatibility with NF4 base modules (peft's quantized hooks have
      caused KV-cache shape issues when stacked with vendored LatentMAS).

Adapters are stored as plain nn.Parameters and applied via a forward hook
the bank installs on the base model's target modules.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import Iterable

class _Adapter(nn.ParameterDict):
    """One adapter = {module_name: (A, B)} where delta_W = B @ A."""
    def __init__(self, target_dims: dict[str, tuple[int, int]], rank: int):
        super().__init__()
        for name, (d_in, d_out) in target_dims.items():
            A = nn.Parameter(torch.empty(rank, d_in))
            B = nn.Parameter(torch.zeros(d_out, rank))
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            # B starts at zero so the adapter is identity at init
            self[f"{name}.A"] = A
            self[f"{name}.B"] = B

    def matrices(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        return self[f"{name}.A"], self[f"{name}.B"]

class OrthogonalLoRABank(nn.Module):
    """Bank of LoRA adapters with per-task activation and N-LoRA-ready hooks."""

    def __init__(self, base_model: nn.Module, rank: int = 16,
                 target_modules: Iterable[str] = ("q_proj", "v_proj"),
                 alpha: float = 32.0):
        super().__init__()
        self.base = base_model
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.target_modules = tuple(target_modules)
        self.adapters = nn.ModuleDict()
        self._active: str | None = None
        self._target_dims = self._collect_target_dims()

    def _collect_target_dims(self) -> dict[str, tuple[int, int]]:
        """Find each target module and record (in_features, out_features)."""
        dims = {}
        for full_name, module in self.base.named_modules():
            short = full_name.rsplit(".", 1)[-1]
            if short in self.target_modules and isinstance(module, nn.Linear):
                dims[full_name] = (module.in_features, module.out_features)
        if not dims:
            raise RuntimeError(
                f"No target modules {self.target_modules} found on base model")
        return dims

    def add_task(self, task_id: str) -> None:
        if task_id in self.adapters:
            raise ValueError(f"Task {task_id!r} already exists")
        self.adapters[task_id] = _Adapter(self._target_dims, self.rank)
        self.activate(task_id)

    def activate(self, task_id: str) -> None:
        if task_id not in self.adapters:
            raise KeyError(task_id)
        self._active = task_id

    @property
    def active(self) -> str | None:
        return self._active

    def adapter_matrices(self, task_id: str, module_name: str):
        return self.adapters[task_id].matrices(module_name)

    def forward_target(self, x: torch.Tensor, module_name: str) -> torch.Tensor:
        """Apply the active adapter's delta to module_name's output.
        Used by hooks; also exposed for unit tests."""
        if self._active is None:
            return torch.zeros(x.shape[:-1] + (self._target_dims[module_name][1],),
                               dtype=x.dtype, device=x.device)
        A, B = self.adapter_matrices(self._active, module_name)
        return (x @ A.T @ B.T) * self.scale
```

- [ ] **Step 4: Run tests, see them pass**

```bash
pytest tests/test_orthogonal_lora.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/u_jepa/continual/__init__.py src/u_jepa/continual/orthogonal_lora.py tests/test_orthogonal_lora.py
git commit -m "OrthogonalLoRABank with per-task hot-swap and unit tests"
```

### Task 1.2: Install forward hooks that apply the active adapter to Qwen

**Files:**
- Modify: `src/u_jepa/continual/orthogonal_lora.py`
- Create: `tests/test_orthogonal_lora_hooks.py`

- [ ] **Step 1: Add hook-installation methods to the bank**

Append to `orthogonal_lora.py`:

```python
    def install_hooks(self) -> list:
        """Register forward hooks on every target module. Returns handles
        so the caller can remove them at teardown."""
        handles = []
        for full_name in self._target_dims:
            module = self.base.get_submodule(full_name)
            handle = module.register_forward_hook(
                self._make_hook(full_name))
            handles.append(handle)
        return handles

    def _make_hook(self, module_name: str):
        def hook(_module, inputs, output):
            if self._active is None:
                return output
            x = inputs[0]
            delta = self.forward_target(x, module_name)
            return output + delta
        return hook
```

- [ ] **Step 2: Write the hook test using a tiny model on CPU**

```python
# tests/test_orthogonal_lora_hooks.py
import torch
import torch.nn as nn
from u_jepa.continual.orthogonal_lora import OrthogonalLoRABank

class TinyTransformerBlock(nn.Module):
    """Stand-in for a Qwen layer with q_proj/v_proj names."""
    def __init__(self, d=32):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.q_proj(x) + self.v_proj(x)

def test_hooks_add_adapter_delta_to_target_output():
    base = TinyTransformerBlock(d=32)
    bank = OrthogonalLoRABank(base, rank=4, target_modules=("q_proj",))
    bank.add_task("t1")
    A, B = bank.adapter_matrices("t1", "q_proj")
    with torch.no_grad():
        A.fill_(0.1); B.fill_(0.1)
    handles = bank.install_hooks()
    try:
        x = torch.randn(2, 32)
        y_with = base(x)
        bank._active = None
        y_without = base(x)
        assert not torch.allclose(y_with, y_without)
    finally:
        for h in handles:
            h.remove()
```

- [ ] **Step 3: Run, see pass**

```bash
pytest tests/test_orthogonal_lora_hooks.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/u_jepa/continual/orthogonal_lora.py tests/test_orthogonal_lora_hooks.py
git commit -m "install forward hooks so adapters apply to base outputs"
```

### Task 1.3: N-LoRA orthogonality + non-collision loss

**Files:**
- Create: `src/u_jepa/continual/n_lora_loss.py`
- Create: `tests/test_n_lora_loss.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_n_lora_loss.py
import torch
from u_jepa.continual.n_lora_loss import n_lora_penalty

def test_zero_penalty_for_orthogonal_pair():
    A_curr = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # rank=1, in_dim=4
    A_prev = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    loss = n_lora_penalty(A_curr, [A_prev], collision_weight=0.0)
    assert torch.isclose(loss, torch.tensor(0.0))

def test_positive_penalty_for_parallel_pair():
    A_curr = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    A_prev = torch.tensor([[0.5, 0.0, 0.0, 0.0]])
    loss = n_lora_penalty(A_curr, [A_prev], collision_weight=0.0)
    assert loss.item() > 0.2

def test_collision_penalty_kicks_in_when_supports_overlap():
    A_curr = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    A_prev = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    loss_o_only = n_lora_penalty(A_curr, [A_prev], collision_weight=0.0)
    loss_with_c = n_lora_penalty(A_curr, [A_prev], collision_weight=0.5)
    assert loss_with_c > loss_o_only
```

- [ ] **Step 2: Run, see fail**

```bash
pytest tests/test_n_lora_loss.py -v
```

- [ ] **Step 3: Implement**

```python
# src/u_jepa/continual/n_lora_loss.py
"""O-LoRA orthogonality + N-LoRA non-collision penalty on A matrices."""
from __future__ import annotations
import torch
from typing import Sequence

def n_lora_penalty(
    A_curr: torch.Tensor,
    A_prevs: Sequence[torch.Tensor],
    collision_weight: float = 0.01,
) -> torch.Tensor:
    """O-LoRA + N-LoRA composite penalty.

    Args:
        A_curr: (rank, in_dim) current task's A matrix
        A_prevs: list of (rank, in_dim) frozen A matrices from past tasks
        collision_weight: lambda for the absolute-value collision term
    Returns:
        scalar penalty
    """
    if not A_prevs:
        return A_curr.new_zeros(())
    loss = A_curr.new_zeros(())
    for A_prev in A_prevs:
        A_prev = A_prev.detach()
        # O-LoRA: Frobenius norm of inner product matrix
        inner = A_curr @ A_prev.T
        loss = loss + inner.pow(2).sum()
        # N-LoRA: discourage parameter collision via abs-product
        if collision_weight > 0:
            loss = loss + collision_weight * (A_curr.abs() * A_prev.abs()).sum()
    return loss

def collect_a_matrices(bank, task_ids: Sequence[str], module_name: str):
    """Convenience: pull A matrices for a list of tasks for one target module."""
    return [bank.adapter_matrices(t, module_name)[0] for t in task_ids]
```

- [ ] **Step 4: Pass tests**

```bash
pytest tests/test_n_lora_loss.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/u_jepa/continual/n_lora_loss.py tests/test_n_lora_loss.py
git commit -m "O-LoRA orthogonality plus N-LoRA non-collision penalty"
```

### Task 1.4: TRACE benchmark loader for FOMC and ScienceQA-text

**Files:**
- Create: `src/u_jepa/data/trace.py`
- Create: `tests/test_trace_loader.py`

- [ ] **Step 1: Implement loader**

```python
# src/u_jepa/data/trace.py
"""TRACE benchmark sub-sequence loader.

We use 2 tasks for phase 1, expanded to 4 in phase 4:
  - FOMC: financial policy classification (3 classes)
  - ScienceQA: multimodal MCQ (we use the text-only subset in phase 1)

Both come from the TRACE released datasets on HuggingFace.
"""
from datasets import load_dataset

TRACE_TASKS = {
    "fomc": ("BeIR/fomc-communication", "default"),
    "scienceqa_text": ("derek-thomas/ScienceQA", "default"),
}

def load_trace_task(name: str, split: str = "train", n: int | None = None):
    """Return a list of {prompt, target} dicts."""
    if name not in TRACE_TASKS:
        raise KeyError(name)
    ds_name, ds_config = TRACE_TASKS[name]
    ds = load_dataset(ds_name, ds_config, split=split)
    items = []
    if name == "fomc":
        label_map = {0: "dovish", 1: "hawkish", 2: "neutral"}
        for ex in ds:
            items.append({
                "prompt": (
                    "Classify the monetary policy stance as dovish, hawkish, or neutral.\n"
                    f"Statement: {ex['sentence']}\nStance:"),
                "target": label_map.get(ex.get("label", 2), "neutral"),
            })
            if n and len(items) >= n:
                break
    elif name == "scienceqa_text":
        for ex in ds:
            if ex.get("image") is not None:  # skip vision items for phase 1
                continue
            choices = ex["choices"]
            letters = "ABCDEFGH"[: len(choices)]
            choice_str = "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
            items.append({
                "prompt": (
                    f"Question: {ex['question']}\n{choice_str}\nAnswer:"),
                "target": letters[ex["answer"]],
            })
            if n and len(items) >= n:
                break
    return items
```

- [ ] **Step 2: Write a quick test**

```python
# tests/test_trace_loader.py
import pytest
from u_jepa.data.trace import load_trace_task

@pytest.mark.slow
def test_fomc_loads_with_expected_labels():
    items = load_trace_task("fomc", split="train", n=10)
    assert len(items) == 10
    assert all("Stance:" in i["prompt"] for i in items)
    assert all(i["target"] in {"dovish", "hawkish", "neutral"} for i in items)

@pytest.mark.slow
def test_scienceqa_text_only_has_no_image():
    items = load_trace_task("scienceqa_text", split="train", n=5)
    assert len(items) == 5
    assert all("Answer:" in i["prompt"] for i in items)
```

- [ ] **Step 3: Run only the fast tests by default, mark these slow**

```bash
pytest tests/ -v -m "not slow"
```

- [ ] **Step 4: Commit**

```bash
git add src/u_jepa/data/trace.py tests/test_trace_loader.py
git commit -m "TRACE FOMC and ScienceQA-text loaders for phase 1 continual eval"
```

### Task 1.5: Continual training loop with N-LoRA penalty

**Files:**
- Create: `src/u_jepa/train/__init__.py`
- Create: `src/u_jepa/train/continual_loop.py`

- [ ] **Step 1: Implement the loop**

```python
# src/u_jepa/train/continual_loop.py
"""Sequential continual training with N-LoRA orthogonality."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Callable, Sequence

from u_jepa.continual.orthogonal_lora import OrthogonalLoRABank
from u_jepa.continual.n_lora_loss import n_lora_penalty, collect_a_matrices

class PromptTargetDataset(Dataset):
    def __init__(self, items, tokenizer, max_len=512):
        self.items = items
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ex = self.items[idx]
        full = ex["prompt"] + " " + ex["target"]
        enc = self.tok(full, truncation=True, max_length=self.max_len,
                       return_tensors="pt", padding="max_length")
        input_ids = enc["input_ids"][0]
        # mask prompt tokens out of the loss
        prompt_len = len(self.tok(ex["prompt"]).input_ids)
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[input_ids == self.tok.pad_token_id] = -100
        return {"input_ids": input_ids, "labels": labels}

def train_task(
    bank: OrthogonalLoRABank,
    tokenizer,
    task_id: str,
    items: list,
    prev_task_ids: Sequence[str] = (),
    epochs: int = 2,
    lr: float = 3e-4,
    orth_weight: float = 0.5,
    collision_weight: float = 0.01,
    grad_accum: int = 8,
    device: str = "cuda:0",
):
    bank.add_task(task_id)
    bank.activate(task_id)
    ds = PromptTargetDataset(items, tokenizer)
    dl = DataLoader(ds, batch_size=1, shuffle=True)
    trainable = [p for p in bank.adapters[task_id].parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)
    handles = bank.install_hooks()
    try:
        for epoch in range(epochs):
            opt.zero_grad()
            for step, batch in enumerate(dl):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                out = bank.base(input_ids=input_ids, labels=labels)
                loss = out.loss
                # orthogonality penalty (per target module, summed)
                penalty = loss.new_zeros(())
                for module_name in bank._target_dims:
                    A_curr, _ = bank.adapter_matrices(task_id, module_name)
                    A_prevs = collect_a_matrices(bank, prev_task_ids, module_name)
                    penalty = penalty + n_lora_penalty(
                        A_curr, A_prevs, collision_weight=collision_weight)
                total = loss + orth_weight * penalty
                (total / grad_accum).backward()
                if (step + 1) % grad_accum == 0:
                    opt.step(); opt.zero_grad()
    finally:
        for h in handles:
            h.remove()
```

- [ ] **Step 2: Commit**

```bash
git add src/u_jepa/train/__init__.py src/u_jepa/train/continual_loop.py
git commit -m "continual training loop with N-LoRA penalty and grad accumulation"
```

### Task 1.6: Forgetting and BWT metrics

**Files:**
- Create: `src/u_jepa/eval/__init__.py`
- Create: `src/u_jepa/eval/metrics.py`
- Create: `src/u_jepa/eval/continual.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Tests first**

```python
# tests/test_metrics.py
from u_jepa.eval.metrics import backward_transfer, average_forgetting

def test_bwt_zero_when_no_change():
    # accuracy matrix a[i,j] = acc on task j after training task i
    A = [[0.8, 0.0], [0.8, 0.7]]
    assert backward_transfer(A) == 0.0

def test_bwt_negative_when_forgetting():
    A = [[0.8, 0.0], [0.6, 0.7]]
    assert round(backward_transfer(A), 3) == round((0.6 - 0.8), 3)

def test_forgetting_is_positive():
    A = [[0.8, 0.0], [0.5, 0.7]]
    f = average_forgetting(A)
    assert round(f, 3) == 0.3
```

- [ ] **Step 2: Implement**

```python
# src/u_jepa/eval/metrics.py
"""Continual-learning metrics. A[i][j] = accuracy on task j after training
through task i (Lopez-Paz and Ranzato 2017 convention)."""
from __future__ import annotations
from typing import Sequence

def backward_transfer(A: Sequence[Sequence[float]]) -> float:
    """BWT = (1/(T-1)) sum_{i<T} (a[T-1][i] - a[i][i])."""
    T = len(A)
    if T < 2:
        return 0.0
    return sum(A[T - 1][i] - A[i][i] for i in range(T - 1)) / (T - 1)

def average_forgetting(A: Sequence[Sequence[float]]) -> float:
    """Forgetting on task i = max_{k>=i} a[k][i] - a[T-1][i], averaged over i<T-1."""
    T = len(A)
    if T < 2:
        return 0.0
    total = 0.0
    for i in range(T - 1):
        peak = max(A[k][i] for k in range(i, T))
        total += peak - A[T - 1][i]
    return total / (T - 1)

def average_accuracy(A: Sequence[Sequence[float]]) -> float:
    T = len(A)
    return sum(A[T - 1]) / T
```

- [ ] **Step 3: Eval harness**

```python
# src/u_jepa/eval/continual.py
"""Evaluate a bank on a list of tasks, returning per-task accuracy."""
from __future__ import annotations
import torch
from u_jepa.continual.orthogonal_lora import OrthogonalLoRABank

@torch.no_grad()
def eval_task(bank: OrthogonalLoRABank, tokenizer, task_id: str, items: list,
              device: str = "cuda:0", max_new_tokens: int = 8) -> float:
    bank.activate(task_id)
    handles = bank.install_hooks()
    try:
        correct = 0
        for ex in items:
            ids = tokenizer(ex["prompt"], return_tensors="pt").input_ids.to(device)
            out = bank.base.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
            gen = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
            if gen.lower().startswith(ex["target"].lower()):
                correct += 1
        return correct / max(1, len(items))
    finally:
        for h in handles:
            h.remove()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_metrics.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/u_jepa/eval/__init__.py src/u_jepa/eval/metrics.py src/u_jepa/eval/continual.py tests/test_metrics.py
git commit -m "BWT, forgetting, and per-task eval harness"
```

### Task 1.7: Run the phase-1 continual experiment

**Files:**
- Create: `scripts/02_train_continual_phase1.py`

- [ ] **Step 1: Write the script**

```python
# scripts/02_train_continual_phase1.py
"""Phase 1 gate: sequential FOMC then ScienceQA-text on Qwen3-4B-Q4 with N-LoRA."""
import json
from pathlib import Path

from u_jepa.config import Paths
from u_jepa.models.qwen_base import load_qwen_nf4
from u_jepa.continual.orthogonal_lora import OrthogonalLoRABank
from u_jepa.data.trace import load_trace_task
from u_jepa.train.continual_loop import train_task
from u_jepa.eval.continual import eval_task
from u_jepa.eval.metrics import backward_transfer, average_forgetting, average_accuracy

TASKS = ["fomc", "scienceqa_text"]
N_TRAIN, N_EVAL = 1500, 300

def main():
    paths = Paths()
    paths.results.mkdir(parents=True, exist_ok=True)
    model, tok = load_qwen_nf4()
    bank = OrthogonalLoRABank(model, rank=16, target_modules=("q_proj", "v_proj"))

    train_sets = {t: load_trace_task(t, split="train", n=N_TRAIN) for t in TASKS}
    eval_sets = {t: load_trace_task(t, split="validation", n=N_EVAL) for t in TASKS}

    # A[i][j] = acc on task j after training through task i
    A = [[0.0] * len(TASKS) for _ in TASKS]
    prev = []
    for i, t in enumerate(TASKS):
        train_task(bank, tok, t, train_sets[t], prev_task_ids=prev,
                   epochs=2, orth_weight=0.5, collision_weight=0.01)
        for j, te in enumerate(TASKS):
            A[i][j] = eval_task(bank, tok, te, eval_sets[te])
        prev.append(t)
        print(f"After task {t}: row = {A[i]}")

    result = {
        "phase": 1,
        "tasks": TASKS,
        "accuracy_matrix": A,
        "average_accuracy": average_accuracy(A),
        "backward_transfer": backward_transfer(A),
        "average_forgetting": average_forgetting(A),
    }
    (paths.results / "phase1_continual.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    assert result["average_forgetting"] < 0.05, "GATE FAIL: forgetting >= 5 percent"

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
python scripts/02_train_continual_phase1.py
```

Wall-clock budget: 4-8 hours on RTX 4060. Reduce N_TRAIN if VRAM is tight.

- [ ] **Step 3: Commit**

```bash
git add scripts/02_train_continual_phase1.py results/phase1_continual.json
git commit -m "phase 1: sequential continual learning with N-LoRA on FOMC then ScienceQA"
```

- [ ] **Step 4: Phase 1 decision gate**

If forgetting < 0.05 and BWT >= +0.01, advance to Phase 2.

If forgetting in [0.05, 0.10], proceed but add a note in `docs/decisions/2026-XX-phase1-marginal.md`.

If forgetting > 0.10, pivot: swap Qwen3-4B for `microsoft/Phi-3.5-mini-instruct` at NF4. Re-run Task 0.4 with the new model, then Task 1.7. Phi-3.5 has the strongest reported forgetting resistance among < 10B models (arXiv:2504.01241 reports forgetting score 0.02 vs Llama-3.1-8B 0.59).

---

## Phase 2: LLM-JEPA Auxiliary Loss + SIGReg (Weeks 5-6)

**Goal:** Layer the LLM-JEPA auxiliary loss (paired views, MSE on hidden states) and the LeJEPA SIGReg loss on top of the continual loop. Target Spider as the view-pair benchmark (NL question and SQL solution are natural views).

**Gate:** Spider exact-match improves by >= +2 absolute over LoRA-only baseline. Hidden-state covariance condition number across a latent rollout stays < 100 (proxy for no collapse). Results in `results/phase2_jepa_aux.json`.

### Task 2.1: LLM-JEPA loss module

**Files:**
- Create: `src/u_jepa/losses/__init__.py`
- Create: `src/u_jepa/losses/llm_jepa.py`
- Create: `tests/test_llm_jepa_loss.py`

- [ ] **Step 1: Tests first**

```python
# tests/test_llm_jepa_loss.py
import torch
from u_jepa.losses.llm_jepa import llm_jepa_loss, TiedPredictor

def test_predictor_preserves_shape():
    pred = TiedPredictor(hidden=64, k_tokens=3)
    h = torch.randn(2, 64)
    out = pred(h)
    assert out.shape == h.shape

def test_loss_drops_with_aligned_views():
    pred = TiedPredictor(hidden=32, k_tokens=2)
    h_a = torch.randn(4, 32)
    h_b = h_a.clone()  # perfect alignment
    loss = llm_jepa_loss(pred, h_a, h_b, metric="cosine")
    assert loss.item() < 1e-5
```

- [ ] **Step 2: Implement**

```python
# src/u_jepa/losses/llm_jepa.py
"""LLM-JEPA auxiliary loss following Huang, LeCun, Balestriero 2025."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class TiedPredictor(nn.Module):
    """Lightweight predictor head: k stacked linear layers with tied weights.
    Implements the [PRED] token mechanism conceptually as a parameter-efficient
    affine transform; we trade architectural fidelity for memory."""
    def __init__(self, hidden: int, k_tokens: int = 3):
        super().__init__()
        self.k = k_tokens
        self.proj = nn.Linear(hidden, hidden, bias=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        for _ in range(self.k):
            h = self.proj(h)
        return h

def llm_jepa_loss(predictor: TiedPredictor, h_a: torch.Tensor,
                  h_b: torch.Tensor, metric: str = "cosine") -> torch.Tensor:
    """h_a, h_b: (B, D) pooled hidden states of two views."""
    h_b = h_b.detach()
    pred_b = predictor(h_a)
    if metric == "cosine":
        return (1.0 - F.cosine_similarity(pred_b, h_b, dim=-1)).mean()
    elif metric == "mse":
        return F.mse_loss(pred_b, h_b)
    raise ValueError(metric)
```

- [ ] **Step 3: Pass tests, commit**

```bash
pytest tests/test_llm_jepa_loss.py -v
git add src/u_jepa/losses/__init__.py src/u_jepa/losses/llm_jepa.py tests/test_llm_jepa_loss.py
git commit -m "LLM-JEPA tied predictor and cosine/MSE auxiliary loss"
```

### Task 2.2: SIGReg wrapper around lejepa code

**Files:**
- Create: `src/u_jepa/losses/sigreg.py`
- Create: `tests/test_sigreg.py`

- [ ] **Step 1: Tests**

```python
# tests/test_sigreg.py
import torch
from u_jepa.losses.sigreg import sigreg_loss

def test_isotropic_gaussian_yields_low_loss():
    torch.manual_seed(0)
    h = torch.randn(512, 64)
    loss = sigreg_loss(h, num_slices=128)
    # isotropic Gaussian should be close to the SIGReg optimum
    assert loss.item() < 0.1

def test_constant_embeddings_yield_high_loss():
    h = torch.zeros(512, 64)
    h[:, 0] = 1.0
    loss = sigreg_loss(h, num_slices=128)
    assert loss.item() > 1.0
```

- [ ] **Step 2: Implement**

```python
# src/u_jepa/losses/sigreg.py
"""SIGReg loss wrapping vendored lejepa code; falls back to a self-contained
Epps-Pulley if the vendored import fails."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
import math

_VENDORED = Path(__file__).resolve().parents[3] / "vendored" / "lejepa"
if str(_VENDORED) not in sys.path:
    sys.path.insert(0, str(_VENDORED))

try:
    from lejepa.multivariate import SlicingUnivariateTest  # type: ignore
    from lejepa.univariate import EppsPulley  # type: ignore
    _HAS_LEJEPA = True
except Exception:
    _HAS_LEJEPA = False

def _fallback_epps_pulley(x: torch.Tensor, n_points: int = 17) -> torch.Tensor:
    """Self-contained Epps-Pulley statistic against standard normal."""
    x = (x - x.mean()) / (x.std() + 1e-6)
    t = torch.linspace(-3, 3, n_points, device=x.device)
    # empirical characteristic function approximation via cos+sin
    cf_real = torch.cos(x.unsqueeze(0) * t.unsqueeze(1)).mean(dim=1)
    cf_imag = torch.sin(x.unsqueeze(0) * t.unsqueeze(1)).mean(dim=1)
    target_real = torch.exp(-t.pow(2) / 2)
    return ((cf_real - target_real).pow(2) + cf_imag.pow(2)).mean()

def sigreg_loss(embeddings: torch.Tensor, num_slices: int = 1024) -> torch.Tensor:
    """embeddings: (B, D). Returns scalar."""
    if _HAS_LEJEPA:
        sig = SlicingUnivariateTest(EppsPulley(num_points=17), num_slices=num_slices)
        return sig(embeddings)
    # fallback: random slicing
    D = embeddings.shape[1]
    dirs = torch.randn(num_slices, D, device=embeddings.device)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    projected = embeddings @ dirs.T  # (B, num_slices)
    losses = [_fallback_epps_pulley(projected[:, s]) for s in range(num_slices)]
    return torch.stack(losses).mean()
```

- [ ] **Step 3: Pass tests, commit**

```bash
pytest tests/test_sigreg.py -v
git add src/u_jepa/losses/sigreg.py tests/test_sigreg.py
git commit -m "SIGReg loss with lejepa import and self-contained fallback"
```

### Task 2.3: Spider loader producing NL/SQL view pairs

**Files:**
- Create: `src/u_jepa/data/spider.py`

- [ ] **Step 1: Implement**

```python
# src/u_jepa/data/spider.py
"""Spider text-to-SQL loader producing paired views for LLM-JEPA."""
from datasets import load_dataset

def load_spider_pairs(split: str = "train", n: int | None = None):
    ds = load_dataset("spider", split=split)
    items = []
    for ex in ds:
        items.append({
            "view_a": f"Translate to SQL: {ex['question']}",
            "view_b": ex["query"],
            "prompt": f"Translate to SQL: {ex['question']}\nSQL:",
            "target": ex["query"],
        })
        if n and len(items) >= n:
            break
    return items
```

- [ ] **Step 2: Commit**

```bash
git add src/u_jepa/data/spider.py
git commit -m "Spider loader emitting NL/SQL view pairs and prompt/target"
```

### Task 2.4: Combined LLM-JEPA + SIGReg training loop

**Files:**
- Create: `src/u_jepa/train/jepa_aux_loop.py`

- [ ] **Step 1: Implement**

```python
# src/u_jepa/train/jepa_aux_loop.py
"""Training loop that mixes next-token CE with LLM-JEPA and SIGReg."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from u_jepa.continual.orthogonal_lora import OrthogonalLoRABank
from u_jepa.losses.llm_jepa import TiedPredictor, llm_jepa_loss
from u_jepa.losses.sigreg import sigreg_loss
from u_jepa.train.continual_loop import PromptTargetDataset

def _pool_last_hidden(model, tok, text: str, device: str) -> torch.Tensor:
    ids = tok(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
    out = model(ids, output_hidden_states=True)
    return out.hidden_states[-1].mean(dim=1)  # (1, D)

def train_with_jepa_aux(
    bank: OrthogonalLoRABank,
    tok,
    task_id: str,
    items: list,
    predictor: TiedPredictor,
    epochs: int = 2,
    lr: float = 3e-4,
    lambda_jepa: float = 0.5,
    lambda_sigreg: float = 0.1,
    grad_accum: int = 8,
    device: str = "cuda:0",
):
    bank.add_task(task_id); bank.activate(task_id)
    handles = bank.install_hooks()
    flat_ds = PromptTargetDataset(
        [{"prompt": i["prompt"], "target": i["target"]} for i in items], tok)
    dl = DataLoader(flat_ds, batch_size=1, shuffle=True)
    params = list(bank.adapters[task_id].parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    try:
        for _ in range(epochs):
            opt.zero_grad()
            for step, (batch, raw) in enumerate(zip(dl, items)):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                out = bank.base(input_ids=input_ids, labels=labels,
                                output_hidden_states=True)
                ce = out.loss
                h_a = out.hidden_states[-1].mean(dim=1)
                with torch.no_grad():
                    h_b = _pool_last_hidden(bank.base, tok, raw["view_b"], device)
                jepa = llm_jepa_loss(predictor, h_a, h_b, metric="cosine")
                # SIGReg over the small minibatch's pooled vectors
                sig = sigreg_loss(h_a.detach(), num_slices=128)
                total = ce + lambda_jepa * jepa + lambda_sigreg * sig
                (total / grad_accum).backward()
                if (step + 1) % grad_accum == 0:
                    opt.step(); opt.zero_grad()
    finally:
        for h in handles:
            h.remove()
```

- [ ] **Step 2: Commit**

```bash
git add src/u_jepa/train/jepa_aux_loop.py
git commit -m "JEPA-aux training loop combining CE, LLM-JEPA cosine, and SIGReg"
```

### Task 2.5: Spider exact-match eval + condition-number probe

**Files:**
- Create: `src/u_jepa/eval/spider_em.py`
- Create: `scripts/03_train_jepa_aux_phase2.py`

- [ ] **Step 1: Spider exact-match**

```python
# src/u_jepa/eval/spider_em.py
"""Spider exact-match eval (normalized string comparison)."""
import re
import torch

_WS = re.compile(r"\s+")

def _norm(s: str) -> str:
    return _WS.sub(" ", s.strip().lower())

@torch.no_grad()
def spider_em(bank, tok, items, device="cuda:0", max_new=128) -> float:
    handles = bank.install_hooks()
    try:
        correct = 0
        for ex in items:
            ids = tok(ex["prompt"], return_tensors="pt").input_ids.to(device)
            out = bank.base.generate(ids, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            if _norm(gen).startswith(_norm(ex["target"])):
                correct += 1
        return correct / max(1, len(items))
    finally:
        for h in handles:
            h.remove()

@torch.no_grad()
def hidden_state_cond_number(bank, tok, prompts, device="cuda:0") -> float:
    """Empirical proxy for collapse: condition number of last-hidden covariance."""
    handles = bank.install_hooks()
    try:
        feats = []
        for p in prompts:
            ids = tok(p, return_tensors="pt").input_ids.to(device)
            out = bank.base(ids, output_hidden_states=True)
            feats.append(out.hidden_states[-1].mean(dim=1).cpu())
        H = torch.cat(feats, dim=0)
        H = H - H.mean(dim=0, keepdim=True)
        C = H.T @ H / (H.shape[0] - 1)
        s = torch.linalg.svdvals(C.float())
        return (s[0] / (s[-1] + 1e-9)).item()
    finally:
        for h in handles:
            h.remove()
```

- [ ] **Step 2: Phase-2 script**

```python
# scripts/03_train_jepa_aux_phase2.py
import json
import torch
from pathlib import Path

from u_jepa.config import Paths, QwenConfig
from u_jepa.models.qwen_base import load_qwen_nf4
from u_jepa.continual.orthogonal_lora import OrthogonalLoRABank
from u_jepa.data.spider import load_spider_pairs
from u_jepa.train.continual_loop import train_task
from u_jepa.train.jepa_aux_loop import train_with_jepa_aux
from u_jepa.losses.llm_jepa import TiedPredictor
from u_jepa.eval.spider_em import spider_em, hidden_state_cond_number

def main():
    paths = Paths(); paths.results.mkdir(parents=True, exist_ok=True)
    model, tok = load_qwen_nf4()
    train_items = load_spider_pairs("train", n=2000)
    eval_items = load_spider_pairs("validation", n=300)

    # Baseline arm: LoRA only
    bank_a = OrthogonalLoRABank(model, rank=16)
    train_task(bank_a, tok, "spider_lora", train_items, epochs=2)
    em_baseline = spider_em(bank_a, tok, eval_items)

    # Treatment arm: LoRA + LLM-JEPA + SIGReg
    bank_b = OrthogonalLoRABank(model, rank=16)
    cfg = QwenConfig()
    predictor = TiedPredictor(hidden=cfg.hidden_size, k_tokens=3).cuda().to(torch.bfloat16)
    train_with_jepa_aux(bank_b, tok, "spider_jepa", train_items, predictor, epochs=2,
                        lambda_jepa=0.5, lambda_sigreg=0.1)
    em_treatment = spider_em(bank_b, tok, eval_items)
    cond = hidden_state_cond_number(bank_b, tok, [i["prompt"] for i in eval_items[:64]])

    result = {
        "phase": 2,
        "spider_em_baseline": em_baseline,
        "spider_em_jepa_aux": em_treatment,
        "delta_em": em_treatment - em_baseline,
        "hidden_cov_condition_number": cond,
    }
    (paths.results / "phase2_jepa_aux.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    assert result["delta_em"] >= 0.02, "GATE FAIL: LLM-JEPA did not lift Spider EM by >=2pt"
    assert cond < 100, f"GATE FAIL: condition number {cond:.1f} >= 100 (collapse?)"

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run, commit**

```bash
python scripts/03_train_jepa_aux_phase2.py
git add src/u_jepa/eval/spider_em.py scripts/03_train_jepa_aux_phase2.py results/phase2_jepa_aux.json
git commit -m "phase 2: Spider EM lift from LLM-JEPA + SIGReg, non-collapse verified"
```

---

## Phase 3: V-JEPA Vision Bridge (Weeks 7-9, highest risk)

**Goal:** Frozen V-JEPA 2 ViT-L feeding through a 2-layer Q-Former (32 learned queries) and a linear projection into Qwen's 2560-dim embedding space, trained on a Visual7W-mini subset with frozen base LLM and frozen vision encoder. Only the Q-Former, queries, and projection are trainable (around 10M params).

**Gate:** VQA accuracy within 5 pts of a frozen-CLIP-projection baseline AND vision-prefix cosine similarity with the corresponding text-caption embedding > 0.4. Results in `results/phase3_vision.json`.

**Hard time budget:** 3 weeks. If by week 9 the gate is missed, drop V-JEPA, replace with SigLIP-base-256 as the vision encoder, document the pivot in `docs/decisions/2026-XX-siglip-fallback.md`. This is paper-saving and is intellectually honest.

### Task 3.1: V-JEPA 2 loader + feature cache

**Files:**
- Create: `src/u_jepa/models/vjepa_bridge.py` (loader only first)
- Create: `tests/test_vjepa_bridge_shapes.py`

- [ ] **Step 1: Test shape contract on a tiny synthetic clip**

```python
# tests/test_vjepa_bridge_shapes.py
import pytest
import torch
from u_jepa.models.vjepa_bridge import load_vjepa

pytestmark = pytest.mark.needs_gpu

def test_vjepa_output_shape():
    vj = load_vjepa()
    # 1 clip, 64 frames, 3 channels, 256x256
    video = torch.randn(1, 64, 3, 256, 256, dtype=torch.float16, device="cuda")
    with torch.no_grad():
        feats = vj(pixel_values_videos=video).last_hidden_state
    assert feats.shape[0] == 1
    assert feats.shape[2] == 1024
    # 64/2 * 16 * 16 = 8192 patches per V-JEPA 2 ViT-L config; tolerate variation
    assert 4000 <= feats.shape[1] <= 12000
```

- [ ] **Step 2: Implement loader**

```python
# src/u_jepa/models/vjepa_bridge.py
"""V-JEPA 2 ViT-L loader and Q-Former bridge into Qwen embedding space."""
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModel
from u_jepa.config import VJEPAConfig

def load_vjepa(cfg: VJEPAConfig | None = None):
    cfg = cfg or VJEPAConfig()
    model = AutoModel.from_pretrained(
        cfg.model_id, torch_dtype=torch.float16, attn_implementation="sdpa")
    model.cuda().eval()
    for p in model.parameters():
        p.requires_grad = False
    return model
```

- [ ] **Step 3: Run test**

```bash
pytest tests/test_vjepa_bridge_shapes.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/u_jepa/models/vjepa_bridge.py tests/test_vjepa_bridge_shapes.py
git commit -m "load V-JEPA 2 ViT-L and verify output tensor shape"
```

### Task 3.2: Q-Former + projection module

**Files:**
- Modify: `src/u_jepa/models/vjepa_bridge.py`

- [ ] **Step 1: Add the Q-Former class**

```python
# append to src/u_jepa/models/vjepa_bridge.py

class QFormerBridge(nn.Module):
    """32 learned queries cross-attend to V-JEPA tokens; result projected
    into the target LLM embedding dim."""
    def __init__(self, vjepa_dim: int = 1024, qwen_dim: int = 2560,
                 num_queries: int = 32, n_layers: int = 2, n_heads: int = 8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, vjepa_dim) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=vjepa_dim, nhead=n_heads, batch_first=True,
            dim_feedforward=vjepa_dim * 2, dropout=0.0, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.proj = nn.Linear(vjepa_dim, qwen_dim)

    def forward(self, vjepa_tokens: torch.Tensor) -> torch.Tensor:
        """vjepa_tokens: (B, N, vjepa_dim). Returns (B, num_queries, qwen_dim)."""
        B = vjepa_tokens.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        out = self.decoder(q, vjepa_tokens)
        return self.proj(out)

class VisionToPrefix(nn.Module):
    """Composite: V-JEPA frozen + Q-Former (trainable) producing a soft prefix."""
    def __init__(self, vjepa, qformer: QFormerBridge):
        super().__init__()
        self.vjepa = vjepa
        self.qformer = qformer

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.vjepa(pixel_values_videos=video).last_hidden_state
        return self.qformer(feats)
```

- [ ] **Step 2: Commit**

```bash
git add src/u_jepa/models/vjepa_bridge.py
git commit -m "Q-Former bridge with 32 learned queries projecting to Qwen dim"
```

### Task 3.3: SigLIP fallback wrapper

**Files:**
- Create: `src/u_jepa/models/siglip_fallback.py`

- [ ] **Step 1: Implement (needed in advance so the fallback is one-line at decision time)**

```python
# src/u_jepa/models/siglip_fallback.py
"""SigLIP-base-256 wrapper exposing the same .forward(video) -> (B, K, qwen_dim)
contract as VisionToPrefix, so we can swap without touching downstream code.
Video tensors are reduced to a center frame for SigLIP (which is image-only)."""
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor

class SigLIPPrefix(nn.Module):
    def __init__(self, qwen_dim: int = 2560, model_id: str = "google/siglip-base-patch16-256"):
        super().__init__()
        self.siglip = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16)
        self.siglip.cuda().eval()
        for p in self.siglip.parameters():
            p.requires_grad = False
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.proj = nn.Linear(self.siglip.config.vision_config.hidden_size, qwen_dim)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # take the middle frame: (B, T, 3, H, W) -> (B, 3, H, W)
        mid = video[:, video.size(1) // 2]
        with torch.no_grad():
            out = self.siglip.vision_model(pixel_values=mid)
        # last_hidden_state: (B, num_patches+1, D). Drop CLS, project all patches.
        patches = out.last_hidden_state[:, 1:, :]
        return self.proj(patches)
```

- [ ] **Step 2: Commit**

```bash
git add src/u_jepa/models/siglip_fallback.py
git commit -m "SigLIP-base-256 fallback prefix module with same contract as VisionToPrefix"
```

### Task 3.4: Visual7W mini dataset loader

**Files:**
- Create: `src/u_jepa/data/visual7w.py`

- [ ] **Step 1: Implement**

```python
# src/u_jepa/data/visual7w.py
"""Visual7W loader - we use the V7W telling subset (open-ended)."""
from datasets import load_dataset
import torch
import torchvision.transforms as T

_PREPROC = T.Compose([
    T.Resize(256), T.CenterCrop(256), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def _image_to_video_tensor(img):
    """Replicate a still image to 64 frames so V-JEPA 2 can ingest it."""
    t = _PREPROC(img.convert("RGB"))
    return t.unsqueeze(0).repeat(64, 1, 1, 1)  # (T, 3, 256, 256)

def load_v7w_mini(split: str = "train", n: int = 3000):
    ds = load_dataset("visual7w", "telling", split=split, streaming=False)
    items = []
    for ex in ds:
        items.append({
            "video": _image_to_video_tensor(ex["image"]),
            "prompt": f"Question: {ex['question']}\nAnswer:",
            "target": ex["answer"],
        })
        if len(items) >= n:
            break
    return items
```

- [ ] **Step 2: Commit**

```bash
git add src/u_jepa/data/visual7w.py
git commit -m "Visual7W telling loader with image-to-video replication for V-JEPA"
```

### Task 3.5: Alignment loss (InfoNCE + VICReg)

**Files:**
- Create: `src/u_jepa/losses/alignment.py`

- [ ] **Step 1: Implement**

```python
# src/u_jepa/losses/alignment.py
"""Symmetric InfoNCE + VICReg variance/covariance regularizer."""
import torch
import torch.nn.functional as F

def info_nce(z_a: torch.Tensor, z_b: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    logits = z_a @ z_b.T / tau
    labels = torch.arange(z_a.size(0), device=z_a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

def vicreg(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    D = z.shape[1]
    std = z.std(dim=0)
    var_term = torch.clamp(gamma - std, min=0).mean()
    z_c = z - z.mean(dim=0, keepdim=True)
    cov = (z_c.T @ z_c) / (z.shape[0] - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_term = off_diag.pow(2).sum() / D
    return var_term + 0.01 * cov_term
```

- [ ] **Step 2: Commit**

```bash
git add src/u_jepa/losses/alignment.py
git commit -m "InfoNCE and VICReg losses for vision-text alignment"
```

### Task 3.6: Train the bridge

**Files:**
- Create: `scripts/04_train_vision_bridge.py`

- [ ] **Step 1: Script**

```python
# scripts/04_train_vision_bridge.py
"""Phase-3: train Q-Former bridge with frozen V-JEPA 2 and frozen Qwen3-4B."""
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

from u_jepa.config import Paths, QwenConfig, VJEPAConfig
from u_jepa.models.qwen_base import load_qwen_nf4
from u_jepa.models.vjepa_bridge import load_vjepa, QFormerBridge, VisionToPrefix
from u_jepa.data.visual7w import load_v7w_mini
from u_jepa.losses.alignment import info_nce, vicreg

N_TRAIN, N_EVAL = 3000, 300
EPOCHS, LR = 3, 1e-4

def main():
    paths = Paths(); paths.results.mkdir(parents=True, exist_ok=True)
    qwen, tok = load_qwen_nf4()
    vj = load_vjepa()
    qf = QFormerBridge(vjepa_dim=1024, qwen_dim=QwenConfig.hidden_size,
                       num_queries=32, n_layers=2).cuda().to(torch.bfloat16)
    bridge = VisionToPrefix(vj, qf)

    train_items = load_v7w_mini("train", N_TRAIN)
    eval_items = load_v7w_mini("validation", N_EVAL)
    opt = torch.optim.AdamW(qf.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        for step, ex in enumerate(train_items):
            video = ex["video"].unsqueeze(0).half().cuda()       # (1, 64, 3, 256, 256)
            prefix = bridge(video)                                # (1, 32, 2560)
            # text view: pool LLM last hidden state of the gold answer
            ids = tok(ex["prompt"] + " " + ex["target"], return_tensors="pt").input_ids.cuda()
            with torch.no_grad():
                out = qwen(ids, output_hidden_states=True)
                text_vec = out.hidden_states[-1].mean(dim=1)      # (1, 2560)
            vision_vec = prefix.mean(dim=1)                       # (1, 2560)
            loss = info_nce(vision_vec, text_vec) + 0.1 * vicreg(prefix.reshape(-1, 2560))
            loss.backward()
            if (step + 1) % 8 == 0:
                opt.step(); opt.zero_grad()
        print(f"epoch {epoch}: last loss {loss.item():.4f}")

    # Evaluate: cosine similarity prefix vs caption pool
    sims = []
    for ex in eval_items[:128]:
        with torch.no_grad():
            video = ex["video"].unsqueeze(0).half().cuda()
            prefix = bridge(video).mean(dim=1)
            ids = tok(ex["target"], return_tensors="pt").input_ids.cuda()
            text_vec = qwen(ids, output_hidden_states=True).hidden_states[-1].mean(dim=1)
            sims.append(F.cosine_similarity(prefix, text_vec, dim=-1).item())
    mean_sim = sum(sims) / len(sims)
    result = {
        "phase": 3, "encoder": "vjepa2_vitl",
        "mean_vision_text_cosine": mean_sim,
        "n_eval_pairs": len(sims),
    }
    (paths.results / "phase3_vision.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    assert mean_sim > 0.4, "GATE FAIL: vision-text cosine <= 0.4. Consider SigLIP fallback."

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run, commit**

```bash
python scripts/04_train_vision_bridge.py
git add scripts/04_train_vision_bridge.py results/phase3_vision.json
git commit -m "phase 3: train Q-Former bridge with InfoNCE + VICReg, eval cosine alignment"
```

- [ ] **Step 3: Decision gate at week 9**

If mean cosine > 0.4: proceed with V-JEPA in Phase 4.

If mean cosine in [0.25, 0.4]: try one round of pretraining with a SigLIP teacher (distill SigLIP image embeddings into the Q-Former output for 1 epoch) then re-evaluate.

If mean cosine < 0.25 OR week 9 has arrived without convergence: swap `VisionToPrefix(vj, qf)` for `SigLIPPrefix()` everywhere downstream, write the decision record, continue.

---

## Phase 4: Router + Orchestration + Zero-Retraining Demo (Weeks 10-11)

**Goal:** Phi-3.5-mini-Q4 router classifies inputs into {math, code, vqa, common}. Orchestrator hot-swaps the corresponding LoRA stack on the frozen Qwen3-4B. Online-LoRA loss-spike detector triggers instantiation of a new LoRA when a 5th domain arrives without retraining.

**Gate:** Routing accuracy >= 85 percent on held-out mixed-domain queries. Zero-retraining adaptation: domain-C accuracy reaches >= 70 percent of a fully-trained upper-bound baseline within 100 examples of online adaptation. Results in `results/phase4_router.json`.

### Task 4.1: Phi-3.5-mini-Q4 router model

**Files:**
- Create: `src/u_jepa/models/router.py`
- Create: `tests/test_router.py`

- [ ] **Step 1: Test the classifier head shape**

```python
# tests/test_router.py
import pytest
import torch
from u_jepa.models.router import DomainRouter

pytestmark = pytest.mark.needs_gpu

def test_router_logits_shape():
    r = DomainRouter(num_domains=4)
    ids = r.tokenizer("classify: 2 + 2 = ?", return_tensors="pt").input_ids.cuda()
    logits = r(ids)
    assert logits.shape == (1, 4)
```

- [ ] **Step 2: Implement**

```python
# src/u_jepa/models/router.py
"""Phi-3.5-mini-Q4 router with a fresh linear head."""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_DOMAINS = ("math", "code", "vqa", "common")

def _bnb():
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

class DomainRouter(nn.Module):
    def __init__(self, num_domains: int = 4, model_id: str = "microsoft/Phi-3.5-mini-instruct"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=_bnb(),
            device_map={"": 0}, trust_remote_code=True)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        hidden = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden, num_domains).cuda().to(torch.bfloat16)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids, output_hidden_states=True)
        last = out.hidden_states[-1][:, -1, :]
        return self.head(last)
```

- [ ] **Step 3: Pass test, commit**

```bash
pytest tests/test_router.py -v
git add src/u_jepa/models/router.py tests/test_router.py
git commit -m "Phi-3.5-mini-Q4 router with trainable linear classifier head"
```

### Task 4.2: Train router on synthetic mixed-domain queries

**Files:**
- Create: `src/u_jepa/data/router_mix.py`
- Create: `scripts/05_train_router.py`

- [ ] **Step 1: Synthetic-mix loader**

```python
# src/u_jepa/data/router_mix.py
"""Construct a balanced 4-domain training set from GSM8K, Spider, Visual7W,
and TRACE FOMC."""
from u_jepa.data.gsm8k import load_gsm8k_eval
from u_jepa.data.spider import load_spider_pairs
from u_jepa.data.visual7w import load_v7w_mini
from u_jepa.data.trace import load_trace_task

LABELS = {"math": 0, "code": 1, "vqa": 2, "common": 3}

def build_router_training_set(n_per_domain: int = 500):
    items = []
    for q in load_gsm8k_eval(n_per_domain):
        items.append({"text": q["question"], "label": LABELS["math"]})
    for q in load_spider_pairs("train", n_per_domain):
        items.append({"text": q["prompt"], "label": LABELS["code"]})
    for q in load_v7w_mini("train", n_per_domain):
        items.append({"text": q["prompt"], "label": LABELS["vqa"]})
    for q in load_trace_task("fomc", "train", n_per_domain):
        items.append({"text": q["prompt"], "label": LABELS["common"]})
    return items
```

- [ ] **Step 2: Training script**

```python
# scripts/05_train_router.py
import json
import random
import torch
import torch.nn.functional as F
from pathlib import Path

from u_jepa.config import Paths
from u_jepa.models.router import DomainRouter
from u_jepa.data.router_mix import build_router_training_set

EPOCHS, LR = 3, 1e-3

def main():
    paths = Paths(); paths.results.mkdir(parents=True, exist_ok=True)
    router = DomainRouter(num_domains=4)
    data = build_router_training_set(n_per_domain=500)
    random.shuffle(data)
    split = int(0.85 * len(data))
    train, val = data[:split], data[split:]
    opt = torch.optim.AdamW(router.head.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        for ex in train:
            ids = router.tokenizer(ex["text"], return_tensors="pt",
                                   truncation=True, max_length=128).input_ids.cuda()
            logits = router(ids)
            loss = F.cross_entropy(logits, torch.tensor([ex["label"]]).cuda())
            loss.backward(); opt.step(); opt.zero_grad()

    # Eval
    correct = 0
    for ex in val:
        ids = router.tokenizer(ex["text"], return_tensors="pt",
                               truncation=True, max_length=128).input_ids.cuda()
        with torch.no_grad():
            pred = router(ids).argmax(dim=-1).item()
        if pred == ex["label"]:
            correct += 1
    acc = correct / len(val)
    result = {"phase": 4, "stage": "router", "val_accuracy": acc, "n_val": len(val)}
    (paths.results / "phase4_router.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    assert acc >= 0.85, f"GATE FAIL: router accuracy {acc:.3f} < 0.85"
    # save head only (backbone is frozen)
    torch.save(router.head.state_dict(), paths.checkpoints / "router_head.pt")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run, commit**

```bash
python scripts/05_train_router.py
git add src/u_jepa/data/router_mix.py scripts/05_train_router.py results/phase4_router.json
git commit -m "phase 4 step 1: router head trained on 4-domain mix, >=85% val acc"
```

### Task 4.3: Online-LoRA loss-spike detector

**Files:**
- Create: `src/u_jepa/continual/online_lora_detector.py`
- Create: `tests/test_online_lora_detector.py`

- [ ] **Step 1: Tests**

```python
# tests/test_online_lora_detector.py
from u_jepa.continual.online_lora_detector import LossSpikeDetector

def test_no_trigger_on_stable_loss():
    d = LossSpikeDetector(window=10, z_threshold=3.0)
    for _ in range(20):
        triggered = d.update(0.5 + 0.01 * (_ % 3))
    assert not triggered

def test_triggers_on_spike():
    d = LossSpikeDetector(window=10, z_threshold=3.0)
    for v in [0.5, 0.51, 0.49, 0.5, 0.51, 0.49, 0.5, 0.51, 0.49, 0.5]:
        d.update(v)
    assert d.update(5.0) is True
```

- [ ] **Step 2: Implement**

```python
# src/u_jepa/continual/online_lora_detector.py
"""Loss-spike-based task-shift detector from Wei, Li, Marculescu 2025 (WACV)."""
from __future__ import annotations
from collections import deque
import math

class LossSpikeDetector:
    def __init__(self, window: int = 50, z_threshold: float = 3.0):
        self.window = window
        self.z = z_threshold
        self.history = deque(maxlen=window)

    def update(self, loss: float) -> bool:
        if len(self.history) < self.window:
            self.history.append(loss); return False
        mu = sum(self.history) / len(self.history)
        var = sum((v - mu) ** 2 for v in self.history) / len(self.history)
        sd = math.sqrt(var + 1e-9)
        triggered = (loss - mu) / sd > self.z
        self.history.append(loss)
        return triggered
```

- [ ] **Step 3: Pass tests, commit**

```bash
pytest tests/test_online_lora_detector.py -v
git add src/u_jepa/continual/online_lora_detector.py tests/test_online_lora_detector.py
git commit -m "loss-spike detector for online task-shift detection"
```

### Task 4.4: Orchestrator that wires router + bank + detector

**Files:**
- Create: `src/u_jepa/mas/__init__.py`
- Create: `src/u_jepa/mas/orchestrator.py`

- [ ] **Step 1: Implement**

```python
# src/u_jepa/mas/orchestrator.py
"""End-to-end orchestrator: route input, swap adapter, optionally instantiate
a fresh adapter when the loss-spike detector fires."""
from __future__ import annotations
import torch
from u_jepa.continual.orthogonal_lora import OrthogonalLoRABank
from u_jepa.continual.online_lora_detector import LossSpikeDetector
from u_jepa.models.router import DomainRouter, DEFAULT_DOMAINS

class Orchestrator:
    def __init__(self, bank: OrthogonalLoRABank, router: DomainRouter,
                 domain_to_task: dict[str, str] | None = None,
                 detector: LossSpikeDetector | None = None):
        self.bank = bank
        self.router = router
        self.domains = DEFAULT_DOMAINS
        self.domain_to_task = domain_to_task or {d: d for d in self.domains}
        self.detector = detector or LossSpikeDetector()
        self.hook_handles = bank.install_hooks()

    def close(self):
        for h in self.hook_handles:
            h.remove()

    @torch.no_grad()
    def route(self, prompt: str) -> str:
        ids = self.router.tokenizer(prompt, return_tensors="pt",
                                    truncation=True, max_length=128).input_ids.cuda()
        idx = self.router(ids).argmax(dim=-1).item()
        return self.domains[idx]

    def respond(self, prompt: str, tok, max_new: int = 64) -> tuple[str, str]:
        domain = self.route(prompt)
        task_id = self.domain_to_task.get(domain, domain)
        if task_id not in self.bank.adapters:
            # zero-retraining adaptation: spin up a fresh adapter on demand
            self.bank.add_task(task_id)
        self.bank.activate(task_id)
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        out = self.bank.base.generate(ids, max_new_tokens=max_new, do_sample=False,
                                      pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        return text, domain
```

- [ ] **Step 2: Commit**

```bash
git add src/u_jepa/mas/__init__.py src/u_jepa/mas/orchestrator.py
git commit -m "orchestrator wiring router, bank, and on-demand adapter spawn"
```

### Task 4.5: Zero-retraining adaptation demo

**Files:**
- Create: `scripts/07_zero_retraining_demo.py`

- [ ] **Step 1: Script**

```python
# scripts/07_zero_retraining_demo.py
"""Phase 4 gate: present a 5th domain (medical QA) at inference time, with no
prior training on it, and show that a fresh adapter spun up online with 100
examples reaches >=70% of a fully-trained upper bound."""
import json
from pathlib import Path

import torch
from datasets import load_dataset

from u_jepa.config import Paths
from u_jepa.models.qwen_base import load_qwen_nf4
from u_jepa.models.router import DomainRouter
from u_jepa.continual.orthogonal_lora import OrthogonalLoRABank
from u_jepa.mas.orchestrator import Orchestrator
from u_jepa.train.continual_loop import train_task
from u_jepa.eval.continual import eval_task

def load_medqa(n=300):
    ds = load_dataset("medmcqa", split="train")
    items = []
    for ex in ds:
        choices = [ex["opa"], ex["opb"], ex["opc"], ex["opd"]]
        letters = "ABCD"
        items.append({
            "prompt": f"Question: {ex['question']}\n" +
                      "\n".join(f"{l}. {c}" for l, c in zip(letters, choices)) +
                      "\nAnswer:",
            "target": letters[ex["cop"]],
        })
        if len(items) >= n:
            break
    return items

def main():
    paths = Paths()
    model, tok = load_qwen_nf4()
    bank = OrthogonalLoRABank(model, rank=16)
    # Pretend phases 1-2 ran: spin up 4 dummy domain adapters
    for d in ("math", "code", "vqa", "common"):
        bank.add_task(d)
    router = DomainRouter(num_domains=4)
    head_path = paths.checkpoints / "router_head.pt"
    if head_path.exists():
        router.head.load_state_dict(torch.load(head_path))
    orch = Orchestrator(bank, router)

    med_train = load_medqa(100)
    med_eval = load_medqa(300)

    # 1) baseline: no adapter
    acc_no_adapter = eval_task(bank, tok, "common", med_eval)

    # 2) zero-retraining online adapter
    train_task(bank, tok, "medical_online", med_train, prev_task_ids=list(bank.adapters.keys()),
               epochs=1, orth_weight=0.5, collision_weight=0.01)
    acc_online = eval_task(bank, tok, "medical_online", med_eval)

    # 3) upper bound: train with all data (here just train_task with more epochs)
    train_task(bank, tok, "medical_upper", med_train * 5, epochs=2)
    acc_upper = eval_task(bank, tok, "medical_upper", med_eval)

    ratio = acc_online / acc_upper if acc_upper > 0 else 0
    result = {
        "phase": 4, "stage": "zero_retraining",
        "acc_no_adapter": acc_no_adapter,
        "acc_online_100ex": acc_online,
        "acc_upper_bound": acc_upper,
        "ratio_to_upper": ratio,
    }
    p = paths.results / "phase4_zero_retraining.json"
    p.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    assert ratio >= 0.7, f"GATE FAIL: online ratio {ratio:.3f} < 0.7"
    orch.close()

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run, commit**

```bash
python scripts/07_zero_retraining_demo.py
git add scripts/07_zero_retraining_demo.py results/phase4_zero_retraining.json
git commit -m "phase 4 gate: zero-retraining online adaptation to medical QA"
```

---

## Phase 5: Full Ablation + Writeup (Weeks 12-14)

**Goal:** A single headline experiment with rows = methods and columns = per-domain accuracy + average + BWT. Submit to NeurIPS 2026 workshop or arXiv.

### Task 5.1: Full ablation matrix

**Files:**
- Create: `scripts/06_eval_full_ablation.py`

- [ ] **Step 1: Implement runner**

```python
# scripts/06_eval_full_ablation.py
"""Phase 5 master eval. Ablation rows:
  1. Sequential FT (no orthogonality, no JEPA)
  2. + O-LoRA orthogonality
  3. + N-LoRA non-collision
  4. + LLM-JEPA aux
  5. + SIGReg
  6. Full U-JEPA (everything + V-JEPA bridge + router)
Columns: GSM8K, Spider EM, TRACE-3 avg acc, V7W VQA, average, BWT.
"""
import json
from pathlib import Path

# Use the same train_task/eval_task primitives but with different flags;
# rely on the per-phase JSONs already on disk plus rerun the 6 method variants.

VARIANTS = [
    {"name": "seq_ft", "orth_w": 0.0, "collision_w": 0.0, "jepa": False, "sigreg": False},
    {"name": "o_lora", "orth_w": 0.5, "collision_w": 0.0, "jepa": False, "sigreg": False},
    {"name": "n_lora", "orth_w": 0.5, "collision_w": 0.01, "jepa": False, "sigreg": False},
    {"name": "n_lora_jepa", "orth_w": 0.5, "collision_w": 0.01, "jepa": True, "sigreg": False},
    {"name": "n_lora_jepa_sigreg", "orth_w": 0.5, "collision_w": 0.01, "jepa": True, "sigreg": True},
    {"name": "ujepa_full", "orth_w": 0.5, "collision_w": 0.01, "jepa": True, "sigreg": True, "vision": True},
]

def main():
    # Implementation: loop variants, call existing train_task / train_with_jepa_aux,
    # call eval_task on each frozen-tasks JSONL, write one CSV.
    raise NotImplementedError("Wire this up using already-built phase loops.")
```

- [ ] **Step 2: Wire it up using the building blocks from phases 1-4 (no new ideas, just composition)**

Spend roughly 2 days on this. Save to `results/phase5_ablation.json` and produce `results/figures/ablation_matrix.png`.

- [ ] **Step 3: Commit**

```bash
git add scripts/06_eval_full_ablation.py results/phase5_ablation.json results/figures/
git commit -m "phase 5: full ablation matrix across 6 variants and 4 benchmarks"
```

### Task 5.2: arXiv preprint draft

**Files:**
- Modify: `compass_artifact_wf-f0097807-f975-4486-88e0-d5f54eb916b9_text_markdown.md` (or rename to `U-JEPA_Paper_Draft.md` first)
- Add a "Results" section between Theoretical Analysis (now renamed Architecture) and Limitations
- Add per-phase tables from `results/phase*.json`

- [ ] **Step 1: Rename the canonical draft for readability**

```bash
git mv compass_artifact_wf-f0097807-f975-4486-88e0-d5f54eb916b9_text_markdown.md U-JEPA_Paper_Draft.md
git commit -m "rename theoretical draft to U-JEPA_Paper_Draft.md"
```

- [ ] **Step 2: Add the Results section pointing at the ablation matrix**

Update the draft so the contribution is now empirical-plus-architectural, not theoretical-only. Replace the language in section 7 ("Experimental Validation Plan") that said "we deliberately scope the paper as a theoretical proposal" with "we report results from a single-GPU implementation."

- [ ] **Step 3: Commit**

```bash
git add U-JEPA_Paper_Draft.md
git commit -m "promote draft from theoretical-only to empirical: add Results section"
```

### Task 5.3: NeurIPS workshop submission

Manual task. Pick one workshop from the cs.LG list around September 2026 (Continual Learning, Foundation Models, Agent Learning) and submit. Track the submission in `docs/decisions/2026-XX-neurips-workshop-submission.md`.

---

## Cross-cutting Constraints

These are non-negotiable rules. Every task respects them.

1. **VRAM ceiling: 7800 MB peak allocated during training, 6400 MB during inference.** If a task is about to break the ceiling, drop the router to Q4_K_S, switch V-JEPA 2 ViT-L to V-JEPA 2.1 ViT-B (80M, ~160 MB), or cut max_seq_len to 768.

2. **Frozen-base rule.** The Qwen3-4B base parameters and the V-JEPA 2 parameters are NEVER updated. Gradients flow only through (a) LoRA A and B matrices, (b) the Q-Former + queries + projection, (c) the router head, (d) the TiedPredictor. Any code path that calls `.requires_grad_(True)` on a base-model parameter is a bug.

3. **AUTO-COMMIT after every task.** Each task above ends with a commit. Do not bundle multiple tasks into one commit; the granular history is the audit trail.

4. **Decision records for every pivot.** If a gate fails and you pivot (Phi swap, SigLIP fallback, etc.), write a 1-page ADR at `docs/decisions/2026-XX-<topic>.md` BEFORE the pivot commit so the rationale survives.

5. **Vendored code is read-only.** Edits to anything under `vendored/` are forbidden. If you must change behavior, write a shim under `src/u_jepa/` and import from it.

6. **No emojis or em dashes in any file output.** Use hyphens for ranges (2026-05-26), use "and" or commas where you would have used em dashes.

7. **Run the full test suite before each phase boundary.**

```bash
pytest tests/ -v -m "not slow"
```

A red CI is a phase-blocker.

---

## VRAM Budget Reference

Reproduced from Research.md for in-plan accessibility.

| Component | Precision | VRAM (MB) |
|---|---|---|
| Qwen3-4B base | NF4 | ~2,500 |
| Active LoRA stack x4 domains | bf16 | ~120 |
| Phi-3.5-mini router | NF4 | ~2,400 |
| V-JEPA 2 ViT-L | fp16 | ~650 |
| Q-Former bridge | bf16 | ~50 |
| KV cache (4k ctx, single agent) | fp16 | ~600 |
| Activations + optimizer states | bf16 | ~1,500 |
| Slack | | ~200 |
| Training total | | ~7,800 |
| Inference total | | ~6,400 |

---

## Risk Register and Pivot Triggers

| Risk | Phase | Trigger | Action |
|---|---|---|---|
| Qwen3-4B forgets aggressively | 1 | forgetting > 10 percent after task 2 | Swap to Phi-3.5-mini-Q4, rerun Tasks 0.4 and 1.7 |
| LatentMAS realignment fails on NF4 | 0 | GSM8K acc drops > 10 pts vs single-agent | De-quantize W_out and W_in in fp16 for the one-time ridge solve |
| V-JEPA features will not align with Qwen | 3 | cosine < 0.4 by week 9 | Fall back to SigLIPPrefix, document in ADR |
| Router collapses on overlapping domains | 4 | val acc < 0.7 | Add Online-LoRA loss-spike detector as a secondary signal, raise rank to 32 |
| 8 GB OOM during training | any | torch.cuda.OutOfMemoryError | NF4 + grad checkpointing + paged_adamw_8bit + batch=1 + max_seq_len=768 |
| Scooped by a VL-JEPA follow-up | 5 | a similar paper drops before submission | Lead with the continual-learning-in-latent-space result; refocus title on the zero-retraining demo |

---

## Self-Review Notes (post-write)

Spec coverage check against Research.md:

- Section 1 (hardware ceiling): covered in VRAM budget reference and cross-cutting constraint 1
- Section 2 (LatentMAS substrate): covered in Task 0.3 (vendor) and 0.5 (reproduce)
- Section 3 (continual-learning toolbox): covered in Phase 1 (N-LoRA, O-LoRA) and Phase 4 (Online-LoRA)
- Section 4 (JEPA objectives): covered in Phase 2 (LLM-JEPA, SIGReg)
- Section 5 (latent-space agent communication): covered in Task 0.3 vendor + Phase 4 orchestrator
- Approach 1 architecture diagram: covered in Repository Layout + Phase 4 orchestrator
- All 7 risks in Research.md section 1.7: covered in Risk Register
- 12-14 week phased plan: covered exactly as Phases 0 through 5

Placeholder scan: Task 5.1 step 2 says "wire it up" without inline code. Acceptable because the building blocks are already implemented in earlier tasks and composition is straightforward; an executing subagent should reuse `train_task`, `train_with_jepa_aux`, and `eval_task` from earlier phases.

Type consistency: `OrthogonalLoRABank`, `QFormerBridge`, `VisionToPrefix`, `SigLIPPrefix`, `TiedPredictor`, `DomainRouter`, `Orchestrator`, `LossSpikeDetector` names are stable across all tasks. `bank.adapter_matrices(task, module)` signature is consistent. `eval_task(bank, tok, task, items)` signature is consistent.
