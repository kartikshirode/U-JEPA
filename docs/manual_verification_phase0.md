# Phase 0 manual verification on Kaggle

This is the end-to-end smoke test for the Phase 0 LatentMAS reproduction
on GSM8K. Run this on a Kaggle T4 x2 session after the repo has been
cloned to `/kaggle/working/U-JEPA` and `pip install -r requirements-kaggle.txt`
has finished. The unit tests cover the patch helpers in isolation;
this checklist confirms the whole pipeline runs together.

## What you should see at each stage

Add these cells to a Kaggle notebook in the order below. Each cell has a
short list of patterns the output must contain. If a pattern is missing,
stop and read the failure section under that cell.

### Cell 1: GPU and env sanity

```python
import torch, sys, os
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i} = {p.name}, {p.total_memory // (1024**3)} GiB")
```

Expected output patterns:
- `cuda available: True`
- `device count: 2`
- both lines say `Tesla T4` and `14` or `15` GiB

If `device count` is 1, the notebook is on a single-GPU runtime and the
Phase 0 config's `device2="cuda:1"` will crash later. Switch the
accelerator to `GPU T4 x2` in the right sidebar.

### Cell 2: import the patch helpers and verify they applied

```python
import sys
sys.path.insert(0, "/kaggle/working/U-JEPA/scripts")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "phase0", "/kaggle/working/U-JEPA/scripts/01_repro_latentmas_gsm8k.py"
)
phase0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase0)

import transformers.activations as act
print("PytorchGELUTanh shim present:", hasattr(act, "PytorchGELUTanh"))
print("NewGELUActivation shim present:", hasattr(act, "NewGELUActivation"))
print("GELUActivation shim present:", hasattr(act, "GELUActivation"))

import vllm
print("vllm.LLM.__init__ patched:", "patched" in vllm.LLM.__init__.__qualname__ or "_patched" in vllm.LLM.__init__.__qualname__)
```

Expected output patterns:
- `[patch] vllm.LLM defaults: max_model_len=8192, max_num_seqs=2, ...`
- `[patch] transformers.activations shimmed for autoawq compatibility`
- `[patch] _build_latent_realign_matrix compute moved to CPU ...`
- all three GELU shim lines say `True`
- `vllm.LLM.__init__ patched: True`

If the activations shim line is missing, autoawq's import chain will
break later during model load. Check whether transformers was downgraded
by a stray dependency; the patch is a noop when the classes already exist.

### Cell 3: dry-run config check

```python
ns = phase0.build_namespace()
print("model:", ns.model_name)
print("task:", ns.task, "split:", ns.split, "max_samples:", ns.max_samples)
print("tensor_parallel_size:", ns.tensor_parallel_size, "device:", ns.device, "device2:", ns.device2)
print("generate_bs:", ns.generate_bs)
assert ns.generate_bs >= 2, "vendored run_batch_vllm needs bs>=2"
```

Expected:
- `model: Qwen/Qwen3-14B-AWQ`
- `tensor_parallel_size: 1` (vLLM lives on cuda:0 only)
- `device2: cuda:1` (HF latent-path model lives on the other GPU)
- `generate_bs: 2`

### Cell 4: load the AWQ model and confirm VRAM split

```python
from u_jepa.util.env import prepare
env = prepare()
print("env:", env.name)  # expect "kaggle"

# Heavy: this triggers the vLLM init + the second HF model load
from models import ModelWrapper
mw = ModelWrapper(ns.model_name, ns.device, use_vllm=True, args=ns)

# After load, check that cuda:0 holds vLLM weights and cuda:1 holds the HF model
import torch
for i in range(2):
    free = torch.cuda.mem_get_info(i)[0] / 1024**3
    total = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f"cuda:{i} free={free:.2f} GiB, total={total:.2f} GiB, used={total-free:.2f} GiB")
```

Expected:
- `env: kaggle`
- model load logs include `Loading vLLM model on cuda:0`
- both GPUs report used VRAM between 4 and 12 GiB; neither sits near zero (means the split failed) nor near 14.5 (means it OOMed)

### Cell 5: tiny sanity run (5 problems, not 250)

```python
import argparse
small_ns = argparse.Namespace(**{**phase0.PHASE_CFG, "max_samples": 5})
acc, correct, preds, elapsed = phase0.run_inference(small_ns)
print(f"5-problem smoke: acc={acc:.2f}, correct={correct}/{len(preds)}, time={elapsed:.1f}s")
```

Expected:
- prints `[1/5]` through `[5/5]` lines with `pred=...`, `gold=...`, `ok=True|False`
- final accuracy lands in `0.2` to `1.0` range; with N=5 the variance is huge so do not gate on this number, just confirm the loop ran
- time per sample is somewhere in `5s` to `60s` range
- no `RuntimeError` or `CUDA out of memory`

If a `prompt_embeds != input_tokens` assertion fires, the prefix-cache
override in the vllm patch did not stick. Re-run cell 2 from a fresh
Python kernel.

### Cell 6: full gate run

```python
acc, correct, preds, elapsed = phase0.run_inference(ns)
print(f"FINAL: acc={acc:.3f}, correct={correct}/{len(preds)}, time={elapsed:.1f}s")
print("GATE", "PASS" if acc >= 0.65 else "FAIL", f"({acc:.3f} vs 0.65)")
```

Expected:
- 250 lines of per-problem output
- final accuracy `>= 0.65` (Phase 0 gate)
- time per sample roughly the same as the smoke test

If accuracy lands between `0.55` and `0.65`, the model is working but
LatentMAS is underperforming. Try bumping `latent_steps` from 4 to 6 in
PHASE_CFG and rerunning. If it lands below `0.55`, the latent path is
probably broken; check that the second HF model loaded on cuda:1.

## Common failures and fixes

- **OOM on cuda:1 during `ModelWrapper.__init__`**: the realign matrix
  build did not get moved to CPU. Re-run cell 2 to reapply the patch and
  confirm the `[patch] _build_latent_realign_matrix compute moved to CPU`
  line appears.
- **`AttributeError: module 'transformers.activations' has no attribute 'PytorchGELUTanh'`**:
  autoawq imported before the shim was applied. Restart the kernel and
  run cell 2 first.
- **`Assertion failed: len(inputs_embeds) == len(input_tokens)`**: prefix
  caching slipped through. Confirm cell 2 printed
  `enable_prefix_caching=False`. If the vendored ModelWrapper bypassed
  the patch by importing vllm before cell 2, restart the kernel.
- **`ValueError: tensor_parallel_size 2 not supported`**: PHASE_CFG was
  edited to TP=2. Phase 0 ships with TP=1; revert that change.
