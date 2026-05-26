"""Phase 0 gate: reproduce LatentMAS GSM8K accuracy on Qwen3-14B-AWQ.

Calls vendored LatentMASMethod directly with our own argparse.Namespace,
bypassing run.py's argparse choices list which is hardcoded to
{Qwen3-4B, Qwen3-14B} only. Using AWQ-quantized weights so 14B fits
into 2x T4 16GB with plenty of room for KV cache and activations.

Memory budget per T4 (16 GiB total reported as 14.56):
  Qwen3-14B-AWQ weights (4-bit): ~3.5 GiB per GPU at TP=2
  Activations + KV cache budget:  ~10 GiB per GPU
  Slack:                          ~1 GiB

Gate: GSM8K accuracy >= 65% on 250 problems.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Path bootstrap for u_jepa package + vendored LatentMAS
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
_VENDORED = _REPO_ROOT / "vendored" / "LatentMAS"
for p in (_SRC, _VENDORED):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from u_jepa.util.env import detect, prepare

# Patch vllm.LLM to default max_model_len so vendored ModelWrapper does not
# request the full 40960-token native context (way larger than our T4 KV cache
# budget at ~10k tokens). Done at import-time so it lands before vendored
# LatentMAS imports vllm. Safe: setdefault only fills in when caller did not.
def _patch_vllm_max_model_len(default_max_model_len: int = 8192) -> None:
    try:
        import vllm  # type: ignore
    except ImportError:
        return  # local Windows path
    _orig = vllm.LLM.__init__

    def _patched(self, *args, **kwargs):
        kwargs.setdefault("max_model_len", default_max_model_len)
        kwargs.setdefault("max_num_seqs", 16)
        return _orig(self, *args, **kwargs)

    vllm.LLM.__init__ = _patched
    print(f"[patch] vllm.LLM default max_model_len={default_max_model_len}, max_num_seqs=16")

_patch_vllm_max_model_len(default_max_model_len=8192)


def _patch_transformers_activations_for_autoawq() -> None:
    """autoawq imports PytorchGELUTanh / NewGELUActivation / GELUActivation
    from transformers.activations; transformers >= 4.53 dropped those class
    names in favor of an ACT2FN dict. Add thin shims so autoawq's import
    chain succeeds on Kaggle's transformers (currently 4.57)."""
    try:
        import transformers.activations as _act
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        return  # local Windows path

    if not hasattr(_act, "PytorchGELUTanh"):
        class PytorchGELUTanh(nn.Module):
            def forward(self, x):
                return F.gelu(x, approximate="tanh")
        _act.PytorchGELUTanh = PytorchGELUTanh

    if not hasattr(_act, "NewGELUActivation"):
        class NewGELUActivation(nn.Module):
            def forward(self, x):
                return F.gelu(x, approximate="tanh")
        _act.NewGELUActivation = NewGELUActivation

    if not hasattr(_act, "GELUActivation"):
        class GELUActivation(nn.Module):
            def forward(self, x):
                return F.gelu(x)
        _act.GELUActivation = GELUActivation

    print("[patch] transformers.activations shimmed for autoawq compatibility")

_patch_transformers_activations_for_autoawq()

PHASE_CFG = dict(
    method="latent_mas",
    model_name="Qwen/Qwen3-14B-AWQ",
    task="gsm8k",
    split="test",
    prompt="sequential",
    max_samples=250,
    generate_bs=1,
    latent_steps=4,
    max_new_tokens=512,
    temperature=0.6,
    top_p=0.95,
    use_vllm=True,
    enable_prefix_caching=True,
    use_second_HF_model=True,
    latent_space_realign=True,
    # tensor_parallel_size=1 so vLLM lives entirely on cuda:0 and the second
    # HF model (used for latent-path hidden states) gets cuda:1 to itself.
    # AWQ Qwen3-14B is ~4.7 GB so single-GPU vLLM has plenty of room.
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
    device="cuda:0",
    device2="cuda:1",
    text_mas_context_length=-1,
    think=False,
    seed=42,
)


def build_namespace() -> argparse.Namespace:
    return argparse.Namespace(**PHASE_CFG)


def run_inference(args: argparse.Namespace) -> Tuple[float, int, List[Dict], float]:
    """Load model, build method, iterate dataset, return (acc, correct, preds, elapsed)."""
    # Imports happen here so the script can `--help` cleanly without CUDA
    from models import ModelWrapper
    from methods.latent_mas import LatentMASMethod
    from data import load_gsm8k
    from utils import auto_device, set_seed

    set_seed(args.seed)
    device = auto_device(args.device)
    print(f"loading {args.model_name} via vLLM with TP={args.tensor_parallel_size}")
    model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)

    method = LatentMASMethod(
        model,
        latent_steps=args.latent_steps,
        judger_max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        generate_bs=args.generate_bs,
        args=args,
    )

    dataset_iter = list(load_gsm8k(split=args.split))
    if args.max_samples > 0:
        dataset_iter = dataset_iter[: args.max_samples]
    total = len(dataset_iter)
    print(f"running {total} GSM8K problems")

    preds: List[Dict] = []
    processed = 0
    batch: List[Dict] = []
    t0 = time.time()

    for item in dataset_iter:
        batch.append(item)
        if len(batch) == args.generate_bs or processed + len(batch) == total:
            results = (
                method.run_batch_vllm(batch)
                if args.method == "latent_mas" and args.use_vllm
                else method.run_batch(batch)
            )
            for res in results:
                preds.append(res)
                processed += 1
                print(
                    f"[{processed}/{total}] "
                    f"pred={res.get('prediction')} "
                    f"gold={res.get('gold')} "
                    f"ok={res.get('correct')}"
                )
            batch = []

    elapsed = time.time() - t0
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / max(1, len(preds))
    return acc, correct, preds, elapsed


def main() -> int:
    env = prepare()
    print(f"env: {env.name}, repo_root: {env.repo_root}")

    if not env.can_run_vllm:
        print(f"[skip] env={env.name} cannot run vLLM. Use kaggle/phase0/.")
        return 0

    args = build_namespace()
    print(f"config: {PHASE_CFG}")

    try:
        acc, correct, preds, elapsed = run_inference(args)
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    result = {
        "phase": 0,
        "stage": "latentmas_baseline",
        "config": PHASE_CFG,
        "model": args.model_name,
        "split": args.split,
        "n_eval": len(preds),
        "accuracy": acc,
        "correct": correct,
        "elapsed_sec": round(elapsed, 2),
        "sec_per_sample": round(elapsed / max(1, len(preds)), 2),
    }
    out_json = env.results_dir / "phase0_baseline.json"
    try:
        out_json.write_text(json.dumps(result, indent=2))
    except OSError as e:
        print(f"[warn] could not write {out_json}: {e}")
    print(json.dumps(result, indent=2))

    if acc < 0.65:
        print(f"GATE FAIL: accuracy {acc:.3f} < 0.65")
        return 2
    print(f"GATE PASS: accuracy {acc:.3f} >= 0.65")
    return 0


if __name__ == "__main__":
    sys.exit(main())
