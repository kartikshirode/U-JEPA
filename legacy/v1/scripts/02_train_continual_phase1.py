"""Phase 1: sequential continual learning with N-LoRA on Qwen3-14B-Instruct.

Train FOMC then ScienceQA-text on a frozen NF4 Qwen3-14B base. Each task
gets its own adapter in the OrthogonalLoRABank with N-LoRA penalty pushing
its A matrices orthogonal to and non-colliding with the prior task's.

Gate: average_forgetting < 0.05 AND backward_transfer >= -0.02
Pivot trigger: if forgetting > 0.10, swap base to Phi-3.5-mini-Q4 and rerun.

Runs on Kaggle T4 x2 with NF4 quantization. Local Windows skips with a
clear message because bitsandbytes wheels are Linux-only practically.
"""
from __future__ import annotations
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Path bootstrap so the script runs without `pip install -e .`
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from u_jepa.config import HardwareConfig, LoraConfig, QwenConfig
from u_jepa.util.env import detect, prepare

TASKS = ["fomc", "scienceqa_text"]
N_TRAIN_PER_TASK = 1500
N_EVAL_PER_TASK = 300


def main() -> int:
    env = prepare()
    print(f"env: {env.name}")

    if env.name == "local_win":
        print("[skip] local Windows: bitsandbytes NF4 path not viable.")
        print("Use kaggle/phase1/ notebook on Kaggle GPU T4 x2 instead.")
        return 0

    # Heavy imports only after env check so local Win can run --help cleanly
    import torch
    from u_jepa.continual.orthogonal_lora import OrthogonalLoRABank
    from u_jepa.data.trace import load_trace_task
    from u_jepa.eval.continual import build_accuracy_matrix
    from u_jepa.eval.metrics import (
        average_accuracy, average_forgetting, backward_transfer,
    )
    from u_jepa.models.qwen_base import load_qwen_nf4, qwen_vram_usage_mb
    from u_jepa.train.continual_loop import train_task

    qwen_cfg = QwenConfig()
    lora_cfg = LoraConfig()
    hw_cfg = HardwareConfig()

    print(f"loading {qwen_cfg.model_id} at NF4")
    t0 = time.time()
    model, tok = load_qwen_nf4(qwen_cfg)
    print(f"loaded in {time.time() - t0:.1f}s, VRAM ~{qwen_vram_usage_mb(model)} MiB")

    bank = OrthogonalLoRABank(
        model, rank=lora_cfg.rank, target_modules=lora_cfg.target_modules,
        alpha=lora_cfg.alpha,
    )
    print(f"target modules: {len(bank._target_dims)} matched")

    # Datasets
    train_sets, eval_sets = {}, {}
    for t in TASKS:
        print(f"loading {t} train...")
        train_sets[t] = load_trace_task(t, split="train", n=N_TRAIN_PER_TASK)
        # validation split fallback to test if not present
        try:
            eval_sets[t] = load_trace_task(t, split="validation", n=N_EVAL_PER_TASK)
        except Exception:
            eval_sets[t] = load_trace_task(t, split="test", n=N_EVAL_PER_TASK)
        print(f"  {t}: train={len(train_sets[t])}, eval={len(eval_sets[t])}")

    # Sequential training + per-step eval matrix
    A_matrix: list[list[float]] = []
    seen: list[str] = []
    train_stats: list[dict] = []

    for t in TASKS:
        print(f"\n=== training task: {t} (prev: {seen}) ===")
        stats = train_task(
            bank, tok, t, train_sets[t],
            prev_task_ids=seen,
            epochs=2, lr=3e-4,
            orth_weight=0.5, collision_weight=0.01,
            grad_accum=hw_cfg.grad_accum,
            max_len=hw_cfg.max_seq_len,
            device=hw_cfg.device,
        )
        train_stats.append(stats)
        seen.append(t)
        print(f"  stats: {stats}")

        print(f"  evaluating bank against all {len(TASKS)} tasks...")
        row = build_accuracy_matrix(
            bank, tok, eval_sets, seen,
            device=hw_cfg.device, task_order=TASKS,
        )
        A_matrix.append(row)
        print(f"  row: {row}")

    result = {
        "phase": 1,
        "stage": "continual_n_lora",
        "config": {
            "qwen": asdict(qwen_cfg),
            "lora": asdict(lora_cfg),
            "n_train_per_task": N_TRAIN_PER_TASK,
            "n_eval_per_task": N_EVAL_PER_TASK,
        },
        "tasks": TASKS,
        "accuracy_matrix": A_matrix,
        "average_accuracy": average_accuracy(A_matrix),
        "backward_transfer": backward_transfer(A_matrix),
        "average_forgetting": average_forgetting(A_matrix),
        "train_stats": train_stats,
    }
    out_path = env.results_dir / "phase1_continual.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    forgetting = result["average_forgetting"]
    if forgetting > 0.05:
        print(f"GATE FAIL: avg forgetting {forgetting:.3f} > 0.05")
        if forgetting > 0.10:
            print("PIVOT TRIGGER: forgetting > 0.10 - swap base to Phi-3.5-mini-Q4")
        return 1
    print(f"GATE PASS: avg forgetting {forgetting:.3f} <= 0.05")
    return 0


if __name__ == "__main__":
    sys.exit(main())
