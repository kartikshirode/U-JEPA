"""Phase 2: LLM-JEPA auxiliary loss + SIGReg on Spider, two-arm comparison.

Loads Qwen3-14B once at NF4, then runs two independent LoRA arms on top of
the SAME frozen base:
  baseline arm  = LoRA + standard CE only (a fresh bank)
  treatment arm = LoRA + CE + LLM-JEPA cosine + SIGReg (a fresh bank)

Both arms are evaluated on the same Spider validation slice with the same
string-level exact-match proxy. Gates:
  (1) treatment EM - baseline EM >= +0.02 absolute
  (2) treatment hidden-state covariance condition number < 100

Runs on Kaggle (T4, NF4). Local Windows is a no-op (bitsandbytes).
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from u_jepa.config import HardwareConfig, LoraConfig, QwenConfig
from u_jepa.util.env import prepare

N_TRAIN = 800      # per arm; chosen so 2 arms + Spider eval fit in 9h on T4
N_EVAL = 200
MAX_NEW = 96       # most Spider SQL targets are <60 tokens; leaves headroom
EPOCHS = 2
LR = 3e-4
LAMBDA_JEPA = 0.5
LAMBDA_SIGREG = 0.1


def main() -> int:
    env = prepare()
    print(f"env: {env.name}")

    if env.name == "local_win":
        print("[skip] local Windows: bitsandbytes NF4 path not viable.")
        print("Use kaggle/phase2/ notebook on Kaggle GPU T4 instead.")
        return 0

    import gc

    import torch
    from u_jepa.continual.orthogonal_lora import OrthogonalLoRABank
    from u_jepa.data.spider import load_spider_pairs
    from u_jepa.eval.spider_em import hidden_state_cond_number, spider_em
    from u_jepa.losses.llm_jepa import TiedPredictor
    from u_jepa.losses.sigreg import using_lejepa
    from u_jepa.models.qwen_base import (
        load_qwen_nf4, model_input_device, qwen_vram_usage_mb,
    )
    from u_jepa.train.continual_loop import train_task
    from u_jepa.train.jepa_aux_loop import train_with_jepa_aux

    qwen_cfg = QwenConfig()
    lora_cfg = LoraConfig()
    hw_cfg = HardwareConfig()

    print(f"loading {qwen_cfg.model_id} at NF4")
    t0 = time.time()
    model, tok = load_qwen_nf4(qwen_cfg)
    print(f"loaded in {time.time() - t0:.1f}s, VRAM ~{qwen_vram_usage_mb(model)} MiB")
    print(f"sigreg backend: {'lejepa' if using_lejepa() else 'fallback'}")

    print("loading spider pairs...")
    train_items = load_spider_pairs("train", n=N_TRAIN)
    eval_items = load_spider_pairs("validation", n=N_EVAL)
    print(f"  spider: train={len(train_items)}, eval={len(eval_items)}")

    # --- Arm A: LoRA only -------------------------------------------------
    print("\n=== arm A: LoRA-only baseline ===")
    bank_a = OrthogonalLoRABank(
        model, rank=lora_cfg.rank, target_modules=lora_cfg.target_modules,
        alpha=lora_cfg.alpha,
    )
    stats_a = train_task(
        bank_a, tok, "spider_lora", train_items, prev_task_ids=(),
        epochs=EPOCHS, lr=LR, orth_weight=0.0, collision_weight=0.0,
        grad_accum=hw_cfg.grad_accum, max_len=hw_cfg.max_seq_len,
        device=hw_cfg.device,
    )
    em_a = spider_em(bank_a, tok, eval_items, task_id="spider_lora",
                     device=hw_cfg.device, max_new_tokens=MAX_NEW,
                     max_len=hw_cfg.max_seq_len)
    print(f"baseline EM: {em_a:.4f}")

    # Free arm-A adapter and optimizer state before allocating arm B; otherwise
    # bank_a's LoRA params plus arm-A AdamW moments stay resident in VRAM
    # alongside the new bank, which can push a 15 GB T4 into OOM.
    del bank_a
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Arm B: LoRA + LLM-JEPA + SIGReg ---------------------------------
    print("\n=== arm B: LoRA + LLM-JEPA + SIGReg ===")
    bank_b = OrthogonalLoRABank(
        model, rank=lora_cfg.rank, target_modules=lora_cfg.target_modules,
        alpha=lora_cfg.alpha,
    )
    # Predictor in bf16 to keep its parameters and AdamW moments small. The
    # JEPA loss is an auxiliary signal and tolerates the precision tradeoff;
    # the trained adapter is what ultimately matters for downstream EM.
    predictor = TiedPredictor(
        hidden=qwen_cfg.hidden_size, k_tokens=3,
    ).to(model_input_device(model)).to(torch.bfloat16)
    stats_b = train_with_jepa_aux(
        bank_b, tok, "spider_jepa", train_items, predictor,
        epochs=EPOCHS, lr=LR,
        lambda_jepa=LAMBDA_JEPA, lambda_sigreg=LAMBDA_SIGREG,
        grad_accum=hw_cfg.grad_accum, max_len=hw_cfg.max_seq_len,
        device=hw_cfg.device,
    )
    em_b = spider_em(bank_b, tok, eval_items, task_id="spider_jepa",
                     device=hw_cfg.device, max_new_tokens=MAX_NEW,
                     max_len=hw_cfg.max_seq_len)
    print(f"jepa+sigreg EM: {em_b:.4f}")

    print("computing hidden-state condition number on treatment arm...")
    cond = hidden_state_cond_number(
        bank_b, tok, [x["prompt"] for x in eval_items[:64]],
        task_id="spider_jepa", device=hw_cfg.device,
        max_len=hw_cfg.max_seq_len,
    )
    print(f"cond number: {cond:.2f}")

    delta = em_b - em_a
    result = {
        "phase": 2,
        "stage": "jepa_aux_spider",
        "config": {
            "qwen": asdict(qwen_cfg),
            "lora": asdict(lora_cfg),
            "n_train": N_TRAIN, "n_eval": N_EVAL,
            "max_new_tokens": MAX_NEW,
            "epochs": EPOCHS, "lr": LR,
            "lambda_jepa": LAMBDA_JEPA, "lambda_sigreg": LAMBDA_SIGREG,
        },
        "sigreg_backend": "lejepa" if using_lejepa() else "fallback",
        "spider_em_baseline": em_a,
        "spider_em_jepa_aux": em_b,
        "delta_em": delta,
        "hidden_cov_condition_number": cond,
        "train_stats_baseline": stats_a,
        "train_stats_jepa_aux": stats_b,
    }
    out_path = env.results_dir / "phase2_jepa_aux.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    fail = []
    if delta < 0.02:
        fail.append(f"delta EM {delta:.4f} < 0.02")
    if cond != cond or cond >= 100:
        fail.append(f"cond number {cond:.2f} >= 100 (collapse?)")
    if fail:
        print("GATE FAIL: " + "; ".join(fail))
        return 1
    print(f"GATE PASS: delta_em={delta:.4f} cond={cond:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
