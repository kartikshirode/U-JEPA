"""Per-task eval harness for sequential continual learning."""
from __future__ import annotations
import torch

from u_jepa.continual.orthogonal_lora import OrthogonalLoRABank


@torch.no_grad()
def eval_task(
    bank: OrthogonalLoRABank,
    tokenizer,
    task_id: str,
    items: list[dict],
    device: str = "cuda:0",
    max_new_tokens: int = 8,
) -> float:
    """Greedy decode and prefix-match the target string. Returns accuracy in [0,1].

    Match is case-insensitive prefix because targets are short labels
    (dovish/hawkish/neutral, A/B/C/D, etc.) and the model may emit extra
    tokens we do not want to penalize.
    """
    bank.activate(task_id)
    handles = bank.install_hooks()
    try:
        correct = 0
        for ex in items:
            ids = tokenizer(ex["prompt"], return_tensors="pt").input_ids.to(device)
            out = bank.base.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            gen = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
            if gen.lower().startswith(ex["target"].lower()):
                correct += 1
        return correct / max(1, len(items))
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def build_accuracy_matrix(
    bank: OrthogonalLoRABank,
    tokenizer,
    eval_sets: dict[str, list[dict]],
    seen_tasks: list[str],
    device: str = "cuda:0",
) -> list[float]:
    """Evaluate the bank on every task in eval_sets and return one row of
    the accuracy matrix (one entry per task in the same order as eval_sets)."""
    row = []
    for tname in eval_sets:
        if tname in seen_tasks:
            row.append(eval_task(bank, tokenizer, tname, eval_sets[tname], device=device))
        else:
            # Task not yet seen: 0 by convention
            row.append(0.0)
    return row
