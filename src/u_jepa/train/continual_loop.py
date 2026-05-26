"""Sequential continual training loop with N-LoRA penalty.

Each task gets a fresh adapter added to the bank. The adapter is
trained for `epochs` passes over the task data while the per-step
N-LoRA penalty pushes its A matrices orthogonal to and non-colliding
with all previously trained adapters.

Memory shape:
  base model: frozen, NF4 (no gradients)
  trainable: only the active adapter's A and B (a few MB per task)
  gradients: bf16, accumulated over micro-batches
  KV cache during training: standard CE on labels, no need for past_kv
"""
from __future__ import annotations
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from u_jepa.continual.orthogonal_lora import OrthogonalLoRABank
from u_jepa.continual.n_lora_loss import n_lora_penalty_over_bank


class PromptTargetDataset(Dataset):
    """Tokenize {prompt, target} pairs and mask the prompt out of the loss."""

    def __init__(self, items: list[dict], tokenizer, max_len: int = 512):
        self.items = items
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.items[idx]
        full = ex["prompt"] + " " + ex["target"]
        enc = self.tok(
            full,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            padding="max_length",
        )
        input_ids = enc["input_ids"][0]
        attn_mask = enc["attention_mask"][0]

        # Mask prompt tokens out of the loss so we only train the target
        prompt_ids = self.tok(ex["prompt"], truncation=True, max_length=self.max_len).input_ids
        prompt_len = len(prompt_ids)
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attn_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
        }


def train_task(
    bank: OrthogonalLoRABank,
    tokenizer,
    task_id: str,
    items: list[dict],
    prev_task_ids: Sequence[str] = (),
    epochs: int = 2,
    lr: float = 3e-4,
    orth_weight: float = 0.5,
    collision_weight: float = 0.01,
    grad_accum: int = 8,
    max_len: int = 512,
    device: str = "cuda:0",
) -> dict:
    """Train one new task on top of `prev_task_ids` adapters in the bank.

    Returns a dict with training stats (final loss components).
    """
    bank.add_task(task_id)
    bank.activate(task_id)

    ds = PromptTargetDataset(items, tokenizer, max_len=max_len)
    dl = DataLoader(ds, batch_size=1, shuffle=True)

    trainable = [p for p in bank.adapters[task_id].parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)

    handles = bank.install_hooks()
    last_ce, last_orth = 0.0, 0.0
    step_count = 0
    try:
        for _epoch in range(epochs):
            opt.zero_grad(set_to_none=True)
            for step, batch in enumerate(dl):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attn = batch["attention_mask"].to(device)

                out = bank.base(input_ids=input_ids, attention_mask=attn, labels=labels)
                ce = out.loss

                orth = n_lora_penalty_over_bank(
                    bank,
                    current_task=task_id,
                    prev_tasks=list(prev_task_ids),
                    collision_weight=collision_weight,
                )

                total = ce + orth_weight * orth
                (total / grad_accum).backward()

                last_ce = float(ce.detach())
                last_orth = float(orth.detach()) if isinstance(orth, torch.Tensor) else 0.0
                step_count += 1

                if (step + 1) % grad_accum == 0:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            # flush remainder
            opt.step()
            opt.zero_grad(set_to_none=True)
    finally:
        for h in handles:
            h.remove()

    return {
        "task_id": task_id,
        "n_examples": len(items),
        "epochs": epochs,
        "steps": step_count,
        "final_ce": last_ce,
        "final_orth_penalty": last_orth,
    }
