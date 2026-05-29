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


def _resolve_input_device(model, fallback: str):
    """Device the model wants token inputs on (embedding layer device).

    Falls back to the caller-supplied string if the model exposes neither
    an input embedding nor any parameters (e.g. a stub in tests).
    """
    try:
        emb = model.get_input_embeddings()
        if emb is not None and getattr(emb, "weight", None) is not None:
            return emb.weight.device
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return fallback


class PromptTargetDataset(Dataset):
    """Tokenize {prompt, target} pairs and mask the prompt out of the loss."""

    def __init__(self, items: list[dict], tokenizer, max_len: int = 512,
                 pad_to_max: bool = False):
        # Drop rows whose target tokenizes to nothing. An all-masked label
        # row makes CrossEntropy divide by zero valid tokens and returns NaN,
        # which then poisons every gradient through the accumulation window.
        kept = []
        for ex in items:
            tgt = ex.get("target", "")
            if not str(tgt).strip():
                continue
            n_tgt = len(tokenizer(str(tgt), add_special_tokens=False)["input_ids"])
            if n_tgt < 1:
                continue
            kept.append(ex)
        self.items = kept
        self.tok = tokenizer
        self.max_len = max_len
        # With batch_size=1 a single sequence never needs padding to line up
        # with batch siblings, so default to no padding to save compute and
        # memory. Set pad_to_max=True only if batching >1 without a collator.
        self.pad_to_max = pad_to_max

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.items[idx]
        prompt = ex["prompt"]
        target = ex["target"]

        # Tokenize the prompt alone to learn its exact token length, then
        # tokenize the target as a continuation. Concatenating the two
        # avoids the BPE merge across the prompt or target boundary that
        # made the separately-counted prompt_len fall on the wrong side
        # of a token, leaking the last prompt token into the loss or
        # masking out the first target token entirely.
        prompt_enc = self.tok(prompt, add_special_tokens=True)
        target_enc = self.tok(target, add_special_tokens=False)
        prompt_ids = list(prompt_enc["input_ids"])
        target_ids = list(target_enc["input_ids"])

        # Reserve at least one target token after truncation. If the prompt
        # alone already fills max_len, drop the head of the prompt so the
        # target survives. This prevents all-masked labels from producing
        # NaN gradients (CE on -100-only targets is undefined in PyTorch).
        keep_target = min(len(target_ids), max(1, self.max_len - 1))
        target_ids = target_ids[:keep_target]
        keep_prompt = max(0, self.max_len - len(target_ids))
        if len(prompt_ids) > keep_prompt:
            prompt_ids = prompt_ids[-keep_prompt:] if keep_prompt > 0 else []

        ids = prompt_ids + target_ids
        attn = [1] * len(ids)
        pad_id = self.tok.pad_token_id
        if pad_id is None:
            pad_id = getattr(self.tok, "eos_token_id", 0) or 0
        pad_count = self.max_len - len(ids)
        if self.pad_to_max and pad_count > 0:
            ids = ids + [pad_id] * pad_count
            attn = attn + [0] * pad_count

        input_ids = torch.tensor(ids, dtype=torch.long)
        attn_mask = torch.tensor(attn, dtype=torch.long)
        labels = input_ids.clone()
        prompt_len = len(prompt_ids)
        labels[:prompt_len] = -100
        labels[attn_mask == 0] = -100

        # Belt-and-suspenders: at least one real target token must remain or
        # the CE loss is NaN. The __init__ filter and keep_target>=1 logic
        # should guarantee this, so a failure here is a real bug, not data.
        if int((labels != -100).sum()) < 1:
            raise ValueError(
                f"row {idx} has no unmasked label tokens "
                f"(prompt_len={prompt_len}, target={self.items[idx].get('target')!r})"
            )

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

    # Resolve the device the base model actually expects inputs on. With a
    # multi-GPU device_map the embedding layer may not be on `device`, so
    # honour the model's real input device instead of a hardcoded string.
    in_device = _resolve_input_device(bank.base, device)

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
            pending = 0  # micro-batches accumulated since the last opt.step()
            for step, batch in enumerate(dl):
                input_ids = batch["input_ids"].to(in_device)
                labels = batch["labels"].to(in_device)
                attn = batch["attention_mask"].to(in_device)

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
                pending += 1

                last_ce = float(ce.detach())
                last_orth = float(orth.detach()) if isinstance(orth, torch.Tensor) else 0.0
                step_count += 1

                if (step + 1) % grad_accum == 0:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    pending = 0
            # Only flush if there is a real partial accumulation left over;
            # an unconditional step() either runs on zeroed grads (waste) or
            # under-weights the partial batch by a factor of grad_accum.
            if pending > 0:
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
