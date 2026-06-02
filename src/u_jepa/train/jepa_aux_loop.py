"""Combined training loop: CE on next-token + LLM-JEPA + SIGReg.

This is Phase 2's main training entry point. Per item, the model sees the NL
prompt (view A), produces (a) the standard CE loss against the SQL target and
(b) a pooled hidden state h_a. A second no-grad forward on the SQL (view B)
produces the JEPA target h_b. The TiedPredictor maps h_a toward h_b under a
cosine loss. SIGReg is computed over the token-level hidden states of view A
to keep them from collapsing to a low-rank subspace; per-token vectors give
SIGReg enough samples to be meaningful even with batch_size=1.

The loop reuses the OrthogonalLoRABank from Phase 1 so the same hooks and
device-routing logic apply.
"""
from __future__ import annotations

import time
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from u_jepa.continual.orthogonal_lora import OrthogonalLoRABank
from u_jepa.losses.llm_jepa import TiedPredictor, llm_jepa_loss
from u_jepa.losses.sigreg import sigreg_loss
from u_jepa.train.continual_loop import PromptTargetDataset, _resolve_input_device
from u_jepa.util.prompting import format_chat_prompt


class SpiderJEPADataset(Dataset):
    """Yields a PromptTargetDataset row PLUS the raw view_b string per item.

    The view_b string is tokenized lazily inside the train loop so we do not
    waste memory on pre-tokenized SQL strings we may never reach if training
    stops early.
    """

    def __init__(self, items: list[dict], tokenizer, max_len: int = 512):
        # Reuse PromptTargetDataset so view-A tokenization (chat template,
        # prompt masking, NaN guard) lives in one place. Then derive view_b
        # strings from PromptTargetDataset.items directly so the filter
        # logic is not duplicated (and cannot drift out of sync).
        self.inner = PromptTargetDataset(items, tokenizer, max_len=max_len)
        self.view_b = [
            ex.get("view_b", ex.get("target", "")) for ex in self.inner.items
        ]
        assert len(self.view_b) == len(self.inner), (
            f"view_b/inner length mismatch: {len(self.view_b)} vs {len(self.inner)}"
        )

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> dict:
        # Copy the row so downstream mutations (like adding view_b_text) do
        # not leak back into a cached tensor inside PromptTargetDataset.
        row = dict(self.inner[idx])
        row["view_b_text"] = self.view_b[idx]
        row["view_b_idx"] = idx
        return row


def _spider_collate(batch: list[dict]) -> dict:
    """Collate that keeps view_b_text as a plain string list."""
    if len(batch) != 1:
        raise NotImplementedError(
            f"_spider_collate expects batch_size=1, got {len(batch)}"
        )
    out = {
        k: v for k, v in batch[0].items()
        if k not in ("view_b_text", "view_b_idx")
    }
    out = {k: v.unsqueeze(0) if torch.is_tensor(v) and v.dim() == 1 else v
           for k, v in out.items()}
    out["view_b_text"] = batch[0]["view_b_text"]
    out["view_b_idx"] = batch[0]["view_b_idx"]
    return out


@torch.no_grad()
def _pool_view_b(model, tokenizer, text: str, in_device, max_len: int) -> torch.Tensor:
    """No-grad forward on view B; return mean-pooled last hidden state (1, D).

    View B is the SQL answer; we feed it as raw text (no chat template) so the
    target embedding reflects the SQL surface form, not the assistant-turn
    wrapping. The caller is responsible for deactivating any active adapter
    before calling so the cached target is base-only.
    """
    enc = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_len,
        add_special_tokens=True,
    )
    ids = enc["input_ids"].to(in_device)
    attn = enc["attention_mask"].to(in_device)
    out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
    last = out.hidden_states[-1].detach()  # (1, T, D)
    mask = attn.unsqueeze(-1).to(last.dtype)
    pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    # Release the full hidden-state tuple right after pooling so the 41-layer
    # stack does not sit in memory across the next train step.
    del out
    return pooled  # (1, D)


@torch.no_grad()
def _precompute_view_b_cache(
    bank: OrthogonalLoRABank,
    tokenizer,
    view_b_texts: list[str],
    in_device,
    max_len: int,
    log_every: int = 100,
) -> list[torch.Tensor]:
    """Pool every view-B string ONCE with all adapters deactivated.

    Returns a list of (1, D) CPU tensors in the same order as view_b_texts.
    This halves training time by removing the per-step view-B forward, and
    pins the target representation to the frozen base so adapter updates do
    not chase a moving target.
    """
    prev_active = bank._active
    bank._active = None
    cache: list[torch.Tensor] = []
    t0 = time.monotonic()
    try:
        for i, text in enumerate(view_b_texts):
            pooled = _pool_view_b(bank.base, tokenizer, text, in_device, max_len)
            cache.append(pooled.detach().to("cpu"))
            if log_every > 0 and (i + 1) % log_every == 0:
                elapsed = time.monotonic() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                eta = (len(view_b_texts) - (i + 1)) / max(rate, 1e-6)
                print(
                    f"[jepa:precompute] {i+1}/{len(view_b_texts)} "
                    f"rate={rate:.2f}/s eta={eta:.0f}s",
                    flush=True,
                )
    finally:
        bank._active = prev_active
    return cache


def train_with_jepa_aux(
    bank: OrthogonalLoRABank,
    tokenizer,
    task_id: str,
    items: list[dict],
    predictor: TiedPredictor,
    epochs: int = 2,
    lr: float = 3e-4,
    lambda_jepa: float = 0.5,
    lambda_sigreg: float = 0.1,
    grad_accum: int = 8,
    max_len: int = 512,
    device: str = "cuda:0",
    sigreg_slices: int = 64,
    log_every: int = 25,
) -> dict:
    """Train one task with CE + LLM-JEPA + SIGReg.

    Adapter for `task_id` is added to the bank on entry and left activated on
    exit so the caller can evaluate immediately without touching bank state.
    """
    bank.add_task(task_id)
    bank.activate(task_id)
    in_device = _resolve_input_device(bank.base, device)

    ds = SpiderJEPADataset(items, tokenizer, max_len=max_len)
    dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=_spider_collate)

    trainable = (
        [p for p in bank.adapters[task_id].parameters() if p.requires_grad]
        + [p for p in predictor.parameters() if p.requires_grad]
    )
    opt = torch.optim.AdamW(trainable, lr=lr)

    # Precompute view-B pooled embeddings ONCE with the adapter deactivated.
    # Halves per-step compute and pins the JEPA target to the frozen base so
    # the adapter is not chasing a moving target across epochs.
    print(
        f"[jepa:{task_id}] precomputing view-B cache for {len(ds.view_b)} rows...",
        flush=True,
    )
    view_b_cache = _precompute_view_b_cache(
        bank, tokenizer, ds.view_b, in_device, max_len,
    )

    handles = bank.install_hooks()
    total_steps = epochs * len(dl)
    t0 = time.monotonic()
    last_ce = last_jepa = last_sig = 0.0
    step_count = 0
    print(
        f"[jepa:{task_id}] start: rows={len(ds)} epochs={epochs} steps={total_steps} "
        f"lr={lr} lambda_jepa={lambda_jepa} lambda_sigreg={lambda_sigreg} "
        f"grad_accum={grad_accum}",
        flush=True,
    )
    try:
        for epoch in range(epochs):
            opt.zero_grad(set_to_none=True)
            pending = 0
            for step, batch in enumerate(dl):
                input_ids = batch["input_ids"].to(in_device)
                labels = batch["labels"].to(in_device)
                attn = batch["attention_mask"].to(in_device)

                out = bank.base(
                    input_ids=input_ids, attention_mask=attn, labels=labels,
                    output_hidden_states=True,
                )
                ce = out.loss
                last_hidden = out.hidden_states[-1]  # (1, T, D)
                # Drop the full 41-layer hidden-state tuple right after we
                # capture the layer we need; otherwise it sits in memory for
                # the duration of the SIGReg / JEPA forward.
                del out.hidden_states
                del out
                mask_a = attn.unsqueeze(-1).to(last_hidden.dtype)
                h_a_pooled = (last_hidden * mask_a).sum(dim=1) / mask_a.sum(dim=1).clamp(min=1)

                # Fetch the cached view-B embedding for this row; move to the
                # active device just before the loss so the predictor matmul
                # does not see a CPU vs CUDA mix.
                h_b = view_b_cache[batch["view_b_idx"]].to(in_device)
                pred_dtype = next(predictor.parameters()).dtype
                jepa = llm_jepa_loss(
                    predictor,
                    h_a_pooled.to(pred_dtype),
                    h_b.to(pred_dtype),
                    metric="cosine",
                )

                # SIGReg over per-token last-hidden vectors of view A. With
                # batch_size=1 the pooled vector has no companions; per-token
                # vectors give SIGReg ~T ~= 512 samples to slice against.
                flat = last_hidden.reshape(-1, last_hidden.shape[-1])
                # Mask out padding tokens so SIGReg does not "see" them.
                m = attn.reshape(-1).bool()
                tokens = flat[m]
                if tokens.shape[0] >= 16:
                    # Keep tokens in their native dtype so we do not inflate
                    # activation memory by upcasting the whole slice to fp32.
                    sig = sigreg_loss(tokens, num_slices=sigreg_slices)
                else:
                    sig = torch.zeros((), device=in_device, dtype=last_hidden.dtype)

                total = ce + lambda_jepa * jepa + lambda_sigreg * sig
                (total / grad_accum).backward()
                pending += 1

                last_ce = float(ce.detach())
                last_jepa = float(jepa.detach())
                last_sig = float(sig.detach())
                step_count += 1

                if log_every > 0 and step_count % log_every == 0:
                    elapsed = time.monotonic() - t0
                    sps = step_count / max(elapsed, 1e-6)
                    eta = (total_steps - step_count) / max(sps, 1e-6)
                    print(
                        f"[jepa:{task_id}] ep{epoch+1}/{epochs} "
                        f"step {step_count}/{total_steps} "
                        f"ce={last_ce:.4f} jepa={last_jepa:.4f} sig={last_sig:.4f} "
                        f"sps={sps:.2f} elapsed={elapsed:.0f}s eta={eta:.0f}s",
                        flush=True,
                    )

                if (step + 1) % grad_accum == 0:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    pending = 0
            if pending > 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
    finally:
        for h in handles:
            h.remove()

    print(
        f"[jepa:{task_id}] done in {time.monotonic()-t0:.0f}s steps={step_count} "
        f"final_ce={last_ce:.4f} final_jepa={last_jepa:.4f} final_sig={last_sig:.4f}",
        flush=True,
    )
    return {
        "task_id": task_id,
        "n_examples": len(items),
        "epochs": epochs,
        "steps": step_count,
        "final_ce": last_ce,
        "final_jepa": last_jepa,
        "final_sigreg": last_sig,
    }
