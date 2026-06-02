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
        # prompt masking, NaN guard) lives in one place. Then re-run the same
        # keep predicate over the source items so the surviving view_b strings
        # stay in lock-step with PromptTargetDataset's filtered rows.
        self.inner = PromptTargetDataset(items, tokenizer, max_len=max_len)
        kept = []
        for ex in items:
            tgt = ex.get("target", "")
            if not str(tgt).strip():
                continue
            n_tgt = len(tokenizer(str(tgt), add_special_tokens=False)["input_ids"])
            if n_tgt < 1:
                continue
            kept.append(ex)
        self.view_b = [ex.get("view_b", ex.get("target", "")) for ex in kept]
        assert len(self.view_b) == len(self.inner), (
            f"view_b/inner length mismatch: {len(self.view_b)} vs {len(self.inner)}"
        )

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> dict:
        row = self.inner[idx]
        row["view_b_text"] = self.view_b[idx]
        return row


def _spider_collate(batch: list[dict]) -> dict:
    """Collate that keeps view_b_text as a plain string list."""
    assert len(batch) == 1, "JEPA loop is batch_size=1"
    out = {k: v for k, v in batch[0].items() if k != "view_b_text"}
    out = {k: v.unsqueeze(0) if torch.is_tensor(v) and v.dim() == 1 else v
           for k, v in out.items()}
    out["view_b_text"] = batch[0]["view_b_text"]
    return out


@torch.no_grad()
def _pool_view_b(model, tokenizer, text: str, in_device, max_len: int) -> torch.Tensor:
    """No-grad forward on view B; return mean-pooled last hidden state (1, D)."""
    formatted, used_template = format_chat_prompt(tokenizer, text)
    enc = tokenizer(
        formatted, return_tensors="pt", truncation=True, max_length=max_len,
        add_special_tokens=not used_template,
    )
    ids = enc["input_ids"].to(in_device)
    attn = enc["attention_mask"].to(in_device)
    out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
    last = out.hidden_states[-1]  # (1, T, D)
    mask = attn.unsqueeze(-1).to(last.dtype)
    pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return pooled  # (1, D)


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
                mask_a = attn.unsqueeze(-1).to(last_hidden.dtype)
                h_a_pooled = (last_hidden * mask_a).sum(dim=1) / mask_a.sum(dim=1).clamp(min=1)

                h_b = _pool_view_b(
                    bank.base, tokenizer, batch["view_b_text"], in_device, max_len,
                )
                # Predictor lives in fp32 for stable updates; cast inputs in to
                # match its weight dtype so torch does not silently upcast the
                # whole graph or refuse the matmul.
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
                    sig = sigreg_loss(tokens.to(pred_dtype), num_slices=sigreg_slices)
                else:
                    sig = torch.zeros((), device=in_device, dtype=pred_dtype)

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
