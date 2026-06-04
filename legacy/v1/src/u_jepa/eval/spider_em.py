"""Spider exact-match eval (string-level proxy) + hidden-state cond number.

Real Spider exact-match is a component-aware metric that parses the SQL AST
and compares SELECT/FROM/WHERE clauses set-wise. We use a normalized
prefix-match instead because (a) Phase 2's gate is a DELTA between two arms
trained and scored with the same metric, so consistency matters more than
absolute correctness, and (b) we want to avoid bringing the heavy Spider
evaluator and its dependency tree onto Kaggle.

`hidden_state_cond_number` is the collapse proxy: the condition number of the
empirical covariance of mean-pooled last-layer hidden states. A healthy model
sits below ~100; a collapsed one shoots toward infinity.
"""
from __future__ import annotations

import re
import time

import torch

from u_jepa.continual.orthogonal_lora import OrthogonalLoRABank
from u_jepa.eval.continual import _resolve_input_device
from u_jepa.util.prompting import format_chat_prompt


_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    s = _WS.sub(" ", s.strip().lower()).rstrip(";").strip()
    # Collapse the common pretty-printed spacing around parentheses so a
    # generation like "count( * )" matches a gold "count(*)".
    s = s.replace("( ", "(").replace(" )", ")")
    return s


@torch.no_grad()
def spider_em(
    bank: OrthogonalLoRABank,
    tokenizer,
    items: list[dict],
    task_id: str | None = None,
    device: str = "cuda:0",
    max_new_tokens: int = 128,
    log_every: int = 25,
    max_len: int = 1024,
) -> float:
    """Return the fraction of items whose normalized generation prefixes match.

    If `task_id` is given, activate that adapter before evaluating. Hooks are
    installed for the duration and removed in a finally block so the bank is
    not left with stray callbacks attached.
    """
    if task_id is not None:
        bank.activate(task_id)
    handles = bank.install_hooks()
    in_device = _resolve_input_device(bank.base, device)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    n = len(items)
    t0 = time.monotonic()
    print(f"[spider_em] start: rows={n} max_new={max_new_tokens}", flush=True)
    try:
        correct = 0
        for i, ex in enumerate(items, start=1):
            formatted, used_template = format_chat_prompt(tokenizer, ex["prompt"])
            enc = tokenizer(
                formatted, return_tensors="pt",
                truncation=True, max_length=max_len,
                add_special_tokens=not used_template,
            )
            ids = enc["input_ids"].to(in_device)
            attn = enc["attention_mask"].to(in_device)
            out = bank.base.generate(
                ids, attention_mask=attn, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=pad_id,
            )
            gen = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            if i <= 3:
                # Smoke check: dump the raw decode for the first few rows so a
                # broken chat template or generation kwargs show up immediately
                # instead of after a 200-row eval all returning 0.
                print(f"[spider_em] sample {i} gen={gen!r} gold={ex['target']!r}",
                      flush=True)
            if _norm(gen).startswith(_norm(ex["target"])):
                correct += 1
            if log_every > 0 and i % log_every == 0:
                elapsed = time.monotonic() - t0
                rate = i / max(elapsed, 1e-6)
                eta = (n - i) / max(rate, 1e-6)
                print(
                    f"[spider_em] {i}/{n} acc={correct/i:.3f} "
                    f"rate={rate:.2f}/s elapsed={elapsed:.0f}s eta={eta:.0f}s",
                    flush=True,
                )
        acc = correct / max(1, n)
        print(f"[spider_em] done: acc={acc:.3f} ({correct}/{n}) in "
              f"{time.monotonic()-t0:.0f}s", flush=True)
        return acc
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def hidden_state_cond_number(
    bank: OrthogonalLoRABank,
    tokenizer,
    prompts: list[str],
    task_id: str | None = None,
    device: str = "cuda:0",
    max_len: int = 512,
    log_every: int = 16,
) -> float:
    """Condition number of the empirical covariance of pooled last-hidden states.

    High values indicate a collapsed or degenerate representation; the Phase 2
    gate is <100. We pool over real tokens only so padding does not pull the
    mean toward zero artificially.
    """
    if task_id is not None:
        bank.activate(task_id)
    handles = bank.install_hooks()
    in_device = _resolve_input_device(bank.base, device)
    n = len(prompts)
    t0 = time.monotonic()
    print(f"[cond] start: prompts={n} max_len={max_len}", flush=True)
    try:
        feats = []
        for i, p in enumerate(prompts, start=1):
            formatted, used_template = format_chat_prompt(tokenizer, p)
            enc = tokenizer(
                formatted, return_tensors="pt", truncation=True,
                max_length=max_len, add_special_tokens=not used_template,
            )
            ids = enc["input_ids"].to(in_device)
            attn = enc["attention_mask"].to(in_device)
            out = bank.base(input_ids=ids, attention_mask=attn,
                            output_hidden_states=True)
            last = out.hidden_states[-1]
            mask = attn.unsqueeze(-1).to(last.dtype)
            pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            feats.append(pooled.float().cpu())
            if log_every > 0 and i % log_every == 0:
                elapsed = time.monotonic() - t0
                rate = i / max(elapsed, 1e-6)
                eta = (n - i) / max(rate, 1e-6)
                print(
                    f"[cond] {i}/{n} rate={rate:.2f}/s "
                    f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                    flush=True,
                )
        H = torch.cat(feats, dim=0)
        if H.shape[0] < 2:
            print(f"[cond] done: not enough samples ({H.shape[0]} < 2), NaN",
                  flush=True)
            return float("nan")
        H = H - H.mean(dim=0, keepdim=True)
        C = H.T @ H / (H.shape[0] - 1)
        s = torch.linalg.svdvals(C)
        smallest = s[-1].clamp(min=1e-12)
        cond = (s[0] / smallest).item()
        print(
            f"[cond] done: cond={cond:.2f} sigma_max={s[0].item():.4g} "
            f"sigma_min={s[-1].item():.4g} in {time.monotonic()-t0:.0f}s",
            flush=True,
        )
        return cond
    finally:
        for h in handles:
            h.remove()
