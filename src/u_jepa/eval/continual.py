"""Per-task eval harness for sequential continual learning."""
from __future__ import annotations
from typing import Sequence

import torch

from u_jepa.continual.orthogonal_lora import OrthogonalLoRABank


def _match_generation(generated: str, target: str) -> bool:
    """Decide whether a model generation counts as correct for a label target.

    Long targets (dovish, hawkish, neutral) use a case-insensitive prefix
    match. Single-character labels like A..H in ScienceQA must match the
    first non-space character exactly AND the character that follows must
    be a non-letter (whitespace, punctuation, or end of string). This
    prevents trivial false positives like the generation "absolute zero"
    counting as the label "A".
    """
    gen_stripped = generated.strip()
    tgt_stripped = target.strip()
    if not tgt_stripped:
        return False
    if len(tgt_stripped) == 1:
        if not gen_stripped:
            return False
        first = gen_stripped[0]
        if first.lower() != tgt_stripped.lower():
            return False
        # The next char (if any) must not be a letter, otherwise the model
        # produced a longer word that merely starts with the right letter.
        if len(gen_stripped) == 1:
            return True
        nxt = gen_stripped[1]
        return not nxt.isalpha()
    return gen_stripped.lower().startswith(tgt_stripped.lower())


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

    Match is case-insensitive prefix for multi-char targets and exact
    first-character match for single-char targets (A..H letters) so we
    do not get false positives from generations like "absolute" matching
    target "A".
    """
    bank.activate(task_id)
    handles = bank.install_hooks()
    # If the tokenizer has no explicit pad token (some chat models ship
    # without one), fall back to the EOS id so generate() does not crash
    # with a "pad token id is required" warning that silences the output.
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    try:
        correct = 0
        for ex in items:
            enc = tokenizer(ex["prompt"], return_tensors="pt")
            ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            out = bank.base.generate(
                ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
            gen = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            if _match_generation(gen, ex["target"]):
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
    task_order: Sequence[str] | None = None,
) -> list[float]:
    """Evaluate the bank on every task and return one row of A.

    `task_order` is the explicit ordering of columns (defaults to the
    insertion order of eval_sets so older callers keep working). Passing
    it explicitly avoids relying on dict-iteration order, which is a
    language guarantee on CPython but easy to misread when the dict is
    rebuilt or merged elsewhere.
    """
    order = list(task_order) if task_order is not None else list(eval_sets.keys())
    row: list[float] = []
    for tname in order:
        if tname not in eval_sets:
            row.append(0.0)
            continue
        if tname in seen_tasks:
            row.append(eval_task(bank, tokenizer, tname, eval_sets[tname], device=device))
        else:
            # Task not yet seen: 0 by convention
            row.append(0.0)
    return row
