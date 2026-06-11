"""Edit-success, locality, and perplexity metrics for knowledge editing.

The F1 spike uses these to judge whether GRACE actually edited GPT-2-XL without
wrecking nearby facts or the model's general fluency. The perplexity and
locality functions are reused by the Phase 2 probe (two of its three signals).

String scoring (`normalize`, `match_prefix`) and the logit math
(`perplexity_from_logits`) are pure, so they unit-test on a CPU stub without a
real model. The generation wrappers (`greedy_generate`, `score_edits`,
`sequence_perplexity`) take any HF causal-LM and tokenizer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_WS = re.compile(r"\s+")
_STRIP = " .,:;!?\"'“”‘’"


def normalize(s: str) -> str:
    """Lowercase, collapse internal whitespace, trim surrounding punctuation."""
    return _WS.sub(" ", str(s).strip().lower()).strip(_STRIP)


def match_prefix(generation: str, target: str) -> bool:
    """True when the normalized generation begins with the normalized target.

    Edit targets here are words or short phrases (CounterFact style), so a
    prefix match on the continuation is enough. An empty target never matches,
    so a model that emits nothing cannot score a free hit.
    """
    t = normalize(target)
    if not t:
        return False
    return normalize(generation).startswith(t)


def perplexity_from_logits(logits, targets) -> float:
    """exp(mean token cross-entropy).

    `logits[i]` is the distribution predicting `targets[i]`; the caller does the
    shift. Pure tensor math, no model. Uniform logits over a vocab of V return
    perplexity ~= V; a distribution peaked on the right token returns ~1.
    """
    import torch
    import torch.nn.functional as F

    ce = F.cross_entropy(logits.float(), targets, reduction="mean")
    return float(torch.exp(ce))


@dataclass
class Edit:
    """One knowledge edit, CounterFact-shaped.

    `neighborhood` items are {prompt, ground_truth} dicts that must stay
    unchanged after the edit (the locality check). `paraphrases` are alternate
    phrasings of the edited fact that should also flip to `target_new`
    (generalization).
    """

    prompt: str
    target_new: str
    ground_truth: str
    subject: str = ""
    paraphrases: list[str] = field(default_factory=list)
    neighborhood: list[dict] = field(default_factory=list)


def _input_device(model):
    """Device the model wants token inputs on (embedding layer, then any param)."""
    try:
        emb = model.get_input_embeddings()
        if emb is not None and getattr(emb, "weight", None) is not None:
            return emb.weight.device
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except StopIteration:
        import torch

        return torch.device("cpu")


def greedy_generate(model, tok, prompt: str, max_new_tokens: int = 8) -> str:
    """Greedy-decode the continuation after `prompt`, return decoded text only."""
    import torch

    dev = _input_device(model)
    enc = tok(prompt, return_tensors="pt")
    ids = enc["input_ids"].to(dev)
    attn = enc.get("attention_mask")
    attn = attn.to(dev) if attn is not None else None
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    with torch.no_grad():
        out = model.generate(
            ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad,
        )
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def score_edits(model, tok, edits: list[Edit], max_new_tokens: int = 8) -> dict:
    """Edit-success, locality, and paraphrase-success over a list of Edits.

    - edit_success: fraction whose generation matches target_new.
    - locality_preserved: fraction of neighborhood prompts still matching their
      ground_truth (None if no neighborhood probes supplied).
    - paraphrase_success: fraction of paraphrases matching target_new (None if
      none supplied).
    Returns the aggregates plus a per-edit detail list for eyeballing.
    """
    succ = total = 0
    loc_ok = loc_total = 0
    para_ok = para_total = 0
    details: list[dict] = []

    for e in edits:
        gen = greedy_generate(model, tok, e.prompt, max_new_tokens)
        hit = match_prefix(gen, e.target_new)
        succ += int(hit)
        total += 1
        d: dict[str, Any] = {
            "prompt": e.prompt,
            "target_new": e.target_new,
            "generation": gen,
            "success": hit,
            "neighborhood": [],
            "paraphrases": [],
        }
        for nb in e.neighborhood:
            g = greedy_generate(model, tok, nb["prompt"], max_new_tokens)
            ok = match_prefix(g, nb["ground_truth"])
            loc_ok += int(ok)
            loc_total += 1
            d["neighborhood"].append(
                {
                    "prompt": nb["prompt"],
                    "ground_truth": nb["ground_truth"],
                    "generation": g,
                    "preserved": ok,
                }
            )
        for pp in e.paraphrases:
            g = greedy_generate(model, tok, pp, max_new_tokens)
            ok = match_prefix(g, e.target_new)
            para_ok += int(ok)
            para_total += 1
            d["paraphrases"].append({"prompt": pp, "generation": g, "success": ok})
        details.append(d)

    return {
        "edit_success": succ / max(1, total),
        "locality_preserved": (loc_ok / loc_total) if loc_total else None,
        "paraphrase_success": (para_ok / para_total) if para_total else None,
        "n_edits": total,
        "n_neighborhood": loc_total,
        "n_paraphrase": para_total,
        "details": details,
    }


def sequence_perplexity(model, tok, text: str, max_len: int = 512) -> float:
    """Next-token perplexity of `text` under the model. NaN if under 2 tokens."""
    import torch

    dev = _input_device(model)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len)
    ids = enc["input_ids"].to(dev)
    if ids.shape[1] < 2:
        return float("nan")
    with torch.no_grad():
        logits = model(input_ids=ids).logits  # (1, T, V)
    # logits at position t predict token t+1, so drop the last logit and the
    # first target to line them up.
    shift_logits = logits[0, :-1, :]
    shift_targets = ids[0, 1:]
    return perplexity_from_logits(shift_logits, shift_targets)


def mean_heldout_perplexity(model, tok, texts: list[str]) -> float:
    """Mean per-sequence perplexity over held-out probe texts, ignoring NaNs."""
    vals = [sequence_perplexity(model, tok, t) for t in texts]
    vals = [v for v in vals if v == v]  # drop NaN
    return (sum(vals) / len(vals)) if vals else float("nan")
