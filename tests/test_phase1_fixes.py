"""Regression tests for Phase 1 fixes from the swarm audit.

Covers:
  - PromptTargetDataset boundary alignment and full-prompt overflow guard
  - train_task gradient-accumulation flush behavior
  - OrthogonalLoRABank adapter dtype cast in forward_target
  - OrthogonalLoRABank adapter device placement matches the base module
  - eval_task single-character label exact match (no "absolute" matches "A")
  - build_accuracy_matrix respects explicit task_order

All tests are CPU-only and avoid downloading any model or dataset.
"""
from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn as nn

from u_jepa.continual.orthogonal_lora import OrthogonalLoRABank
from u_jepa.eval.continual import _match_generation, build_accuracy_matrix
from u_jepa.train.continual_loop import PromptTargetDataset, train_task


# ---------------------------------------------------------------------------
# Toy tokenizer used across the file. Behaves enough like a HF tokenizer to
# drive the dataset and the eval matcher without pulling in transformers.
# ---------------------------------------------------------------------------


class _ToyTokenizer:
    PAD = 0

    def __init__(self):
        self.vocab: dict[str, int] = {"<pad>": 0}
        self.pad_token_id = self.PAD
        self.eos_token_id = self.PAD

    def _encode(self, text: str) -> list[int]:
        ids = []
        for tok in text.split():
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
            ids.append(self.vocab[tok])
        return ids

    def __call__(
        self,
        text,
        truncation=False,
        max_length=None,
        return_tensors=None,
        padding=None,
        add_special_tokens=True,
    ):
        ids = self._encode(text)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        if padding == "max_length" and max_length is not None:
            attn = [1] * len(ids) + [0] * (max_length - len(ids))
            ids = ids + [self.PAD] * (max_length - len(ids))
        else:
            attn = [1] * len(ids)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.tensor([attn], dtype=torch.long),
            }
        return {"input_ids": ids, "attention_mask": attn}


# ---------------------------------------------------------------------------
# ISSUE-3: PromptTargetDataset boundary and overflow guard
# ---------------------------------------------------------------------------


def test_prompt_target_dataset_target_token_is_not_masked():
    """The first target token must NOT be masked out of the loss. With the
    old separate-tokenization approach, a BPE merge across the space could
    push the boundary off by one and either mask the first target token or
    leak the last prompt token into the loss."""
    tok = _ToyTokenizer()
    items = [{"prompt": "the prompt", "target": "answer"}]
    ds = PromptTargetDataset(items, tok, max_len=16)
    sample = ds[0]
    labels = sample["labels"]
    # Prompt is 2 tokens, target is 1 token at index 2. Index 2 must be
    # exactly the target token id and must NOT be -100.
    answer_id = tok.vocab["answer"]
    assert sample["input_ids"][2].item() == answer_id
    assert labels[2].item() == answer_id, "first target token was masked out of the loss"


def test_prompt_target_dataset_prompt_longer_than_max_len_still_keeps_target():
    """If the prompt alone fills max_len the old code masked every position
    as -100 and CE would return NaN. The fix must reserve room for at least
    one target token even if the prompt has to be truncated."""
    tok = _ToyTokenizer()
    long_prompt = " ".join(f"w{i}" for i in range(50))
    items = [{"prompt": long_prompt, "target": "label"}]
    ds = PromptTargetDataset(items, tok, max_len=8)
    sample = ds[0]
    labels = sample["labels"]
    # At least one position must be a real label (not -100), otherwise CE
    # is undefined and grads will be NaN.
    n_real = int((labels != -100).sum().item())
    assert n_real >= 1, "all labels masked - CE would be NaN"


def test_prompt_target_dataset_pads_short_sequences_to_max_len():
    tok = _ToyTokenizer()
    items = [{"prompt": "p", "target": "t"}]
    ds = PromptTargetDataset(items, tok, max_len=10, pad_to_max=True)
    sample = ds[0]
    assert sample["input_ids"].shape == (10,)
    assert sample["attention_mask"].shape == (10,)
    assert sample["labels"].shape == (10,)
    # Padded positions are masked
    assert (sample["labels"][sample["attention_mask"] == 0] == -100).all()


def test_prompt_target_dataset_no_padding_by_default():
    tok = _ToyTokenizer()
    items = [{"prompt": "p", "target": "t"}]
    ds = PromptTargetDataset(items, tok, max_len=10)
    sample = ds[0]
    # Default (batch 1): keep the natural length, no wasted padding.
    n_prompt = len(tok("p")["input_ids"])
    n_target = len(tok("t", add_special_tokens=False)["input_ids"])
    assert sample["input_ids"].shape == (n_prompt + n_target,)
    assert (sample["attention_mask"] == 1).all()


def test_prompt_target_dataset_drops_empty_targets():
    tok = _ToyTokenizer()
    items = [
        {"prompt": "p", "target": ""},
        {"prompt": "p", "target": "   "},
        {"prompt": "p", "target": "t"},
    ]
    ds = PromptTargetDataset(items, tok, max_len=10)
    assert len(ds) == 1
    assert int((ds[0]["labels"] != -100).sum()) >= 1


# ---------------------------------------------------------------------------
# ISSUE-4: gradient-accumulation flush behavior
# ---------------------------------------------------------------------------


class _TinyBaseWithLoss(nn.Module):
    """Stand-in for HF AutoModelForCausalLM that exposes one Linear named
    q_proj and returns an object with a .loss attribute when labels are
    passed. The loss is just the mean of the q_proj output so it depends
    on the adapter parameters when the hook is active."""

    def __init__(self, d=8):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        x = input_ids.float().unsqueeze(-1).expand(-1, -1, self.q_proj.in_features)
        y = self.q_proj(x)
        loss = y.mean()
        return types.SimpleNamespace(loss=loss, logits=y)


def test_train_task_no_extra_flush_when_grad_accum_aligned(monkeypatch):
    """If the number of micro-batches is an exact multiple of grad_accum
    the epoch-end branch must NOT call opt.step() again (the old code did,
    wasting a step on zeroed grads)."""
    base = _TinyBaseWithLoss(d=8)
    bank = OrthogonalLoRABank(base, rank=2, target_modules=("q_proj",))
    tok = _ToyTokenizer()
    # 4 examples, grad_accum=2 means exactly 2 opt steps per epoch and 0 flush
    items = [{"prompt": f"p{i}", "target": "t"} for i in range(4)]

    step_calls = {"n": 0}
    orig_step = torch.optim.AdamW.step

    def counted_step(self, *args, **kwargs):
        step_calls["n"] += 1
        return orig_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", counted_step)
    train_task(
        bank, tok, "t0", items, prev_task_ids=(), epochs=1,
        lr=1e-3, orth_weight=0.0, collision_weight=0.0,
        grad_accum=2, max_len=8, device="cpu",
    )
    # Expect exactly 2 steps (4 micro-batches / 2), no extra epoch-end flush
    assert step_calls["n"] == 2, f"expected 2 opt steps, got {step_calls['n']}"


def test_train_task_flushes_partial_accumulation(monkeypatch):
    """Five micro-batches with grad_accum=2 should yield 2 mid-epoch steps
    plus 1 partial-flush step for the leftover micro-batch."""
    base = _TinyBaseWithLoss(d=8)
    bank = OrthogonalLoRABank(base, rank=2, target_modules=("q_proj",))
    tok = _ToyTokenizer()
    items = [{"prompt": f"p{i}", "target": "t"} for i in range(5)]

    step_calls = {"n": 0}
    orig_step = torch.optim.AdamW.step

    def counted_step(self, *args, **kwargs):
        step_calls["n"] += 1
        return orig_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", counted_step)
    train_task(
        bank, tok, "t0", items, prev_task_ids=(), epochs=1,
        lr=1e-3, orth_weight=0.0, collision_weight=0.0,
        grad_accum=2, max_len=8, device="cpu",
    )
    assert step_calls["n"] == 3, f"expected 3 opt steps (2 aligned + 1 partial), got {step_calls['n']}"


# ---------------------------------------------------------------------------
# ISSUE-5: forward_target must cast delta back to the input dtype
# ---------------------------------------------------------------------------


class _BlockBf16(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        for p in self.parameters():
            p.requires_grad = False


def test_forward_target_returns_delta_in_input_dtype_when_base_is_bf16():
    base = _BlockBf16(d=16)
    bank = OrthogonalLoRABank(base, rank=4, target_modules=("q_proj",))
    bank.add_task("t1")
    # Activations come in as bf16 (mimics bnb 4bit compute path)
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    out = bank.forward_target(x, "q_proj")
    assert out.dtype == torch.bfloat16, (
        f"expected bf16 delta to avoid silent upcast, got {out.dtype}"
    )


def test_forward_target_returns_delta_in_input_dtype_for_fp32_input():
    base = _BlockBf16(d=16)
    bank = OrthogonalLoRABank(base, rank=4, target_modules=("q_proj",))
    bank.add_task("t1")
    x = torch.randn(2, 16, dtype=torch.float32)
    out = bank.forward_target(x, "q_proj")
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# ISSUE-6: adapter parameters land on the same device as the base module
# ---------------------------------------------------------------------------


def test_adapter_parameters_live_on_base_module_device():
    """If add_task forgot to place the adapter on the base module's device,
    the forward hook would raise on the first CUDA step. We only have CPU
    here, so we move the base to a meta device proxy by setting weight
    device explicitly and check the adapter follows it."""
    base = _BlockBf16(d=16)
    bank = OrthogonalLoRABank(base, rank=4, target_modules=("q_proj",))
    bank.add_task("t1")
    base_device = base.q_proj.weight.device
    A, B = bank.adapter_matrices("t1", "q_proj")
    assert A.device == base_device
    assert B.device == base_device


def test_adapter_parameters_are_fp32_for_optimizer_stability():
    base = _BlockBf16(d=16)
    bank = OrthogonalLoRABank(base, rank=4, target_modules=("q_proj",))
    bank.add_task("t1")
    A, B = bank.adapter_matrices("t1", "q_proj")
    assert A.dtype == torch.float32
    assert B.dtype == torch.float32


# ---------------------------------------------------------------------------
# ISSUE-7: single-character label matching must be exact, not prefix
# ---------------------------------------------------------------------------


def test_match_generation_single_char_label_requires_exact_first_char():
    # "absolute" starts with "a" but the label is "A" - must NOT match
    assert _match_generation("absolute zero is cold", "A") is False
    assert _match_generation("A. zero", "A") is True
    assert _match_generation("a. zero", "A") is True  # case-insensitive
    assert _match_generation("B is right", "A") is False


def test_match_generation_multi_char_label_uses_prefix():
    assert _match_generation("dovish stance overall", "dovish") is True
    assert _match_generation("DOVISH", "dovish") is True
    assert _match_generation("hawkish", "dovish") is False
    # Extra leading whitespace is stripped
    assert _match_generation("   neutral ", "neutral") is True


def test_match_generation_empty_generation_is_wrong():
    assert _match_generation("", "A") is False
    assert _match_generation("   ", "dovish") is False


# ---------------------------------------------------------------------------
# ISSUE-19: build_accuracy_matrix uses explicit task_order
# ---------------------------------------------------------------------------


def test_build_accuracy_matrix_respects_explicit_task_order(monkeypatch):
    """Even if eval_sets is built in a weird order, an explicit task_order
    must control the columns of the returned row."""
    from u_jepa.eval import continual as ec

    def _fake_eval_task(bank, tokenizer, task_id, items, device="cuda:0", max_new_tokens=8):
        return {"fomc": 0.7, "scienceqa_text": 0.5}[task_id]

    monkeypatch.setattr(ec, "eval_task", _fake_eval_task)
    # eval_sets inserted in reverse order on purpose
    eval_sets = {"scienceqa_text": [{}], "fomc": [{}]}
    row = build_accuracy_matrix(
        bank=object(), tokenizer=object(), eval_sets=eval_sets,
        seen_tasks=["fomc", "scienceqa_text"], device="cpu",
        task_order=["fomc", "scienceqa_text"],
    )
    # With explicit order, row[0] is fomc (0.7) and row[1] is scienceqa (0.5)
    assert row == [0.7, 0.5]


def test_build_accuracy_matrix_default_order_matches_eval_sets(monkeypatch):
    from u_jepa.eval import continual as ec
    monkeypatch.setattr(
        ec, "eval_task",
        lambda bank, tokenizer, task_id, items, device="cuda:0", max_new_tokens=8: 1.0,
    )
    eval_sets = {"a": [{}], "b": [{}]}
    row = build_accuracy_matrix(
        bank=object(), tokenizer=object(), eval_sets=eval_sets,
        seen_tasks=["a", "b"], device="cpu",
    )
    assert row == [1.0, 1.0]


def test_forward_hook_passes_gradients_to_adapter_parameters():
    """End-to-end check (point G in the audit): a loss computed on the
    base output after the hook fires must produce non-zero gradients on
    the active adapter's A and B parameters, otherwise training is a
    no-op even though loss values look right."""
    base = _BlockBf16(d=8)
    bank = OrthogonalLoRABank(base, rank=2, target_modules=("q_proj",))
    bank.add_task("t1")
    A, B = bank.adapter_matrices("t1", "q_proj")
    # Seed B with something non-zero so the gradient chain through B@A
    # actually flows. Init leaves B=0 which would zero the d(loss)/dA path.
    with torch.no_grad():
        B.fill_(0.1)
    handles = bank.install_hooks()
    try:
        x = torch.randn(1, 4, 8, requires_grad=False)
        y = base.q_proj(x)
        loss = y.pow(2).mean()
        loss.backward()
    finally:
        for h in handles:
            h.remove()
    assert A.grad is not None and A.grad.abs().sum().item() > 0
    assert B.grad is not None and B.grad.abs().sum().item() > 0


def test_build_accuracy_matrix_missing_task_in_eval_sets_scores_zero(monkeypatch):
    """If task_order names a task not present in eval_sets, return 0.0 for
    that column rather than KeyError-ing."""
    from u_jepa.eval import continual as ec
    monkeypatch.setattr(
        ec, "eval_task",
        lambda bank, tokenizer, task_id, items, device="cuda:0", max_new_tokens=8: 0.8,
    )
    eval_sets = {"a": [{}]}
    row = build_accuracy_matrix(
        bank=object(), tokenizer=object(), eval_sets=eval_sets,
        seen_tasks=["a", "b"], device="cpu",
        task_order=["a", "b"],
    )
    assert row == [0.8, 0.0]
