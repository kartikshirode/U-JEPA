"""Pure-Python tests for continual_loop.PromptTargetDataset and
eval.continual.build_accuracy_matrix.

We avoid loading any real HF tokenizer or model. A toy whitespace
tokenizer is enough to exercise the masking logic, and a fake bank +
tokenizer suffice for the unseen-task row builder.
"""
from __future__ import annotations
from typing import Iterable

import pytest
import torch

from u_jepa.eval.continual import build_accuracy_matrix
from u_jepa.train.continual_loop import PromptTargetDataset


class _ToyTokenizer:
    """Minimal stand-in: assigns one int id per whitespace token and pads
    with PAD=0. Returns the same dict shape the dataset code expects."""

    PAD = 0

    def __init__(self):
        self.vocab: dict[str, int] = {"<pad>": 0}

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
        # mimic HF's non-tensor return (a BatchEncoding-like dict)
        return _Enc(ids, attn)


class _Enc(dict):
    def __init__(self, ids, attn):
        super().__init__(input_ids=ids, attention_mask=attn)

    @property
    def input_ids(self):
        return self["input_ids"]


def test_prompt_target_dataset_masks_prompt_tokens():
    tok = _ToyTokenizer()
    items = [{"prompt": "classify this please", "target": "dovish"}]
    ds = PromptTargetDataset(items, tok, max_len=16)
    sample = ds[0]
    labels = sample["labels"]
    input_ids = sample["input_ids"]
    attn = sample["attention_mask"]
    assert labels.shape == input_ids.shape == attn.shape == (16,)
    # First three positions correspond to the prompt - must be -100.
    assert (labels[:3] == -100).all()
    # Target position(s) (here token 3) should NOT be masked.
    assert (labels[3] != -100).item()
    # Padded positions must also be -100.
    assert (labels[attn == 0] == -100).all()


def test_prompt_target_dataset_truncation_does_not_overrun():
    tok = _ToyTokenizer()
    items = [{"prompt": "a b c d e f g h", "target": "yes"}]
    ds = PromptTargetDataset(items, tok, max_len=4)
    sample = ds[0]
    assert sample["input_ids"].shape == (4,)
    assert sample["labels"].shape == (4,)


def test_prompt_target_dataset_len_matches_items():
    tok = _ToyTokenizer()
    items = [{"prompt": "p1", "target": "t1"}, {"prompt": "p2", "target": "t2"}]
    ds = PromptTargetDataset(items, tok, max_len=8)
    assert len(ds) == 2


def test_build_accuracy_matrix_unseen_tasks_score_zero(monkeypatch):
    """build_accuracy_matrix must return 0.0 for any task not yet in
    seen_tasks; we should never even try to call eval_task on it."""
    from u_jepa.eval import continual as ec

    seen_calls = []

    def _fake_eval_task(bank, tokenizer, task_id, items, device="cuda:0", max_new_tokens=8):
        seen_calls.append(task_id)
        return 0.5

    monkeypatch.setattr(ec, "eval_task", _fake_eval_task)
    eval_sets = {"fomc": [{}], "scienceqa_text": [{}]}
    row = build_accuracy_matrix(
        bank=object(),
        tokenizer=object(),
        eval_sets=eval_sets,
        seen_tasks=["fomc"],
        device="cpu",
    )
    assert row == [0.5, 0.0]
    assert seen_calls == ["fomc"], "should not invoke eval_task on unseen task"


def test_build_accuracy_matrix_all_seen(monkeypatch):
    from u_jepa.eval import continual as ec

    monkeypatch.setattr(
        ec, "eval_task",
        lambda bank, tokenizer, task_id, items, device="cuda:0", max_new_tokens=8: 0.9,
    )
    eval_sets = {"a": [{}], "b": [{}], "c": [{}]}
    row = build_accuracy_matrix(
        bank=object(), tokenizer=object(), eval_sets=eval_sets,
        seen_tasks=["a", "b", "c"], device="cpu",
    )
    assert row == [0.9, 0.9, 0.9]
