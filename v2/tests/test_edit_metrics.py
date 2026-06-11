"""Tests for edit_metrics.

The string scoring and logit math are pure and tested directly. score_edits and
sequence_perplexity run on a toy tokenizer + stub model on CPU, mirroring the v1
test_spider_em pattern, so no real model loads.
"""

from __future__ import annotations

import types

import pytest

from u_jepa_v2.eval import edit_metrics as em
from u_jepa_v2.eval.edit_metrics import Edit


# --- pure string scoring --------------------------------------------------


def test_normalize_lowercases_collapses_and_trims_punct():
    assert em.normalize("  The  Eiffel Tower. ") == "the eiffel tower"
    assert em.normalize("Rome!") == "rome"


def test_match_prefix_true_on_leading_target():
    assert em.match_prefix("Rome, the capital of Italy", "Rome") is True
    assert em.match_prefix("rome", "Rome") is True  # case-insensitive


def test_match_prefix_false_when_target_not_leading():
    assert em.match_prefix("Paris is the answer", "Rome") is False


def test_match_prefix_empty_target_never_matches():
    # A model that emits nothing must not score a free hit.
    assert em.match_prefix("", "Rome") is False
    assert em.match_prefix("anything", "") is False


# --- perplexity_from_logits ----------------------------------------------


def test_perplexity_uniform_logits_approx_vocab_size():
    torch = pytest.importorskip("torch")
    V = 50
    logits = torch.zeros(4, V)  # uniform -> CE = ln(V) -> ppl = V
    targets = torch.tensor([0, 1, 2, 3])
    ppl = em.perplexity_from_logits(logits, targets)
    assert abs(ppl - V) < 1e-3


def test_perplexity_peaked_logits_approx_one():
    torch = pytest.importorskip("torch")
    logits = torch.full((3, 10), -1e4)
    targets = torch.tensor([2, 5, 7])
    for i, t in enumerate(targets):
        logits[i, t] = 1e4  # almost all mass on the right token
    ppl = em.perplexity_from_logits(logits, targets)
    assert ppl < 1.01


# --- stub model + toy tokenizer for the generation wrappers ---------------


class _ToyTokenizer:
    def __init__(self):
        self.vocab = {"<pad>": 0}
        self.pad_token_id = 0
        self.eos_token_id = 0

    def _encode(self, text):
        ids = []
        for tok in str(text).split():
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
            ids.append(self.vocab[tok])
        return ids

    def __call__(self, text, truncation=False, max_length=None, return_tensors=None):
        import torch

        ids = self._encode(text)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        attn = [1] * len(ids)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.tensor([attn], dtype=torch.long),
            }
        return {"input_ids": ids, "attention_mask": attn}

    def decode(self, ids, skip_special_tokens=False):
        inv = {v: k for k, v in self.vocab.items()}
        toks = [inv.get(int(i), "?") for i in ids]
        return " ".join(t for t in toks if t != "<pad>")


def _stub_model(reply_words, tok, logits_vocab=None):
    """Stub HF-ish model: generate() appends canned tokens, __call__ returns
    fixed logits. Exposes get_input_embeddings + a parameter for device probing."""
    import torch
    import torch.nn as nn

    reply_ids = tok._encode(" ".join(reply_words))
    V = logits_vocab or 16

    class _Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(64, 4)

        def get_input_embeddings(self):
            return self.emb

        def generate(self, input_ids, attention_mask=None, max_new_tokens=8,
                     do_sample=False, pad_token_id=0):
            prefix = input_ids[0].tolist()
            return torch.tensor([prefix + reply_ids[:max_new_tokens]], dtype=torch.long)

        def forward(self, input_ids=None, **kw):
            T = input_ids.shape[1]
            # uniform logits so perplexity is deterministic (~V)
            logits = torch.zeros(1, T, V)
            return types.SimpleNamespace(logits=logits)

    return _Stub()


def test_score_edits_counts_success_and_locality():
    tok = _ToyTokenizer()
    # Model always replies "Rome" regardless of prompt.
    model = _stub_model(["Rome"], tok)
    edits = [
        Edit(
            prompt="Eiffel Tower city",
            target_new="Rome",
            ground_truth="Paris",
            neighborhood=[{"prompt": "Louvre city", "ground_truth": "Rome"}],
            paraphrases=["where Eiffel Tower"],
        ),
    ]
    out = em.score_edits(model, tok, edits, max_new_tokens=2)
    assert out["edit_success"] == 1.0          # generation "Rome" matches target
    assert out["locality_preserved"] == 1.0    # neighborhood gt is also "Rome" here
    assert out["paraphrase_success"] == 1.0
    assert out["n_edits"] == 1
    assert out["details"][0]["success"] is True


def test_score_edits_locality_fails_when_neighborhood_changes():
    tok = _ToyTokenizer()
    model = _stub_model(["Rome"], tok)  # everything answers "Rome"
    edits = [
        Edit(
            prompt="Eiffel Tower city",
            target_new="Rome",
            ground_truth="Paris",
            neighborhood=[{"prompt": "Louvre city", "ground_truth": "Paris"}],
        ),
    ]
    out = em.score_edits(model, tok, edits)
    assert out["edit_success"] == 1.0
    assert out["locality_preserved"] == 0.0    # Louvre should be Paris, model said Rome
    assert out["paraphrase_success"] is None


def test_sequence_perplexity_matches_uniform_expectation():
    tok = _ToyTokenizer()
    model = _stub_model(["x"], tok, logits_vocab=20)
    ppl = em.sequence_perplexity(model, tok, "one two three four")
    assert abs(ppl - 20) < 1e-3


def test_sequence_perplexity_nan_on_single_token():
    tok = _ToyTokenizer()
    model = _stub_model(["x"], tok)
    val = em.sequence_perplexity(model, tok, "single")
    assert val != val  # NaN


def test_mean_heldout_perplexity_ignores_nan():
    tok = _ToyTokenizer()
    model = _stub_model(["x"], tok, logits_vocab=10)
    mean = em.mean_heldout_perplexity(model, tok, ["a b c", "x"])  # second is 1 token -> NaN
    assert abs(mean - 10) < 1e-3
