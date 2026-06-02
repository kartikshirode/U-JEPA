"""Spider EM matcher + cond number on tiny CPU stubs."""
from __future__ import annotations

import types

import torch
import torch.nn as nn

from u_jepa.continual.orthogonal_lora import OrthogonalLoRABank
from u_jepa.eval.spider_em import _norm, hidden_state_cond_number, spider_em


class _ToyTokenizer:
    def __init__(self):
        self.vocab = {"<pad>": 0, "select": 1, "from": 2, "t": 3}
        self.pad_token_id = 0
        self.eos_token_id = 0

    def _encode(self, text):
        ids = []
        for tok in str(text).split():
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
            ids.append(self.vocab[tok])
        return ids

    def __call__(self, text, truncation=False, max_length=None,
                 return_tensors=None, padding=None, add_special_tokens=True):
        ids = self._encode(text)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        attn = [1] * len(ids)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long),
                    "attention_mask": torch.tensor([attn], dtype=torch.long)}
        return {"input_ids": ids, "attention_mask": attn}

    def decode(self, ids, skip_special_tokens=False):
        inv = {v: k for k, v in self.vocab.items()}
        toks = [inv.get(int(i), "?") for i in ids]
        return " ".join(t for t in toks if t != "<pad>")


class _StubGenLM(nn.Module):
    """Stub LM that echoes a fixed reply via .generate()."""
    def __init__(self, d=8, reply=("select", "from", "t")):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.d = d
        self._reply = reply
        for p in self.parameters():
            p.requires_grad = False

    def get_input_embeddings(self):
        return nn.Embedding(64, self.d)

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                output_hidden_states=False):
        B, T = input_ids.shape
        x = input_ids.float().unsqueeze(-1).expand(-1, -1, self.d)
        h = self.v_proj(self.q_proj(x))
        out = types.SimpleNamespace(loss=h.mean(), logits=h)
        if output_hidden_states:
            out.hidden_states = (h,)
        return out

    def generate(self, input_ids, attention_mask=None, max_new_tokens=8,
                 do_sample=False, pad_token_id=0):
        # Lookup token ids for the canned reply via a closure on _vocab.
        # Tests assign self._reply_ids before calling.
        reply_ids = getattr(self, "_reply_ids", [1, 2, 3])
        prefix = input_ids[0].tolist()
        return torch.tensor([prefix + reply_ids], dtype=torch.long)


def test_norm_collapses_whitespace_and_lowercases():
    assert _norm("  SELECT  *  FROM   T") == "select * from t"


def test_spider_em_counts_normalized_prefix_match():
    tok = _ToyTokenizer()
    base = _StubGenLM(d=8)
    base._reply_ids = [tok.vocab["select"], tok.vocab["from"], tok.vocab["t"]]
    bank = OrthogonalLoRABank(base, rank=2, target_modules=("q_proj", "v_proj"))
    bank.add_task("spider")
    items = [
        {"prompt": "what", "target": "select from t"},  # exact match
        {"prompt": "huh",  "target": "SELECT FROM T"},  # case-insensitive
        {"prompt": "no",   "target": "drop everything"},  # mismatch
    ]
    acc = spider_em(bank, tok, items, task_id="spider", device="cpu",
                    max_new_tokens=4, log_every=0)
    assert abs(acc - 2/3) < 1e-6


def test_hidden_state_cond_number_finite_for_random_inputs():
    torch.manual_seed(0)
    tok = _ToyTokenizer()
    base = _StubGenLM(d=8)
    bank = OrthogonalLoRABank(base, rank=2, target_modules=("q_proj", "v_proj"))
    bank.add_task("t")
    prompts = [f"prompt number {i}" for i in range(20)]
    cond = hidden_state_cond_number(bank, tok, prompts, task_id="t",
                                    device="cpu", max_len=16)
    assert cond == cond  # not NaN
    assert cond > 0
