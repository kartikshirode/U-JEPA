"""Label masking guarantees for PromptTargetDataset.

Two failure modes we never want:
  1. An all -100 label row. Cross entropy over an empty target set is undefined
     in PyTorch, which poisons the running loss and can stall or crash training.
     This happens when a long prompt is truncated and the target falls off the
     end.
  2. The prompt tokens leaking into the loss (not masked), which would train the
     model to parrot the question instead of answering it.

The existing test_continual_helpers and test_phase1_fixes touch masking; here we
also assert the exact target token ids survive unmasked and that CE on the
truncated row is actually finite (the concrete consequence of guard 1).
"""
import torch

from u_jepa.train.continual_loop import PromptTargetDataset


class _ToyTokenizer:
    """Whitespace tokenizer with the call signature PromptTargetDataset uses:
    tok(text, add_special_tokens=...) -> {"input_ids": [...]}."""

    PAD = 0

    def __init__(self):
        self.vocab = {"<pad>": 0, "<eos>": 1}
        self.pad_token_id = self.PAD
        self.eos_token_id = 1

    def _encode(self, text):
        ids = []
        for tok in text.split():
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
            ids.append(self.vocab[tok])
        return ids

    def __call__(self, text, add_special_tokens=True, **kwargs):
        return {"input_ids": self._encode(text)}


def test_long_prompt_never_yields_all_masked_row():
    tok = _ToyTokenizer()
    long_prompt = " ".join("w%d" % i for i in range(50))
    ds = PromptTargetDataset([{"prompt": long_prompt, "target": "answer"}], tok, max_len=8)
    item = ds[0]
    assert item["input_ids"].shape == item["labels"].shape
    n_unmasked = int((item["labels"] != -100).sum().item())
    assert n_unmasked >= 1, "long-prompt example produced an all -100 label row"


def test_normal_example_masks_prompt_keeps_exact_target_ids():
    tok = _ToyTokenizer()
    # Encode prompt/target first so we know their exact ids and lengths.
    prompt_ids = tok(" ".join(["what", "is", "two", "plus", "two"]))["input_ids"]
    target_ids = tok("the answer is four", add_special_tokens=False)["input_ids"]
    ds = PromptTargetDataset(
        [{"prompt": "what is two plus two", "target": "the answer is four"}],
        tok,
        max_len=64,
    )
    item = ds[0]
    labels = item["labels"]
    input_ids = item["input_ids"]
    p = len(prompt_ids)
    t = len(target_ids)

    # The whole prompt span is masked out of the loss.
    assert (labels[:p] == -100).all(), "prompt span leaked into the loss"

    # The exact target token ids appear unmasked, in order, right after it.
    assert labels[p : p + t].tolist() == target_ids
    # And they are present on the input side too (not dropped).
    assert input_ids[p : p + t].tolist() == target_ids

    # Padding past the real sequence is masked.
    attn = item["attention_mask"]
    assert (labels[attn == 0] == -100).all()


def test_no_nan_loss_on_truncated_row():
    # Concrete consequence of guard 1: CE over the masked labels is finite.
    tok = _ToyTokenizer()
    long_prompt = " ".join("w%d" % i for i in range(50))
    ds = PromptTargetDataset([{"prompt": long_prompt, "target": "answer"}], tok, max_len=8)
    item = ds[0]
    labels = item["labels"]
    vocab = int(max(item["input_ids"].max().item(), labels.max().item())) + 5
    logits = torch.randn(len(labels), vocab)
    loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)
    assert torch.isfinite(loss), "loss is NaN/Inf on a truncated row"
