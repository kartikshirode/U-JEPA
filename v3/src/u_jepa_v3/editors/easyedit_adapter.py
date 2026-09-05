"""Wraps EasyEdit's BaseEditor so every method looks identical from above.

edit() returns (metrics, edited_model, weights_copy). Keeping the edited model
is the whole job of this class. Dropping it, as an earlier version did, means
every probe reads the untouched base and every sequential edit restarts from it.

The easyeditor import is deferred to first use so payload construction stays
testable on a laptop with no CUDA and no easyeditor installed.
"""
from __future__ import annotations

from ..schema import ApplyResult, EditCandidate

SUPPORTED_METHODS = ("ultraedit", "alphaedit", "rome", "memit", "wise", "grace")


class HFResponder:
    """Greedy short-generation responder over a HuggingFace model."""

    def __init__(self, model, tokenizer, max_new_tokens: int = 24) -> None:
        self._model = model
        self._tok = tokenizer
        self._max_new_tokens = max_new_tokens

    def answer(self, prompts: list[str]) -> list[str]:
        import torch

        if not prompts:
            return []
        batch = self._tok(prompts, return_tensors="pt", padding=True).to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **batch, max_new_tokens=self._max_new_tokens, do_sample=False,
                pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id,
            )
        cut = batch["input_ids"].shape[1]
        return [self._tok.decode(row[cut:], skip_special_tokens=True).strip() for row in out]


class EasyEditAdapter:
    def __init__(self, method: str, hparams_path: str, sequential: bool = True,
                 max_new_tokens: int = 24) -> None:
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"method {method!r} not in {SUPPORTED_METHODS}")
        self.method = method
        self.hparams_path = hparams_path
        self.sequential = sequential
        self.max_new_tokens = max_new_tokens
        self.name = f"easyedit:{method}"
        self.edited_model = None
        self._editor = None
        self._tokenizer = None

    def to_easyedit_payload(self, batch: list[EditCandidate]) -> dict:
        """Map candidates onto edit()'s keyword arguments.

        ground_truth stays None on purpose: we assert the new value rather than
        claiming to know what the model currently believes, and guessing would
        inject an assumption into every measurement downstream.
        """
        return {
            "prompts": [c.prompt for c in batch],
            "target_new": [c.object for c in batch],
            "subject": [c.subject for c in batch],
            "ground_truth": None,
            "sequential_edit": self.sequential,
        }

    def _ensure_editor(self):
        if self._editor is not None:
            return self._editor
        from easyeditor import BaseEditor, get_hparams  # deferred on purpose

        hparams = get_hparams(self.method, self.hparams_path)
        self._editor = BaseEditor.from_hparams(hparams)
        self._tokenizer = getattr(self._editor, "tok", None)
        return self._editor

    def apply(self, batch: list[EditCandidate]) -> list[ApplyResult]:
        if not batch:
            return []
        payload = self.to_easyedit_payload(batch)
        try:
            editor = self._ensure_editor()
            _, edited_model, _ = editor.edit(**payload)
            self.edited_model = edited_model
        except Exception as exc:  # one bad batch must not kill a 100K-edit run
            return [ApplyResult(c, False, f"{type(exc).__name__}: {exc}") for c in batch]
        return [ApplyResult(c, True, None) for c in batch]

    def responder(self) -> HFResponder:
        if self.edited_model is None:
            raise RuntimeError(
                "no model yet: apply() has not succeeded, so there is nothing to probe. "
                "Measure the untouched base through a separate base responder."
            )
        return HFResponder(self.edited_model, self._tokenizer, self.max_new_tokens)
