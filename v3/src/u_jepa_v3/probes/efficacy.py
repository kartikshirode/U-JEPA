"""Did the edit take, and did the neighbours survive.

Answers are normalised before comparison because exact string match on raw
generation measures formatting rather than knowledge. v1 learned that the
expensive way when a chat model's preamble pushed the label out of the eval
window and corrupted a whole accuracy table.
"""
from __future__ import annotations

import re
import string

from ..editors.base import Responder
from ..schema import EditCandidate

_ARTICLES = {"a", "an", "the"}


def normalize_answer(text: str) -> str:
    """Lowercase, drop punctuation and articles, collapse whitespace."""
    lowered = text.lower().strip()
    stripped = lowered.translate(str.maketrans("", "", string.punctuation))
    words = [w for w in re.split(r"\s+", stripped) if w and w not in _ARTICLES]
    return " ".join(words)


def _accuracy(got: list[str], want: list[str]) -> float:
    if not want:
        return 0.0
    hits = sum(normalize_answer(g) == normalize_answer(w) for g, w in zip(got, want))
    return hits / len(want)


def efficacy(responder: Responder, candidates: list[EditCandidate]) -> float:
    """Share of edits whose new object the model now returns."""
    if not candidates:
        return 0.0
    got = responder.answer([c.prompt for c in candidates])
    return _accuracy(got, [c.object for c in candidates])


def locality(responder: Responder, pairs: list[tuple[str, str]]) -> float:
    """Share of unrelated (prompt, expected) pairs still answered correctly."""
    if not pairs:
        return 0.0
    return _accuracy(responder.answer([p for p, _ in pairs]), [a for _, a in pairs])
