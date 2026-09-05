"""One interface for every editor, and the only way to obtain a responder.

Editors expose responder() rather than taking one, because the previous design
let a caller hold a responder that was never bound to the edited model. Probes
then measured the untouched model for the whole run and every test passed.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..schema import ApplyResult, EditCandidate


@runtime_checkable
class Responder(Protocol):
    def answer(self, prompts: list[str]) -> list[str]:
        ...


@runtime_checkable
class Editor(Protocol):
    name: str

    def apply(self, batch: list[EditCandidate]) -> list[ApplyResult]:
        """Apply every candidate in order. Never raises on a single failure."""
        ...

    def responder(self) -> Responder:
        """A responder reflecting every edit applied so far."""
        ...
