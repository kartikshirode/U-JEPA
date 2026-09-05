"""An editor that records instead of editing, so the harness tests on CPU.

Its responder answers from the edits it accepted. That is deliberate: a stub
whose responder ignored edits would reproduce the exact bug this design exists
to prevent, and would do it invisibly.
"""
from __future__ import annotations

from ..schema import ApplyResult, EditCandidate

UNEDITED = "<unedited>"


class _StubResponder:
    def __init__(self, table: dict[str, str]) -> None:
        self._table = table

    def answer(self, prompts: list[str]) -> list[str]:
        return [self._table.get(p, UNEDITED) for p in prompts]


class StubEditor:
    name = "stub"

    def __init__(self, fail_keys: set[str] | None = None) -> None:
        self.fail_keys = fail_keys or set()
        self.applied: list[EditCandidate] = []
        self._answers: dict[str, str] = {}

    def apply(self, batch: list[EditCandidate]) -> list[ApplyResult]:
        results = []
        for candidate in batch:
            self.applied.append(candidate)
            if candidate.key in self.fail_keys:
                results.append(ApplyResult(candidate, False, "stub-forced failure"))
                continue
            self._answers[candidate.prompt] = candidate.object
            results.append(ApplyResult(candidate, True, None))
        return results

    def responder(self) -> _StubResponder:
        return _StubResponder(dict(self._answers))
