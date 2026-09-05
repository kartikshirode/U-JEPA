"""SST, MMLU, MRPC and NLI, matching UltraEdit's own evaluation set.

Same four suites they report, so v3 numbers sit beside theirs without
translation. This is also the stealth detector: an attack that leaves these flat
while corrupting target knowledge is the dangerous kind, and EditRisk-Bench
found that is the common case.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..editors.base import Responder
from .efficacy import _accuracy

REQUIRED_SUITES = ("sst", "mmlu", "mrpc", "nli")


@dataclass(frozen=True)
class GeneralAbility:
    sst: float
    mmlu: float
    mrpc: float
    nli: float

    @property
    def mean(self) -> float:
        return (self.sst + self.mmlu + self.mrpc + self.nli) / 4


def general_ability(
    responder: Responder, suites: dict[str, list[tuple[str, str]]]
) -> GeneralAbility:
    missing = [s for s in REQUIRED_SUITES if s not in suites]
    if missing:
        raise ValueError(f"general_ability needs all of {REQUIRED_SUITES}, missing {missing}")

    scores = {}
    for suite in REQUIRED_SUITES:
        pairs = suites[suite]
        got = responder.answer([p for p, _ in pairs]) if pairs else []
        scores[suite] = _accuracy(got, [a for _, a in pairs])
    return GeneralAbility(**scores)
