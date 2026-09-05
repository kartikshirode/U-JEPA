"""What the gate is allowed to see, and the shape every signal has.

The gate decides before an edit is applied. It therefore sees an entry the way
an operator's ingestion pipeline sees one: a claim, an account that submitted it,
a position in the stream, and whatever the operator already trusts.

GateInput exists because the harness knows the answer. FeedEntry carries
is_poison and attack_family, and EditCandidate carries source, which the attack
generators set to the family name. A signal that read any of those would score
perfectly and mean nothing. So the gate never receives a FeedEntry. It receives a
redacted view built by from_entry, which drops the labels and replaces source
with the simulated account. Discipline would have been enough right up until it
was not, and this is the same class of mistake as the editor that returned a
responder bound to the base model.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Callable, Protocol, runtime_checkable

from ..data.relation_prior import RelationPrior
from ..schema import Decision, EditCandidate, EditKind, FeedEntry
from .provenance import SourceTrust


@dataclass(frozen=True)
class GateInput:
    """One entry as the gate sees it. No ground truth reachable from here."""

    entry_id: str
    position: int
    source: str
    candidate: EditCandidate

    @classmethod
    def from_entry(cls, entry: FeedEntry, source: str) -> "GateInput":
        redacted = replace(
            entry.candidate,
            source=source,
            is_adversarial=False,
            risk_category=None,
        )
        return cls(entry_id=entry.entry_id, position=entry.position,
                   source=source, candidate=redacted)

    @property
    def key(self) -> str:
        return self.candidate.key


@dataclass
class GateContext:
    """Everything the operator knows before deciding, updated as the feed streams.

    object_vocab and slot_values start empty and fill from admitted entries, so
    the gate bootstraps from its own history rather than from an oracle. Seed
    them from a trusted snapshot when one exists; `prime` does that.
    """

    prior: RelationPrior | None = None
    trust: SourceTrust | None = None
    belief: Callable[[list[str]], list[str]] | None = None
    window: int = 200
    object_vocab: dict[str, set[str]] = field(default_factory=dict)
    slot_values: dict[str, str] = field(default_factory=dict)
    slot_writes: dict[str, int] = field(default_factory=dict)
    _recent: deque = field(default_factory=deque)

    def prime(self, candidates: list[EditCandidate]) -> None:
        """Load a trusted snapshot: what each relation's objects look like."""
        for candidate in candidates:
            self.object_vocab.setdefault(candidate.relation_id, set()).add(candidate.object)
            self.slot_values[candidate.key] = candidate.object

    def observe(self, gate_input: GateInput) -> None:
        """Record an admitted entry. Refused entries must not be recorded.

        An attacker who can push a refused entry into the trusted vocabulary
        gets to define what counts as normal, which is the slow version of the
        same attack.
        """
        candidate = gate_input.candidate
        self.object_vocab.setdefault(candidate.relation_id, set()).add(candidate.object)
        self.slot_values[candidate.key] = candidate.object
        self.slot_writes[candidate.key] = self.slot_writes.get(candidate.key, 0) + 1
        self._recent.append((gate_input.position, gate_input.source,
                             candidate.subject_id))
        self._trim(gate_input.position)

    def _trim(self, position: int) -> None:
        while self._recent and self._recent[0][0] < position - self.window:
            self._recent.popleft()

    def recent_from_source(self, source: str, position: int) -> int:
        self._trim(position)
        return sum(1 for _, src, _ in self._recent if src == source)

    def recent_for_subject(self, subject_id: str, position: int) -> int:
        self._trim(position)
        return sum(1 for _, _, subj in self._recent if subj == subject_id)


@runtime_checkable
class Signal(Protocol):
    """One suspicion score per entry, in [0, 1], higher meaning more suspicious.

    Batched rather than per entry because the belief signal costs a generation
    call, and one call for 64 entries is not 64 calls.
    """

    name: str

    def score(self, batch: list[GateInput], ctx: GateContext) -> list[float]:
        ...


@dataclass(frozen=True)
class GateScore:
    entry_id: str
    total: float
    per_signal: dict[str, float]


@dataclass(frozen=True)
class GateDecision:
    entry_id: str
    decision: Decision
    score: GateScore

    @property
    def admitted(self) -> bool:
        return self.decision is Decision.ADMIT


def is_revision(gate_input: GateInput, ctx: GateContext) -> bool:
    """A revision as the gate can tell, which is not the corpus tag.

    WikiBigEdit's `update` tag says the slot changed between two Wikidata
    snapshots. The gate does not have those snapshots. It has its own record of
    what it admitted, so a revision is an entry writing a slot it has seen
    before with a different value. The corpus tag is used only as a fallback for
    slots the gate has never seen.
    """
    known = ctx.slot_values.get(gate_input.key)
    if known is not None:
        return known != gate_input.candidate.object
    return gate_input.candidate.kind is EditKind.REVISION
