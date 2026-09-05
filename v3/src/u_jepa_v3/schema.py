"""The shared vocabulary. Every corpus normalises into EditCandidate.

FeedEntry wraps a candidate with its position in the simulated maintenance feed
and its relationship to other entries. That relationship is what RQ1 studies: a
poisoned entry, and the revert that corrects it some distance later.

On EditKind. WikiBigEdit tags rows `new` or `update`, which maps to accretion
and revision. It is a useful feature and a cost dimension. It is NOT a safety
bypass: a newly added criminal conviction is harmful without colliding with any
existing slot, and `new` means absent from the earlier Wikidata snapshot rather
than absent from the model's parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EditKind(str, Enum):
    ACCRETION = "accretion"   # the entity did not hold this relation before
    REVISION = "revision"     # an existing object is being replaced


class Decision(str, Enum):
    ADMIT = "admit"
    REFUSE = "refuse"
    QUARANTINE = "quarantine"   # plausible but unverified, held pending evidence


@dataclass(frozen=True)
class EditCandidate:
    subject_id: str
    subject: str
    relation_id: str
    relation: str
    object_id: str | None
    object: str
    prompt: str
    kind: EditKind
    source: str
    timestep: int
    is_adversarial: bool
    risk_category: str | None
    n_hops: int

    def __post_init__(self) -> None:
        for field in ("subject_id", "relation_id", "object", "prompt"):
            if not getattr(self, field):
                raise ValueError(f"{field} must not be blank")
        if self.n_hops < 1:
            raise ValueError(f"n_hops must be >= 1, got {self.n_hops}")
        if self.is_adversarial and not self.risk_category:
            raise ValueError("adversarial candidates need a risk_category")
        if not self.is_adversarial and self.risk_category:
            raise ValueError("benign candidates must not carry a risk_category")

    @property
    def key(self) -> str:
        """The fact slot being written to, ignoring the value."""
        return f"{self.subject_id}:{self.relation_id}"


@dataclass(frozen=True)
class FeedEntry:
    """One entry in the simulated maintenance feed.

    `reverts` holds the entry_id this entry corrects, which is how a revert is
    represented. A revert is a legitimate entry carrying the true value, so it
    is never itself poison.
    """

    candidate: EditCandidate
    position: int
    entry_id: str
    is_poison: bool
    reverts: str | None
    attack_family: str | None

    def __post_init__(self) -> None:
        if self.is_poison and not self.attack_family:
            raise ValueError("poison entries need an attack_family")
        if not self.is_poison and self.attack_family:
            raise ValueError("benign entries must not carry an attack_family")
        if self.is_poison and self.reverts:
            raise ValueError("a poison entry cannot also be a revert")


@dataclass(frozen=True)
class ApplyResult:
    candidate: EditCandidate
    succeeded: bool
    error: str | None = None
