"""Who submitted the claim, and how much that account has earned.

WikiBigEdit has no submitter column, so provenance has to be simulated. That is
a modelling assumption and it is stated here rather than buried: entries are
attributed to a pool of accounts, the attacker controls a few of them, and those
accounts also carry ordinary traffic.

The cover traffic is the point. Give the attacker accounts that submit nothing
but poison and the source becomes the label; a gate would then score perfectly
on an artefact of the simulation. Real vandalism accounts on Wikidata mix real
edits with bad ones, which is what makes them hard.

Trust is a smoothed record of being reverted. It updates only from corrections
the operator actually observed, so it is late by construction. An attacker who
burns an account gets one campaign out of it, which is the property worth
measuring rather than assuming away.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..schema import FeedEntry

DEFAULT_PRIOR_TRUST = 0.5
DEFAULT_PRIOR_WEIGHT = 10.0


@dataclass(frozen=True)
class SourceRecord:
    source: str
    n_seen: int
    n_reverted: int

    @property
    def revert_rate(self) -> float:
        return self.n_reverted / self.n_seen if self.n_seen else 0.0


class SourceTrust:
    """Beta-smoothed trust per account, in [0, 1], 1 being never reverted.

    prior_weight is how many clean entries an unknown account is treated as
    already having. Low, and one revert destroys an account. High, and a
    long-lived account can absorb a campaign without moving. 10 is a starting
    point, not a result; RQ3 sweeps it.
    """

    def __init__(self, prior_trust: float = DEFAULT_PRIOR_TRUST,
                 prior_weight: float = DEFAULT_PRIOR_WEIGHT) -> None:
        if not 0.0 <= prior_trust <= 1.0:
            raise ValueError(f"prior_trust must be in [0, 1], got {prior_trust}")
        if prior_weight <= 0:
            raise ValueError(f"prior_weight must be > 0, got {prior_weight}")
        self.prior_trust = prior_trust
        self.prior_weight = prior_weight
        self._seen: dict[str, int] = {}
        self._reverted: dict[str, int] = {}

    def observe(self, source: str, reverted: bool) -> None:
        self._seen[source] = self._seen.get(source, 0) + 1
        if reverted:
            self._reverted[source] = self._reverted.get(source, 0) + 1

    def record(self, source: str) -> SourceRecord:
        return SourceRecord(source, self._seen.get(source, 0),
                            self._reverted.get(source, 0))

    def trust(self, source: str) -> float:
        seen = self._seen.get(source, 0)
        clean = seen - self._reverted.get(source, 0)
        return ((self.prior_weight * self.prior_trust + clean)
                / (self.prior_weight + seen))

    def known(self) -> list[str]:
        return sorted(self._seen)


def simulate_sources(
    feed: list[FeedEntry],
    seed: int,
    n_sources: int = 8,
    n_attacker_sources: int = 2,
    cover_rate: float = 0.35,
) -> dict[str, str]:
    """Attribute every entry to an account. Returns entry_id to source.

    Poison always comes from an attacker account. cover_rate is the share of
    benign entries that also come from one, which is what stops the account
    being a perfect label. At 0.35 an attacker account is roughly a third
    ordinary traffic, so trust degrades slowly and a gate that leans on it alone
    will refuse a lot of good edits.
    """
    if n_attacker_sources < 1 or n_attacker_sources > n_sources:
        raise ValueError(f"n_attacker_sources must be in [1, {n_sources}], "
                         f"got {n_attacker_sources}")
    if not 0.0 <= cover_rate <= 1.0:
        raise ValueError(f"cover_rate must be in [0, 1], got {cover_rate}")

    rng = random.Random(seed)
    names = [f"src-{i:02d}" for i in range(n_sources)]
    attacker = names[:n_attacker_sources]
    honest = names[n_attacker_sources:] or names

    out: dict[str, str] = {}
    for entry in feed:
        if entry.is_poison:
            out[entry.entry_id] = rng.choice(attacker)
        elif rng.random() < cover_rate:
            out[entry.entry_id] = rng.choice(attacker)
        else:
            out[entry.entry_id] = rng.choice(honest)
    return out


class TrustTracker:
    """Feeds outcomes into SourceTrust at the time the operator would learn them.

    An entry is not credited the moment it is applied. It sits pending until
    either a correction arrives, which counts against its account, or it survives
    `lag` positions unchallenged, which counts for it. Crediting at admission
    instead would let an attacker earn trust from the very entries being
    measured, and would make the gate look better than it is by exactly the
    amount of poison it let through.
    """

    def __init__(self, trust: SourceTrust, lag: int = 200) -> None:
        if lag < 1:
            raise ValueError(f"lag must be >= 1, got {lag}")
        self.trust = trust
        self.lag = lag
        self._pending: dict[str, tuple[str, int]] = {}

    def submitted(self, entry_id: str, source: str, position: int) -> None:
        self._pending[entry_id] = (source, position)

    def reverted(self, entry_id: str) -> bool:
        """Charge a revert to whoever submitted the entry being corrected."""
        found = self._pending.pop(entry_id, None)
        if found is None:
            return False
        self.trust.observe(found[0], reverted=True)
        return True

    def advance(self, position: int) -> int:
        """Credit everything that has stood unchallenged for long enough."""
        matured = [eid for eid, (_, pos) in self._pending.items()
                   if pos <= position - self.lag]
        for entry_id in matured:
            source, _ = self._pending.pop(entry_id)
            self.trust.observe(source, reverted=False)
        return len(matured)

    def pending(self) -> int:
        return len(self._pending)


def attacker_sources(n_sources: int = 8, n_attacker_sources: int = 2) -> list[str]:
    """The accounts simulate_sources treats as attacker controlled.

    For scoring the simulation, never for the gate. Passing this into a signal
    would hand it the answer.
    """
    return [f"src-{i:02d}" for i in range(n_attacker_sources)][:n_sources]
