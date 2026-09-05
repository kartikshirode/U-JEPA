"""Per-relation statistics offered to the gate as candidate features.

WHAT update_share IS. Per relation, the share of its rows in the diff stream
that are revisions rather than additions. Composition of observed change.

WHAT IT IS NOT. Volatility, or the probability that a fact of this relation
changes. The denominator holds only rows that already changed, so a relation
posts a high value simply by rarely gaining new subjects. Getting the real
number needs revisions over statements at risk, which needs Wikidata property
statement counts. See v3/spikes/q1_volatility/FINDINGS.md.

Q1 did show the share is stable enough to predict from a relation's own past,
split-half Spearman 0.695. That makes it a reasonable feature to offer a
classifier. Whether it helps a decision is RQ3, and the answer may be no.

Concentration rides along because a relation whose updates all land in one
timestep is worth a human look. It does not classify anything: elections and
transfer windows are lumpy real change, and scheduled bot passes can spread
evenly.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from ..schema import EditCandidate, EditKind

DEFAULT_THRESHOLD = 0.1
DEFAULT_MIN_SUPPORT = 200


@dataclass(frozen=True)
class RelationStats:
    relation_id: str
    n_rows: int
    n_updates: int
    update_share: float
    concentration: float


class RelationPrior:
    """Maps a relation to the composition of its observed change."""

    def __init__(self, stats: dict[str, RelationStats], n_rows_total: int) -> None:
        self._stats = stats
        self._n_rows_total = n_rows_total

    @classmethod
    def from_candidates(
        cls, candidates: list[EditCandidate], min_support: int = DEFAULT_MIN_SUPPORT
    ) -> "RelationPrior":
        rows: Counter[str] = Counter()
        updates: Counter[str] = Counter()
        per_step: dict[str, Counter[int]] = defaultdict(Counter)
        for c in candidates:
            rows[c.relation_id] += 1
            if c.kind is EditKind.REVISION:
                updates[c.relation_id] += 1
                per_step[c.relation_id][c.timestep] += 1

        stats: dict[str, RelationStats] = {}
        for relation_id, n in rows.items():
            if n < min_support:
                continue
            n_up = updates[relation_id]
            steps = per_step[relation_id]
            stats[relation_id] = RelationStats(
                relation_id=relation_id,
                n_rows=n,
                n_updates=n_up,
                update_share=n_up / n,
                concentration=(max(steps.values()) / n_up) if n_up else 0.0,
            )
        return cls(stats, sum(rows.values()))

    def __contains__(self, relation_id: str) -> bool:
        return relation_id in self._stats

    def stats(self, relation_id: str) -> RelationStats:
        if relation_id not in self._stats:
            raise KeyError(f"relation {relation_id} has no prior")
        return self._stats[relation_id]

    def update_share(self, relation_id: str) -> float:
        return self.stats(relation_id).update_share

    def is_low(self, relation_id: str, threshold: float = DEFAULT_THRESHOLD) -> bool:
        return self.update_share(relation_id) < threshold

    def coverage(self) -> float:
        """Share of all rows in a relation that cleared min_support."""
        if not self._n_rows_total:
            return 0.0
        return sum(s.n_rows for s in self._stats.values()) / self._n_rows_total
