"""What removing a bad edit actually costs, once you accept that correcting it does not work.

An operator who discovers poison has two remedies. Write the true value on top,
which is cheap and is what RQ1 measures. Or rebuild: take the untouched base
model and replay every admitted edit except the bad ones. The second is exact by
construction, since the poison was never applied to the model that results. It
costs a full replay of the ledger.

So the interesting number is not whether rebuilding works. It is what it costs,
and how that cost grows with how late the poison was found. A gate that stops
one entry in ten thousand is worth very little if a miss is cheap to undo, and
worth a great deal if undoing a miss means replaying 200,000 edits.

The ledger holds admitted edits only. A refused entry was never applied, so
replaying it would be a bug rather than fidelity.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..editors.base import Editor
from ..probes.efficacy import efficacy
from ..probes.elicitation import elicitation_rate
from ..schema import EditCandidate

DEFAULT_BATCH = 32


@dataclass(frozen=True)
class LedgerEntry:
    entry_id: str
    position: int
    candidate: EditCandidate


@dataclass
class ShadowLedger:
    """Ordered record of what was applied, which is what a replay needs.

    Order is preserved because sequential editing is not commutative: two edits
    to the same slot leave a different model depending on which went last, and
    lifelong normalization makes even unrelated edits order dependent.
    """

    entries: list[LedgerEntry] = field(default_factory=list)

    def record(self, entry_id: str, position: int, candidate: EditCandidate) -> None:
        self.entries.append(LedgerEntry(entry_id, position, candidate))

    def __len__(self) -> int:
        return len(self.entries)

    def replay_plan(self, drop_ids: set[str]) -> list[EditCandidate]:
        """Everything admitted except the named entries, in the original order."""
        unknown = drop_ids - {e.entry_id for e in self.entries}
        if unknown:
            raise KeyError(
                f"asked to drop entries that were never admitted: {sorted(unknown)}. "
                "A refused entry has nothing to roll back."
            )
        return [e.candidate for e in self.entries if e.entry_id not in drop_ids]

    def position_of(self, entry_id: str) -> int:
        for entry in self.entries:
            if entry.entry_id == entry_id:
                return entry.position
        raise KeyError(f"{entry_id!r} is not in the ledger")

    def cost_of_dropping(self, drop_ids: set[str]) -> int:
        """Edits that have to be replayed, being everything the drop does not remove.

        The whole ledger, less the dropped entries. Not "everything after the
        earliest drop": replay starts from the base model, so edits before the
        poison are re-applied too. Anything cheaper needs a checkpoint of the
        weights at that point, which is a design choice with its own storage
        bill rather than a free saving.
        """
        return len(self.replay_plan(drop_ids))


@dataclass(frozen=True)
class RollbackAudit:
    n_ledger: int
    n_dropped: int
    n_replayed: int
    seconds: float
    residual_direct: float
    residual_leading: float
    benign_efficacy: float

    @property
    def seconds_per_edit(self) -> float:
        return self.seconds / self.n_replayed if self.n_replayed else 0.0

    def as_dict(self) -> dict:
        return {"n_ledger": self.n_ledger, "n_dropped": self.n_dropped,
                "n_replayed": self.n_replayed, "seconds": round(self.seconds, 2),
                "seconds_per_edit": round(self.seconds_per_edit, 4),
                "residual_direct": self.residual_direct,
                "residual_leading": self.residual_leading,
                "benign_efficacy": self.benign_efficacy}


def audit_rollback(
    editor_factory,
    ledger: ShadowLedger,
    drop_ids: set[str],
    poisoned: list[EditCandidate],
    benign_sample: list[EditCandidate],
    batch_size: int = DEFAULT_BATCH,
) -> RollbackAudit:
    """Replay the ledger without the dropped entries onto a fresh model, then probe.

    editor_factory must return an editor over an untouched base model. Handing in
    the editor that already carries the poison measures nothing, which is the
    same trap the responder redesign closed, so the factory is required rather
    than an editor.

    residual_direct and residual_leading should both be zero. They are measured
    anyway: a non-zero value means the replay is not reproducing a clean model,
    and the run is telling you the base was not clean or the editor is not
    deterministic.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    plan = ledger.replay_plan(drop_ids)
    editor: Editor = editor_factory()

    started = time.perf_counter()
    for start in range(0, len(plan), batch_size):
        editor.apply(plan[start : start + batch_size])
    seconds = time.perf_counter() - started

    responder = editor.responder()
    return RollbackAudit(
        n_ledger=len(ledger),
        n_dropped=len(drop_ids),
        n_replayed=len(plan),
        seconds=seconds,
        residual_direct=elicitation_rate(responder, poisoned, mode="direct"),
        residual_leading=elicitation_rate(responder, poisoned, mode="leading"),
        benign_efficacy=efficacy(responder, benign_sample),
    )
