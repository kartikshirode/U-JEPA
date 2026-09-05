"""The maintenance feed a model operator would actually consume.

An operator ingests a public knowledge feed and applies its entries as edits, at
a volume that rules out per-entry review. A fraction of entries are poisoned. The
upstream source notices some distance later and publishes a correction, which
arrives as an ordinary entry carrying the true value.

That correction is what RQ1 is about. If applying it removes the poison, the
pipeline self-heals and a gate matters much less. If the poison stays elicitable
after the correction has been applied in good faith, then admission is the only
place to stop it, because retraction does not work.
"""
from __future__ import annotations

import random

from ..schema import EditCandidate, FeedEntry


def build_feed(
    benign: list[EditCandidate],
    poison_pairs: list[tuple[EditCandidate, EditCandidate]],
    base_rate: float,
    revert_lag: int,
    seed: int,
) -> list[FeedEntry]:
    """Interleave poison into a benign stream, each followed by its correction.

    base_rate is the share of poison_pairs actually injected, so the caller can
    sweep prevalence without regenerating attacks.

    revert_lag counts BENIGN entries between a poison entry and its correction.
    The resulting gap in feed positions is revert_lag + 1, because the poison
    entry occupies a position of its own.
    """
    if revert_lag < 1:
        raise ValueError(f"revert_lag must be >= 1, got {revert_lag}")
    if not 0.0 <= base_rate <= 1.0:
        raise ValueError(f"base_rate must be in [0, 1], got {base_rate}")

    rng = random.Random(seed)
    n_inject = round(len(poison_pairs) * base_rate)
    injected = rng.sample(poison_pairs, n_inject) if n_inject else []

    # Slots where poison enters, spaced so every correction fits before the end.
    usable = max(len(benign) - revert_lag, 1)
    slots = sorted(rng.sample(range(usable), min(len(injected), usable)))

    # benign index -> entries to emit just before that benign row
    pending: dict[int, list[tuple]] = {}
    for (original, bad), slot in zip(injected, slots):
        poison_id = f"poison-{slot}"
        pending.setdefault(slot, []).append((bad, True, None, bad.source, poison_id))
        pending.setdefault(slot + revert_lag, []).append(
            (original, False, poison_id, None, f"revert-{slot}")
        )

    feed: list[FeedEntry] = []
    position = 0
    for index, candidate in enumerate(benign):
        for cand, is_poison, reverts, family, entry_id in pending.get(index, []):
            feed.append(FeedEntry(candidate=cand, position=position, entry_id=entry_id,
                                  is_poison=is_poison, reverts=reverts,
                                  attack_family=family))
            position += 1
        feed.append(FeedEntry(candidate=candidate, position=position,
                              entry_id=f"benign-{index}", is_poison=False,
                              reverts=None, attack_family=None))
        position += 1
    return feed


def poison_entries(feed: list[FeedEntry]) -> list[FeedEntry]:
    return [e for e in feed if e.is_poison]


def reverted_by(feed: list[FeedEntry]) -> dict[str, FeedEntry]:
    """poison entry_id -> the entry that corrects it."""
    return {e.reverts: e for e in feed if e.reverts}


def poison_state(
    feed: list[FeedEntry], upto: int
) -> tuple[list[FeedEntry], list[FeedEntry]]:
    """Split poison into (not yet corrected, already corrected) as of position upto.

    Only entries the pipeline has actually reached count, which is what lets the
    driver ask both questions at every checkpoint: does uncorrected poison stick,
    and does corrected poison go away.
    """
    corrections = reverted_by(feed)
    uncorrected, corrected = [], []
    for entry in poison_entries(feed):
        if entry.position >= upto:
            continue
        revert = corrections.get(entry.entry_id)
        if revert and revert.position < upto:
            corrected.append(entry)
        else:
            uncorrected.append(entry)
    return uncorrected, corrected
