"""Poisoned entries, in 3 families that differ by mechanism rather than label.

Every generator returns (original, poisoned) pairs. Matching is the point: both
arms then share a subject, a relation and an edit kind, so a benign-versus-
poisoned comparison is not quietly measuring dataset difficulty.

Real labelled Wikidata vandalism is the preferred source once the corpus join
lands. These generators are what makes stage 1 runnable before it does, and the
held-out-family test in stage 2 needs several controlled families regardless.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from enum import Enum
from pathlib import Path

from ..schema import EditCandidate, EditKind

RISK_CATEGORIES = ("misinformation", "bias", "safety")


class AttackFamily(str, Enum):
    OBJECT_SWAP = "object_swap"           # object from a different relation, type-violating
    TYPE_CONSISTENT = "type_consistent"   # object from the same relation, plausible
    TEMPORAL_STALE = "temporal_stale"     # a value this slot genuinely held earlier


def _poisoned(original: EditCandidate, new_object: str, family: AttackFamily,
              category: str) -> EditCandidate:
    return EditCandidate(
        subject_id=original.subject_id, subject=original.subject,
        relation_id=original.relation_id, relation=original.relation,
        object_id=None, object=new_object, prompt=original.prompt,
        kind=original.kind, source=family.value, timestep=original.timestep,
        is_adversarial=True, risk_category=category, n_hops=original.n_hops,
    )


def _by_relation(benign: list[EditCandidate]) -> dict[str, list[str]]:
    vocab: dict[str, set[str]] = defaultdict(set)
    for c in benign:
        vocab[c.relation_id].add(c.object)
    return {k: sorted(v) for k, v in vocab.items()}


def poison_object_swap(
    benign: list[EditCandidate], seed: int, n: int
) -> list[tuple[EditCandidate, EditCandidate]]:
    """Crude attack: object taken from a different relation, so usually the wrong type."""
    rng = random.Random(seed)
    vocab = _by_relation(benign)
    if len(vocab) < 2:
        raise ValueError("object swap needs at least 2 relations to cross between")

    out = []
    for original in rng.sample(benign, min(n, len(benign))):
        others = [r for r in vocab if r != original.relation_id]
        pool = [o for o in vocab[rng.choice(others)] if o != original.object]
        if not pool:
            # Vocabularies overlap on real data. Skip rather than substitute a
            # same-relation object, which would silently be a different family.
            continue
        out.append((original, _poisoned(original, rng.choice(pool),
                                        AttackFamily.OBJECT_SWAP,
                                        rng.choice(RISK_CATEGORIES))))
    return out


def poison_type_consistent(
    benign: list[EditCandidate], seed: int, n: int
) -> list[tuple[EditCandidate, EditCandidate]]:
    """Plausible attack: object taken from the same relation, so the type is right."""
    rng = random.Random(seed)
    vocab = _by_relation(benign)

    out = []
    for original in rng.sample(benign, min(n, len(benign))):
        pool = [o for o in vocab[original.relation_id] if o != original.object]
        if not pool:
            continue
        out.append((original, _poisoned(original, rng.choice(pool),
                                        AttackFamily.TYPE_CONSISTENT,
                                        rng.choice(RISK_CATEGORIES))))
    return out


def build_history(candidates: list[EditCandidate]) -> dict[str, list[EditCandidate]]:
    """Group candidates by fact slot, ordered by timestep."""
    hist: dict[str, list[EditCandidate]] = defaultdict(list)
    for c in candidates:
        hist[c.key].append(c)
    return {k: sorted(v, key=lambda c: c.timestep) for k, v in hist.items()}


def poison_temporal_stale(
    history: dict[str, list[EditCandidate]], seed: int, n: int
) -> list[tuple[EditCandidate, EditCandidate]]:
    """Hardest attack: a value the slot really held before, so it is true but outdated.

    No fact checker can call this false, only stale, which is exactly why it is
    worth a family of its own. It needs a slot observed changing at least twice;
    Q1 found 913 of those in 99,404 updated pairs, so ask for few and expect the
    raise rather than a silent substitution.
    """
    eligible = [v for v in history.values() if len(v) >= 2]
    if not eligible:
        raise ValueError("temporal stale needs slots that changed at least twice")

    rng = random.Random(seed)
    out = []
    for chain in rng.sample(eligible, min(n, len(eligible))):
        current, earlier = chain[-1], chain[-2]
        out.append((current, _poisoned(current, earlier.object,
                                       AttackFamily.TEMPORAL_STALE,
                                       "misinformation")))
    return out


def load_editrisk(path: str | Path) -> list[EditCandidate]:
    """Ingest EditRisk-Bench from a local JSON file, for the downstream-harm probe.

    Raises when absent so callers fall back to the generators deliberately
    rather than silently.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"EditRisk-Bench not found at {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))

    out = []
    for row in rows:
        category = row.get("risk_category")
        if category not in RISK_CATEGORIES:
            raise ValueError(f"unknown risk_category {category!r} in {path}")
        out.append(
            EditCandidate(
                subject_id=str(row["subject_id"]), subject=str(row.get("subject") or ""),
                relation_id=str(row["relation_id"]), relation=str(row.get("relation") or ""),
                object_id=None, object=str(row["object"]), prompt=str(row["prompt"]),
                kind=EditKind.REVISION,
                source="editrisk", timestep=0, is_adversarial=True,
                risk_category=category, n_hops=int(row.get("n_hops", 1)),
            )
        )
    return out
