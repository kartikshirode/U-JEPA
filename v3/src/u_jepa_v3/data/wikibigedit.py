"""Benign edit corpus from 8 Wikidata snapshot diffs (2024-02-01 to 2024-07-01).

The `tag` column carries `new` or `update`, which maps onto EditKind. Rows with
a blank tag (about 1.4%) are dropped because we cannot say which they are, and
rows with a null subject_id or relation_id are dropped because they cannot be
keyed.
"""
from __future__ import annotations

import json
import random

import pandas as pd

from ..schema import EditCandidate, EditKind

REPO_ID = "lukasthede/WikiBigEdit"

# Order defines the timestep index. Do not sort.
TIMESTEP_FILES = [
    "wiki_big_edit_20240201_20240220.json",
    "wiki_big_edit_20240220_20240301.json",
    "wiki_big_edit_20240301_20240320.json",
    "wiki_big_edit_20240320_20240401.json",
    "wiki_big_edit_20240401_20240501.json",
    "wiki_big_edit_20240501_20240601.json",
    "wiki_big_edit_20240601_20240620.json",
    "wiki_big_edit_20240620_20240701.json",
]

TAG_TO_KIND = {"new": EditKind.ACCRETION, "update": EditKind.REVISION}


def load_raw() -> pd.DataFrame:
    """Download every timestep and concatenate, adding a `timestep` column."""
    from huggingface_hub import hf_hub_download

    frames = []
    for step, name in enumerate(TIMESTEP_FILES):
        path = hf_hub_download(REPO_ID, name, repo_type="dataset")
        with open(path, "r", encoding="utf-8") as handle:
            rows = json.load(handle)
        frame = pd.DataFrame(rows)
        frame["timestep"] = step
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _prompt_for(row) -> str:
    rephrase = row.get("rephrase")
    if isinstance(rephrase, str) and rephrase.strip():
        return rephrase
    return f"What is the {row['relation']} of {row['subject']}?"


def to_candidates(frame: pd.DataFrame) -> list[EditCandidate]:
    """Normalise raw rows into EditCandidate, dropping unusable ones."""
    out: list[EditCandidate] = []
    for row in frame.to_dict("records"):
        kind = TAG_TO_KIND.get(row.get("tag"))
        if kind is None:
            continue
        if not row.get("subject_id") or not row.get("relation_id"):
            continue
        if pd.isna(row.get("subject_id")) or pd.isna(row.get("relation_id")):
            continue
        out.append(
            EditCandidate(
                subject_id=str(row["subject_id"]),
                subject=str(row.get("subject") or ""),
                relation_id=str(row["relation_id"]),
                relation=str(row.get("relation") or ""),
                object_id=(str(row["object_id"]) if row.get("object_id") else None),
                object=str(row["object"]),
                prompt=_prompt_for(row),
                kind=kind,
                source="wikibigedit",
                timestep=int(row["timestep"]),
                is_adversarial=False,
                risk_category=None,
                n_hops=1,
            )
        )
    out.sort(key=lambda c: (c.timestep, c.key))
    return out


def sample_candidates(
    candidates: list[EditCandidate], n: int, seed: int
) -> list[EditCandidate]:
    """Seeded uniform sample, preserving timestep order in the result.

    A sorted prefix would bias toward low-numbered Q-ids and would make every
    seed identical, which turns "5 seeds" into one run reported five times.
    """
    if n >= len(candidates):
        return list(candidates)
    picked = random.Random(seed).sample(candidates, n)
    picked.sort(key=lambda c: (c.timestep, c.key))
    return picked


def load_candidates(n: int | None = None, seed: int = 0) -> list[EditCandidate]:
    candidates = to_candidates(load_raw())
    return sample_candidates(candidates, n, seed) if n else candidates
