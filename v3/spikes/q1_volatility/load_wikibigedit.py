"""Download and flatten the WikiBigEdit timestep diffs into one table.

WikiBigEdit ships as 8 JSON files, each holding the facts that changed between
two Wikidata snapshots between 2024-02-01 and 2024-07-01. Every row is a
(subject_id, relation_id, object_id) triple plus question phrasings.

We only need the triple, the tag, and which timestep the row came from. The
timestep index is what makes volatility measurable: a (subject, relation) pair
that shows up in several timesteps changed its object several times.

Cached under ~/.cache/huggingface after the first run, so reruns are offline.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pandas as pd

REPO_ID = "lukasthede/WikiBigEdit"

# Ordered oldest to newest. The order is the timestep index, so do not sort
# these alphabetically by accident; the names happen to sort correctly but
# relying on that is fragile.
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

KEEP_COLUMNS = [
    "tag",
    "subject",
    "subject_id",
    "relation",
    "relation_id",
    "object",
    "object_id",
]


@dataclass(frozen=True)
class LoadReport:
    """What actually got loaded, for the record in the results JSON."""

    n_files: int
    n_rows: int
    rows_per_timestep: dict[int, int]
    n_unique_relations: int
    n_unique_sr_pairs: int


def _download(filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(REPO_ID, filename, repo_type="dataset")


def load_all() -> tuple[pd.DataFrame, LoadReport]:
    """Return one dataframe of every timestep, with a `timestep` column added.

    timestep is 0-based and follows TIMESTEP_FILES order, so timestep 0 is the
    2024-02-01 to 2024-02-20 window.
    """
    frames = []
    for step, filename in enumerate(TIMESTEP_FILES):
        path = _download(filename)
        with open(path, "r", encoding="utf-8") as handle:
            rows = json.load(handle)
        frame = pd.DataFrame(rows)
        missing = [c for c in KEEP_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"{filename} is missing columns {missing}")
        frame = frame[KEEP_COLUMNS].copy()
        frame["timestep"] = step
        frame["window"] = filename.removeprefix("wiki_big_edit_").removesuffix(".json")
        frames.append(frame)

    table = pd.concat(frames, ignore_index=True)

    # A handful of rows carry a null relation_id or subject_id. They cannot be
    # keyed, so drop them rather than letting them form a bogus "nan" group.
    before = len(table)
    table = table.dropna(subset=["subject_id", "relation_id"])
    dropped = before - len(table)
    if dropped:
        print(f"  dropped {dropped} rows with a null subject_id or relation_id")

    report = LoadReport(
        n_files=len(TIMESTEP_FILES),
        n_rows=len(table),
        rows_per_timestep=table["timestep"].value_counts().sort_index().to_dict(),
        n_unique_relations=table["relation_id"].nunique(),
        n_unique_sr_pairs=len(table.drop_duplicates(["subject_id", "relation_id"])),
    )
    return table, report


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    df, rep = load_all()
    print(f"rows                {rep.n_rows:,}")
    print(f"unique relations    {rep.n_unique_relations:,}")
    print(f"unique (subj,rel)   {rep.n_unique_sr_pairs:,}")
    print(f"rows per timestep   {rep.rows_per_timestep}")
    print(f"tag counts          {df['tag'].value_counts().to_dict()}")
