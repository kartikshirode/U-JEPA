"""Q1 follow-up: is churn a property of the relation, or a bot pass?

analyze_volatility.py found that relations differ enormously in how often their
facts get revised. That is only useful to the architecture if the difference is
a stable trait you could predict in advance. It is useless if it is one curation
sweep landing in one snapshot.

Wikidata revisions mix two things the design cares about separately: the world
changing (a boxer remarries) and the database being tidied (a bot reclassifies
17,000 astronomical objects). Both look identical in a diff. Time is what tells
them apart, because real-world change trickles and curation arrives in a lump.

Two measures:

  concentration  the largest share of a relation's updates falling in any one
                 timestep. Near 1.0 means a lump, so probably curation.
  split-half     Spearman correlation between churn measured on timesteps 0-3
                 and churn measured on timesteps 4-7. High means churn is a
                 stable trait of the relation and can be predicted; low means it
                 is noise and layer assignment cannot lean on it.

Split-half is the number the architecture actually hangs on. Writes
stability.json next to this file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from load_wikibigedit import load_all

MIN_SUPPORT = 200
MIN_UPDATES_PER_HALF = 20
EARLY = (0, 1, 2, 3)
LATE = (4, 5, 6, 7)

OUT_PATH = Path(__file__).parent / "stability.json"


def concentration(updates: pd.DataFrame) -> pd.DataFrame:
    """Largest share of a relation's updates sitting in a single timestep."""
    per = (
        updates.groupby(["relation_id", "timestep"])
        .size()
        .rename("n")
        .reset_index()
    )
    totals = per.groupby("relation_id")["n"].sum().rename("n_updates")
    peak = per.groupby("relation_id")["n"].max().rename("peak")
    steps = per.groupby("relation_id")["timestep"].nunique().rename("n_timesteps_seen")
    out = pd.concat([totals, peak, steps], axis=1).reset_index()
    out["concentration"] = out["peak"] / out["n_updates"]
    return out


def split_half(table: pd.DataFrame) -> dict:
    """Does churn measured early predict churn measured late?"""
    half_stats = {}
    for name, steps in (("early", EARLY), ("late", LATE)):
        part = table[table["timestep"].isin(steps)]
        grouped = part.groupby("relation_id").agg(
            n_rows=("tag", "size"),
            n_updates=("tag", lambda s: int((s == "update").sum())),
        )
        grouped["churn"] = grouped["n_updates"] / grouped["n_rows"]
        half_stats[name] = grouped

    joined = half_stats["early"].join(
        half_stats["late"], lsuffix="_early", rsuffix="_late", how="inner"
    )
    # Require enough rows on both sides or the ratio is noise pretending to be signal.
    joined = joined[
        (joined["n_rows_early"] >= MIN_UPDATES_PER_HALF)
        & (joined["n_rows_late"] >= MIN_UPDATES_PER_HALF)
    ]
    if len(joined) < 10:
        return {"n": int(len(joined)), "note": "too few relations to correlate"}

    rho, p = sps.spearmanr(joined["churn_early"], joined["churn_late"])
    pear, p_pear = sps.pearsonr(joined["churn_early"], joined["churn_late"])
    return {
        "n_relations": int(len(joined)),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": float(f"{p:.3e}"),
        "pearson_r": round(float(pear), 4),
        "pearson_p": float(f"{p_pear:.3e}"),
        "min_rows_per_half": MIN_UPDATES_PER_HALF,
    }


def main() -> None:
    table, _ = load_all()
    updates = table[table["tag"] == "update"].copy()

    conc = concentration(updates)
    supported = conc[conc["n_updates"] >= 50].copy()

    names = table.drop_duplicates("relation_id").set_index("relation_id")["relation"]
    supported["relation"] = supported["relation_id"].map(names)

    sh = split_half(table)

    # How much of the whole update stream comes from lumpy relations.
    lumpy = supported[supported["concentration"] >= 0.8]
    spread = supported[supported["concentration"] < 0.5]

    results = {
        "split_half_churn": sh,
        "concentration": {
            "n_relations_scored": int(len(supported)),
            "min_updates": 50,
            "median": round(float(supported["concentration"].median()), 4),
            "share_lumpy_ge_0_8": round(float((supported["concentration"] >= 0.8).mean()), 4),
            "share_spread_lt_0_5": round(float((supported["concentration"] < 0.5).mean()), 4),
            "updates_in_lumpy_relations": int(lumpy["n_updates"].sum()),
            "updates_in_spread_relations": int(spread["n_updates"].sum()),
            "total_updates_scored": int(supported["n_updates"].sum()),
        },
        "lumpiest": [
            {
                "relation": r.relation,
                "relation_id": r.relation_id,
                "n_updates": int(r.n_updates),
                "concentration": round(float(r.concentration), 3),
                "n_timesteps_seen": int(r.n_timesteps_seen),
            }
            for r in supported.nlargest(12, "n_updates")
            .sort_values("concentration", ascending=False)
            .itertuples()
        ],
        "most_spread": [
            {
                "relation": r.relation,
                "relation_id": r.relation_id,
                "n_updates": int(r.n_updates),
                "concentration": round(float(r.concentration), 3),
                "n_timesteps_seen": int(r.n_timesteps_seen),
            }
            for r in supported[supported["n_updates"] >= 100]
            .nsmallest(12, "concentration")
            .itertuples()
        ],
    }

    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("split-half churn correlation:")
    for k, v in sh.items():
        print(f"  {k:22} {v}")
    print("\nconcentration:")
    for k, v in results["concentration"].items():
        print(f"  {k:32} {v}")
    print("\nbiggest relations, by how lumpy their updates are:")
    for r in results["lumpiest"]:
        print(f"  {str(r['relation'])[:34]:36} {r['relation_id']:>7} "
              f"n={r['n_updates']:>6} conc={r['concentration']:.3f} "
              f"steps={r['n_timesteps_seen']}/8")
    print("\nmost evenly spread (real-world churn candidates):")
    for r in results["most_spread"]:
        print(f"  {str(r['relation'])[:34]:36} {r['relation_id']:>7} "
              f"n={r['n_updates']:>6} conc={r['concentration']:.3f} "
              f"steps={r['n_timesteps_seen']}/8")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
