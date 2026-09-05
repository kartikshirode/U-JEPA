"""Q1 spike: does knowledge split cleanly into invariant and volatile?

WHAT THIS MEASURES, AND WHAT IT DOES NOT. The headline number here is
`update_share`: per relation, the share of its rows in the diff stream that are
revisions rather than additions. That is the composition of observed change. It
is NOT volatility, and it is NOT the probability that a fact of that relation
changes.

The difference matters. A true volatility estimate needs statements at risk in
the denominator: of every Wikidata statement using property P, what share got
revised in the window. This script cannot compute that, because the corpus only
contains rows that already changed. A relation can therefore post a high
update_share simply by rarely gaining new subjects, while a genuinely churning
relation that is also growing fast posts a low one.

Getting the real number needs Wikidata property statement counts from the query
service, which is a prerequisite before any of this may set a threshold. Until
then, read update_share as a composition statistic and nothing more.

Method. WikiBigEdit gives 8 Wikidata snapshot diffs covering 2024-02-01 to
2024-07-01. Each row is a (subject, relation, object) triple that changed in
that window, tagged `new` when the entity gained the property and `update` when
an existing object was replaced. Those two are very different events and the
script keeps them apart throughout.

  update_share  per relation, the share of its rows that are updates rather
                than new facts. Composition of change, not rate of change.
  recurrence    per relation, the share of its updated (subject, relation)
                pairs that got updated in more than one timestep. This one is a
                genuine rate, but only over pairs already known to have changed.

Recurrence has high precision and low recall by construction: five months is
short, so a relation that genuinely turns over every four years shows zero
recurrence here. Read a high value as proof of volatility and a low value as
absence of evidence, not as proof of invariance.

Writes results.json next to this file. No GPU, no network past the dataset.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from load_wikibigedit import load_all

# Relations below this many rows give unstable ratios, so they are reported
# separately rather than mixed into the headline distribution.
MIN_SUPPORT = 200

OUT_PATH = Path(__file__).parent / "results.json"


def tag_breakdown(table: pd.DataFrame) -> dict:
    """How much of the edit stream is growth versus revision."""
    counts = table["tag"].value_counts().to_dict()
    total = int(len(table))
    return {
        "total_rows": total,
        "counts": {str(k): int(v) for k, v in counts.items()},
        "share": {str(k): round(int(v) / total, 4) for k, v in counts.items()},
    }


def recurrence_table(updates: pd.DataFrame) -> pd.DataFrame:
    """Per (subject, relation) pair, how many distinct timesteps updated it."""
    per_pair = (
        updates.groupby(["subject_id", "relation_id"])["timestep"]
        .nunique()
        .rename("n_timesteps")
        .reset_index()
    )
    return per_pair


def per_relation_stats(table: pd.DataFrame, updates: pd.DataFrame) -> pd.DataFrame:
    """Assemble update_share and recurrence for every relation."""
    all_rows = (
        table.groupby("relation_id")
        .agg(n_rows=("tag", "size"), relation=("relation", "first"))
        .reset_index()
    )

    n_updates = (
        updates.groupby("relation_id").size().rename("n_updates").reset_index()
    )

    pairs = recurrence_table(updates)
    rec = (
        pairs.assign(recurring=lambda d: d["n_timesteps"] > 1)
        .groupby("relation_id")
        .agg(
            n_updated_pairs=("n_timesteps", "size"),
            n_recurring_pairs=("recurring", "sum"),
            mean_updates_per_pair=("n_timesteps", "mean"),
            max_updates_per_pair=("n_timesteps", "max"),
        )
        .reset_index()
    )

    stats = all_rows.merge(n_updates, on="relation_id", how="left").merge(
        rec, on="relation_id", how="left"
    )
    for col in [
        "n_updates",
        "n_updated_pairs",
        "n_recurring_pairs",
        "max_updates_per_pair",
    ]:
        stats[col] = stats[col].fillna(0).astype(int)
    stats["mean_updates_per_pair"] = stats["mean_updates_per_pair"].fillna(0.0)

    stats["update_share"] = stats["n_updates"] / stats["n_rows"]
    stats["recurrence"] = np.where(
        stats["n_updated_pairs"] > 0,
        stats["n_recurring_pairs"] / stats["n_updated_pairs"].replace(0, np.nan),
        0.0,
    )
    stats["recurrence"] = stats["recurrence"].fillna(0.0)
    return stats.sort_values("n_rows", ascending=False).reset_index(drop=True)


def bimodality(values: np.ndarray) -> dict:
    """Is the update_share distribution two humps or one smear?

    Uses the sample bimodality coefficient, (skew^2 + 1) / kurtosis, with the
    usual 5/9 = 0.555 reference point. Above that leans bimodal. It is a coarse
    instrument, so the histogram is reported alongside it and the mass sitting
    at the extremes is what the decision should actually rest on.
    """
    values = values[~np.isnan(values)]
    n = len(values)
    if n < 4:
        return {"n": n, "coefficient": None, "note": "too few relations"}
    mean = values.mean()
    std = values.std(ddof=1)
    if std == 0:
        return {"n": n, "coefficient": None, "note": "zero variance"}
    z = (values - mean) / std
    skew = (n / ((n - 1) * (n - 2))) * np.sum(z**3)
    kurt = np.sum(z**4) / n
    coeff = (skew**2 + 1) / kurt
    hist, edges = np.histogram(values, bins=10, range=(0.0, 1.0))
    return {
        "n": int(n),
        "coefficient": round(float(coeff), 4),
        "threshold": 0.5556,
        "leans_bimodal": bool(coeff > 0.5556),
        "histogram_counts": [int(c) for c in hist],
        "histogram_edges": [round(float(e), 2) for e in edges],
        "share_below_0_1": round(float((values < 0.1).mean()), 4),
        "share_above_0_9": round(float((values > 0.9).mean()), 4),
    }


def main() -> None:
    table, report = load_all()
    updates = table[table["tag"] == "update"].copy()

    tags = tag_breakdown(table)
    stats = per_relation_stats(table, updates)
    supported = stats[stats["n_rows"] >= MIN_SUPPORT].copy()

    share_shape = bimodality(supported["update_share"].to_numpy())
    rec_shape = bimodality(supported["recurrence"].to_numpy())

    pairs = recurrence_table(updates)
    step_hist = pairs["n_timesteps"].value_counts().sort_index().to_dict()

    def top(frame: pd.DataFrame, col: str, n: int = 15, ascending: bool = False):
        cols = [
            "relation",
            "relation_id",
            "n_rows",
            "n_updates",
            "update_share",
            "recurrence",
            "mean_updates_per_pair",
        ]
        out = frame.sort_values(col, ascending=ascending).head(n)[cols]
        return [
            {
                "relation": r.relation,
                "relation_id": r.relation_id,
                "n_rows": int(r.n_rows),
                "n_updates": int(r.n_updates),
                "update_share": round(float(r.update_share), 4),
                "recurrence": round(float(r.recurrence), 4),
                "mean_updates_per_pair": round(float(r.mean_updates_per_pair), 3),
            }
            for r in out.itertuples()
        ]

    results = {
        "load": {
            "n_files": report.n_files,
            "n_rows": report.n_rows,
            "rows_per_timestep": {str(k): int(v) for k, v in report.rows_per_timestep.items()},
            "n_unique_relations": report.n_unique_relations,
            "n_unique_sr_pairs": report.n_unique_sr_pairs,
            "window": "2024-02-01 to 2024-07-01",
        },
        "tags": tags,
        "updates": {
            "n_update_rows": int(len(updates)),
            "n_updated_pairs": int(len(pairs)),
            "pairs_by_n_timesteps": {str(k): int(v) for k, v in step_hist.items()},
            "share_pairs_updated_more_than_once": round(
                float((pairs["n_timesteps"] > 1).mean()), 4
            ),
        },
        "relations": {
            "n_total": int(len(stats)),
            "n_with_support": int(len(supported)),
            "min_support": MIN_SUPPORT,
        },
        "update_share_distribution": share_shape,
        "recurrence_distribution": rec_shape,
        "highest_update_share": top(supported, "update_share"),
        "lowest_update_share": top(supported, "update_share", ascending=True),
        "most_recurring": top(supported, "recurrence"),
    }

    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    stats.to_csv(OUT_PATH.parent / "per_relation.csv", index=False)

    print(f"rows {report.n_rows:,}  relations {report.n_unique_relations:,}")
    print(f"tags {tags['share']}")
    print(
        f"update pairs {len(pairs):,}  "
        f"share updated >1x {results['updates']['share_pairs_updated_more_than_once']}"
    )
    print(f"pairs by n_timesteps {step_hist}")
    print(f"relations with >={MIN_SUPPORT} rows: {len(supported)}")
    print(f"update_share      {share_shape}")
    print(f"recurrence {rec_shape}")
    print("\nhighest update share:")
    for r in results["highest_update_share"][:10]:
        print(f"  {r['relation'][:34]:36} {r['relation_id']:>7} "
              f"n={r['n_rows']:>6} update_share={r['update_share']:.3f} rec={r['recurrence']:.3f}")
    print("\nlowest update share:")
    for r in results["lowest_update_share"][:10]:
        print(f"  {r['relation'][:34]:36} {r['relation_id']:>7} "
              f"n={r['n_rows']:>6} update_share={r['update_share']:.3f} rec={r['recurrence']:.3f}")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
