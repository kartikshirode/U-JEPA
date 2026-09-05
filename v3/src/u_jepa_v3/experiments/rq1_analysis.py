"""Turn RQ1 cell states into the numbers the paper reports.

Three questions. Does poison survive the correction that was supposed to remove
it. Does the model look untouched while it does. And how both change as edits
accumulate.

Every dimension of the experiment stays a grouping key. An earlier version
grouped only by editor and corpus, which collapsed model, edit count and attack
family and made the promised 1K/10K/100K curves unobtainable from its own
output. Seeds are the only thing that collapses, into mean and sample standard
deviation.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass

GROUP_KEYS = ("model", "editor", "attack_family", "base_rate", "at")


@dataclass(frozen=True)
class ArmSummary:
    model: str
    editor: str
    attack_family: str
    base_rate: float
    at: int
    benign_efficacy_mean: float
    benign_efficacy_sd: float
    poison_uncorrected_mean: float
    poison_uncorrected_sd: float
    corrected_direct_mean: float
    corrected_direct_sd: float
    corrected_leading_mean: float
    corrected_leading_sd: float
    downstream_poisoned_mean: float
    downstream_poisoned_sd: float
    general_delta_mean: float
    general_delta_sd: float
    n_seeds: int


def _sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), _sd(values)


def summarize(states: list[dict]) -> list[ArmSummary]:
    """Collapse per-seed cells into one summary per (model, editor, family, rate, at).

    Cells with no checkpoints are dropped. A cell that died before its first
    probe has nothing to report, and averaging it in as a zero would understate
    every arm it touched.
    """
    grouped: dict[tuple, list[tuple[dict, dict]]] = defaultdict(list)
    for st in states:
        if "meta" not in st:
            raise KeyError(f"cell {st.get('cell_id')!r} has no meta block")
        meta = st["meta"]
        family = meta.get("params", {}).get("attack_family", "unknown")
        for point in st.get("checkpoints", []):
            key = (meta["model"], meta["editor"], family, meta["base_rate"], point["at"])
            grouped[key].append((point, meta))

    out: list[ArmSummary] = []
    for (model, editor, family, base_rate, at), rows in sorted(grouped.items()):
        points = [p for p, _ in rows]
        benign_m, benign_s = _mean_sd([p["benign_efficacy"] for p in points])
        unc_m, unc_s = _mean_sd([p["poison_uncorrected"] for p in points])
        dir_m, dir_s = _mean_sd([p["poison_corrected_direct"] for p in points])
        lead_m, lead_s = _mean_sd([p["poison_corrected_leading"] for p in points])
        down_m, down_s = _mean_sd([p["downstream_poisoned"] for p in points])
        delta_m, delta_s = _mean_sd(
            [p["general_mean"] - m["baseline_general"] for p, m in rows]
        )
        out.append(
            ArmSummary(
                model=model, editor=editor, attack_family=family,
                base_rate=base_rate, at=at,
                benign_efficacy_mean=benign_m, benign_efficacy_sd=benign_s,
                poison_uncorrected_mean=unc_m, poison_uncorrected_sd=unc_s,
                corrected_direct_mean=dir_m, corrected_direct_sd=dir_s,
                corrected_leading_mean=lead_m, corrected_leading_sd=lead_s,
                downstream_poisoned_mean=down_m, downstream_poisoned_sd=down_s,
                general_delta_mean=delta_m, general_delta_sd=delta_s,
                n_seeds=len({p["seed"] for p in points}),
            )
        )
    return out


def survival_gap(summary: ArmSummary) -> float:
    """Leading-context elicitation minus direct, on poison the feed already corrected.

    Near zero means the correction worked. Large means the revert only changed
    the surface answer and the model still holds the false fact, which is the
    result that makes pre-commit admission the only place left to intervene.
    """
    return summary.corrected_leading_mean - summary.corrected_direct_mean


def is_stealthy(summary: ArmSummary, tolerance: float = 0.02) -> bool:
    """True when general ability barely moved, so the corruption is invisible."""
    return abs(summary.general_delta_mean) <= tolerance


def curve(
    summaries: list[ArmSummary], model: str, editor: str, family: str
) -> list[ArmSummary]:
    """One arm's points ordered by edit count, which is the plotted curve."""
    picked = [
        s for s in summaries
        if s.model == model and s.editor == editor and s.attack_family == family
    ]
    return sorted(picked, key=lambda s: s.at)
