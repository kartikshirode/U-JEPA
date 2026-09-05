"""Turn gated cell states into the two numbers a gate has to justify itself with.

How much poison did it keep out, and how many good edits did it destroy doing
it. Reporting the first without the second is how a gate that refuses everything
posts perfect recall.

The transfer question is separate and is the one that decides whether any of it
generalises: a summary whose calibrated_on differs from its attack_family was
tuned on attacks it was not then tested on. Same-family rows are the ceiling,
not the result.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass

from .rq1_analysis import ArmSummary

GROUP_KEYS = ("model", "editor", "attack_family", "calibrated_on", "at")


@dataclass(frozen=True)
class GateSummary:
    model: str
    editor: str
    attack_family: str
    calibrated_on: str
    at: int
    sensitivity_mean: float
    sensitivity_sd: float
    benign_blocked_mean: float
    benign_blocked_sd: float
    precision_at_deployment_mean: float
    precision_at_deployment_sd: float
    corrected_leading_mean: float
    corrected_leading_sd: float
    benign_efficacy_mean: float
    benign_efficacy_sd: float
    n_seeds: int

    @property
    def held_out(self) -> bool:
        """True when the thresholds never saw this attack family.

        calibrated_on is a comma separated list, because thresholds are normally
        fitted on more than one family. Comparing it as a single string would
        call every multi-family calibration held out, including the ones that
        contain the evaluation family, which is the exact case the flag exists
        to catch.
        """
        if self.calibrated_on in ("", "none"):
            return False
        families = {name.strip() for name in self.calibrated_on.split(",")}
        return self.attack_family not in families


def _sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), _sd(values)


def summarize_gated(states: list[dict]) -> list[GateSummary]:
    """Collapse per-seed gated cells, keeping every experimental dimension."""
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for st in states:
        if "meta" not in st:
            raise KeyError(f"cell {st.get('cell_id')!r} has no meta block")
        meta = st["meta"]
        family = meta.get("params", {}).get("attack_family", "unknown")
        calibrated_on = meta.get("calibrated_on", "none")
        for point in st.get("checkpoints", []):
            if "gate" not in point:
                raise KeyError(
                    f"cell {st.get('cell_id')!r} has a checkpoint with no gate block. "
                    "That is an ungated RQ1 cell; summarize it with rq1_analysis."
                )
            key = (meta["model"], meta["editor"], family, calibrated_on, point["at"])
            grouped[key].append(point)

    out: list[GateSummary] = []
    for (model, editor, family, calibrated_on, at), points in sorted(grouped.items()):
        sens_m, sens_s = _mean_sd([p["gate"]["sensitivity"] for p in points])
        cost_m, cost_s = _mean_sd([p["gate"]["false_positive_rate"] for p in points])
        prec_m, prec_s = _mean_sd([p["precision_at_deployment"] for p in points])
        lead_m, lead_s = _mean_sd([p["poison_corrected_leading"] for p in points])
        eff_m, eff_s = _mean_sd([p["benign_efficacy"] for p in points])
        out.append(GateSummary(
            model=model, editor=editor, attack_family=family,
            calibrated_on=calibrated_on, at=at,
            sensitivity_mean=sens_m, sensitivity_sd=sens_s,
            benign_blocked_mean=cost_m, benign_blocked_sd=cost_s,
            precision_at_deployment_mean=prec_m, precision_at_deployment_sd=prec_s,
            corrected_leading_mean=lead_m, corrected_leading_sd=lead_s,
            benign_efficacy_mean=eff_m, benign_efficacy_sd=eff_s,
            n_seeds=len({p["seed"] for p in points}),
        ))
    return out


@dataclass(frozen=True)
class NetBenefit:
    model: str
    editor: str
    attack_family: str
    at: int
    poison_reduction: float
    benign_blocked: float
    efficacy_loss: float
    held_out: bool

    @property
    def worth_it(self) -> bool:
        """A crude screen, not a decision.

        More poison kept out than good edits destroyed. Whether that trade is
        acceptable depends on what the good edits were and what the poison would
        have done, and no ratio settles it. It is here to catch the case where a
        gate is strictly worse than no gate.
        """
        return self.poison_reduction > self.benign_blocked


def net_benefit(gated: GateSummary, ungated: ArmSummary) -> NetBenefit:
    """Gated minus undefended, on arms that must match in everything else."""
    mismatch = [
        name for name, a, b in (
            ("model", gated.model, ungated.model),
            ("editor", gated.editor, ungated.editor),
            ("attack_family", gated.attack_family, ungated.attack_family),
            ("at", gated.at, ungated.at),
        ) if a != b
    ]
    if mismatch:
        raise ValueError(
            f"cannot subtract arms that differ in {mismatch}. The comparison is only "
            "meaningful between a gated and an ungated run of the same cell"
        )

    return NetBenefit(
        model=gated.model, editor=gated.editor, attack_family=gated.attack_family,
        at=gated.at,
        poison_reduction=ungated.corrected_leading_mean - gated.corrected_leading_mean,
        benign_blocked=gated.benign_blocked_mean,
        efficacy_loss=ungated.benign_efficacy_mean - gated.benign_efficacy_mean,
        held_out=gated.held_out,
    )


def pair_arms(gated: list[GateSummary],
              ungated: list[ArmSummary]) -> list[NetBenefit]:
    """Match gated and ungated summaries on their shared keys and subtract.

    Unmatched summaries are dropped rather than compared against a default. An
    arm with no control is not a weaker result, it is not a result.
    """
    index = {(u.model, u.editor, u.attack_family, u.at): u for u in ungated}
    out = []
    for g in gated:
        control = index.get((g.model, g.editor, g.attack_family, g.at))
        if control is not None:
            out.append(net_benefit(g, control))
    return out
