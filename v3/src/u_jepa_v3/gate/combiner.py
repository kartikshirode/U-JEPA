"""Turn several weak signals into one decision, and pick the thresholds honestly.

Two thresholds, because refusing and asking a human are different actions with
different costs. Above refuse_at the entry never reaches the editor. Between the
two it is quarantined, which is a claim that a person will look at it, so the
quarantine rate is a staffing number and not a free win.

Thresholds are fitted, never chosen by eye, and fitted against the base rate the
operator actually faces. A gate calibrated on a balanced sample and reported by
AUROC can look excellent and still refuse mostly good edits in deployment,
because precision falls with prevalence and AUROC does not move at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..power import precision_at_base_rate
from ..schema import Decision
from .base import GateContext, GateDecision, GateInput, GateScore, Signal

DEFAULT_REFUSE_AT = 0.7
DEFAULT_QUARANTINE_AT = 0.5


@dataclass(frozen=True)
class Thresholds:
    refuse_at: float = DEFAULT_REFUSE_AT
    quarantine_at: float = DEFAULT_QUARANTINE_AT

    def __post_init__(self) -> None:
        for name in ("refuse_at", "quarantine_at"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.quarantine_at > self.refuse_at:
            raise ValueError(
                f"quarantine_at {self.quarantine_at} is above refuse_at "
                f"{self.refuse_at}, which would refuse entries too weak to quarantine"
            )


class LinearCombiner:
    """Weighted mean of the signals, with the weights held explicit.

    Linear on purpose. With one labelled campaign to calibrate against, anything
    with more capacity fits the campaign rather than the attack, and the per
    signal breakdown is the part an operator can argue with.
    """

    def __init__(self, signals: list[Signal], weights: dict[str, float] | None = None,
                 thresholds: Thresholds | None = None) -> None:
        if not signals:
            raise ValueError("a combiner needs at least one signal")
        names = [s.name for s in signals]
        if len(set(names)) != len(names):
            raise ValueError(f"signal names must be unique, got {names}")

        self.signals = signals
        self.thresholds = thresholds or Thresholds()
        raw = weights or {name: 1.0 for name in names}
        unknown = set(raw) - set(names)
        if unknown:
            raise KeyError(f"weights name signals that are not present: {sorted(unknown)}")
        missing = set(names) - set(raw)
        if missing:
            raise KeyError(f"no weight for {sorted(missing)}")
        if any(w < 0 for w in raw.values()):
            raise ValueError("weights must not be negative")
        total = sum(raw.values())
        if total <= 0:
            raise ValueError("weights must not sum to zero")
        self.weights = {name: raw[name] / total for name in names}

    def score(self, batch: list[GateInput], ctx: GateContext) -> list[GateScore]:
        if not batch:
            return []

        columns: dict[str, list[float]] = {}
        for signal in self.signals:
            values = signal.score(batch, ctx)
            if len(values) != len(batch):
                raise RuntimeError(
                    f"signal {signal.name!r} returned {len(values)} scores for "
                    f"{len(batch)} entries"
                )
            for value in values:
                if not 0.0 <= value <= 1.0:
                    raise RuntimeError(
                        f"signal {signal.name!r} returned {value}, outside [0, 1]"
                    )
            columns[signal.name] = values

        out = []
        for index, item in enumerate(batch):
            per_signal = {name: columns[name][index] for name in columns}
            total = sum(per_signal[name] * self.weights[name] for name in per_signal)
            out.append(GateScore(item.entry_id, total, per_signal))
        return out

    def decide(self, batch: list[GateInput], ctx: GateContext) -> list[GateDecision]:
        out = []
        for score in self.score(batch, ctx):
            if score.total >= self.thresholds.refuse_at:
                decision = Decision.REFUSE
            elif score.total >= self.thresholds.quarantine_at:
                decision = Decision.QUARANTINE
            else:
                decision = Decision.ADMIT
            out.append(GateDecision(score.entry_id, decision, score))
        return out

    def with_thresholds(self, thresholds: Thresholds) -> "LinearCombiner":
        return LinearCombiner(self.signals, dict(self.weights), thresholds)


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    sensitivity: float
    false_positive_rate: float
    precision: float

    def as_dict(self) -> dict:
        return {"threshold": round(self.threshold, 4),
                "sensitivity": round(self.sensitivity, 4),
                "false_positive_rate": round(self.false_positive_rate, 4),
                "precision": round(self.precision, 4)}


@dataclass(frozen=True)
class Calibration:
    thresholds: Thresholds
    base_rate: float
    target_precision: float
    met_target: bool
    sensitivity: float
    false_positive_rate: float
    precision: float
    quarantine_rate: float
    n_poison: int
    n_benign: int
    points: list[OperatingPoint] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "refuse_at": round(self.thresholds.refuse_at, 4),
            "quarantine_at": round(self.thresholds.quarantine_at, 4),
            "base_rate": self.base_rate,
            "target_precision": self.target_precision,
            "met_target": self.met_target,
            "sensitivity": round(self.sensitivity, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "precision": round(self.precision, 4),
            "quarantine_rate": round(self.quarantine_rate, 4),
            "n_poison": self.n_poison, "n_benign": self.n_benign,
        }


def sweep(scores: list[float], labels: list[bool],
          base_rate: float) -> list[OperatingPoint]:
    """Every distinct operating point, precision computed at the deployment rate."""
    if len(scores) != len(labels):
        raise ValueError(f"{len(scores)} scores against {len(labels)} labels")
    n_poison = sum(labels)
    n_benign = len(labels) - n_poison
    if n_poison == 0 or n_benign == 0:
        raise ValueError(
            f"calibration needs both classes, got {n_poison} poison and "
            f"{n_benign} benign"
        )

    candidates = sorted({round(s, 6) for s in scores})
    points = []
    for threshold in candidates:
        flagged = [(s >= threshold) for s in scores]
        tp = sum(f and l for f, l in zip(flagged, labels))
        fp = sum(f and not l for f, l in zip(flagged, labels))
        sensitivity = tp / n_poison
        fpr = fp / n_benign
        points.append(OperatingPoint(
            threshold=threshold, sensitivity=sensitivity, false_positive_rate=fpr,
            precision=precision_at_base_rate(sensitivity, fpr, base_rate),
        ))
    return points


def calibrate(scores: list[float], labels: list[bool], base_rate: float,
              target_precision: float = 0.9,
              quarantine_recall: float = 0.9) -> Calibration:
    """Pick the pair of thresholds, maximising recall subject to precision.

    The refusal threshold is the lowest one whose precision at the deployment
    base rate clears the target, which is the most poison you can stop while
    keeping the promise you made about false refusals. If nothing clears it, the
    best available point comes back with met_target False rather than a
    threshold that quietly misses the target.

    The quarantine threshold is the highest one still reaching quarantine_recall,
    so the band between them is as narrow as that recall allows. A wide band is
    a large human review queue.
    """
    if not 0.0 < target_precision <= 1.0:
        raise ValueError(f"target_precision must be in (0, 1], got {target_precision}")
    if not 0.0 < quarantine_recall <= 1.0:
        raise ValueError(f"quarantine_recall must be in (0, 1], got {quarantine_recall}")

    points = sweep(scores, labels, base_rate)
    n_poison = sum(labels)
    n_benign = len(labels) - n_poison

    clearing = [p for p in points if p.precision >= target_precision]
    if clearing:
        best = max(clearing, key=lambda p: (p.sensitivity, -p.threshold))
        met = True
    else:
        best = max(points, key=lambda p: (p.precision, p.sensitivity))
        met = False

    reaching = [p for p in points if p.sensitivity >= quarantine_recall]
    quarantine_at = min(max(reaching, key=lambda p: p.threshold).threshold,
                        best.threshold) if reaching else 0.0

    flagged_quarantine = sum(1 for s in scores if quarantine_at <= s < best.threshold)
    return Calibration(
        thresholds=Thresholds(refuse_at=best.threshold, quarantine_at=quarantine_at),
        base_rate=base_rate, target_precision=target_precision, met_target=met,
        sensitivity=best.sensitivity, false_positive_rate=best.false_positive_rate,
        precision=best.precision,
        quarantine_rate=flagged_quarantine / len(scores) if scores else 0.0,
        n_poison=n_poison, n_benign=n_benign, points=points,
    )
