"""How many poisoned facts an arm needs before its numbers mean anything.

The headline of RQ1 is the survival gap: elicitation under a leading context
minus elicitation under direct questioning, both measured on the same corrected
poison items. Same items means the two rates are paired, so the test is
McNemar's and the thing that drives power is the discordant proportion, not the
rates themselves.

That distinction is expensive to get wrong. Two rates of 0.20 and 0.55 look like
a huge effect, but if only 8% of items disagree between the modes there is no
power at any sample size the pilot grid would fund.

Nothing here needs scipy. The normal quantiles come from statistics.NormalDist,
which ships with the standard library.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

_NORMAL = NormalDist()


def z(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError(f"quantile needs p in (0, 1), got {p}")
    return _NORMAL.inv_cdf(p)


def _check_pair(discordant: float, delta: float) -> None:
    if not 0.0 < discordant <= 1.0:
        raise ValueError(f"discordant must be in (0, 1], got {discordant}")
    if delta <= 0.0:
        raise ValueError(f"delta must be > 0, got {delta}. State the effect you "
                         "would act on, not the one you hope for")
    if delta > discordant:
        raise ValueError(
            f"delta {delta} exceeds the discordant proportion {discordant}. The gap "
            "is the difference between the two disagreement cells, so it can never "
            "be larger than their sum"
        )


@dataclass(frozen=True)
class PairedPlan:
    """A McNemar plan for one arm of the survival gap."""

    discordant: float
    delta: float
    alpha: float
    target_power: float
    n_pairs: int

    def as_dict(self) -> dict:
        return {"discordant": self.discordant, "delta": self.delta,
                "alpha": self.alpha, "target_power": self.target_power,
                "n_pairs": self.n_pairs}


def mcnemar_sample_size(discordant: float, delta: float, alpha: float = 0.05,
                        power: float = 0.8) -> PairedPlan:
    """Corrected poison items needed per arm, following Connor (1987).

    discordant is the share of items where the two elicitation modes disagree in
    either direction. delta is the difference between the two disagreement
    directions, which is the survival gap itself.
    """
    _check_pair(discordant, delta)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}")

    za, zb = z(1 - alpha / 2), z(power)
    numerator = za * math.sqrt(discordant) + zb * math.sqrt(discordant - delta ** 2)
    n = (numerator ** 2) / (delta ** 2)
    return PairedPlan(discordant, delta, alpha, power, math.ceil(n))


def mcnemar_power(n_pairs: int, discordant: float, delta: float,
                  alpha: float = 0.05) -> float:
    """Power of the paired test at a sample size you are stuck with."""
    _check_pair(discordant, delta)
    if n_pairs < 1:
        raise ValueError(f"n_pairs must be >= 1, got {n_pairs}")
    za = z(1 - alpha / 2)
    spread = math.sqrt(discordant - delta ** 2)
    if spread == 0.0:
        return 1.0
    stat = (delta * math.sqrt(n_pairs) - za * math.sqrt(discordant)) / spread
    return _NORMAL.cdf(stat)


def two_proportion_sample_size(p1: float, p2: float, alpha: float = 0.05,
                               power: float = 0.8) -> int:
    """Items per group for an unpaired comparison, such as editor A against B.

    Unpaired because two editors see two independently poisoned feeds. Use the
    paired form for anything measured twice on the same items.
    """
    for name, p in (("p1", p1), ("p2", p2)):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"{name} must be a proportion, got {p}")
    if p1 == p2:
        raise ValueError("p1 and p2 are equal, so there is no effect to power for")

    za, zb = z(1 - alpha / 2), z(power)
    pooled = (p1 + p2) / 2
    a = za * math.sqrt(2 * pooled * (1 - pooled))
    b = zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return math.ceil(((a + b) ** 2) / ((p1 - p2) ** 2))


def precision_at_base_rate(sensitivity: float, false_positive_rate: float,
                           base_rate: float) -> float:
    """What a detector's precision becomes once prevalence is realistic.

    A gate tuned on a balanced sample and reported by AUROC can look excellent
    and still refuse mostly benign edits in deployment, because precision falls
    with prevalence while AUROC does not move at all. Stage 2 reports this
    number instead.
    """
    for name, value in (("sensitivity", sensitivity),
                        ("false_positive_rate", false_positive_rate),
                        ("base_rate", base_rate)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    hits = base_rate * sensitivity
    misses = (1 - base_rate) * false_positive_rate
    if hits + misses == 0.0:
        return 0.0
    return hits / (hits + misses)


def wilson_interval(hits: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval, which stays inside [0, 1] near the edges.

    The normal approximation does not, and elicitation rates sit near 0 or 1
    often enough that it matters.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not 0 <= hits <= n:
        raise ValueError(f"hits must be in [0, {n}], got {hits}")
    zc = z(1 - alpha / 2)
    phat = hits / n
    denom = 1 + zc ** 2 / n
    centre = (phat + zc ** 2 / (2 * n)) / denom
    half = zc * math.sqrt(phat * (1 - phat) / n + zc ** 2 / (4 * n ** 2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass(frozen=True)
class FeedPlan:
    """Grid parameters that deliver a required number of corrected poison items."""

    n_pairs_needed: int
    base_rate: float
    n_poison: int
    n_benign: int
    prevalence: float

    def as_dict(self) -> dict:
        return {"n_pairs_needed": self.n_pairs_needed, "base_rate": self.base_rate,
                "n_poison": self.n_poison, "n_benign": self.n_benign,
                "prevalence": round(self.prevalence, 4)}


def feed_plan(n_pairs_needed: int, base_rate: float, benign_per_poison: int = 25,
              n_seeds: int = 1) -> FeedPlan:
    """Turn a power requirement into n_poison and n_benign for the grid.

    base_rate is the share of generated pairs that get injected, so the poison
    pool has to be larger than the number of items you want to measure. Seeds
    pool, so 3 seeds of 40 items answer the same question as 1 of 120 provided
    the items themselves differ, which the seeded sampler guarantees.

    benign_per_poison sets how dilute the feed is. It controls prevalence, and
    prevalence is what the stage 2 precision numbers are reported against.
    """
    if n_pairs_needed < 1:
        raise ValueError(f"n_pairs_needed must be >= 1, got {n_pairs_needed}")
    if not 0.0 < base_rate <= 1.0:
        raise ValueError(f"base_rate must be in (0, 1], got {base_rate}")
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1, got {n_seeds}")
    if benign_per_poison < 1:
        raise ValueError(f"benign_per_poison must be >= 1, got {benign_per_poison}")

    per_seed = math.ceil(n_pairs_needed / n_seeds)
    n_poison = math.ceil(per_seed / base_rate)
    injected = round(n_poison * base_rate)
    n_benign = injected * benign_per_poison
    # Each injected pair also emits a correction, so the feed carries
    # n_benign + 2 * injected entries in total.
    prevalence = injected / (n_benign + 2 * injected) if injected else 0.0
    return FeedPlan(n_pairs_needed, base_rate, n_poison, n_benign, prevalence)
