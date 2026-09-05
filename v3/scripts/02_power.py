"""How big the arms have to be. Run this before any number gets reported.

The survival gap is two elicitation rates measured on the same corrected poison
items, so the test is McNemar's and the driver is the discordant proportion: the
share of items where direct and leading questioning disagree. Rates alone say
nothing about power.

    python v3/scripts/02_power.py --discordant 0.35 --delta 0.15 --seeds 3

Nothing here is measured. Every input is an assumption, and the point of the
script is to make the assumption explicit before it is spent on GPU hours. Once
a pilot exists, re-run it with the pilot's own discordant proportion.
"""
from __future__ import annotations

import argparse

from u_jepa_v3.power import (
    feed_plan,
    mcnemar_power,
    mcnemar_sample_size,
    two_proportion_sample_size,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--discordant", type=float, default=0.35,
                        help="share of items where the two modes disagree")
    parser.add_argument("--delta", type=float, default=0.15,
                        help="survival gap worth detecting")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.8)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--base-rate", type=float, default=0.25,
                        help="share of the generated attack pool that gets injected")
    parser.add_argument("--benign-per-poison", type=int, default=25)
    parser.add_argument("--compare", type=float, nargs=2, metavar=("P1", "P2"),
                        default=(0.55, 0.30),
                        help="two editor survival rates to size an unpaired comparison")
    args = parser.parse_args(argv)

    plan = mcnemar_sample_size(args.discordant, args.delta, args.alpha, args.power)
    print("paired test on the survival gap (McNemar)")
    print(f"  discordant proportion   {plan.discordant}")
    print(f"  gap worth detecting     {plan.delta}")
    print(f"  alpha / power           {plan.alpha} / {plan.target_power}")
    print(f"  corrected poison items  {plan.n_pairs} per arm")

    print("\n  power at other sizes:")
    for n in (25, 50, 100, plan.n_pairs, 2 * plan.n_pairs):
        print(f"    n = {n:5}   power {mcnemar_power(n, args.discordant, args.delta):.3f}")

    unpaired = two_proportion_sample_size(*args.compare, alpha=args.alpha,
                                          power=args.power)
    print(f"\nunpaired comparison of {args.compare[0]} against {args.compare[1]}")
    print(f"  items per editor        {unpaired}")

    feed = feed_plan(plan.n_pairs, args.base_rate, args.benign_per_poison, args.seeds)
    print(f"\ngrid parameters, spread over {args.seeds} seeds")
    print(f"  n_poison                {feed.n_poison}")
    print(f"  n_benign                {feed.n_benign}")
    print(f"  base_rate               {feed.base_rate}")
    print(f"  prevalence in the feed  {feed.prevalence:.4f}")
    print(f"  edits per cell          {feed.n_benign + 2 * round(feed.n_poison * feed.base_rate)}")

    print("\nWhat is assumed rather than measured: the discordant proportion, the gap")
    print("worth detecting, and that seeds pool. Replace the first with the pilot's own")
    print("value as soon as one exists.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
