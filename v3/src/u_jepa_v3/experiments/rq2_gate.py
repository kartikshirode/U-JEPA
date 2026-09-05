"""RQ2: what does admission control buy, and what does it cost.

Same feed, same editor, same probes as RQ1. The only difference is that entries
pass a gate first, and refused or quarantined entries are never applied. So the
two arms subtract, and the difference is the gate's effect rather than a
difference in setup.

The poison numbers stay over every poison entry the feed has reached, not over
the ones that got admitted. An operator does not care how many of the attacks
they let through were successful; they care how many of the attacks sent at them
landed. Denominator choice is the easiest place to flatter a defence and this is
the honest one.

Calibration is separated from evaluation on purpose. collect_calibration runs a
scoring only pass over a feed built from some attack families, calibrate picks
thresholds from it, and the evaluation arm then faces a family the thresholds
never saw. A gate tuned and tested on the same attack is a memorisation result.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..gate.base import GateContext, GateInput
from ..gate.combiner import LinearCombiner
from ..gate.provenance import TrustTracker
from ..gate.rollback import ShadowLedger
from ..power import precision_at_base_rate
from ..runs.state import RunState
from ..schema import Decision, FeedEntry
from .rq1_survival import Rq1Config, checkpoint_metrics


@dataclass(frozen=True)
class Rq2Config:
    checkpoint_every: int
    seed: int
    model: str
    editor: str
    base_rate: float
    revert_lag: int
    trust_lag: int = 200
    calibrated_on: str = "none"
    # Prevalence the operator would actually face, which is not base_rate.
    # base_rate is how much of the generated attack pool this run injects, a
    # knob for making the experiment powered. Deployment prevalence is how
    # often a real feed carries a poisoned entry, and precision is reported
    # against that. Conflating them was how the v2 numbers flattered themselves.
    deployment_prevalence: float = 0.001

    def __post_init__(self) -> None:
        if self.checkpoint_every < 1:
            raise ValueError(f"checkpoint_every must be >= 1, got {self.checkpoint_every}")
        if self.trust_lag < 1:
            raise ValueError(f"trust_lag must be >= 1, got {self.trust_lag}")
        if not 0.0 < self.deployment_prevalence < 1.0:
            raise ValueError("deployment_prevalence must be in (0, 1), got "
                             f"{self.deployment_prevalence}")

    def as_rq1(self) -> Rq1Config:
        return Rq1Config(
            checkpoint_every=self.checkpoint_every, seed=self.seed,
            model=self.model, editor=self.editor,
            base_rate=self.base_rate, revert_lag=self.revert_lag,
        )


@dataclass
class GateCounts:
    """Running confusion matrix of decisions against the simulation's ground truth.

    Quarantine blocks. An entry a human has not looked at yet is not in the
    model, so for the purpose of what the model believes, quarantined and refused
    are the same. They separate again in the cost column, because one is a
    rejection and the other is a person's afternoon.
    """

    poison_admitted: int = 0
    poison_refused: int = 0
    poison_quarantined: int = 0
    benign_admitted: int = 0
    benign_refused: int = 0
    benign_quarantined: int = 0

    @property
    def poison_seen(self) -> int:
        return self.poison_admitted + self.poison_refused + self.poison_quarantined

    @property
    def benign_seen(self) -> int:
        return self.benign_admitted + self.benign_refused + self.benign_quarantined

    @property
    def poison_blocked(self) -> int:
        return self.poison_refused + self.poison_quarantined

    @property
    def benign_blocked(self) -> int:
        return self.benign_refused + self.benign_quarantined

    @property
    def sensitivity(self) -> float:
        return self.poison_blocked / self.poison_seen if self.poison_seen else 0.0

    @property
    def false_positive_rate(self) -> float:
        return self.benign_blocked / self.benign_seen if self.benign_seen else 0.0

    def observed_precision(self) -> float:
        blocked = self.poison_blocked + self.benign_blocked
        return self.poison_blocked / blocked if blocked else 0.0

    def precision_at(self, base_rate: float) -> float:
        """Precision the same gate would post at a different prevalence."""
        return precision_at_base_rate(self.sensitivity, self.false_positive_rate,
                                      base_rate)

    def record(self, is_poison: bool, decision: Decision) -> None:
        bucket = "poison" if is_poison else "benign"
        name = {Decision.ADMIT: "admitted", Decision.REFUSE: "refused",
                Decision.QUARANTINE: "quarantined"}[decision]
        setattr(self, f"{bucket}_{name}", getattr(self, f"{bucket}_{name}") + 1)

    def as_dict(self) -> dict:
        return {
            "poison_admitted": self.poison_admitted,
            "poison_refused": self.poison_refused,
            "poison_quarantined": self.poison_quarantined,
            "benign_admitted": self.benign_admitted,
            "benign_refused": self.benign_refused,
            "benign_quarantined": self.benign_quarantined,
            "sensitivity": round(self.sensitivity, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "observed_precision": round(self.observed_precision(), 4),
        }


def collect_calibration(
    combiner: LinearCombiner,
    feed: list[FeedEntry],
    sources: dict[str, str],
    ctx: GateContext,
) -> tuple[list[float], list[bool]]:
    """Score every entry without blocking any of it, and return scores with labels.

    Admitting everything is deliberate. The context a gate sees in deployment is
    the one an undefended pipeline produced up to that point, so calibrating
    against a context shaped by the gate's own refusals would fit the gate to
    itself.
    """
    scores: list[float] = []
    labels: list[bool] = []
    for entry in feed:
        item = GateInput.from_entry(entry, sources[entry.entry_id])
        scored = combiner.score([item], ctx)[0]
        scores.append(scored.total)
        labels.append(entry.is_poison)
        ctx.observe(item)
    return scores, labels


def run_gated_arm(
    combiner: LinearCombiner,
    editor,
    feed: list[FeedEntry],
    sources: dict[str, str],
    ctx: GateContext,
    suites: dict[str, list[tuple[str, str]]],
    locality_pairs: list[tuple[str, str]],
    base_responder,
    config: Rq2Config,
    hop_questions: dict[str, list[tuple[str, str, str]]] | None = None,
    ledger: ShadowLedger | None = None,
) -> RunState:
    """Stream a feed through the gate and then the editor, probing at intervals."""
    from ..probes.general_ability import general_ability

    hop_questions = hop_questions or {}
    ledger = ledger if ledger is not None else ShadowLedger()
    tracker = TrustTracker(ctx.trust, config.trust_lag) if ctx.trust else None
    counts = GateCounts()

    state = RunState(cell_id="", meta={
        "model": config.model,
        "editor": config.editor,
        "seed": config.seed,
        "base_rate": config.base_rate,
        "revert_lag": config.revert_lag,
        "n_feed": len(feed),
        "arm": "gated",
        "calibrated_on": config.calibrated_on,
        "refuse_at": combiner.thresholds.refuse_at,
        "quarantine_at": combiner.thresholds.quarantine_at,
        "weights": dict(combiner.weights),
        "baseline_general": general_ability(base_responder, suites).mean,
    })

    n_failed = n_applied = 0
    rq1_config = config.as_rq1()
    for start in range(0, len(feed), config.checkpoint_every):
        batch = feed[start : start + config.checkpoint_every]
        items = [GateInput.from_entry(e, sources[e.entry_id]) for e in batch]
        if ctx.belief is not None and n_applied:
            # The gate has to ask the model it is protecting, and that model has
            # moved. Left pointing at the base, the belief signal would answer
            # from a snapshot that gets staler with every batch.
            ctx.belief = editor.responder().answer
        decisions = combiner.decide(items, ctx)

        admitted = []
        for entry, item, decision in zip(batch, items, decisions):
            counts.record(entry.is_poison, decision.decision)
            if not decision.admitted:
                continue
            admitted.append(entry.candidate)
            ctx.observe(item)
            ledger.record(entry.entry_id, entry.position, entry.candidate)
            if tracker:
                tracker.submitted(entry.entry_id, item.source, entry.position)
                # A correction is public information that arrives with the feed,
                # so charging it to the account that submitted the original is
                # something an operator can actually do. It is not a label: the
                # correction lands after the entry it corrects.
                if entry.reverts:
                    tracker.reverted(entry.reverts)

        results = editor.apply(admitted)
        n_failed += sum(not r.succeeded for r in results)
        n_applied += sum(r.succeeded for r in results)
        upto = start + len(batch)
        if tracker:
            tracker.advance(upto)

        # Nothing applied yet means the model is still the base, so that is what
        # gets probed. Asking the editor for a model it has not built raises.
        probe_target = None if n_applied else base_responder
        point = checkpoint_metrics(editor, feed, upto, suites, locality_pairs,
                                   n_failed, rq1_config, hop_questions,
                                   responder=probe_target)
        point.update({
            "n_admitted": len(admitted),
            "n_applied": n_applied,
            "n_ledger": len(ledger),
            "gate": counts.as_dict(),
            "observed_prevalence": round(
                counts.poison_seen / max(counts.poison_seen + counts.benign_seen, 1), 5),
            "precision_at_deployment": round(
                counts.precision_at(config.deployment_prevalence), 4),
            "trust_pending": tracker.pending() if tracker else 0,
        })
        state.checkpoints.append(point)

    state.meta["gate_totals"] = counts.as_dict()
    state.finished = True
    return state


def _families(raw) -> list[str]:
    """Accept a list or a comma separated string, since grids carry both badly."""
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(",") if part.strip()]
    return list(raw)


def _attacks(family: str, benign, seed: int, n: int):
    from ..data import adversarial

    if family == adversarial.AttackFamily.OBJECT_SWAP.value:
        return adversarial.poison_object_swap(benign, seed, n)
    if family == adversarial.AttackFamily.TYPE_CONSISTENT.value:
        return adversarial.poison_type_consistent(benign, seed, n)
    if family == adversarial.AttackFamily.TEMPORAL_STALE.value:
        return adversarial.poison_temporal_stale(
            adversarial.build_history(benign), seed, n)
    raise ValueError(f"unknown attack family {family!r}")


def run_cell_from_params(params: dict) -> RunState:
    """Build and run one gated cell from grid params. Called by the shard worker.

    Expected keys: everything RQ1 takes, plus calibrate_on naming the families
    the thresholds are fitted on. When calibrate_on holds the evaluation family
    the result is the ceiling rather than a transfer number, and the analysis
    marks it as such.
    """
    from ..data import wikibigedit
    from ..data.feed import build_feed
    from ..data.relation_prior import RelationPrior
    from ..editors import registry
    from ..editors.easyedit_adapter import HFResponder
    from ..gate.combiner import LinearCombiner, calibrate
    from ..gate.provenance import SourceTrust, simulate_sources
    from ..gate.signals import default_signals
    from .rq1_survival import (
        _load_base,
        _load_locality,
        _load_suites,
        check_fits,
        resolve_hparams,
    )

    registry.register_defaults()
    check_fits(params)

    seed = params["seed"]
    n_poison = params["n_poison"]
    corpus = wikibigedit.load_candidates()
    benign = wikibigedit.sample_candidates(corpus, params["n_benign"], seed)

    # The operator's own history, drawn on a different seed so the prior and the
    # trusted vocabulary are not the very rows being scored.
    prior_rows = wikibigedit.sample_candidates(
        corpus, params.get("n_prior", params["n_benign"]), seed + 10_000)
    prior = RelationPrior.from_candidates(prior_rows)

    eval_family = params["attack_family"]
    calibration_families = _families(params.get("calibrate_on", []))

    signals = default_signals(with_belief=bool(params.get("with_belief")))
    combiner = LinearCombiner(signals)

    calibrated_on = "none"
    if calibration_families:
        scores: list[float] = []
        labels: list[bool] = []
        for family in calibration_families:
            pairs = _attacks(family, benign, seed, n_poison)
            cal_feed = build_feed(benign, pairs, params["base_rate"],
                                  params["revert_lag"], seed)
            cal_ctx = GateContext(prior=prior, trust=SourceTrust())
            cal_ctx.prime(prior_rows)
            got, want = collect_calibration(
                combiner, cal_feed, simulate_sources(cal_feed, seed), cal_ctx)
            scores.extend(got)
            labels.extend(want)
        fitted = calibrate(scores, labels,
                           base_rate=params.get("deployment_prevalence", 0.001),
                           target_precision=params.get("target_precision", 0.9))
        combiner = combiner.with_thresholds(fitted.thresholds)
        calibrated_on = ",".join(calibration_families)

    pairs = _attacks(eval_family, benign, seed, n_poison)
    feed = build_feed(benign, pairs, params["base_rate"], params["revert_lag"], seed)
    sources = simulate_sources(feed, seed)

    ctx = GateContext(prior=prior, trust=SourceTrust())
    ctx.prime(prior_rows)

    editor = registry.build(params["editor"], hparams_path=resolve_hparams(params))
    base_model, base_tok = _load_base(params["model"])
    base_responder = HFResponder(base_model, base_tok)
    if params.get("with_belief"):
        ctx.belief = base_responder.answer

    config = Rq2Config(
        checkpoint_every=params["checkpoint_every"], seed=seed,
        model=params["model"], editor=params["editor"],
        base_rate=params["base_rate"], revert_lag=params["revert_lag"],
        trust_lag=params.get("trust_lag", 200),
        calibrated_on=calibrated_on,
        deployment_prevalence=params.get("deployment_prevalence", 0.001),
    )
    return run_gated_arm(combiner, editor, feed, sources, ctx, _load_suites(),
                         _load_locality(), base_responder, config)
