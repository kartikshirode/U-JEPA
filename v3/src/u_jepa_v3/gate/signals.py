"""The evidence the gate weighs. Each signal returns suspicion in [0, 1].

None of these is a fact checker. They are cheap properties of a claim and of the
stream it arrived in, and every one of them fires on some legitimate edits. That
is the whole difficulty: a real knowledge feed is mostly true revisions to
things that recently changed, which is also what an attack looks like from the
outside.

Two of the three attack families are expected to be reachable. Object swap
violates type, so the vocabulary signal should catch it. A type consistent swap
looks normal in isolation and needs the belief and stream signals. A temporal
stale attack asserts a value the slot genuinely held, so no signal here can call
it false, only unusual; if the gate stops it at all it will be through churn and
provenance. That prediction is written down before the numbers exist so it can
be wrong in public.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..probes.efficacy import normalize_answer
from .base import GateContext, GateInput, is_revision

# What a signal returns when it has no basis to judge. Not zero, because zero
# means "looks fine" and would let an unknown relation through on confidence
# nobody has.
ABSTAIN = 0.5

# An object nobody has used for this relation yet. Most genuinely new facts look
# like this, so it is a mild prior rather than an accusation.
UNSEEN_OBJECT = 0.15


@dataclass
class TypeViolationSignal:
    """Is this object the kind of thing this relation takes.

    Cheap and effective against a crude swap, and blind to a careful one. The
    vocabulary comes from what the gate has admitted, so a long enough run of
    accepted poison teaches it that poison is normal. That is a real weakness
    and the reason ctx.observe is called on admitted entries only.
    """

    name: str = "type_violation"
    unseen: float = UNSEEN_OBJECT

    def score(self, batch: list[GateInput], ctx: GateContext) -> list[float]:
        out = []
        for item in batch:
            candidate = item.candidate
            own = ctx.object_vocab.get(candidate.relation_id, set())
            if candidate.object in own:
                out.append(0.0)
                continue
            elsewhere = any(candidate.object in objects
                            for relation, objects in ctx.object_vocab.items()
                            if relation != candidate.relation_id)
            out.append(1.0 if elsewhere else self.unseen)
        return out


@dataclass
class PriorMismatchSignal:
    """A revision to a relation whose rows are rarely revisions.

    Uses update_share from the relation prior, which is the composition of
    observed change and not a volatility rate. It is being used here as what it
    is: a stable per relation feature, split-half Spearman 0.695 in the Q1 spike.
    Whether it carries decision value is exactly what stage 2 measures.
    """

    name: str = "prior_mismatch"

    def score(self, batch: list[GateInput], ctx: GateContext) -> list[float]:
        out = []
        for item in batch:
            if not is_revision(item, ctx):
                out.append(0.0)
                continue
            relation = item.candidate.relation_id
            if ctx.prior is None or relation not in ctx.prior:
                out.append(ABSTAIN)
                continue
            out.append(1.0 - ctx.prior.update_share(relation))
        return out


@dataclass
class SourceTrustSignal:
    """How often this account has been reverted, smoothed toward the prior.

    Late by construction. Trust only moves when a correction arrives, which is
    after the damage. It is in the combiner to measure how much that lateness
    costs, not because it is expected to carry the gate.
    """

    name: str = "source_trust"

    def score(self, batch: list[GateInput], ctx: GateContext) -> list[float]:
        if ctx.trust is None:
            return [ABSTAIN] * len(batch)
        return [1.0 - ctx.trust.trust(item.source) for item in batch]


@dataclass
class BurstSignal:
    """Volume from one account, or attention on one subject, inside the window.

    A campaign is rarely one edit. Scheduled bot passes are also not one edit,
    which is why this fires on plenty of benign traffic and cannot be used alone.
    """

    name: str = "burst"
    cap: int = 25

    def __post_init__(self) -> None:
        if self.cap < 1:
            raise ValueError(f"cap must be >= 1, got {self.cap}")

    def score(self, batch: list[GateInput], ctx: GateContext) -> list[float]:
        out = []
        for item in batch:
            from_source = ctx.recent_from_source(item.source, item.position)
            on_subject = ctx.recent_for_subject(item.candidate.subject_id, item.position)
            out.append(min(max(from_source, on_subject) / self.cap, 1.0))
        return out


@dataclass
class SlotChurnSignal:
    """This slot has already been written during this run.

    A fact that gets rewritten repeatedly is either genuinely unsettled or being
    fought over. Both are worth a human, which is what quarantine is for.
    """

    name: str = "slot_churn"
    cap: int = 3

    def __post_init__(self) -> None:
        if self.cap < 1:
            raise ValueError(f"cap must be >= 1, got {self.cap}")

    def score(self, batch: list[GateInput], ctx: GateContext) -> list[float]:
        return [min(ctx.slot_writes.get(item.key, 0) / self.cap, 1.0) for item in batch]


@dataclass
class BeliefContradictionSignal:
    """The model already answers this slot, and with something else.

    The only signal that asks the model anything, and the only expensive one:
    one short generation per entry. It fires on every genuine revision too, so
    on its own it is close to useless. It is here because paired with a low
    update share it separates "this relation changes and this one changed" from
    "this relation never changes and something just changed it".
    """

    name: str = "belief_contradiction"

    def score(self, batch: list[GateInput], ctx: GateContext) -> list[float]:
        if ctx.belief is None:
            raise RuntimeError(
                "belief_contradiction needs GateContext.belief, a callable taking "
                "prompts and returning answers. Drop the signal or supply a model."
            )
        if not batch:
            return []

        answers = ctx.belief([item.candidate.prompt for item in batch])
        if len(answers) != len(batch):
            raise RuntimeError(
                f"belief returned {len(answers)} answers for {len(batch)} prompts"
            )

        out = []
        for item, answer in zip(batch, answers):
            held = normalize_answer(answer)
            if not held:
                out.append(ABSTAIN)
            elif held == normalize_answer(item.candidate.object):
                out.append(0.0)
            else:
                out.append(1.0)
        return out


def default_signals(with_belief: bool = False) -> list:
    """The set stage 2 starts from. Belief is opt-in because it costs a forward pass."""
    signals = [TypeViolationSignal(), PriorMismatchSignal(), SourceTrustSignal(),
               BurstSignal(), SlotChurnSignal()]
    if with_belief:
        signals.append(BeliefContradictionSignal())
    return signals
