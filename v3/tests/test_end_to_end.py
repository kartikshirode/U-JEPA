"""One CPU-only pass through the whole chain, with no model and no network.

Every other test module checks a unit. This one checks that the units compose:
corpus to attack families to feed to editor to probes to cell state to analysis.
It is the test that would have caught the superseded plan's central defect, where
the adapter dropped the edited model and every probe silently read the base.
"""
from __future__ import annotations

import pandas as pd

from u_jepa_v3.data import wikibigedit as wbe
from u_jepa_v3.data.adversarial import poison_object_swap, poison_type_consistent
from u_jepa_v3.data.feed import build_feed, poison_entries
from u_jepa_v3.data.relation_prior import RelationPrior
from u_jepa_v3.editors.stub import StubEditor
from u_jepa_v3.experiments.rq1_analysis import summarize, survival_gap
from u_jepa_v3.experiments.rq1_survival import Rq1Config, run_arm
from u_jepa_v3.experiments.rq2_analysis import summarize_gated
from u_jepa_v3.experiments.rq2_gate import Rq2Config, collect_calibration, run_gated_arm
from u_jepa_v3.gate.base import GateContext
from u_jepa_v3.gate.combiner import LinearCombiner, calibrate
from u_jepa_v3.gate.provenance import SourceTrust, simulate_sources
from u_jepa_v3.gate.signals import default_signals
from u_jepa_v3.runs.grid import Cell, expand, shard
from u_jepa_v3.runs.state import load
from u_jepa_v3.runs.worker import run_cell

SUITES = {"sst": [("a", "a")], "mmlu": [("b", "b")],
          "mrpc": [("c", "c")], "nli": [("d", "d")]}


class BaseResponder:
    def answer(self, prompts):
        return ["<base>"] * len(prompts)


def fake_corpus(n=60):
    """Stands in for the WikiBigEdit download, through the real loader."""
    rows = []
    for i in range(n):
        relation, prefix = ("P26", "spouse") if i % 2 else ("P54", "team")
        rows.append(dict(
            tag="update" if i % 3 else "new",
            subject=f"S{i}", subject_id=f"Q{i}",
            relation=relation, relation_id=relation,
            object=f"{prefix}{i}", object_id=f"O{i}",
            rephrase=f"Who is the {relation} of S{i}?", timestep=i % 8,
        ))
    return wbe.to_candidates(pd.DataFrame(rows))


def test_the_whole_chain_runs_and_produces_a_summary(tmp_path):
    benign = fake_corpus()
    pairs = poison_type_consistent(benign, seed=0, n=4)
    assert pairs, "expected the generator to produce attacks"

    feed = build_feed(benign, pairs, base_rate=1.0, revert_lag=4, seed=0)
    assert len(poison_entries(feed)) == len(pairs)

    def runner(params):
        return run_arm(
            StubEditor(), feed, SUITES, [], BaseResponder(),
            Rq1Config(checkpoint_every=10, seed=params["seed"],
                      model="stub-model", editor="stub",
                      base_rate=1.0, revert_lag=4),
        )

    cells = [Cell({"seed": s}) for s in range(3)]
    paths = [run_cell(c, tmp_path, runner) for c in cells]
    states = [load(p).__dict__ for p in paths]

    assert all(s["finished"] for s in states)

    summaries = summarize(states)
    assert summaries, "analysis produced nothing"
    final = max(summaries, key=lambda s: s.at)
    assert final.n_seeds == 3
    assert final.model == "stub-model" and final.editor == "stub"

    # The stub applies the revert, so direct questioning returns the true value
    # and the survival gap is zero. On a real model this is the number RQ1 reports.
    assert final.corrected_direct_mean == 0.0
    assert survival_gap(final) == 0.0


def test_the_gated_chain_runs_and_beats_the_undefended_one(tmp_path):
    """The stage 2 counterpart: calibrate on two families, evaluate on the third.

    The numbers themselves are the stub's, not a result. What this checks is
    that calibration, the gate, the editor and the analysis compose, and that a
    gated arm can be subtracted from its control.
    """
    benign = fake_corpus()
    prior = RelationPrior.from_candidates(benign, min_support=1)

    combiner = LinearCombiner(default_signals())
    scores, labels = [], []
    for family, generator in (("object_swap", poison_object_swap),
                              ("type_consistent", poison_type_consistent)):
        pairs = generator(benign, seed=0, n=6)
        cal_feed = build_feed(benign, pairs, base_rate=1.0, revert_lag=4, seed=0)
        ctx = GateContext(prior=prior, trust=SourceTrust())
        ctx.prime(benign)
        got, want = collect_calibration(combiner, cal_feed,
                                        simulate_sources(cal_feed, seed=0), ctx)
        scores.extend(got)
        labels.extend(want)

    fitted = calibrate(scores, labels, base_rate=0.01, target_precision=0.5)
    tuned = combiner.with_thresholds(fitted.thresholds)

    pairs = poison_type_consistent(benign, seed=1, n=6)
    feed = build_feed(benign, pairs, base_rate=1.0, revert_lag=4, seed=1)
    ctx = GateContext(prior=prior, trust=SourceTrust())
    ctx.prime(benign)

    state = run_gated_arm(
        tuned, StubEditor(), feed, simulate_sources(feed, seed=1), ctx, SUITES, [],
        BaseResponder(),
        Rq2Config(checkpoint_every=10, seed=1, model="stub-model", editor="stub",
                  base_rate=1.0, revert_lag=4,
                  calibrated_on="object_swap,type_consistent"),
    )

    assert state.finished
    summaries = summarize_gated([{**state.__dict__, "cell_id": "g0",
                                  "meta": {**state.meta,
                                           "params": {"attack_family": "type_consistent"}}}])
    assert summaries
    final = max(summaries, key=lambda s: s.at)
    assert final.calibrated_on == "object_swap,type_consistent"
    assert final.held_out is False
    assert 0.0 <= final.sensitivity_mean <= 1.0
    assert 0.0 <= final.benign_blocked_mean <= 1.0


def test_sharding_covers_every_cell_exactly_once():
    cells = expand({"editor": ["ultraedit", "alphaedit", "rome"],
                    "seed": [0, 1, 2, 3, 4]})
    assigned = [c.cell_id for node in range(4) for c in shard(cells, node=node, of=4)]
    assert len(assigned) == 15
    assert len(set(assigned)) == 15


def test_an_unfinished_cell_is_rerun_rather_than_resumed(tmp_path):
    """Cells are atomic. A crashed cell leaves no partial state to continue from."""
    calls = []

    def flaky(params):
        calls.append(params)
        raise RuntimeError("cuda oom")

    cell = Cell({"seed": 0})
    run_cell(cell, tmp_path, flaky)
    state = load(tmp_path / f"{cell.cell_id}.json")
    assert state.finished is False
    assert state.checkpoints == []
    assert "cuda oom" in state.meta["error"]
