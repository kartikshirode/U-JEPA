"""The ledger, and what replaying it without the bad entries costs."""
from __future__ import annotations

import pytest

from u_jepa_v3.editors.stub import StubEditor
from u_jepa_v3.gate.rollback import ShadowLedger, audit_rollback
from u_jepa_v3.schema import EditCandidate, EditKind


def cand(subj, obj):
    return EditCandidate(
        subject_id=subj, subject=f"S{subj}", relation_id="P26", relation="spouse",
        object_id=None, object=obj, prompt=f"Who is the spouse of S{subj}?",
        kind=EditKind.REVISION, source="wikibigedit", timestep=0,
        is_adversarial=False, risk_category=None, n_hops=1,
    )


def poison(subj, obj):
    return EditCandidate(
        subject_id=subj, subject=f"S{subj}", relation_id="P26", relation="spouse",
        object_id=None, object=obj, prompt=f"Who is the spouse of S{subj}?",
        kind=EditKind.REVISION, source="type_consistent", timestep=0,
        is_adversarial=True, risk_category="misinformation", n_hops=1,
    )


def filled(n_benign=20):
    ledger = ShadowLedger()
    for i in range(n_benign):
        ledger.record(f"benign-{i}", i, cand(f"Q{i}", f"true{i}"))
    ledger.record("poison-0", n_benign, poison("QP", "false"))
    return ledger


def test_the_ledger_keeps_the_order_edits_were_applied_in():
    ledger = filled(5)
    assert [e.entry_id for e in ledger.entries][:3] == ["benign-0", "benign-1", "benign-2"]
    assert len(ledger) == 6


def test_the_replay_plan_leaves_out_exactly_the_dropped_entries():
    ledger = filled(5)
    plan = ledger.replay_plan({"poison-0"})
    assert len(plan) == 5
    assert all(not c.is_adversarial for c in plan)


def test_dropping_something_that_was_never_applied_raises():
    with pytest.raises(KeyError, match="never admitted"):
        filled(3).replay_plan({"refused-9"})


def test_the_cost_is_the_whole_ledger_less_the_drops():
    """Replay starts from the base model, so earlier edits are re-applied too."""
    ledger = filled(100)
    assert ledger.cost_of_dropping({"poison-0"}) == 100


def test_the_position_of_an_entry_is_recoverable():
    ledger = filled(4)
    assert ledger.position_of("poison-0") == 4
    with pytest.raises(KeyError):
        ledger.position_of("nope")


def test_replaying_without_the_poison_leaves_no_trace_of_it():
    ledger = filled(10)
    bad = poison("QP", "false")
    audit = audit_rollback(StubEditor, ledger, {"poison-0"}, [bad],
                           [cand("Q0", "true0")])
    assert audit.residual_direct == 0.0
    assert audit.residual_leading == 0.0
    assert audit.benign_efficacy == 1.0
    assert audit.n_replayed == 10
    assert audit.n_dropped == 1


def test_replaying_with_the_poison_still_in_shows_it_coming_back():
    """The control. A zero residual means the drop worked, not that the probe is blind."""
    ledger = filled(10)
    bad = poison("QP", "false")
    audit = audit_rollback(StubEditor, ledger, set(), [bad], [cand("Q0", "true0")])
    assert audit.residual_direct == 1.0
    assert audit.n_replayed == 11


def test_the_audit_reports_a_per_edit_cost():
    audit = audit_rollback(StubEditor, filled(10), {"poison-0"}, [], [])
    assert audit.seconds >= 0.0
    assert audit.seconds_per_edit == pytest.approx(audit.seconds / 10)


def test_an_empty_replay_does_not_divide_by_zero():
    ledger = ShadowLedger()
    ledger.record("only", 0, cand("Q1", "x"))
    audit = audit_rollback(StubEditor, ledger, {"only"}, [], [])
    assert audit.n_replayed == 0
    assert audit.seconds_per_edit == 0.0


def test_a_zero_batch_size_is_refused():
    with pytest.raises(ValueError, match="batch_size"):
        audit_rollback(StubEditor, filled(2), set(), [], [], batch_size=0)
