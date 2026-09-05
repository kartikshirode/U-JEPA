import pytest
from u_jepa_v3.editors import registry
from u_jepa_v3.editors.base import Editor, Responder
from u_jepa_v3.editors.stub import StubEditor
from u_jepa_v3.schema import EditCandidate, EditKind


def cand(i=0, obj="o"):
    return EditCandidate(
        subject_id=f"Q{i}", subject="s", relation_id="P1", relation="r",
        object_id=None, object=obj, prompt=f"p{i}", kind=EditKind.REVISION,
        source="test", timestep=0, is_adversarial=False,
        risk_category=None, n_hops=1,
    )


def test_stub_satisfies_both_protocols():
    e = StubEditor()
    assert isinstance(e, Editor)
    assert isinstance(e.responder(), Responder)


def test_responder_reflects_an_applied_edit():
    e = StubEditor()
    assert e.responder().answer(["p0"]) == ["<unedited>"]
    e.apply([cand(0, "Paris Fury")])
    assert e.responder().answer(["p0"]) == ["Paris Fury"]


def test_a_later_edit_to_the_same_prompt_wins():
    e = StubEditor()
    e.apply([cand(0, "first")])
    e.apply([cand(0, "second")])
    assert e.responder().answer(["p0"]) == ["second"]


def test_a_failed_edit_does_not_change_the_answer():
    e = StubEditor(fail_keys={"Q0:P1"})
    e.apply([cand(0, "nope")])
    assert e.responder().answer(["p0"]) == ["<unedited>"]


def test_stub_records_what_it_applied():
    e = StubEditor()
    e.apply([cand(0), cand(1)])
    assert [c.subject_id for c in e.applied] == ["Q0", "Q1"]


def test_failure_is_reported_per_candidate():
    results = StubEditor(fail_keys={"Q1:P1"}).apply([cand(0), cand(1)])
    assert [r.succeeded for r in results] == [True, False]
    assert results[1].error == "stub-forced failure"


def test_registry_builds_and_lists():
    registry.register("stub", StubEditor)
    assert isinstance(registry.build("stub"), StubEditor)
    assert "stub" in registry.available()


def test_registry_rejects_unknown_name():
    with pytest.raises(KeyError, match="nope"):
        registry.build("nope")
