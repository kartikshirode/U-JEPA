import os
import pytest
from u_jepa_v3.editors.base import Editor
from u_jepa_v3.editors.easyedit_adapter import EasyEditAdapter, SUPPORTED_METHODS
from u_jepa_v3.schema import EditCandidate, EditKind


def cand(i=0, obj="Paris Fury"):
    return EditCandidate(
        subject_id=f"Q{i}", subject="Tyson Fury", relation_id="P26", relation="spouse",
        object_id=None, object=obj, prompt="Who is the spouse of Tyson Fury?",
        kind=EditKind.REVISION, source="test", timestep=0,
        is_adversarial=False, risk_category=None, n_hops=1,
    )


class FakeEasyEditor:
    """Stands in for EasyEdit BaseEditor. Returns a new model object each edit."""
    def __init__(self):
        self.calls = []
        self.model_counter = 0

    def edit(self, **payload):
        self.calls.append(payload)
        self.model_counter += 1
        return {"metrics": []}, f"model-v{self.model_counter}", None


def test_adapter_satisfies_the_protocol():
    assert isinstance(EasyEditAdapter("ultraedit", "x.yaml"), Editor)


def test_name_includes_the_method():
    assert EasyEditAdapter("alphaedit", "x.yaml").name == "easyedit:alphaedit"


def test_rejects_unsupported_method():
    with pytest.raises(ValueError, match="notamethod"):
        EasyEditAdapter("notamethod", "x.yaml")


def test_payload_maps_prompt_subject_and_object():
    a = EasyEditAdapter("ultraedit", "x.yaml")
    payload = a.to_easyedit_payload([cand(0), cand(1, "Someone Else")])
    assert payload["target_new"] == ["Paris Fury", "Someone Else"]
    assert payload["subject"] == ["Tyson Fury", "Tyson Fury"]
    assert payload["sequential_edit"] is True
    lengths = {len(v) for v in payload.values() if isinstance(v, list)}
    assert lengths == {2}


def test_apply_keeps_the_returned_model(monkeypatch):
    a = EasyEditAdapter("ultraedit", "x.yaml")
    fake = FakeEasyEditor()
    monkeypatch.setattr(a, "_ensure_editor", lambda: fake)
    a.apply([cand(0)])
    assert a.edited_model == "model-v1"


def test_each_edit_continues_from_the_previous_model(monkeypatch):
    a = EasyEditAdapter("ultraedit", "x.yaml")
    fake = FakeEasyEditor()
    monkeypatch.setattr(a, "_ensure_editor", lambda: fake)
    a.apply([cand(0)])
    a.apply([cand(1)])
    assert a.edited_model == "model-v2"


def test_responder_before_any_edit_raises_rather_than_silently_reading_the_base():
    a = EasyEditAdapter("ultraedit", "x.yaml")
    with pytest.raises(RuntimeError, match="no model"):
        a.responder()


def test_a_failed_batch_reports_per_candidate_and_keeps_the_old_model(monkeypatch):
    a = EasyEditAdapter("ultraedit", "x.yaml")

    class Boom:
        def edit(self, **payload):
            raise RuntimeError("cuda oom")

    monkeypatch.setattr(a, "_ensure_editor", lambda: Boom())
    results = a.apply([cand(0), cand(1)])
    assert [r.succeeded for r in results] == [False, False]
    assert "cuda oom" in results[0].error
    assert a.edited_model is None


def test_empty_batch_yields_empty_results():
    assert EasyEditAdapter("ultraedit", "x.yaml").apply([]) == []


@pytest.mark.skipif(
    os.environ.get("U_JEPA_V3_RUN_GPU") != "1",
    reason="needs a GPU and easyeditor; set U_JEPA_V3_RUN_GPU=1",
)
def test_real_edit_changes_the_answer():
    a = EasyEditAdapter("ultraedit", os.environ["U_JEPA_V3_HPARAMS"])
    target = cand(0, "Paris Fury")
    a.apply([target])
    assert "paris" in a.responder().answer([target.prompt])[0].lower()
