import pytest
from u_jepa_v3.runs.state import RunState, is_finished, load, save


def test_round_trips(tmp_path):
    p = tmp_path / "cell.json"
    s = RunState(cell_id="abc", checkpoints=[{"at": 10, "efficacy": 0.5}],
                 finished=False, meta={"editor": "stub"})
    save(s, p)
    assert load(p) == s


def test_state_carries_no_resume_counter():
    # Resuming mid-cell would continue from the wrong model, so the field that
    # made it possible is deliberately absent.
    assert "n_applied" not in RunState.__dataclass_fields__


def test_meta_and_checkpoints_default_to_empty(tmp_path):
    p = tmp_path / "cell.json"
    save(RunState(cell_id="abc"), p)
    loaded = load(p)
    assert loaded.meta == {} and loaded.checkpoints == []


def test_write_leaves_no_temp_file(tmp_path):
    p = tmp_path / "cell.json"
    save(RunState("abc"), p)
    assert list(tmp_path.glob("*.tmp")) == []


def test_existing_file_survives_a_failed_write(tmp_path):
    p = tmp_path / "cell.json"
    save(RunState("abc", checkpoints=[{"at": 5}]), p)
    with pytest.raises(TypeError):
        save(RunState("abc", checkpoints=[{"bad": {1, 2}}]), p)
    assert load(p).checkpoints == [{"at": 5}]


def test_is_finished_false_for_missing_file(tmp_path):
    assert is_finished(tmp_path / "nope.json") is False


def test_is_finished_true_only_when_the_flag_is_set(tmp_path):
    p = tmp_path / "cell.json"
    save(RunState("abc", finished=False), p)
    assert is_finished(p) is False
    save(RunState("abc", finished=True), p)
    assert is_finished(p) is True


def test_is_finished_false_for_a_corrupt_file(tmp_path):
    p = tmp_path / "cell.json"
    p.write_text("{not json", encoding="utf-8")
    assert is_finished(p) is False
