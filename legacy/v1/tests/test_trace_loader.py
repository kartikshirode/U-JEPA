"""TRACE loader smoke tests. Marked slow because they download from HF."""
import pytest
from u_jepa.data.trace import load_trace_task, TASK_LOADERS


def test_task_loader_registry_has_phase1_tasks():
    assert "fomc" in TASK_LOADERS
    assert "scienceqa_text" in TASK_LOADERS


def test_unknown_task_name_raises():
    with pytest.raises(KeyError):
        load_trace_task("not_a_real_task", split="train", n=1)


@pytest.mark.slow
def test_fomc_loads_with_expected_labels():
    items = load_trace_task("fomc", split="train", n=10)
    assert len(items) <= 10
    assert all("Stance:" in i["prompt"] for i in items)
    assert all(i["target"] in {"dovish", "hawkish", "neutral"} for i in items)


@pytest.mark.slow
def test_scienceqa_text_skips_image_items():
    items = load_trace_task("scienceqa_text", split="train", n=5)
    assert len(items) <= 5
    assert all("Answer:" in i["prompt"] for i in items)
    assert all(len(i["target"]) == 1 and i["target"].isalpha() for i in items)
