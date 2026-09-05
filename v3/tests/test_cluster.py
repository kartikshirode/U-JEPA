"""The memory arithmetic that decides which arms are runnable on MIG slices."""
from __future__ import annotations

import pytest

from u_jepa_v3 import cluster


def test_an_8b_model_with_a_banded_method_does_not_fit_an_18gb_slice():
    """The finding that reshaped the pilot grid, kept as a test so it stays true."""
    report = cluster.plan_fit("meta-llama/Meta-Llama-3-8B-Instruct", "alphaedit",
                              "bf16", "1g.18gb")
    assert report.fits is False
    assert report.weights_gb == pytest.approx(16.06, abs=0.1)
    assert any("does not fit" in note for note in report.notes)


def test_a_3b_model_fits_with_room_to_spare():
    report = cluster.plan_fit("meta-llama/Llama-3.2-3B-Instruct", "alphaedit",
                              "bf16", "1g.18gb")
    assert report.fits is True
    assert report.headroom_gb > 2.0


def test_alphaedit_costs_twice_what_memit_costs():
    """It holds a null space projection alongside each covariance."""
    spec = cluster.KNOWN_MODELS["meta-llama/Meta-Llama-3-8B-Instruct"]
    assert (cluster.editor_overhead_gb("alphaedit", spec)
            == pytest.approx(2 * cluster.editor_overhead_gb("memit", spec)))


def test_rome_is_one_layer_and_ultraedit_is_not_counted():
    spec = cluster.KNOWN_MODELS["meta-llama/Llama-3.2-3B-Instruct"]
    assert cluster.editor_overhead_gb("rome", spec) > 0
    assert cluster.editor_overhead_gb("rome", spec) < cluster.editor_overhead_gb("memit", spec)
    assert cluster.editor_overhead_gb("ultraedit", spec) == 0.0


def test_fp32_doubles_the_weights():
    spec = cluster.KNOWN_MODELS["gpt2-xl"]
    assert (cluster.weights_gb(spec, "fp32")
            == pytest.approx(2 * cluster.weights_gb(spec, "bf16")))


def test_a_slice_hands_over_less_than_its_nominal_size():
    assert cluster.slice_budget_gb("1g.18gb") < 18.0
    assert cluster.slice_budget_gb("1g.24gb") > cluster.slice_budget_gb("1g.18gb")


def test_an_unknown_flavour_and_an_unknown_model_both_raise():
    with pytest.raises(KeyError):
        cluster.slice_budget_gb("1g.40gb")
    with pytest.raises(KeyError, match="config.json"):
        cluster.plan_fit("nobody/nothing", "rome", "bf16")


def test_a_thin_margin_is_called_out_rather_than_passed_silently():
    report = cluster.plan_fit("EleutherAI/gpt-j-6b", "rome", "fp16", "1g.18gb")
    assert report.fits is True
    assert 0 < report.headroom_gb < 2.0
    assert any("headroom" in note for note in report.notes)


def test_slurm_context_reads_an_array_task():
    ctx = cluster.slurm_context({
        "SLURM_JOB_ID": "1234", "SLURM_ARRAY_TASK_ID": "5",
        "SLURM_ARRAY_TASK_COUNT": "14", "CUDA_VISIBLE_DEVICES": "0",
        "SLURMD_NODENAME": "aicoeserver03",
    })
    assert ctx.under_slurm and ctx.is_array
    assert (ctx.array_task_id, ctx.array_task_count) == (5, 14)


def test_a_bare_shell_is_not_slurm_and_not_an_array():
    ctx = cluster.slurm_context({})
    assert ctx.under_slurm is False
    assert ctx.is_array is False


def test_a_non_numeric_task_id_does_not_crash_the_worker():
    ctx = cluster.slurm_context({"SLURM_JOB_ID": "7", "SLURM_ARRAY_TASK_ID": "n/a"})
    assert ctx.under_slurm is True
    assert ctx.is_array is False


def test_device_report_is_empty_without_cuda_rather_than_raising():
    assert isinstance(cluster.device_report(), list)
