import pytest
from u_jepa_v3 import env


@pytest.mark.parametrize(
    "capability,expected",
    [((7, 5), "fp16"), ((8, 0), "bf16"), ((9, 0), "bf16"), (None, "fp32")],
)
def test_dtype_follows_capability(capability, expected, monkeypatch):
    monkeypatch.delenv("U_JEPA_V3_DTYPE", raising=False)
    assert env.preferred_dtype_str(capability) == expected


def test_env_var_overrides_capability(monkeypatch):
    monkeypatch.setenv("U_JEPA_V3_DTYPE", "fp32")
    assert env.preferred_dtype_str((9, 0)) == "fp32"


def test_rejects_unknown_dtype_override(monkeypatch):
    monkeypatch.setenv("U_JEPA_V3_DTYPE", "int4")
    with pytest.raises(ValueError, match="int4"):
        env.preferred_dtype_str((9, 0))


def test_run_root_prefers_env_and_creates_it(monkeypatch, tmp_path):
    monkeypatch.setenv("U_JEPA_V3_RUN_DIR", str(tmp_path / "runs"))
    assert env.run_root() == tmp_path / "runs"
    assert env.run_root().is_dir()


def test_summary_has_required_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("U_JEPA_V3_RUN_DIR", str(tmp_path))
    monkeypatch.delenv("U_JEPA_V3_DTYPE", raising=False)
    keys = env.summarize().as_dict()
    for key in ("python", "torch", "cuda_available", "device_count",
                "capability", "dtype", "run_root"):
        assert key in keys
