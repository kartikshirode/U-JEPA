"""First thing to run on the HPC box. Answers the open hardware questions.

The v3 design was written against "4 H200s". Baramati cuts those cards into MIG
slices and schedules the slice, so a job gets 18 GB and not 141. This reports
what is actually visible, then does the memory arithmetic for the planned arms,
so a cell that cannot fit is known before it queues rather than after it OOMs.

Run:  python v3/scripts/00_smoke_gpu.py
Or:   sbatch v3/slurm/00_smoke.slurm

Exits non-zero when something the plan depends on is missing, so it can gate a
job script.
"""
from __future__ import annotations

import subprocess
import sys

# The arms the pilot grid intends to run, as (model, method).
PLANNED = [
    ("meta-llama/Llama-3.2-3B-Instruct", "ultraedit"),
    ("meta-llama/Llama-3.2-3B-Instruct", "alphaedit"),
    ("meta-llama/Llama-3.2-3B-Instruct", "rome"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "alphaedit"),
    ("meta-llama/Meta-Llama-3-8B-Instruct", "alphaedit"),
]


def section(title: str) -> None:
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


def check_slurm() -> None:
    section("slurm context")
    from u_jepa_v3.cluster import slurm_context

    ctx = slurm_context()
    if not ctx.under_slurm:
        print("  not under Slurm. Fine for a login node check; the real run is an array job.")
        return
    for key, value in ctx.as_dict().items():
        print(f"  {key:20} {value}")


def check_torch() -> tuple[bool, int]:
    section("torch and devices")
    try:
        import torch
    except ImportError:
        print("  FAIL torch is not installed")
        return False, 0

    print(f"  torch            {torch.__version__}")
    print(f"  cuda available   {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("  FAIL no CUDA. Nothing in stage 1 can run.")
        return False, 0

    from u_jepa_v3.cluster import device_report

    devices = device_report()
    print(f"  device count     {len(devices)}")
    for device in devices:
        flag = "  <- looks like a MIG slice" if device["looks_like_mig"] else ""
        print(f"    [{device['index']}] {device['name']}  {device['total_gb']} GB  "
              f"capability {device['capability']}{flag}")
    return True, len(devices)


def check_dtype() -> None:
    section("dtype (must be derived, never hardcoded)")
    from u_jepa_v3.env import device_capability, preferred_dtype_str, summarize

    cap = device_capability()
    dtype = preferred_dtype_str(cap)
    print(f"  capability       {cap}")
    print(f"  chosen dtype     {dtype}")
    if cap and cap[0] >= 8 and dtype != "bf16":
        print("  WARN Hopper or Ampere should choose bf16. Check U_JEPA_V3_DTYPE.")
    print(f"  full summary     {summarize().as_dict()}")


def check_fit() -> None:
    """Whether each planned arm fits the slice this job was actually given."""
    section("memory budget for the planned arms")
    from u_jepa_v3.cluster import device_report, plan_fit, slice_budget_gb
    from u_jepa_v3.env import device_capability, preferred_dtype_str

    devices = device_report()
    budget = devices[0]["total_gb"] if devices else slice_budget_gb("1g.18gb") + 0.8
    dtype = preferred_dtype_str(device_capability())
    print(f"  budget from      {'this device' if devices else 'the 1g.18gb flavour'}"
          f" ({budget:.1f} GB nominal), dtype {dtype}\n")

    for model, method in PLANNED:
        report = plan_fit(model, method, dtype, budget)
        verdict = "fits" if report.fits else "DOES NOT FIT"
        print(f"  {verdict:12} {method:10} {model.split('/')[-1]:26} "
              f"needs {report.required_gb:5.1f} of {report.budget_gb:5.1f} GB")
        for note in report.notes:
            print(f"               note: {note}")


def check_topology(count: int) -> None:
    section("topology (does anything larger than one slice have a path)")
    if count < 2:
        print("  one visible device. Under a MIG allocation that is expected, and it")
        print("  means every job is capped at that slice. Nothing may assume otherwise.")
        return

    import torch

    print("  peer access:")
    for i in range(count):
        for j in range(i + 1, count):
            ok = torch.cuda.can_device_access_peer(i, j)
            print(f"    {i} <-> {j}  {'yes' if ok else 'no'}")
    print("  MIG instances do not peer with each other. Two slices are two small")
    print("  GPUs, not one large one.")

    try:
        out = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True,
                             text=True, timeout=30)
        print("\n  nvidia-smi topo -m:")
        for line in out.stdout.splitlines()[:12]:
            print(f"    {line}")
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"  nvidia-smi unavailable ({type(exc).__name__})")


def check_easyedit() -> bool:
    section("easyedit")
    try:
        import easyeditor  # noqa: F401
        print("  easyeditor imports")
        print("  next: python v3/scripts/04_check_hparams.py v3/hparams/")
        return True
    except ImportError as exc:
        print(f"  NOT INSTALLED ({exc})")
        print("  Stage 1 needs it. Install into this environment, then validate the")
        print("  hparams in v3/hparams/ with scripts/04_check_hparams.py.")
        return False


def check_probe_sets() -> bool:
    section("probe sets")
    from u_jepa_v3.experiments.rq1_survival import PROBE_DIR_ENV, _load_suites

    try:
        suites = _load_suites()
        for name, pairs in suites.items():
            print(f"  {name:6} {len(pairs)} pairs")
        return True
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"  NOT READY: {exc}")
        print(f"  Build them on the login node, then point {PROBE_DIR_ENV} at them:")
        print("    python v3/scripts/01_build_probes.py --out ~/probes")
        return False


def main() -> int:
    check_slurm()
    ok, count = check_torch()
    if not ok:
        return 2

    check_dtype()
    check_fit()
    check_topology(count)
    has_editor = check_easyedit()
    has_probes = check_probe_sets()

    section("verdict")
    blockers = []
    if not has_editor:
        blockers.append("easyeditor not installed")
    if not has_probes:
        blockers.append("probe sets not built")

    if blockers:
        print("  Harness is fine. Stage 1 is blocked on:")
        for b in blockers:
            print(f"    - {b}")
        print("\n  The CPU suite still passes without these; run `pytest` in v3/.")
        return 1

    print("  Ready for a stage 1 pilot.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
