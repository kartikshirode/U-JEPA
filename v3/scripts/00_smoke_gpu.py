"""First thing to run on the HPC box. Answers the open hardware questions.

The v3 design assumes no single job exceeds 141 GB because the topology of the 4
H200s was never confirmed. This reports what is actually there, so that
assumption can be dropped or kept on evidence rather than caution.

Run:  python v3/scripts/00_smoke_gpu.py

Exits non-zero when something the plan depends on is missing, so it can gate a
job script.
"""
from __future__ import annotations

import subprocess
import sys


def section(title: str) -> None:
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


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

    count = torch.cuda.device_count()
    print(f"  device count     {count}")
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        print(f"    [{i}] {props.name}  "
              f"{props.total_memory / 1e9:.1f} GB  "
              f"capability {props.major}.{props.minor}")
    return True, count


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


def check_topology(count: int) -> None:
    section("topology (decides whether the 141 GB per-job cap can be dropped)")
    if count < 2:
        print("  only one visible device, nothing to pair")
        return

    import torch

    peer = {}
    for i in range(count):
        for j in range(count):
            if i < j:
                peer[(i, j)] = torch.cuda.can_device_access_peer(i, j)
    print("  peer access:")
    for (i, j), ok in peer.items():
        print(f"    {i} <-> {j}  {'yes' if ok else 'no'}")

    try:
        out = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True,
                             text=True, timeout=30)
        print("\n  nvidia-smi topo -m:")
        for line in out.stdout.splitlines()[:12]:
            print(f"    {line}")
        if "NV" in out.stdout:
            print("\n  NVLink present. The 70B arm and larger single jobs are on the table;")
            print("  record this in the spec's open items before relying on it.")
        else:
            print("\n  No NVLink markers. Keep the 141 GB per-job cap.")
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"  nvidia-smi unavailable ({type(exc).__name__}); peer access above is the signal")


def check_easyedit() -> bool:
    section("easyedit")
    try:
        import easyeditor  # noqa: F401
        print("  easyeditor imports")
        return True
    except ImportError as exc:
        print(f"  NOT INSTALLED ({exc})")
        print("  Stage 1 needs it. Install into this environment, then fetch hparams")
        print("  for ultraedit, alphaedit, rome and memit against the chosen 8B model.")
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
        print(f"  Build them and point {PROBE_DIR_ENV} at the directory.")
        return False


def main() -> int:
    ok, count = check_torch()
    if not ok:
        return 2

    check_dtype()
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
