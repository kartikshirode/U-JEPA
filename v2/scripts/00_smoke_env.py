"""Phase 0 step 1: print the env summary and assert what the build plan requires.

Run me first on every fresh Kaggle session. Fails loudly if assumptions break.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from u_jepa_v2 import env  # noqa: E402


def main() -> int:
    s = env.summary()
    print(json.dumps(s.as_dict(), indent=2, sort_keys=True))

    if s.device == "cpu":
        print("\n[warn] running on CPU. Phase 0 will be slow; that's expected for local smoke.")
        return 0

    if s.gpu_name and "T4" in s.gpu_name and s.native_bf16:
        print("\n[fail] T4 should not report native bf16. Hardware/driver detection is wrong.", file=sys.stderr)
        return 2

    if s.gpu_vram_gb is not None and s.gpu_vram_gb < 14:
        print(f"\n[warn] GPU VRAM = {s.gpu_vram_gb:.1f} GB. Plan assumes ~15 GB (T4).", file=sys.stderr)

    print("\n[ok] env check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
