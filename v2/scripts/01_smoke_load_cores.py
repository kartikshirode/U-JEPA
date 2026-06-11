"""Phase 0 step 2: load both frozen cores and confirm VRAM headroom.

Loads GPT-2-XL and Qwen2.5-1.5B-Instruct one at a time (NOT together, to keep
peak VRAM honest), runs a short generation, and reports VRAM peak per model.

Skipped paths:
- `--only gpt2-xl` or `--only qwen2.5-1.5b` to load just one.
- `--nf4` to load Qwen in NF4 (only meaningful for later adapter work; cores
  fit in fp16 on T4).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from u_jepa_v2 import env, persistence  # noqa: E402
from u_jepa_v2.models import core  # noqa: E402


PROMPT = "Q: What city is the Eiffel Tower in? A:"


def _reset_peak():
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_gb() -> float | None:
    import torch
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def smoke_one(name: str, *, dtype: str, nf4: bool, cache_dir: str | None) -> dict:
    import torch
    _reset_peak()
    t0 = time.monotonic()
    model, tok = core.load_core(name, dtype=dtype, nf4=nf4, cache_dir=cache_dir)
    load_s = time.monotonic() - t0

    device = next(model.parameters()).device
    inputs = tok(PROMPT, return_tensors="pt").to(device)

    t1 = time.monotonic()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=12,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    gen_s = time.monotonic() - t1
    text = tok.decode(out[0], skip_special_tokens=True)

    record = {
        "model": name,
        "dtype": dtype,
        "nf4": nf4,
        "load_seconds": round(load_s, 2),
        "generate_seconds": round(gen_s, 2),
        "peak_vram_gb": round(_peak_gb() or 0.0, 3),
        "output": text,
    }

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(core.CORE_MODELS), default=None)
    ap.add_argument("--nf4", action="store_true")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    print(json.dumps(env.summary().as_dict(), indent=2, sort_keys=True))
    targets = [args.only] if args.only else list(core.CORE_MODELS)
    print(f"\n[plan] smoke-loading: {targets}\n", flush=True)

    results: list[dict] = []
    for name in targets:
        print(f"[load] {name} (dtype={args.dtype}, nf4={args.nf4})", flush=True)
        rec = smoke_one(name, dtype=args.dtype, nf4=args.nf4, cache_dir=args.cache_dir)
        print(json.dumps(rec, indent=2), flush=True)
        results.append(rec)

    payload = {"env": env.summary().as_dict(), "results": results}
    if args.out:
        persistence.save_json(args.out, payload)
        print(f"\n[ok] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
