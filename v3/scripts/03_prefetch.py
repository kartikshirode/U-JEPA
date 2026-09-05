"""Pull every weight and dataset onto shared storage. Login node only.

Baramati's compute nodes have no outbound network; the login node does. So
everything a job needs gets fetched here first, into a cache on /home that the
jobs read with HF_HUB_OFFLINE=1. A job that tries to download instead fails
several minutes in, after the queue wait.

    export HF_HOME=/home/$USER/.cache/huggingface
    python v3/scripts/03_prefetch.py --model meta-llama/Llama-3.2-3B-Instruct

Gated repositories need `huggingface-cli login` first. Llama is gated; Qwen is
not.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from u_jepa_v3.cluster import KNOWN_MODELS
from u_jepa_v3.data.wikibigedit import REPO_ID, TIMESTEP_FILES


def prefetch_model(name: str) -> bool:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print(f"\n{name}")
    try:
        config = AutoConfig.from_pretrained(name)
        print(f"  config     {config.num_hidden_layers} layers, "
              f"hidden {config.hidden_size}, intermediate {config.intermediate_size}")
        AutoTokenizer.from_pretrained(name)
        print("  tokenizer  cached")
        # Weights only, never onto a device. The login node has no GPU.
        AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto",
                                             low_cpu_mem_usage=True)
        print("  weights    cached")
        return True
    except Exception as exc:
        print(f"  FAILED {type(exc).__name__}: {exc}")
        return False


def prefetch_dataset() -> bool:
    from huggingface_hub import hf_hub_download

    print(f"\n{REPO_ID}")
    ok = True
    for name in TIMESTEP_FILES:
        try:
            path = hf_hub_download(REPO_ID, name, repo_type="dataset")
            size = Path(path).stat().st_size / 1e6
            print(f"  {name}  {size:.1f} MB")
        except Exception as exc:
            print(f"  {name}  FAILED {type(exc).__name__}: {exc}")
            ok = False
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", action="append", default=[],
                        help="repeatable; defaults to the two models that fit a slice")
    parser.add_argument("--skip-dataset", action="store_true")
    args = parser.parse_args(argv)

    models = args.model or ["meta-llama/Llama-3.2-3B-Instruct",
                            "Qwen/Qwen2.5-1.5B-Instruct"]
    cache = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")
    print(f"HF cache: {cache or 'default (~/.cache/huggingface)'}")
    if cache and cache.startswith("/tmp"):
        print("  WARNING that cache is node local. Compute nodes will not see it.")

    failures = []
    for name in models:
        if name not in KNOWN_MODELS:
            print(f"\n{name}\n  note: no ModelSpec, so the fit check cannot size it")
        if not prefetch_model(name):
            failures.append(name)

    if not args.skip_dataset and not prefetch_dataset():
        failures.append(REPO_ID)

    print("\n" + "=" * 60)
    if failures:
        print("failed: " + ", ".join(failures))
        print("Gated repos need `huggingface-cli login` and an accepted licence.")
        return 1
    print("everything cached. Jobs can now run with HF_HUB_OFFLINE=1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
