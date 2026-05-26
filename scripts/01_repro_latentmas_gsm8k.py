"""Phase 0 gate: reproduce LatentMAS GSM8K accuracy on Qwen3-14B.

Strategy: shell out to the vendored LatentMAS run.py with the chosen config.
Capture its final JSON line and save under results/phase0_baseline.json.

Run this on Kaggle with GPU T4 x2 (Qwen3-14B in bf16 needs ~28 GB combined
across the two T4s via tensor parallelism). Locally on Windows this script
detects the environment and prints a skip message.

Gate: accuracy >= 0.65 on 250 GSM8K test problems.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

# Self-contained bootstrap: works without `pip install -e .`
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from u_jepa.config import QwenBaselineConfig
from u_jepa.util.env import detect, prepare

JSON_LINE_RE = re.compile(r"^\{.*\}\s*$")


def build_cmd(vendored_dir, cfg: QwenBaselineConfig) -> list[str]:
    cmd = [
        sys.executable,
        str(vendored_dir / "run.py"),
        "--method", "latent_mas",
        "--model_name", cfg.model_name,
        "--task", cfg.task,
        "--prompt", cfg.prompt,
        "--max_samples", str(cfg.max_samples),
        "--generate_bs", str(cfg.generate_bs),
        "--latent_steps", str(cfg.latent_steps),
        "--use_vllm",
        "--enable_prefix_caching",
        "--latent_space_realign",
        "--tensor_parallel_size", str(cfg.tensor_parallel_size),
        "--gpu_memory_utilization", str(cfg.gpu_memory_utilization),
    ]
    return cmd


def extract_final_json(stdout: str) -> dict | None:
    """vendored run.py prints a final json.dumps(...) line. Find the last one."""
    last = None
    for line in stdout.splitlines():
        line = line.strip()
        if JSON_LINE_RE.match(line):
            try:
                obj = json.loads(line)
                if "accuracy" in obj:
                    last = obj
            except json.JSONDecodeError:
                pass
    return last


def main() -> int:
    env = prepare()
    print(f"env: {env.name}, repo_root: {env.repo_root}")

    cfg = QwenBaselineConfig()
    if env.is_kaggle and "TENSOR_PARALLEL_SIZE" in os.environ:
        cfg = QwenBaselineConfig(tensor_parallel_size=int(os.environ["TENSOR_PARALLEL_SIZE"]))

    if not env.can_run_vllm:
        print(f"[skip] env={env.name} cannot run vLLM (Linux + CUDA required).")
        print("Use kaggle/notebooks/phase0_baseline.py on Kaggle GPU T4 x2 instead.")
        return 0

    vendored = env.repo_root / "vendored" / "LatentMAS"
    if not (vendored / "run.py").exists():
        print(f"FAIL: vendored LatentMAS missing at {vendored}")
        return 1

    cmd = build_cmd(vendored, cfg)
    print("Running:", " ".join(cmd))
    log_path = env.results_dir / "phase0_baseline.log"

    proc = subprocess.run(cmd, cwd=str(vendored), capture_output=True, text=True)
    log_path.write_text(
        f"=== exit code {proc.returncode} ===\n"
        f"=== stdout ===\n{proc.stdout}\n"
        f"=== stderr ===\n{proc.stderr}\n"
    )

    if proc.returncode != 0:
        print(f"FAIL: run.py exited {proc.returncode}. See {log_path}")
        return proc.returncode

    final = extract_final_json(proc.stdout)
    if final is None:
        print(f"FAIL: no accuracy JSON in stdout. See {log_path}")
        return 1

    result = {
        "phase": 0,
        "stage": "latentmas_baseline",
        "config": asdict(cfg),
        **final,
    }
    out_json = env.results_dir / "phase0_baseline.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    acc = float(result.get("accuracy", 0))
    if acc < 0.65:
        print(f"GATE FAIL: accuracy {acc:.3f} < 0.65")
        return 2
    print(f"GATE PASS: accuracy {acc:.3f} >= 0.65")
    return 0


if __name__ == "__main__":
    sys.exit(main())
