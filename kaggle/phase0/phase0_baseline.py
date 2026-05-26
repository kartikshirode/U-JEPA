"""Kaggle notebook body for Phase 0: reproduce LatentMAS on Qwen3-14B-NF4.

To run: create a Kaggle notebook with GPU T4 x1, Internet On, paste the
cells below in order. Or use jupytext to convert this .py into .ipynb.

Cells are delimited by `# %%`.
"""

# %% [markdown]
# # U-JEPA Phase 0: LatentMAS baseline on Qwen3-14B-NF4
#
# Gate: GSM8K accuracy >= 65 percent at >= 10 tok/s. Output goes to
# /kaggle/working/results/phase0_baseline.json.

# %% Cell 1: clone repo and install deps
import subprocess, os, sys
if not os.path.exists("/kaggle/working/U-JEPA"):
    subprocess.run(["git", "clone", "https://github.com/kartikshirode/U-JEPA.git",
                    "/kaggle/working/U-JEPA"], check=True)
os.chdir("/kaggle/working/U-JEPA")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                "requirements-kaggle.txt"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)

# %% Cell 2: HuggingFace auth + cache setup
import os
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as e:
    print(f"Warning: no HF_TOKEN secret found ({e}). Public models will still load.")
os.environ.setdefault("HF_HOME", "/kaggle/working/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/kaggle/working/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/kaggle/working/hf_cache")

# %% Cell 3: env smoke
import torch
print(f"torch {torch.__version__}, cuda {torch.version.cuda}, GPUs: {torch.cuda.device_count()}")
print(f"GPU 0: {torch.cuda.get_device_name(0)}, "
      f"{torch.cuda.get_device_properties(0).total_memory // (1024**2)} MiB")

# %% Cell 4: run the Phase 0 reproduction script
import subprocess, sys
subprocess.run([sys.executable, "scripts/01_repro_latentmas_gsm8k.py"], check=True)

# %% Cell 5: show results
import json
from pathlib import Path
p = Path("/kaggle/working/results/phase0_baseline.json")
if p.exists():
    print(json.dumps(json.loads(p.read_text()), indent=2))
else:
    print("No results file yet")
