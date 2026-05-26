# Running U-JEPA on Kaggle

Heavy compute (LatentMAS reproduction, continual training, vision-bridge alignment, full ablations) runs on Kaggle GPUs because vLLM is Linux-only and Qwen3-14B does not fit on the local RTX 4060.

## One-time setup

1. **Make a Kaggle account.** Phone-verify it so you can enable internet inside notebooks. Internet is required to clone the GitHub repo and pull HuggingFace models the first time.

2. **Add your HuggingFace token as a Kaggle Secret.** Settings tab inside a notebook, Add-ons, Secrets, name it `HF_TOKEN`. Then the notebook reads it via `from kaggle_secrets import UserSecretsClient; HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")`.

3. **Install the Kaggle CLI locally** so you can push notebooks and pull results without clicking through the web UI:

```bash
pip install kaggle
# Then put kaggle.json (API token from kaggle.com/settings) at:
#   Windows: C:\Users\<you>\.kaggle\kaggle.json
#   Linux/Mac: ~/.kaggle/kaggle.json
```

## Per-phase notebook template

Each phase has a corresponding notebook under `kaggle/notebooks/`. They all follow the same shape:

```python
# Cell 1: clone repo + install deps
!git clone https://github.com/kartikshirode/U-JEPA.git
%cd U-JEPA
!pip install -q -r requirements-kaggle.txt
!pip install -q -e .

# Cell 2: HF auth
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

# Cell 3: run the phase script
!python scripts/01_repro_latentmas_gsm8k.py
```

## Settings to set in every notebook

- **Accelerator: GPU T4 x1** (or P100 if available). Phase 3+ can use T4 x2 for V-JEPA training.
- **Internet: On** (required for clone + HF download).
- **Persistence: Files only** (so /kaggle/working survives between sessions; HF cache is preserved this way).

## Outputs

Every script writes JSON results to `/kaggle/working/results/` and checkpoints to `/kaggle/working/checkpoints/`. After the notebook finishes, click "Save Version" then "Quick Save" to commit the output as a new Kaggle Dataset version. From local you can pull it with:

```bash
kaggle kernels output <your-username>/<notebook-slug> -p ./results
```

## Pre-staged model caches (recommended)

Re-downloading Qwen3-14B (~28 GB) each session burns half a session. Better: do it once in a setup notebook that downloads + saves to /kaggle/working/hf_cache, then commit as a Kaggle Dataset. Future notebooks attach that dataset under `/kaggle/input/u-jepa-hf-cache` and copy or symlink it as the HF cache.

Datasets to pre-stage:
- `Qwen/Qwen3-14B-Instruct`
- `microsoft/Phi-3.5-mini-instruct`
- `facebook/vjepa2-vitl-fpc64-256`

## Session limit reminders

Kaggle sessions die at 9 hours. Every long-running script must:
- Checkpoint adapter weights every N steps
- Save running results to `/kaggle/working/results/<phase>_partial.json`
- Resume from last checkpoint if `results/<phase>_partial.json` exists
