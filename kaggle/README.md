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

## Per-phase notebook layout

Each phase has its own folder under `kaggle/` (e.g. `kaggle/phase0/`, `kaggle/phase1/`) containing:
- `<phase>.ipynb`: the actual notebook pushed to Kaggle
- `kernel-metadata.json`: kernel config used by `kaggle kernels push`

Every notebook follows the same shape:

1. Cleanup cell that removes any stale `/kaggle/working/hf_cache` from a prior run and forces `/tmp/hf_cache` to exist (HF cache cannot live in `/kaggle/working` because of the 20 GB quota).
2. Setup cell: `git clone` the repo (or `git pull` if already present), `pip install -r requirements-kaggle.txt`, `pip install -e .`. Installs run without `-q` so pip errors surface in the kernel log.
3. HF auth cell: pull `HF_TOKEN` from Kaggle secrets if present, and pin `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE` to `/tmp/hf_cache`.
4. GPU smoke cell: print torch + CUDA versions and per-GPU memory, assert at least one CUDA device.
5. Run cell: invoke the phase script via `subprocess.run` with the cache env vars propagated.
6. Show-results cell: print the JSON output and tail the log if one exists.

## Settings already encoded in kernel-metadata.json

- **Accelerator: NvidiaTeslaT4** (canonical name; the Kaggle UI exposes this as T4 x2).
- **Internet: On** (`enable_internet: true`).
- **Privacy: Private** (`is_private: true`); the source GitHub repo must be public for `git clone` to work without auth.
- **GPU: enabled** (`enable_gpu: true`); the `--accelerator` CLI flag alone is not enough.

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
