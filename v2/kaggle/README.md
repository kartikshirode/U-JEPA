# Running U-JEPA v2 on Kaggle

Heavy runs go to Kaggle's free T4. The local RTX 4060 is for unit tests and
small smoke checks. v2 cores are small (GPT-2-XL 1.5B, Qwen2.5-1.5B), so a
single T4 is plenty; there is no need for the T4 x2 setup v1 used for Qwen3-14B.

## Account

The kernel metadata points at `kartikshirode`. The repo-root `kaggle.json` is a
different account (`chinmayishirode`) and is gitignored; make sure your local
Kaggle CLI is configured for `kartikshirode` before pushing
(`~/.kaggle/kaggle.json`), or these pushes land on the wrong account.

## Per-spike / per-phase layout

Each notebook lives in its own folder with a `kernel-metadata.json`:

```
kaggle/
  build_f1_notebook.py        generator for the F1 notebook (keeps the ipynb valid JSON)
  f1_spike/
    f1_spike.ipynb            the notebook pushed to Kaggle
    kernel-metadata.json      kernel config for `kaggle kernels push`
```

Notebooks are generated from a Python builder rather than hand-edited, because a
hand-written ipynb is easy to corrupt into invalid JSON (v1 lost a push that
way). Edit `build_*.py`, re-run it, then push.

## The standard cell shape

1. Cleanup: drop any stale `/kaggle/working/hf_cache`, force `/tmp/hf_cache`
   (HF cache cannot sit in `/kaggle/working` because of the 20 GB quota).
2. Setup: clone the repo (and any upstream like EasyEdit), `pip install -e v2`.
   Installs run without `-q` so dependency conflicts show in the log.
3. HF auth: pull `HF_TOKEN` from Kaggle secrets if present; pin the cache env
   vars. Public models load without a token.
4. Env gate: run `scripts/00_smoke_env.py`. It now exits non-zero if the GPU is
   off on Kaggle, so a forgotten accelerator fails fast instead of burning a
   session.
5. Run: shell out to the phase/spike script with `--out` under
   `/kaggle/working/results/`.
6. Show: print the result JSON.

## Pushing and pulling

```bash
# from v2/kaggle/<folder>/
kaggle kernels push
# after it finishes, pull the output JSON back to local
kaggle kernels output kartikshirode/<slug> -p ../../results
```

## Session limits

9 hour hard cap, 30 h/week. The F1 spike is minutes, not hours, so it does not
need checkpoint/resume. Later phases that approach the cap must save resumable
state through `u_jepa_v2.persistence` and reload from `/kaggle/input/<slug>`.
