"""Generate f1_spike/f1_spike.ipynb as valid nbformat-4 JSON.

Hand-written notebooks shipped to Kaggle must be real JSON, not pseudo-XML; v1
lost a push to that. This builds the cell list in Python and dumps it, so the
output always parses. Re-run after editing a cell:  python build_f1_notebook.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": list(lines)}


CELLS = [
    md(
        "# U-JEPA v2 - F1 spike: EasyEdit GRACE on GPT-2-XL\n",
        "\n",
        "Resolves open fork F1 (EasyEdit-GRACE vs a hand-rolled codebook). The\n",
        "spike script loads GPT-2-XL with our own loader, scores a tiny set of\n",
        "CounterFact-style edits before and after running GRACE, and writes a\n",
        "verdict JSON. Single T4, internet on. If EasyEdit fights the install,\n",
        "that failure is the result: the verdict comes back HANDROLL.",
    ),
    code(
        "# Cell 1: move the HF cache off the 20 GB /kaggle/working quota.\n",
        "# GPT-2-XL is small (~6 GB fp32, ~3 GB fp16), so this is hygiene, not\n",
        "# strictly required, but keep the pattern that worked in v1.\n",
        "import os, shutil\n",
        "stale = '/kaggle/working/hf_cache'\n",
        "if os.path.isdir(stale):\n",
        "    shutil.rmtree(stale, ignore_errors=True)\n",
        "os.makedirs('/tmp/hf_cache', exist_ok=True)\n",
        "print('hf cache -> /tmp/hf_cache')",
    ),
    code(
        "# Cell 2: clone the repo + EasyEdit, install both.\n",
        "import subprocess, sys, os\n",
        "if not os.path.exists('/kaggle/working/U-JEPA'):\n",
        "    subprocess.run(['git', 'clone', '--depth', '1',\n",
        "                    'https://github.com/kartikshirode/U-JEPA.git',\n",
        "                    '/kaggle/working/U-JEPA'], check=True)\n",
        "if not os.path.exists('/kaggle/working/EasyEdit'):\n",
        "    # Clone EasyEdit for its GRACE hparams yaml (hparams/GRACE/gpt2-xl.yaml).\n",
        "    subprocess.run(['git', 'clone', '--depth', '1',\n",
        "                    'https://github.com/zjunlp/EasyEdit.git',\n",
        "                    '/kaggle/working/EasyEdit'], check=True)\n",
        "# Install our package (editable) and EasyEdit (PyPI). Run without -q so a\n",
        "# dependency conflict surfaces in the log; a hard conflict here is itself\n",
        "# an F1 finding.\n",
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '-e',\n",
        "                '/kaggle/working/U-JEPA/v2'], check=True)\n",
        "subprocess.run([sys.executable, '-m', 'pip', 'install', 'easyeditor'], check=False)\n",
        "print('installs attempted')",
    ),
    code(
        "# Cell 3: HF cache env (gpt2-xl is public, no token needed) + optional secret.\n",
        "import os\n",
        "try:\n",
        "    from kaggle_secrets import UserSecretsClient\n",
        "    os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')\n",
        "except Exception as e:\n",
        "    print(f'no HF_TOKEN secret ({e}); public models still load')\n",
        "for k in ('HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE'):\n",
        "    os.environ.setdefault(k, '/tmp/hf_cache')",
    ),
    code(
        "# Cell 4: env gate. Exits non-zero if the GPU is off on Kaggle.\n",
        "import subprocess, sys\n",
        "subprocess.run([sys.executable,\n",
        "                '/kaggle/working/U-JEPA/v2/scripts/00_smoke_env.py'],\n",
        "               check=True)",
    ),
    code(
        "# Cell 5: run the spike. dtype fp16 for the T4. Points --hparams at the\n",
        "# cloned EasyEdit GRACE yaml. The script never hard-crashes; it writes a\n",
        "# verdict either way.\n",
        "import subprocess, sys, os\n",
        "env = dict(os.environ)\n",
        "subprocess.run([sys.executable,\n",
        "                '/kaggle/working/U-JEPA/v2/scripts/02_spike_f1_grace.py',\n",
        "                '--model', 'gpt2-xl', '--dtype', 'fp16',\n",
        "                '--hparams', '/kaggle/working/EasyEdit/hparams/GRACE/gpt2-xl.yaml',\n",
        "                '--out', '/kaggle/working/results/f1_spike.json'],\n",
        "               check=True, env=env)",
    ),
    code(
        "# Cell 6: show the verdict and the numbers.\n",
        "import json\n",
        "from pathlib import Path\n",
        "p = Path('/kaggle/working/results/f1_spike.json')\n",
        "if p.exists():\n",
        "    r = json.loads(p.read_text())\n",
        "    print('VERDICT:', r.get('verdict'), '-', r.get('verdict_rationale'))\n",
        "    print('baseline edit_success:', r.get('baseline', {}).get('edit_success'))\n",
        "    print('post edit_success    :', r.get('post', {}).get('edit_success'))\n",
        "    print('baseline ppl / post ppl:', r.get('baseline_perplexity'), '/', r.get('post_perplexity'))\n",
        "    print('easyedit_error:', r.get('easyedit_error'))\n",
        "else:\n",
        "    print('no results file written')",
    ),
]

NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = HERE / "f1_spike" / "f1_spike.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(NB, indent=1) + "\n", encoding="utf-8")
print(f"wrote {out} ({len(CELLS)} cells)")
