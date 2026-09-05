# The cluster this actually runs on

The v3 design was written against "4 H200s, 141 GB each". That is not what the
cluster hands out. Baramati cuts those cards into MIG slices and schedules the
slice, so a job gets 18 GB and a separate device with no path to any other. The
design's per-job memory assumption was wrong by a factor of 8 in the direction
that matters, and everything below follows from fixing it.

Access details, accounts and passwords stay out of this repo. They live in the
local kit folder on the laptop.

## Layout

| Node | Role | GPUs |
|---|---|---|
| aicoeserver01 | login and master | none |
| aicoeserver02 | storage, serves `/data` over NFS | none |
| aicoeserver03 | compute | 2x H200 NVL, exposed as `gpu:1g.18gb:14` |
| aicoeserver04 | compute | 2x H200 NVL, exposed as `gpu:1g.18gb:14` |
| aicoeserver05 | compute | 2x RTX PRO 6000 Blackwell, exposed as `gpu:1g.24gb:8` |

One partition, `gpu`. Slurm 24.11.7 on Rocky 9, 256 CPUs and about 1 TB of RAM
per compute node. `/home` is shared and 128 T, which is where everything for
this project lives; `/data` is root owned with no user directory.

## What the slices changed

A MIG slice is a whole GPU as far as your process is concerned, just a small
one, and instances do not peer with each other. Two slices are two small GPUs.
So there is no configuration in which a single job sees more than 18 GB (or 24
on aicoeserver05), and model parallelism across slices would go through host
memory.

Run the arithmetic and the model choice falls out. `v3/src/u_jepa_v3/cluster.py`
does it, and `scripts/00_smoke_gpu.py` prints it for the slice you were actually
given:

| Model | Method | Needs | Fits 18 GB |
|---|---|---|---|
| Llama-3.2-3B, bf16 | AlphaEdit | 10.6 GB | yes |
| Llama-3.2-3B, bf16 | ROME | 8.2 GB | yes |
| Qwen2.5-1.5B, bf16 | AlphaEdit | 7.8 GB | yes |
| Llama-3-8B, bf16 | ROME | 18.4 GB | no, though it clears 24 with 3 GB spare |
| Llama-3-8B, bf16 | AlphaEdit | 25.8 GB | no, and it does not clear 24 either |

Usable memory is under the nominal size: a slice reserves some and the CUDA
context takes its cut, so 18 GB is 15.9 GB to work with and 24 is 21.5.

The weights are the smaller half of the problem for the banded methods. MEMIT
and AlphaEdit hold an `intermediate x intermediate` fp32 covariance per edited
layer, which is 822 MB a layer at 8B and 268 MB at 3B, and AlphaEdit holds a
null space projection of the same shape beside each one.

So the primary model is Llama-3.2-3B-Instruct with Qwen2.5-1.5B-Instruct as the
second family, and the 8B arm survives only for single layer methods on the
24 GB slices. Two model families is better evidence than one large model anyway,
and this is the version of that argument that runs.

`check_fits` in `rq1_survival.py` refuses an arm that cannot fit before it loads
anything, so the reason lands in the cell's error field instead of arriving as
an OOM 40 minutes into a queue slot.

## Traps that already cost time on earlier projects

These came from the Adversarial IDS and Vaani work on the same cluster. Each one
was found the expensive way once.

- `srun` fails with "Job credential expired". Everything goes through `sbatch`.
- `--cpus-per-task` defaults to 1 and silently single threads the job. The job
  scripts here set 8.
- CRLF line endings make `sbatch` refuse a script outright. `.gitattributes`
  pins `*.slurm` and `*.sh` to LF in the working tree, not just in the object
  store, because this repo is edited on Windows. Run `dos2unix` after any scp
  that might have converted them back.
- Windows ssh strips quotes, so a python one liner or heredoc sent through `ssh`
  arrives mangled. Write the file, copy it, then run it.
- Compute nodes have no outbound network. The login node does. Everything gets
  fetched on the login node into a cache on `/home`, and jobs run with
  `HF_HUB_OFFLINE=1`.
- Only the `torch-gpu` environment is built for sm_120, which is what
  aicoeserver05's Blackwell cards need.

## The order to do things in

Login node, once:

```bash
export HF_HOME=/home/$USER/.cache/huggingface
python v3/scripts/03_prefetch.py --model meta-llama/Llama-3.2-3B-Instruct
python v3/scripts/01_build_probes.py --out /home/$USER/probes --n 200
export U_JEPA_V3_PROBE_DIR=/home/$USER/probes
```

Then confirm the slice and the prerequisites:

```bash
mkdir -p logs
sbatch v3/slurm/00_smoke.slurm
```

Exit 2 means no CUDA reached the job. Exit 1 means the harness is fine and stage
1 is still blocked on easyeditor or the probe sets. Once easyeditor is in the
environment, validate the hparams before spending a slot on them:

```bash
python v3/scripts/04_check_hparams.py v3/hparams/
```

Those YAML files were written from published templates on a laptop with no
EasyEdit installed. The checker builds the real HyperParams object, which is
what settles whether a field name is right.

Then check the shard split and submit:

```bash
python -m u_jepa_v3.runs.worker --grid v3/grids/rq1_pilot.json \
    --out runs/rq1_pilot --node 0 --of 14 --dry-run
sbatch v3/slurm/worker_array.slurm v3/grids/rq1_pilot.json runs/rq1_pilot
```

Each array task takes one slice and one interleaved shard of the grid, and
writes one JSON per finished cell named by that cell's hash. Nothing is shared
between tasks, so two of them can never collide, and a task that dies leaves
only its own unfinished cell behind.

## Still unconfirmed

The whole table above comes from the cluster's own user guide and from notes
taken during earlier sessions, not from a run of this project's code. Nothing
here is settled until `00_smoke.slurm` has run once and printed what the
allocation actually contains. In particular: whether both 18 GB nodes are
usable in practice given other users, whether the conda environment is called
`torch-gpu-pip` or `pytorch-gpu-pip`, and whether easyeditor installs cleanly
against the CUDA build that is there.
