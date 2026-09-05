# U-JEPA v3

Admission control for automated knowledge maintenance. Design in
`docs/superpowers/specs/2026-09-05-u-jepa-v3-design.md`, implementation plan in
`docs/superpowers/plans/2026-09-05-u-jepa-v3-harness-and-rq1.md`, cluster notes
in `v3/docs/cluster-baramati.md`.

Stage 0 (harness), stage 1 (RQ1) and stage 2 (the gate) are all built. Stage 2
is a mechanism with no numbers in it: the signals, the combiner, the calibration
procedure and the rollback audit exist and are tested, and which signals survive
is an empirical question that RQ1 and RQ2 answer.

## What RQ1 asks

An operator keeps a model current from a public feed. Some entries are poisoned.
The upstream source notices and publishes a correction, which the operator
applies in good faith. Does that correction actually remove the poison?

The editing literature says probably not. Edits suppress rather than erase, so
the pipeline can look healthy while the model still holds a false fact that a
paraphrase or a leading question pulls straight back out. If that holds for the
modern stable editors, admission is the only place left to intervene.

## What RQ2 asks

Given that, what does gating the feed buy, and what does it cost. Same feed,
same editor, same probes, with entries passing a gate first. The two arms
subtract. Precision gets reported at the prevalence an operator actually faces,
not at the balanced rate the calibration sample happens to have, because
precision falls with prevalence and AUROC does not move at all.

## Install and test

```bash
cd v3
pip install -e ".[dev]"
pytest -q
```

Everything runs on CPU with no network and no model. 279 tests, 1 skipped, the
skip being the GPU path in the EasyEdit adapter.

## First hour on the cluster

Read `v3/docs/cluster-baramati.md` first. The short version: the H200s are cut
into 18 GB MIG slices, the slice is what you get, and the design's old "no
single job above 141 GB" line was wrong by a factor of 8.

Login node, which is the only machine with a network:

```bash
export HF_HOME=/home/$USER/.cache/huggingface
python v3/scripts/03_prefetch.py --model meta-llama/Llama-3.2-3B-Instruct
python v3/scripts/01_build_probes.py --out /home/$USER/probes --n 200
export U_JEPA_V3_PROBE_DIR=/home/$USER/probes
```

Then confirm what the allocation contains:

```bash
mkdir -p logs && sbatch v3/slurm/00_smoke.slurm
```

It reports the slice, the derived dtype, whether anything pairs, and the memory
arithmetic for each planned arm. Exit 2 means no CUDA reached the job, exit 1
means the harness is fine and stage 1 is blocked on easyeditor or the probe
sets.

Two things still need doing there and neither is code. Install easyeditor, then
run `python v3/scripts/04_check_hparams.py v3/hparams/`; those YAML files were
written from published templates on a laptop with no EasyEdit, and the checker
builds the real HyperParams object, which is what settles the field names.
UltraEdit's file is the least certain of the four.

## Running a grid

Check the split before spending slots on it:

```bash
python -m u_jepa_v3.runs.worker --grid grids/rq1_pilot.json \
    --out runs/rq1_pilot --node 0 --of 14 --dry-run
```

Then one array task per slice:

```bash
sbatch v3/slurm/worker_array.slurm v3/grids/rq1_pilot.json runs/rq1_pilot
sbatch v3/slurm/worker_array.slurm v3/grids/rq2_pilot.json runs/rq2_pilot
```

`--node` and `--of` default to the array task coordinates, so the same command
works inside a job script and outside one. The grid's `arm` key picks the
driver, rq1 for the undefended pipeline and rq2 for the gated one.

Grid sizes come from `scripts/02_power.py` rather than from round numbers. The
survival gap is two rates on the same items, so it is a paired test and the
thing that drives power is how often the two elicitation modes disagree. The
earlier pilot grid asked for 40 poison pairs at a 0.05 base rate, which is 2
poisoned facts per cell.

## Four things that are deliberate and look wrong

**Cells are atomic.** Kill a task and the interrupted cell reruns from zero;
only finished cells are skipped. Weights, editor normalization state and RNG
state are never checkpointed, so continuing a partial cell would continue from
the wrong model while believing tens of thousands of edits had landed. An
earlier version did exactly that and every number after a restart was silently
wrong.

**Editors own their responder.** There is no way to obtain a probe target that
is not bound to the edits applied so far. The earlier adapter discarded the
model `edit()` returns, so probes read the untouched base for entire runs, and
the test doubles hid it because the fake responder was never updated either.

**The gate never sees a FeedEntry.** It gets a `GateInput`, which drops
`is_poison` and `attack_family` and replaces `candidate.source` with the
simulated account. The attack generators write the family name into `source`, so
a signal reading it would score perfectly and mean nothing. Discipline would
have been enough right up until it was not.

**Attacker accounts carry ordinary traffic.** `simulate_sources` gives them a
share of benign entries too. Without that the account is the label again, and a
provenance signal would post a result that is an artefact of the simulation.

## What is not built

Stages 3 to 6: the preprint, the belief-state predictor and energy, the adaptive
white-box attacker, and the ablation sweep. Stage 4 is the one that decides
whether the JEPA name stays on the project.
