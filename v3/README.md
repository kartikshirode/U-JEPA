# U-JEPA v3

Admission control for automated knowledge maintenance. Design in
`docs/superpowers/specs/2026-09-05-u-jepa-v3-design.md`, implementation plan in
`docs/superpowers/plans/2026-09-05-u-jepa-v3-harness-and-rq1.md`.

Stage 0 (harness) and stage 1 (RQ1) are built. The gate itself is stage 2 and has
no code yet.

## What RQ1 asks

An operator keeps a model current from a public feed. Some entries are poisoned.
The upstream source notices and publishes a correction, which the operator
applies in good faith. Does that correction actually remove the poison?

The editing literature says probably not. Edits suppress rather than erase, so
the pipeline can look healthy while the model still holds a false fact that a
paraphrase or a leading question pulls straight back out. If that holds for the
modern stable editors, admission is the only place left to intervene.

## Install and test

```bash
cd v3
pip install -e ".[dev]"
pytest -q
```

Everything runs on CPU with no network and no model. 138 tests, 1 skipped, the
skip being the GPU path in the EasyEdit adapter.

## First 30 minutes on the HPC box

```bash
python v3/scripts/00_smoke_gpu.py
```

That reports device count, VRAM, compute capability, the derived dtype, and
whether the cards share NVLink. The design currently assumes no single job
exceeds 141 GB because the topology was never confirmed; this is what confirms or
drops that.

It then checks two things stage 1 needs and the harness does not:

1. `easyeditor` installed, plus hparams for ultraedit, alphaedit, rome and memit
   against the chosen 8B model.
2. Probe sets built. Write `sst.json`, `mmlu.json`, `mrpc.json`, `nli.json` and
   `locality.json` as JSON lists of `[prompt, expected]` pairs, then point
   `U_JEPA_V3_PROBE_DIR` at that directory. Match UltraEdit's evaluation set so
   the numbers sit beside theirs, and pin the versions.

Build the locality set from WikiBigEdit's `loc` and `loc_ans` columns.

## Running a grid

Check the shard split before spending GPU hours on it:

```bash
python -m u_jepa_v3.runs.worker --grid grids/rq1_pilot.json \
    --out runs/rq1_pilot --node 0 --of 4 --dry-run
```

Then one process per GPU, no collectives:

```bash
for n in 0 1 2 3; do
  python -m u_jepa_v3.runs.worker --grid grids/rq1_pilot.json \
      --out runs/rq1_pilot --node $n --of 4 &
done
wait
```

`grids/rq1_pilot.json` is a starting point, not a result. Its `hparams` path is a
placeholder and its sizes are round numbers rather than powered ones. Run the
power calculation for the survival gap before any of it becomes a reported
number.

## Two things that are deliberate and look wrong

**Cells are atomic.** Kill a node and the interrupted cell reruns from zero;
only finished cells are skipped. Weights, editor normalization state and RNG
state are never checkpointed, so continuing a partial cell would continue from
the wrong model while believing tens of thousands of edits had landed. An earlier
version did exactly that and every number after a restart was silently wrong.

**Editors own their responder.** There is no way to obtain a probe target that is
not bound to the edits applied so far. The earlier adapter discarded the model
`edit()` returns, so probes read the untouched base for entire runs, and the test
doubles hid it because the fake responder was never updated either.

## What is not built

Stage 2, the gate: provenance, verification, the combiner, and the shadow-copy
rollback audit. It gets its own plan once RQ1 numbers exist, because the signal
design depends on which attacks survive and what stealth looks like in practice.
