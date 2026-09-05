"""What the target cluster actually offers, and whether a cell fits in it.

The v3 design was written against "4 H200s, 141 GB each". The cluster it runs on
is Baramati (Benchmark Computer Solutions), where those H200 NVL cards are cut
into MIG slices and the slice is the schedulable unit. A job asks for
`gpu:1g.18gb:1` and gets 18 GB, not 141. Slices are separate devices with no
peer access, so two of them are two small GPUs rather than one large one.

That changes which arms are runnable. An 8B model in bf16 is 16 GB of weights
before the editor allocates anything, and MEMIT and AlphaEdit hold a covariance
per edited layer. This module does the arithmetic up front so a cell that cannot
fit is refused at submission rather than discovered by an OOM 40 minutes in.

Every number here is an estimate with a stated basis. Measured numbers replace
them the first time a real arm runs; `FitReport.notes` says which is which.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

BYTES_PER_PARAM = {"fp16": 2, "bf16": 2, "fp32": 4}

# Baramati gres flavours, from the Slurm user guide and confirmed by sinfo.
# aicoeserver03 and 04 carry 14 of the 18 GB flavour, aicoeserver05 8 of the
# 24 GB one. Ask for a flavour and Slurm picks a node that has it.
SLICE_GB = {"1g.18gb": 18.0, "1g.24gb": 24.0}

# A MIG slice does not hand over its nominal size. Some is reserved, and the
# CUDA context takes its cut before the first tensor is allocated.
USABLE_FRACTION = 0.93
CUDA_CONTEXT_GB = 0.8


@dataclass(frozen=True)
class ModelSpec:
    """The four numbers that decide memory. Read them off config.json."""

    name: str
    n_params_b: float
    hidden: int
    intermediate: int
    n_layers: int


KNOWN_MODELS = {
    m.name: m
    for m in (
        ModelSpec("Qwen/Qwen2.5-1.5B-Instruct", 1.54, 1536, 8960, 28),
        ModelSpec("meta-llama/Llama-3.2-1B-Instruct", 1.24, 2048, 8192, 16),
        ModelSpec("meta-llama/Llama-3.2-3B-Instruct", 3.21, 3072, 8192, 28),
        ModelSpec("EleutherAI/gpt-j-6b", 6.05, 4096, 16384, 28),
        ModelSpec("Qwen/Qwen2.5-7B-Instruct", 7.62, 3584, 18944, 28),
        ModelSpec("meta-llama/Meta-Llama-3-8B-Instruct", 8.03, 4096, 14336, 32),
        ModelSpec("gpt2-xl", 1.56, 1600, 6400, 48),
    )
}

# Short names for hparams files, so a grid can name an editor and a model and
# the path falls out instead of being a fourth dimension nothing checks.
MODEL_SLUG = {
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen2.5-1.5b",
    "meta-llama/Llama-3.2-1B-Instruct": "llama3.2-1b",
    "meta-llama/Llama-3.2-3B-Instruct": "llama3.2-3b",
    "EleutherAI/gpt-j-6b": "gpt-j-6b",
    "Qwen/Qwen2.5-7B-Instruct": "qwen2.5-7b",
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama3-8b",
    "gpt2-xl": "gpt2-xl",
}


def hparams_slug(model: str) -> str:
    if model not in MODEL_SLUG:
        raise KeyError(
            f"no hparams slug for {model!r}. Add one to cluster.MODEL_SLUG and write "
            f"the matching yaml files under v3/hparams/"
        )
    return MODEL_SLUG[model]


# How many MLP layers each method writes to, and therefore how many second
# moment matrices it holds. ROME edits one layer. MEMIT spreads over a band,
# 5 in the published configs. AlphaEdit holds the same band plus a null space
# projection of the same shape per layer, so it pays twice.
EDIT_LAYERS = {"rome": 1, "memit": 5, "alphaedit": 5, "ultraedit": 0,
               "wise": 0, "grace": 0, "stub": 0}
COVARIANCE_MULTIPLIER = {"alphaedit": 2}


@dataclass(frozen=True)
class FitReport:
    model: str
    method: str
    dtype: str
    weights_gb: float
    editor_gb: float
    activation_gb: float
    required_gb: float
    budget_gb: float
    fits: bool
    notes: list[str] = field(default_factory=list)

    @property
    def headroom_gb(self) -> float:
        return self.budget_gb - self.required_gb

    def as_dict(self) -> dict:
        return {
            "model": self.model, "method": self.method, "dtype": self.dtype,
            "weights_gb": round(self.weights_gb, 2),
            "editor_gb": round(self.editor_gb, 2),
            "activation_gb": round(self.activation_gb, 2),
            "required_gb": round(self.required_gb, 2),
            "budget_gb": round(self.budget_gb, 2),
            "headroom_gb": round(self.headroom_gb, 2),
            "fits": self.fits, "notes": list(self.notes),
        }


def weights_gb(spec: ModelSpec, dtype: str) -> float:
    if dtype not in BYTES_PER_PARAM:
        raise ValueError(f"dtype {dtype!r} not in {sorted(BYTES_PER_PARAM)}")
    return spec.n_params_b * 1e9 * BYTES_PER_PARAM[dtype] / 1e9


def editor_overhead_gb(method: str, spec: ModelSpec) -> float:
    """Second moment statistics the locate-and-edit methods keep resident.

    Each is intermediate x intermediate in fp32, one per edited layer. That is
    822 MB per layer for an 8B model and 268 MB for a 3B one, which is the
    difference between an arm fitting an 18 GB slice and not.

    UltraEdit keeps running mean and variance over the key space instead, which
    is linear in the hidden size and rounds to zero here. WISE and GRACE keep a
    side memory that grows with the edit count; that is a runtime concern rather
    than a load time one, so it is not counted and is called out in the notes.
    """
    layers = EDIT_LAYERS.get(method, 0)
    if not layers:
        return 0.0
    per_layer = spec.intermediate * spec.intermediate * 4 / 1e9
    return per_layer * layers * COVARIANCE_MULTIPLIER.get(method, 1)


def slice_budget_gb(flavour_or_gb: str | float) -> float:
    """Usable bytes for a gres flavour, after the reservation and the context."""
    if isinstance(flavour_or_gb, str):
        if flavour_or_gb not in SLICE_GB:
            raise KeyError(f"unknown gres flavour {flavour_or_gb!r}; "
                           f"have {sorted(SLICE_GB)}")
        nominal = SLICE_GB[flavour_or_gb]
    else:
        nominal = float(flavour_or_gb)
    return nominal * USABLE_FRACTION - CUDA_CONTEXT_GB


def plan_fit(
    model: str,
    method: str,
    dtype: str,
    budget: str | float = "1g.18gb",
    activation_gb: float = 1.5,
) -> FitReport:
    """Estimate whether one editing arm fits one device.

    activation_gb covers the KV cache and the generation buffers for the probe
    batches. 1.5 GB is a working allowance for short greedy answers at a batch
    of 32; raise it before blaming the model.
    """
    spec = KNOWN_MODELS.get(model)
    notes: list[str] = []
    if spec is None:
        raise KeyError(
            f"{model!r} has no ModelSpec. Add one from its config.json: "
            "hidden_size, intermediate_size, num_hidden_layers and the parameter count."
        )

    w = weights_gb(spec, dtype)
    e = editor_overhead_gb(method, spec)
    budget_gb = slice_budget_gb(budget)
    required = w + e + activation_gb

    if method in ("wise", "grace"):
        notes.append(f"{method} grows a side memory with the edit count; "
                     "this estimate covers load time only")
    if method not in EDIT_LAYERS:
        notes.append(f"no overhead model for {method!r}; counted as zero")
    if required > budget_gb:
        notes.append("does not fit one slice. Options: a smaller model, a "
                     "1g.24gb slice on aicoeserver05, or a single layer method")
    elif budget_gb - required < 2.0:
        notes.append("under 2 GB of headroom. A longer probe batch will OOM")

    return FitReport(
        model=model, method=method, dtype=dtype,
        weights_gb=w, editor_gb=e, activation_gb=activation_gb,
        required_gb=required, budget_gb=budget_gb,
        fits=required <= budget_gb, notes=notes,
    )


@dataclass(frozen=True)
class SlurmContext:
    """The parts of the Slurm environment that change how the worker behaves."""

    job_id: str | None
    array_task_id: int | None
    array_task_count: int | None
    visible_devices: str | None
    node: str | None

    @property
    def under_slurm(self) -> bool:
        return self.job_id is not None

    @property
    def is_array(self) -> bool:
        return self.array_task_id is not None and self.array_task_count is not None

    def as_dict(self) -> dict:
        return {
            "job_id": self.job_id, "array_task_id": self.array_task_id,
            "array_task_count": self.array_task_count,
            "visible_devices": self.visible_devices, "node": self.node,
            "under_slurm": self.under_slurm, "is_array": self.is_array,
        }


def _int_or_none(raw: str | None) -> int | None:
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None


def slurm_context(environ: dict | None = None) -> SlurmContext:
    env = os.environ if environ is None else environ
    return SlurmContext(
        job_id=env.get("SLURM_JOB_ID") or env.get("SLURM_JOBID"),
        array_task_id=_int_or_none(env.get("SLURM_ARRAY_TASK_ID")),
        array_task_count=_int_or_none(env.get("SLURM_ARRAY_TASK_COUNT")),
        visible_devices=env.get("CUDA_VISIBLE_DEVICES"),
        node=env.get("SLURMD_NODENAME"),
    )


def device_report() -> list[dict]:
    """Per visible device: name, memory, capability and whether it is a MIG slice.

    Empty when torch is missing or there is no CUDA, so callers can print it on
    a laptop without guarding.
    """
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []

    out = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        total = props.total_memory / 1e9
        out.append({
            "index": index,
            "name": props.name,
            "total_gb": round(total, 1),
            "capability": f"{props.major}.{props.minor}",
            # A MIG instance reports the parent card's name with a MIG suffix,
            # and always a fraction of its memory. Either signal alone is weak,
            # so both are recorded and the caller decides.
            "looks_like_mig": "MIG" in props.name.upper() or total < 40.0,
        })
    return out
