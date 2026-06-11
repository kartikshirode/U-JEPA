"""Load the frozen core models (GPT-2-XL, Qwen2.5-1.5B).

Phase 0 contract: returns a model in eval() with all params frozen, plus its
tokenizer. fp16 default for T4 parity. NF4 path is wired but optional (cores
do not need it at 1.5B; NF4 is reserved for any later adapter training).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Dtype = Literal["fp16", "bf16", "fp32"]


@dataclass(frozen=True)
class CoreSpec:
    short_name: str
    hf_id: str
    family: str  # "gpt2" or "qwen"
    params_b: float
    notes: str = ""


CORE_MODELS: dict[str, CoreSpec] = {
    "gpt2-xl": CoreSpec(
        short_name="gpt2-xl",
        hf_id="gpt2-xl",
        family="gpt2",
        params_b=1.5,
        notes="Editing-literature baseline (GRACE, ROME, MEMIT paper-comparable).",
    ),
    "qwen2.5-1.5b": CoreSpec(
        short_name="qwen2.5-1.5b",
        hf_id="Qwen/Qwen2.5-1.5B-Instruct",
        family="qwen",
        params_b=1.5,
        notes="Real-system core. Apache-2.0, better than GPT-2 at instruction following.",
    ),
}


def resolve_core(name: str) -> CoreSpec:
    key = name.strip().lower()
    if key in CORE_MODELS:
        return CORE_MODELS[key]
    for spec in CORE_MODELS.values():
        if spec.hf_id.lower() == key:
            return spec
    raise KeyError(f"Unknown core model {name!r}. Known: {sorted(CORE_MODELS)}")


def _torch_dtype(dtype: Dtype) -> Any:
    import torch
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]


def freeze_(model: Any) -> Any:
    """Freeze every parameter in-place. Returns the model for chaining."""
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_core(
    name: str,
    *,
    dtype: Dtype = "fp16",
    nf4: bool = False,
    device: str = "auto",
    cache_dir: str | None = None,
    trust_remote_code: bool = False,
) -> tuple[Any, Any]:
    """Load a frozen core model + tokenizer.

    `device='auto'` uses HF device_map='auto' (places on cuda if present, else
    cpu). Any other value ('cpu', 'cuda', 'cuda:0', ...) pins the whole model
    to that device via device_map={'': device}.
    `nf4=True` requires bitsandbytes; only meaningful if you ever train an adapter.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = resolve_core(name)

    tok = AutoTokenizer.from_pretrained(
        spec.hf_id,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    load_kwargs: dict[str, Any] = {
        "cache_dir": cache_dir,
        "trust_remote_code": trust_remote_code,
    }
    if device == "auto":
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = {"": device}

    # torch_dtype applies in both branches: under nf4 it sets the dtype of the
    # modules bitsandbytes leaves unquantized (otherwise they load fp32).
    load_kwargs["torch_dtype"] = _torch_dtype(dtype)

    if nf4:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            raise RuntimeError("nf4=True needs bitsandbytes installed.") from e
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=_torch_dtype(dtype),
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **load_kwargs)
    freeze_(model)
    model.eval()
    return model, tok
