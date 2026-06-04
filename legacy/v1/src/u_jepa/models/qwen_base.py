"""Qwen3-14B-Instruct loader at NF4 for adapter training.

We use this for Phase 1+ where we own the loader and train adapters on
top of a frozen quantized base. Phase 0 reproduction uses vendored
LatentMAS' ModelWrapper which loads via vLLM separately.
"""
from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from u_jepa.config import QwenConfig


def _bnb_config(cfg: QwenConfig):
    """Build a BitsAndBytesConfig matching cfg. Import lazily so the local
    Windows env (no bitsandbytes) does not crash on import of this module."""
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=cfg.quant_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=True,
    )


def load_qwen_nf4(cfg: QwenConfig | None = None, device_map: str | dict | None = None):
    """Load Qwen3-14B-Instruct at NF4 and return (model, tokenizer).

    Default pins the whole model to cuda:0 ({"": 0}). NF4 14B weights are
    ~8 GB, plus a couple GB of activations at seq 512 batch 1, which fits in a
    single 15 GB T4 with room to spare. Keeping every layer on one device
    avoids the device-mismatch crash that device_map='auto' can cause when
    it splits the model across cuda:0 and cuda:1 while the training loop
    forces inputs onto a single hardcoded device. The second T4 stays free
    for other work. Pass device_map='auto' explicitly to opt into sharding.
    """
    cfg = cfg or QwenConfig()
    if device_map is None:
        # Whole model on cuda:0 if a GPU exists, else CPU offload so an
        # import-time call does not hard-crash on a CPU-only box.
        device_map = {"": 0} if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=_bnb_config(cfg),
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def model_input_device(model) -> "torch.device":
    """Device that token inputs must live on before the first forward.

    With device_map='auto' the embedding layer may sit on a different GPU
    than cuda:0, so inputs hardcoded to 'cuda:0' would mismatch. Resolve the
    actual input-embedding device and fall back to the first parameter's
    device, then CPU, so callers can move inputs there reliably.
    """
    try:
        emb = model.get_input_embeddings()
        if emb is not None and emb.weight is not None:
            return emb.weight.device
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def qwen_vram_usage_mb(model) -> int:
    """Approximate VRAM held by a model's parameters and buffers, in MiB."""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    for b in model.buffers():
        total_bytes += b.numel() * b.element_size()
    return total_bytes // (1024 * 1024)


def target_module_names(model, target_short_names=("q_proj", "v_proj")) -> list[str]:
    """Return full names of every nn.Linear whose short name matches.

    Useful for sanity-checking which modules the LoRA bank will hook.
    Qwen3 attention blocks use q_proj/k_proj/v_proj/o_proj naming.
    """
    import torch.nn as nn
    matches = []
    for full_name, module in model.named_modules():
        short = full_name.rsplit(".", 1)[-1]
        if short in target_short_names and isinstance(module, nn.Linear):
            matches.append(full_name)
    return matches
