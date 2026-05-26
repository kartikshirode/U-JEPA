"""Central config dataclasses. Every script reads its config from here."""
from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class HardwareConfig:
    device: str = "cuda:0"
    vram_ceiling_mb: int = 7800
    grad_accum: int = 8
    micro_batch: int = 1
    max_seq_len: int = 1024
    bf16_compute: bool = True

@dataclass(frozen=True)
class QwenConfig:
    model_id: str = "Qwen/Qwen3-4B-Instruct"
    quant_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    hidden_size: int = 2560
    n_layers: int = 36

@dataclass(frozen=True)
class VJEPAConfig:
    model_id: str = "facebook/vjepa2-vitl-fpc64-256"
    out_dim: int = 1024
    frames_per_clip: int = 64
    crop: int = 256

@dataclass(frozen=True)
class LoraConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple = ("q_proj", "v_proj")

@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[2]
    results: Path = field(init=False)
    checkpoints: Path = field(init=False)
    hf_cache: Path = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "results", self.root / "results")
        object.__setattr__(self, "checkpoints", self.root / "checkpoints")
        object.__setattr__(self, "hf_cache", self.root / "hf_cache")
