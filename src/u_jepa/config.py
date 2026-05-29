"""Central config dataclasses. Every script reads its config from here."""
from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class HardwareConfig:
    device: str = "cuda:0"
    vram_ceiling_mb: int = 7800
    grad_accum: int = 8
    micro_batch: int = 1
    # 512 is plenty for FOMC (one short sentence) and ScienceQA-text (a
    # question plus a few short choices). 1024 just wasted padding compute and
    # memory at batch 1 since each sequence already pads to its own length.
    max_seq_len: int = 512
    bf16_compute: bool = True

@dataclass(frozen=True)
class QwenConfig:
    # Default for phases 1+ where we own the loader and train adapters.
    # Use Qwen3-14B-Instruct for instruction-tuned reasoning quality.
    model_id: str = "Qwen/Qwen3-14B-Instruct"
    quant_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    hidden_size: int = 5120     # Qwen3-14B
    n_layers: int = 48          # Qwen3-14B

@dataclass(frozen=True)
class QwenBaselineConfig:
    """For Phase 0 reproduction via vendored LatentMAS run.py.

    The vendored argparse hardcodes choices to {Qwen3-4B, Qwen3-14B} (base
    models, no Instruct). We use the same model they validated on.
    """
    model_name: str = "Qwen/Qwen3-14B"
    tensor_parallel_size: int = 2        # dual T4 on Kaggle
    gpu_memory_utilization: float = 0.85
    max_samples: int = 250
    generate_bs: int = 8
    latent_steps: int = 4
    task: str = "gsm8k"
    prompt: str = "sequential"

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
