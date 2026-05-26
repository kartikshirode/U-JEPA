"""Hot-swappable LoRA bank for continual learning on a frozen base model.

Design notes:
  Why not peft.PeftModel:
    - We need O(1) per-task activation by flipping a pointer (peft requires
      add_adapter then set_adapter calls that involve a per-module walk).
    - We need direct access to A and B matrices for the N-LoRA orthogonality
      and non-collision penalties without going through peft's wrappers.
    - peft's quantized-base hooks have shape issues stacking on top of vendored
      LatentMAS KV-cache plumbing (their past_key_values prepending is sensitive
      to module-order changes).

  Adapters are plain nn.Parameters. Forward hooks add the active adapter's
  delta to the base module's output. Base parameters stay frozen.
"""
from __future__ import annotations
import math
from typing import Iterable

import torch
import torch.nn as nn


def _safe_key(module_name: str) -> str:
    """ParameterDict keys cannot contain dots. Replace with double-underscore.
    Full module names from named_modules() look like 'model.layers.0.q_proj'."""
    return module_name.replace(".", "__")


class _Adapter(nn.ParameterDict):
    """One adapter: ParameterDict keyed by '<safe_module_name>__A' and '__B'."""

    def __init__(self, target_dims: dict[str, tuple[int, int]], rank: int):
        super().__init__()
        for name, (d_in, d_out) in target_dims.items():
            key = _safe_key(name)
            A = nn.Parameter(torch.empty(rank, d_in))
            B = nn.Parameter(torch.zeros(d_out, rank))
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            # B stays zero so delta_W = B @ A is zero at init (no change at first step)
            self[f"{key}__A"] = A
            self[f"{key}__B"] = B

    def matrices(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        key = _safe_key(name)
        return self[f"{key}__A"], self[f"{key}__B"]


class OrthogonalLoRABank(nn.Module):
    """Bank of LoRA adapters with per-task activation."""

    def __init__(
        self,
        base_model: nn.Module,
        rank: int = 16,
        target_modules: Iterable[str] = ("q_proj", "v_proj"),
        alpha: float = 32.0,
    ):
        super().__init__()
        self.base = base_model
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.target_modules = tuple(target_modules)
        self.adapters = nn.ModuleDict()
        self._active: str | None = None
        self._target_dims = self._collect_target_dims()

    def _collect_target_dims(self) -> dict[str, tuple[int, int]]:
        """Walk the base model and find each target module's (in, out) dims."""
        dims: dict[str, tuple[int, int]] = {}
        for full_name, module in self.base.named_modules():
            short = full_name.rsplit(".", 1)[-1]
            if short in self.target_modules and isinstance(module, nn.Linear):
                dims[full_name] = (module.in_features, module.out_features)
        if not dims:
            raise RuntimeError(
                f"No target modules {self.target_modules} found on base model"
            )
        return dims

    def add_task(self, task_id: str) -> None:
        if task_id in self.adapters:
            raise ValueError(f"task {task_id!r} already exists")
        self.adapters[task_id] = _Adapter(self._target_dims, self.rank)
        self.activate(task_id)

    def activate(self, task_id: str) -> None:
        if task_id not in self.adapters:
            raise KeyError(task_id)
        self._active = task_id

    @property
    def active(self) -> str | None:
        return self._active

    def adapter_matrices(self, task_id: str, module_name: str):
        return self.adapters[task_id].matrices(module_name)

    def forward_target(self, x: torch.Tensor, module_name: str) -> torch.Tensor:
        """Apply the active adapter's delta to module_name's output.

        Used by forward hooks; also exposed for unit tests.
        """
        if self._active is None:
            d_out = self._target_dims[module_name][1]
            return torch.zeros(x.shape[:-1] + (d_out,), dtype=x.dtype, device=x.device)
        A, B = self.adapter_matrices(self._active, module_name)
        return (x @ A.T @ B.T) * self.scale

    def install_hooks(self) -> list:
        """Register forward hooks on every target module.

        Returns hook handles so the caller can remove them at teardown.
        """
        handles = []
        for full_name in self._target_dims:
            module = self.base.get_submodule(full_name)
            handles.append(module.register_forward_hook(self._make_hook(full_name)))
        return handles

    def _make_hook(self, module_name: str):
        def hook(_module, inputs, output):
            if self._active is None:
                return output
            x = inputs[0]
            return output + self.forward_target(x, module_name)
        return hook
