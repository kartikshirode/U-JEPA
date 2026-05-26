"""Forward-hook integration test: adapter delta actually shows up at base output."""
import torch
import torch.nn as nn

from u_jepa.continual.orthogonal_lora import OrthogonalLoRABank


class _MiniBlock(nn.Module):
    """Two linears named q_proj and v_proj summed - stand-in for one Qwen layer."""
    def __init__(self, d=32):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.q_proj(x) + self.v_proj(x)


def test_hooks_add_active_adapter_delta_to_base_output():
    base = _MiniBlock(d=32)
    bank = OrthogonalLoRABank(base, rank=4, target_modules=("q_proj",))
    bank.add_task("t1")
    # Non-zero delta: set A and B so B@A is meaningful
    with torch.no_grad():
        A, B = bank.adapter_matrices("t1", "q_proj")
        A.fill_(0.2)
        B.fill_(0.3)
    handles = bank.install_hooks()
    try:
        x = torch.randn(2, 32)
        y_with = base(x)
        bank._active = None
        y_without = base(x)
        # Hook should make these differ
        assert not torch.allclose(y_with, y_without)
        # Delta is exactly the adapter contribution
        delta = y_with - y_without
        expected = bank.forward_target(x, "q_proj") if False else None  # not active now
        # Verify the delta matches what active=t1 would produce
        bank.activate("t1")
        manual_delta = bank.forward_target(x, "q_proj")
        assert torch.allclose(delta, manual_delta, atol=1e-5)
    finally:
        for h in handles:
            h.remove()


def test_hooks_are_noop_when_no_adapter_active():
    base = _MiniBlock(d=32)
    bank = OrthogonalLoRABank(base, rank=4, target_modules=("q_proj",))
    handles = bank.install_hooks()
    try:
        x = torch.randn(2, 32)
        y_hook = base(x)
        # Without hooks
        for h in handles:
            h.remove()
        handles = []
        y_no_hook = base(x)
        assert torch.allclose(y_hook, y_no_hook)
    finally:
        for h in handles:
            h.remove()
