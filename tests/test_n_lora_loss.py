"""N-LoRA orthogonality + non-collision penalty (Yang et al., COLING 2025)."""
import torch

from u_jepa.continual.n_lora_loss import n_lora_penalty


def test_zero_penalty_when_no_prev_tasks():
    A = torch.randn(4, 16)
    loss = n_lora_penalty(A, [], collision_weight=0.01)
    assert loss.item() == 0.0


def test_zero_penalty_for_orthogonal_pair_no_collision_term():
    A_curr = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    A_prev = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    loss = n_lora_penalty(A_curr, [A_prev], collision_weight=0.0)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_orthogonal_pair_with_collision_term_still_zero():
    """Orthogonal supports (disjoint non-zero indices) means zero collision too."""
    A_curr = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    A_prev = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    loss = n_lora_penalty(A_curr, [A_prev], collision_weight=0.5)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_parallel_pair_yields_nonzero_orthogonality_penalty():
    A_curr = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    A_prev = torch.tensor([[0.5, 0.0, 0.0, 0.0]])
    loss = n_lora_penalty(A_curr, [A_prev], collision_weight=0.0)
    assert loss.item() > 0.2


def test_collision_term_adds_extra_penalty_for_overlapping_supports():
    A_curr = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    A_prev = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    loss_o_only = n_lora_penalty(A_curr, [A_prev], collision_weight=0.0)
    loss_with_c = n_lora_penalty(A_curr, [A_prev], collision_weight=0.5)
    assert loss_with_c > loss_o_only


def test_penalty_accumulates_across_multiple_prev_tasks():
    A_curr = torch.tensor([[1.0, 0.0, 0.0]])
    A_prev1 = torch.tensor([[0.5, 0.0, 0.0]])
    A_prev2 = torch.tensor([[0.5, 0.0, 0.0]])
    loss_one = n_lora_penalty(A_curr, [A_prev1], collision_weight=0.0)
    loss_two = n_lora_penalty(A_curr, [A_prev1, A_prev2], collision_weight=0.0)
    assert torch.isclose(loss_two, loss_one * 2)


def test_a_prev_is_detached_no_gradient_flows_back():
    A_curr = torch.tensor([[1.0, 0.0]], requires_grad=True)
    A_prev = torch.tensor([[0.5, 0.0]], requires_grad=True)
    loss = n_lora_penalty(A_curr, [A_prev], collision_weight=0.0)
    loss.backward()
    assert A_curr.grad is not None
    assert A_prev.grad is None  # detached inside the loss
