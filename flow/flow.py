"""AFlow-style state flow :math:`F(s)=Q_b(s)V_\\phi(s)` and edge factors."""

import torch


def compute_state_flow(Q_b: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """State flow :math:`F(s)=Q_b(s)V_\\phi(s)` (stub)."""
    return torch.zeros_like(Q_b)


def compute_edge_flow(F_s: torch.Tensor, policy_probs: torch.Tensor) -> torch.Tensor:
    """Edge flow from :math:`F(s)` and :math:`\\pi_\\theta(a\\mid s)` (stub)."""
    return torch.zeros_like(policy_probs)
