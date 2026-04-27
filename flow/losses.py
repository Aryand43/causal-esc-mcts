"""AFlow losses :math:`L_{\\mathrm{flow}}` and ranking :math:`L_{\\mathrm{rank}}`."""

import torch


def flow_consistency_loss(flow_edges: torch.Tensor, flow_from_policy: torch.Tensor) -> torch.Tensor:
    """Flow consistency term (stub scalar)."""
    return torch.zeros(())


def ranking_loss(v_winner: torch.Tensor, v_loser: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Pairwise ranking hinge on :math:`V_\\phi` (stub scalar)."""
    return torch.zeros(())
