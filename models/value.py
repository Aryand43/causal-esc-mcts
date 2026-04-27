"""Value :math:`V_\\phi(s)` for AFlow flow and MCTS backups."""

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """Placeholder scalar :math:`V_\\phi(s)` per state."""

    def __init__(self, state_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.zeros(state.shape[0], 1)
