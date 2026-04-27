"""Policy :math:`\\pi_\\theta(a\\mid s)` logits over a fixed action dimension."""

import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """Placeholder head for :math:`\\pi_\\theta` (PUCT priors later)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self._action_dim = action_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.zeros(state.shape[0], self._action_dim)
