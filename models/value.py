"""Value network V_θ(s): state vector → scalar value estimate."""

from __future__ import annotations

import torch
import torch.nn as nn


class _SharedTrunk(nn.Module):
    """Shared feature extractor trunk."""

    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueNetwork(nn.Module):
    """
    Value network V_θ(s).

    Maps a flat state vector s ∈ ℝ^{state_dim} to a scalar value
    estimate in [-1, 1] via tanh output activation.

    Used for MCTS leaf evaluation — replaces high-variance random rollout.

    Architecture: SharedTrunk → Linear(hidden_dim, 1) → Tanh

    Parameters
    ----------
    state_dim  : Flat state vector dimension
    hidden_dim : Hidden layer width (default 256)
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self._trunk = _SharedTrunk(state_dim, hidden_dim)
        self._head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state : Flat state tensor, shape [state_dim] or [B, state_dim]

        Returns
        -------
        Scalar value estimate, shape [..., 1], range [-1, 1]
        """
        h = self._trunk(state)
        return torch.tanh(self._head(h))

    def value(self, state_tensor: torch.Tensor) -> float:
        """Convenience wrapper returning a Python float (no grad)."""
        with torch.no_grad():
            return float(self.forward(state_tensor).item())
