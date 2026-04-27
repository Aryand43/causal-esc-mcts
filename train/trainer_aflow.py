"""AFlow baseline trainer: :math:`L_{\\mathrm{flow}}+\\lambda L_{\\mathrm{rank}}` from trajectories."""

from __future__ import annotations

from typing import Any

from models.policy import PolicyNetwork
from models.value import ValueNetwork


class AFlowTrainer:
    """Trains :math:`\\pi_\\theta,V_\\phi` with trajectory-level flow consistency (stub)."""

    def __init__(
        self,
        policy: PolicyNetwork,
        value: ValueNetwork,
        config: dict[str, Any],
    ) -> None:
        self.policy = policy
        self.value = value
        self.config = config

    def train_step(self, batch: Any) -> None:
        """One optimization step on an ESC batch (not implemented)."""
        pass
