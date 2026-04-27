"""Causal MCTS trainer: MCTS targets + flow / ranking losses."""

from __future__ import annotations

from typing import Any

from mcts.mcts import MCTS
from models.policy import PolicyNetwork
from models.transition import TransitionModel
from models.value import ValueNetwork


class CausalMCTSTrainer:
    """Joint training of :math:`\\pi,V,f_\\theta` with MCTS on :math:`f_\\theta` (stub)."""

    def __init__(
        self,
        policy: PolicyNetwork,
        value: ValueNetwork,
        transition: TransitionModel,
        mcts: MCTS,
        config: dict[str, Any],
    ) -> None:
        self.policy = policy
        self.value = value
        self.transition = transition
        self.mcts = mcts
        self.config = config

    def train_step(self, batch: Any) -> None:
        """One step using MCTS rollouts and AFlow-style losses (not implemented)."""
        pass
