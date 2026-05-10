"""Causal MCTS trainer: value regression on synthetic batches (stub rollout hooks)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from mcts.mcts import MCTS
from models.policy import PolicyNetwork
from models.transition import TransitionModel
from models.value import ValueNetwork


class CausalMCTSTrainer:
    """Joint optimizer over π, V, f_θ with a placeholder MSE value target."""

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

        lr = config.get("learning_rate", config.get("learningrate", 1e-4))
        self.optimizer = torch.optim.Adam(
            list(policy.parameters())
            + list(value.parameters())
            + list(transition.parameters()),
            lr=lr,
        )

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        states = batch["states"]
        values = self.value(states)
        target = torch.zeros_like(values)

        loss = F.mse_loss(values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": float(loss.item())}
