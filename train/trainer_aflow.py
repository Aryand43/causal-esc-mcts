"""AFlow baseline trainer from synthetic or trajectory batches."""

from __future__ import annotations

from typing import Any

import torch

from flow import compute_edge_flow, compute_state_flow, flow_consistency_loss, ranking_loss
from models.policy import PolicyNetwork
from models.value import ValueNetwork


class AFlowTrainer:
    """Trains π_θ and V_φ with flow consistency and pairwise ranking on values."""

    def __init__(
        self,
        policy: PolicyNetwork,
        value: ValueNetwork,
        config: dict[str, Any],
    ) -> None:
        self.policy = policy
        self.value = value
        self.config = config
        lr = config.get("learning_rate", config.get("learningrate", 1e-4))
        self.optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value.parameters()),
            lr=lr,
        )

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        states = batch["states"]
        Q_b = batch["Q_b"]
        V = batch["V"]
        policy_probs = batch["policy_probs"]

        F_s = compute_state_flow(Q_b, V)
        F_edges = compute_edge_flow(F_s, policy_probs)
        flow_from_policy = F_edges.detach()

        loss_flow = flow_consistency_loss(F_edges, flow_from_policy)

        v_all = self.value(states)
        b = states.shape[0]
        m = b // 2
        if m > 0:
            v_winner = v_all[:m]
            v_loser = v_all[m : 2 * m]
            margin = self.config.get("gamma_margin", self.config.get("gammamargin", 1.0))
            loss_rank = ranking_loss(
                v_winner,
                v_loser,
                margin=float(margin),
            )
        else:
            loss_rank = torch.tensor(0.0, device=states.device)

        w_flow = float(self.config.get("flow_loss_weight", 1.0))
        w_rank = float(self.config.get("ranking_loss_weight", 1.0))
        loss = w_flow * loss_flow + w_rank * loss_rank

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "loss_flow": float(loss_flow.item()),
            "loss_rank": float(loss_rank.item()),
        }
