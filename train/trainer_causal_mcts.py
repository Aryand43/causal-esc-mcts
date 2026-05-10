"""Causal MCTS trainer: MCTS action selection plus value + AFlow-style regularizers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from esc.state import ESCState
from flow import compute_edge_flow, compute_state_flow, flow_consistency_loss, ranking_loss
from mcts.mcts import MCTS
from models.policy import PolicyNetwork
from models.transition import TransitionModel
from models.value import ValueNetwork


class CausalMCTSTrainer:
    """Joint optimizer over π, V, and f_θ with MCTS-collected rewards."""

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

        lr = float(config.get("learning_rate", config.get("learningrate", 1e-4)))
        self.optimizer = torch.optim.Adam(
            list(policy.parameters())
            + list(value.parameters())
            + list(transition.parameters()),
            lr=lr,
        )

    def train_step(self, states: list[ESCState]) -> dict[str, float]:
        self.mcts.policy_network.eval()
        self.mcts.value_network.eval()

        rewards: list[float] = []
        with torch.no_grad():
            for state in states:
                action = self.mcts.search(state)
                _next_state, reward, _done, _info = self.mcts.env.step(state, action)
                rewards.append(float(reward))

        self.mcts.policy_network.train()
        self.mcts.value_network.train()

        S = torch.stack([s.to_tensor() for s in states])
        R = torch.tensor(rewards, dtype=S.dtype, device=S.device)
        policy_probs = self.mcts.policy_network(S)
        values_2d = self.mcts.value_network(S)
        values = values_2d.squeeze(-1)

        value_loss = F.mse_loss(values, R)

        num_actions = policy_probs.shape[-1]
        V = values_2d.expand(-1, num_actions)
        Q_b = policy_probs
        F_s = compute_state_flow(Q_b, V)
        F_edges = compute_edge_flow(F_s, policy_probs.detach())
        flow_from_policy = compute_edge_flow(
            F_s.detach(), policy_probs.detach()
        ).detach()
        flow_loss = flow_consistency_loss(F_edges, flow_from_policy)

        margin = float(
            self.config.get("gamma_margin", self.config.get("gammamargin", 1.0))
        )
        v_winner = values_2d.mean()
        rank_loss = ranking_loss(
            v_winner,
            torch.zeros_like(v_winner),
            margin=margin,
        )

        w_flow = float(
            self.config.get("flow_loss_weight", self.config.get("flowlossweight", 1.0))
        )
        w_rank = float(
            self.config.get(
                "ranking_loss_weight", self.config.get("rankinglossweight", 1.0)
            )
        )
        loss = value_loss + w_flow * flow_loss + w_rank * rank_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "value_loss": float(value_loss.item()),
            "flow_loss": float(flow_loss.item()),
            "rank_loss": float(rank_loss.item()),
        }
