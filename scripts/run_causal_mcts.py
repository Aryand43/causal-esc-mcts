"""Script entry for causal MCTS training on synthetic state batches."""

from __future__ import annotations

import os

import torch

from esc.action import ESCAction
from esc.env import ESCEnv
from esc.state import ESCState
from mcts.mcts import MCTS
from models.policy import PolicyNetwork
from models.transition import LinearTransitionModel
from models.value import ValueNetwork
from train.trainer_causal_mcts import CausalMCTSTrainer
from train.utils import load_env_config


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(root, "config", "aflow_baseline.yaml")
    config = load_env_config(cfg_path)

    state_dim = ESCState.get_state_dim()
    transition_model = LinearTransitionModel(
        state_dim=state_dim,
        n_causes=ESCState.N_C,
        d_e=ESCState.D_E,
        n_strategies=ESCAction.NUM_STRATEGIES,
    )
    env = ESCEnv(
        transition_model=transition_model,
        max_horizon=10,
    )

    num_actions = ESCAction.NUM_STRATEGIES * ESCState.N_C
    policy = PolicyNetwork(state_dim=state_dim, action_dim=num_actions)
    value = ValueNetwork(state_dim=state_dim)
    planner = MCTS(
        env=env,
        policy_network=policy,
        value_network=value,
        num_simulations=int(config.get("num_simulations", 10)),
    )

    trainer = CausalMCTSTrainer(
        policy,
        value,
        transition_model,
        planner,
        config=config,
    )

    batch_size = int(config.get("batch_size", config.get("batchsize", 4)))

    for step in range(10):
        states = torch.randn(batch_size, state_dim)
        metrics = trainer.train_step({"states": states})
        print(f"[CausalMCTS] step={step} loss={metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
