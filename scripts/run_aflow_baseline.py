"""Script entry for AFlow baseline training on synthetic batches."""

from __future__ import annotations

import os

import torch

from esc.action import ESCAction
from esc.state import ESCState
from models.policy import PolicyNetwork
from models.value import ValueNetwork
from train.trainer_aflow import AFlowTrainer
from train.utils import load_env_config


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(root, "config", "aflow_baseline.yaml")
    config = load_env_config(cfg_path)

    state_dim = ESCState.get_state_dim()
    num_actions = ESCAction.NUM_STRATEGIES * ESCState.N_C

    policy = PolicyNetwork(state_dim=state_dim, action_dim=num_actions)
    value = ValueNetwork(state_dim=state_dim)
    trainer = AFlowTrainer(policy, value, config=config)

    batch_size = int(config.get("batch_size", config.get("batchsize", 4)))

    for step in range(10):
        states = torch.randn(batch_size, state_dim)
        Q_b = torch.randn(batch_size, 1)
        V = torch.randn(batch_size, 1)
        policy_probs = torch.softmax(torch.randn(batch_size, num_actions), dim=-1)
        batch = {
            "states": states,
            "Q_b": Q_b,
            "V": V,
            "policy_probs": policy_probs,
        }
        metrics = trainer.train_step(batch)
        print(f"[AFlow] step={step} loss={metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
