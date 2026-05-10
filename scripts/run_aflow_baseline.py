"""Script entry for AFlow baseline training on synthetic batches."""

from __future__ import annotations

import os

import torch

from esc.action import ESCAction
from esc.state import ESCState
from models.policy import PolicyNetwork
from models.value import ValueNetwork
from train.trainer_aflow import AFlowTrainer
from train.utils import load_merged_config


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = load_merged_config(root)

    state_dim = ESCState.get_state_dim()
    num_actions = ESCAction.NUM_STRATEGIES * ESCState.N_C

    policy = PolicyNetwork(state_dim=state_dim, action_dim=num_actions)
    value = ValueNetwork(state_dim=state_dim)
    trainer = AFlowTrainer(policy, value, config=config)

    batch_size = int(config.get("batch_size", config.get("batchsize", 4)))
    max_steps = int(config.get("max_steps", config.get("maxsteps", 10)))

    for step in range(max_steps):
        states = torch.randn(batch_size, state_dim)
        metrics = trainer.train_step(states)
        print(f"[AFlow] step={step} loss={metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
