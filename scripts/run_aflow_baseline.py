"""Script entry for AFlow baseline training (artifacts or synthetic batches)."""

from __future__ import annotations

import os
from itertools import cycle

import torch
from torch.utils.data import DataLoader

from data.collate import collate_state_tensors
from esc.action import ESCAction
from esc.state import ESCState
from models.policy import PolicyNetwork
from models.value import ValueNetwork
from train.trainer_aflow import AFlowTrainer
from train.train_data import ESCStateTensorDataset
from train.utils import load_merged_config

_SYNTHETIC_WARNING = (
    "WARNING: Running in synthetic smoke-test mode. "
    "This does not reproduce the paper experiments."
)


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

    states_path = os.path.join(root, "artifacts", "states", "train.pt")
    loader_cycle = None
    if os.path.isfile(states_path):
        ds = ESCStateTensorDataset(states_path)
        if len(ds) > 0:
            bs = min(batch_size, len(ds))
            loader = DataLoader(
                ds,
                batch_size=bs,
                shuffle=True,
                collate_fn=collate_state_tensors,
            )
            loader_cycle = cycle(loader)

    if loader_cycle is None:
        print(_SYNTHETIC_WARNING)

    for step in range(max_steps):
        if loader_cycle is not None:
            states = next(loader_cycle)
        else:
            states = torch.randn(batch_size, state_dim)
        metrics = trainer.train_step(states)
        print(f"[AFlow] step={step} loss={metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
