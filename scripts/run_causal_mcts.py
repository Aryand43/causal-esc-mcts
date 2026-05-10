"""Script entry for causal MCTS training with ESCEnv root states."""

from __future__ import annotations

import os

from esc.action import ESCAction
from esc.env import ESCEnv
from esc.state import ESCState
from mcts.mcts import MCTS
from models.policy import PolicyNetwork
from models.transition import LinearTransitionModel
from models.value import ValueNetwork
from train.trainer_causal_mcts import CausalMCTSTrainer
from train.utils import load_merged_config


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = load_merged_config(root)

    state_dim = ESCState.get_state_dim()
    num_actions = ESCAction.NUM_STRATEGIES * ESCState.N_C

    transition_model = LinearTransitionModel(
        state_dim=state_dim,
        n_causes=ESCState.N_C,
        d_e=ESCState.D_E,
        n_strategies=ESCAction.NUM_STRATEGIES,
    )
    env = ESCEnv(
        transition_model=transition_model,
        config=config,
    )

    policy = PolicyNetwork(state_dim=state_dim, action_dim=num_actions)
    value = ValueNetwork(state_dim=state_dim)
    planner = MCTS(
        env=env,
        policy_network=policy,
        value_network=value,
        config=config,
    )

    trainer = CausalMCTSTrainer(
        policy,
        value,
        transition_model,
        planner,
        config=config,
    )

    batch_size = int(config.get("batch_size", config.get("batchsize", 4)))
    max_steps = int(config.get("max_steps", config.get("maxsteps", 10)))

    for step in range(max_steps):
        states = [env.reset(initial_turns=None) for _ in range(batch_size)]
        metrics = trainer.train_step(states)
        print(f"[CausalMCTS] step={step} loss={metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
