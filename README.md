# causal-esc-mcts

Scaffold for latency-aware causal MCTS over an Emotional Support Conversation (ESC) MDP with an AFlow-style baseline.

## Requirements

- Python 3.10+
- A TeX distribution for local math compilation/rendering (recommended on Windows: [MiKTeX](https://miktex.org/download))

## Running the code

```bash
# Run tests
pytest

# AFlow baseline smoke test (synthetic states, flow + ranking losses)
python -m scripts.run_aflow_baseline

# Causal MCTS training smoke test (MCTS search + env step + value / flow losses)
python -m scripts.run_causal_mcts

# Interactive demo with Qwen backbone stub
python -m inference.interactivecli
```

- **pytest** checks imports and the ESC / reward / env / transition / MCTS pipeline.
- **run_aflow_baseline** loads merged `config/env.yaml` and `config/aflowbaseline.yaml`, runs a short AFlow-style training loop on random state batches.
- **run_causal_mcts** builds `ESCEnv` with `LinearTransitionModel`, runs MCTS from batched `env.reset()` roots, and steps the trainers.
- **interactivecli** runs a small read–eval loop: MCTS selects an action, the backbone stub prints a placeholder reply.

Hyperparameters (learning rate, `maxsteps`, `num_simulations`, `cpuct`, `max_horizon`, loss weights, device, seed, backbone model id) live in those YAML files and are read by the env, MCTS, and trainers.
