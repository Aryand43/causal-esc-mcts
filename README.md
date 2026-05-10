# causal-esc-mcts

Scaffold for latency-aware causal MCTS over an Emotional Support Conversation (ESC) MDP with an AFlow-style baseline.

## Method (ESC + AFlow + Causal MCTS)

The dialogue is framed as an MDP with state

\[
s_t \;=\; \operatorname{concat}\!\left(\bar{H}_t,\; \bar{C}_t,\; e_t,\; p_t\right),
\]

where:
\[
\bar{H}_t = \text{pooled recent turn encodings}, \quad
\bar{C}_t = \text{pooled causal-graph embeddings}, \quad
e_t = \text{current emotion vector}, \quad
p_t \in \Delta^2 = \text{phase distribution (exploration / comforting / action)}.
\]

The reward is additively decomposed as
\[
R_t \;=\; R_t^{\text{cause}} + R_t^{\text{emotion}} + R_t^{\text{phase}},
\]
capturing cause-resolution progress, movement toward a target emotion, and phase progression.

A learned transition \(f_\theta(s_t, a_t)\) predicts next emotion, phase logits, and resolution deltas so MCTS can plan without full LLM rollouts. At inference time, **Qwen-9B** (stub in `models/backbone_qwen.py`) is intended only for encoding dialogue into this state and for generating the final assistant reply conditioned on the chosen ESC action.

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
