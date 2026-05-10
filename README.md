# causal-esc-mcts

Scaffold for latency-aware causal MCTS over an Emotional Support Conversation (ESC) MDP with an AFlow-style baseline.

## Method (ESC + AFlow + Causal MCTS)

The dialogue is framed as an MDP. The state is

\[
s_t = \text{concat}\bigl(\bar H_t,\,\bar C_t,\,e_t,\,p_t\bigr),
\]

pooling the recent turn encodings \(\bar H_t\) and cause embeddings \(\bar C_t\) from a causal graph, the current emotion vector \(e_t\), and a soft three-way phase distribution \(p_t\) (exploration / comforting / action). The reward decomposes as \(R = R_{\text{cause}} + R_{\text{emotion}} + R_{\text{phase}}\) over cause resolution progress, movement toward a target emotion, and phase progression.

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
