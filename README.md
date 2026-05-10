# causal-esc-mcts

Scaffold for latency-aware causal MCTS over an Emotional Support Conversation (ESC) MDP with an AFlow-style baseline.

## Method (ESC + AFlow + Causal MCTS)

The dialogue is framed as an MDP with state

\[
\begin{aligned}
s_t &= \operatorname{concat}\!\left(\bar{H}_t,\bar{C}_t,e_t,p_t\right),\\
\bar{H}_t &:= \text{pooled recent turn encodings},\\
\bar{C}_t &:= \text{pooled causal-graph embeddings},\\
e_t &:= \text{current emotion vector},\\
p_t &\in \Delta^2,
\end{aligned}
\]
where \(p_t\) is the phase distribution over \(\{\text{exploration},\text{comforting},\text{action}\}\).

The reward is additively decomposed as
\[
R_t = R_t^{\mathrm{cause}} + R_t^{\mathrm{emotion}} + R_t^{\mathrm{phase}},
\]
capturing cause-resolution progress, movement toward a target emotion, and phase progression.

A learned transition model supports planning without full LLM rollouts:
\[
(\hat{e}_{t+1}, \hat{p}_{t+1}, \hat{\delta}_{t+1}) = f_\theta(s_t, a_t),
\]
where \(\hat{e}_{t+1}\), \(\hat{p}_{t+1}\), and \(\hat{\delta}_{t+1}\) are predicted next-step emotion, phase logits, and resolution deltas.

At inference time, **Qwen-9B** (stub in `models/backbone_qwen.py`) is used only to encode dialogue into state features and generate the final assistant reply conditioned on the selected ESC action.

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
