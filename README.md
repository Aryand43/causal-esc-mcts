# causal-esc-mcts

Scaffold for latency-aware causal MCTS over an Emotional Support Conversation (ESC) MDP with an AFlow-style baseline.

## Requirements

- Python 3.10+
- A TeX distribution for local math compilation/rendering (recommended on Windows: [MiKTeX](https://miktex.org/download))

## Method (ESC + AFlow + Causal MCTS)

The dialogue is framed as an MDP
$$
\mathcal{M}=(\mathcal{S},\mathcal{A},P,R,\gamma)
$$
with state

$$
\begin{aligned}
s_t &= \operatorname{concat}\!\left(\bar{H}_t,\bar{C}_t,e_t,p_t\right),\\
\bar{H}_t &\in \mathbb{R}^{d_H},\qquad
\bar{C}_t \in \mathbb{R}^{d_C},\qquad
e_t \in \mathbb{R}^{d_e},\\
p_t &\in \Delta^2,\qquad
\Delta^2=\left\{p\in\mathbb{R}_{\ge 0}^3:\sum_{i=1}^{3}p_i=1\right\}.
\end{aligned}
$$
The three phase components of $p_t$ correspond to $\{\text{exploration},\text{comforting},\text{action}\}$.

The reward is additively decomposed as
$$
R_t = R_t^{\mathrm{cause}} + R_t^{\mathrm{emotion}} + R_t^{\mathrm{phase}}
$$
where the three terms capture cause-resolution progress, movement toward a target emotion, and phase progression.

A learned transition model supports planning without full LLM rollouts:
$$
\left(\hat{e}_{t+1},\hat{p}_{t+1},\hat{\delta}_{t+1}\right) = f_\theta(s_t,a_t)
$$
with $\hat{e}_{t+1}$ (next-step emotion), $\hat{p}_{t+1}$ (phase logits/distribution), and $\hat{\delta}_{t+1}$ (resolution deltas).

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
