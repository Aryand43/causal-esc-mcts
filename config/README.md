# config -- YAML Configuration Files

This directory contains the two flat YAML configuration files that control all hyperparameters for training, search, and environment setup. They are loaded and merged by `train/utils.py::load_merged_config()`, with `aflowbaseline.yaml` keys taking precedence over `env.yaml` keys.

---

## `env.yaml` -- Environment and Backbone Config

Controls the hardware target, reproducibility seed, and which backbone model to use.

| Key | Default | Type | Description |
|---|---|---|---|
| `device` | `"cuda"` | str | Torch device for model computation. Set to `"cpu"` on machines without a GPU. |
| `seed` | `42` | int | Global RNG seed passed to `utils.seed.set_global_seed()`. |
| `backbone_model_name` | `"Qwen/Qwen-1.5-9B-Chat"` | str | Hugging Face model ID for the Qwen backbone. Passed to `QwenBackbone.__init__()`. |

**When to change `env.yaml`:**

- Different GPU machines: update `device` to the appropriate CUDA device string (`"cuda:0"`, `"cuda:1"`, etc.).
- Reproducibility experiments: change `seed` to a different integer and re-run preprocessing and training.
- Backbone ablation: change `backbone_model_name` to a different HF model ID when comparing backbone sizes.

---

## `aflowbaseline.yaml` -- Training and Search Hyperparameters

Controls all training loop, MCTS search, and loss function parameters. These are shared between the AFlow baseline and the Causal MCTS trainer.

| Key | Default | Type | Used by | Description |
|---|---|---|---|---|
| `learningrate` | `1e-4` | float | Both trainers | Adam learning rate. |
| `batchsize` | `4` | int | Both scripts | Number of states per training step. |
| `maxsteps` | `10` | int | Both scripts | Total number of gradient steps. Increase significantly for real training runs. |
| `flowlossweight` | `1.0` | float | Both trainers | Weight `w_flow` on the flow consistency loss term. Set to `0.0` to ablate AFlow. |
| `rankinglossweight` | `1.0` | float | Both trainers | Weight `w_rank` on the ranking loss term. Set to `0.0` to ablate the ranking objective. |
| `gammamargin` | `1.0` | float | Both trainers | Margin `m` in the ranking loss `max(0, m - (v_winner - v_loser))`. |
| `num_simulations` | `10` | int | MCTS | Search budget per step. Use 50-200 for real training. |
| `max_horizon` | `10` | int | ESCEnv | Maximum conversation turns per episode. Set to 20 for full-length episodes. |
| `cpuct` | `1.0` | float | MCTS | PUCT exploration constant. Higher values encourage broader search. |

---

## Config loading

Both YAML files use a flat `key: value` format with no nesting or special YAML features. They are parsed by `train/utils.py::load_env_config()`, which does not require PyYAML: it splits each line on the first colon and auto-parses the value as int, float, or string.

Comments in the YAML files (lines starting with `#` or inline `# text`) are stripped before parsing.

The merged config dict is passed directly to trainers, MCTS, and the environment. All consumer code accepts both the underscore form (`learning_rate`) and the no-underscore form (`learningrate`) for multi-word keys, since the YAML files use the no-underscore form.

---

## Recommended values for real training runs

For a full training run targeting EMNLP-quality results:

```yaml
# aflowbaseline.yaml (suggested overrides)
learningrate: 3e-4
batchsize: 32
maxsteps: 5000
flowlossweight: 1.0
rankinglossweight: 0.5
gammamargin: 1.0
num_simulations: 50
max_horizon: 20
cpuct: 1.5
```

The default values in the repository are intentionally small (`maxsteps: 10`, `num_simulations: 10`) so that `python -m scripts.run_aflow_baseline` completes in under one minute on any machine. This makes the smoke-test loop fast without requiring config edits.
