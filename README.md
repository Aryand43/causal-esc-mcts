# causal-esc-mcts

Latency-aware causal MCTS over an Emotional Support Conversation (ESC) MDP, with an AFlow-style flow baseline. This repository provides offline ESC data ingestion (Hugging Face ESConv and Cornell ESC via ConvoKit), serialized state artifacts, and training entry points that consume those artifacts without coupling search code to dataset backends.

---

## Quick Start

From a fresh clone, run the pipeline in this order:

```bash
pip install -r requirements-data-full.txt
python -m scripts.preprocessdatasets --skip-cornell
python -m scripts.validatedatapipeline
python -m scripts.runaflowbaseline
python -m scripts.runcausalmcts
pytest
```

Or use the canonical end-to-end runner:

```bash
bash scripts/run_end_to_end.sh
```

| Step | Command | What it does |
|------|---------|--------------|
| 1 | `pip install -r requirements-data-full.txt` | Installs PyTorch, `datasets` (ESConv), and ConvoKit (Cornell ESC). |
| 2 | `python -m scripts.preprocessdatasets --skip-cornell` | Downloads ESConv, normalizes dialogs, and writes JSONL + serialized state tensors under `artifacts/`. Use `--skip-cornell` for a lightweight HF-only smoke run; omit it for the full ESConv + Cornell benchmark. |
| 3 | `python -m scripts.validatedatapipeline` | Sanity-checks `artifacts/processed/conversations.jsonl` and `artifacts/states/train.pt` (shapes, bundles, state dim). |
| 4 | `python -m scripts.runaflowbaseline` | Trains the AFlow-style policy/value networks on `artifacts/states/train.pt` (or synthetic batches if artifacts are missing). |
| 5 | `python -m scripts.runcausalmcts` | Trains policy, value, and transition models with MCTS on serialized root states (or env-generated fallbacks if artifacts are missing). |
| 6 | `pytest` | Runs unit tests (integration tests that download corpora are skipped by default). |

**Smoke-test mode:** Default hyperparameters in `config/aflowbaseline.yaml` are intentionally small so the pipeline finishes quickly on any machine. Values such as `maxsteps: 10` and `num_simulations: 10` are for smoke testing only—not paper-scale training. For full experiments, raise these (see [config/README.md](config/README.md); suggested overrides include `maxsteps: 5000`, `num_simulations: 50`, `max_horizon: 20`).

**Synthetic fallback:** If `artifacts/states/train.pt` is missing or empty, the training scripts print a warning and use random or env-generated states. That path verifies the training loop but does **not** reproduce paper experiments.

---

## Artifacts

Preprocessing writes gitignored outputs under `artifacts/`:

| Path | Description |
|------|-------------|
| `artifacts/processed/conversations.jsonl` | Normalized `ConversationRecord` entries (one JSON object per line). |
| `artifacts/states/train.pt` | Training split: state bundles + stacked tensors for dataloaders. |
| `artifacts/states/valid.pt` | Validation split. |
| `artifacts/states/test.pt` | Test split. |

Each `.pt` file contains a `"bundles"` list (for `ESCStateBundleDataset`) and optional `"state_tensors"` (for `ESCStateTensorDataset`).

---

## How this repository maps to the paper

| Paper component | Repository location |
|-----------------|---------------------|
| ESC MDP (state, causal graph, actions, reward) | `esc/` |
| MCTS planner (PUCT, tree search) | `mcts/` |
| Policy, value, transition models | `models/` |
| AFlow-style flow objectives | `flow/`, `train/trainer_aflow.py` |
| Causal MCTS training | `train/trainer_causal_mcts.py`, `scripts/runcausalmcts.py` |
| Offline data pipeline (ESConv + Cornell) | `data/`, `scripts/preprocessdatasets.py` |
| Hyperparameters | `config/env.yaml`, `config/aflowbaseline.yaml` |

The `data/` package owns corpus I/O; trainers read only serialized states under `artifacts/states/`, not live HF or ConvoKit APIs.

---

## Project structure

```text
causal-esc-mcts/
├── data/                    # Offline dataset layer
├── esc/                     # ESCState, ESCEnv, CausalGraph, rewards
├── mcts/                    # MCTS planner, TreeNode, PUCT
├── models/                  # PolicyNetwork, ValueNetwork, TransitionModel
├── flow/                    # AFlow-style flow objectives and losses
├── train/                   # Trainers and PyTorch datasets
├── inference/               # Interactive CLI
├── scripts/
│   ├── preprocessdatasets.py      # canonical alias → preprocess_datasets
│   ├── validatedatapipeline.py    # canonical alias → validate_data_pipeline
│   ├── runaflowbaseline.py        # canonical alias → run_aflow_baseline
│   ├── runcausalmcts.py           # canonical alias → run_causal_mcts
│   └── run_end_to_end.sh          # full pipeline runner
├── artifacts/               # Gitignored outputs (created by preprocessing)
├── config/
└── tests/
```

Legacy underscore module names (`scripts.preprocess_datasets`, etc.) remain available and delegate to the same implementations.

---

## Requirements

- **Python 3.10+** (recommended for ConvoKit / spaCy stacks).
- `requirements-data-full.txt` for ESConv + Cornell; `requirements.txt` if you intentionally skip Cornell.

---

## Subdirectory documentation

| Directory | README |
|-----------|--------|
| `esc/` | [ESC MDP](esc/README.md) |
| `mcts/` | [MCTS planner](mcts/README.md) |
| `models/` | [Networks and backbone](models/README.md) |
| `data/` | [Data pipeline](data/README.md) |
| `flow/` | [Flow objectives](flow/README.md) |
| `train/` | [Trainers and datasets](train/README.md) |
| `inference/` | [Interactive CLI](inference/README.md) |
| `scripts/` | [Entry points](scripts/README.md) |
| `config/` | [YAML configuration](config/README.md) |
| `tests/` | [Test suite](tests/README.md) |
| `utils/` | [Seed and logging](utils/README.md) |

---

## CI

GitHub Actions runs lint (`ruff`) and unit tests (`pytest -m "not integration"`) on Python 3.10 and 3.11. See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).
