# causal-esc-mcts

Latency-aware **causal MCTS** over an Emotional Support Conversation (ESC) MDP with an AFlow-style baseline.

**Supports EMNLP 2026 submission:** reproducible ESC data ingestion (HF ESConv + Cornell ESC via ConvoKit), offline state serialization, and training scripts that consume artifacts without coupling search code to dataset backends.

---

## Project structure

```text
causal-esc-mcts/
├── data/                    # Offline dataset layer (no HF / ConvoKit at import time in callers)
│   ├── conversation_schema.py
│   ├── dataset_sources.py   # Lazy loaders for ESConv + Cornell
│   ├── preprocess.py
│   ├── build_states.py      # ConversationRecord → ESCState
│   ├── serialization.py
│   └── collate.py
├── esc/                     # ESCState, ESCEnv, CausalGraph, rewards
├── mcts/
├── models/
├── train/
│   ├── train_data.py        # PyTorch datasets over serialized states
│   └── trainer_*.py
├── scripts/
│   ├── preprocess_datasets.py
│   ├── validate_data_pipeline.py
│   ├── run_aflow_baseline.py
│   └── run_causal_mcts.py
├── artifacts/               # Gitignored outputs (create via preprocessing)
│   ├── processed/           # e.g. conversations.jsonl
│   └── states/              # train.pt, valid.pt, test.pt
├── config/
├── tests/
├── requirements.txt
└── requirements-data-full.txt   # Adds ConvoKit / Cornell (recommended)
```

---

## Quick start (paper-style pipeline)

```bash
# 1. Full deps including Cornell ESC / ConvoKit (recommended default)
pip install -r requirements-data-full.txt

# 2. Offline preprocessing → artifacts/processed/*.jsonl + artifacts/states/*.pt
python -m scripts.preprocess_datasets

# 3. Sanity-check serialized tensors when artifacts exist
python -m scripts.validate_data_pipeline

# 4. Short training smoke (uses artifacts/states/train.pt when present)
python -m scripts.run_aflow_baseline
python -m scripts.run_causal_mcts

# 5. Unit tests (skips network-heavy integration tests by default)
pytest
# HF download smoke (optional):  pytest -m integration
```

- **`requirements-data-full.txt`** — includes **`datasets`** (HF ESConv) plus **`convokit`** (Cornell emotional-support). Use **`requirements.txt`** only if you intentionally skip Cornell (`--skip-cornell`).

---

## Datasets

| Source | Access | Role |
|--------|--------|------|
| **ESConv** | Hugging Face [`thu-coai/esconv`](https://huggingface.co/datasets/thu-coai/esconv) | ESC dialogs with strategy labels (`datasets` loader; lazy import in `data/dataset_sources.py`). |
| **Cornell ESC** | ConvoKit [`emotional-support`](https://convokit.cornell.edu/documentation/support.html) | Parallel ESC benchmark (~1.3k sessions); merged into the same `ConversationRecord` schema. |

Together, after normalization and filtering, the pipeline targets on the order of **~2.3k conversations** (sources overlap; exact counts depend on split filters and min-turn thresholds). Training code reads **only** serialized bundles under `artifacts/states/`—not HF or ConvoKit APIs.

---

## GPU cluster deployment

1. Clone this repository.
2. Create a Python **3.10+** environment on the node.
3. `pip install -r requirements-data-full.txt`
4. Run **`python -m scripts.preprocess_datasets`** once (writes caches under ConvoKit / HF directories on shared or local disk).
5. **`python -m scripts.validate_data_pipeline`** to confirm `artifacts/states/*.pt` shapes.
6. **`pytest`** for regression checks (no integration markers unless you opt in).
7. Launch **`python -m scripts.run_aflow_baseline`** / **`python -m scripts.run_causal_mcts`** with merged YAML configs in `config/`.

Hyperparameters (`learning_rate`, `max_steps`, `num_simulations`, `cpuct`, `max_horizon`, loss weights, device, seed, backbone id) live in `config/env.yaml` and `config/aflowbaseline.yaml`.

---

## Requirements

- **Python 3.10+** (recommended for ConvoKit / spaCy stacks).
- Optional: a TeX distribution for local math compilation ([MiKTeX](https://miktex.org/download) on Windows).

---

## Scripts overview

| Script | Purpose |
|--------|---------|
| `scripts.preprocess_datasets` | Download/normalize ESConv + Cornell (Cornell **required by default**; fails with clear instructions unless `--skip-cornell`). |
| `scripts.validate_data_pipeline` | Validates JSONL + `train.pt` when artifacts exist. |
| `scripts.run_aflow_baseline` | AFlow-style trainer; prefers `artifacts/states/train.pt`. |
| `scripts.run_causal_mcts` | MCTS + trainer; prefers bundled roots from artifacts. |
| `inference.interactivecli` | Stub backbone read–eval loop. |

---

## EMNLP submission note

This repository is structured so reviewers can **clone → install → preprocess → validate → train** without patching dataset paths inside `mcts/` or `models/`. The **`data/`** package owns corpus I/O; **`train/train_data.py`** exposes tensor datasets over serialized states only.
