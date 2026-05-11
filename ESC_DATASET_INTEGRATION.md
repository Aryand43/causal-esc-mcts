# ESC Dataset Integration Plan

## Goal

Integrate two emotional support conversation datasets into the existing `aryand43-causal-esc-mcts` repository so the project can be cloned on a GPU cluster, prepared offline, and run end-to-end through MCTS training with clean compilation and smoke tests.

Datasets:
- ESConv from Hugging Face: `thu-coai/esconv`
- Cornell Emotional Support Corpus from ConvoKit: `emotional-support`

This plan emphasizes:
- Separation of concerns.
- Single responsibility.
- Explicit data contracts.
- Offline preprocessing.
- Reproducible cluster execution.
- Minimal coupling between raw data, encoded data, and training code.

---

## Design Principles

### 1. Separation of concerns
Each layer should have one job:
- **Raw dataset loaders** only fetch and normalize source data.
- **Preprocessors** convert raw conversations into a unified internal format.
- **Encoders** map conversation turns into tensors.
- **Dataset classes** expose tensors to training code.
- **Trainers** consume only prepared tensors and never touch dataset-specific APIs.

### 2. Single responsibility
Each module should do one thing well:
- `dataset_sources.py`: load ESConv and Cornell.
- `conversation_schema.py`: define a canonical conversation record.
- `preprocess.py`: clean, align, and serialize data.
- `build_esc_state.py`: convert canonical records to `ESCState`.
- `train_data.py`: expose PyTorch datasets and dataloaders.

### 3. Dependency inversion
Training code should depend on abstract data contracts, not on Hugging Face or ConvoKit directly. This keeps MCTS and AFlow stable even if the source datasets change.

### 4. Reproducibility
All preprocessing should be deterministic given a seed. Save artifacts to disk so GPU-cluster training does not require re-downloading or re-parsing corpora.

---

## Current Repository Context

The repo already contains:
- `esc/` with `ESCState`, `ESCAction`, `CausalGraph`, `ESCEnv`
- `models/` with policy, value, transition, backbone stubs
- `mcts/` with tree search
- `train/` with trainer entry points
- `scripts/` for smoke runs and training

The repository already expects a structured ESC state with:
- history embeddings
- causal graph nodes and edges
- emotion vector
- phase embedding
- turn index
- target emotion

So the dataset integration should produce exactly those inputs, without mixing dataset parsing into model code.

---

## Recommended Folder Layout

Add a new data layer only:

```text
data/
  __init__.py
  conversation_schema.py
  dataset_sources.py
  preprocess.py
  build_states.py
  serialization.py
  collate.py

artifacts/
  raw/
  processed/
  states/

scripts/
  preprocess_datasets.py
  validate_data_pipeline.py
```

Keep dataset code out of `esc/`, `mcts/`, and `models/`. Those directories should stay focused on ESC logic, search, and learning.

---

## Canonical Data Contract

All raw datasets should be normalized into one structure.

### Conversation record
```python
@dataclass
class ConversationRecord:
    conversation_id: str
    source: str
    turns: list[str]
    speaker_roles: list[str]
    annotations: dict[str, Any]
    metadata: dict[str, Any]
```

### Why this matters
This lets ESConv and Cornell share the same downstream path even though their raw formats differ.

### Required fields
- `turns`: ordered utterance strings
- `speaker_roles`: seeker/supporter role per turn
- `annotations`: strategy labels, emotion labels, survey scores if available
- `metadata`: source, split, topic, problem type, emotion type

---

## Dataset Loaders

### ESConv loader
Implement a loader that reads ESConv from Hugging Face and converts each conversation into `ConversationRecord`.

Responsibilities:
- download or cache dataset
- extract dialogue turns in order
- preserve supporter strategy labels if present
- normalize role names to `seeker` and `supporter`

### Cornell loader
Implement a loader using ConvoKit.

Responsibilities:
- download corpus once
- iterate conversations
- extract utterances in order
- capture annotation fields
- normalize metadata

### Important rule
These loaders must not create tensors. They only produce Python records.

---

## Preprocessing Pipeline

Create a deterministic preprocessing step that transforms raw records into clean canonical records.

### Steps
1. Remove empty utterances.
2. Strip excessive whitespace.
3. Enforce alternating roles when possible.
4. Truncate very long conversations to the project horizon if needed.
5. Drop conversations that are too short for ESC training.
6. Standardize strategy labels into your internal strategy set.
7. Save the normalized records to disk.

### Output format
Use JSONL for raw/normalized conversation records.

Recommended path:
```text
artifacts/processed/conversations.jsonl
```

### Validation rules
Each record should satisfy:
- at least 2 turns
- first turn is seeker or clearly labeled
- supporter strategies are mapped if available
- conversation ID is stable
- source is recorded

---

## Mapping to ESC States

Create a dedicated state builder that converts a canonical conversation record into an `ESCState`.

### Responsibilities
- generate history embeddings
- build causal graph nodes from identified problems/causes
- initialize resolution probabilities
- assign target emotion if available
- set phase embedding
- set turn index

### Required separation
This module should know about `ESCState`, `CausalGraph`, and encoders, but it should not know about training loops or MCTS.

### Preferred behavior
If a conversation has no extracted causes:
- create placeholder causes
- keep graph structurally valid
- allow the pipeline to continue

This is important for robustness on cluster runs.

---

## Offline Encoding

Use offline preprocessing to avoid repeated GPU-heavy encoding during training.

### Recommended workflow
1. Load raw datasets.
2. Normalize to canonical records.
3. Run encoder to create tensors.
4. Serialize the resulting state objects or tensor bundles.
5. Train from serialized artifacts.

### Why offline
This reduces:
- startup time
- dependency on external downloads
- runtime variance
- coupling between data loading and model training

### Suggested serialized artifacts
- `artifacts/states/train.pt`
- `artifacts/states/valid.pt`
- `artifacts/states/test.pt`

You can also store:
- conversation metadata JSON
- tensor dictionaries per example
- precomputed state vectors for fast smoke tests

---

## Training Integration

Training code should read only processed artifacts.

### AFlow baseline
The AFlow trainer should consume batched state tensors from a PyTorch dataset.

### Causal MCTS
The MCTS pipeline should be able to reset an environment from a prepared state or from a state generated from a processed conversation.

### Rule
Do not call Hugging Face or ConvoKit inside `train/` or `mcts/`.

### Desired flow
```text
raw dataset -> canonical record -> encoded ESCState -> serialized artifact -> dataset -> trainer -> MCTS
```

---

## Smoke Tests

Add high-signal tests that confirm the pipeline compiles and connects cleanly.

### Test 1: Import test
Verify these import paths succeed:
- `esc.state`
- `esc.env`
- `mcts.mcts`
- `train.traineraflow`
- new `data.*` modules

### Test 2: Dataset loading test
Load one sample from each dataset source and confirm:
- non-empty turns
- source field exists
- normalization succeeds

### Test 3: Preprocessing test
Run preprocessing on a tiny subset and confirm:
- JSONL is written
- records are valid
- no malformed conversations are emitted

### Test 4: State-building test
Build one `ESCState` from a normalized record and confirm:
- tensor shapes match expectations
- `state.totensor()` works
- causal graph resolution tensor has correct length

### Test 5: End-to-end smoke test
Run one short training or simulation step:
- initialize environment
- initialize MCTS
- search one action
- step environment
- confirm no runtime errors

---

## Git Ignore Policy

Use a clean `.gitignore` that excludes:
- Python caches
- virtual environments
- build artifacts
- Hugging Face cache
- ConvoKit cache
- model checkpoints
- dataset downloads
- generated artifacts that are reproducible
- IDE files

### Suggested entries
```gitignore
__pycache__/
*.pyc
*.pyo
*.pyd
*.so
.venv/
venv/
.env
.idea/
.vscode/
.DS_Store

# Python packaging
build/
dist/
*.egg-info/

# Caches
.cache/
pytest_cache/
.mypy_cache/
.ruff_cache/

# ML artifacts
checkpoints/
runs/
logs/
output/
artifacts/raw/
artifacts/processed/
artifacts/states/

# Dataset caches
~/.cache/huggingface/
.huggingface/
.convokit/
```

### Important note
Do not ignore source code, configs, or test files. Only ignore generated or machine-specific files.

---

## Minimal Dependency Split

Keep dependencies separated by purpose.

### Data dependencies
- `datasets`
- `convokit`
- `pandas`
- `pyarrow`
- `tqdm`

### Model/training dependencies
- `torch`
- `transformers`
- `numpy`

### Testing dependencies
- `pytest`

This avoids bloating runtime environments with unnecessary packages.

---

## Cluster Deployment Path

On the GPU cluster, the project should run in this order:

1. Clone repository.
2. Install dependencies.
3. Run dataset preprocessing offline.
4. Verify serialized states exist.
5. Run smoke tests.
6. Launch training or MCTS simulation.

### Expected command sequence
```bash
python -m scripts.preprocess_datasets
pytest
python -m scripts.runaflowbaseline
python -m scripts.runcausalmcts
```

---

## Acceptance Criteria

The integration is complete when all of the following are true:
- ESConv and Cornell load into the same canonical record format.
- Canonical records convert into valid `ESCState` objects.
- Training code consumes only serialized tensors or prepared datasets.
- MCTS can run one full simulation step from a loaded state.
- Smoke tests pass locally.
- A fresh clone on the GPU cluster can reproduce the pipeline without manual dataset surgery.

---

## Implementation Order

1. Add `data/conversation_schema.py`.
2. Add `data/dataset_sources.py`.
3. Add `data/preprocess.py`.
4. Add `data/build_states.py`.
5. Add `data/serialization.py`.
6. Add `scripts/preprocess_datasets.py`.
7. Add smoke tests.
8. Update `.gitignore`.
9. Wire training scripts to processed artifacts.
10. Run end-to-end compile and MCTS smoke test.

---

## Final Principle

The repository should treat datasets as inputs to a stable ESC pipeline, not as special cases embedded in the search or model code. If the data layer stays isolated, the rest of the system can evolve cleanly without breaking the MCTS training path.