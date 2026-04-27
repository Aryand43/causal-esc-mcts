"""Training utilities (seed, optimizers; no config I/O in scaffold)."""

from __future__ import annotations

import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Reproducibility for ESC / AFlow experiments (stub)."""
    pass


def load_env_config(path: str) -> dict:
    """Load run config from ``path`` (stub; add YAML/JSON when needed)."""
    _ = path
    return {}


def create_optimizers(models: dict[str, nn.Module], lr: float) -> dict[str, torch.optim.Optimizer]:
    """Build optimizers for each submodule (stub)."""
    raise NotImplementedError
