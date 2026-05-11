"""Training utilities (seed helpers; flat YAML config load, no trajectory I/O)."""

from __future__ import annotations

import os

import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Reproducibility for ESC / AFlow experiments (stub)."""
    _ = seed
    pass


def _parse_scalar(val: str) -> int | float | str:
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def load_env_config(path: str) -> dict:
    """Load a flat ``key: value`` YAML file (one level; no imports)."""
    cfg: dict = {}
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if ":" not in line:
                continue
            key, _, rest = line.partition(":")
            key = key.strip()
            val = rest.strip()
            if not key:
                continue
            cfg[key] = _parse_scalar(val) if val else None
    return cfg


def load_merged_config(root: str) -> dict:
    """Load ``env.yaml`` then ``aflowbaseline.yaml`` (training keys override env keys)."""
    env_cfg = load_env_config(os.path.join(root, "config", "env.yaml"))
    train_cfg = load_env_config(os.path.join(root, "config", "aflowbaseline.yaml"))
    return {**env_cfg, **train_cfg}
