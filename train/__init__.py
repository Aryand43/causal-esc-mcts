"""Trainer entry points for AFlow and causal MCTS."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from train.trainer_aflow import AFlowTrainer
    from train.trainer_causal_mcts import CausalMCTSTrainer

__all__ = ["AFlowTrainer", "CausalMCTSTrainer"]


def __getattr__(name: str) -> Any:
    if name == "AFlowTrainer":
        from train.trainer_aflow import AFlowTrainer

        return AFlowTrainer
    if name == "CausalMCTSTrainer":
        from train.trainer_causal_mcts import CausalMCTSTrainer

        return CausalMCTSTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
