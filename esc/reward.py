"""Reward :math:`R = \\alpha R_{\\mathrm{cause}} + \\beta R_{\\mathrm{emotion}} + \\gamma R_{\\mathrm{phase}}`."""

from __future__ import annotations

from typing import Optional

import torch

from esc.state import ESCState


def get_phase_index(phase_embedding: torch.Tensor) -> int:
    """Scalar phase index from :math:`p_t` (stub)."""
    _ = phase_embedding
    return 0


def compute_reward(
    state: ESCState,
    next_state: ESCState,
    cause_resolution_probs: Optional[torch.Tensor],
) -> float:
    """Decomposed ESC reward for transition :math:`(s_t, a_t, s_{t+1})` (stub)."""
    _ = (state, next_state, cause_resolution_probs)
    return 0.0
