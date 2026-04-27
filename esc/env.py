"""ESC MDP :math:`M=(S,A,P,R)` environment interface (no dynamics implemented)."""

from __future__ import annotations

from typing import Optional

import torch

from esc.action import ESCAction
from esc.state import ESCState


class ESCEnv:
    """Finite-horizon emotional-support MDP; :math:`P` will use learned :math:`f_\\theta` later."""

    def __init__(self, max_horizon: int = 20) -> None:
        self._max_horizon = max_horizon

    @property
    def max_horizon(self) -> int:
        return self._max_horizon

    @property
    def k_hist(self) -> int:
        return 1

    @property
    def d_h(self) -> int:
        return 1

    @property
    def n_causes(self) -> int:
        return 1

    @property
    def d_c(self) -> int:
        return 1

    @property
    def d_e(self) -> int:
        return 1

    @property
    def d_p(self) -> int:
        return 1

    def reset(self, initial_turns: Optional[list[str]] = None) -> ESCState:
        """Sample or build initial state :math:`s_0` (placeholder zeros)."""
        _ = initial_turns
        return ESCState(
            history_embeddings=torch.zeros(1, 1),
            cause_embeddings=torch.zeros(1, 1),
            emotion_vector=torch.zeros(1),
            phase_embedding=torch.zeros(1),
        )

    def step(
        self, state: ESCState, action: ESCAction
    ) -> tuple[ESCState, float, bool, dict]:
        """One MDP step; reward uses decomposition in :mod:`esc.reward` (stub)."""
        _ = action
        return state, 0.0, False, {}
