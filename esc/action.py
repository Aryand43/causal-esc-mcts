"""Actions :math:`a_t = (\\sigma, c_i)` in the ESC MDP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from esc.state import ESCState


@dataclass
class ESCAction:
    """Strategy–cause pair :math:`(\\sigma, c_i)` with optional embedding :math:`\\psi(\\sigma,c_i)`."""

    strategy_id: int
    cause_index: int
    embedding: Optional[torch.Tensor] = field(default=None, repr=False)

    def __hash__(self) -> int:
        return hash((self.strategy_id, self.cause_index))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ESCAction):
            return NotImplemented
        return self.strategy_id == other.strategy_id and self.cause_index == other.cause_index


def generate_candidate_actions(
    state: ESCState, num_strategies: int, num_causes: int
) -> list[ESCAction]:
    """Candidate set :math:`\\mathcal{A}(s_t)` for policy :math:`\\pi_\\theta(a\\mid s)` (stub)."""
    _ = (state, num_strategies, num_causes)
    return []
