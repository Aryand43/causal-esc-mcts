"""ESC state :math:`s_t = \\phi(H_t, G_t)` for the finite-horizon MDP."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ESCState:
    """Pooled history, causes, emotion :math:`e_t`, and phase :math:`p_t` as tensors."""

    history_embeddings: torch.Tensor
    cause_embeddings: torch.Tensor
    emotion_vector: torch.Tensor
    phase_embedding: torch.Tensor

    def to_tensor(self) -> torch.Tensor:
        """Concatenate factors into :math:`s_t \\in \\mathbb{R}^d` (placeholder layout)."""
        return torch.zeros(1)

    @classmethod
    def from_dialogue(cls, turns: list[str]) -> ESCState:
        """Build :math:`s_t` from dialogue via backbone + causal graph (placeholder zeros)."""
        _ = turns
        return cls(
            history_embeddings=torch.zeros(1, 1),
            cause_embeddings=torch.zeros(1, 1),
            emotion_vector=torch.zeros(1),
            phase_embedding=torch.zeros(1),
        )
