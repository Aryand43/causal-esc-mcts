"""Learned transition :math:`f_\\theta(s_t,a_t)` for low-latency MCTS rollouts."""

import torch
import torch.nn as nn


class TransitionModel(nn.Module):
    """Maps :math:`(s_t,a_t)` to emotion delta, cause resolution, and next phase (stub outputs)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        emotion_dim: int,
        num_causes: int,
        phase_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self._emotion_dim = emotion_dim
        self._num_causes = num_causes
        self._phase_dim = phase_dim

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        b = state.shape[0]
        return {
            "delta_emotion": torch.zeros(b, self._emotion_dim),
            "cause_resolution_probs": torch.zeros(b, self._num_causes),
            "next_phase_logits": torch.zeros(b, self._phase_dim),
        }
