"""Qwen 9B backbone hook for encoding dialogue and final response (Hugging Face later)."""

from __future__ import annotations

import torch
import torch.nn as nn


class QwenBackbone(nn.Module):
    """Lazy HF :math:`\\texttt{AutoModel}` / tokenizer for ESC (not wired in scaffold)."""

    def __init__(self, model_name: str, *, dummy_hist_dim: tuple[int, int] = (4, 768)) -> None:
        super().__init__()
        self.model_name = model_name

    def load(self) -> None:
        """Load pretrained weights (stub)."""
        pass

    def encode_dialogue(self, turns: list[str]) -> torch.Tensor:
        """Dialogue embeddings for :math:`H_t` (stub zeros)."""
        return torch.zeros(1, 1)

    def generate_response(self, prompt: str) -> str:
        """Single LLM response conditioned on :math:`(s_t,a_t)` (stub)."""
        return ""
