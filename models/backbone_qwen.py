"""
Stub for the Qwen dialogue backbone (encoding + final response generation).

**Integration point:** This module is the *only* place that must be changed to
wire in the real HuggingFace ``Qwen-1.5-9B-Chat`` weights and tokenizer. All
other ESC / MCTS / training code should consume
``QwenBackbone.encode_dialogue`` / ``QwenBackbone.generate_response`` and stay
agnostic to HF internals.

The canonical SOTA backbone for this project is **Qwen-9B**, per supervisor
instruction (ESC encoding and reply generation at inference time).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from esc.state import ESCState


class QwenBackbone(nn.Module):
    """
    Lazy HF ``AutoModel`` / tokenizer hook for ESC (stub until weights are wired).

    ``encode_dialogue`` returns tensors and cause dicts shaped for
    :meth:`esc.state.ESCState.from_dialogue` (history window, emotion, causes).
    """

    def __init__(self, model_name: str, *, dummy_hist_dim: tuple[int, int] = (4, 768)) -> None:
        super().__init__()
        _ = dummy_hist_dim
        self.model_name = model_name

    def load(self) -> None:
        """Load pretrained weights (stub)."""
        pass

    def encode_dialogue(self, turns: list[str]) -> dict[str, torch.Tensor | list[dict]]:
        """
        Encode dialogue for ESC state construction (stub: correct shapes, zero data).

        Returns
        -------
        dict with keys:

        - ``history`` : ``(ESCState.K_HISTORY_WINDOW, ESCState.D_H)``
        - ``emotion`` : ``(ESCState.D_E,)``
        - ``causes``  : list of ``N_C`` dicts with ``"label"`` (str) and
          ``"embedding"`` tensor ``(ESCState.D_C,)``

        The window keeps the most recent ``K_HISTORY_WINDOW`` turns; if there are
        fewer turns, rows are zero-padded at the beginning. Embeddings are zeros
        until the real model is plugged in.
        """
        k, d_h = ESCState.K_HISTORY_WINDOW, ESCState.D_H
        d_e, d_c, n_c = ESCState.D_E, ESCState.D_C, ESCState.N_C

        n = len(turns)
        history = torch.zeros(k, d_h)
        if n > 0:
            # Stub: rows stay zero; a real backbone would write embeddings into the last
            # min(n, k) rows (most recent turns), with zero padding at the start when n < k.
            pass

        emotion = torch.zeros(d_e)
        causes: list[dict] = [
            {"label": f"cause_{i}", "embedding": torch.zeros(d_c)}
            for i in range(n_c)
        ]
        return {"history": history, "emotion": emotion, "causes": causes}

    def generate_response(self, prompt: str) -> str:
        """Single LLM response conditioned on planner context (stub)."""
        _ = prompt
        return "[Qwen-9B stub reply]"
