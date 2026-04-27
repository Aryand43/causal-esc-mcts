"""MCTS tree node for PUCT backups over :math:`Q(s,a)`, :math:`N`, priors :math:`P`."""

from __future__ import annotations

from typing import Optional

import torch

from esc.action import ESCAction
from esc.state import ESCState


class TreeNode:
    """Search node; children keyed by :math:`a \\in \\mathcal{A}(s)`."""

    def __init__(
        self,
        state: ESCState,
        parent: Optional[TreeNode] = None,
        *,
        prior: float = 1.0,
    ) -> None:
        self.state = state
        self.parent = parent
        self.children: dict[ESCAction, TreeNode] = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P = prior

    def expand(self, actions: list[ESCAction], priors: torch.Tensor) -> None:
        """Attach children with policy priors :math:`\\pi_\\theta(a\\mid s)` (stub)."""
        pass

    def update(self, value: float) -> None:
        """Backup Monte Carlo return into :math:`N,W,Q` (stub)."""
        pass
