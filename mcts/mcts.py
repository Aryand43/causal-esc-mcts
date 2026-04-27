"""PUCT MCTS using :math:`f_\\theta` rollouts instead of full LLM simulation."""

from esc.action import ESCAction
from esc.state import ESCState
from models.policy import PolicyNetwork
from models.transition import TransitionModel
from models.value import ValueNetwork


class MCTS:
    """Causal MCTS planner; selection uses :math:`Q+ c\\pi\\sqrt{N/(1+N(\\cdot))}` (not implemented)."""

    def __init__(
        self,
        policy: PolicyNetwork,
        value: ValueNetwork,
        transition: TransitionModel,
        c_puct: float = 1.0,
        num_simulations: int = 10,
        *,
        num_strategies: int = 2,
        num_causes: int = 2,
    ) -> None:
        self.policy = policy
        self.value = value
        self.transition = transition

    def search(self, root_state: ESCState) -> ESCAction:
        """Return an action after simulations on :math:`f_\\theta` (stub)."""
        _ = root_state
        return ESCAction(strategy_id=0, cause_index=0)
