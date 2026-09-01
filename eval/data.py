"""Build held-out evaluation instances from processed ESConv test-split conversations."""

from __future__ import annotations

import os
from typing import Any

from data import iter_jsonl


def build_eval_instances(
    root: str, *, n: int = 40, min_context: int = 2
) -> list[dict[str, Any]]:
    """
    One (context, gold_reply, gold_strategy) triple per test conversation:
    the first supporter turn with >= min_context preceding turns and a
    strategy label that survived normalization.
    """
    jsonl_path = os.path.join(root, "artifacts", "processed", "conversations.jsonl")
    instances: list[dict[str, Any]] = []
    for rec in iter_jsonl(jsonl_path):
        if rec.metadata.get("split") != "test":
            continue
        strategies = rec.annotations.get("turn_strategies") or []
        for t in range(min_context, len(rec.turns)):
            if t >= len(rec.speaker_roles) or rec.speaker_roles[t] != "supporter":
                continue
            gold_strategy = strategies[t] if t < len(strategies) else None
            if gold_strategy is None:
                continue
            instances.append(
                {
                    "conversation_id": rec.conversation_id,
                    "turn_index": t,
                    "context": list(rec.turns[:t]),
                    "gold_reply": rec.turns[t],
                    "gold_strategy": gold_strategy,
                }
            )
            break
        if len(instances) >= n:
            break
    return instances
