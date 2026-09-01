"""Cross-seed variance + paired significance testing.

Library functions (paired_bootstrap, wilcoxon_test) plus a CLI
(`python -m eval.significance`) that aggregates results/metrics/*.json
into results/metrics/summary.json: mean +/- std per system across seeds,
and pairwise paired-bootstrap / Wilcoxon tests on the seed=42 per-example
scores (same held-out instances across systems, so the pairing is exact).
"""

from __future__ import annotations

import glob
import json
import os
import random
import statistics
from collections.abc import Sequence

from scipy.stats import wilcoxon

SYSTEMS = ["causal_mcts", "flow_ablation", "random_floor"]
SEEDS = [42, 43, 44]
HEADLINE_METRICS = [
    "bleu",
    "rouge1",
    "rouge2",
    "rougeL",
    "bertscore_f1_mean",
    "distinct_2",
    "strategy_accuracy",
]


def paired_bootstrap(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_resamples: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    assert len(scores_a) == len(scores_b) and len(scores_a) > 0
    n = len(scores_a)
    rng = random.Random(seed)
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    obs_mean = sum(diffs) / n

    resample_means = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        resample_means.append(sum(diffs[i] for i in idx) / n)
    resample_means.sort()
    lo = resample_means[int(0.025 * n_resamples)]
    hi = resample_means[min(int(0.975 * n_resamples), n_resamples - 1)]
    p_value = 2 * min(
        sum(1 for m in resample_means if m <= 0) / n_resamples,
        sum(1 for m in resample_means if m >= 0) / n_resamples,
    )
    return {"mean_diff": obs_mean, "ci_lo": lo, "ci_hi": hi, "bootstrap_p": min(p_value, 1.0)}


def wilcoxon_test(scores_a: Sequence[float], scores_b: Sequence[float]) -> dict[str, float]:
    try:
        stat, p = wilcoxon(scores_a, scores_b)
        return {"statistic": float(stat), "p_value": float(p)}
    except ValueError:
        return {"statistic": float("nan"), "p_value": 1.0}


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics_dir = os.path.join(root, "results", "metrics")

    per_system: dict[str, dict[int, dict]] = {s: {} for s in SYSTEMS}
    for path in glob.glob(os.path.join(metrics_dir, "*_seed*.json")):
        with open(path) as f:
            data = json.load(f)
        sys_name = data["system"]
        seed = data["seed"]
        if sys_name in per_system:
            per_system[sys_name][seed] = data

    summary: dict = {"per_system_mean_std": {}, "pairwise_significance": {}}

    for sys_name in SYSTEMS:
        runs = per_system[sys_name]
        if not runs:
            continue
        entry = {}
        for metric in HEADLINE_METRICS:
            vals = [runs[s][metric] for s in runs if metric in runs[s]]
            if not vals:
                continue
            entry[metric] = {
                "mean": statistics.mean(vals),
                "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                "n_seeds": len(vals),
                "values": vals,
            }
        summary["per_system_mean_std"][sys_name] = entry

    seed_for_sig = 42
    if all(seed_for_sig in per_system[s] for s in SYSTEMS):
        for a, b in [("causal_mcts", "flow_ablation"), ("causal_mcts", "random_floor")]:
            data_a = per_system[a][seed_for_sig]["per_example"]
            data_b = per_system[b][seed_for_sig]["per_example"]
            pair_key = f"{a}_vs_{b}"
            summary["pairwise_significance"][pair_key] = {}
            for metric_key in ["rougeL", "bertscore_f1"]:
                sa = data_a[metric_key]
                sb = data_b[metric_key]
                if len(sa) != len(sb) or not sa:
                    continue
                summary["pairwise_significance"][pair_key][metric_key] = {
                    "bootstrap": paired_bootstrap(sa, sb),
                    "wilcoxon": wilcoxon_test(sa, sb),
                }
            strat_a = [
                1.0 if p == g else 0.0
                for p, g in zip(data_a["predicted_strategy"], data_a["gold_strategy"])
            ]
            strat_b = [
                1.0 if p == g else 0.0
                for p, g in zip(data_b["predicted_strategy"], data_b["gold_strategy"])
            ]
            summary["pairwise_significance"][pair_key]["strategy_correct"] = {
                "bootstrap": paired_bootstrap(strat_a, strat_b),
            }

    out_path = os.path.join(metrics_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[significance] wrote {out_path}")
    for sys_name, entry in summary["per_system_mean_std"].items():
        print(f"[significance] {sys_name}: " + ", ".join(
            f"{k}={v['mean']:.4f}+/-{v['std']:.4f}" for k, v in entry.items()
        ))


if __name__ == "__main__":
    main()
