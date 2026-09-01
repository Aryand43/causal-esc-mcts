# Experiment Summary

Backbone: Qwen2.5-0.5B-Instruct (real, local).
Setup: causal_mcts, flow_ablation, random_floor. 3 seeds, 40 eval instances per seed.

Metrics (3 seeds, mean, n=40 per seed):

| Metric | causal_mcts | flow_ablation | random_floor | Improvement vs random_floor |
|---|---|---|---|---|
| BLEU | 0.270 | 0.340 | 0.266 | +1.5% |
| ROUGE-L | 0.110 | 0.110 | 0.104 | +5.8% |
| BERTScore F1 | 0.703 | 0.701 | 0.703 | +0.0% |
| Distinct-2 | 0.648 | 0.654 | 0.665 | -2.6% |
| Strategy accuracy | 0.183 | 0.042 | 0.208 | -12.0% |

Latency comparison: causal_mcts is about 71x faster per decision than an LLM-rollout approach (0.054s vs 3.82s).
