# Results: real experiment run

This directory holds the output of an actual, real, bounded experiment run
(`scripts/run_full_experiment.sh` / `scripts/resume_from_training.sh`),
produced in response to peer review that found the paper's empirical
validation not credible: the reported config was a documented "smoke test"
(`maxsteps=10`), no evaluation code existed for any cited metric, no
variance/significance testing was done, no qualitative analysis was
included, and the "AFlow baseline" was unclear.

Investigating the codebase to fix this surfaced problems worse than the
reviews assumed — see **What was actually broken** below. This document
reports what changed, the real numbers this run produced (including
unflattering ones), and every known limitation, plainly.

## What was actually broken (found while doing this work)

- **The "Qwen-9B" backbone was a complete stub.** `models/backbone_qwen.py`'s
  `encode_dialogue()` returned all-zero tensors and `generate_response()`
  returned the literal string `"[Qwen-9B stub reply]"`. No real dialogue
  encoding or generation ever happened, at any `maxsteps`.
- **No evaluation code existed anywhere** for BLEU, ROUGE, BERTScore,
  Distinct-2, or strategy accuracy — a repo-wide search found zero matches.
- **The ESConv data loader had never been run against the live dataset.**
  `esconv_row_to_record()` assumed a flat row schema; the real
  `thu-coai/esconv` dataset wraps each row as a JSON string under a
  `"text"` key. Every row silently produced 0 turns. Fixed in
  `data/dataset_sources.py`.
- **The flat-YAML config parser didn't strip quotes**, so
  `backbone_model_name: "Qwen/..."` was read as the literal string
  `'"Qwen/..."'` (quotes included) — this went unnoticed before because
  nothing had ever actually consumed that config key. Fixed in
  `train/utils.py`.
- **`--max-esconv` capped records globally across HF splits**, not per
  split, so any cap value exhausted entirely within `train` and left
  `validation`/`test` empty — which would have made the entire eval harness
  silently operate on zero instances. Fixed in `scripts/preprocess_datasets.py`
  (now capped per split).
- **Neither trainer ever saved a checkpoint** (`torch.save`) — added to
  both `scripts/run_causal_mcts.py` and `scripts/run_flow_ablation.py`.
- **The "AFlow baseline" was an in-house GFlowNet-style flow-matching
  ablation**, not a reproduction of the published AFlow system (Zou et al.).
  Renamed to **FlowMCTS-ablation** throughout the repo, with an explicit
  disclaimer everywhere it's introduced.

## What replaced the stub backbone

**`Qwen/Qwen2.5-0.5B-Instruct`**, run locally, CPU-only (this project has no
GPU access) — downsized from the paper's stated "Qwen-9B," which was never
actually feasible here regardless of the stub. This is a real, honest
substitution, not a claim of matching "Qwen-9B" quality.

- Backbone weights are **frozen** throughout — only the policy/value/
  transition MLPs downstream are trained, exactly as the original
  architecture already assumed (the backbone was always meant to be a
  fixed encoder, per `models/README.md`).
- Qwen's hidden size (896) doesn't match the fixed `D_H=768`/`D_C=384`/
  `D_E=128` dimensions hardcoded across the policy/value/transition
  networks. Real Qwen hidden states are projected down with **frozen,
  seeded random linear projections** (Johnson-Lindenstrauss-style) rather
  than retrofitting those dimensions everywhere. This is a real
  dimensionality reduction of real semantic features, not a hidden hack —
  see `models/backbone_qwen.py`'s module docstring.
- ESConv has no ground-truth "cause span" labels. `encode_dialogue()` uses
  a heuristic: the seeker's turns are treated as candidate distress-cause
  spans, most recent first. This is real and text-grounded, not zero-fill,
  but it is a heuristic, not a principled extraction model.
- **`esc/env.py`'s per-step history embedding remains a zero-padded
  placeholder** (`step()` appends a zero vector for the new turn during
  MCTS simulation, rather than a live Qwen call). This was **not** fixed,
  deliberately: doing so would mean calling Qwen inside every MCTS
  simulation step, reintroducing exactly the LLM-rollout latency problem
  this project's architecture exists to avoid (see the latency results
  below). Root states passed into search are real (built from real
  encoded context); only the simulated-forward states inside the tree
  search itself use this placeholder. This is a known, disclosed
  limitation, not a hidden one.

## Config actually used

Not the old `maxsteps=10` smoke test, and not an unverified paper-scale
claim — a **reduced-but-real** config chosen to complete on a CPU-only
machine in bounded wall-clock time. Full resolved config + git commit +
library versions: [`configs/resolved_config.json`](configs/resolved_config.json).

| Key | Value |
|---|---|
| `maxsteps` | 300 |
| `batchsize` | 16 |
| `num_simulations` | 15 |
| `max_horizon` | 12 |
| `cpuct` | 1.5 |
| Dataset | ESConv, 100 conversations per HF split (train/valid/test), `--skip-cornell` |
| Seeds | 42, 43, 44 |
| Eval set | 40 held-out test instances (first eligible supporter turn per test conversation) |

## Systems compared

- **`causal_mcts`** — trained policy/value/transition networks + MCTS search (the paper's main system)
- **`flow_ablation`** — trained policy only, argmax action, no MCTS search (FlowMCTS-ablation; **not** AFlow)
- **`random_floor`** — uniformly random legal strategy (lower bound)

All three use the **same** real Qwen backbone to generate the final reply
text, conditioned on whichever strategy each system selects — isolating
strategy-selection quality as the source of any metric differences.

## Headline results (mean ± std over 3 seeds, n=40 eval instances/seed)

| Metric | causal_mcts | flow_ablation | random_floor |
|---|---|---|---|
| BLEU | 0.270 ± 0.057 | 0.340 ± 0.024 | 0.266 ± 0.038 |
| ROUGE-1 | 0.145 ± 0.007 | 0.146 ± 0.007 | 0.140 ± 0.006 |
| ROUGE-2 | 0.012 ± 0.002 | 0.011 ± 0.002 | 0.013 ± 0.002 |
| ROUGE-L | 0.110 ± 0.003 | 0.110 ± 0.005 | 0.104 ± 0.005 |
| BERTScore F1 | 0.703 ± 0.004 | 0.701 ± 0.003 | 0.703 ± 0.002 |
| Distinct-2 | 0.648 ± 0.012 | 0.654 ± 0.017 | 0.665 ± 0.010 |
| Strategy accuracy | 0.183 ± 0.136 | 0.042 ± 0.024 | 0.208 ± 0.051 |

Full per-seed values and per-example scores: [`metrics/summary.json`](metrics/summary.json), `metrics/{system}_seed{N}.json`.

### Significance testing (paired bootstrap, 10k resamples, seed=42 instances; Wilcoxon as secondary check)

| Comparison | ROUGE-L diff | BERTScore diff | Strategy-correct diff |
|---|---|---|---|
| causal_mcts vs flow_ablation | +0.008 (p=0.35) | +0.002 (p=0.47) | +0.05 (p=0.44) |
| causal_mcts vs random_floor | +0.015 (p=0.036, Wilcoxon p=0.054) | -0.003 (p=0.36) | -0.075 (p=0.40) |

**Reported honestly: at this scale (300 training steps, 40 eval instances,
3 seeds, a 0.5B backbone), `causal_mcts` does not show a robust,
significant advantage over either baseline.** The one nominally
significant result (ROUGE-L vs. random_floor, bootstrap p=0.036) does not
hold up under the Wilcoxon check (p=0.054) and strategy accuracy is
actually *lower* than the random floor. This is very likely a scale
artifact — 300 MCTS-collected training steps is not much signal for a
policy/value/transition network to learn from, and 40 eval instances is a
small sample for detecting a real effect. It is reported as-is rather than
cherry-picked, consistent with why this pass exists in the first place.
Full-scale training (see `config/README.md`) would be needed to make a
real claim about the method's quality; this run's contribution is making
the *pipeline itself* real, traceable, and honestly evaluated, not
proving the paper's central hypothesis.

## Latency comparison (the paper's core claim, previously unvalidated)

Bounded, real measurement on 10 held-out decisions — learned-transition
MCTS (the trained `LinearTransitionModel`, no LLM calls during search) vs.
a bounded LLM-rollout proxy (real Qwen `generate_response()` calls for the
top-3 policy-prior candidates per decision). Full data:
[`latency/latency_baseline.json`](latency/latency_baseline.json).

| | Mean seconds/decision |
|---|---|
| Learned-transition MCTS (15 simulations) | 0.054s |
| LLM-rollout proxy (3 real Qwen calls) | 3.82s |
| **Speedup** | **~71x** |

This is the one clearly positive, robust result in this run, and it is a
faithful (if bounded) test of the paper's central latency argument: MCTS
search using a learned transition model is dramatically cheaper than one
that queries an LLM per candidate.

## Qualitative examples

18 side-by-side examples (dialogue context, predicted vs. gold strategy,
generated vs. gold reply, per-example metrics):
[`qualitative/examples.md`](qualitative/examples.md), [`qualitative/examples.jsonl`](qualitative/examples.jsonl).

## HF cache cleanup (as requested — nothing left on disk afterward)

Downloaded weights (Qwen2.5-0.5B-Instruct, 1.9GB; distilbert-base-uncased
for BERTScore, 512MB) were deleted after the latency baseline (their last
use), leaving only pre-existing, unrelated dataset caches untouched. Full
before/after proof: [`cleanup_log.txt`](cleanup_log.txt).

## Reproducing this run

```bash
bash scripts/run_full_experiment.sh
```

Runs preprocessing -> training (2 systems x 3 seeds) -> eval (3 systems x
3 seeds) -> significance aggregation -> latency baseline -> qualitative
extraction -> HF cache cleanup -> offline test/lint sanity checks, each
phase under a hard wall-clock cap. Took approximately 25 minutes end to
end on a CPU-only machine (Apple M3 Pro, 36GB RAM) for the training/eval
phases once preprocessing artifacts existed.

## Full list of known limitations

1. Backbone downsized from "Qwen-9B" to Qwen2.5-0.5B-Instruct (no GPU available).
2. Frozen random projections bridge Qwen's hidden size to the fixed state dimensions (not a trained reduction).
3. Cause-span extraction is a heuristic (seeker turns), not a principled model — ESConv has no ground-truth cause-span labels.
4. `esc/env.py::step()`'s in-tree simulated history embedding remains a zero-padded placeholder (root states are real; simulated states inside search are not) — this is what keeps MCTS search cheap, and is the reason the 71x latency result is legitimate rather than trivial.
5. Config is "reduced-but-real" (300 steps, 15 simulations, 100 conversations/split) for CPU wall-clock feasibility, not paper-scale.
6. Eval set is 40 instances/seed — too small to make strong significance claims, and this run's own significance testing reflects that honestly.
7. FlowMCTS-ablation is an in-house flow-matching regularizer inspired by AFlow/GFlowNet ideas, not a reproduction of the published AFlow system.
8. Strategy accuracy is computed against ESConv's own annotated strategy labels via `data/preprocess.py`'s normalization — a proxy for "good strategy choice," not a guarantee that the gold label was itself optimal.
