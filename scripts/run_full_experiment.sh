#!/usr/bin/env bash
# Real, bounded end-to-end experiment run: preprocess -> train (2x3 seeds) ->
# eval (3x3 seeds) -> significance -> latency baseline -> qualitative -> cleanup.
#
# Every phase runs under a hard wall-clock cap (macOS has no `timeout(1)`,
# so this uses a perl alarm() wrapper instead) so the run cannot silently
# hang or balloon past the user's "don't run too long" instruction.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

run_capped() {
  local cap_seconds="$1"; shift
  echo "[run_full_experiment] (cap=${cap_seconds}s) $*"
  perl -e 'alarm shift; exec @ARGV' "$cap_seconds" "$@"
  local rc=$?
  if [ "$rc" -eq 142 ]; then
    echo "[run_full_experiment] TIMED OUT after ${cap_seconds}s: $*" >&2
  elif [ "$rc" -ne 0 ]; then
    echo "[run_full_experiment] FAILED (exit $rc): $*" >&2
  fi
  return $rc
}

mkdir -p results/logs results/configs

echo "=== 0. Record resolved config + environment for reproducibility ==="
python -c "
import json, os, subprocess, sys
os.chdir('$ROOT')
from train.utils import load_merged_config
cfg = load_merged_config('$ROOT')
try:
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
except Exception:
    commit = 'unknown'
import torch, transformers
record = {
    'resolved_config': cfg,
    'git_commit': commit,
    'python_version': sys.version,
    'torch_version': torch.__version__,
    'transformers_version': transformers.__version__,
}
with open('results/configs/resolved_config.json', 'w') as f:
    json.dump(record, f, indent=2)
print(json.dumps(record, indent=2))
"

echo "=== 1. Preprocessing (capped subset, real Qwen encoder) ==="
# --max-esconv is a PER-SPLIT cap (train/validation/test each independently
# capped), so 100 here yields up to ~300 conversations total with a real,
# non-empty test split for eval.
run_capped 600 python -m scripts.preprocess_datasets --skip-cornell --max-esconv 100 \
  2>&1 | tee results/logs/preprocess.txt

echo "=== 2. Validate artifacts ==="
run_capped 60 python -m scripts.validate_data_pipeline 2>&1 | tee results/logs/validate.txt
if grep -q "skip train.pt" results/logs/validate.txt; then
  echo "[run_full_experiment] ABORT: train.pt missing, preprocessing failed." >&2
  exit 1
fi

echo "=== 3. Training: causal_mcts + flow_ablation, seeds 42/43/44 ==="
for seed in 42 43 44; do
  run_capped 300 python -m scripts.run_causal_mcts --seed "$seed" \
    2>&1 | tee "results/logs/causal_mcts_seed${seed}_train.txt"
  if grep -q "synthetic smoke-test mode" "results/logs/causal_mcts_seed${seed}_train.txt"; then
    echo "[run_full_experiment] ABORT: causal_mcts seed=$seed used synthetic fallback." >&2
    exit 1
  fi
  run_capped 180 python -m scripts.run_flow_ablation --seed "$seed" \
    2>&1 | tee "results/logs/flow_ablation_seed${seed}_train.txt"
  if grep -q "synthetic smoke-test mode" "results/logs/flow_ablation_seed${seed}_train.txt"; then
    echo "[run_full_experiment] ABORT: flow_ablation seed=$seed used synthetic fallback." >&2
    exit 1
  fi
done

echo "=== 4. Eval timing probe (5 generate() calls) ==="
python -c "
import time, os
os.chdir('$ROOT')
from models.backbone_qwen import QwenBackbone
from train.utils import load_merged_config
cfg = load_merged_config('$ROOT')
bb = QwenBackbone(cfg.get('backbone_model_name', 'Qwen/Qwen2.5-0.5B-Instruct'))
bb.load()
t0 = time.perf_counter()
for _ in range(5):
    bb.generate_response('As the Supporter, respond supportively to someone feeling anxious about work.')
elapsed = (time.perf_counter() - t0) / 5
print(f'PROBE_SECONDS_PER_CALL={elapsed:.3f}')
" 2>&1 | tee results/logs/timing_probe.txt

EVAL_N=40
PROBE_S=$(grep -o 'PROBE_SECONDS_PER_CALL=.*' results/logs/timing_probe.txt | cut -d= -f2)
if [ -n "${PROBE_S:-}" ]; then
  OVER=$(python3 -c "print(1 if float('$PROBE_S') > 2.5 else 0)")
  if [ "$OVER" = "1" ]; then
    EVAL_N=20
    echo "[run_full_experiment] generate() is slow (${PROBE_S}s/call) -> trimming eval set to $EVAL_N"
  fi
fi

echo "=== 5. Eval: causal_mcts / flow_ablation / random_floor x seeds 42/43/44 (n=$EVAL_N) ==="
for seed in 42 43 44; do
  for system in causal_mcts flow_ablation random_floor; do
    run_capped 900 python -m eval.run_eval --system "$system" --seed "$seed" --n "$EVAL_N" \
      2>&1 | tee "results/logs/eval_${system}_seed${seed}.txt"
  done
done

echo "=== 6. Significance aggregation ==="
run_capped 120 python -m eval.significance 2>&1 | tee results/logs/significance.txt

echo "=== 7. Latency baseline (bounded, real Qwen, LAST model use) ==="
run_capped 400 python -m eval.latency_baseline 2>&1 | tee results/logs/latency_baseline.txt

echo "=== 8. Qualitative extraction ==="
run_capped 60 python -m eval.qualitative 2>&1 | tee results/logs/qualitative.txt

echo "=== 9. HF cache cleanup ==="
run_capped 60 python -m scripts.cleanup_hf_cache 2>&1 | tee results/logs/cleanup.txt

echo "=== 10. Offline sanity checks ==="
run_capped 120 pytest -m "not integration" --tb=short -q 2>&1 | tee results/logs/pytest.txt
run_capped 60 ruff check . 2>&1 | tee results/logs/ruff.txt

echo "=== DONE ==="
