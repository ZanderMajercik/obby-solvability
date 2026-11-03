#!/bin/bash
# RL-FT: finetunes the policy on each query with scripts/finetune_ppo.py, once
# per (query, seed). A run stops when the max episode return reaches the solve
# threshold (10) or at TIMEOUT. Runs all 56 queries; --eval-set runs 16.
#
# Each run writes result.json ({solved, solve_update, solve_time_sec, ...}).
# The RL-FT solve rate of a query is the fraction of seeds solved within 90 s,
# as the paper scores it (scripts/figures/make_paper_figs.py).
#
# Settings: configs/rl/finetune.yaml. POLICY_CKPT sets the policy
# checkpoint (default: configs/base.yaml); EXTRA_FINETUNE_FLAGS adds finetune_ppo.py flags (e.g. --disable-step-obs).
# TIMEOUT is in seconds ("300" or "300s"). Do not share the GPU with a training run.
#
# Relative paths (POLICY_CKPT, BASE_LOG_DIR, ...) start at the repo root.
#
# Detached launch:
#   mkdir -p logs_rl && setsid nohup scripts/bash/run_rl_finetune.sh < /dev/null \
#     > logs_rl/rl_finetune.log 2>&1 &

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PY="${PY:-python}"
KERNEL_CACHE="${KERNEL_CACHE:-./build/cache}"
source "$(dirname "${BASH_SOURCE[0]}")/_config.sh"
load_config configs/rl/finetune.yaml
TIMEOUT="${TIMEOUT%s}"              # wall-clock seconds per (query, seed)
EXTRA_FINETUNE_FLAGS="${EXTRA_FINETUNE_FLAGS:-}"

source "$(dirname "${BASH_SOURCE[0]}")/_queries.sh"
for arg in "$@"; do [ "$arg" = "--eval-set" ] && EVAL_SET=1; done
if [ "${EVAL_SET:-0}" = "1" ]; then
  STEMS=("${EVAL_SET_QUERIES[@]}")
else
  STEMS=("${MASTER_QUERIES[@]}")
fi
BASE_LOG_DIR="${BASE_LOG_DIR:-results/runs/$([ "${EVAL_SET:-0}" = 1 ] && echo eval_set || echo all)}"
BASE_LOG="$BASE_LOG_DIR/$VARIANT"
QUERIES=()
for s in "${STEMS[@]}"; do QUERIES+=("$QUERY_ROOT/$s.json"); done

if [ ! -f "$POLICY_CKPT" ]; then
  echo "ERROR: policy checkpoint not found: $POLICY_CKPT (see the README)" >&2
  exit 1
fi

echo "=== RL finetune: ${#QUERIES[@]} queries, ckpt=$POLICY_CKPT, seeds [$SEEDS], timeout=${TIMEOUT}s ==="

run() {
  local QUERY_PATH=$1 SEED=$2
  local REL="${QUERY_PATH#${QUERY_ROOT}/}"; local STEM="${REL%.json}"
  local LOG_DIR="$BASE_LOG/$STEM/seed${SEED}"
  mkdir -p "$LOG_DIR"
  local LOG="$LOG_DIR/seed${SEED}.log"
  local RESULT="$LOG_DIR/result.json"

  # Skip a finished run (result.json with a 'solved' key).
  if [ -f "$RESULT" ] && python3 -c "import json,sys; d=json.load(open('$RESULT')); sys.exit(0 if 'solved' in d else 1)" 2>/dev/null; then
    echo "[SKIP] $STEM seed=$SEED (solved=$(python3 -c "import json; print(json.load(open('$RESULT'))['solved'])" 2>/dev/null))"
    return 0
  fi

  echo "[RUN ] $STEM seed=$SEED  $(date)"
  local TIMER_FLAG="$LOG_DIR/.timer_fired"; rm -f "$TIMER_FLAG"
  MADRONA_MWGPU_KERNEL_CACHE="$KERNEL_CACHE" \
  stdbuf -oL -eL "$PY" -u scripts/finetune_ppo.py \
    $METHOD_FLAGS \
    --query-path "$QUERY_PATH" \
    --seed "$SEED" \
    --gpu-id "$GPU_ID" \
    --result-path "$RESULT" \
    $EXTRA_FINETUNE_FLAGS \
    >> "$LOG" 2>&1 &
  local PY_PID=$!
  # At TIMEOUT: SIGTERM (finetune_ppo.py then writes result.json), SIGKILL 5 s later.
  ( sleep "$TIMEOUT" && touch "$TIMER_FLAG" && kill "$PY_PID" 2>/dev/null; sleep 5 && kill -9 "$PY_PID" 2>/dev/null ) &
  local TIMER_PID=$!
  wait "$PY_PID"
  kill "$TIMER_PID" 2>/dev/null; wait "$TIMER_PID" 2>/dev/null

  # No result.json: decide between a GPU fault, a crash and a timeout.
  if [ ! -f "$RESULT" ]; then
    if grep -q "CUDA unknown error\|initCUDA\|Unable to determine the device handle" "$LOG" 2>/dev/null; then
      # A GPU fault, not a timeout. Write no result, so a relaunch runs it again
      # instead of scoring it as unsolved.
      echo "[GPU-FAULT] $STEM seed=$SEED - no result written so the slot re-runs"
      return 0
    fi
    if [ ! -f "$TIMER_FLAG" ]; then
      # Python stopped before the timer fired: a crash, not a timeout. Write no
      # result, so a relaunch runs it again.
      echo "[CRASH] $STEM seed=$SEED - see $LOG; no result written so the slot re-runs"
      return 0
    fi
    python3 - "$RESULT" "$QUERY_PATH" "$POLICY_CKPT" "$TIMEOUT" <<'PYEOF'
import json, sys
result, query, ckpt, timeout = sys.argv[1:5]
json.dump({'query': query, 'ckpt': ckpt, 'solved': False,
           'solve_update': None, 'solve_time_sec': None,
           'best_reward_max': None, 'total_updates': None,
           'total_time_sec': float(timeout), 'timed_out': True}, open(result, 'w'), indent=2)
PYEOF
    echo "[TIMEOUT] $STEM seed=$SEED - fallback result written"
  fi
  rm -f "$TIMER_FLAG"
}

for SEED in $SEEDS; do
  echo "=== SEED $SEED  $(date) ==="
  for QUERY_PATH in "${QUERIES[@]}"; do run "$QUERY_PATH" "$SEED"; done
done
echo "=== ALL DONE  $(date) ==="
echo "Per-(query,seed) results under: $BASE_LOG/<query>/seed<N>/result.json"
