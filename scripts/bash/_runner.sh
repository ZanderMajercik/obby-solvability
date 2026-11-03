#!/bin/bash
# Shared runner for the go_explore_search_solvable.py methods (GE and CFLGE families).
#
# run_method.sh sets CONFIG (a file under configs/) and sources this file. The config
# gives the method's flags, VARIANT and the env defaults (configs/base.yaml).
# --eval-set (or EVAL_SET=1) selects the 16-query subset instead of all 56.
#
# Env overrides: the config's env (SEEDS, TIMEOUT, NUM_WORLDS, GPU_ID,
# POLICY_CKPT, DYN_CKPT, ...), BASE_LOG_DIR, EXTRA_GESS_FLAGS.
#
# Flag order: the config's flags, then EXTRA_GESS_FLAGS. argparse keeps the
# last value of a flag.
#
# TIMEOUT is the wall-clock budget per run, in seconds. A run that hits it is
# killed, and its log gets a [TIMEOUT] line, so a relaunch skips it as finished.
# Exit code 137 counts as a timeout only if the run used its full budget;
# otherwise it is an OOM kill, and a relaunch runs it again.
#
# The script runs from the repo root, so relative paths in env vars start there.

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PY="${PY:-python}"
source "$(dirname "${BASH_SOURCE[0]}")/_config.sh"
: "${CONFIG:?set CONFIG (a file under configs/) before sourcing}"
load_config "$CONFIG"
for arg in "$@"; do [ "$arg" = "--eval-set" ] && EVAL_SET=1; done
# The kernel cache must belong to the build that Python imports (see README).
# Set GESS_ENTRY and KERNEL_CACHE to run against a different build.
GESS_ENTRY="${GESS_ENTRY:-scripts/go_explore_search_solvable.py}"
KERNEL_CACHE="${KERNEL_CACHE:-./build/cache}"
TIMEOUT="${TIMEOUT%s}"
EXTRA_GESS_FLAGS="${EXTRA_GESS_FLAGS:-}"
EVAL_SET="${EVAL_SET:-0}"
# wandb is off by default (--wandb-mode disabled). Set WANDB_ENABLED=1 to log
# each run to WANDB_PROJECT (configs/base.yaml). PBGE starts wandb itself, so it
# always runs with WANDB_ENABLED=0 (no second wandb.init).
WANDB_ENABLED="${WANDB_ENABLED:-0}"
[[ " $METHOD_FLAGS " == *" --policy-based "* ]] && WANDB_ENABLED=0

# The query sets are in _queries.sh, shared with the RL scripts.
source "$(dirname "${BASH_SOURCE[0]}")/_queries.sh"

if [ "$EVAL_SET" = "1" ]; then
  SET_NAME="eval_set"
  BASE_LOG_DIR="${BASE_LOG_DIR:-results/runs/eval_set}"
  QUERIES=("${EVAL_SET_QUERIES[@]}")
else
  SET_NAME="master"
  BASE_LOG_DIR="${BASE_LOG_DIR:-results/runs/all}"
  QUERIES=("${MASTER_QUERIES[@]}")
fi

BASE_LOG="$BASE_LOG_DIR/$VARIANT"
# One CSV row per run (written by the search through --log-path): query, seed,
# time_sec, solution_found, bins_discovered, solve_step, num_*_bins.
RESULTS_CSV="${RESULTS_CSV:-$BASE_LOG/results.csv}"
mkdir -p "$(dirname "$RESULTS_CSV")"

run() {
  local QUERY=$1
  local SEED=$2
  local TAG="${QUERY//\//__}"
  local LOG_DIR="$BASE_LOG/$QUERY/seed${SEED}"
  mkdir -p "$LOG_DIR"
  local LOG="$LOG_DIR/seed${SEED}.log"
  if [ -f "$LOG" ] && grep -qE "GoExplore time|\[TIMEOUT\]" "$LOG"; then
    echo "[SKIP] $QUERY seed=$SEED"
    return 0
  fi
  echo "[RUN ] $QUERY seed=$SEED  $(date)"
  local t0; t0=$(date +%s)
  local WANDB_ARGS
  if [ "$WANDB_ENABLED" = "1" ]; then
    WANDB_ARGS=(--wandb --wandb-project "$WANDB_PROJECT" \
      --wandb-run-name "${VARIANT}_${TAG}_seed${SEED}" \
      --wandb-tags "$SET_NAME" $WANDB_TAGS_EXTRA "$TAG" \
      --wandb-group "${SET_NAME}_${VARIANT}")
  else
    WANDB_ARGS=(--wandb-mode disabled)
  fi
  MADRONA_MWGPU_KERNEL_CACHE="$KERNEL_CACHE" timeout --foreground -s TERM -k 10s "${TIMEOUT}s" \
    stdbuf -oL -eL "$PY" -u $GESS_ENTRY \
      $METHOD_FLAGS \
      --gpu-id "$GPU_ID" --seed "$SEED" \
      --num-worlds "$NUM_WORLDS" \
      --query-path "$QUERY_ROOT/${QUERY}.json" \
      --log-path "$RESULTS_CSV" \
      $EXTRA_GESS_FLAGS \
      "${WANDB_ARGS[@]}" \
      2>&1 | tee "$LOG"
  local rc=${PIPESTATUS[0]}
  local used=$(( $(date +%s) - t0 ))
  if [ "$rc" -eq 124 ] || { [ "$rc" -eq 137 ] && [ "$used" -ge "$TIMEOUT" ]; }; then
    echo "[TIMEOUT] $QUERY seed=$SEED killed after ${TIMEOUT}s (rc=$rc)" | tee -a "$LOG"
  fi
}

echo "=== Set: $SET_NAME (${#QUERIES[@]} queries), variant: $VARIANT, seeds: $SEEDS ==="

# One pass over all queries per seed, then the next seed.
for SEED in $SEEDS; do
  for QUERY in "${QUERIES[@]}"; do
    run "$QUERY" "$SEED"
  done
done

echo "=== ALL DONE  $(date) ==="
