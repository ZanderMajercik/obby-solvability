#!/bin/bash
# RL: runs the policy alone (no Go-Explore) with scripts/infer.py, once per query.
# The policy restarts when its episodes end; a query stops at MAX_SOLVES total
# solves or at TIMEOUT seconds. Runs all 56 queries; --eval-set runs 16.
# The solve rate of a query is its number of solves within 300 s (at most 5),
# divided by 5 (scripts/figures/make_paper_figs.py).
#
# results.csv columns:
#   solve_count  solves in the parallel worlds, capped at MAX_SOLVES. A solve is a
#                world episode that ends with reward >= ~10 (see infer.py).
#   solveN_wall  seconds from the query's start to its Nth solve (from the
#                epoch on each [SOLVE] line).
#
# Settings: configs/rl/inference.yaml. POLICY_CKPT sets the policy
# checkpoint (default: configs/base.yaml); EXTRA_INFER_FLAGS adds infer.py flags (e.g. --disable-step-obs).
# Do not share the GPU with a training run: 8192 worlds need the memory.
#
# Relative paths (POLICY_CKPT, BASE_LOG_DIR, ...) start at the repo root.
#
# Detached launch:
#   mkdir -p logs_rl && setsid nohup scripts/bash/run_rl_inference.sh < /dev/null \
#     > logs_rl/rl_inference.log 2>&1 &

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PY="${PY:-python}"
KERNEL_CACHE="${KERNEL_CACHE:-./build/cache}"
source "$(dirname "${BASH_SOURCE[0]}")/_config.sh"
load_config configs/rl/inference.yaml   # uses SEED (seed of the first attempt), not SEEDS
TIMEOUT="${TIMEOUT%s}"                  # wall-clock seconds per query
# Extra infer.py flags. With --level-file, infer.py ignores the level-generation
# flags (--waypoint, --random-spawn, --spawn-block-index, --perturb) and
# --transform-params.
EXTRA_INFER_FLAGS="${EXTRA_INFER_FLAGS:-}"

source "$(dirname "${BASH_SOURCE[0]}")/_queries.sh"
for arg in "$@"; do [ "$arg" = "--eval-set" ] && EVAL_SET=1; done
if [ "${EVAL_SET:-0}" = "1" ]; then
  STEMS=("${EVAL_SET_QUERIES[@]}")
else
  STEMS=("${MASTER_QUERIES[@]}")
fi
BASE_LOG_DIR="${BASE_LOG_DIR:-results/runs/$([ "${EVAL_SET:-0}" = 1 ] && echo eval_set || echo all)}"
BASE_LOG="$BASE_LOG_DIR/$VARIANT"
RESULTS_CSV="${RESULTS_CSV:-$BASE_LOG/results.csv}"
QUERIES=()
for s in "${STEMS[@]}"; do QUERIES+=("$QUERY_ROOT/$s.json"); done

if [ ! -f "$POLICY_CKPT" ]; then
  echo "ERROR: policy checkpoint not found: $POLICY_CKPT (see the README)" >&2
  exit 1
fi

mkdir -p "$(dirname "$RESULTS_CSV")"
if [ ! -f "$RESULTS_CSV" ]; then
  echo "query,solve_count,attempts,total_steps,wall_time_sec,timed_out,solve1_step,solve2_step,solve3_step,solve4_step,solve5_step,solve1_wall,solve2_wall,solve3_wall,solve4_wall,solve5_wall" > "$RESULTS_CSV"
fi

echo "=== RL inference: ${#QUERIES[@]} queries, policy=$POLICY_CKPT, max_solves=$MAX_SOLVES, timeout=${TIMEOUT}s ==="

run_query() {
  local QUERY_PATH=$1
  local REL="${QUERY_PATH#${QUERY_ROOT}/}"
  local STEM="${REL%.json}"
  local LOG_DIR="$BASE_LOG/$STEM"
  mkdir -p "$LOG_DIR"
  local LOG="$LOG_DIR/infer.log"

  if [ -f "$LOG" ] && grep -q "^\[RESULT\]" "$LOG"; then
    echo "[SKIP] $STEM"; return 0
  fi

  local solve_count=0 cumulative_steps=0 attempt=0 timed_out=0 start
  start=$(date +%s)
  declare -a ms_step=() ms_wall=()
  echo "=== [START] $STEM  $(date) ===" | tee -a "$LOG"

  while true; do
    local now elapsed remaining rc
    now=$(date +%s); elapsed=$(( now - start )); remaining=$(( TIMEOUT - elapsed ))
    if [ "$remaining" -le 0 ]; then
      timed_out=1; echo "[TIMEOUT] $STEM solve_count=$solve_count" | tee -a "$LOG"; break
    fi
    attempt=$(( attempt + 1 ))
    local tmp cur_seed; tmp=$(mktemp); cur_seed=$(( SEED + attempt - 1 ))
    echo "[ATTEMPT $attempt] $STEM seed=$cur_seed elapsed=${elapsed}s remaining=${remaining}s cumsolves=$solve_count" | tee -a "$LOG"

    MADRONA_MWGPU_KERNEL_CACHE="$KERNEL_CACHE" \
      timeout --foreground -s TERM -k 5s "${remaining}s" \
      stdbuf -oL -eL "$PY" -u scripts/infer.py \
        $METHOD_FLAGS \
        --gpu-id "$GPU_ID" \
        --level-file "$QUERY_PATH" \
        --seed "$cur_seed" \
        $EXTRA_INFER_FLAGS \
      2>&1 | grep -v "^Total steps:" | tee -a "$LOG" > "$tmp"
    rc=${PIPESTATUS[0]}
    # 0: finished (infer.py also exits 0 on SIGTERM). 124 or 137: stopped by
    # `timeout`, but 137 counts only if the attempt used its full budget (else
    # it is the OOM killer). Anything else is a crash: write no [RESULT], so a
    # relaunch runs the query again.
    local used=$(( $(date +%s) - now ))
    if [ "$rc" -ne 0 ] && ! { [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; } || \
       { [ "$rc" -eq 137 ] && [ "$used" -lt "$remaining" ]; }; then
      echo "[CRASH] $STEM attempt=$attempt rc=$rc - no result written, rerun to retry" | tee -a "$LOG"
      rm -f "$tmp"; return 1
    fi

    local attempt_steps attempt_solves
    attempt_steps=$(grep "^Total Steps:" "$tmp" | tail -1 | grep -oP 'Total Steps:\s*\K[0-9]+' || echo 0)
    attempt_solves=$(grep "Policy 0 succeeded in" "$tmp" | tail -1 | grep -oP 'in \K[0-9]+' || echo 0)
    attempt_steps="${attempt_steps:-0}"; attempt_solves="${attempt_solves:-0}"

    while IFS= read -r line; do
      local n s ep; n=$(echo "$line" | grep -oP 'n=\K[0-9]+'); s=$(echo "$line" | grep -oP 'step=\K[0-9]+')
      ep=$(echo "$line" | grep -oP 'epoch=\K[0-9]+' || true)
      local cum_n=$(( solve_count + n ))
      if [ "$cum_n" -le "$MAX_SOLVES" ] && [ -z "${ms_step[$cum_n]:-}" ]; then
        ms_step[$cum_n]=$(( cumulative_steps + s ))
        ms_wall[$cum_n]=$(( ${ep:-$(date +%s)} - start ))
      fi
    done < <(grep "^\[SOLVE\]" "$tmp")
    rm -f "$tmp"

    cumulative_steps=$(( cumulative_steps + attempt_steps )); solve_count=$(( solve_count + attempt_solves ))
    echo "[ATTEMPT_RESULT] attempt=$attempt solves=$attempt_solves cumsolves=$solve_count steps=$attempt_steps cumsteps=$cumulative_steps" | tee -a "$LOG"

    if [ "$solve_count" -ge "$MAX_SOLVES" ]; then
      echo "[DONE] $STEM reached $MAX_SOLVES solves in $attempt attempts" | tee -a "$LOG"; break
    fi
    now=$(date +%s); elapsed=$(( now - start ))
    if [ "$elapsed" -ge "$TIMEOUT" ]; then
      timed_out=1; echo "[TIMEOUT] $STEM solve_count=$solve_count" | tee -a "$LOG"; break
    fi
  done

  local wall_time=$(( $(date +%s) - start ))
  for i in 1 2 3 4 5; do ms_step[$i]="${ms_step[$i]:-}"; ms_wall[$i]="${ms_wall[$i]:-}"; done  # CSV has 5 columns
  echo "[RESULT] $STEM solve_count=${solve_count} attempts=$attempt total_steps=${cumulative_steps} wall_time=${wall_time}s timed_out=${timed_out}  $(date)" | tee -a "$LOG"
  echo "$STEM,$solve_count,$attempt,$cumulative_steps,$wall_time,$timed_out,${ms_step[1]},${ms_step[2]},${ms_step[3]},${ms_step[4]},${ms_step[5]},${ms_wall[1]},${ms_wall[2]},${ms_wall[3]},${ms_wall[4]},${ms_wall[5]}" >> "$RESULTS_CSV"
}

for QUERY_PATH in "${QUERIES[@]}"; do run_query "$QUERY_PATH"; done
echo "=== ALL DONE  $(date) ==="
echo "Results: $RESULTS_CSV"
