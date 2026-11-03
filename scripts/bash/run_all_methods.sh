#!/bin/bash
# Reproduce the paper and supplement figures, in two stages:
#   1. Runs: the three families run one after another and write to
#      results/runs/all/ (56 queries x seeds 31-35, one run at a time):
#        scripts/bash/run_family.sh ge     GE, GE-RS, GE-PS, GE-RL, PBGE
#        scripts/bash/run_family.sh cflge  CFLGE, CFLGE-Frozen, CFLGE-FT
#        scripts/bash/run_family.sh rl     RL, RL-FT
#      Each family also runs alone, with the same arguments.
#   2. Figures: scripts/figures/make_paper_figs.py reads the runs and writes the
#      figures and CSVs to results/figs/all/. With --eval-set: .../eval_set/.
#
#   scripts/bash/run_all_methods.sh                       # all 56 queries
#   scripts/bash/run_all_methods.sh --eval-set          # 16-query eval set
#   scripts/bash/run_all_methods.sh --figs-only           # stage 2 only, from existing runs
#   METHODS="vanilla rl_inference" scripts/bash/run_all_methods.sh
#
# Settings come from configs/ (see configs/README.md). Every run has a 300 s
# budget. A relaunch skips the runs that finished.
# Env overrides: OUT (runs), FIGS, METHODS, PY, and the configs' env (SEEDS,
# QUERY_LIST, POLICY_CKPT, DYN_CKPT, GPU_ID, NUM_WORLDS, ...).
set -u
cd "$(dirname "$0")/../.."

SET=all
ARGS=()
FIGS_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --eval-set) SET=eval_set; ARGS+=(--eval-set); shift ;;
    --figs-only) FIGS_ONLY=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
OUT="${OUT:-results/runs/$SET}"
FIGS="${FIGS:-results/figs/$SET}"
PY="${PY:-python}"
export BASE_LOG_DIR="$OUT" METHODS_ANY_FAMILY=1
source scripts/bash/_methods.sh
for m in ${METHODS:-}; do
  [[ " $ALL_METHODS " == *" $m "* ]] || { echo "unknown method: $m (choose from: $ALL_METHODS)" >&2; exit 2; }
done

if [ "$FIGS_ONLY" = 0 ]; then
  for fam in ge cflge rl; do
    scripts/bash/run_family.sh $fam "${ARGS[@]}" || exit $?
  done
fi
echo "### $(date) figures: $FIGS"
"${FIG_PY:-$PY}" scripts/figures/make_paper_figs.py --runs "$OUT" --out "$FIGS"
