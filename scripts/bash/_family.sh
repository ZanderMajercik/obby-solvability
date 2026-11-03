#!/bin/bash
# Sourced by run_family.sh, which sets FAMILY_METHODS (the family's methods, in
# run order). Runs them one at a time with run_method.sh.
#
#   --eval-set   run the 16-query eval set instead of all 56
#   METHODS="a b"  run only these methods of the family
#
# Runs go to $BASE_LOG_DIR/<variant>/ (default: results/runs/all, or
# results/runs/eval_set with --eval-set). The runs are sequential because
# the times are only comparable with one run on the GPU.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${FAMILY_METHODS:?set FAMILY_METHODS before sourcing}"

REG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --eval-set) REG=--eval-set; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
export BASE_LOG_DIR="${BASE_LOG_DIR:-results/runs/$([ -n "$REG" ] && echo eval_set || echo all)}"
export MADRONA_MWGPU_KERNEL_CACHE="${MADRONA_MWGPU_KERNEL_CACHE:-./build/cache}"
export WANDB_ENABLED="${WANDB_ENABLED:-0}"

RUN=""
for m in ${METHODS:-$FAMILY_METHODS}; do
  case " $FAMILY_METHODS " in
    *" $m "*) RUN="$RUN $m" ;;
    *) [ -z "${METHODS_ANY_FAMILY:-}" ] && { echo "not in this family: $m (choose from: $FAMILY_METHODS)" >&2; exit 2; } ;;
  esac
done

stamp() { echo "### $(date) $*"; }
for m in $RUN; do
  stamp "start $m (out=$BASE_LOG_DIR)"
  scripts/bash/run_method.sh $m $REG
  stamp "done $m"
done
