#!/bin/bash
# Run every method of one family, one at a time (the times are only comparable
# with one run on the GPU).
#
#   scripts/bash/run_family.sh <ge|cflge|rl> [--eval-set]
#   METHODS="vanilla pbge" scripts/bash/run_family.sh ge     # only these methods
#
#   ge     GE, GE-RS, GE-PS, GE-RL, PBGE
#   cflge  CFLGE, CFLGE-Frozen, CFLGE-FT
#   rl     RL, RL-FT
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_methods.sh"
case "${1:-}" in
  ge)    FAMILY_METHODS=$GE_METHODS ;;
  cflge) FAMILY_METHODS=$CFLGE_METHODS ;;
  rl)    FAMILY_METHODS=$RL_METHODS ;;
  *) echo "usage: $0 <ge|cflge|rl> [--eval-set]" >&2; exit 2 ;;
esac
shift
source "$(dirname "${BASH_SOURCE[0]}")/_family.sh"
