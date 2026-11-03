#!/bin/bash
# The benchmark query sets, shared by _runner.sh (GE and CFLGE families) and the
# RL scripts (run_rl_inference.sh, run_rl_finetune.sh).
#
# MASTER_QUERIES: the 56 queries, ordered easy to hard by the paper's GE mean
# time (tiers: data/query_tiers.csv).
# EVAL_SET_QUERIES: a 16-query subset (6 easy, 5 medium, 5 hard).
#
# EVAL_SET=1 selects the subset; EVAL_SET=0 selects all 56.

EVAL_SET_QUERIES=(
  "Designer1/Query1"
  "RunToSafety/Query3"
  "3SpinFwd/Query1"
  "123SpinPlank/Query5"
  "Designer3/Query1"
  "RidePlatDoor/Query2"
  "123SpinPlank/Query4"
  "Designer2/Query2"
  "SpikyLogRoll/Query2"
  "SineWave/Query1"
  "RevSpinLily/Query1"
  "SineWave/Query2"
  "Hard/SpinnerTrussButton"
  "RevSpinLily/Query3"
  "PlatSlideCheck/Query3"
  "Designer3/Query4"
)

MASTER_QUERIES=(
  "Designer1/Query2"
  "Designer1/Query1"
  "RunToSafety/Query3"
  "Designer1/Query3"
  "3SpinFwd/Query3"
  "3SpinFwd/Query1"
  "3SpinFwd/Query2"
  "Designer1/Query5"
  "Designer1/Query4"
  "123SpinPlank/Query3"
  "123SpinPlank/Query5"
  "123SpinPlank/Query1"
  "Designer3/Query1"
  "Designer1/Query6"
  "123SpinPlank/Query2"
  "RidePlatDoor/Query2"
  "SpikyLogRoll/Query1"
  "Designer3/Query2"
  "123SpinPlank/Query4"
  "RunToSafety/Query1"
  "Designer2/Query2"
  "PlatSlideCheck/Query2"
  "SpikyLogRoll/Query2"
  "SlashingLava/Query1"
  "RidePlatDoor/Query1"
  "MultiPlatRide/Query3"
  "Designer2/Query4"
  "123SpinPlank/Query6"
  "SineWave/Query1"
  "Designer2/Query1"
  "MultiPlatRide/Query1"
  "MultiPlatRide/Query2"
  "Designer3/Query6"
  "RunToSafety/Query2"
  "RidePlatDoor/Query3"
  "RevSpinLily/Query1"
  "TwoWaySlowSpin/Query1"
  "Designer3/Query3"
  "Designer3/Query5"
  "SineWave/Query2"
  "TwoWaySlowSpin/Query2"
  "Hard/SpinnerTrussButton"
  "SlashingLava/Query2"
  "Hard/UpAndDownSmall"
  "Hard/BigOne"
  "Hard/UpAndDownFull"
  "Hard/UpAndDownMid"
  "Hard/RevSpinLily9Jumps"
  "RevSpinLily/Query3"
  "RevSpinLily/Query2"
  "PlatSlideCheck/Query3"
  "PlatSlideCheck/Query1"
  "Designer2/Query3"
  "Hard/BlockButtonWallWrap"
  "Designer3/Query4"
  "RidePlatDoor/Query4"
)

# QUERY_LIST="Designer1/Query1 Hard/BigOne" replaces both sets with only these
# queries (for smoke tests and single-query reruns).
if [ -n "${QUERY_LIST:-}" ]; then
  read -ra EVAL_SET_QUERIES <<< "$QUERY_LIST"
  MASTER_QUERIES=("${EVAL_SET_QUERIES[@]}")
fi
