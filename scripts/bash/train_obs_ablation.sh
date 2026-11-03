#!/bin/bash
# Train one policy and its forward-dynamics encoder (PPO, scripts/simplePPO.py)
# in one observation space.
#
# Usage: scripts/bash/train_obs_ablation.sh <lidaronly|physonly|fullobs>
#
#   lidaronly  --disable-phys-obs --disable-step-obs   lidar (512) + the 35 scalars (the policy's space)
#   physonly   --disable-lidar    --disable-step-obs   entity block (288) + the 35 scalars
#   fullobs    --disable-step-obs                      both channels
#
# All three turn off steps-remaining, so only the lidar and entity channels differ.
#
# Every --save-frequency updates, the run saves:
#   policy   -> $CKPT_DIR/<update>.pth
#   encoder  -> $CKPT_DIR/dynamics/<update>.pth
# simplePPO.py also writes $CKPT_DIR/argv.txt.
#
# The observation flags zero features; they do not change the width (always
# 836). So checkpoints from different spaces have the same shapes, and only
# argv.txt tells them apart: do not rename or merge the checkpoint directories.
#
# Env overrides: PY, NUM_WORLDS, GPU_ID, NUM_UPDATES, SAVE_FREQUENCY,
# EVAL_FREQUENCY, LOG_FREQUENCY, SEED, CKPT_ROOT, RUN_NAME, KERNEL_CACHE,
# DYN_EPOCHS, DYN_MINIBATCH, WANDB_MODE, EXTRA_FLAGS, EXIT_GRACE, plus the eight action
# transform params (DELAY_BUFFER_SIZE, P_DROP, P_RANDOM, P_REPEAT,
# RATE_FREQUENCY, RATE_FREQUENCY_STD, NOISE_MEAN, NOISE_STD).
#
# Smoke test:
#   NUM_WORLDS=1024 NUM_UPDATES=20 SAVE_FREQUENCY=10 EVAL_FREQUENCY=10 \
#   CKPT_ROOT=./ckpts/smoke scripts/bash/train_obs_ablation.sh lidaronly
#
# Detached launch:
#   mkdir -p ckpts/ppo_lidaronly && setsid nohup scripts/bash/train_obs_ablation.sh lidaronly < /dev/null \
#     > ckpts/ppo_lidaronly/train.log 2>&1 &

set -u
set -o pipefail

OBS_SPACE="${1:-}"
case "$OBS_SPACE" in
  lidaronly) OBS_FLAGS="--disable-phys-obs --disable-step-obs" ;;
  physonly)  OBS_FLAGS="--disable-lidar --disable-step-obs" ;;
  fullobs)   OBS_FLAGS="--disable-step-obs" ;;
  *) echo "usage: $0 <lidaronly|physonly|fullobs>" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-python}"
KERNEL_CACHE="${KERNEL_CACHE:-./build/cache}"
NUM_WORLDS="${NUM_WORLDS:-8192}"
GPU_ID="${GPU_ID:-0}"
NUM_UPDATES="${NUM_UPDATES:-150000}"
SAVE_FREQUENCY="${SAVE_FREQUENCY:-100}"
EVAL_FREQUENCY="${EVAL_FREQUENCY:-500}"
LOG_FREQUENCY="${LOG_FREQUENCY:-10}"
SEED="${SEED:-31}"
CKPT_ROOT="${CKPT_ROOT:-./ckpts}"
RUN_NAME="${RUN_NAME:-ppo_${OBS_SPACE}}"
CKPT_DIR="$CKPT_ROOT/$RUN_NAME"
# Encoder training per PPO update: 1 epoch with 4096-sample minibatches costs
# about 16x less time than the argparse defaults (2 epochs, 512).
DYN_EPOCHS="${DYN_EPOCHS:-1}"
DYN_MINIBATCH="${DYN_MINIBATCH:-4096}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"
# RUN_INFERENCE=1 adds a periodic policy evaluation ("INFERENCE EVAL - Success
# Rate" in the log) every EVAL_FREQUENCY updates, in the training simulator.
RUN_INFERENCE="${RUN_INFERENCE:-0}"
if [ "$RUN_INFERENCE" = "1" ]; then
  INFER_FLAGS="--run-inference-during-training --eval-frequency $EVAL_FREQUENCY"
else
  INFER_FLAGS=""
fi
# Action-transform and actuation-noise parameters. simplePPO.py needs all eight
# when --transform-params-file is not given (they have no argparse defaults).
# These values give an identity action transform with a little actuation noise.
DELAY_BUFFER_SIZE="${DELAY_BUFFER_SIZE:-1}"
P_DROP="${P_DROP:-0.0}"
P_RANDOM="${P_RANDOM:-0.0}"
P_REPEAT="${P_REPEAT:-0.0}"
RATE_FREQUENCY="${RATE_FREQUENCY:-1}"
RATE_FREQUENCY_STD="${RATE_FREQUENCY_STD:-0.1}"
NOISE_MEAN="${NOISE_MEAN:-0.0}"
NOISE_STD="${NOISE_STD:-0.01}"
# simplePPO.py calls wandb.init() unconditionally; WANDB_MODE (disabled by
# default) decides where it logs. Set WANDB_MODE=online to log to wandb.
export WANDB_MODE="${WANDB_MODE:-disabled}"

# Level types to train on. The default is all 21 (TrainParameters1.get_generators()).
# Set LEVEL_TYPES to a space-separated list for another set, e.g. the 14 types
# that the policy was trained on:
#   LEVEL_TYPES="Plank TriplePlank ThickPathJumps Tightrope LavaStairs \
#     LavaSpinner MoveAnywhereLavaSpinner DoubleLavaSpinner LavaCheckerboard \
#     RotatingPlatforms WallJumps Truss LavaRingDrop SideToSideBlocks"
# simplePPO.py records the list in argv.txt.
_LT_OVERRIDE="${LEVEL_TYPES:-}"
if [ -n "$_LT_OVERRIDE" ]; then
  read -r -a LEVEL_TYPES <<< "$_LT_OVERRIDE"
else
  LEVEL_TYPES=(
    Plank TriplePlank ThickPathJumps Tightrope ConveyorTightrope LavaStairs
    LavaJumps LavaDoors LavaSpinner MoveAnywhereLavaSpinner DoubleLavaSpinner
    LavaCheckerboard RotatingPlatforms Trampoline WallJumps Truss LavaRingDrop
    SideToSideBlocks BlockPush OneButtonPressWall DextrousButtonPressWall
  )
fi

mkdir -p "$CKPT_DIR"

echo "=== train_obs_ablation: $OBS_SPACE ==="
echo "  obs flags   : $OBS_FLAGS   (obs_dim is 836 for BOTH spaces)"
echo "  ckpt dir    : $CKPT_DIR"
echo "  run name    : $RUN_NAME"
echo "  worlds      : $NUM_WORLDS on GPU $GPU_ID"
echo "  updates     : $NUM_UPDATES (save every $SAVE_FREQUENCY, eval every $EVAL_FREQUENCY)"
echo "  dynamics    : epochs=$DYN_EPOCHS minibatch=$DYN_MINIBATCH"
echo "  level types : ${#LEVEL_TYPES[@]}"
echo "  python      : $PY"
echo "  wandb mode  : $WANDB_MODE"
echo "  extra       : ${EXTRA_FLAGS:-<none>}"
echo

# Training runs in the background so that this script can stop a process that
# hangs during shutdown: after simplePPO.py prints "TRAINING FINISHED" (every
# checkpoint written), the process gets EXIT_GRACE seconds to exit on its own.
EXIT_GRACE="${EXIT_GRACE:-120}"
TRAIN_LOG="$CKPT_DIR/train_stdout.log"
MADRONA_MWGPU_KERNEL_CACHE="$KERNEL_CACHE" "$PY" -u scripts/simplePPO.py \
  --gpu-sim --gpu-id "$GPU_ID" --num-worlds "$NUM_WORLDS" --reward-mode Dense1 \
  --ckpt-dir "$CKPT_DIR" --run-name "$RUN_NAME" \
  --num-updates "$NUM_UPDATES" --save-frequency "$SAVE_FREQUENCY" \
  --log-frequency "$LOG_FREQUENCY" \
  --steps-per-update 40 --num-bptt-chunks 1 --lr 2e-4 --entropy-loss-coef 0.001 \
  --num-components 1 \
  --level-types "${LEVEL_TYPES[@]}" \
  --random-spawn --waypoint \
  --delay-buffer-size "$DELAY_BUFFER_SIZE" --p-drop "$P_DROP" \
  --p-random "$P_RANDOM" --p-repeat "$P_REPEAT" \
  --rate-frequency "$RATE_FREQUENCY" --rate-frequency-std "$RATE_FREQUENCY_STD" \
  --noise-mean "$NOISE_MEAN" --noise-std "$NOISE_STD" \
  $OBS_FLAGS \
  --train-dynamics --dynamics-pred-hidden 256 --dynamics-lr 1e-4 \
  --dynamics-epochs "$DYN_EPOCHS" --dynamics-minibatch "$DYN_MINIBATCH" \
  $INFER_FLAGS \
  --seed "$SEED" \
  $EXTRA_FLAGS > >(tee "$TRAIN_LOG") 2>&1 &
TRAIN_PID=$!

waited=0
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  if grep -q "^TRAINING FINISHED" "$TRAIN_LOG" 2>/dev/null; then
    waited=$((waited + 1))
    if [ "$waited" -ge "$EXIT_GRACE" ]; then
      echo "[EXIT-HANG] training finished but the process did not exit within ${EXIT_GRACE}s; stopping it (outputs are complete)"
      kill "$TRAIN_PID" 2>/dev/null; sleep 5; kill -9 "$TRAIN_PID" 2>/dev/null
      wait "$TRAIN_PID" 2>/dev/null
      exit 0
    fi
  fi
  sleep 1
done
wait "$TRAIN_PID"
