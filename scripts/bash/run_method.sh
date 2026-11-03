#!/bin/bash
# Run one method on the 56 queries x SEEDS (or the 16-query set with --eval-set).
#
#   scripts/bash/run_method.sh <method | config.yaml> [--eval-set]
#
# Methods (configs in configs/, see configs/README.md):
#   vanilla           GE: sticky-random actions; bins selected with weight
#                     1/sqrt(times_seen+1).
#   random_selection  GE-RS: after each restore, the policy drives the explore
#                     episode with probability 0.5, else sticky-random actions do.
#   policy_selection  GE-PS: the policy drives the explore episode the first time a
#                     bin is selected after it was found or updated; sticky-random
#                     actions drive later selections.
#   only_rl_policy    GE-RL: the policy drives every explore episode.
#   pbge              PBGE: per query, trains a fresh PPO policy to return to
#                     sampled goal bins, then explores from there.
#   cflge_scratch     CFLGE: the encoder starts from random weights and trains
#                     during the search; states are selected by k-NN density in
#                     its latent space.
#   cflge_frozen      CFLGE-Frozen: the encoder from DYN_CKPT, not trained.
#   cflge_finetune    CFLGE-FT: the encoder from DYN_CKPT, trained further.
#   rl_inference      RL: the policy alone (scripts/bash/run_rl_inference.sh).
#   rl_finetune       RL-FT: PPO finetuning per query (scripts/bash/run_rl_finetune.sh).
# A path to a .yaml config runs that config like the GE and CFLGE methods.
#
# Env overrides: the config's env (SEEDS, TIMEOUT, NUM_WORLDS, GPU_ID,
# POLICY_CKPT, DYN_CKPT, ...), QUERY_LIST, BASE_LOG_DIR, EXTRA_GESS_FLAGS.
#
# Detached launch:
#   mkdir -p results/runs/all && setsid nohup scripts/bash/run_method.sh policy_selection \
#     < /dev/null > results/runs/all/policy_selection.log 2>&1 &
set -u
HERE="$(dirname "${BASH_SOURCE[0]}")"
source "$HERE/_methods.sh"
[ $# -ge 1 ] || { echo "usage: $0 <method | config.yaml> [--eval-set]  (methods: $ALL_METHODS)" >&2; exit 2; }
M=$1; shift
case "$M" in
  rl_inference) exec "$HERE/run_rl_inference.sh" "$@" ;;
  rl_finetune)  exec "$HERE/run_rl_finetune.sh" "$@" ;;
  *.yaml)       CONFIG=$M ;;
  *)            CONFIG=$(method_config "$M") || { echo "unknown method: $M (choose from: $ALL_METHODS)" >&2; exit 2; } ;;
esac
source "$HERE/_runner.sh"
