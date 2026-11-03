#!/bin/bash
# Sourced by the run scripts. `load_config CONFIG` exports what the config sets:
# its env defaults (SEEDS, TIMEOUT, NUM_WORLDS, GPU_ID, POLICY_CKPT, ...),
# VARIANT, WANDB_TAGS_EXTRA and METHOD_FLAGS (the program's flags).
# The resolver needs PyYAML. CFG_PY sets its Python (default: $PY, else python).
load_config() {
  local out
  out=$("${CFG_PY:-${PY:-python}}" scripts/config.py "$1" --shell) || exit 2
  eval "$out"
}
