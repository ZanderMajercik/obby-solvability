#!/bin/bash
# The methods and their configs, shared by run_method.sh, run_family.sh and
# run_all_methods.sh. `method_config NAME` prints the config of a method.
GE_METHODS="vanilla random_selection policy_selection only_rl_policy pbge"
CFLGE_METHODS="cflge_scratch cflge_frozen cflge_finetune"
RL_METHODS="rl_inference rl_finetune"
ALL_METHODS="$GE_METHODS $CFLGE_METHODS $RL_METHODS"

method_config() {
  case "$1" in
    vanilla|random_selection|policy_selection|only_rl_policy|pbge) echo "configs/ge/$1.yaml" ;;
    cflge_scratch|cflge_frozen|cflge_finetune) echo "configs/cflge/${1#cflge_}.yaml" ;;
    rl_inference) echo "configs/rl/inference.yaml" ;;
    rl_finetune) echo "configs/rl/finetune.yaml" ;;
    *) return 1 ;;
  esac
}
