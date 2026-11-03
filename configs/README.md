# Configs

Each method's settings are one YAML file. A file names its parent with
`extends:` and holds only what differs from it. `scripts/config.py` merges the
chain and gives the run scripts the program's CLI flags and env defaults; the
programs themselves take only CLI flags. Run a config with
`scripts/bash/run_method.sh <method>` (the names are in the top-level README) or
`scripts/bash/run_method.sh path/to/config.yaml`.

```
base.yaml                     seeds, budget, worlds, GPU, checkpoints (all methods)
├── gess.yaml                 go_explore_search_solvable.py: Go-steps, trajectory cap, bins
│   ├── ge/family.yaml        GE family
│   │   ├── ge/vanilla.yaml               GE
│   │   ├── ge/policy.yaml                + the policy
│   │   │   ├── ge/random_selection.yaml  GE-RS
│   │   │   ├── ge/policy_selection.yaml  GE-PS
│   │   │   └── ge/only_rl_policy.yaml    GE-RL
│   │   └── ge/pbge.yaml                  PBGE
│   └── cflge/family.yaml     CFLGE family: density selection, cell-free archive, encoder
│       ├── cflge/scratch.yaml            CFLGE
│       ├── cflge/frozen.yaml             CFLGE-Frozen
│       └── cflge/finetune.yaml           CFLGE-FT
└── rl/family.yaml            RL family: the policy alone
    ├── rl/inference.yaml                 RL
    └── rl/finetune.yaml                  RL-FT
```

## Keys

| Key | Meaning |
|---|---|
| `extends` | parent file, relative to this one |
| `description` | one line, shown by `--show` |
| `family` | `ge`, `cflge` or `rl` |
| `variant` | output directory under `results/runs/all/` (or `eval_set/`), and the wandb name |
| `wandb_tags` | extra wandb tags |
| `env` | variables the run scripts read. A variable set in the caller's environment wins. |
| `flags` | program flags: `true` → `--flag`, `false` → left out, a list → `--flag a b`, else `--flag value` |

`env` and `flags` merge key by key down the chain (the child wins). A value can
use `${NAME}`: the caller's environment first, else the merged `env`
(e.g. `policy: ${POLICY_CKPT}`).

Per-run flags (`--seed`, `--gpu-id`, `--num-worlds` for gess, the query path,
the log path, wandb) come from the run scripts, not from the configs.

## Commands

```bash
python scripts/config.py configs/ge/policy_selection.yaml --show                    # chain, env, flags
python scripts/config.py --check     # every config resolves
```
