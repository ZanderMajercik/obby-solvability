<div align="center">

# Providing Rapid Design Feedback for 3D Obstacle Course Games Using Constrained Solvability Queries

[Zander Majercik](https://zandermajercik.github.io/),
[Sharon Zhang](https://sxzhang25.github.io/),
[William Wang](https://willwng.github.io/),
[Tejan Karmali](https://tejank10.github.io/),
[Fangjun Zhou](https://fangjunzhou.github.io/),
[Yucheng Yuan](https://scholar.google.com/citations?user=2rrJap0AAAAJ&hl=en),
[Jean-Peic Chou](https://jeanpeic.github.io/),
[Maneesh Agrawala](https://graphics.stanford.edu/~maneesh/),
[Kayvon Fatahalian](https://graphics.stanford.edu/~kayvonf/)

**SIGGRAPH Asia 2026**

[🌐 Project page](https://zandermajercik.github.io/interactive-obstacle-course-feedback/) |
[📝 BibTeX](#citation)

<a href="https://zandermajercik.github.io/interactive-obstacle-course-feedback/">
  <img src="https://zandermajercik.github.io/interactive-obstacle-course-feedback/static/images/teaser.png" alt="Teaser" width="90%">
</a>

</div>

## Contents

- [Installation](#installation)
- [Checkpoints](#checkpoints)
- [Experimenting over methods](#experimenting-over-methods)
- [Running a method](#running-a-method)
- [Configs](#configs)
- [Training a policy](#training-a-policy)
- [Queries](#queries)
- [Design tool](#design-tool)
- [License](#license) · [Citation](#citation)

## Installation

**Requirements**
- Linux with an NVIDIA GPU (24 GB for the default 8192 worlds) and the CUDA 12
  toolkit. Tested on an RTX 4090 and an RTX 3090.
- Python 3.11 or 3.12 and PyTorch with CUDA (tested with 2.4.1+cu121).
- CMake ≥ 3.24 and a C++20 compiler, plus the
  [Madrona dependencies](https://github.com/shacklettbp/madrona#dependencies).

```bash
git lfs install                              
git clone --recursive <REPO_URL> mpbRbx      
cd mpbRbx

# A fresh environment
conda create -n mpbrbx python=3.12 -y
conda activate mpbrbx

# PyTorch with CUDA first, so that the next steps do not pull a different build.
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e obby-dsl                      

mkdir build && cd build
# Make sure to build the Python module for the same Python you installed the packages into.
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPython_EXECUTABLE=$(which python) ..
make -j
cd ..
pip install -e .                             # the simulator + training package
```

Set `MADRONA_MWGPU_KERNEL_CACHE` to some path in the environment cache compiled GPU kernels 
for subsequent compilations.
```bash
export MADRONA_MWGPU_KERNEL_CACHE=./build/cache
```

<details>
<summary>Troubleshooting: <code>undefined symbol: __nvJitLinkComplete_12_4</code></summary>

If importing torch fails with this error, add the nvjitlink library that ships
with torch to the loader path:
```bash
export LD_LIBRARY_PATH=<env>/lib/python3.X/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
```
</details>

## Checkpoints

The trained policy checkpoints are stored in this repository with [Git LFS](https://git-lfs.com),
at the paths the run scripts use. If they are missing, ensure you've run:
```bash
git lfs install && git lfs pull
```

| Checkpoint | File | md5 | Used by |
|---|---|---|---|
| Policy (PPO, 150k updates) | `ckpts/policy.pth` | `5b1c859f85c9abad35c79a988bc36f3a` | GE-RS, GE-PS, GE-RL, RL, RL-FT |
| Encoder (forward dynamics) | `ckpts/dynamics_encoder.pth` | `9ac1d731cb8a06dd4cf042bdbada9f41` | CFLGE-Frozen, CFLGE-FT |

GE, CFLGE (scratch) and PBGE need no checkpoint. To use other files, set
`POLICY_CKPT` and `DYN_CKPT`.

## Experimenting over methods

The methods are grouped in three families: GoExplore with a hand-designed binning function, GoExplore with a learned latent, and rollouts with an RL policy (no GoExplore). Each command runs on the 56 queries × seeds 31–35, with a 300 s budget per run:
```bash
scripts/bash/run_family.sh ge       # GE, GE-RS, GE-PS, GE-RL, PBGE
scripts/bash/run_family.sh cflge    # CFLGE, CFLGE-Frozen, CFLGE-FT
scripts/bash/run_family.sh rl       # RL, RL-FT
python scripts/figures/make_paper_figs.py --runs results/runs/all --out results/figs/all
```
`scripts/bash/run_all_methods.sh` runs the three families and then the figures.

## Running a method

```bash
scripts/bash/run_method.sh <method> [--eval-set]
```
`<method>` is a name from the table below, or the path to a config (e.g.
`configs/ge/my_variant.yaml`). `--eval-set` runs the eval set instead of all 56
queries: 16 queries (6 easy, 5 medium, 5 hard, listed in `scripts/bash/_queries.sh`),
written to `results/runs/eval_set/`. The families and `run_all_methods.sh` take it too. To run
only some methods of a family: `METHODS="vanilla pbge" scripts/bash/run_family.sh ge`.

| Method (paper name) | `<method>` | Config |
|---|---|---|
| GE | `vanilla` | `configs/ge/vanilla.yaml` |
| GE-RS | `random_selection` | `configs/ge/random_selection.yaml` |
| GE-PS | `policy_selection` | `configs/ge/policy_selection.yaml` |
| GE-RL | `only_rl_policy` | `configs/ge/only_rl_policy.yaml` |
| PBGE | `pbge` | `configs/ge/pbge.yaml` |
| CFLGE | `cflge_scratch` | `configs/cflge/scratch.yaml` |
| CFLGE-Frozen | `cflge_frozen` | `configs/cflge/frozen.yaml` |
| CFLGE-FT | `cflge_finetune` | `configs/cflge/finetune.yaml` |
| RL | `rl_inference` | `configs/rl/inference.yaml` |
| RL-FT | `rl_finetune` | `configs/rl/finetune.yaml` |

Environment variables override the run settings, for example
`SEEDS=31 QUERY_LIST="Designer1/Query1" scripts/bash/run_method.sh policy_selection`
runs one query and one seed (see [Configs](#configs)). To keep a long run going
after you log out:
```bash
mkdir -p results/runs/all && setsid nohup scripts/bash/run_family.sh ge < /dev/null \
    > results/runs/all/ge.log 2>&1 &
```

## Configs

Every setting lives in `configs/`, in a tree: a config names its parent with
`extends:` and overrides only what differs. `configs/base.yaml` holds what all
methods share (seeds, budget, checkpoints), `configs/gess.yaml` what the GE and
CFLGE families share, and each family directory has one file per method (see
[`configs/README.md`](configs/README.md)). To see what a method runs:
```bash
python scripts/config.py configs/ge/policy_selection.yaml --show
```
A variable set in the environment overrides the config's `env` value, e.g.
`SEEDS=31 POLICY_CKPT=my.pth scripts/bash/run_family.sh ge`. Other
variables: `QUERY_LIST` (space-separated queries, e.g. `"Designer1/Query1"`),
`BASE_LOG_DIR` (output root), and `EXTRA_GESS_FLAGS` (extra flags for the GE and
CFLGE methods).

<details>
<summary>Outputs and logging</summary>

Each run writes to `results/runs/all/<variant>/<query>/seed<N>/`, where
`<variant>` is the config's `variant` (e.g. `goexplore_policy_selection`). Every
Go-Explore run also appends a row to `<variant>/results.csv`.
</details>

## Training a policy

```bash
scripts/bash/train_obs_ablation.sh lidaronly   # lidar only: the policy's observation space
scripts/bash/train_obs_ablation.sh fullobs     # lidar + entity/physics observations
scripts/bash/train_obs_ablation.sh physonly    # entity/physics observations only
```
Each run is 150k PPO updates over 8192 worlds, about 1.5 days on one RTX 4090. The same run trains the
forward-dynamics encoder used by CFLGE-Frozen/FT and saves it to
`<run>/dynamics/`. For a quick check, use `NUM_WORLDS=1024 NUM_UPDATES=20`.

## Queries

We include all queries from the paper's supplementary tables with the same naming convention:
`query_data/<Level>/QueryN.json` is the N-th query posed on that level.

## Design tool

The Roblox Studio design tool is in [`robloxTestHarness/`](robloxTestHarness/README.md).
Its backend is `scripts/slider_backend.py`, which runs GE-PS by default.

## License

MIT; see [`LICENSE`](LICENSE). The Madrona Engine submodule (`external/madrona`)
has its own license.

## Citation

```bibtex
@article{majercik2026solvability,
  author  = {Majercik, Zander and Zhang, Sharon and Wang, William and
             Karmali, Tejan and Zhou, Fangjun and Yuan, Yucheng and
             Chou, Jean-Peic and Agrawala, Maneesh and Fatahalian, Kayvon},
  title   = {Providing Rapid Design Feedback for 3D Obstacle Course Games
             Using Constrained Solvability Queries},
  year    = {2026},
}
```
