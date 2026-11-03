import matplotlib.pyplot as plt

import madrona_puzzle_bench
from madrona_puzzle_bench import SimFlags, RewardMode
from madrona_puzzle_bench_learn.action import DiscreteActionDistributions

from madrona_puzzle_bench_learn import LearningState, ActionTransform, profile, AMPState
from madrona_puzzle_bench_learn import (
    train, profile, TrainConfig, PPOConfig, SimInterface, LearningState, EMANormalizer
)
from madrona_scripts.policy import make_policy, setup_obs

from tqdm import tqdm

from madrona_scripts.parameters.params_train import TrainParameters1
from madrona_scripts.parameters.params_transforms import TransformParameters
from generateLevels import loadJsonDescriptionsFromDisk, jsonDescriptionsToTensor, generateJsonLevels

import argparse
import numpy as np
import torch
import warnings
import json
from pathlib import Path

import itertools
import threading
import time
import os
import random

from cflge.density import estimate_density, sample_geometric_with_max, \
    KNN_K_NEIGHBORS, GEO_SAMPLING_P, KNN_MAX_REFERENCE_LATENTS
from cflge.encoders import BackboneEncoder, build_encoder_trainer, load_dynamics_ckpt_encoder
from cflge.kmeans import CellFreeKMeans
from go_binning import get_level_binning_function, BinningFunction, \
    create_binning_function, draw_level, draw_grid, setup_interactive_plot

torch.manual_seed(0)
warnings.filterwarnings("error")

# A query's max_traj_jumps limits jumps (adds a jump-count bin axis) only when
# 0 < max_traj_jumps < LIMITED_JUMPS_MAX; larger values mean unlimited.
LIMITED_JUMPS_MAX = 10
# Bins along the dynamic-block axis when the level has an Unanchored (pushable) part.
DYN_BINS_WITH_UNANCHORED = 8


arg_parser = argparse.ArgumentParser()

# General args
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--seed', type=int, required=True)

# World gen args
arg_parser.add_argument('--reward-mode', type=str, required=True)
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--gpu-sim', action='store_true')
# Observation-space flags. They zero parts of the observation but keep its width
# (obs_dim is always 836), so a checkpoint trained in another space still loads.
arg_parser.add_argument('--disable-phys-obs', action='store_true',
        help='Suppress the entity block (16x[15 phys + 3 type one-hot]).')
arg_parser.add_argument('--disable-lidar', action='store_true',
        help='Zero the 128-ray lidar depths and flatten the hit-type one-hot.')
arg_parser.add_argument('--disable-step-obs', action=argparse.BooleanOptionalAction,
        default=True, help='Zero steps_remaining.')
arg_parser.add_argument('--query-path', type=str)
arg_parser.add_argument('--level-types', type=str, nargs='+',
        help="A set of level types backed by generators from the DSL.")

# Learning args
arg_parser.add_argument('--exploration-steps', type=int, required=True)
arg_parser.add_argument('--max-traj-steps', type=int, required=True)
arg_parser.add_argument('--max-traj-jumps', type=int, default=0)
arg_parser.add_argument('--num-steps', type=int, required=True)
arg_parser.add_argument('--num-x-bins', type=int, required=True)
arg_parser.add_argument('--num-y-bins', type=int, required=True)
arg_parser.add_argument('--num-z-bins', type=int, default=5)
arg_parser.add_argument('--num-t-bins', type=int, default=5)
arg_parser.add_argument('--repeat-p', type=float, default=0.95)
arg_parser.add_argument('--break-when-solved', action='store_true')
arg_parser.add_argument('--store-successful-trajs', action='store_true')

arg_parser.add_argument('--policy', type=str, default="",
        help="Policy checkpoint path.")
arg_parser.add_argument('--ckpt-path', type=str,
        help="PBGE only: starting policy when --policy is not given.")
arg_parser.add_argument('--use-action-percent', type=float, default=0.5)
arg_parser.add_argument('--use-policy-selection', action='store_true')
arg_parser.add_argument('--archive-update', choices=['reward_then_length', 'length'],
        default='length',
        help="When a new state replaces a bin's archived state. length (default): the "
             "shorter trajectory wins and selecting a bin also counts as seen. "
             "reward_then_length: the higher cumulative reward wins, equal reward needs a "
             "strictly shorter trajectory, and times_seen counts explore visits only. "
             "CFLGE ignores this flag and always uses length.")


# Policy-based Go-Explore (PBGE) args
arg_parser.add_argument('--policy-based', action='store_true',
        help='Run policy-based GoExplore (PBGE) with a PPO training loop.')
arg_parser.add_argument('--num-channels', type=int, default=256)
arg_parser.add_argument('--separate-values', action='store_true')
arg_parser.add_argument('--no-level-obs', action='store_true')
arg_parser.add_argument('--ckpt-dir', type=str, default='GoExplorePBGE')
arg_parser.add_argument('--steps-per-update', type=int, default=40)

# State-selection weighting
arg_parser.add_argument('--selection-mode', type=str, default='archive',
        choices=['archive', 'density'],
        help="State-selection weighting. "
             "'archive' = Go-Explore count weights (compute_cell_weights). "
             "'density' = k-NN density estimation over the slot latents (cflge.density).")
arg_parser.add_argument('--knn-k', type=int, default=KNN_K_NEIGHBORS,
        help="k for k-NN density estimation in density mode.")
arg_parser.add_argument('--knn-candidates', type=int, default=-1,
        help="Max candidate entries subsampled from latent buffer for k-NN "
             "(-1 = binning_function.num_bins()).")
arg_parser.add_argument('--knn-candidates-frac', type=float, default=None,
        help="Candidates as fraction of binning_function.num_bins(). Overrides --knn-candidates.")
arg_parser.add_argument('--knn-ref-mode', type=str, default='uniform',
        choices=['uniform', 'stratified'],
        help="How to subsample reference latents in density k-NN. "
             "'uniform' (default) = random sample of max_ref slots. "
             "'stratified' = sample evenly from each non-empty cell-free k-means cluster, "
             "so sparse regions are not under-represented. Requires --cell-free.")
arg_parser.add_argument('--geo-p', type=float, default=GEO_SAMPLING_P,
        help="Geometric distribution p for density-mode bin selection.")
arg_parser.add_argument('--density-latent', type=str, default='backbone',
        choices=['backbone', 'dynamics_ckpt'],
        help="Latent representation for density mode. "
             "'backbone' = policy backbone actor features (requires --policy or "
             "--encoder-init-ckpt for sizing; works with --train-encoder). "
             "'dynamics_ckpt' = frozen encoder loaded from a saved "
             "DynamicsTrainer checkpoint. Requires --dynamics-ckpt-path. "
             "No further training; uses the encoder's latent_dim output as "
             "the slot-pool latent.")
arg_parser.add_argument('--dynamics-ckpt-path', type=str, default=None,
        help="Path to a saved DynamicsTrainer checkpoint (e.g. "
             "ckpts/dynamics_encoder.pth). Required when --density-latent dynamics_ckpt.")
arg_parser.add_argument('--pool-size', type=int, default=77_000,
        help="Slot pool size (POOL_SIZE). Doubles as the latent-buffer size when "
             "--selection-mode density is on, since each slot stores its latent "
             "alongside its checkpoint/trajectory.")
arg_parser.add_argument('--skip-slot-trajectories', action='store_true',
        help="Don't allocate per-slot pos/action/bin trajectory tensors. "
             "Saves ~7 GB of GPU memory at POOL_SIZE=77k, max_traj_steps=3000. "
             "Use when you don't need to replay a slot's past trajectory "
             "(e.g., pure density mode where each restart starts a fresh trajectory).")
arg_parser.add_argument('--cell-free', action='store_true',
        help="Cell-free latent buffer (only meaningful with --selection-mode density). "
             "Drops the one-slot-per-coarse-bin invariant. All alive worlds are "
             "inserted into the slot pool every step; eviction (when the pool fills "
             "up) is by diversity-aware online k-means + age, not by per-bin score.")
arg_parser.add_argument('--cell-free-k', type=int, default=None,
        help="Number of k-means centroids for --cell-free eviction (default: pool_size/50).")
arg_parser.add_argument('--cell-free-small-quantile', type=float, default=0.2,
        help="Fraction of smallest non-empty clusters protected from eviction (default: 0.2).")

# Encoder and forward-dynamics training (CFLGE)
arg_parser.add_argument('--train-encoder', action='store_true',
        help="Train an encoder online via forward-dynamics prediction. Sized from "
             "--policy or --encoder-init-ckpt. With --density-latent backbone, the "
             "density latents come from this encoder.")
arg_parser.add_argument('--encoder-lr', type=float, default=1e-4)
arg_parser.add_argument('--encoder-epochs', type=int, default=2)
arg_parser.add_argument('--encoder-minibatch', type=int, default=512)
arg_parser.add_argument('--encoder-train-every', type=int, default=1,
        help="Train every N inner exploration steps (1 = every step).")
arg_parser.add_argument('--encoder-pred-hidden', type=int, default=256,
        help="Hidden dim for the forward-dynamics MLP.")
arg_parser.add_argument('--encoder-init-ckpt', type=str, default=None,
        help="Path to a saved DynamicsTrainer state_dict (e.g. ckpts/dynamics_encoder.pth) "
             "to initialize the encoder + predictor + action_embed from. "
             "The trainer uses the FULL process_obs dim (~836) to match how the ckpt was trained.")

# wandb logging
arg_parser.add_argument('--wandb', action='store_true',
        help="Log encoder-trainer + GoExplore metrics to wandb.")
arg_parser.add_argument('--wandb-project', type=str, default="obby-solvability")
arg_parser.add_argument('--wandb-run-name', type=str, default=None)
arg_parser.add_argument('--wandb-tags', type=str, nargs='+', default=None)
arg_parser.add_argument('--wandb-mode', type=str, default="online",
        choices=["online", "offline", "disabled"])
arg_parser.add_argument('--wandb-group', type=str, default=None)

class GoExplore:
    def __init__(self,
                 num_worlds: int,
                 exploration_phases: int,
                 exploration_steps: int,
                 max_traj_steps: int,
                 max_traj_jumps: int,
                 sim_flags: SimFlags,
                 reward_mode: RewardMode,
                 json_levels_cpu: np.ndarray,
                 binning_fun: BinningFunction,
                 repeat_p: float,
                 break_when_solved: bool,
                 store_successful_trajs: bool,
                 gpu_sim: bool,
                 gpu_id: int,
                 target_return: float,
                 seed: int,
                 device: torch.device,
                 world_simulator = None,
                 action_policy = None,
                 encoder_backbone = None,
                 use_policy_selection = False,
                 archive_update = 'length',
                 verbose = False,
                 allow_trajectory_recording = False,
                 # Density-mode and cell-free (CFLGE) options; defaults are off.
                 selection_mode: str = 'archive',
                 knn_k: int = KNN_K_NEIGHBORS,
                 knn_candidates: int = -1,
                 knn_candidates_frac: float = None,
                 knn_ref_mode: str = 'uniform',
                 geo_p: float = GEO_SAMPLING_P,
                 pool_size: int = 45_000, # TODO: resotre, 77_000
                 skip_slot_trajectories: bool = False,
                 track_block_trajectories: bool = False,
                 cell_free: bool = False,
                 cell_free_k: int = None,
                 cell_free_small_quantile: float = 0.2,
                 # The caller builds encoder and on_step (see
                 # cell_free_latent_explore_from_set_sim). `encoder` exposes
                 # encode_target(obs) -> latent. on_step(pre_obs, pre_actions,
                 # post_obs) runs once per sim step in explore_from_state.
                 encoder=None,
                 on_step=None,
                 bin_coord_obs=None,
                 wandb_logger=None):
        
        if not world_simulator:
            self.worlds = madrona_puzzle_bench.SimManager(
                exec_mode = madrona_puzzle_bench.madrona.ExecMode.CUDA if gpu_sim else madrona_puzzle_bench.madrona.ExecMode.CPU,
                gpu_id = gpu_id,
                num_worlds = num_worlds,
                rand_seed = seed,
                sim_flags = int(sim_flags),
                reward_mode = reward_mode,
                episode_len = 3000,
                reward_per_dist = 0.05,
                levels_per_episode = 1,
                button_width = 1.3 * 1.0,
                door_width = 20.0 / 3.,
                slack_reward = -0.005,
            )
            self.worlds.init()
        else:
            self.worlds = world_simulator
        self.device = device

        self.exploration_step = 0

        if json_levels_cpu is not None:
            # Load the level, if given. Otherwise the simulator already holds it.
            self.json_levels_cpu = json_levels_cpu
            json_indices = self.worlds.json_index_tensor().to_torch()
            json_levels = self.worlds.json_level_descriptions_tensor().to_torch()
            json_levels_torch = torch.from_numpy(json_levels_cpu).to(device)
            json_levels[:] = json_levels_torch
            json_indices[:] = 0
        # Reset every world to load the level.
        
        self.resets = self.worlds.reset_tensor().to_torch()
        checkpoint_reset = self.worlds.checkpoint_reset_tensor().to_torch()
        checkpoint_reset[..., 0] = 0
        self.resets[..., 0] = 1
        self.worlds.step()  # produces the first checkpoint

        self.num_worlds = num_worlds
        self.max_traj_steps = max_traj_steps
        self.max_traj_jumps = max_traj_jumps
        self.dones = self.worlds.done_tensor().to_torch()
        self.rewards = self.worlds.reward_tensor().to_torch()
        self.checkpoints = self.worlds.checkpoint_tensor().to_torch()
        self.checkpoint_reset = self.worlds.checkpoint_reset_tensor().to_torch()
        self.world_indices = torch.arange(self.num_worlds, device=self.device)

        self.worlds_reset_this_step = torch.ones(self.num_worlds, dtype=torch.bool, device=self.device)
        self.sim_steps_this_explore_step = torch.zeros(self.num_worlds, dtype=torch.int32, device=self.device)
        self.new_sampled_bins = torch.zeros(self.num_worlds, dtype=torch.bool, device=self.device)

        # Observations, agent position and the other tensors the search reads.
        enum_counts_tensor = self.worlds.enum_counts_tensor().to_torch().to(device)
        self.obs, num_obs_features = setup_obs(self.worlds, enum_counts_tensor, bin_coord_obs=bin_coord_obs)
        self.bin_coord_obs = bin_coord_obs
        self.agent_pos = self.worlds.agent_pos_tensor().to_torch()
        self.agent_jumped = self.worlds.agent_jump_performed_tensor().to_torch()
        self.steps_remaining = self.worlds.steps_remaining_tensor().to_torch()
        self.dynamic_block_pos = self.worlds.dynamic_block_pos_tensor().to_torch()
        self.sim_obs = (self.obs, self.agent_pos, self.steps_remaining, self.dynamic_block_pos)
        self.action_policy = action_policy
        # Encoder backbone reference. When encoder_backbone is None (every CLI
        # path), the encoder is sized from action_policy.
        self.encoder_backbone = encoder_backbone if encoder_backbone is not None else action_policy

        # Starting checkpoint and positions, used to replay trajectories later.
        self.starting_checkpoints = self.checkpoints[0].clone()
        self.starting_positions = self.agent_pos.clone()

        self.actions = self.worlds.action_tensor().to_torch()
        self.actions_num_buckets = [2, 8, 3, 2]
        self.repeat_p = repeat_p
        self.target_return = target_return
        self.break_when_solved = break_when_solved
        self.finished = False
        self.solve_step = None
        self.max_return = 0

        self.use_policy_selection = use_policy_selection
        assert archive_update in ('reward_then_length', 'length'), archive_update
        self.archive_update = archive_update

        # Density-mode selection settings.
        self.selection_mode          = selection_mode
        self.knn_k                   = knn_k
        self.knn_candidates          = knn_candidates
        self.knn_candidates_frac     = knn_candidates_frac
        self.knn_ref_mode            = knn_ref_mode
        self.geo_p                   = geo_p
        self.pool_size_arg           = pool_size  # used to size POOL_SIZE below
        self.skip_slot_trajectories  = skip_slot_trajectories
        self.slot_latents            = None       # lazy-allocated in density mode
        self._world_latents          = None       # most recent world latents, set per step
        self.wandb                   = wandb_logger

        # Cell-free buffer (density mode with cell_free=True): the slot pool
        # drops the one-slot-per-bin rule. Every alive world is inserted each
        # step, and eviction uses online k-means plus slot age (self.cf).
        self.cell_free                  = bool(cell_free)

        # Built by the caller; GoExplore only calls them.
        self.encoder              = encoder
        self.on_step              = on_step

        
        # Binning
        self.num_exploration_steps = exploration_steps
        self.num_exploration_phases = exploration_phases
        self.binning_function = binning_fun
        self._num_coarse_bins = binning_fun.num_bins()

        self.POOL_SIZE = self.pool_size_arg
        self.cf = CellFreeKMeans(self.POOL_SIZE, device, k=cell_free_k,
                                 small_quantile=cell_free_small_quantile)
        self.MAX_TRAJ_HISTORY = 3000

        # Slot index -> coarse bin index (-1 = free). This tensor alone
        # defines which slots are occupied.
        self._slot_to_coarse = torch.full(
            (self.POOL_SIZE,), -1, dtype=torch.int64, device=device)

        # Per-bin statistics, indexed by coarse bin.
        _nc = self._num_coarse_bins
        self.bin_times_chosen = torch.zeros(_nc, device=device, dtype=torch.int32)
        self.bin_visited = torch.zeros(_nc, device=device, dtype=torch.bool)
        self.bin_times_chosen_since_new = torch.zeros(_nc, device=device, dtype=torch.int32)
        self.bin_times_seen = torch.zeros(_nc, device=device, dtype=torch.int64)
        self.bin_times_checkpoint_updated = torch.zeros(_nc, device=device, dtype=torch.int32)

        # Bins sampled at the start of the current exploration step.
        self.current_step_sampled_bins = None

        # Per-bin history of archived checkpoints.
        self.bin_checkpoint_idx = torch.ones(_nc, device=device, dtype=torch.int32) * -1
        # (bin, 30 checkpoints, [x, y, z, t, step created, times chosen, mean new bins, variance new bins]).
        self.bin_history = torch.zeros((_nc, 30, 8), device=device)
        
        # Per-slot state: the checkpoint at the end of its trajectory.
        self.slot_checkpoints = torch.zeros((self.POOL_SIZE,
                                            self.checkpoints.shape[-1]),
                                            dtype=self.checkpoints.dtype,
                                            device=device)
        # Bin and position the slot's exploration started from.
        self.slot_starting_bin = torch.ones(self.POOL_SIZE,
                                           dtype=torch.int64,
                                           device=device) * -1
        self.slot_starting_pos = torch.zeros((self.POOL_SIZE, 3),
                                            device=device)

        if self.skip_slot_trajectories:
            self.slot_pos_trajectories = None
            self.slot_bin_trajectories = None
            self.slot_action_trajectories = None
        else:
            self.slot_pos_trajectories = torch.zeros((self.POOL_SIZE,
                                                 self.max_traj_steps, 3),
                                                 device=device)
            # Bin sequence of each trajectory.
            self.slot_bin_trajectories = torch.ones((self.POOL_SIZE,
                                                 self.max_traj_steps, 1),
                                                 dtype=torch.int32,
                                                 device=device) * -1
            self.slot_action_trajectories = torch.zeros((self.POOL_SIZE,
                                                        self.max_traj_steps, 4),
                                                        dtype=torch.int32,
                                                        device=device)
        self.slot_trajectories_step = torch.zeros(self.POOL_SIZE,
                                                 dtype=torch.int32,
                                                 device=device)
        self.slot_rewards = torch.zeros(self.POOL_SIZE, device=device)
        
        self.slot_jumps_remaining = torch.zeros(self.POOL_SIZE,
                                                 dtype=torch.int32,
                                                 device=device)
        # Dynamic-block position at the slot's end state, to re-bin archived states without a restore.
        self.slot_dynamic_block_pos = torch.zeros(
            (self.POOL_SIZE, self.MAX_TRAJ_HISTORY, 3),
            device=device, dtype=self.dynamic_block_pos.dtype)
        #self.slot_steps_remaining = torch.zeros(
        #    (self.POOL_SIZE,) + self.steps_remaining.shape[1:],
        #    device=device, dtype=self.steps_remaining.dtype)
        # Per-step dynamic-block positions along each trajectory, for the
        # design-tool backend's visualizations (off for the paper's runs).
        # block_trajs holds the trajectory of each world, slot_block_trajectories
        # that of each archive slot.
        self.block_trajs = None
        self.slot_block_trajectories = None
        #if track_block_trajectories and not skip_slot_trajectories:
        #    block_shape = tuple(self.dynamic_block_pos.shape[1:])
        #    self.block_trajs = torch.zeros((num_worlds, self.max_traj_steps) + block_shape,
         #                                  device=device, dtype=self.dynamic_block_pos.dtype)
        #    self.slot_block_trajectories = torch.zeros((self.POOL_SIZE, self.max_traj_steps) + block_shape,
         #                                              device=device, dtype=self.dynamic_block_pos.dtype)

        self.curr_rewards = torch.zeros(num_worlds, device=device)
        self.bin_count = torch.zeros(self._num_coarse_bins, device=device).int()

        # Per-world state for action rate limiting.
        self.world_limit_count = torch.zeros(num_worlds,
                                             device=device,
                                             dtype=torch.int32)
        self.world_prev_action = torch.zeros((num_worlds,
                                              len(self.actions_num_buckets)),
                                             device=device,
                                             dtype=torch.int32)
        self.rate_limit = -1

        # Successful trajectories (with store_successful_trajs).
        self.store_successful_trajs = store_successful_trajs
        self.successful_trajs_pos = []
        self.successful_trajs_bin = []
        self.successful_trajs_action = []
        self.successful_trajs_starting_checkpoint = []
        self.successful_trajs_dynamic_block_pos = []  # filled with track_block_trajectories

        self.verbose = verbose
        return

    def generate_new_actions(self, action_policy = None, should_use_policy = None, world_indices = None):
        # Actions applied on the previous step.
        prev_applied_actions = self.actions.clone()

        # Uniform random actions; the policy overrides them below where it is used.
        logits = torch.ones(self.num_worlds,
                            sum(self.actions_num_buckets), device=self.device)
        action_dist = DiscreteActionDistributions(
            self.actions_num_buckets, logits=logits)
        actions, _ = action_dist.sample_out()

        # With probability repeat_p, repeat the previously applied action
        # (worlds that use the policy are excluded).
        mask = (torch.rand(self.num_worlds, device=self.device) < self.repeat_p)

        if action_policy is not None:
            action_dist, _, _ = action_policy([], *self.obs)
            policy_actions, _ = action_dist.sample_out()
            actions[should_use_policy, ...] = policy_actions[should_use_policy]
            mask = torch.logical_and(mask, ~should_use_policy)

        
        actions = actions.int()
        actions[mask] = prev_applied_actions[mask]

        # Write the actions only for world_indices (default: all worlds).
        if world_indices is None:
            world_indices = self.world_indices
        self.actions[world_indices] = actions[world_indices]

    # Unnormalized selection weight of each cell.
    def compute_cell_weights(self, ind):
        # Count score from the Go-Explore Nature paper. bin_times_seen is the
        # number of exploration steps on which the bin was seen.
        count_score = 1 / torch.pow(self.bin_times_seen[ind] + 1, 0.5)

        return count_score

    # Slots per chunk when select_winning_bins scores the archive in chunks
    # (after the single gather ran out of GPU memory).
    SCORE_CHUNK = 8192
    _score_in_chunks = False

    def get_scores(self, rewards, steps, jumps_remaining, trajs):
        """
        Score trajectories for the archive update; a higher score wins.
        Only the step score is used: shorter trajectories win.
        """
        step_score = 1.0 / (1 + steps.float())

        # Other candidate scores; not part of the returned score.
        jump_score = jumps_remaining.float() / self.max_traj_jumps
        idx_expanded = steps.view(steps.size(0), 1, 1).expand(-1, 1, 3)
        idx_expanded = torch.maximum(idx_expanded - 1, torch.zeros_like(idx_expanded))
        height_score = torch.gather(trajs, dim=1, index=idx_expanded.long()).squeeze(1)[..., 2]
        # The reward is 0.05 per unit of progress toward the goal.
        reward_score = rewards * 20
        return step_score


    def _slot_trajs(self, idx):
        """Return (pos, bin, action, steps, starting_pos) for the given slot indices.

        With --skip-slot-trajectories the per-step tensors are zeros. ``steps``
        is always real: it does not depend on max_traj_steps.
        """
        n = idx.shape[0]
        steps = self.slot_trajectories_step[idx]
        if self.slot_pos_trajectories is not None:
            pos = self.slot_pos_trajectories[idx]
            binstr = self.slot_bin_trajectories[idx]
            act = self.slot_action_trajectories[idx]
            starting_pos = self.slot_pos_trajectories[idx, steps - 1, :]
        else:
            pos = torch.zeros(n, self.max_traj_steps, 3, device=self.device)
            binstr = torch.zeros(n, self.max_traj_steps, 1, dtype=torch.int32, device=self.device)
            act = torch.zeros(n, self.max_traj_steps, 4, dtype=torch.int32, device=self.device)
            starting_pos = torch.zeros(n, 3, device=self.device)
        return pos, binstr, act, steps, starting_pos

    # Go-Explore step 1: select a state from the archive.
    def _select_state_density(self, world_indices_to_select):
        """Density-based state selection over the slot pool itself.

        ``self.slot_latents`` holds one latent per occupied slot (written by
        update_archive). The k-NN density is estimated in latent space, the
        candidates are sorted by ascending density (most novel first), and a
        geometric rank sample picks the slots. Returns the same tuple as the
        archive path of select_state.
        """
        n_select = world_indices_to_select.shape[0]

        # Occupied slots.
        valid_slots = torch.where(self._slot_to_coarse != -1)[0]
        if valid_slots.shape[0] == 0 or self.slot_latents is None:
            # No slot is filled yet: restart from slot 0 (the initial checkpoint).
            sampled_slots = torch.zeros(n_select, dtype=torch.long, device=self.device)
            sampled_coarse_bins = self._slot_to_coarse[sampled_slots].clamp_min(0)
            self.curr_rewards[world_indices_to_select] = self.slot_rewards[sampled_slots]
            pos, binstr, act, steps, starting_pos = self._slot_trajs(sampled_slots)
            return (
                self.slot_checkpoints[sampled_slots],
                pos, binstr, steps, act,
                self.slot_jumps_remaining[sampled_slots],
                sampled_coarse_bins, starting_pos,
                self.slot_dynamic_block_pos[sampled_slots],
            )

        # Subsample the candidates to at most n_cand_max slots.
        n_xy = self.binning_function.num_bins() if hasattr(self.binning_function, "num_bins") else self._num_coarse_bins
        if self.knn_candidates_frac is not None:
            n_cand_max = max(1, int(n_xy * self.knn_candidates_frac))
        elif self.knn_candidates == -1:
            n_cand_max = n_xy
        else:
            n_cand_max = self.knn_candidates
        if valid_slots.shape[0] > n_cand_max:
            perm = torch.randperm(valid_slots.shape[0], device=self.device)[:n_cand_max]
            cand_slot_indices = valid_slots[perm]
        else:
            cand_slot_indices = valid_slots

        candidate_latents = self.slot_latents[cand_slot_indices]
        max_ref = KNN_MAX_REFERENCE_LATENTS

        # Reference set. 'stratified' takes the same number of slots from each
        # k-means cluster, so sparse regions of latent space are not
        # under-represented. Without initialized clusters it is uniform.
        use_stratified = (
            self.knn_ref_mode == 'stratified'
            and self.cell_free
            and self.cf.initialized
            and self.cf.slot_cluster_id is not None
        )
        if use_stratified and valid_slots.shape[0] > max_ref:
            cluster_ids = self.cf.slot_cluster_id[valid_slots]
            assigned = cluster_ids >= 0
            cl = cluster_ids[assigned]
            slots_assigned = valid_slots[assigned]
            if cl.numel() == 0:
                ref_latents = self.slot_latents[valid_slots]
            else:
                # Remap the cluster ids to 0..k_active-1.
                unique_clusters, inverse = torch.unique(cl, return_inverse=True)
                k_active = int(unique_clusters.numel())
                per_cluster = max(1, max_ref // k_active)

                # Sort by a random key, then stably by cluster, so the slots of
                # each cluster are in random order.
                r = torch.rand(cl.shape[0], device=self.device)
                order_r = torch.argsort(r)
                inv_after_r = inverse[order_r]
                slots_after_r = slots_assigned[order_r]
                order_c = torch.argsort(inv_after_r, stable=True)
                final_inv = inv_after_r[order_c]
                final_slots = slots_after_r[order_c]

                # Position of each slot within its cluster.
                counts = torch.bincount(inverse, minlength=k_active)
                cluster_starts = torch.cat([
                    torch.zeros(1, dtype=torch.long, device=self.device),
                    counts.cumsum(0)[:-1],
                ])
                positions = (
                    torch.arange(final_inv.shape[0], device=self.device)
                    - cluster_starts[final_inv]
                )
                keep = positions < per_cluster
                strat_slots = final_slots[keep]
                ref_latents = self.slot_latents[strat_slots]
                # Use the whole stratified set: estimate_density must not
                # subsample it again.
                max_ref = ref_latents.shape[0]
        else:
            ref_latents = self.slot_latents[valid_slots]

        density = estimate_density(candidate_latents, ref_latents, k=self.knn_k,
                                   max_samples=max_ref)
        sorted_slot_indices = cand_slot_indices[torch.argsort(density, descending=False)]
        geo_idx = torch.from_numpy(
            sample_geometric_with_max(
                p=self.geo_p, max_value=sorted_slot_indices.shape[0] - 1, size=n_select)
        ).long().to(self.device)
        sampled_slots = sorted_slot_indices[geo_idx]
        sampled_coarse_bins = self._slot_to_coarse[sampled_slots].clamp_min(0)

        # Update the per-bin statistics, as the archive path does.
        unique_coarse_bins, bin_counts = torch.unique(sampled_coarse_bins, return_counts=True)
        self.new_sampled_bins[world_indices_to_select] = self.bin_times_chosen_since_new[sampled_coarse_bins] == 0
        self.bin_times_chosen[unique_coarse_bins] += bin_counts
        self.bin_times_chosen_since_new[unique_coarse_bins] += bin_counts
        self.bin_times_seen[unique_coarse_bins] += bin_counts
        self.bin_history[unique_coarse_bins, self.bin_checkpoint_idx[unique_coarse_bins], 5] += bin_counts

        self.curr_rewards[world_indices_to_select] = self.slot_rewards[sampled_slots]
        pos, binstr, act, steps, starting_pos = self._slot_trajs(sampled_slots)
        return (
            self.slot_checkpoints[sampled_slots],
            pos, binstr, steps, act,
            self.slot_jumps_remaining[sampled_slots],
            sampled_coarse_bins, starting_pos,
            self.slot_dynamic_block_pos[sampled_slots],
        )

    def select_state(self, world_indices_to_select):
        # Density mode samples by latent novelty instead of cell weights.
        if self.selection_mode == 'density':
            return self._select_state_density(world_indices_to_select)

        # Weight the visited bins with the Go-Explore count score.
        valid_coarse_bins = torch.nonzero(self.bin_visited).flatten()
        bin_weights = self.compute_cell_weights(valid_coarse_bins)
        bin_weights = bin_weights / torch.sum(bin_weights)

        all_bin_weights = torch.zeros_like(self.bin_visited).to(torch.float32)
        all_bin_weights[valid_coarse_bins] = bin_weights

        # Occupied slots.
        valid_slots = torch.where(self._slot_to_coarse != -1)[0]
        #print("Valid slots", valid_slots)
        valid_slot_bins = self._slot_to_coarse[valid_slots]

        # Split each bin's weight over its slots.
        slot_bins, slots_per_bin = torch.unique(valid_slot_bins, return_counts=True)
        all_bin_weights[slot_bins] /= slots_per_bin

        slot_weights = all_bin_weights[valid_slot_bins]

        assert len(slot_weights) > 0
        # One sample per world: indices into valid_slots.
        sampled_slot_indices = torch.multinomial(slot_weights,
            num_samples=world_indices_to_select.shape[0],
            replacement=True).type(torch.int)
        
        sampled_slots = valid_slots[sampled_slot_indices]
        
        sampled_coarse_bins = self._slot_to_coarse[valid_slots[sampled_slot_indices]]
        
        unique_coarse_bins, bin_counts = torch.unique(sampled_coarse_bins, return_counts=True)

        # Update the per-bin statistics.
        self.new_sampled_bins[world_indices_to_select] = self.bin_times_chosen_since_new[sampled_coarse_bins] == 0
        self.bin_times_chosen[unique_coarse_bins] += bin_counts
        self.bin_times_chosen_since_new[unique_coarse_bins] += bin_counts
        # With 'length', selecting a bin also counts as seeing it. With
        # 'reward_then_length' (paper Alg. 1), only explore visits count.
        if self.archive_update == 'length':
            self.bin_times_seen[unique_coarse_bins] += bin_counts
        self.bin_history[unique_coarse_bins, self.bin_checkpoint_idx[unique_coarse_bins], 5] += bin_counts

        # Read the sampled slots.
        self.curr_rewards[world_indices_to_select] = self.slot_rewards[sampled_slots]
        selected_ckpts = self.slot_checkpoints[sampled_slots]
        selected_pos_trajs, selected_bin_trajs, selected_action_trajs, selected_steps, selected_starting_pos = \
            self._slot_trajs(sampled_slots)
        selected_jumps_remaining = self.slot_jumps_remaining[sampled_slots]
        # Dynamic-block positions of the sampled slots.
        selected_dynamic_block_trajs = self.slot_dynamic_block_pos[sampled_slots]
        #if self.block_trajs is not None:
        #    self.block_trajs[world_indices_to_select] = self.slot_block_trajectories[sampled_slots]
        # sampled_coarse_bins become the start bins in update_archive.
        return selected_ckpts, selected_pos_trajs, selected_bin_trajs, selected_steps, selected_action_trajs, selected_jumps_remaining, sampled_coarse_bins, selected_starting_pos, selected_dynamic_block_trajs

    # Go-Explore step 2: restore the selected states.
    def go_to_state(self, states):
        self.checkpoints[:] = states
        self.checkpoint_reset[:, 0] = 1
        self.worlds.step()

        if self.verbose:
            print("Avg y position of resampled states: ", torch.mean(self.agent_pos[..., 1]))

        return
    
    def compute_world_latents(self):
        """Compute latents for all current worlds via the configured encoder.

        Returns a (num_worlds, latent_dim) tensor. Lazy-allocates
        ``self.slot_latents`` on the first call so update_archive can store
        per-slot latents matching the dim.

        Density mode only. The encoder (EncoderTrainer, DynamicsCkptEncoder or
        BackboneEncoder) comes from ``GoExplore(encoder=...)``.
        """
        assert self.encoder is not None, \
            "selection_mode='density' requires an encoder; the CFLGE wrapper " \
            "must build one (via load_dynamics_ckpt_encoder, build_encoder_trainer, " \
            "or BackboneEncoder) and pass it in as encoder=."
        latents = self.encoder.encode_target(self.obs).detach().float()
        if self.slot_latents is None:
            self.slot_latents = torch.zeros(
                (self.POOL_SIZE, latents.shape[-1]),
                dtype=torch.float32, device=self.device)
        return latents

    # Go-Explore step 3: explore from the restored states (one sim step).
    def explore_from_state(self, trajs, trajs_bin, steps, actions, jumps_remaining, starting_bin, starting_pos, dynamic_block_trajs):

        with profile('pre_step'):
            # Worlds with no jumps left cannot jump.
            # TODO: the policy does not observe jumps_remaining.
            self.actions[jumps_remaining <= 0, 3] = 0

            ckpt_reset_this_step = torch.where(self.checkpoint_reset > 0)[0]
            reset_this_step = torch.where(self.resets > 0)[0]

            # Copy the pre-step observations and actions for on_step (CFLGE
            # encoder training). obs can contain None (the bin_coord_obs slot,
            # used only by PBGE).
            _pre_obs = tuple(o.clone() if o is not None else None for o in self.obs) if self.on_step is not None else None
            _pre_actions = self.actions.clone() if self.on_step is not None else None

        with profile('worlds.step'):
            self.worlds.step()

        with profile('reward_bookkeep'):
            record_jump = torch.isin(torch.where(self.checkpoint_reset == 0)[0], torch.where(self.agent_jumped > 0)[0])
            jumps_remaining[record_jump] -= 1

            rewards = self.rewards.squeeze(1)
            dones = self.dones.squeeze(1)

        with profile('map_states_to_bins'):
            curr_bins = self.map_states_to_bins(self.sim_obs, jumps_remaining)

        with profile('latent_extract'):
            self._world_latents = self.compute_world_latents() if self.selection_mode == 'density' else None

        with profile('reward_bookkeep'):
            no_reset_rewards = ~torch.isin(self.world_indices, torch.cat([ckpt_reset_this_step, reset_this_step]))
            if len(no_reset_rewards) > 0:
                self.curr_rewards[no_reset_rewards] += rewards[no_reset_rewards]
            self.max_return = max(self.max_return, torch.max(self.curr_rewards).item())

        with profile('reset_handling'):
            with profile('reset_success_check'):
                success = self.worlds.success_tensor().to_torch()
                should_reset = False
                if torch.any(success):
                    if self.store_successful_trajs:
                        ind_success = torch.where(success == 1)[0]
                        for ind in ind_success:
                            traj = trajs[ind, :steps[ind], :].clone()
                            bin_traj = trajs_bin[ind, :steps[ind], :].clone()
                            action = actions[ind, :steps[ind], :].clone()
                            ckpt = self.starting_checkpoints.clone()
                            # Skip a trajectory that starts where a stored one starts.
                            skip = False
                            for pos_traj in self.successful_trajs_pos:
                                if torch.equal(traj[0], pos_traj[0]): skip = True
                            if skip: continue
                            self.successful_trajs_pos.append(traj)
                            self.successful_trajs_bin.append(bin_traj)
                            self.successful_trajs_action.append(action)
                            self.successful_trajs_starting_checkpoint.append(ckpt)
                            #if self.block_trajs is not None:
                            self.successful_trajs_dynamic_block_pos.append(dynamic_block_trajs)
                            should_reset = True
                    if not self.finished:
                        self.finished = True
                        self.solve_step = self.exploration_step
                        print("Found a solution!")
                    if self.verbose:
                        print("Total bins: ", torch.sum(self.bin_visited))
                        print("Total trajectory points: ", torch.sum(self.slot_trajectories_step))
                    if self.break_when_solved:
                        return

            with profile('reset_ind_compute'):
                ind_reset_parts = [
                    # Episode ended.
                    torch.where(dones > 0)[0],
                    # Trajectory cap reached.
                    torch.where(steps >= self.max_traj_steps)[0],
                    # Explore budget of this Go step used up.
                    torch.where(self.sim_steps_this_explore_step >= self.num_exploration_steps)[0],
                ]
                ind_reset = torch.unique(torch.cat(ind_reset_parts))

            # After a stored success, restart every world (only when the search
            # continues to collect more solutions).
            if should_reset: ind_reset = self.world_indices

            self.worlds_reset_this_step[:] = False
            self.worlds_reset_this_step[ind_reset] = True

            if len(ind_reset) > 0:
                with profile('reset_select_state'):
                    # Select new states for the reset worlds; they skip the archive update below.
                    _ckpts, _trajs, _bin_trajs, _steps, _actions, _jumps_remaining, _sampled_bins, _sampled_pos, _dynamic_block_trajs = self.select_state(ind_reset)
                with profile('reset_state_restore'):
                    # Load the selected trajectories; the restore happens on the next sim step.
                    trajs[ind_reset, :, :] = _trajs.clone()
                    trajs_bin[ind_reset, :, :] = _bin_trajs.clone()
                    actions[ind_reset, :, :] = _actions.clone()
                    dynamic_block_trajs[ind_reset, :, :] = _dynamic_block_trajs.clone()
                    steps[ind_reset] = _steps.clone()
                    jumps_remaining[ind_reset] = _jumps_remaining.clone()
                    starting_bin[ind_reset] = _sampled_bins
                    starting_pos[ind_reset] = _sampled_pos


                    self.checkpoints[ind_reset] = _ckpts
                    self.checkpoint_reset[ind_reset, 0] = 1

                    # The restore takes one sim step, which does not count toward the budget.
                    self.sim_steps_this_explore_step[ind_reset] = -1

        with profile('traj_record'):
            # Append this step to each trajectory. Worlds that were reset or
            # restored this step are skipped.

            worlds_to_skip = torch.zeros_like(self.world_indices, dtype=torch.bool)
            worlds_to_skip[torch.unique(torch.cat((ind_reset, ckpt_reset_this_step, reset_this_step)))] = True
            no_reset = ~worlds_to_skip


            pos = self.agent_pos.clone()
            trajs[self.world_indices[no_reset], steps[no_reset], :] = pos[no_reset]

            action = self.actions.clone()
            actions[self.world_indices[no_reset], steps[no_reset], :] = action[no_reset]

            trajs_bin[self.world_indices[no_reset], steps[no_reset], :] = curr_bins[no_reset].unsqueeze(1).int()

            if self.block_trajs is not None:
                self.block_trajs[self.world_indices[no_reset], steps[no_reset]] = self.dynamic_block_pos[no_reset]

            steps[no_reset] += 1
            self.sim_steps_this_explore_step[no_reset] += 1

            # Worlds inside the grid that were not reset.
            alive = no_reset & (curr_bins >= 0)

            # Count each visit in bin_times_seen, before the archive update deduplicates.
            unique_world_coarse, _, world_coarse_counts = torch.unique(
                curr_bins[alive], return_inverse=True, return_counts=True)

            self.bin_times_seen[unique_world_coarse] += world_coarse_counts

        with profile('update_archive'):
            self.update_archive(
                coarse_bins=curr_bins[alive],
                start_bins=starting_bin[alive],
                rewards=self.curr_rewards[alive],
                trajs=trajs[alive],
                trajs_bin=trajs_bin[alive],
                steps=steps[alive],
                actions=actions[alive],
                jumps_remaining=jumps_remaining[alive],
                checkpoints=self.checkpoints[alive],
                starting_pos=starting_pos[alive],
                dynamic_block_trajs=dynamic_block_trajs[alive],
                world_latents=(self._world_latents[alive] if self._world_latents is not None else None))

        # Pass the transition to the caller (CFLGE trains its encoder here).
        if self.on_step is not None:
            self.on_step(_pre_obs, _pre_actions, self.obs)





    def map_states_to_bins(self, states, jumps_remaining) -> torch.Tensor:
        """Return coarse bin indices for the current simulator state.

        Returns -1 for worlds whose agent position falls outside the binning grid.
        """
        obs, agent_pos, steps_remaining, dynamic_block_pos = states
        #print("Agent positions", agent_pos)
        #print("steps_remaining", steps_remaining)
        #print("jumps_remaining", jumps_remaining)
        #print("dynamic_bloc_pos", dynamic_block_pos)

        # TODO: restore
        return self.binning_function.apply(
            agent_pos, (5000 - steps_remaining), jumps_remaining.unsqueeze(1), dynamic_block_pos)
    
    def select_winning_bins(self, world_coarse_bins, world_rewards, world_steps, world_jumps_remaining, world_trajs, archive_coarse_bins, score_fn):

        n_world = world_coarse_bins.shape[0]
        n_archive = archive_coarse_bins.shape[0]
    
        world_scores = score_fn(world_rewards, world_steps, world_jumps_remaining, world_trajs)

        active_slots = self._slot_to_coarse > -1
        active_steps = self.slot_trajectories_step[active_slots]
        # Score the archive. get_scores uses only the steps, but the position
        # trajectories are still gathered: that cost grows with the archive and
        # is part of the measured wall-clock time. At about 42k cells one gather
        # is about 1.5 GB and can run out of GPU memory; after that the run
        # gathers in chunks, which changes its timing.
        archive_scores = None
        if not self._score_in_chunks:
            try:
                if self.slot_pos_trajectories is not None:
                    active_pos = self.slot_pos_trajectories[active_slots]
                else:
                    active_pos = torch.zeros(int(active_slots.sum().item()),
                                             self.max_traj_steps, 3, device=self.device)
                archive_scores = score_fn(
                    self.slot_rewards[active_slots],
                    active_steps,
                    self.slot_jumps_remaining[active_slots],
                    active_pos,
                )
            except torch.OutOfMemoryError:
                active_pos = None
                torch.cuda.empty_cache()
                self._score_in_chunks = True
                print(f"select_winning_bins: archive gather ran out of GPU memory at "
                      f"{int(active_slots.sum().item())} cells; scoring in chunks from now on")
        if archive_scores is None:
            active_idx = torch.nonzero(active_slots).squeeze(1)
            scores = []
            for idx in torch.split(active_idx, self.SCORE_CHUNK):
                if self.slot_pos_trajectories is not None:
                    pos = self.slot_pos_trajectories[idx]
                else:
                    pos = torch.zeros(idx.shape[0], self.max_traj_steps, 3, device=self.device)
                scores.append(score_fn(self.slot_rewards[idx], self.slot_trajectories_step[idx],
                                       self.slot_jumps_remaining[idx], pos))
            archive_scores = (torch.cat(scores) if scores
                              else torch.zeros(0, device=self.device))
        assert len(archive_coarse_bins) == len(archive_scores)

        all_coarse = torch.cat([world_coarse_bins, archive_coarse_bins])
        all_scores = torch.cat([world_scores, archive_scores])
        n_total = n_world + n_archive

        # Sort best first, so the first entry of each bin is its winner.
        if self.archive_update == 'reward_then_length':
            # Paper Alg. 1: the higher cumulative reward wins; at equal reward a
            # strictly shorter trajectory wins; on a full tie the archived state
            # stays. Stable sorts, least significant key first.
            all_rewards = torch.cat([world_rewards, self.slot_rewards[active_slots]]).float()
            is_archive = torch.arange(n_total, device=self.device) >= n_world
            ind_sort = torch.argsort(is_archive.to(torch.int8), descending=True, stable=True)
            ind_sort = ind_sort[torch.argsort(all_scores[ind_sort], descending=True, stable=True)]
            ind_sort = ind_sort[torch.argsort(all_rewards[ind_sort], descending=True, stable=True)]
        else:
            ind_sort = torch.argsort(all_scores, descending=True)
        all_coarse_sorted = all_coarse[ind_sort]

        # Sorted position of each bin's first (best) entry.
        unique_coarse, inv = torch.unique(all_coarse_sorted, return_inverse=True)
        first_sorted_pos = torch.zeros(unique_coarse.shape[0], device=self.device, dtype=torch.int64)
        first_sorted_pos.scatter_reduce_(0, inv, torch.arange(n_total, device=self.device), reduce='amin', include_self=False)

        winning_orig_idx = ind_sort[first_sorted_pos]
        return winning_orig_idx

    def select_states_to_store(self, world_coarse_bins, world_rewards, world_trajs,
                               world_steps, world_jumps_remaining,
                               archive_coarse_bins, world_latents=None):
        """Decide which world states to write and which archive entries to retain.

        Two paths:
          * cell-free density mode (self.cell_free with world_latents): keep
            every alive world, and when the pool would overflow, evict slots by
            the online k-means + age rule.
          * otherwise: in each coarse bin, the best state (world or archive)
            wins.

        Returns (world_keep, archive_keep) bool masks.
        """
        n_world = world_coarse_bins.shape[0]
        n_archive = archive_coarse_bins.shape[0]

        # Cell-free buffer: no per-bin selection, only eviction.
        if self.cell_free and self.selection_mode == 'density' and world_latents is not None:
            self.cf.lazy_init(world_latents.shape[-1])
            world_keep = torch.ones(n_world, device=self.device, dtype=torch.bool)
            archive_keep = torch.ones(n_archive, device=self.device, dtype=torch.bool)
            budget = self.POOL_SIZE - n_archive
            if n_world > budget:
                n_evict = n_world - budget
                evict_slots = self.cf.pick_evictions(n_evict, self._slot_to_coarse >= 0)
                valid_slots = torch.where(self._slot_to_coarse >= 0)[0]
                slot_to_active_idx = torch.full((self.POOL_SIZE,), -1, dtype=torch.long, device=self.device)
                slot_to_active_idx[valid_slots] = torch.arange(valid_slots.shape[0], device=self.device)
                evict_active = slot_to_active_idx[evict_slots]
                archive_keep[evict_active] = False
                # Before initialization slots have no cluster (id -1), so the
                # running means are updated only after it.
                if self.cf.initialized:
                    self.cf.remove(evict_slots, self.slot_latents)
            return world_keep, archive_keep

        winning_orig_idx = self.select_winning_bins(world_coarse_bins=world_coarse_bins,
                                                    world_rewards=world_rewards,
                                                    world_trajs=world_trajs,
                                                    world_steps=world_steps,
                                                    world_jumps_remaining=world_jumps_remaining,
                                                    archive_coarse_bins=archive_coarse_bins,
                                                    score_fn=self.get_scores)
        winning_idxs_by_score = [winning_orig_idx]

        #print(n_archive)

        world_keep = torch.zeros(n_world, device=self.device, dtype=torch.bool)
        archive_keep = torch.zeros(n_archive, device=self.device, dtype=torch.bool)

        for winning_idx in winning_idxs_by_score:
            is_world_win = winning_idx < n_world
            world_keep[winning_idx[is_world_win]] = True
            archive_keep[winning_idx[~is_world_win] - n_world] = True

        #print("select states world and archive", world_keep, archive_keep)

        # TODO: a full pool stops the run; there is no eviction outside cell-free mode.
        total_saved_states = torch.sum(world_keep) + torch.sum(archive_keep)
        if total_saved_states >= self.POOL_SIZE:
            raise RuntimeError(
                f"GoExplore slot pool exhausted: need {total_saved_states} slots, "
                f"only {self.POOL_SIZE} available. Increase _POOL_SIZE.")


        return world_keep, archive_keep

    def update_archive(self,
                       coarse_bins,
                       start_bins,
                       rewards,
                       trajs,
                       trajs_bin,
                       steps,
                       actions,
                       jumps_remaining,
                       checkpoints,
                       starting_pos,
                       dynamic_block_trajs,
                       world_latents=None):

        # Work on a copy, so that a concurrent reader (the design-tool
        # backend) does not see a half-updated map.
        slot_to_coarse_copy = self._slot_to_coarse.clone()

        # Occupied archive slots.
        valid_slots = torch.where(slot_to_coarse_copy >= 0)[0]    # (n_occupied,)
        archive_coarse = slot_to_coarse_copy[valid_slots]         # (n_occupied,)

        world_keep, archive_keep = self.select_states_to_store(
            coarse_bins, rewards, trajs, steps, jumps_remaining,
            archive_coarse, world_latents=world_latents)

        # TODO: restore, debugging
        #print("world keep", world_keep, "archive keep", archive_keep)
        
        # Free the archive slots that lost.
        slot_to_coarse_copy[valid_slots[~archive_keep]] = -1

        # Worlds outside the grid (bin -1) are not stored.
        world_keep[coarse_bins == -1] = False

        # A bin that lost its slot and gained none is no longer visited.
        # TODO: this assumes one slot per bin (cell-free mode recomputes it below).
        world_winner_coarse = coarse_bins[world_keep]
        bin_lost = ~torch.isin(archive_coarse[~archive_keep], world_winner_coarse)
        self.bin_visited[archive_coarse[~archive_keep][bin_lost]] = False

        # Free slots, including the ones freed above.
        free_slots = torch.where(slot_to_coarse_copy < 0)[0]

        n_winners = world_winner_coarse.shape[0]
        winner_slots = free_slots[:n_winners]

        slot_to_coarse_copy[winner_slots] = world_winner_coarse

        # Write the winning world states into their slots.
        self.slot_rewards[winner_slots] = rewards[world_keep]
        self.slot_checkpoints[winner_slots] = checkpoints[world_keep]
        if self.slot_pos_trajectories is not None:
            self.slot_pos_trajectories[winner_slots] = trajs[world_keep]
            self.slot_bin_trajectories[winner_slots] = trajs_bin[world_keep]
            self.slot_action_trajectories[winner_slots] = actions[world_keep]
        self.slot_jumps_remaining[winner_slots] = jumps_remaining[world_keep]
        self.slot_trajectories_step[winner_slots] = steps[world_keep]
        self.slot_dynamic_block_pos[winner_slots] = dynamic_block_trajs[world_keep]

#        curr_dynamic_block_pos[world_keep]
        #self.slot_steps_remaining[winner_slots] = steps[world_keep]
        self.slot_starting_bin[winner_slots] = start_bins[world_keep]
        self.slot_starting_pos[winner_slots] = starting_pos[world_keep]
        if self.slot_latents is not None and world_latents is not None:
            self.slot_latents[winner_slots] = world_latents[world_keep]
        #if self.slot_block_trajectories is not None and block_trajs is not None:
        #    self.slot_block_trajectories[winner_slots] = block_trajs[world_keep]

        # Cell-free bookkeeping: insert step, cluster assignment and running means.
        if self.cell_free and self.selection_mode == 'density' and world_latents is not None and winner_slots.shape[0] > 0:
            n_w = winner_slots.shape[0]
            self.cf.slot_insert_step[winner_slots] = self.cf.global_step
            self.cf.global_step += n_w
            if not self.cf.initialized:
                # Before initialization the slots stay unclustered; the
                # centroids are initialized at the end of this call.
                live_after = int((self._slot_to_coarse >= 0).sum().item()) + n_w
                if live_after >= self.cf.init_samples and self.cf.k > 0:
                    pass
            else:
                ids = self.cf.nearest(world_latents[world_keep])
                self.cf.slot_cluster_id[winner_slots] = ids
                self.cf.insert(world_latents[world_keep].float(), ids)

        # Per-bin statistics for the bins that got a new state.
        self.bin_visited[world_winner_coarse] = True
        self.bin_times_checkpoint_updated[world_winner_coarse] += 1
        self.bin_times_chosen_since_new[world_winner_coarse] = 0

        if n_winners > 0:
            # Append the new checkpoint to each bin's history.
            self.bin_checkpoint_idx[world_winner_coarse] = torch.minimum(
                self.bin_checkpoint_idx[world_winner_coarse] + 1,
                torch.full_like(self.bin_checkpoint_idx[world_winner_coarse],
                                self.bin_history.size(1) - 1))
            history_steps = self.slot_trajectories_step[winner_slots].unsqueeze(1)
            final_pos_index = history_steps.to(torch.int64).view(
                history_steps.shape[0], 1, 1).expand(-1, 1, 3)
            if self.slot_pos_trajectories is not None:
                history_positions = torch.gather(
                    self.slot_pos_trajectories[winner_slots], dim=1,
                    index=torch.maximum(final_pos_index - 1,
                                        torch.zeros_like(final_pos_index))).squeeze()
            else:
                history_positions = torch.zeros(
                    winner_slots.shape[0], 3, device=self.device)
            if len(history_positions.shape) == 1:
                history_positions = history_positions.unsqueeze(0)
            history_points = torch.cat((history_positions, history_steps), dim=1)
            ckpt_entry = torch.zeros((n_winners, 4), device=self.device)
            ckpt_entry[..., 0] = self.exploration_step
            new_ckpt_entries = torch.cat((history_points, ckpt_entry), dim=1)
            self.bin_history[world_winner_coarse,
                             self.bin_checkpoint_idx[world_winner_coarse], ...] = new_ckpt_entries
            
        self._slot_to_coarse = slot_to_coarse_copy

        # Cell-free: several slots can share a bin, so recompute bin_visited
        # from the final slot occupancy.
        if self.cell_free and self.selection_mode == 'density':
            self.bin_visited.zero_()
            occupied_bins = self._slot_to_coarse[self._slot_to_coarse >= 0]
            if occupied_bins.numel() > 0:
                self.bin_visited[occupied_bins] = True

        # Initialize the centroids once enough slots are live.
        if (self.cell_free and self.selection_mode == 'density'
                and self.cf.centroids is not None and not self.cf.initialized):
            n_live = int((self._slot_to_coarse >= 0).sum().item())
            if n_live >= self.cf.init_samples and self.cf.k > 0:
                self.cf.initialize_centroids(self._slot_to_coarse >= 0, self.slot_latents)

        return None

def draw_interactive(go_explore):
    def redraw(level, x_grid, y_grid, binning_function, ix, iy):
        plt.title(f"GoExplore trajectories, "
                  f"repeat p = {go_explore.repeat_p}\n"
                  f"max return = {round(go_explore.max_return, 2)}")
        plt.clf()
        draw_level(level)
        highlighted_bin = binning_function.apply(torch.tensor([ix, iy], device=go_explore.device))
        highlighted_bin = highlighted_bin.item()

        # Draw the grid and mark the selected bin.
        draw_grid(x_grid.cpu().numpy(), y_grid.cpu().numpy())
        plt.scatter(ix, iy, marker='x', color='red', s=10)
        plt.text(ix, iy, str(highlighted_bin), fontsize=8,
                 verticalalignment='bottom', horizontalalignment='right')

        traj = go_explore.slot_pos_trajectories[highlighted_bin]
        step = go_explore.slot_trajectories_step[highlighted_bin]

        x, y = traj[:step, 0].cpu(), traj[:step, 1].cpu()
        plt.plot(x, y, marker='^', markersize=2, color="blue")
        return

    x_grid_bin, y_grid_bin = go_explore.binning_function.get_grid()
    setup_interactive_plot(go_explore.json_levels_cpu,
                           x_grid_bin, y_grid_bin,
                           go_explore.binning_function,
                           draw_fn=redraw)
    plt.show()


def save_successful_trajs(go_explore, save_path):
    if not go_explore.store_successful_trajs:
        print("Not storing successful trajectories.")
        return
    
    if len(go_explore.successful_trajs_pos) == 0:
        print("No success found!")
        return
    ind = torch.all(torch.eq(go_explore.starting_positions, go_explore.successful_trajs_pos[0][0]), dim=1)
    if go_explore.verbose:
        print(go_explore.starting_checkpoints)
    
    # json_levels_cpu is set only when the level came from a file.
    json_levels_cpu = getattr(go_explore, 'json_levels_cpu', None)
    
    torch.save({
        "trajectories": go_explore.successful_trajs_pos,
        "actions": go_explore.successful_trajs_action,
        "json_levels_cpu": json_levels_cpu,
        "starting_checkpoints": go_explore.starting_checkpoints
    }, save_path)
    print(f"Saved to {save_path}")
    return

def _run_explore_loop(
    go_explore,
    num_steps,
    exploration_steps,
    num_worlds,
    max_traj_steps,
    max_traj_jumps,
    action_policy,
    use_policy_selection,
    use_action_percent,
    solution_streaming_info,
    dev,
    resume_slot=None,
    explore=True,
):
    """Seed the archive, then run the Go steps (outer loop) and sim steps (inner loop).

    resume_slot (design tool only): a dict of one archived state of an earlier
    search (see _copy_slot); the archive is seeded with it instead of the
    starting state of every world. explore=False only seeds the archive.

    The caller builds go_explore (and its encoder / on_step, if any) and runs
    _finalize_run afterwards. Returns the search time in seconds, or None when
    the search was cancelled.
    """
    go_explore.USE_ACTION_PERCENT = use_action_percent
    solution_streaming_info["current_go_explore"] = go_explore

    if resume_slot is not None:
        r = resume_slot
        go_explore.update_archive(
            r["bin"], r["starting_bin"], r["reward"], r["traj"], r["bin_traj"], r["steps"],
            r["actions"], r["jumps_remaining"], r["checkpoint"], r["starting_pos"],
            dynamic_block_trajs=r["dynamic_block_pos"],
        )
        if not explore:
            return 0.0
        return _explore_from_archive(go_explore, num_steps, exploration_steps, num_worlds,
                                     action_policy, use_policy_selection, use_action_percent,
                                     solution_streaming_info)

    # Seed the archive with the starting state of every world.
    start_bin = go_explore.map_states_to_bins(
        go_explore.sim_obs,
        # starting jumps remaining
        torch.ones(num_worlds, dtype=torch.int32, device=dev) * max_traj_jumps
    )

    # TODO: restore
    print("start bin", start_bin)

    go_explore.update_archive(
        # current bins
        start_bin,
        # starting bins for the rollout
        start_bin,
        # rewards
        torch.zeros(num_worlds, device=dev),
        # trajs
        torch.zeros(num_worlds, max_traj_steps, 3, device=dev),
        # traj_bins
        torch.ones(num_worlds, max_traj_steps, 1, dtype=torch.int32, device=dev) * -1,
        # steps
        torch.zeros(num_worlds, device=dev, dtype=torch.int),
        # actions
        torch.zeros(num_worlds, max_traj_steps, 4, dtype=torch.int32, device=dev),
        # jumps_remaining
        torch.ones(num_worlds, dtype=torch.int32, device=dev) * max_traj_jumps,
        # checkpoints
        go_explore.checkpoints,
        # Starting positions
        torch.zeros(num_worlds, 3, device=dev),
        dynamic_block_trajs=torch.zeros((num_worlds, go_explore.MAX_TRAJ_HISTORY) + go_explore.dynamic_block_pos.shape[1:], device=dev, dtype=go_explore.dynamic_block_pos.dtype),
    )
    if not explore:
        return 0.0
    return _explore_from_archive(go_explore, num_steps, exploration_steps, num_worlds,
                                 action_policy, use_policy_selection, use_action_percent,
                                 solution_streaming_info)


def _explore_from_archive(go_explore, num_steps, exploration_steps, num_worlds,
                          action_policy, use_policy_selection, use_action_percent,
                          solution_streaming_info):
    """The Go steps (outer loop) and sim steps (inner loop), from a seeded archive."""
    # Select and restore the first states.
    ckpts, trajs, trajs_bin, steps, actions, jumps_remaining, sampled_bins, sampled_pos, dynamic_block_trajs = go_explore.select_state(go_explore.world_indices)

    go_explore.go_to_state(ckpts)

    # Per-world trajectories being explored; update_archive stores them when they win a bin.
    explore_trajs = trajs.clone()
    explore_bin_trajs = trajs_bin.clone()
    explore_steps = steps.clone()
    explore_actions = actions.clone()
    explore_jumps_remaining = jumps_remaining.clone()
    explore_starting_bin = sampled_bins.clone()
    explore_starting_pos = sampled_pos.clone()
    explore_dynamic_block_trajs = dynamic_block_trajs.clone()


    # Per world: does the policy choose the actions of this explore episode?
    should_use_policy = torch.zeros(num_worlds, device=go_explore.device, dtype=torch.bool)

    start_time = time.time()
    _prev_bins_visited = 0
    _prev_step_end = start_time
    pbar = tqdm(range(num_steps), desc="GoExplore: ", unit="step")
    for i in pbar:
        if solution_streaming_info["should_stop"]:
            solution_streaming_info["should_stop"] = False
            return None
        if go_explore.finished and go_explore.break_when_solved:
            break

        go_explore.exploration_step = i

        for j in range(exploration_steps):
            # The design tool can cancel between sim steps.
            if solution_streaming_info["should_stop"]:
                solution_streaming_info["should_stop"] = False
                return None
            if go_explore.finished and go_explore.break_when_solved:
                break
            # On restore, decide whether the policy drives the new episode.
            if torch.any(torch.nonzero(go_explore.worlds_reset_this_step)):
                if action_policy is not None:
                    if use_policy_selection:
                        # GE-PS: use the policy when the restored bin is new.
                        should_use_policy[go_explore.worlds_reset_this_step] = go_explore.new_sampled_bins[go_explore.worlds_reset_this_step]
                    else:
                        # Use policy with probability use_action_percent (GE-RS 0.5, GE-RL 1.0).
                        should_use_policy[go_explore.worlds_reset_this_step] = torch.rand(should_use_policy[go_explore.worlds_reset_this_step].shape, device=go_explore.device) < use_action_percent


            with profile('generate_new_actions'):
                go_explore.generate_new_actions(action_policy, should_use_policy)
            go_explore.explore_from_state(
                explore_trajs,
                explore_bin_trajs,
                explore_steps,
                explore_actions,
                explore_jumps_remaining,
                explore_starting_bin,
                explore_starting_pos,
                explore_dynamic_block_trajs
            )

        # Print the profile of this Go step (GE_PROFILE=1).
        if os.environ.get('GE_PROFILE') == '1':
            print(f"\n[PROFILE outer step {i}]")
            profile.commit()
            profile.report()
            profile.reset()

        # Bins found in this Go step.
        n_bins_visited = int(go_explore.bin_visited.sum().item())
        n_new_bins = max(0, n_bins_visited - _prev_bins_visited)
        _prev_bins_visited = n_bins_visited
        pbar.set_postfix(bins=n_bins_visited, new=n_new_bins, refresh=False)
        if go_explore.verbose:
            print(f"  step {i}: bins_visited={n_bins_visited}  new_this_step={n_new_bins}")
        if go_explore.wandb is not None:
            now = time.time()
            occupied = (go_explore._slot_to_coarse >= 0)
            archive_size = int(occupied.sum().item())
            n_seen = int((go_explore.bin_times_seen > 0).sum().item())
            n_chosen = int((go_explore.bin_times_chosen > 0).sum().item())
            metrics = {
                "goexplore/exploration_step": i,
                "goexplore/n_bins_visited": n_bins_visited,
                "goexplore/n_new_bins_this_step": n_new_bins,
                "goexplore/n_bins_seen": n_seen,
                "goexplore/n_bins_chosen": n_chosen,
                "goexplore/archive_size": archive_size,
                "goexplore/pool_occupancy_frac": archive_size / max(1, go_explore.POOL_SIZE),
                "goexplore/finished": int(go_explore.finished),
                "goexplore/elapsed_sec": now - start_time,
                "goexplore/step_time_sec": now - _prev_step_end,
            }
            if go_explore.cell_free and go_explore.cf.initialized:
                cf_counts = go_explore.cf.counts.float()
                active = cf_counts > 0
                metrics["goexplore/cf_n_clusters_active"] = int(active.sum().item())
                metrics["goexplore/cf_mean_cluster_size"] = float(cf_counts[active].mean().item()) if active.any() else 0.0
                metrics["goexplore/cf_max_cluster_size"] = int(cf_counts.max().item())
            go_explore.wandb.log(metrics)
            _prev_step_end = now

    return time.time() - start_time


def _finalize_run(go_explore, world_simulator, record_trajectories, dev):
    """After the search: optionally write a replay file for the viewer, and
    save the successful trajectories (with store_successful_trajs)."""
    go_explore_traj_file = "GoExploreTrajectories.out"
    go_explore_level_file = "GoExploreLevels.out"
    # Replay the successful trajectories into checkpoint logs for the viewer.
    if record_trajectories:
        with open(go_explore_level_file, "wb") as out_file:
            json_levels_out = world_simulator.json_level_descriptions_tensor().to_torch().to(dev)
            out_file.write(json_levels_out.cpu().numpy().tobytes())
        with open(go_explore_traj_file, "wb") as record_log:
            valid_bins = torch.nonzero(go_explore.bin_visited).flatten()

            num_trajs = len(go_explore.successful_trajs_action)
            replay_starting_checkpoints = go_explore.starting_checkpoints.unsqueeze(0).expand(num_trajs, -1).contiguous()
            replay_actions = torch.from_numpy(np.array([t.cpu() for t in go_explore.successful_trajs_action]))

            if go_explore.verbose:
                print(replay_starting_checkpoints.shape)
                print(replay_actions.shape)

                print(f"Total bins: {replay_starting_checkpoints.size(0)}")

            sim_checkpoints = world_simulator.checkpoint_tensor().to_torch()
            sim_checkpoint_reset = world_simulator.checkpoint_reset_tensor().to_torch()
            sim_actions = world_simulator.action_tensor().to_torch()

            # Restore the starting checkpoint in every world.
            sim_checkpoints[:replay_starting_checkpoints.size(0), ...] = replay_starting_checkpoints
            sim_checkpoints[replay_starting_checkpoints.size(0):, ...] = replay_starting_checkpoints[0, ...]
            sim_checkpoint_reset[:, 0] = 1
            world_simulator.step()

            if go_explore.verbose:
                print(replay_starting_checkpoints.size(0))

            # Replay the actions and write one checkpoint per step.
            for step_idx in range(replay_actions.shape[1]):
                sim_checkpoints[:replay_starting_checkpoints.size(0), ...].cpu().numpy().tofile(record_log)
                sim_actions[:replay_actions.size(0), ...] = replay_actions[..., step_idx, :]
                sim_actions[replay_actions.size(0):, ...] = 0
                world_simulator.step()
                step_idx += 1

            print("GoExplore trajectories saved for replay in ", go_explore_traj_file)

    # Save the successful trajectories as a .pt file.
    if go_explore.store_successful_trajs:
        save_string = "successful_trajs.pt"
        save_successful_trajs(go_explore, save_string)
        if go_explore.verbose:
            print(f"Saved {len(go_explore.successful_trajs_pos)} successful trajectory(ies) to {save_string}")


def create_go_explore(
    world_simulator,
    level_json,
    num_worlds,
    num_steps,
    num_x_bins,
    num_y_bins,
    num_z_bins,
    num_t_bins,
    num_j_bins,
    num_dyn_bins,
    exploration_steps,
    max_traj_steps,
    max_traj_jumps,
    bounds,
    repeat_p,
    break_when_solved,
    store_successful_trajs,
    gpu_sim,
    gpu_id,
    seed,
    dev,
    action_policy,
    use_policy_selection,
    sim_flags,
    pool_size=77_000,
    archive_update='length',
    skip_slot_trajectories=False,
    wandb_logger=None,
    track_block_trajectories=False,
    # CFLGE options (encoder_backbone, selection_mode, knn_*, cell_free*,
    # geo_p), passed on to GoExplore.__init__. The caller attaches the encoder
    # and on_step to the returned object.
    **cflge_kwargs,
):
    target_return = 10.0
    reward_mode = getattr(RewardMode, "Dense1")

    # Margins added to the level bounds for the bin grid (jump height, and
    # horizontal reach). The height margin is not needed when the level is
    # already taller than a jump.
    APPROX_JUMP_HEIGHT = 9
    APPROX_HORIZONTAL_MOBILITY = 7

    MAX_EPISODE_LEN = 3000

    binning_function = get_level_binning_function(
        level_json, 
        num_x_bins, 
        num_y_bins, 
        num_z_bins, 
        num_t_bins, 
        MAX_EPISODE_LEN, 
        MAX_EPISODE_LEN - (max_traj_steps + 1), # Decrease min by 1 to include the endpoint.
        num_j_bins,
        max_traj_jumps,
        -1, # Decrease min by 1 to include the endpoint.
        num_dyn_bins,
        APPROX_HORIZONTAL_MOBILITY, 
        APPROX_JUMP_HEIGHT,
        bounds,
        dev)

    return GoExplore(num_worlds, num_steps, exploration_steps,
                           max_traj_steps, max_traj_jumps, sim_flags,
                           reward_mode, None, binning_function, repeat_p,
                           break_when_solved, store_successful_trajs,
                           gpu_sim, gpu_id,
                           target_return,
                           seed,
                           dev,
                           world_simulator=world_simulator,
                           action_policy=action_policy,
                           use_policy_selection=use_policy_selection,
                           pool_size=pool_size,
                           archive_update=archive_update,
                           skip_slot_trajectories=skip_slot_trajectories,
                           wandb_logger=wandb_logger,
                           track_block_trajectories=track_block_trajectories,
                           **cflge_kwargs)

def explore_from_set_sim(world_simulator,
                         level_json,
                         num_worlds,
                         num_steps,
                         num_x_bins,
                         num_y_bins,
                         num_z_bins,
                         num_t_bins,
                         num_j_bins,
                         num_dyn_bins,
                         exploration_steps,
                         max_traj_steps,
                         max_traj_jumps,
                         bounds,
                         repeat_p,
                         break_when_solved,
                         store_successful_trajs,
                         gpu_sim,
                         gpu_id,
                         seed,
                         dev,
                         action_policy,
                         use_policy_selection,
                         solution_streaming_info,
                         record_trajectories,
                         use_action_percent=0.5,
                         pool_size=40_000, # TODO: restore, 77_000
                         archive_update='length',
                         skip_slot_trajectories=False,
                         wandb_logger=None,
                         track_block_trajectories=False,
                         starting_slot_idx=-1,
                         run_in_background=False):
    """GE, GE-RS, GE-PS and GE-RL.

    The design-tool backend (scripts/slider_backend.py) also sets:
      track_block_trajectories  record dynamic-block positions along trajectories;
      starting_slot_idx         resume from this archive slot of the previous search
                                (solution_streaming_info["current_go_explore"]);
      run_in_background         start the search in a thread and return at once.
    """
    resume_slot = None
    prev = solution_streaming_info.get("current_go_explore")
    if starting_slot_idx != -1:
        assert prev is not None, "starting_slot_idx needs a previous search"
        resume_slot = _copy_slot(prev, starting_slot_idx)
        # The resumed trajectory keeps its steps; the new budget comes on top.
        max_traj_steps += int(resume_slot["steps"].max())
        resume_slot = _pad_slot(resume_slot, max_traj_steps)
    if prev is not None and (run_in_background or starting_slot_idx != -1):
        # Design tool: free the previous search before allocating the next.
        solution_streaming_info["current_go_explore"] = None
        del prev
        torch.cuda.empty_cache()

    go_explore = create_go_explore(
        world_simulator,
        level_json,
        num_worlds,
        num_steps,
        num_x_bins,
        num_y_bins,
        num_z_bins,
        num_t_bins,
        num_j_bins,
        num_dyn_bins,
        exploration_steps,
        max_traj_steps,
        max_traj_jumps,
        bounds,
        repeat_p,
        break_when_solved,
        store_successful_trajs,
        gpu_sim,
        gpu_id,
        seed,
        dev,
        action_policy,
        use_policy_selection,
        SimFlags.Default,
        pool_size=pool_size,
        archive_update=archive_update,
        skip_slot_trajectories=skip_slot_trajectories,
        wandb_logger=wandb_logger,
        track_block_trajectories=track_block_trajectories,
    )

    def run():
        # GE, GE-RS, GE-PS, GE-RL: no encoder and no on_step.
        total_time = _run_explore_loop(
            go_explore,
            num_steps=num_steps,
            exploration_steps=exploration_steps,
            num_worlds=num_worlds,
            max_traj_steps=max_traj_steps,
            max_traj_jumps=max_traj_jumps,
            action_policy=action_policy,
            use_policy_selection=use_policy_selection,
            use_action_percent=use_action_percent,
            solution_streaming_info=solution_streaming_info,
            dev=dev,
            resume_slot=resume_slot,
        )
        if total_time is None:
            return None
        _finalize_run(go_explore, world_simulator, record_trajectories, dev)
        return go_explore

    if run_in_background:
        # Seed the archive before returning, so the first status request of
        # the design tool already sees it; the Go steps run in the thread.
        _run_explore_loop(go_explore, num_steps=num_steps, exploration_steps=exploration_steps,
                          num_worlds=num_worlds, max_traj_steps=max_traj_steps,
                          max_traj_jumps=max_traj_jumps, action_policy=action_policy,
                          use_policy_selection=use_policy_selection,
                          use_action_percent=use_action_percent,
                          solution_streaming_info=solution_streaming_info, dev=dev,
                          resume_slot=resume_slot, explore=False)

        def run_seeded():
            total_time = _explore_from_archive(go_explore, num_steps, exploration_steps, num_worlds,
                                               action_policy, use_policy_selection,
                                               use_action_percent, solution_streaming_info)
            if total_time is not None:
                _finalize_run(go_explore, world_simulator, record_trajectories, dev)

        threading.Thread(target=run_seeded, daemon=True).start()
        return go_explore
    return run()


def _copy_slot(go_explore, slot):
    """One archived state of go_explore, as batch-1 tensors for update_archive."""
    g, i = go_explore, slot
    one = lambda t: t[i].clone().unsqueeze(0)
    return {
        "bin": one(g._slot_to_coarse), "starting_bin": one(g.slot_starting_bin),
        "reward": one(g.slot_rewards), "traj": one(g.slot_pos_trajectories),
        "bin_traj": one(g.slot_bin_trajectories), "steps": one(g.slot_trajectories_step),
        "actions": one(g.slot_action_trajectories), "jumps_remaining": one(g.slot_jumps_remaining),
        "checkpoint": one(g.slot_checkpoints), "starting_pos": one(g.slot_starting_pos),
        "dynamic_block_pos": one(g.slot_dynamic_block_pos)
    }


def _pad_slot(slot, max_traj_steps):
    """Pad the per-step arrays of a copied slot to max_traj_steps steps."""
    out = dict(slot)
    for k, fill in (("traj", 0), ("bin_traj", -1), ("actions", 0)):
        t = out[k]
        if t is None or t.shape[1] >= max_traj_steps:
            continue
        pad = torch.full((t.shape[0], max_traj_steps - t.shape[1]) + tuple(t.shape[2:]), fill,
                         dtype=t.dtype, device=t.device)
        out[k] = torch.cat([t, pad], dim=1)
    return out


def cell_free_latent_explore_from_set_sim(
        world_simulator,
        level_json,
        num_worlds,
        num_steps,
        num_x_bins,
        num_y_bins,
        num_z_bins,
        num_t_bins,
        num_j_bins,
        num_dyn_bins,
        exploration_steps,
        max_traj_steps,
        max_traj_jumps,
        bounds,
        repeat_p,
        break_when_solved,
        store_successful_trajs,
        gpu_sim,
        gpu_id,
        seed,
        dev,
        action_policy,
        use_policy_selection,
        solution_streaming_info,
        record_trajectories,
        use_action_percent=0.5,
        # CFLGE options.
        density_latent='backbone',
        dynamics_ckpt_path=None,
        encoder_backbone=None,
        pool_size=120_000,
        skip_slot_trajectories=True,
        cell_free_k=None,
        cell_free_small_quantile=0.2,
        train_encoder=False,
        encoder_lr=1e-4,
        encoder_epochs=2,
        encoder_minibatch=512,
        encoder_train_every=1,
        encoder_pred_hidden=256,
        encoder_init_ckpt=None,
        knn_k=KNN_K_NEIGHBORS,
        knn_candidates=-1,
        knn_candidates_frac=None,
        knn_ref_mode='uniform',
        geo_p=GEO_SAMPLING_P,
        wandb_logger=None,
):
    """CFLGE driver: cell-free latent go-explore.

    Like explore_from_set_sim, but with selection_mode='density' and
    cell_free=True: states are selected by k-NN density in an encoder's
    latent space, and the slot pool is a cell-free buffer. It also builds the
    encoder and the on_step callback that trains it.
    """
    go_explore = create_go_explore(
        world_simulator,
        level_json,
        num_worlds,
        num_steps,
        num_x_bins,
        num_y_bins,
        num_z_bins,
        num_t_bins,
        num_j_bins,
        num_dyn_bins,
        exploration_steps,
        max_traj_steps,
        max_traj_jumps,
        bounds,
        repeat_p,
        break_when_solved,
        store_successful_trajs,
        gpu_sim,
        gpu_id,
        seed,
        dev,
        action_policy,
        use_policy_selection,
        SimFlags.Default,
        encoder_backbone=encoder_backbone,
        selection_mode='density',
        knn_k=knn_k,
        knn_candidates=knn_candidates,
        knn_candidates_frac=knn_candidates_frac,
        knn_ref_mode=knn_ref_mode,
        geo_p=geo_p,
        pool_size=pool_size,
        skip_slot_trajectories=skip_slot_trajectories,
        cell_free=True,
        cell_free_k=cell_free_k,
        cell_free_small_quantile=cell_free_small_quantile,
        wandb_logger=wandb_logger,
    )

    # The encoder, by (density_latent, train_encoder):
    #   'dynamics_ckpt'             -> frozen checkpoint encoder (CFLGE-Frozen)
    #   'backbone' + train_encoder  -> forward-dynamics trainer (CFLGE, CFLGE-FT)
    #   'backbone' + encoder_backbone, no training -> the policy backbone
    encoder = None
    if density_latent == 'dynamics_ckpt':
        if dynamics_ckpt_path is None:
            raise ValueError("--density-latent dynamics_ckpt requires --dynamics-ckpt-path.")
        encoder = load_dynamics_ckpt_encoder(
            dynamics_ckpt_path, dev, verbose=go_explore.verbose)
    elif train_encoder:
        encoder = build_encoder_trainer(
            go_explore,
            encoder_init_ckpt=encoder_init_ckpt,
            encoder_pred_hidden=encoder_pred_hidden,
            encoder_lr=encoder_lr,
            encoder_epochs=encoder_epochs,
            encoder_minibatch=encoder_minibatch,
            verbose=go_explore.verbose,
            wandb_logger=wandb_logger,
        )
    elif encoder_backbone is not None:
        encoder = BackboneEncoder(encoder_backbone)
    # Otherwise the encoder is None, and compute_world_latents raises.

    go_explore.encoder = encoder

    # on_step runs once per sim step with (pre_obs, pre_actions, post_obs). With
    # train_encoder it stores the transition and trains the encoder every
    # encoder_train_every steps.
    on_step = None
    if train_encoder and encoder is not None:
        _train_every = max(1, encoder_train_every)
        _step = [0]
        def on_step(pre_obs, pre_actions, post_obs):
            encoder.store_obs_t(pre_obs)
            encoder.store_step(pre_actions, post_obs)
            _step[0] += 1
            if (_step[0] % _train_every) == 0:
                encoder.train_minibatches()

    go_explore.on_step = on_step

    total_time = _run_explore_loop(
        go_explore,
        num_steps=num_steps,
        exploration_steps=exploration_steps,
        num_worlds=num_worlds,
        max_traj_steps=max_traj_steps,
        max_traj_jumps=max_traj_jumps,
        action_policy=action_policy,
        use_policy_selection=use_policy_selection,
        use_action_percent=use_action_percent,
        solution_streaming_info=solution_streaming_info,
        dev=dev,
    )
    if total_time is None:
        return None

    _finalize_run(go_explore, world_simulator, record_trajectories, dev)
    return go_explore


world_simulator = None

if __name__ == "__main__":
    import csv
    from pathlib import Path

    # Profiling is on only with GE_PROFILE=1; otherwise profile(...) blocks do nothing.
    if os.environ.get('GE_PROFILE') != '1':
        profile.disable()

    arg_parser.add_argument('--query-paths', type=str, nargs='+',
                            help='One or more query JSON files to run. Overrides --query-path.')
    arg_parser.add_argument('--seeds', type=int, nargs='+',
                            help='List of seeds to run. When provided, overrides --seed for iteration.')
    arg_parser.add_argument('--log-path', type=Path, default=None,
                            help='CSV file to append results to.')
    # Print the parsed args as JSON and exit before any simulator setup (to
    # check what a run script passes).
    arg_parser.add_argument('--print-args-and-exit', action='store_true', help=argparse.SUPPRESS)

    args = arg_parser.parse_args()
    if args.print_args_and_exit:
        print("ARGS_JSON " + json.dumps({k: (str(v) if isinstance(v, Path) else v)
                                         for k, v in vars(args).items() if k != "print_args_and_exit"},
                                        sort_keys=True))
        raise SystemExit(0)

    if args.policy_based:
        # PBGE's search and trainer live in scripts/pbge.py. PBGE runs with the
        # profiler on (as in the paper's runs), so undo the GE_PROFILE gate above.
        profile.enable()
        from pbge import make_trainer, new_query_policy, policy_based_explore_from_set_sim

    # --query-paths, else --query-path.
    if args.query_paths:
        query_paths = args.query_paths
    elif args.query_path:
        query_paths = [args.query_path]
    else:
        raise SystemExit("Must provide --query-path or --query-paths.")

    queries = []
    for qp in query_paths:
        with open(qp, "r") as f:
            data = json.loads(f.read())
        queries.append((qp, data))

    # --seeds, else --seed.
    seeds = args.seeds if args.seeds else [args.seed]

    dev = torch.device(f"cuda:{args.gpu_id}") if args.gpu_sim else torch.device("cpu")

    from madrona_scripts.infer import setup_sim
    world_simulator = setup_sim(
        num_worlds=args.num_worlds,
        episode_len=3000,
        gpu_sim=args.gpu_sim,
        gpu_id=args.gpu_id,
        rand_seed=args.seed,
        use_complex_level=False,
        use_normaabb_obs=False,
        disable_phys_obs=args.disable_phys_obs,
        disable_step_obs=args.disable_step_obs,
        disable_lidar=args.disable_lidar,
        unlock_agent_orientation=False,
        reward_mode=args.reward_mode,
    )
    world_simulator.init()

    ckpt_policy = None
    policy_path = args.policy or args.ckpt_path
    if policy_path:
        _, num_obs_features = setup_obs(world_simulator, world_simulator.enum_counts_tensor().to_torch().to(dev), False)
        ACTION_DIMS = [2, 8, 3, 2]
        ckpt_policy = make_policy(num_obs_features, None, 256, False, ACTION_DIMS)
        weights = LearningState.load_policy_weights(policy_path)
        key_load = ckpt_policy.load_state_dict(weights, strict=False)
        if key_load.missing_keys or key_load.unexpected_keys:
            raise RuntimeError(f"policy checkpoint {policy_path} does not match the model: "
                               f"missing={key_load.missing_keys} unexpected={key_load.unexpected_keys}")
        ckpt_policy = ckpt_policy.to(dev)
    train_wrapper = make_trainer(args, world_simulator, dev, ckpt_policy) if args.policy_based else None
    # wandb logger; does nothing without --wandb.
    from wandb_logger import WandbLogger
    wandb_run = WandbLogger(
        enabled=args.wandb,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        group=args.wandb_group,
        config=vars(args),
    )

    LOG_HEADER = ("query", "seed", "time_sec", "solution_found", "bins_discovered", "solve_step", "num_x_bins", "num_y_bins", "num_z_bins", "num_t_bins", "num_j_bins", "num_dyn_bins")

    for qp, data in queries:
        level_descriptions = [json.dumps(data["level_json"])]
        max_steps = data.get("max_traj_steps", args.max_traj_steps)
        max_jumps = data.get("max_traj_jumps", args.max_traj_jumps)
        bounds = None
        if "boundsPosition" in data:
            bounds = {
                "position": data["boundsPosition"],
                "size": data["boundsExtents"],
            }

        # Per-query bin axes from the level content: animated objects
        # need t bins, Unanchored (pushable) objects need dyn bins, and a
        # jump limit needs j bins. When several axes are on, others get
        # fewer bins, so the total bin count stays bounded.
        has_animated_object = False
        has_dynamic_object = False
        has_limited_jumps = 0 < max_jumps < LIMITED_JUMPS_MAX
        for _, desc in data["level_json"].items():
            for b in desc["behaviors"]:
                if b['name'] == "Unanchored":
                    has_dynamic_object = True
                if "Translate" in b['name'] or "Rotate" in b['name']:
                    has_animated_object = True

        num_x_bins = args.num_x_bins
        num_y_bins = args.num_y_bins
        num_z_bins = args.num_z_bins
        num_t_bins = args.num_t_bins if has_animated_object else 1
        num_j_bins = 1
        num_dyn_bins = 1
        if has_limited_jumps:
            if has_animated_object:
                num_x_bins //= 2
                num_y_bins //= 2
            num_j_bins = max_jumps
        if has_dynamic_object:
            if has_animated_object:
                num_t_bins //= 2
            if has_limited_jumps:
                num_z_bins //= 2
            num_dyn_bins = DYN_BINS_WITH_UNANCHORED
        print(f"Bin counts: x={num_x_bins} y={num_y_bins} z={num_z_bins} "
              f"t={num_t_bins} j={num_j_bins} dyn={num_dyn_bins}  "
              f"(animated={has_animated_object}, dynamic={has_dynamic_object}, "
              f"limited_jumps={has_limited_jumps})")

        # Load the level into the simulator tensor.
        json_levels_tensor = world_simulator.json_level_descriptions_tensor().to_torch()
        json_indices = world_simulator.json_index_tensor().to_torch().to(dev)
        sample_level_jsons = list(level_descriptions)
        sample_level_jsons.extend([sample_level_jsons[0]] * (json_levels_tensor.shape[0] - len(sample_level_jsons)))
        json_indices[..., 0] = 0
        json_levels_cpu_tensor = jsonDescriptionsToTensor(json_levels_tensor.shape, sample_level_jsons, json_levels_physics=None)
        json_levels_tensor[:] = json_levels_cpu_tensor.to(dev)

        for seed in seeds:
            start_time = time.time()
            # PBGE runs policy_based_explore_from_set_sim, CFLGE
            # cell_free_latent_explore_from_set_sim, and the other methods
            # explore_from_set_sim.
            solution_streaming_info = {"should_stop": False, "current_go_explore": None}
            _is_cflge = args.cell_free or args.selection_mode == 'density'
            _search_kwargs = dict(
                world_simulator=world_simulator,
                level_json=json.loads(sample_level_jsons[0]),
                num_worlds=args.num_worlds,
                num_steps=args.num_steps,
                num_x_bins=num_x_bins,
                num_y_bins=num_y_bins,
                num_z_bins=num_z_bins,
                num_t_bins=num_t_bins,
                num_j_bins=num_j_bins,
                num_dyn_bins=num_dyn_bins,
                exploration_steps=args.exploration_steps,
                max_traj_steps=max_steps,
                max_traj_jumps=max_jumps,
                bounds=bounds,
                repeat_p=args.repeat_p,
                break_when_solved=args.break_when_solved,
                store_successful_trajs=args.store_successful_trajs,
                gpu_sim=args.gpu_sim,
                gpu_id=args.gpu_id,
                seed=seed,
                dev=dev,
                solution_streaming_info=solution_streaming_info,
            )
            _common_kwargs = dict(
                **_search_kwargs,
                action_policy=ckpt_policy,
                use_policy_selection=args.use_policy_selection,
                record_trajectories=False,
                use_action_percent=args.use_action_percent,
                pool_size=args.pool_size,
                skip_slot_trajectories=args.skip_slot_trajectories,
                wandb_logger=wandb_run,
            )
            if args.policy_based:
                # A new, untrained policy for each query.
                new_query_policy(train_wrapper, Path(args.ckpt_dir) / Path(qp).stem)
                go_explore = policy_based_explore_from_set_sim(
                    **_search_kwargs,
                    action_policy=train_wrapper.policy,
                    use_policy_selection=False,
                    train_wrapper=train_wrapper,
                )
            elif _is_cflge:
                go_explore = cell_free_latent_explore_from_set_sim(
                    **_common_kwargs,
                    density_latent=args.density_latent,
                    dynamics_ckpt_path=args.dynamics_ckpt_path,
                    encoder_backbone=None,
                    cell_free_k=args.cell_free_k,
                    cell_free_small_quantile=args.cell_free_small_quantile,
                    train_encoder=args.train_encoder,
                    encoder_lr=args.encoder_lr,
                    encoder_epochs=args.encoder_epochs,
                    encoder_minibatch=args.encoder_minibatch,
                    encoder_train_every=args.encoder_train_every,
                    encoder_pred_hidden=args.encoder_pred_hidden,
                    encoder_init_ckpt=args.encoder_init_ckpt,
                    knn_k=args.knn_k,
                    knn_candidates=args.knn_candidates,
                    knn_candidates_frac=args.knn_candidates_frac,
                    knn_ref_mode=args.knn_ref_mode,
                    geo_p=args.geo_p,
                )
            else:
                go_explore = explore_from_set_sim(**_common_kwargs,
                                                  archive_update=args.archive_update)
            elapsed = time.time() - start_time
            solution_found = go_explore.finished if go_explore is not None else False
            bins_discovered = int(go_explore.bin_visited.sum().item()) if go_explore is not None else 0
            solve_step = go_explore.solve_step if go_explore is not None else None
            del go_explore
            import gc; gc.collect()
            torch.cuda.empty_cache()
            print(f"GoExplore time: {elapsed:.4f}s")
            print(f"Query={Path(qp).name} Seed={seed} time={elapsed:.2f}s solution_found={solution_found} bins_discovered={bins_discovered} solve_step={solve_step}")

            if args.log_path is not None:
                write_header = not args.log_path.exists()
                args.log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(args.log_path, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    if write_header:
                        writer.writerow(LOG_HEADER)
                    writer.writerow((qp, str(seed), f"{elapsed:.2f}", str(solution_found), bins_discovered, solve_step, num_x_bins, num_y_bins, num_z_bins, num_t_bins, num_j_bins, num_dyn_bins))

    wandb_run.finish()
