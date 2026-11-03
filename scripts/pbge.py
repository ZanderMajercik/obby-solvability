"""Policy-based Go-Explore (PBGE).

The main of go_explore_search_solvable.py runs every method; with
--policy-based it builds the trainer (make_trainer), gives it a new policy per
query (new_query_policy) and runs policy_based_explore_from_set_sim.

PBGEGoExplore is a subclass of the GoExplore class in
go_explore_search_solvable.py, which every other method (and the design tool's
default search) uses. It takes construction and go_to_state from it and
overrides the methods where PBGE's search differs.
"""

import matplotlib.pyplot as plt

import madrona_puzzle_bench
from madrona_puzzle_bench import SimFlags, RewardMode
from madrona_puzzle_bench_learn.action import DiscreteActionDistributions

from madrona_puzzle_bench_learn import ActionTransform, profile, AMPState
from madrona_puzzle_bench_learn import (
    train, profile, TrainConfig, PPOConfig, SimInterface, EMANormalizer
)
from madrona_scripts.policy import make_policy

from tqdm import tqdm

from madrona_scripts.parameters.params_train import TrainParameters1
from madrona_scripts.parameters.params_transforms import TransformParameters
from generateLevels import loadJsonDescriptionsFromDisk, generateJsonLevels

import numpy as np
import torch
import warnings

import itertools
import random
import threading

from madrona_scripts.go_explore_search_solvable import GoExplore
from go_binning import get_level_binning_function, BinningFunction, \
    create_binning_function, draw_level, draw_grid, setup_interactive_plot

torch.manual_seed(0)
warnings.filterwarnings("error")

class PBGEGoExplore(GoExplore):
    """GoExplore with PBGE's search: in Return mode a goal-conditioned policy
    returns to archived cells, and some worlds replay archived actions for the
    self-imitation (SIL) loss.

    Construction (the simulator, observations, bin statistics and slot state)
    and go_to_state come from the GoExplore class in
    go_explore_search_solvable.py. The methods below are PBGE's own; they keep
    PBGE's archive rules (a 40k pool that drops entries when full, per-step
    trajectories of 3000 steps including dynamic-block positions, its cell
    weights and time-bin axis).
    """

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
                 store_reward_checkpoint: bool,
                 gpu_sim: bool,
                 gpu_id: int,
                 target_return: float,
                 seed: int,
                 device: torch.device,
                 world_simulator = None,
                 action_policy = None,
                 use_policy_selection = False,
                 verbose = False,
                 allow_trajectory_recording = False,
                 use_sil_rollouts = False,
                 bin_coord_obs = None):
        # The shared part. PBGE allocates its own trajectory buffers below
        # (3000 steps, whatever max_traj_steps is), so the base class skips its.
        super().__init__(num_worlds, exploration_phases, exploration_steps,
                         max_traj_steps, max_traj_jumps, sim_flags, reward_mode,
                         json_levels_cpu, binning_fun, repeat_p, break_when_solved,
                         store_successful_trajs, gpu_sim, gpu_id, target_return,
                         seed, device,
                         world_simulator=world_simulator,
                         action_policy=action_policy,
                         use_policy_selection=use_policy_selection,
                         verbose=verbose,
                         allow_trajectory_recording=allow_trajectory_recording,
                         pool_size=40_000,
                         skip_slot_trajectories=True,
                         bin_coord_obs=bin_coord_obs)

        self.action_latch = self.actions.clone()
        self.store_reward_checkpoint = store_reward_checkpoint

        # PBGE: 'Restore' restores archived states; 'Return' makes the policy return to goal bins.
        self.MODE = "Restore"
        self.NUM_GOAL_BINS = 10

        self.MAX_TRAJ_HISTORY = 3000

        self.slot_pos_trajectories = torch.zeros((self.POOL_SIZE,
                                             self.MAX_TRAJ_HISTORY, 3),
                                             device=device)
        # Bin sequence of each trajectory.
        self.slot_bin_trajectories = torch.ones((self.POOL_SIZE,
                                             self.MAX_TRAJ_HISTORY, 1),
                                             dtype=torch.int32,
                                             device=device) * -1
        self.slot_action_trajectories = torch.zeros((self.POOL_SIZE,
                                                    self.MAX_TRAJ_HISTORY, 4),
                                                    dtype=torch.int32,
                                                    device=device)
        # Dynamic-block position history of each slot (same length as the position trajectory).
        self.slot_dynamic_block_pos = torch.zeros(
            (self.POOL_SIZE, self.MAX_TRAJ_HISTORY, 3),
            device=device)

        # Per-world state of the worlds that return to a goal bin (Return mode).
        self.returning_world_trajs_bins = torch.ones(
            (self.num_worlds, self.MAX_TRAJ_HISTORY, 1), dtype=torch.int32, device=device) * -1

        self.returning_world_trajs_final_step_idx = torch.ones(self.num_worlds,
                                                 dtype=torch.int32,
                                                 device=device) * -1
        self.returning_world_steps = torch.ones(self.num_worlds,
                                                 dtype=torch.int32,
                                                 device=device) * -1
        self.exploring_world_goal_cells = torch.ones(self.num_worlds,
                                                 dtype=torch.int32,
                                                 device=device) * -1
        # e_t in the Go-Explore Nature paper (First Return, Then Explore), p. 11, eq. 8.
        self.entropy_threshold = torch.ones(num_worlds, device=device, dtype=torch.int32) * self.max_traj_steps

        self.num_sil_worlds = 0
        self.sil_start_idx = num_worlds
        # Worlds that replay archived action trajectories for the self-imitation (SIL) loss.
        if use_sil_rollouts:
            # 1/16 of the worlds.
            self.num_sil_worlds = (num_worlds // 16)
            self.sil_start_idx = num_worlds - self.num_sil_worlds
            self.sil_actions = torch.zeros((self.num_sil_worlds,
                                            self.MAX_TRAJ_HISTORY, 4),
                                            dtype=torch.int32,
                                            device=device)
            self.sil_steps = torch.zeros(self.num_sil_worlds,
                                        dtype=torch.int32,
                                        device=device)
            self.sil_num_actions = torch.zeros(self.num_sil_worlds,
                                        dtype=torch.int32,
                                        device=device)

    def generate_new_actions(self, action_policy = None, should_use_policy = None, world_indices = None):

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
        actions[mask] = self.action_latch[mask]

        # Worlds that are not returning keep their random actions.
        random_actions_latch = actions[self.returning_world_trajs_final_step_idx == -1]

        # SIL worlds replay their stored actions.
        if self.num_sil_worlds > 0:
            step_idx = self.sil_steps.view(self.sil_actions.shape[0], 1, 1).expand(self.sil_actions.shape[0], 1, self.sil_actions.shape[2])
            actions[self.sil_start_idx:] = torch.gather(self.sil_actions, dim=1, index=step_idx.to(torch.int64)).squeeze(1)

        actions[self.returning_world_trajs_final_step_idx == -1] = random_actions_latch

        # Write the actions only for world_indices (default: all worlds).
        if world_indices is None:
            world_indices = self.world_indices
        self.actions[world_indices] = actions[world_indices]

        # Remember the applied actions for the repeat step.
        self.action_latch = self.actions.clone()



    # Unnormalized selection weight of each cell.
    def compute_bin_weights(self, ind):
        


        if self.MODE == "Restore":
            # Count score from the Go-Explore Nature paper (bin_times_seen = the
            # number of exploration steps on which the bin was seen).
            count_score = 1 / torch.pow(self.bin_times_seen[ind] + 1, 0.5)
        elif self.MODE == "Return":
            # Count score from the policy-based Go-Explore paper.
            count_score = 1 / (0.5 * self.bin_times_seen[ind] + 1)

        score = count_score


        return score

    def get_scores(self, rewards, steps):
        """
        For given trajectories, compute the scores.
        Modify this for custom trajectory preference.
        """
        step_score = 1.0 / (1 + steps.float())

        return step_score

    def get_scores_reward(self, rewards, steps):
        """
        For given trajectories, compute the scores.
        Modify this for custom trajectory preference.
        """
        reward_score = rewards
        return reward_score

    def update_bin_counts(self, bins):
        unique_coarse_bins, bin_counts = torch.unique(bins, return_counts=True)
        self.bin_times_chosen[unique_coarse_bins] += bin_counts
        self.bin_times_chosen_since_new[unique_coarse_bins] += bin_counts
        self.bin_times_seen[unique_coarse_bins] += bin_counts
        self.bin_history[unique_coarse_bins, self.bin_checkpoint_idx[unique_coarse_bins], 5] += bin_counts

    # Go-Explore step 1: select states from the archive.
    def select_state(self, num_bins_to_select):

        # Weight the visited bins with the Go-Explore count score.
        valid_coarse_bins = torch.nonzero(self.bin_visited).flatten()
        bin_weights = self.compute_bin_weights(valid_coarse_bins)

        all_bin_weights = torch.zeros_like(self.bin_visited).to(torch.float32)
        all_bin_weights[valid_coarse_bins] = bin_weights

        # Occupied slots.
        valid_slots = torch.where(self._slot_to_coarse != -1)[0]
        valid_slot_bins = self._slot_to_coarse[valid_slots]

        # Split each bin's weight over its slots.
        slot_bins, slots_per_bin = torch.unique(valid_slot_bins, return_counts=True)
        all_bin_weights[slot_bins] /= slots_per_bin

        slot_weights = all_bin_weights[valid_slot_bins]

        # In Restore mode, each slot's cumulative reward adds to its weight.
        if self.MODE == "Restore":
            slot_weights += self.slot_rewards[valid_slots]

        slot_weights /= torch.sum(slot_weights)


        # One sample per world: indices into valid_slots.
        assert len(slot_weights) > 0
        sampled_slot_indices = torch.multinomial(slot_weights,
            num_samples=num_bins_to_select,
            replacement=True).type(torch.int)
        
        sampled_slots = valid_slots[sampled_slot_indices]
        
        sampled_coarse_bins = self._slot_to_coarse[sampled_slots]
        



        selected_rewards = self.slot_rewards[sampled_slots]
        selected_ckpts = self.slot_checkpoints[sampled_slots]
        selected_pos_trajs = self.slot_pos_trajectories[sampled_slots]
        selected_bin_trajs = self.slot_bin_trajectories[sampled_slots]
        selected_action_trajs = self.slot_action_trajectories[sampled_slots]
        selected_steps = self.slot_trajectories_step[sampled_slots]
        selected_jumps_remaining = self.slot_jumps_remaining[sampled_slots]
        selected_starting_pos = self.slot_pos_trajectories[sampled_slots, self.slot_trajectories_step[sampled_slots] - 1, :]
        selected_dynamic_block_trajs = self.slot_dynamic_block_pos[sampled_slots]
        # sampled_coarse_bins become the start bins in update_archive.
        return  selected_ckpts, \
                selected_pos_trajs, \
                selected_bin_trajs, \
                selected_steps, \
                selected_action_trajs, \
                selected_jumps_remaining, \
                sampled_coarse_bins, \
                selected_starting_pos, \
                selected_dynamic_block_trajs, \
                selected_rewards

    # Go-Explore step 2: restore the selected states.
    def setGoalBins(self, world_indices, bin_indices):
        
        # Every world given here must get a real goal bin.
        # TODO: this fails on some runs (e.g. 3SpinFwd/Query1, SpikyLogRoll/Query1):
        # a returning world reads -1, the fill value of returning_world_trajs_bins.
        # The cause is not known yet.
        assert torch.all(bin_indices != -1)
        
        self.exploring_world_goal_cells[world_indices] = bin_indices.int()

        # The goal-bin observation: the bin's grid coordinates, normalized to [0, 1].
        unnormalized_bin_coord = self.binning_function.index_to_coord(bin_indices)
        num_bins_per_dim = [len(grid) - 1 for grid in self.binning_function.get_grid()]
        normalized_bin_coord = [c / l for c, l in zip(unnormalized_bin_coord, num_bins_per_dim)]
        self.bin_coord_obs[world_indices] = torch.cat([n.unsqueeze(1) for n in normalized_bin_coord], dim=1)

    # Pick new goal bins after the current ones were reached, as in the policy-based Go-Explore paper.
    def sampleNewGoalBins(self, curr_goal_bins):
        # A reached goal bin must be in the archive.
        if torch.any(~torch.isin(curr_goal_bins, torch.where(self.bin_visited)[0])):
            print("A goal was reached but its bin is not in the archive!")
            print("current goal bins:", torch.unique(curr_goal_bins))
            print("Current goal bin points:", self.binning_function.rep_points[torch.unique(curr_goal_bins).int()])
            print("current bins", torch.where(self.bin_visited)[0])
            print("current bins coords", self.binning_function.index_to_coord(torch.where(self.bin_visited)[0]))
            print("Current bin points:", self.binning_function.rep_points[torch.where(self.bin_visited)[0]])
            exit()

        raw_adjacent_bins = self.binning_function.get_adjacent_bins(curr_goal_bins.int())




        adjacent_bins_select = torch.rand_like(raw_adjacent_bins.float())
        adjacent_bins_select[raw_adjacent_bins == curr_goal_bins[:, None]] = -1 # Filter out bins identical to current.
        adjacent_bins = torch.gather(raw_adjacent_bins, 1, adjacent_bins_select.argmax(dim=1, keepdim=True)).flatten()


        valid_bins = torch.nonzero(self.bin_visited).flatten()
        adjacent_bins_select = torch.rand_like(raw_adjacent_bins.float())
        adjacent_bins_select[raw_adjacent_bins == curr_goal_bins[:, None]] = -1 # Filter out bins identical to current.
        adjacent_bins_select[torch.isin(raw_adjacent_bins, valid_bins)] = -1 # Filter out bins that already exist in the archive.
        adjacent_new_bins = torch.gather(raw_adjacent_bins, 1, adjacent_bins_select.argmax(dim=1, keepdim=True)).flatten()

        _, _, _, _, _, _, new_goal_bins, _, _, _ = self.select_state(curr_goal_bins.shape[0])

        ADJACENT_BIN_PROB = 0.225
        ADJACENT_NEW_BIN_PROB = 0.1

        # Per world: a random draw that picks an adjacent new bin, an adjacent bin or an archive bin.
        r = torch.rand_like(new_goal_bins, dtype=torch.float32)

        # When every adjacent bin is already archived, never pick an adjacent new bin.
        r[torch.where(adjacent_new_bins == -1)[0]] *= 1 - ADJACENT_NEW_BIN_PROB 
        r[torch.where(adjacent_new_bins == -1)[0]] += ADJACENT_NEW_BIN_PROB

        should_sample_adjacent_no_archive = torch.where(r < ADJACENT_NEW_BIN_PROB)[0]

        if len(should_sample_adjacent_no_archive) > 0:
            new_goal_bins[should_sample_adjacent_no_archive] = adjacent_new_bins[should_sample_adjacent_no_archive].to(torch.int64)

        should_sample_adjacent = torch.logical_and(
            r >= ADJACENT_NEW_BIN_PROB, 
            r < (ADJACENT_NEW_BIN_PROB + ADJACENT_BIN_PROB)
        )
        if len(should_sample_adjacent) > 0:
            new_goal_bins[should_sample_adjacent] = adjacent_bins[should_sample_adjacent].to(torch.int64)

        return new_goal_bins
    
    def get_current_world_bin_sub_traj(self, world_indices, num_steps_to_gather):
        assert num_steps_to_gather < self.MAX_TRAJ_HISTORY
        bin_traj_ind = torch.zeros((len(world_indices), num_steps_to_gather), dtype=torch.int64, device=self.device)
        bin_traj_ind[...] += self.returning_world_steps[world_indices].unsqueeze(1) 
        bin_traj_ind[...] += torch.arange(num_steps_to_gather, device=self.device).unsqueeze(0)
        ret = torch.gather(self.returning_world_trajs_bins[world_indices].squeeze(2), 1, bin_traj_ind)
        return ret

    # Go-Explore step 3: explore from the restored states (one sim step).
    def explore_from_state(self, 
                           trajs, 
                           trajs_bin, 
                           steps, 
                           actions, 
                           jumps_remaining, 
                           starting_bin, 
                           starting_pos, 
                           dynamic_block_trajs,
                           curr_rewards):

        # Worlds with no jumps left cannot jump.
        # TODO: the policy does not observe jumps_remaining.
        self.actions[jumps_remaining <= 0, 3] = 0

        ckpt_reset_this_step = torch.where(self.checkpoint_reset > 0)[0]
        reset_this_step = torch.where(self.resets > 0)[0]

        self.worlds.step()

        record_jump = torch.isin(torch.where(self.checkpoint_reset == 0)[0], torch.where(self.agent_jumped > 0)[0])
        jumps_remaining[record_jump] -= 1

        rewards = self.rewards.squeeze(1)
        dones = self.dones.squeeze(1)

        curr_bins = self.binning_function.apply(
            self.agent_pos,
            steps.unsqueeze(1),
            jumps_remaining.unsqueeze(1),
            self.dynamic_block_pos
        )

        # Return mode: check whether the returning worlds reached their goal bins.
        if self.MODE == "Return":
            ret_world_indices = torch.where(self.returning_world_trajs_final_step_idx > -1)[0]
            if len(ret_world_indices) > 0:
                curr_world_bin_trajs = self.get_current_world_bin_sub_traj(ret_world_indices, self.NUM_GOAL_BINS)
                # An intermediate goal bin was reached.
                intermediate_mask = torch.logical_and(
                    curr_world_bin_trajs == curr_bins[ret_world_indices, None].int(),
                    curr_bins[ret_world_indices, None] != -1
                ).any(dim=1)
                # The final goal bin was reached.
                final_mask = (self.returning_world_trajs_bins[ret_world_indices, self.returning_world_trajs_final_step_idx[ret_world_indices]].squeeze(1) == curr_bins[ret_world_indices].int())
                assert len(intermediate_mask) == len(final_mask)
                rewards[ret_world_indices[intermediate_mask]] += 0.01
                rewards[ret_world_indices[final_mask]] += 0.02

                # Reaching an intermediate goal restarts the explore budget.
                self.sim_steps_this_explore_step[ret_world_indices[intermediate_mask]] = 0
                self.sim_steps_this_explore_step[ret_world_indices[final_mask]] = 0

            explore_world_indices = torch.where(self.returning_world_trajs_final_step_idx == -1)[0]
            if len(explore_world_indices) > 0:
                hit_explore_goal_mask = torch.logical_and(
                    curr_bins[explore_world_indices] == self.exploring_world_goal_cells[explore_world_indices],
                    curr_bins[explore_world_indices] != -1
                )
                rewards[explore_world_indices[hit_explore_goal_mask]] += 0.03
                self.sim_steps_this_explore_step[explore_world_indices[hit_explore_goal_mask]] = 0


        no_reset_rewards = ~torch.isin(self.world_indices, torch.cat([ckpt_reset_this_step, reset_this_step]))
        if len(no_reset_rewards) > 0:
            curr_rewards[no_reset_rewards] += rewards[no_reset_rewards]

        success = self.worlds.success_tensor().to_torch()
        should_reset = False
        if torch.any(success):
            if self.store_successful_trajs:
                ind_success = torch.where(success == 1)[0]
                for ind in ind_success:
                    traj = trajs[ind, :steps[ind], :].clone()
                    bin_traj = trajs_bin[ind, :steps[ind], :].clone()
                    action = actions[ind, :steps[ind], :].clone()
                    dyn_block_traj = dynamic_block_trajs[ind, :steps[ind], :].clone()
                    ckpt = self.starting_checkpoints.clone()
                    # Skip a trajectory that starts where a stored one starts.
                    skip = False
                    for pos_traj in self.successful_trajs_pos:
                        if torch.equal(traj[0], pos_traj[0]): skip = True
                    if skip: continue
                    self.successful_trajs_pos.append(traj)
                    self.successful_trajs_bin.append(bin_traj)
                    self.successful_trajs_action.append(action)
                    self.successful_trajs_dynamic_block_pos.append(dyn_block_traj)
                    self.successful_trajs_starting_checkpoint.append(ckpt)
                    should_reset = True
            if not self.finished:
                self.finished = self.break_when_solved
                self.solve_step = self.exploration_step
                print("Found a solution!")
            if self.verbose:
                print("Total bins: ", torch.sum(self.bin_visited))
                print("Total trajectory points: ", torch.sum(self.slot_trajectories_step))
            if self.break_when_solved:
                return

        # Worlds to reset: the episode ended, the trajectory cap was reached, or
        # the explore budget is used up.
        # TODO: PBGE does not pick a new goal when a world makes no progress for many steps.

        indices_to_reset = torch.cat(
            (
                torch.where(dones > 0)[0], 
                torch.where(steps >= self.max_traj_steps)[0], 
                # (SIL worlds are excluded from the explore budget)
                torch.where(self.sim_steps_this_explore_step[:self.sil_start_idx] >= self.num_exploration_steps)[0]
            )
        )

        # SIL worlds reset when they run out of actions to replay.
        if self.num_sil_worlds > 0:
            indices_to_reset = torch.cat(
                (
                    indices_to_reset,
                    torch.where(self.sil_steps >= (self.sil_num_actions - 1))[0] + self.sil_start_idx
                )
            )

        ind_reset = torch.unique(
            indices_to_reset
        )

        # After a stored success, restart every world (only when the search
        # continues to collect more solutions).
        if should_reset: ind_reset = self.world_indices

        self.worlds_reset_this_step[:] = False
        self.worlds_reset_this_step[ind_reset] = True

        if len(ind_reset) > 0:
            # Select new states (Restore) or goal trajectories (Return) for the reset worlds.
            _ckpts, _trajs, _bin_trajs, _steps, _actions, _jumps_remaining, _sampled_bins, _sampled_pos, _dynamic_block_trajs, _rewards = self.select_state(ind_reset.shape[0])
            
            # Mark new bins (for GE-PS) before the selection counts change.
            self.new_sampled_bins[ind_reset] = self.bin_times_chosen_since_new[_sampled_bins] == 0
            self.update_bin_counts(_sampled_bins)

            if self.MODE == "Restore":
                # Restore: load the selected trajectories; the restore happens on the next sim step.
                trajs[ind_reset, :, :] = _trajs.clone()
                trajs_bin[ind_reset, :, :] = _bin_trajs.clone()
                actions[ind_reset, :, :] = _actions.clone()
                dynamic_block_trajs[ind_reset, :, :] = _dynamic_block_trajs.clone()
                steps[ind_reset] = _steps.clone()
                jumps_remaining[ind_reset] = _jumps_remaining.clone()
                starting_bin[ind_reset] = _sampled_bins
                starting_pos[ind_reset] = _sampled_pos
                curr_rewards[ind_reset] = _rewards


                self.checkpoints[ind_reset] = _ckpts
                self.checkpoint_reset[ind_reset, 0] = 1
            elif self.MODE == "Return":
                # Return: start from the initial state; the policy returns to the
                # selected bin along its archived bin trajectory.
                trajs[ind_reset, :, :] = torch.zeros_like(trajs[ind_reset, :, :])
                trajs_bin[ind_reset, :, :] = torch.ones_like(trajs_bin[ind_reset, :, :]) * -1
                actions[ind_reset, :, :] = torch.zeros_like(actions[ind_reset, :, :])
                dynamic_block_trajs[ind_reset, :, :] = torch.zeros_like(dynamic_block_trajs[ind_reset, :, :])
                steps[ind_reset] = torch.zeros_like(steps[ind_reset])
                jumps_remaining[ind_reset] = torch.ones_like(jumps_remaining[ind_reset]) * self.max_traj_jumps
                
                # The start state is always the same, so the start bin does not change.
                starting_bin[ind_reset] = starting_bin[ind_reset].clone()

                bt = _bin_trajs.clone()

                # Drop consecutive repeats from each bin trajectory.
                bin_traj_condensed = [torch.unique_consecutive(t) for t in bt.squeeze(2)]
                bin_traj_condensed = [b[:-1] if b[-1] == -1 else b for b in bin_traj_condensed]
                for b in bin_traj_condensed:
                    assert len(b) > 0

                def filterOutOfBoundsBins(x):
                    indices_to_filter = torch.where(x == -2)[0]
                    if len(indices_to_filter) > 0:
                        indices_to_read_from = torch.minimum(indices_to_filter + 1, (len(x) - 1) * torch.ones_like(indices_to_filter))
                        x[indices_to_filter] = x[indices_to_read_from]
                    return x
                bin_traj_condensed = [filterOutOfBoundsBins(t) for t in bin_traj_condensed]
                
                bin_traj_condensed_len = [len(t) for t in bin_traj_condensed]
                
                # Store the condensed bin trajectories and their lengths.
                self.returning_world_trajs_bins[ind_reset] = -1
                rwtb = self.returning_world_trajs_bins[ind_reset]
                rwtfsi = self.returning_world_trajs_final_step_idx[ind_reset]
                for i in range(_bin_trajs.shape[0]):
                    rwtb[i, :len(bin_traj_condensed[i]), :] = bin_traj_condensed[i].unsqueeze(1)
                    rwtfsi[i] = len(bin_traj_condensed[i]) - 1
                self.returning_world_trajs_bins[ind_reset] = rwtb
                self.returning_world_trajs_final_step_idx[ind_reset] = rwtfsi

                
                self.returning_world_steps[ind_reset] = torch.minimum(self.returning_world_trajs_final_step_idx[ind_reset], torch.ones_like(ind_reset)).int()


                # The first goal bin of each new trajectory goes to the simulator.
                bins_to_write = self.returning_world_trajs_bins[ind_reset, self.returning_world_steps[ind_reset]].squeeze(1)

                self.setGoalBins(ind_reset, bins_to_write)

                # Reset the worlds: they may have died or run out of budget.
                self.resets[ind_reset, 0] = 1

                self.entropy_threshold[ind_reset] = _steps.clone()

                if self.num_sil_worlds > 0:
                    # SIL worlds load the actions of their selected trajectory and replay them from step 0.
                    sil_reset = ind_reset[ind_reset >= self.sil_start_idx] 
                    if len(sil_reset) > 0:
                        self.sil_num_actions[sil_reset - self.sil_start_idx] = torch.maximum(_steps[-len(sil_reset):] - 1, torch.zeros_like(_steps[-len(sil_reset):]))
                        self.sil_steps[sil_reset - self.sil_start_idx] = 0
                        self.sil_actions[sil_reset - self.sil_start_idx] = _actions[-len(sil_reset):]

            # The restore takes one sim step, which does not count toward the budget.
            self.sim_steps_this_explore_step[ind_reset] = -1


        # Append this step to each trajectory. Worlds that were reset or
        # restored this step are skipped.

        worlds_to_skip = torch.zeros_like(self.world_indices, dtype=torch.bool)
        worlds_to_skip[torch.unique(torch.cat((ind_reset, ckpt_reset_this_step, reset_this_step)))] = True
        no_reset = ~worlds_to_skip


        pos = self.agent_pos.clone()
        trajs[self.world_indices[no_reset], steps[no_reset], :] = pos[no_reset]

        dyn_pos = self.dynamic_block_pos.clone()
        dynamic_block_trajs[self.world_indices[no_reset], steps[no_reset], :] = dyn_pos[no_reset]

        action = self.actions.clone()
        actions[self.world_indices[no_reset], steps[no_reset], :] = action[no_reset]

        trajs_bin[self.world_indices[no_reset], steps[no_reset], :] = curr_bins[no_reset].unsqueeze(1).int()

        steps[no_reset] += 1
        self.sim_steps_this_explore_step[no_reset] += 1
        if self.num_sil_worlds > 0:
            self.sil_steps[no_reset[self.sil_start_idx:]] += 1

        # Worlds inside the grid that were not reset.
        alive = no_reset & (curr_bins >= 0)

        # Count each visit in bin_times_seen, before the archive update deduplicates.
        unique_world_coarse, _, world_coarse_counts = torch.unique(
            curr_bins[alive], return_inverse=True, return_counts=True)
        
        self.bin_times_seen[unique_world_coarse] += world_coarse_counts

        self.update_archive(
            coarse_bins=curr_bins[alive],
            start_bins=starting_bin[alive],
            rewards=curr_rewards[alive],
            trajs=trajs[alive],
            trajs_bin=trajs_bin[alive],
            steps=steps[alive],
            actions=actions[alive],
            jumps_remaining=jumps_remaining[alive],
            checkpoints=self.checkpoints[alive],
            starting_pos=starting_pos[alive],
            dynamic_block_trajs=dynamic_block_trajs[alive])
    
    def select_winning_bins(self, world_coarse_bins, world_rewards, world_steps, archive_coarse_bins, score_fn):

        n_world = world_coarse_bins.shape[0]
        n_archive = archive_coarse_bins.shape[0]
    
        world_scores = score_fn(world_rewards, world_steps)

        active_slots = self._slot_to_coarse > -1
        archive_scores = score_fn(
            self.slot_rewards[active_slots], 
            self.slot_trajectories_step[active_slots],
        )
        assert len(archive_coarse_bins) == len(archive_scores)

        all_coarse = torch.cat([world_coarse_bins, archive_coarse_bins])
        all_scores = torch.cat([world_scores, archive_scores])
        n_total = n_world + n_archive

        # Sort best first, so the first entry of each bin is its winner.
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
                               archive_coarse_bins):
        """Decide which world states to write and which archive entries to retain.

        For each coarse bin, the highest-scoring state wins (world or archive).
        Returns world_scores (needed by caller), world_keep bool mask, archive_keep bool mask.
        """
        n_world = world_coarse_bins.shape[0]
        n_archive = archive_coarse_bins.shape[0]

        winning_orig_idx = self.select_winning_bins(world_coarse_bins=world_coarse_bins, 
                                                    world_rewards=world_rewards,
                                                    world_steps=world_steps,
                                                    archive_coarse_bins=archive_coarse_bins,
                                                    score_fn=self.get_scores)
        winning_idxs_by_score = [winning_orig_idx]
        if self.store_reward_checkpoint:
            winning_orig_idx_reward = self.select_winning_bins(world_coarse_bins=world_coarse_bins, 
                                                        world_rewards=world_rewards,
                                                        world_steps=world_steps,
                                                        archive_coarse_bins=archive_coarse_bins,
                                                        score_fn=self.get_scores_reward)
            winning_idxs_by_score += [winning_orig_idx_reward]

        world_keep = torch.zeros(n_world, device=self.device, dtype=torch.bool)
        archive_keep = torch.zeros(n_archive, device=self.device, dtype=torch.bool)

        for winning_idx in winning_idxs_by_score:
            is_world_win = winning_idx < n_world
            world_keep[winning_idx[is_world_win]] = True
            archive_keep[winning_idx[~is_world_win] - n_world] = True

        # When the pool is full, drop the last archive entries to make room.
        total_saved_states = torch.sum(world_keep) + torch.sum(archive_keep)
        if total_saved_states >= self.POOL_SIZE:
            archive_keep[self.POOL_SIZE - torch.sum(world_keep) - 1:] = False


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
                       dynamic_block_trajs):

        # Work on a copy, so that a concurrent reader (the design-tool
        # backend) does not see a half-updated map.
        slot_to_coarse_copy = self._slot_to_coarse.clone()

        # Occupied archive slots.
        valid_slots = torch.where(slot_to_coarse_copy >= 0)[0]    # (n_occupied,)
        archive_coarse = slot_to_coarse_copy[valid_slots]         # (n_occupied,)

        # For each coarse bin, select the best state (world or archive).
        world_keep, archive_keep = self.select_states_to_store(
            coarse_bins, rewards, trajs, steps, jumps_remaining,
            archive_coarse)
        
        # Free the archive slots that lost.
        slot_to_coarse_copy[valid_slots[~archive_keep]] = -1

        # Worlds outside the grid (bin -1) are not stored.
        world_keep[coarse_bins == -1] = False

        # A bin that lost its slot and gained none is no longer visited.
        # TODO: this assumes one slot per bin.
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
        self.slot_pos_trajectories[winner_slots] = trajs[world_keep]
        self.slot_bin_trajectories[winner_slots] = trajs_bin[world_keep]
        self.slot_action_trajectories[winner_slots] = actions[world_keep]
        self.slot_jumps_remaining[winner_slots] = jumps_remaining[world_keep]
        self.slot_trajectories_step[winner_slots] = steps[world_keep]
        self.slot_dynamic_block_pos[winner_slots] = dynamic_block_trajs[world_keep]
        self.slot_starting_bin[winner_slots] = start_bins[world_keep]
        self.slot_starting_pos[winner_slots] = starting_pos[world_keep]

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
            history_positions = torch.gather(
                self.slot_pos_trajectories[winner_slots], dim=1,
                index=torch.maximum(final_pos_index - 1,
                                    torch.zeros_like(final_pos_index))).squeeze()
            if len(history_positions.shape) == 1:
                history_positions = history_positions.unsqueeze(0)
            history_points = torch.cat((history_positions, history_steps), dim=1)
            ckpt_entry = torch.zeros((n_winners, 4), device=self.device)
            ckpt_entry[..., 0] = self.exploration_step
            new_ckpt_entries = torch.cat((history_points, ckpt_entry), dim=1)
            self.bin_history[world_winner_coarse,
                             self.bin_checkpoint_idx[world_winner_coarse], ...] = new_ckpt_entries
            
        self._slot_to_coarse = slot_to_coarse_copy

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
    store_reward_checkpoint=False,
    use_sil_rollouts=False,
    bin_coord_obs=None
):
    target_return = 10.0
    reward_mode = getattr(RewardMode, "Dense1")

    # Margins added to the level bounds for the bin grid (jump height, and
    # horizontal reach). The height margin is not needed when the level is
    # already taller than a jump.
    APPROX_JUMP_HEIGHT = 11
    APPROX_HORIZONTAL_MOBILITY = 7

    binning_function = get_level_binning_function(
        level_json, 
        num_x_bins, 
        num_y_bins, 
        num_z_bins, 
        num_t_bins, 
        max_traj_steps, 
        -1, # Decrease min by 1 to include the endpoint.
        num_j_bins,
        max_traj_jumps,
        -1, # Decrease min by 1 to include the endpoint.
        num_dyn_bins,
        APPROX_HORIZONTAL_MOBILITY, 
        APPROX_JUMP_HEIGHT,
        bounds,
        dev)

    return PBGEGoExplore(num_worlds, num_steps, exploration_steps,
                           max_traj_steps, max_traj_jumps, sim_flags,
                           reward_mode, None, binning_function, repeat_p,
                           break_when_solved, store_successful_trajs,
                           store_reward_checkpoint,
                           gpu_sim, gpu_id,
                           target_return,
                           seed,
                           dev,
                           world_simulator=world_simulator,
                           action_policy=action_policy,
                           use_policy_selection=use_policy_selection,
                           use_sil_rollouts=use_sil_rollouts,
                           bin_coord_obs=bin_coord_obs)

def policy_based_explore_from_set_sim(
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
        use_policy_selection,  # False for PBGE
        solution_streaming_info,
        train_wrapper,
        run_in_background=False):
    
    prev_go_explore = solution_streaming_info["current_go_explore"]
    solution_streaming_info["current_go_explore"] = None

    if prev_go_explore is not None:
        # Free the previous search.
        del prev_go_explore
        import gc; gc.collect()
        torch.cuda.empty_cache()

    # The training wrapper must expose the 12 observation tensors that GoExplore uses.
    assert len(train_wrapper.obs) == 12

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
        SimFlags.Default,  # ignored when the simulator already exists
        use_sil_rollouts=True,
        bin_coord_obs=train_wrapper.obs[-1]
    )

    solution_streaming_info["current_go_explore"] = go_explore

    # Seed the archive with the starting state of every world.
    start_bin = go_explore.binning_function.apply(
        # position
        go_explore.agent_pos, 
        # steps
        torch.zeros(num_worlds, dtype=torch.int32, device=dev).unsqueeze(1),
        torch.ones(num_worlds, dtype=torch.int32, device=dev).unsqueeze(1) * max_traj_jumps,
        go_explore.dynamic_block_pos
    )

    start_trajs = torch.zeros(num_worlds, go_explore.MAX_TRAJ_HISTORY, 3, device=dev)
    start_trajs[:, 0, :] = go_explore.agent_pos

    start_bin_trajs = torch.ones(num_worlds, go_explore.MAX_TRAJ_HISTORY, 1, dtype=torch.int32, device=dev) * -1
    start_bin_trajs[:, 0, :] = start_bin.unsqueeze(1)

    go_explore.update_archive(
        # current bins
        coarse_bins=start_bin,
        # starting bins for the rollout
        start_bins=start_bin,
        # rewards
        rewards=torch.zeros(num_worlds, device=dev),
        # trajs
        trajs=start_trajs,
        # traj_bins
        trajs_bin=start_bin_trajs,
        # steps
        steps=torch.zeros(num_worlds, device=dev, dtype=torch.int),
        # actions
        actions=torch.zeros(num_worlds, go_explore.MAX_TRAJ_HISTORY, 4, dtype=torch.int32, device=dev),
        # jumps_remaining
        jumps_remaining=torch.ones(num_worlds, dtype=torch.int32, device=dev) * max_traj_jumps,
        # checkpoints
        checkpoints=go_explore.checkpoints,
        # Starting positions
        starting_pos=go_explore.agent_pos,
        dynamic_block_trajs=torch.zeros((num_worlds, go_explore.MAX_TRAJ_HISTORY) + go_explore.dynamic_block_pos.shape[1:], device=dev, dtype=go_explore.dynamic_block_pos.dtype),
    )

    # PBGE returns to states with the policy instead of restoring them.
    go_explore.MODE = "Return"

    # Every spawn position is outside the bin grid: the archive is empty, so skip the query.
    if torch.all(go_explore._slot_to_coarse == -1):
        print("Warning: archive empty after init (all spawn positions outside binning grid). Skipping query.")
        return go_explore

    # Select and restore the first states.
    ckpts, trajs, trajs_bin, steps, actions, jumps_remaining, sampled_bins, sampled_pos, dynamic_block_trajs, rewards = go_explore.select_state(go_explore.world_indices.shape[0])
    
    # Mark new bins before the selection counts change.
    go_explore.new_sampled_bins[go_explore.world_indices] = go_explore.bin_times_chosen_since_new[sampled_bins] == 0
    go_explore.update_bin_counts(sampled_bins)


    # Optimistically record trajectories to commit whenever a checkpoint is replaced in a bin.
    explore_trajs = trajs.clone()
    explore_bin_trajs = trajs_bin.clone()
    explore_steps = steps.clone()
    explore_actions = actions.clone()
    explore_jumps_remaining = jumps_remaining.clone()
    explore_starting_bin = sampled_bins.clone()
    explore_starting_pos = sampled_pos.clone()
    explore_dynamic_block_trajs = dynamic_block_trajs.clone()
    explore_rewards = rewards.clone()

    total_steps = num_steps * exploration_steps

    # First goal bins.
    new_goal_bins = go_explore.sampleNewGoalBins(sampled_bins)
    go_explore.setGoalBins(go_explore.world_indices, new_goal_bins)

    policy_exploration_worlds = torch.ones(num_worlds, device=go_explore.device, dtype=torch.bool)
    policy_entropy_scales_worlds = torch.ones(num_worlds, device=go_explore.device, dtype=torch.float32)
    sil_worlds = torch.zeros(num_worlds, device=go_explore.device, dtype=torch.bool)
    if go_explore.num_sil_worlds > 0:
        sil_worlds[go_explore.sil_start_idx:] = True

    def computeEntropyScale(t, e_t):
        # e_f, e_p in the Go-Explore Nature paper (First Return, Then Explore), p. 11, eq. 8.
        e_f = 0.01
        e_p = 2
        return 1 + torch.pow(torch.maximum(torch.zeros_like(t), t - e_t) * e_f, e_p)
        

    def explore_callback():
            if torch.any(go_explore.worlds_reset_this_step):
                # A reset world returns with the policy (no random exploration).
                policy_exploration_worlds[go_explore.worlds_reset_this_step] = True

            # Current bin of every world.
            curr_bins = go_explore.binning_function.apply(
                go_explore.agent_pos,
                explore_steps.unsqueeze(1),
                explore_jumps_remaining.unsqueeze(1),
                go_explore.dynamic_block_pos
            )

            ret_world_indices = torch.where(go_explore.returning_world_trajs_final_step_idx > -1)[0]

            # Aliases (no copies) of the returning-world tensors.
            rw_tb = go_explore.returning_world_trajs_bins
            rw_ts = go_explore.returning_world_trajs_final_step_idx
            rw_s = go_explore.returning_world_steps

            # Advance the returning worlds that reached one of their next goal bins.
            if len(ret_world_indices) > 0:
                sub_traj_len = max(go_explore.NUM_GOAL_BINS, int(torch.max(rw_ts[ret_world_indices] - rw_s[ret_world_indices])))
                bin_trajs_to_check = go_explore.get_current_world_bin_sub_traj(ret_world_indices, sub_traj_len)

                # Per world: which of the next bins of its return trajectory it is in.
                mask = torch.logical_and(
                    bin_trajs_to_check == curr_bins[ret_world_indices, None],
                    curr_bins[ret_world_indices, None] != -1
                )
                assert mask.shape[0] == len(ret_world_indices)
                assert mask.shape[1] == sub_traj_len

                # Last matching index of each world.
                mask_indices = torch.zeros_like(mask, dtype=torch.int32)
                mask_indices[...] = torch.arange(sub_traj_len)
                mask_indices[~mask] = -1

                new_index_increment = torch.argmax(mask_indices.int(), dim=1)
                # A match at index 0 still advances by one bin.
                new_index_increment[torch.logical_and(new_index_increment == 0, torch.any(mask, 1))] += 1

                rw_s[ret_world_indices] += new_index_increment
                # Do not advance past the end of the trajectory.
                rw_s[ret_world_indices] = torch.minimum(rw_s[ret_world_indices], rw_ts[ret_world_indices])

                bins_to_write = rw_tb[ret_world_indices, rw_s[ret_world_indices]].squeeze(1)
                # Send the next goal bin to the simulator.
                go_explore.setGoalBins(ret_world_indices, bins_to_write)

                # Worlds that reached the final (target) bin of their trajectory.
                world_final_bins = rw_tb[ret_world_indices, rw_ts[ret_world_indices], :]
                valid_final_mask = world_final_bins.squeeze(-1) != -1
                final_goal_mask = torch.logical_and(
                    torch.logical_and(
                        world_final_bins.squeeze() == curr_bins[ret_world_indices],
                        curr_bins[ret_world_indices] != -1
                    ),
                    valid_final_mask
                )
                goal_reached = ret_world_indices[final_goal_mask]

                if len(goal_reached) > 0:
                    # Clear the return state of the worlds that reached their target.
                    rw_tb[goal_reached] = -1
                    rw_ts[goal_reached] = -1
                    rw_s[goal_reached] = -1
                    go_explore.sim_steps_this_explore_step[goal_reached] = 0

                    # Target reached: half of these worlds get a new target bin ...
                    new_goal_bins = go_explore.sampleNewGoalBins(curr_bins[goal_reached])

                    # ... and half explore randomly.
                    random_exploration = torch.rand_like(goal_reached.float()) < 0.5
                    if go_explore.num_sil_worlds > 0:
                        random_exploration[goal_reached >= go_explore.sil_start_idx] = False
                    policy_exploration_worlds[goal_reached[random_exploration]] = False

                    # SIL worlds are never policy exploration worlds.
                    if go_explore.num_sil_worlds > 0:
                        policy_exploration_worlds[go_explore.sil_start_idx:] = False

                    # Reset returning worlds for random exploration.
                    rw_tb[~policy_exploration_worlds] = -1
                    rw_ts[~policy_exploration_worlds] = -1
                    rw_s[~policy_exploration_worlds] = -1

                    go_explore.setGoalBins(goal_reached[~random_exploration], new_goal_bins[~random_exploration])

                    go_explore.exploring_world_goal_cells[goal_reached[random_exploration]] = -1

                    # Entropy threshold e_t for policy exploration.
                    e_t = 50
                    go_explore.entropy_threshold[goal_reached] = e_t

            # In place: training holds this tensor as interruptible_user_cb.logit_scales.
            policy_entropy_scales_worlds[:] = 1 / computeEntropyScale(
                torch.maximum(torch.zeros_like(go_explore.sim_steps_this_explore_step), go_explore.sim_steps_this_explore_step),
                go_explore.entropy_threshold
            )

            if torch.any(policy_entropy_scales_worlds > 1) or torch.any(policy_entropy_scales_worlds <= 0):
                print("greater than 1 logit scale:", policy_entropy_scales_worlds[policy_entropy_scales_worlds > 1])
                print("less than 0 logit scale:", policy_entropy_scales_worlds[policy_entropy_scales_worlds <= 0])

            go_explore.generate_new_actions(action_policy=None, should_use_policy=None, world_indices=~policy_exploration_worlds)
            # Random actions for the worlds that explore randomly.
            go_explore.explore_from_state(
                explore_trajs,
                explore_bin_trajs,
                explore_steps,
                explore_actions,
                explore_jumps_remaining,
                explore_starting_bin,
                explore_starting_pos,
                explore_dynamic_block_trajs,
                explore_rewards
            )


    class interruptible_user_cb:
        # World weight is 0 if doing random exploration, 1 otherwise.
        world_weights = policy_exploration_worlds
        # Entropy scales are set per step for the *next* step's action selection.
        logit_scales = policy_entropy_scales_worlds
        # 1 for SIL worlds.
        sil_weights = sil_worlds

        def __call__(self, update_idx, update_time, update_results, learning_state):
            if solution_streaming_info["should_stop"]:
                solution_streaming_info["should_stop"] = False
                return False
            return train_wrapper(update_idx, update_time, update_results, learning_state)


    def _run_policy_loop():
        train(
            dev,
            SimInterface(
                step = explore_callback,
                obs = train_wrapper.obs,
                actions = train_wrapper.actions,
                dones = train_wrapper.dones,
                rewards = train_wrapper.rewards,
                resets = train_wrapper.resets,
                checkpoints = train_wrapper.checkpoints,
                checkpoint_resets = train_wrapper.checkpoint_resets,
            ),
            TrainConfig(
                run_name="PolicyBasedGoExplore",
                num_updates=int(total_steps / train_wrapper.steps_per_update),
                steps_per_update=train_wrapper.steps_per_update,
                num_bptt_chunks=1,
                lr=0.0002,
                gamma=0.998,
                gae_lambda=0.95,
                ppo = PPOConfig(
                    num_mini_batches=1,
                    clip_coef=0.2,
                    value_loss_coef=0.5,
                    use_intrinsic_loss=train_wrapper.use_intrinsic_loss,
                    value_loss_intrinsic_coef=0.5,
                    intrinsic_loss_coef=0.5,
                    entropy_coef=0.001,
                    max_grad_norm=0.5,
                    num_epochs=2,
                    # The Go-Explore supplement also clips the value loss, as in PPO.
                    clip_value_loss=True,
                    no_advantages=False,
                    use_sil_loss=True,
                ),
                value_normalizer_decay=0.999,
                mixed_precision=False,
                normalize_advantages=True,
                normalize_values=True,
            ),
            action_policy,
            interruptible_user_cb(),
            None
        )

        # Save the successful trajectories as a .pt file.
        if go_explore.store_successful_trajs:
            save_string = "successful_trajs.pt"
            save_successful_trajs(go_explore, save_string)
            print(f"Saved {len(go_explore.successful_trajs_pos)} successful trajectory(ies) to {save_string}")

    if run_in_background:
        t = threading.Thread(target=_run_policy_loop, daemon=True)
        t.start()
        return go_explore

    _run_policy_loop()
    return go_explore


def make_trainer(args, world_simulator, dev, starting_policy):
    """Build PBGE's PPO training wrapper once; go_explore_search_solvable.py's main
    gives it a new policy for every query (new_query_policy)."""
    from pathlib import Path
    import wandb
    # The PPO trainer logs through wandb; PBGE runs it with wandb disabled.
    wandb.init(mode="disabled")
    from madrona_scripts.simplePPO import PPOTrainingWrapper
    train_params = TrainParameters1(args.seed)
    transform_params = TransformParameters.default_params(args.num_worlds, dev)
    bin_coord_obs = torch.zeros((args.num_worlds, 6), device=dev)
    return PPOTrainingWrapper(
        world_simulator,
        dev,
        args.seed,
        train_params,
        transform_params,
        args.level_types or ["Plank"],
        num_components=1,
        no_level_obs=args.no_level_obs,
        num_channels=args.num_channels,
        separate_value=args.separate_values,
        use_intrinsic_loss=False,
        steps_per_update=args.steps_per_update,
        normalize_values=True,
        single_instance=True,
        disable_random_orientation=True,
        random_spawn=False,
        random_waypoint=False,
        perturb=None,
        perturbation_seed=0,
        run_inference_during_training=False,
        return_based_weighting=False,
        save_frequency=100,
        log_frequency=10,
        eval_frequency=0,
        ckpt_dir=Path(args.ckpt_dir),
        profile_report=False,
        starting_policy=starting_policy,
        level_json=None,
        expert_traj=None,
        bin_coord_obs=bin_coord_obs,
    )


def new_query_policy(train_wrapper, ckpt_dir):
    """Give the trainer a new, untrained policy for the next query, saving to ckpt_dir."""
    ACTION_DIMS = [2, 8, 3, 2]
    train_wrapper.policy = make_policy(
        train_wrapper.num_obs_features, None,
        train_wrapper.num_channels, train_wrapper.separate_value, ACTION_DIMS)
    train_wrapper.ckpt_dir = ckpt_dir
    train_wrapper.ckpt_dir.mkdir(parents=True, exist_ok=True)
