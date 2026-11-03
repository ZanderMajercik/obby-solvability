import argparse
import random
import numpy as np
import sys
import json
import threading
import warnings
from pathlib import Path

import madrona_puzzle_bench
import torch
import wandb
from flask import Flask, request
from madrona_puzzle_bench import SimFlags, RewardMode
from madrona_puzzle_bench_learn import (
    train, profile, TrainConfig, PPOConfig, SimInterface, LearningState
)

app = Flask(__name__)

from generateLevels import loadJsonDescriptionsFromDisk, generateJsonLevels, jsonDescriptionsToTensor, randomize_spawn_blocks, perturb_all_part_positions, select_specific_spawn_block, select_specific_perturbation_block, add_randomized_waypoints, inject_trajectory_waypoints
from infer import run_infer
from parameters.params_train import TrainParameters1, TrainParameters2
from parameters.params_transforms import TransformParameters, generateLimitationStepFunc, _sample_jumps
from policy import make_policy, setup_obs, process_obs
from encoder_trainer import EncoderTrainer
from obby.grammar.distribution import *
from obby.grammar.distribution import Constant

warnings.filterwarnings("error")
import os

# Seed torch once so that torch operations repeat across runs.
torch.manual_seed(0)


arg_parser = argparse.ArgumentParser()

# General args
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--ckpt-dir', type=str, required=True)
arg_parser.add_argument('--run-name', type=str, required=True)
arg_parser.add_argument('--profile-report', action='store_true')
arg_parser.add_argument('--fp16', action='store_true')

# World gen args
arg_parser.add_argument('--level-types', type=str, nargs='+',
        help="A set of level types backed by generators from the DSL.")
arg_parser.add_argument('--level-json-file', type=str, default=None, help='JSON file specification of obby to train.')
arg_parser.add_argument('--num-components', type=int, default=1,
        help="Total number of components drawn from specifications by '--level-types' or '--level-desc'.")
arg_parser.add_argument('--use-fixed-world', action='store_true')
arg_parser.add_argument('--reward-mode', type=str, required=True)
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--gpu-sim', action='store_true')
arg_parser.add_argument('--no-level-obs', action='store_true')
arg_parser.add_argument('--disable-random-orientation', action='store_true')
arg_parser.add_argument('--seed', type=int, default=5)
arg_parser.add_argument('--transform-params-file', type=str, default=None, help='JSON file containing transform params.')
arg_parser.add_argument('--expert-traj', type=str, default=None, help='Filename or directory of expert trajectories.')

# Learning args
arg_parser.add_argument('--num-updates', type=int, required=True)
arg_parser.add_argument('--break-when-solved', action='store_true',
        help='Stop training as soon as any single world reaches the solved reward. '
             'reward_max is the max over ALL worlds, so on a multi-level generalist run '
             'this fires within a few updates -- leave it off for generalist training.')
arg_parser.add_argument('--steps-per-update', type=int, default=40)
arg_parser.add_argument('--num-bptt-chunks', type=int, default=1)
arg_parser.add_argument('--lr', type=float, default=1e-4)
arg_parser.add_argument('--gamma', type=float, default=0.998)
arg_parser.add_argument('--entropy-loss-coef', type=float, default=0.01)
arg_parser.add_argument('--value-loss-coef', type=float, default=0.5)
arg_parser.add_argument('--clip-value-loss', action='store_true')
arg_parser.add_argument('--no-value-norm', action='store_true')
arg_parser.add_argument('--no-advantage-norm', action='store_true')
arg_parser.add_argument('--no-advantages', action='store_true')
arg_parser.add_argument('--value-normalizer-decay', type=float, default=0.999)
arg_parser.add_argument('--use-complex-level', action='store_true')
arg_parser.add_argument('--use-intrinsic-loss', action='store_true')
arg_parser.add_argument('--random-spawn', action='store_true')
arg_parser.add_argument('--save-frequency', type=int, default=100, help='How often to save checkpoints (in number of updates)')
# Forward-dynamics model trained alongside PPO (same model as CFLGE's encoder)
arg_parser.add_argument('--train-dynamics', action='store_true',
        help='Train a forward-dynamics model (same architecture/objective as the CFLGE-scratch '
             'dynamics model) on the same rollout transitions PPO trains on. Checkpointed as '
             '<update>_dynamics.pth alongside the policy.')
arg_parser.add_argument('--dynamics-pred-hidden', type=int, default=256,
        help='Hidden/latent dim of the dynamics encoder+predictor (default: 256).')
arg_parser.add_argument('--dynamics-lr', type=float, default=1e-4, help='Dynamics model Adam lr (default: 1e-4).')
arg_parser.add_argument('--dynamics-epochs', type=int, default=2, help='Dynamics epochs per PPO update (default: 2).')
arg_parser.add_argument('--dynamics-minibatch', type=int, default=512, help='Dynamics minibatch size (default: 512).')
arg_parser.add_argument('--eval-frequency', type=int, default=10, help='How often to run inference evaluation (in number of updates)')
arg_parser.add_argument('--log-frequency', type=int, default=10, help='How often to print training logs (in number of updates)')
arg_parser.add_argument('--run-inference-during-training', action='store_true',
                       help='Run clean inference evaluation during training to track true policy performance. '
                            'Creates a separate CPU simulation with no action/observation transforms to measure '
                            'pure policy capability. Runs every --eval-frequency updates. '
                            'PERFORMANCE NOTE: Adds significant overhead (~5-10 seconds per eval on 300 worlds). '
                            'Disable this flag for faster training if you only care about final performance.')

arg_parser.add_argument('--unlock-agent-orientation', action='store_true')
arg_parser.add_argument('--disable-phys-obs', action='store_true')
arg_parser.add_argument('--disable-step-obs', action='store_true')
arg_parser.add_argument('--disable-lidar', action='store_true')
arg_parser.add_argument('--jumps-remaining-obs', action='store_true', help='Include jump budget in the observation space.')
arg_parser.add_argument('--jumps-remaining-limitation', action='store_true', help='Mask jump actions when per-episode jump budget is exhausted.')

# Architecture args
arg_parser.add_argument('--starting-policy', type=str, default=None, help="Start this training from an existing policy.")
arg_parser.add_argument('--num-channels', type=int, default=256)
arg_parser.add_argument('--separate-value', action='store_true')

# Agent transform args. They have no defaults; params_transforms.py gives the default behavior.
arg_parser.add_argument('--delay-buffer-size', type=int)
arg_parser.add_argument('--p-drop', type=float)
arg_parser.add_argument('--p-random', type=float)
arg_parser.add_argument('--p-repeat', type=float)
arg_parser.add_argument('--rate-frequency', type=int)
arg_parser.add_argument('--rate-frequency-std', type=float)
arg_parser.add_argument('--noise-mean', type=float)
arg_parser.add_argument('--noise-std', type=float)
arg_parser.add_argument('--no-jump', action='store_true')

# Training server
arg_parser.add_argument('--training-server', action='store_true')

arg_parser.add_argument('--single-instance', action='store_true', help='Train on a single level instance across all worlds.')
arg_parser.add_argument('--perturb', type=str, nargs='?', const='all', default=None, 
                       help='Apply physical perturbations to obstacle blocks:\n'
                            '  --perturb         : Perturb all blocks (same as "all")\n'
                            '  --perturb all     : Perturb all blocks\n'
                            '  --perturb random  : Perturb a single randomly selected block\n'
                            '  --perturb X       : Perturb block at index X (0-based)\n'
                            '  (no flag)         : No perturbation applied')
arg_parser.add_argument('--perturbation-seed', type=int, default=None, help='Specific seed for perturbations. If not provided, uses --seed.')
arg_parser.add_argument('--spawn-block-index', type=int, default=None, help='Select a specific spawn block by index (0-based). If not specified, uses default spawn or random spawn if --random-spawn is enabled.')

# Waypoint arguments
arg_parser.add_argument('--waypoint', action='store_true',
                       help='If true, generate levels with randomly spawned waypoints.')

arg_parser.add_argument('--return-based-weighting', action='store_true', help='Enable return-based importance weighting for training.')
arg_parser.add_argument('--weight-temperature', type=float, default=0.1, help='Temperature parameter for return-based weighting (lower = more selective).')

class PPOTrainingWrapper:
    def __init__(self,
                 world_simulator,
                 device,
                 seed,
                 train_params,
                 transform_params,
                 level_types,
                 num_components,
                 no_level_obs,
                 num_channels,
                 separate_value,
                 use_intrinsic_loss,
                 steps_per_update,
                 normalize_values,
                 single_instance,
                 disable_random_orientation,
                 random_spawn,
                 random_waypoint,
                 perturb,
                 perturbation_seed,
                 run_inference_during_training,
                 return_based_weighting,
                 save_frequency,
                 log_frequency,
                 eval_frequency,
                 ckpt_dir,
                 profile_report = False,
                 starting_policy = None,
                 level_json = None,
                 expert_traj = None,
                 bin_coord_obs = None,
                 jumps_remaining_obs = False,
                 jumps_remaining_limitation = False,
                 train_dynamics = False,
                 dynamics_pred_hidden = 256,
                 dynamics_lr = 1e-4,
                 dynamics_epochs = 2,
                 dynamics_minibatch = 512,
                 break_when_solved = False):

        self.device = device

        self.worlds = world_simulator

        self.obs_transform = transform_params.get_obs_transform()
        self.action_transform = transform_params.get_action_transform()


        self.level_selector_rng = np.random.default_rng(seed=seed)

        self.loaded_json_descriptions = None
        if level_json:
            self.loaded_json_descriptions = list(loadJsonDescriptionsFromDisk(level_json).values())


        self.seed = seed
        self.level_types = level_types
        self.num_components = num_components
        self.train_params = train_params
        self.expert_traj = expert_traj
        self.normalize_values = normalize_values

        # Settings used only by the training sim
        self.single_instance = single_instance
        self.disable_random_orientation = disable_random_orientation
        self.random_spawn = random_spawn
        self.random_waypoint = random_waypoint
        self.perturb = perturb
        self.perturbation_seed = perturbation_seed
        self.run_inference_during_training = run_inference_during_training
        self.break_when_solved = break_when_solved
        self.return_based_weighting = return_based_weighting
        self.use_intrinsic_loss = use_intrinsic_loss
        self.save_frequency = save_frequency
        self.log_frequency = log_frequency
        self.eval_frequency = eval_frequency

        self.regenerateJsonDescriptions(self.seed)

        # Map each simulator world to a JSON level description.
        json_indices = self.worlds.json_index_tensor().to_torch()
        num_worlds = json_indices.shape[0]

        json_descriptions_tensor = self.worlds.json_level_descriptions_tensor().to_torch()
        if single_instance:
            # Every world uses the first level description.
            json_indices[:, 0] = 0
        else:
            for i in range(num_worlds):
                json_indices[i, 0] = i % json_descriptions_tensor.shape[0]

        self.num_worlds = num_worlds
        self.steps_per_update = steps_per_update

        self.curr_returns = torch.zeros(num_worlds, device=device)  # cumulative return of each world

        self.training_start_time = None
        self.total_training_time = 0.0  # training time without inference
        self.last_update_time = None

        # Return-based weighting
        self.world_weights = torch.ones(num_worlds, device=device)
        self.return_history = []  # returns kept for debugging

        # Action bucket counts; they must match the simulator's action space.
        actions_num_buckets = [2, 8, 3, 2]

        if jumps_remaining_obs or jumps_remaining_limitation:
            self.jumps_remaining = _sample_jumps(num_worlds, device).unsqueeze(1)
        else:
            self.jumps_remaining = None

        self.bin_coord_obs = bin_coord_obs
        self.obs, self.num_obs_features = setup_obs(
            self.worlds,
            self.worlds.enum_counts_tensor().to_torch().to(device),
            no_level_obs,
            jumps_remaining=self.jumps_remaining if jumps_remaining_obs else None,
            bin_coord_obs=bin_coord_obs,
        )

        self.num_channels = num_channels
        self.separate_value = separate_value
        self.policy = make_policy(self.num_obs_features, None, num_channels, separate_value, actions_num_buckets, intrinsic=use_intrinsic_loss)
        
        if starting_policy:
            weights = LearningState.load_policy_weights(starting_policy)
            key_load = self.policy.load_state_dict(weights, strict=False)
            if key_load.missing_keys or key_load.unexpected_keys:
                # Partial loads are allowed here (e.g. new intrinsic heads), but say so.
                print(f"[warn] starting policy {starting_policy}: missing={key_load.missing_keys} "
                      f"unexpected={key_load.unexpected_keys}")

        self.actions = self.worlds.action_tensor().to_torch()
        self.dones = self.worlds.done_tensor().to_torch()
        self.rewards = self.worlds.reward_tensor().to_torch()
        self.checkpoints = self.worlds.checkpoint_tensor().to_torch()
        self.checkpoint_resets = self.worlds.checkpoint_reset_tensor().to_torch()
        self.resets = self.worlds.reset_tensor().to_torch()

        # Reset the worlds after level generation.
        self.resets.fill_(1.0)
        self.worlds.step()
        self.resets.fill_(0.0)

        self.mean_fps = 0
        self.ckpt_dir = ckpt_dir
        self.profile_report = profile_report

        # Optional forward-dynamics model, trained alongside PPO on the same
        # rollout transitions. It is the same model and objective as CFLGE's encoder
        # (EncoderTrainer / DynamicsFinetuneModel): encode(s_t) plus an action embedding
        # predicts s_{t+1}, with an MSE loss over the full process_obs vector. It is saved
        # as <update>.pth in a dynamics/ folder, in the {model, obs_dim, latent_dim,
        # global_step} format that the CFLGE encoder loader reads.
        self.train_dynamics = train_dynamics
        self.dynamics_trainer = None
        self.dynamics_global_step = 0
        if train_dynamics:
            with torch.no_grad():
                _po = process_obs(*self.obs)
                if isinstance(_po, tuple):
                    _po = torch.cat([_po[0], _po[1].reshape(_po[1].shape[0], -1)], dim=-1)
                target_obs_dim = int(_po.shape[-1])
            self.dynamics_trainer = EncoderTrainer(
                base_backbone=None,
                action_dims=actions_num_buckets,
                device=self.device,
                latent_dim=dynamics_pred_hidden,
                target_obs_dim=target_obs_dim,
                process_obs_fn=process_obs,
                raw_obs_kept_cols=None,  # full observation, no column mask
                hidden_dim=dynamics_pred_hidden,
                lr=dynamics_lr,
                n_epochs=dynamics_epochs,
                minibatch_size=dynamics_minibatch,
                verbose=True,
            )
            # Dynamics checkpoints get their own folder, apart from the policy .pth files.
            self.dynamics_ckpt_dir = self.ckpt_dir / "dynamics"
            self.dynamics_ckpt_dir.mkdir(parents=True, exist_ok=True)
            print(f"[dynamics] training enabled: obs_dim={target_obs_dim} "
                  f"latent_dim={dynamics_pred_hidden} lr={dynamics_lr} "
                  f"epochs={dynamics_epochs} minibatch={dynamics_minibatch} "
                  f"ckpt_dir={self.dynamics_ckpt_dir}")

        # Hooks for PBGE: per-world update weights and logit scales (None = off).
        self.world_weights = None
        self.logit_scales = None
        self.sil_weights = None

    def regenerateJsonDescriptions(self, seed):
        json_descriptions_tensor = self.worlds.json_level_descriptions_tensor().to_torch()

        if self.loaded_json_descriptions:
            json_levels_parts = [self.loaded_json_descriptions[i % len(self.loaded_json_descriptions)] for i in range(json_descriptions_tensor.shape[0])]
        else:
            json_levels_parts = generateJsonLevels(
                json_descriptions_tensor.shape[0],
                seed,
                self.level_types,
                self.level_selector_rng,
                self.num_components,
                self.train_params,
                Constant(0) if self.disable_random_orientation else None
            )
            if self.random_spawn:
                json_levels_parts = [json.loads(json_level_parts) for json_level_parts in json_levels_parts]
                randomize_spawn_blocks(json_levels_parts, self.level_selector_rng)
                json_levels_parts = [json.dumps(json_level_parts) for json_level_parts in json_levels_parts]
            if self.random_waypoint:
                json_levels_parts = [json.loads(json_level_parts) for json_level_parts in json_levels_parts]
                add_randomized_waypoints(json_levels_parts, self.level_selector_rng)
                json_levels_parts = [json.dumps(json_level_parts) for json_level_parts in json_levels_parts]
        
        # Inject waypoints from the expert trajectory, if one is given.
        if self.expert_traj is not None:
            json_levels_parts = [json.loads(json_level_parts) for json_level_parts in json_levels_parts]
            inject_trajectory_waypoints(json_levels_parts, self.expert_traj, num_waypoints=5)
            json_levels_parts = [json.dumps(json_level_parts) for json_level_parts in json_levels_parts]
    
        # Apply the physical perturbations of --perturb.
        if self.perturb is not None:
            json_levels_parts = [json.loads(json_level_parts) for json_level_parts in json_levels_parts]
            perturbation_seed = self.perturbation_seed if self.perturbation_seed is not None else seed
            rng = random.Random(perturbation_seed)
            
            if self.perturb == "all":
                print("Applying perturbations to all blocks")
                for json_level in json_levels_parts:
                    perturb_all_part_positions(json_level, perturbation_seed)
            elif self.perturb == "random":
                print("Applying single random block perturbation")
                for json_level in json_levels_parts:
                    all_blocks = list(json_level.keys())
                    if all_blocks:
                        random_block_index = rng.randint(0, len(all_blocks) - 1)
                        actual_perturb_index = select_specific_perturbation_block([json_level], random_block_index, perturbation_seed)
            elif self.perturb.isdigit():
                block_index = int(self.perturb)
                print(f"Applying perturbation to block {block_index}")
                actual_perturb_index = select_specific_perturbation_block(json_levels_parts, block_index, perturbation_seed)
                if actual_perturb_index != block_index:
                    print(f"  Note: Actual block index perturbed: {actual_perturb_index} (clamped from {block_index})")
            else:
                print(f"Warning: Unknown perturbation argument '{self.perturb}'. Applying all perturbations.")
                for json_level in json_levels_parts:
                    perturb_all_part_positions(json_level, perturbation_seed)
                    
            json_levels_parts = [json.dumps(json_level) for json_level in json_levels_parts]

        json_descriptions_tensor[:] = jsonDescriptionsToTensor(
            json_levels_shape=json_descriptions_tensor.shape, 
            json_levels_parts=json_levels_parts, 
            level_seed=seed,  
            json_levels_traj=self.expert_traj).to(self.device)

    def compute_return_based_weights(self, update_results):
        """Per-world importance weights from the episode returns."""
        with torch.no_grad():
            returns = update_results.returns
            dones = update_results.dones
            
            # Use the last timestep.
            if len(returns.shape) > 1:
                returns = returns[-1]
            if len(dones.shape) > 1:
                dones = dones[-1]
                
            returns = returns.squeeze()
            dones = dones.squeeze()
            
            if not self.return_based_weighting:
                return torch.ones(self.num_worlds, device=self.device, dtype=returns.dtype)
            
            done_mask = dones == 1.0
            
            if done_mask.sum() > 0:
                completed_returns = returns[done_mask]
                
                # Keep the returns for debugging.
                self.return_history.extend(completed_returns.cpu().tolist())
                
                # Softmax over the returns: a higher return gets a higher weight.
                weights_completed = torch.softmax(completed_returns / self.weight_temperature, dim=0)
                
                # Worlds that did not finish get a base weight of 0.1.
                world_weights = torch.full((self.num_worlds,), 0.1, device=self.device, dtype=completed_returns.dtype)
                
                # Scale the finished worlds' weights so that the largest is 1.0.
                max_weight = weights_completed.max()
                scaled_weights = weights_completed / max_weight
                
                world_weights[done_mask] = scaled_weights
                
                self.world_weights = world_weights
                return world_weights
            else:
                # No finished episodes: equal weights.
                return torch.ones(self.num_worlds, device=self.device, dtype=returns.dtype)

    def run_inference_eval(self, num_eval_worlds=300, num_trials=1):
        """Success rate (%) of the current policy, measured with infer.run_infer on
        the training sim (single-instance or multi-instance, as in training)."""

        print(f"  Running inference evaluation on {num_eval_worlds} worlds...")

        sim = self.worlds

        self.regenerateJsonDescriptions(self.seed)

        # TODO: reuse modifiedStep from __main__ instead of repeating this logic.
        json_levels = sim.json_level_descriptions_tensor().to_torch()
        json_indices = sim.json_index_tensor().to_torch()

        if self.single_instance:
            # Every world uses the first level.
            json_indices[:, 0] = 0
        else:
            # Spread the worlds over the levels.
            for i in range(num_eval_worlds):
                json_indices[i, 0] = i % json_levels.shape[0]

        # Run inference with the current limitation settings.
        infer_data = run_infer(
            sim=sim,
            transform_params=transform_params,
            policies=self.policy,
            obs=self.obs,
            num_worlds=num_eval_worlds,
            num_trials=num_trials,
            episode_len=3000,  # same as the training episode length
            device=self.device,
            logging=False,
            single_instance=self.single_instance
        )

        success_count = infer_data['success_per_policy']
        total_attempts = num_eval_worlds * num_trials
        success_rate = (success_count / total_attempts) * 100.0

        return success_rate

    # Forward-dynamics model (trained alongside PPO)
    def _train_dynamics_on_rollout(self, update_results):
        """Feed this update's rollout transitions to the dynamics trainer.

        update_results.obs is a tuple of raw obs components shaped (C, T, N, *f)
        (C = bptt chunks, T = steps/chunk, N = worlds); actions/dones come as
        (C*T, N, *). We form (s_t, a_t) -> s_{t+1} pairs from consecutive steps
        within each chunk and drop transitions where the episode ended at t
        (done_t = 1), since s_{t+1} is then a reset, not the true successor.
        """
        dyn = self.dynamics_trainer
        obs = update_results.obs
        C, T, N = obs[0].shape[0], obs[0].shape[1], obs[0].shape[2]
        if T < 2:
            return
        A = update_results.actions.shape[-1]
        actions = update_results.actions.view(C, T, N, A)
        dones = update_results.dones.view(C, T, N, -1)

        def _flat(x, sl):                       # x: (C, T, N, *f) -> (C*len(sl)*N, *f)
            xs = x[:, sl]
            return xs.reshape(-1, *xs.shape[3:])

        valid = (_flat(dones, slice(0, T - 1)).reshape(-1) == 0)
        if not bool(valid.any()):
            return
        obs_t = [_flat(c, slice(0, T - 1))[valid] for c in obs]
        obs_next = [_flat(c, slice(1, T))[valid] for c in obs]
        act_t = _flat(actions, slice(0, T - 1))[valid]

        dyn.store_obs_t(obs_t)
        dyn.store_step(act_t, obs_next)
        loss = dyn.train_minibatches()
        self.dynamics_global_step += 1
        if loss is not None:
            print(f"    [dynamics] loss={loss:.4f} (transitions={int(act_t.shape[0])})")

    def _save_dynamics_ckpt(self, path):
        dyn = self.dynamics_trainer
        torch.save({
            'model': dyn.model.state_dict(),
            'obs_dim': dyn.target_obs_dim,
            'latent_dim': dyn.latent_dim,
            'action_buckets': dyn.action_buckets,
            'global_step': self.dynamics_global_step,
        }, path)
        print(f"[dynamics] saved {path}")

    # Learning callback, called after each PPO update.
    def __call__(self, update_idx, update_time, update_results, learning_state):
        
        import time
        
        update_id = update_idx + 1

        # Train the forward-dynamics model on this update's rollout transitions
        # (the data PPO just trained on), if it is enabled.
        if self.dynamics_trainer is not None:
            self._train_dynamics_on_rollout(update_results)

        fps = self.num_worlds * self.steps_per_update / update_time
        # Training time: the time between updates, without inference.
        current_time = time.time()
        if self.training_start_time is None:
            self.training_start_time = current_time
            self.last_update_time = current_time
        else:
            time_since_last = current_time - self.last_update_time
            self.total_training_time += time_since_last
            self.last_update_time = current_time
        
        
        # Every eval_frequency updates, run an inference evaluation (if enabled).
        if self.run_inference_during_training and update_id % self.eval_frequency == 0:
            print(f"🔍 Running inference evaluation at update {update_id}...")
            
            # Stop counting training time during inference.
            inference_start_time = time.time()
            
            inference_success_rate = self.run_inference_eval(num_eval_worlds=self.num_worlds, num_trials=1)
            if inference_success_rate is not None:
                print(f"🎯 INFERENCE EVAL - Success Rate: {inference_success_rate:.2f}% ({self.num_worlds} worlds, 1 trial each)")
                
                # Inference metrics use the update ID as their x-axis, not wandb's default step.
                # The default step is not update ID / 10 once inference runs in addition to
                # training, so the inference plots need their own x-axis field.
                wandb.log({
                    "inference_success_rate": inference_success_rate,
                    "inference_metrics/success_rate_vs_updates": inference_success_rate,
                    "inference_update_id": update_id,  # x-axis of the inference plots
                })  # no step argument, so other metrics keep the default step
            else:
                print(f"❌ Inference eval failed")
            
            # Resume the training time; inference time is excluded.
            self.last_update_time = time.time()

        self.mean_fps += (fps - self.mean_fps) / update_id

        skip_log = False
        if update_id != 1 and update_id % self.log_frequency != 0:
            skip_log = True

        ppo = update_results.ppo_stats

        with torch.no_grad():
            reward_max = update_results.rewards.max().cpu().item()

        if not skip_log:
            with torch.no_grad():
                reward_mean = update_results.rewards.mean().cpu().item()
                reward_min = update_results.rewards.min().cpu().item()

                done_count = (update_results.dones == 1.0).sum()

                # Solved episodes: steps where the reward reaches 10 (the goal).
                solve_count = int((update_results.rewards.reshape(-1) >= 9.9999).sum().cpu().item())
                solve_rate = solve_count / self.num_worlds

                return_mean, return_max = 0, 0
                if done_count > 0:
                    return_mean = update_results.returns[update_results.dones == 1.0].mean().cpu().item()
                    return_max = update_results.returns[update_results.dones == 1.0].max().cpu().item()

                value_mean = update_results.values.mean().cpu().item()
                value_min = update_results.values.min().cpu().item()
                value_max = update_results.values.max().cpu().item()

                advantage_mean = update_results.advantages.mean().cpu().item()
                advantage_min = update_results.advantages.min().cpu().item()
                advantage_max = update_results.advantages.max().cpu().item()

                vnorm_mu = 0
                vnorm_sigma = 0
                if self.normalize_values:
                    vnorm_mu = learning_state.value_normalizer.mu.cpu().item()
                    vnorm_sigma = learning_state.value_normalizer.sigma.cpu().item()

            print(f"\nUpdate: {update_id}")
            print(f"    Loss: {ppo.loss: .3e}, A: {ppo.action_loss: .3e}, V: {ppo.value_loss: .3e}, E: {ppo.entropy_loss: .3e}")
            print()
            print(f"    Rewards          => Avg: {reward_mean: .3e}, Min: {reward_min: .3e}, Max: {reward_max: .3e}")
            print(f"    Solves           => Count: {solve_count}, Rate: {solve_rate:.4f} ({solve_count}/{self.num_worlds})")
            print(f"    Values           => Avg: {value_mean: .3e}, Min: {value_min: .3e}, Max: {value_max: .3e}")
            print(f"    Advantages       => Avg: {advantage_mean: .3e}, Min: {advantage_min: .3e}, Max: {advantage_max: .3e}")
            print(f"    Returns          => Mean: {return_mean: .3e}, Max: {return_max: .3e} (over {int(done_count)} done episodes)")
            print(f"    Value Normalizer => Mean: {vnorm_mu: .3e}, σ: {vnorm_sigma :.3e}")
        
            if self.return_based_weighting:
                weight_mean = self.world_weights.mean().cpu().item()
                weight_std = self.world_weights.std().cpu().item()
                weight_max = self.world_weights.max().cpu().item()
                weight_min = self.world_weights.min().cpu().item()
                print(f"    World Weights    => Mean: {weight_mean:.3f}, Std: {weight_std:.3f}, Min: {weight_min:.3f}, Max: {weight_max:.3f}")
        
            wandb.log({
                "update_id": update_id,
                "loss": ppo.loss,
                "action_loss": ppo.action_loss,
                "value_loss": ppo.value_loss,
                "entropy_loss": ppo.entropy_loss,
                "reward_mean": reward_mean,
                "reward_max": reward_max,
                "returns_mean": return_mean,
                "returns_max": return_max,
                "solve_count": solve_count,
                "solve_rate": solve_rate,
                "value_mean": value_mean,
                "value_max": value_max,
                "advantage_mean": advantage_mean,
                "advantage_max": advantage_max,
                "training_time_seconds": self.total_training_time,
                "world_weights_mean": self.world_weights.mean().cpu().item() if self.return_based_weighting else 1.0,
                "world_weights_std": self.world_weights.std().cpu().item() if self.return_based_weighting else 0.0,
                "world_weights_max": self.world_weights.max().cpu().item() if self.return_based_weighting else 1.0,
                }
            )
            if self.use_intrinsic_loss:
                wandb.log({
                    "update_id": update_id,
                    "intrinsic_loss": ppo.intrinsic_loss,
                    "value_loss_intrinsic": ppo.value_loss_intrinsic,
                })

            if self.profile_report:
                print()
                print(f"    FPS: {fps:.0f}, Update Time: {update_time:.2f}, Avg FPS: {self.mean_fps:.0f}")
                print(f"    PyTorch Memory Usage: {torch.cuda.memory_reserved() / 1024 / 1024 / 1024:.3f}GB (Reserved), {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024:.3f}GB (Current)")
                profile.report()

            # Every save_frequency updates, save checkpoints and regenerate the levels.
            if update_id % self.save_frequency == 0:              
                
                if not self.single_instance:
                    self.regenerateJsonDescriptions(self.seed)
                    print("Regenerating obstacles...")

                learning_state.save(update_idx, self.ckpt_dir / f"{update_id}.pth")
                if self.dynamics_trainer is not None:
                    self._save_dynamics_ckpt(self.dynamics_ckpt_dir / f"{update_id}.pth")

        if self.break_when_solved and reward_max >= 9.9999:
            print(f"[SOLVED] reward_max={reward_max:.2f} at update {update_id} — stopping early.")
            return False
        return True

@app.route('/finetunePolicyOnSection', methods=['POST'])
def finetunePolicyOnSection():
    global train_ppo
    section_and_policy = request.get_json()

    # Convert to JSON that json.loads() can parse.
    obbyJsonStr = str(section_and_policy["obbyJson"]).replace("'", "\"").replace("False", "false").replace("True", "true")
    policyDictTensor = {key : torch.tensor(section_and_policy["current_policy_state_dict"][key]).to(train_ppo.device) for key in section_and_policy["current_policy_state_dict"].keys()}

    # TODO: policy finetuning on the received section is not implemented; the route
    # returns the current policy weights unchanged.
    return {key : train_ppo.policy.state_dict()[key].tolist() for key in train_ppo.policy.state_dict().keys()}

if __name__ == "__main__":

    args = arg_parser.parse_args()
    
    normalize_values = not args.no_value_norm
    normalize_advantages = not args.no_advantage_norm

    sim_flags = SimFlags.Default
    print(sim_flags)
    if args.use_fixed_world:
        sim_flags |= SimFlags.UseFixedWorld
    if args.use_complex_level:
        sim_flags |= SimFlags.UseComplexLevel
    if args.unlock_agent_orientation:
        sim_flags |= SimFlags.UnlockAndRandomizeAgentOrientation
    if args.disable_phys_obs:
        sim_flags |= SimFlags.DisablePhysObs
    if args.disable_step_obs:
        sim_flags |= SimFlags.DisableStepObs
    if args.disable_lidar:
        sim_flags |= SimFlags.DisableLidar
    print(sim_flags)

    reward_mode = getattr(RewardMode, args.reward_mode)

    if torch.cuda.is_available() and args.gpu_sim:
        dev = torch.device(f'cuda:{args.gpu_id}')
    else:
        dev = torch.device('cpu')

    ckpt_dir = Path(args.ckpt_dir)

    ckpt_dir.mkdir(exist_ok=True, parents=True)

    with open(os.path.join(ckpt_dir, 'argv.txt'), 'w') as f:
        f.write(' '.join(sys.argv))

    # Create the training parameters and save them.
    train_params = TrainParameters1(level_seed=args.seed)
    train_params.save(os.path.join(ckpt_dir, 'train_params.py'))

    transform_params = TransformParameters(num_worlds=args.num_worlds, device=dev)
    
    if args.transform_params_file is not None:
        transform_params.from_json(args.transform_params_file)
    else:
        assert None not in [args.delay_buffer_size, args.p_drop, args.p_random,
                            args.p_repeat, args.rate_frequency, args.rate_frequency_std,
                            args.noise_mean, args.noise_std], "Must provide all transform parameters if not using a file."
        # Use the transform parameters from the command line.
        transform_params.from_params(
            p_drop = args.p_drop,
            p_random = args.p_random,
            p_repeat = args.p_repeat,
            delay_buffer_size = args.delay_buffer_size,
            rate_frequency = args.rate_frequency,
            rate_frequency_std = args.rate_frequency_std,
            no_jump = args.no_jump,
            noise_mean = args.noise_mean,
            noise_std = args.noise_std,
        )
        
    transform_params.save(os.path.join(ckpt_dir, 'transform_params.py'))
    transform_params.jsonify(os.path.join(ckpt_dir, 'transform_params.json'))
    

    world_simulator = madrona_puzzle_bench.SimManager(
        exec_mode = madrona_puzzle_bench.madrona.ExecMode.CUDA if args.gpu_sim else madrona_puzzle_bench.madrona.ExecMode.CPU,
        gpu_id = args.gpu_id,
        num_worlds = args.num_worlds,
        rand_seed = args.seed,
        sim_flags = (int)(sim_flags),
        reward_mode = reward_mode,
        episode_len = 3000,
        levels_per_episode = 1,
        button_width = 1.3,
        door_width = 20.0 / 3.,
        reward_per_dist = 0.05,
        # A negative slack reward can make the agent learn to jump backwards in the first
        # ~100 updates. It has little effect on long training runs.
        slack_reward = -0.005,
    )

    # init() must run outside any wrapper class that sets up the simulator.
    world_simulator.init()

    train_ppo = PPOTrainingWrapper(
        world_simulator, 
        dev,
        args.seed,
        train_params,
        transform_params,
        args.level_types,
        args.num_components,
        args.no_level_obs,
        args.num_channels,
        args.separate_value,
        args.use_intrinsic_loss,
        args.steps_per_update,
        normalize_values,
        args.single_instance,
        args.disable_random_orientation,
        args.random_spawn,
        args.waypoint,
        args.perturb,
        args.perturbation_seed,
        args.run_inference_during_training,
        args.return_based_weighting,
        args.save_frequency,
        args.log_frequency,
        args.eval_frequency,
        ckpt_dir,
        args.profile_report,
        args.starting_policy,
        args.level_json_file,
        args.expert_traj,
        jumps_remaining_obs=args.jumps_remaining_obs,
        jumps_remaining_limitation=args.jumps_remaining_limitation,
        train_dynamics=args.train_dynamics,
        dynamics_pred_hidden=args.dynamics_pred_hidden,
        dynamics_lr=args.dynamics_lr,
        dynamics_epochs=args.dynamics_epochs,
        dynamics_minibatch=args.dynamics_minibatch,
        break_when_solved=args.break_when_solved,
    )

    # Step function that applies the limitation profile of the simulator.
    limitation_step_func = generateLimitationStepFunc(
        world_simulator,
        train_ppo.obs,
        train_ppo.action_transform,
        train_ppo.obs_transform,
        dev,
        jumps_remaining=train_ppo.jumps_remaining,
        jumps_limitation=args.jumps_remaining_limitation,
    )

    def modifiedStep():
        if not args.single_instance:
            # Randomize each world's index into the JSON description buffer.
            json_indices = train_ppo.worlds.json_index_tensor().to_torch()
            json_indices[:] = torch.randint(0, 1024, json_indices.shape)
        
        # TODO: the trajectory sections for HandTunedTraj are placeholders (0-8), not real values.
        if args.reward_mode == "HandTunedTraj":
            trajectory_section = train_ppo.worlds.trajectory_section_tensor().to_torch()
            test_points = torch.arange(9, device=train_ppo.device)
            trajectory_section[...] = test_points

        limitation_step_func()

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "obby-solvability"),
        config=args,
        name=args.run_name
    )

    if args.training_server:
        # Run the Flask server in a background thread.
        t = threading.Thread(target=app.run, kwargs={"host" : '0.0.0.0', "port" : 3000})
        t.daemon = True
        t.start()
    train(
        dev,
        SimInterface(
            step = modifiedStep,
            obs = train_ppo.obs,
            actions = train_ppo.actions,
            dones = train_ppo.dones,
            rewards = train_ppo.rewards,
            resets = train_ppo.resets,
            checkpoints = train_ppo.checkpoints,
            checkpoint_resets = train_ppo.checkpoint_resets,
        ),
        TrainConfig(
            run_name = args.run_name,
            num_updates = args.num_updates,
            steps_per_update = args.steps_per_update,
            num_bptt_chunks = args.num_bptt_chunks,
            lr = args.lr,
            gamma = args.gamma,
            gae_lambda = 0.95,
            ppo = PPOConfig(
                num_mini_batches=1,
                clip_coef=0.2,
                value_loss_coef=args.value_loss_coef,
                use_intrinsic_loss=args.use_intrinsic_loss,
                value_loss_intrinsic_coef=0.5,  # not tuned
                intrinsic_loss_coef=0.5,  # not tuned
                entropy_coef=args.entropy_loss_coef,
                max_grad_norm=0.5,
                num_epochs=2,
                clip_value_loss=args.clip_value_loss,
                no_advantages=args.no_advantages,
            ),
            value_normalizer_decay = args.value_normalizer_decay,
            mixed_precision = args.fp16,
            normalize_advantages = normalize_advantages,
            normalize_values = normalize_values,
        ),
        train_ppo.policy,
        train_ppo,
        None
    )

    # Every checkpoint is written by now. scripts/bash/train_obs_ablation.sh
    # watches for this line: a process that hangs during shutdown after it
    # is stopped there, with its outputs complete.
    run.finish()
    print(f"TRAINING FINISHED: checkpoints in {args.ckpt_dir}", flush=True)
