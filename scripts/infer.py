import argparse
import json
import os
import random
import signal
import sys
import warnings
import time
import threading
import math
import typing as tp

import madrona_puzzle_bench
import numpy as np
import pandas as pd
import torch
from madrona_puzzle_bench import RewardMode
from madrona_puzzle_bench_learn import LearningState, ActionTransform, profile, AMPState
from obby.grammar.distribution import Constant
from obby.grammar.obby import Obby
from obby.grammar.part import Part

from flask import Flask, request
app = Flask(__name__)

from madrona_scripts.generateLevels import generateJsonLevelsTensor, generateJsonLevels, loadJsonDescriptionsFromDisk, jsonDescriptionsToTensor, randomize_spawn_blocks, perturb_all_part_positions, select_specific_spawn_block, select_specific_perturbation_block, add_randomized_waypoints
from madrona_scripts.parameters.params_train import *
from madrona_scripts.parameters.params_transforms import *
from madrona_scripts.policy import make_policy, setup_obs

warnings.filterwarnings("error")



arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--ckpt-path', type=str)
arg_parser.add_argument('--record-log', type=str)
arg_parser.add_argument('--difficulty-dir', type=str, default="difficulty")
arg_parser.add_argument('--store-positions', action='store_true')
arg_parser.add_argument('--trajectory-log', type=str)
arg_parser.add_argument('--trajectories-out', type=str, default=None)
arg_parser.add_argument('--level-info-out', type=str, default=None)
arg_parser.add_argument('--trajectories', type=str,
        help="JSON trajectory of actions to replay. \
            Contains world index and seed information (assuming same generators passed to --level-types).\
            Each trajectory is assumed to be copied to as many worlds specified: if you specify 8k worlds,\
            the trajectories in the file will play on each of them (useful for parameter sweeping).")
arg_parser.add_argument('--transform-params', type=str, default=None, help='JSON file containing transform params.')

arg_parser.add_argument('--seed', type=int, default=10)


arg_parser.add_argument('--use-fixed-world', action='store_true')
arg_parser.add_argument('--use-best-action', action='store_true')

arg_parser.add_argument('--profile-report', action='store_true')


# Level specification
arg_parser.add_argument('--level-types', type=str, nargs='+',
        help="A set of level types. 'All' denotes every generatable level type. 'Random' denotes random_obby() from the DSL.")
arg_parser.add_argument('--num-components', type=int, default=1,
        help="Total number of components drawn from specifications by '--level-types' or '--level-desc'.")
arg_parser.add_argument('--level-file', type=str, default=None, help='JSON file to load level from.')
arg_parser.add_argument('--level-str', type=str, nargs='+', default=None, help='Any number of JSON str to load level from, delimited by enclosing "".')
arg_parser.add_argument('--duplicate', action='store_true', help='If true, put copies of a single level in all worlds.')
arg_parser.add_argument('--single-instance', action='store_true', help='Infer on a single level instance across all worlds.')
arg_parser.add_argument('--save-json', action='store_true', help='If true, save JSON files for the generated levels.')
arg_parser.add_argument('--sample-params', type=str, nargs='+', default=None, help='The parameter to randomize (fix others).')
arg_parser.add_argument('--num-levels', type=int, default=1, help='The parameter to randomize (fix others).')
arg_parser.add_argument('--random-spawn', action='store_true')


arg_parser.add_argument('--num-worlds', type=int)#, required=True)
arg_parser.add_argument('--num-trials', type=int)#, required=True)
arg_parser.add_argument('--no-level-obs', action='store_true')
arg_parser.add_argument('--enable-random-level-rotation', action='store_true')

arg_parser.add_argument('--num-channels', type=int, default=256)
arg_parser.add_argument('--separate-value', action='store_true')
arg_parser.add_argument('--fp16', action='store_true')

arg_parser.add_argument('--gpu-sim', action='store_true')
arg_parser.add_argument('--use-complex-level', action='store_true')
arg_parser.add_argument('--use-normaabb-obs', action='store_true')
arg_parser.add_argument('--unlock-agent-orientation', action='store_true')
arg_parser.add_argument('--disable-step-obs', action='store_true')
arg_parser.add_argument('--disable-phys-obs', action='store_true')
arg_parser.add_argument('--disable-lidar', action='store_true')

arg_parser.add_argument('--episode-len', type=int, default=1000)
# Results log
arg_parser.add_argument('--store-results', action='store_true')
arg_parser.add_argument('--results-file', type=str)
arg_parser.add_argument('--batch-infer', action='store_true')

# Architecture args
arg_parser.add_argument('--entity-network', action='store_true')
arg_parser.add_argument('--write-binary-out', type=str, default=None,
                        help="JSON level description or folder of JSON level descriptions.")

# Agent transform parameters (no defaults here; the defaults are in params_transforms.py)
arg_parser.add_argument('--delay-buffer-size', type=int, default=1)
arg_parser.add_argument('--p-drop', type=float, default=0.0)
arg_parser.add_argument('--p-random', type=float, default=0.0)
arg_parser.add_argument('--p-repeat', type=float, default=0.0)
arg_parser.add_argument('--rate-frequency', type=int, default=1)
arg_parser.add_argument('--rate-frequency-std', type=float, default=1e-10)
arg_parser.add_argument('--no-jump', action='store_true')
arg_parser.add_argument('--noise-mean', type=float, default=0)
arg_parser.add_argument('--noise-std', type=float, default=1e-10)

# Physical perturbation of the obstacle during inference.
arg_parser.add_argument('--disable-random-orientation', action='store_true', help='Disable random level orientation to match training')
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

# Physics constants of the agent.
GRAVITY = 196.2
MOVE_FORCE = 198.75
JUMP_FORCE = 59.86
AIR_RES_FORCE = 3.5

ACTION_DIMS = [2, 8, 3, 2]

class InferArgs:
    def __init__(
            self,
        num_worlds: int,
        num_trials: int,
        ckpt_path: str,
        level_types: list[str] = ['All'],
        seed: int = 10,
        gpu_sim: bool = False,
        gpu_id: int = 0,
        episode_len: int = 1000,
        record_log: str = None,
        store_positions: bool = False,
        trajectory_log: str = None,
        trajectories_out: str = None,
        write_binary_out: str = "",
        level_info_out: str = None,
        transform_params: str = None,
        trajectories: str = None,
        level_desc: str = None,
        num_components: int = 1,
        use_complex_level: bool = False,
        use_normaabb_obs: bool = False,
        unlock_agent_orientation: bool = False,
        no_level_obs: bool = False,
        difficulty_dir: str = 'difficulty',
        use_fixed_world: bool = False,
        use_best_action: bool = False,
        level_file: str = None,
        level_str: str = None,
        duplicate: bool = False,
        save_json: bool = False,
        sample_params: bool = False,
        num_levels: int = 1,
        num_channels: int = 256,
        separate_value: bool = False,
        fp16: bool = False,
        store_results: bool = False,
        results_file: str = None,
        batch_infer: bool = False,
        entity_network: bool = False,
        delay_buffer_size: int = 1,
        p_drop: float = 0.0,
        p_random: float = 0.0,
        p_repeat: float = 0.0,
        rate_frequency: int = 1,
        rate_frequency_std: float = 1e-10,
        noise_mean: float = 0.0,
        noise_std: float = 1e-10,
        action_dims: list[int] = [2, 8, 3, 2],
        random_spawn_and_goal: bool = False,
        single_instance: bool = False,
        spawn_block_index: int = None,
        perturbation_seed: int = None,
        perturb: str = None,
        **kwargs
    ):
        self.num_worlds = num_worlds
        self.num_trials = num_trials
        self.ckpt_path = ckpt_path
        self.level_types = level_types
        self.seed = seed
        self.gpu_sim = gpu_sim
        self.gpu_id = gpu_id
        self.episode_len = episode_len
        self.record_log = record_log
        self.store_positions = store_positions
        self.trajectory_log = trajectory_log
        self.trajectories_out = trajectories_out
        self.write_binary_out = write_binary_out
        self.level_info_out = level_info_out
        self.trajectories = trajectories

        self.level_desc = level_desc
        self.num_components = num_components
        self.transform_params = transform_params
        self.use_complex_level = use_complex_level
        self.use_normaabb_obs = use_normaabb_obs
        self.unlock_agent_orientation = unlock_agent_orientation
        self.no_level_obs = no_level_obs
        self.difficulty_dir = difficulty_dir,
        self.use_fixed_world = use_fixed_world
        self.use_best_action = use_best_action
        self.level_file = level_file
        self.level_str = level_str
        self.duplicate = duplicate
        self.save_json = save_json
        self.sample_params = sample_params
        self.num_levels = num_levels
        self.num_channels = num_channels
        self.separate_value = separate_value
        self.fp16 = fp16
        self.store_results = store_results
        self.results_file = results_file
        self.batch_infer = batch_infer
        self.entity_network = entity_network
        self.delay_buffer_size = delay_buffer_size
        self.p_drop = p_drop
        self.p_random = p_random
        self.p_repeat = p_repeat
        self.rate_frequency = rate_frequency
        self.rate_frequency_std = rate_frequency_std
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.action_dims = action_dims
        self.random_spawn_and_goal = random_spawn_and_goal
        self.single_instance = single_instance
        self.spawn_block_index = spawn_block_index
        self.perturbation_seed = perturbation_seed
        self.perturb = perturb
        for k, v in kwargs.items():
            self.__dict__[k] = v
        
arg_parser.add_argument('--training-server', action='store_true')
arg_parser.add_argument('--max-solves', type=int, default=None,
                        help='Stop inference early once this many worlds have solved.')

def get_infer_args():
    return arg_parser

def setup_sim(
        num_worlds: int, episode_len: int, 
        gpu_sim: bool, gpu_id: int, 
        rand_seed: int,
        use_complex_level: bool = False, 
        use_normaabb_obs: bool = False, 
        unlock_agent_orientation: bool = False,
        disable_phys_obs: bool = False,
        disable_step_obs: bool = False,
        disable_lidar: bool = False,
        reward_mode: str = "Dense1"
):
    sim_flags = madrona_puzzle_bench.SimFlags.Default | madrona_puzzle_bench.SimFlags.NoActionDelay
    if use_complex_level:
        sim_flags |= madrona_puzzle_bench.SimFlags.UseComplexLevel
    if use_normaabb_obs:
        sim_flags |= madrona_puzzle_bench.SimFlags.UseNormalizedAABBObs
    if unlock_agent_orientation:
        sim_flags |= madrona_puzzle_bench.SimFlags.UnlockAndRandomizeAgentOrientation
    if disable_phys_obs:
        sim_flags |= madrona_puzzle_bench.SimFlags.DisablePhysObs
    if disable_step_obs:
        sim_flags |= madrona_puzzle_bench.SimFlags.DisableStepObs
    if disable_lidar:
        sim_flags |= madrona_puzzle_bench.SimFlags.DisableLidar

    sim = madrona_puzzle_bench.SimManager(
        exec_mode=madrona_puzzle_bench.madrona.ExecMode.CUDA if gpu_sim else madrona_puzzle_bench.madrona.ExecMode.CPU,
        gpu_id=gpu_id,
        num_worlds=num_worlds,
        rand_seed=rand_seed,
        sim_flags=(int)(sim_flags),
        reward_mode=getattr(RewardMode, reward_mode),
        episode_len=episode_len,
        reward_per_dist=0.05,
        levels_per_episode=1,
        button_width=1.3,
        door_width=20.0 / 3.,
        slack_reward=-0.005,
    )
    
    return sim    

def run_infer(
        sim, 
        transform_params: TransformParameters, 
        policies, # one policy or a list of policies
        obs,
        num_worlds: int,
        num_trials: int,
        episode_len: int,
        success_counts=None,
        pre_coordinate_transform_func=None,
        use_complex_level=False,
        use_normaabb_obs=False,
        unlock_agent_orientation=False,
        store_positions=False,
        use_best_action=False,
        record_log=None,
        device=torch.device('cpu'),
        logging=True,
        single_instance: bool = False,
        value_normalizer = None,
        max_solves = None,
    ):

    infer_data = {}

    policy = policies if not isinstance(policies, list) else None

    action_transform = transform_params.get_action_transform()
    obs_transform = transform_params.get_obs_transform()

    if store_positions:
        agent_pos_tensor = sim.agent_pos_tensor().to_torch().to(device)
        agent_jump_performed_tensor = sim.agent_jump_performed_tensor().to_torch().to(device)

    json_indices = sim.json_index_tensor().to_torch()
    json_levels = sim.json_level_descriptions_tensor().to_torch()
    resets = sim.reset_tensor().to_torch().to(device)

    timesteps = sim.timestep_tensor().to_torch()
    
    # The caller sets the JSON levels and indices on the simulator before this call.

    resets[:, 0] = 1

    sim.step()
        
    actions = sim.action_tensor().to_torch().to(device)
    dones = sim.done_tensor().to_torch().to(device)
    rewards = sim.reward_tensor().to_torch()

    timeclock = torch.ones((actions.shape[0],), dtype=torch.int8)  # for the action-delay transform

    ckpts = sim.checkpoint_tensor().to_torch()

    cur_rnn_states = []
    recurrent_cfg_shapes = policies[0].recurrent_cfg.shapes if isinstance(policies, list) else policies.recurrent_cfg.shapes
    for shape in recurrent_cfg_shapes:
        cur_rnn_states.append(torch.zeros(
            *shape[0:2], actions.shape[0], shape[2], dtype=torch.float32, device=torch.device('cpu')))

    if record_log:
        record_log = open(record_log, 'wb')
    else:
        record_log = None

    # Per-step records, filled when store_positions is set.
    all_positions = []
    all_jumps = []
    all_dones = []
    all_rewards = []
    all_timesteps = []
    all_move_dists = []
    all_direction_dists = []
    all_rotate_dists = []
    all_jump_dists = []
    all_values = []

    # Per-world trials remaining.
    if not isinstance(policies, list):
        remainingTrials = torch.zeros(num_worlds, dtype=torch.int32, device=device)
    else:
        # Several policies: each one gets its own block of worlds.
        assert success_counts is not None, "Must initialize success_counts manually if testing multiple policies."
        remainingTrials = torch.zeros(num_worlds * len(policies) * len(success_counts), dtype=torch.int32, device=device)
    

    remainingTrials[...] = num_trials
    if success_counts is None:
        success_counts = [0]

    print("Starting infer loop...")
    total_steps = 0
    start = time.time()
    success_per_world = torch.zeros(num_worlds).to(device)
    _prev_solve_count = 0

    def _print_stats():
        print("Total Steps: ", total_steps)
        print("Total Time: ", time.time() - start)
        for i in range(len(success_counts)):
            print(f"Policy {i} succeeded in {success_counts[i]} / {num_trials * num_worlds}")

    def _sigterm_handler(signum, frame):
        _print_stats()
        sys.exit(0)
    # Only the main thread can install a handler; run_infer also runs in the
    # design-tool backend's request threads (scripts/slider_backend.py).
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _sigterm_handler)

    if value_normalizer is not None:
        amp = AMPState(device, False)

    while remainingTrials.nonzero().numel() > 0:
        with torch.no_grad():
            if policy is not None:
                action_dists, values, cur_rnn_states = policy(cur_rnn_states, *obs)
            else:
                with profile("Multi-Policy Inference"):
                    action_dists = None
                    for i, p in enumerate(policies):
                        a_dists, _, cur_rnn_states = p(cur_rnn_states, *obs)

                        if action_dists is None:
                            action_dists = a_dists
                        else:
                            for j, d in enumerate(action_dists.dists):
                                action_dists.dists[j].probs[i * num_worlds : (i+1)*num_worlds, ...] = \
                                a_dists.dists[j].probs[i * num_worlds : (i+1)*num_worlds, ...]
                                action_dists.dists[j].logits[i * num_worlds : (i+1)*num_worlds, ...] = \
                                a_dists.dists[j].logits[i * num_worlds : (i+1)*num_worlds, ...]

            
            log_probs_out = torch.zeros_like(actions).float()

            if use_best_action:
                action_dists.best(actions)
            else:
                action_dists.sample(actions, log_probs_out)

        if record_log:
            ckpts.cpu().numpy().tofile(record_log)

        # Apply the action transform (limitation profile) before stepping.
        with torch.no_grad():
            actions[:] = action_transform.apply(actions).to(device)

        sim.step()
        total_steps += 1
        print(f"Total steps: {total_steps}")
        
        # Reset any delay buffers for worlds that are done
        action_transform.reset_delay(dones)



        # A world succeeds when its reward reaches the solve reward (depends on the reward mode).
        # Worlds with no trials left do not count.
        rewards[..., 0] *= torch.clamp(remainingTrials, 0, 1)
        official_dones = dones.clone()
        official_dones[..., 0] *= torch.clamp(remainingTrials, 0, 1)

        # success_counts has one counter per equal-sized group of worlds.
        assert rewards.shape[0] % len(success_counts) == 0, "Number of success counts must evenly divide number of worlds"
        worlds_per_success_count = rewards.shape[0] // len(success_counts)

        success_per_world += torch.where(torch.clamp(rewards[:, ..., 0] - 9.9999, 0.0, None) > 0, 1, 0)
        for i in range(len(success_counts)):
            success_counts[i] += torch.clamp(rewards[i * worlds_per_success_count : (i + 1) * worlds_per_success_count, ..., 0] - 9.9999, 0.0, None).nonzero().numel()

        if success_counts[0] > _prev_solve_count:
            cap = max_solves if max_solves is not None else 5
            for n in range(_prev_solve_count + 1, min(success_counts[0], cap) + 1):
                print(f"[SOLVE] n={n} step={total_steps} time={time.time() - start:.2f}s epoch={time.time():.3f}")
            _prev_solve_count = success_counts[0]

        if max_solves is not None and success_counts[0] >= max_solves:
            print(f"[EARLY STOP] reached {success_counts[0]} solves (max_solves={max_solves})")
            break

        remainingTrials -= dones[..., 0]
        remainingTrials = torch.clamp(remainingTrials, 0, None)
        

        # Report values without normalization.
        if value_normalizer is not None:
            values = value_normalizer.invert(amp, values)

        if store_positions:
            all_positions.append(agent_pos_tensor.clone())
            all_jumps.append(agent_jump_performed_tensor.clone())
            all_dones.append(official_dones.clone())
            all_rewards.append(rewards.clone())
            all_timesteps.append(timesteps.clone())
            all_values.append(values.clone())
            all_move_dists.append(action_dists.dists[0].probs.clone())
            all_direction_dists.append(action_dists.dists[1].probs.clone())
            all_rotate_dists.append(action_dists.dists[2].probs.clone())
            all_jump_dists.append(action_dists.dists[3].logits.clone())
            

    if logging:
        _print_stats()

    if store_positions:
        assert len(success_counts) == 1, "store_positions with multiple success counts is not implemented."

        def permuteAndNumpy(x):
            x = torch.stack(x, dim=0)
            x = x.permute(1, 0, 2)
            return x.cpu().numpy()

        all_positions = permuteAndNumpy(all_positions)
        all_jumps = permuteAndNumpy(all_jumps)
        all_dones = permuteAndNumpy(all_dones)
        all_rewards = permuteAndNumpy(all_rewards)
        all_timesteps = permuteAndNumpy(all_timesteps)
        all_values = permuteAndNumpy(all_values)

        all_move_dists = permuteAndNumpy(all_move_dists)
        all_direction_dists = permuteAndNumpy(all_direction_dists)
        all_rotate_dists = permuteAndNumpy(all_rotate_dists)
        all_jump_dists = permuteAndNumpy(all_jump_dists)

        all_positions = np.concatenate(
            [all_rewards, 
            all_positions,
            all_jumps, 
            all_dones, 
            all_timesteps,
            all_values,
            all_move_dists,
            all_direction_dists,
            all_rotate_dists,
            all_jump_dists],
            axis=-1
        )
        infer_data['all_positions'] = all_positions

    if record_log:
        record_log.close()

    infer_data['success_per_world'] = success_per_world.cpu().numpy()
    if len(success_counts) == 1:
        infer_data['success_per_policy'] = success_counts[0]
    else:
        infer_data['success_per_policy'] = success_counts
    return infer_data


def make_json_serializable(d):
    for k, v in d.items():
        if isinstance(v, type):
            d[k] = str(v)
        elif isinstance(v, Part):
            d[k] = str(v)
    return d


def main_infer(args):
    device = torch.device(f'cuda:{args.gpu_id}') if args.gpu_sim else torch.device('cpu')


    sim = setup_sim(
        args.num_worlds,
        args.episode_len,
        args.gpu_sim,
        args.gpu_id,
        rand_seed=args.seed,  
        use_complex_level=args.use_complex_level,
        use_normaabb_obs=args.use_normaabb_obs,
        unlock_agent_orientation=args.unlock_agent_orientation,
        disable_step_obs=args.disable_step_obs,
        disable_phys_obs=args.disable_phys_obs,
        disable_lidar=args.disable_lidar
    )
    
    sim.init()

    transform_params=TransformParameters(
        num_worlds=args.num_worlds, 
        device=device
    )
    transform_params.from_params(
        p_drop=args.p_drop,
        p_random=args.p_random,
        p_repeat=args.p_repeat,
        delay_buffer_size=args.delay_buffer_size,
        rate_frequency=args.rate_frequency,
        rate_frequency_std=args.rate_frequency_std,
        no_jump=args.no_jump,
        noise_mean=args.noise_mean,
        noise_std=args.noise_std
    )

    # Create levels
    level_generation_params = TrainParameters1(level_seed=args.seed)
    level_selector_rng = np.random.default_rng(seed=args.seed)

    json_levels = sim.json_level_descriptions_tensor().to_torch()
    json_indices = sim.json_index_tensor().to_torch()
    if args.level_file:
        levels_json = list(loadJsonDescriptionsFromDisk(args.level_file).values())
        levels_json = [levels_json[i % len(levels_json)] for i in range(1024)]
        levels_json = jsonDescriptionsToTensor(sim.json_level_descriptions_tensor().to_torch().shape, levels_json)
    else:
        json_levels_parts = generateJsonLevels(
            json_levels.shape[0],
            args.seed,
            args.level_types,
            level_selector_rng,
            args.num_components,
            level_generation_params,
            Constant(0) if args.disable_random_orientation else None
        )
        if args.random_spawn:
            json_levels_parts = [json.loads(json_level_parts) for json_level_parts in json_levels_parts]
            randomize_spawn_blocks(json_levels_parts, level_selector_rng)
            json_levels_parts = [json.dumps(json_level_parts) for json_level_parts in json_levels_parts]
        if args.waypoint:
            json_levels_parts = [json.loads(json_level_parts) for json_level_parts in json_levels_parts]
            add_randomized_waypoints(json_levels_parts, level_selector_rng)
            json_levels_parts = [json.dumps(json_level_parts) for json_level_parts in json_levels_parts]
        
        # Select specific spawn block if index is provided (overrides random spawn)
        if args.spawn_block_index is not None:
            print(f"Selecting specific spawn block index: {args.spawn_block_index}")
            json_levels_parts = [json.loads(json_level_parts) for json_level_parts in json_levels_parts]
            actual_spawn_index = select_specific_spawn_block(json_levels_parts, args.spawn_block_index)
            if actual_spawn_index != args.spawn_block_index:
                print(f"  Note: Actual spawn block index used: {actual_spawn_index} (clamped from {args.spawn_block_index})")
            json_levels_parts = [json.dumps(json_level_parts) for json_level_parts in json_levels_parts]
        
        # Apply physical perturbations based on --perturb argument
        if args.perturb is not None:
            json_levels_parts = [json.loads(json_level_parts) for json_level_parts in json_levels_parts]
            perturbation_seed = args.perturbation_seed if args.perturbation_seed is not None else args.seed
            rng = random.Random(perturbation_seed)
            
            if args.perturb == "all":
                print("Applying perturbations to all blocks")
                for json_level in json_levels_parts:
                    perturb_all_part_positions(json_level, perturbation_seed)
            elif args.perturb == "random":
                print("Applying single random block perturbation")
                for json_level in json_levels_parts:
                    # Select a random block index for each level
                    all_blocks = list(json_level.keys())
                    if all_blocks:
                        random_block_index = rng.randint(0, len(all_blocks) - 1)
                        actual_perturb_index = select_specific_perturbation_block([json_level], random_block_index, perturbation_seed)
                        print(f"  Perturbed block {actual_perturb_index}")
            elif args.perturb.isdigit():
                block_index = int(args.perturb)
                print(f"Applying perturbation to block {block_index}")
                actual_perturb_index = select_specific_perturbation_block(json_levels_parts, block_index, perturbation_seed)
                if actual_perturb_index != block_index:
                    print(f"  Note: Actual block index perturbed: {actual_perturb_index} (clamped from {block_index})")
            else:
                print(f"Warning: Unknown perturbation argument '{args.perturb}'. Applying all perturbations.")
                for json_level in json_levels_parts:
                    perturb_all_part_positions(json_level, perturbation_seed)
                    
            json_levels_parts = [json.dumps(json_level) for json_level in json_levels_parts]
        
        levels_json = jsonDescriptionsToTensor(json_levels.shape, json_levels_parts).to(device)
    json_levels[:] = levels_json.to(device)

    if args.write_binary_out:
        print("Writing level to: ", f'{args.write_binary_out}.out')
        with open(f'{args.write_binary_out}.out', 'wb') as f:
            f.write(levels_json.cpu().numpy().tobytes())
    
    # Assign each world a level.
    if args.single_instance:
        # Force all worlds to use the first level description.
        print("Inferring on a single level instance.")
        json_indices[:, 0] = 0
    else:
        for i in range(args.num_worlds):
            json_indices[i, 0] = i % json_levels.shape[0]

    obs, num_obs_features = setup_obs(sim, sim.enum_counts_tensor().to_torch().to(device), args.no_level_obs)

    policy = make_policy(num_obs_features, None, args.num_channels, args.separate_value, ACTION_DIMS)
    if args.ckpt_path:
        weights = LearningState.load_policy_weights(args.ckpt_path)
        key_load = policy.load_state_dict(weights, strict=False)
        if key_load.missing_keys or key_load.unexpected_keys:
            raise RuntimeError(f"policy checkpoint {args.ckpt_path} does not match the model: "
                               f"missing={key_load.missing_keys} unexpected={key_load.unexpected_keys}")
    policy = policy.to(torch.device(f"cuda:{args.gpu_id}")) if args.gpu_sim else policy.to(torch.device('cpu'))
    
    infer_data = run_infer(
        sim=sim,
        transform_params=transform_params,
        policies=policy,
        obs=obs,
        num_worlds=args.num_worlds,
        num_trials=args.num_trials,
        episode_len=args.episode_len,
        success_counts=None,
        pre_coordinate_transform_func=None,
        use_complex_level=args.use_complex_level,
        use_normaabb_obs=args.use_normaabb_obs,
        unlock_agent_orientation=args.unlock_agent_orientation,
        store_positions=args.store_positions,
        use_best_action=args.use_best_action,
        record_log=args.record_log,
        device=device,
        logging=True,
        single_instance=args.single_instance,
        max_solves=args.max_solves,
    )

    return


world_simulator = None
loaded_policies = []
sem = threading.Semaphore()
@app.route('/measureDifficulty', methods=['POST'])
def measureDifficulty():
    args = arg_parser.parse_args()

    global world_simulator
    section_and_policy = request.get_json()
    obbyJsonStr = str(section_and_policy["obbyJson"]).replace("'", "\"").replace("False", "false").replace("True", "true")

    args.level_str = obbyJsonStr

    # Checkpoints for the difficulty-measurement route, keyed by p_random.
    demo_checkpoints = {
        0.0 : "ckpts/DemoObbyUnlimited_seed13/102400.pth"
    }

    # One request at a time.
    sem.acquire()

    if not world_simulator:
        # The server creates the simulator once, on the first request.
        world_simulator = madrona_puzzle_bench.SimManager(
            exec_mode = madrona_puzzle_bench.madrona.ExecMode.CUDA if args.gpu_sim else madrona_puzzle_bench.madrona.ExecMode.CPU,
            gpu_id = args.gpu_id,
            num_worlds = args.num_worlds * len(demo_checkpoints.keys()),
            rand_seed = args.seed,
            sim_flags = int(madrona_puzzle_bench.SimFlags.Default),
            reward_mode = getattr(RewardMode, "Dense1"),
            episode_len = 3000,
            reward_per_dist = 0.05,
            levels_per_episode = 1,
            button_width = 1.3 * 1.0,
            door_width = 20.0 / 3.,
            slack_reward = -0.005,
        )


    if args.gpu_sim:
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')

    # Load each policy once.
    if len(loaded_policies) == 0:
        _, num_obs_features = setup_obs(world_simulator, world_simulator.enum_counts_tensor().to_torch().to(device), args.no_level_obs)
        policy = make_policy(num_obs_features, None, args.num_channels, args.separate_value, ACTION_DIMS)
        for policy_path in sorted(demo_checkpoints.values()):
            print(policy_path)
            weights = LearningState.load_policy_weights(policy_path)
            policy.load_state_dict(weights, strict=False)
            policy = policy.to(torch.device(f"cuda:{args.gpu_id}")) if args.gpu_sim else policy.to(torch.device('cpu'))
            loaded_policies.append(policy)

    success_rate_str = ""

    transform_params = TransformParameters(num_worlds=args.num_worlds * len(demo_checkpoints.keys()), device=device)
    transform_params.from_params(
        p_drop = args.p_drop,
        p_random = args.p_random,
        p_repeat = args.p_repeat,
        delay_buffer_size = args.delay_buffer_size,
        rate_frequency = args.rate_frequency,
        rate_frequency_std = args.rate_frequency_std,
        no_jump = args.no_jump,
        noise_mean = args.noise_mean,
        noise_std = args.noise_std
    )
        


    successes_per_policy, total_trials, levels_info, levels_json = \
        run_infer(args, logging=True, level_types=args.level_types, transform_params=transform_params,
        world_simulator=world_simulator, policies=loaded_policies)
    
    for i, success in enumerate(successes_per_policy):
        ckpt = list(demo_checkpoints.values())[i]
        args.ckpt_path = ckpt
        success_rate_str += "\n" + ckpt[ckpt.index("/") + 1:ckpt.rindex("/")] + f" Succeeded in {success} / {total_trials}"
    
    sem.release()
    return success_rate_str

@app.route('/measureDifficultyMultisample', methods=['POST'])
def measureDifficultyMultisample():
    global world_simulator
    section_and_policy = request.get_json()
    obbyJsonStr = str(section_and_policy["obbyJson"]).replace("'", "\"").replace("False", "false").replace("True", "true")

    args.level_str = obbyJsonStr

    # Checkpoints for the difficulty-measurement route, keyed by p_random.
    demo_checkpoints = {
        0.0 : "ckpts/DemoObbyUnlimited_seed13/102400.pth"
    }

    # Initialize the success counts for 1 default and 6 transformations.
    success_counts = [0 for i in range(7)]


    total_worlds = args.num_worlds * len(demo_checkpoints.keys()) * len(success_counts)

    # One request at a time.
    sem.acquire()

    if not world_simulator:
        # The server creates the simulator once, on the first request.
        world_simulator = madrona_puzzle_bench.SimManager(
            exec_mode = madrona_puzzle_bench.madrona.ExecMode.CUDA if args.gpu_sim else madrona_puzzle_bench.madrona.ExecMode.CPU,
            gpu_id = args.gpu_id,
            num_worlds = total_worlds,
            rand_seed = args.seed,
            sim_flags = int(madrona_puzzle_bench.SimFlags.Default),
            reward_mode = getattr(RewardMode, "Dense1"),
            episode_len = 3000,
            reward_per_dist = 0.05,
            levels_per_episode = 1,
            button_width = 1.3 * 1.0,
            door_width = 20.0 / 3.,
            slack_reward = -0.005,
        )


    if args.gpu_sim:
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')

    # Load each policy once.
    if len(loaded_policies) == 0:
        _, num_obs_features = setup_obs(world_simulator, world_simulator.enum_counts_tensor().to_torch().to(device), args.no_level_obs)
        policy = make_policy(num_obs_features, None, args.num_channels, args.separate_value, ACTION_DIMS)
        for policy_path in sorted(demo_checkpoints.values()):
            print(policy_path)
            weights = LearningState.load_policy_weights(policy_path)
            policy.load_state_dict(weights, strict=False)
            policy = policy.to(torch.device(f"cuda:{args.gpu_id}")) if args.gpu_sim else policy.to(torch.device('cpu'))
            loaded_policies.append(policy)

    success_rate_str = ""

    transform_params = TransformParameters(num_worlds=total_worlds, device=device)
    transform_params.from_params(
        p_drop = args.p_drop,
        p_random = args.p_random,
        p_repeat = args.p_repeat,
        delay_buffer_size = args.delay_buffer_size,
        rate_frequency = args.rate_frequency,
        rate_frequency_std = args.rate_frequency_std,
        no_jump = args.no_jump,
        noise_mean = args.noise_mean,
        noise_std = args.noise_std
    )

    # Applied to every object right before the conversion to Madrona coordinates.
    # Each block of worlds gets one of 7 transforms (none, then +-x, +-y, +-z),
    # as the Roblox tool does.
    def pre_coordinate_transform(obj_desc, json_description_idx):
        transforms = (
            (0,0,0),
            (1,0,0),
            (-1,0,0),
            (0,1,0),
            (0,-1,0),
            (0,0,1),
            (0,0,-1)
        )
        world_idx = min(json_description_idx, total_worlds - 1)
        transform_idx = world_idx // (total_worlds // len(transforms))
        for behavior_idx, behavior_dict in enumerate(obj_desc["behaviors"]):
            if behavior_dict["name"] == "TransformPosition":
                obj_desc["position"] = [x + 5 * t for x, t in zip(obj_desc["position"], transforms[transform_idx])]
            if behavior_dict["name"] == "TransformOrientation":
                obj_desc["orientation"] = [x + 90 * t for x, t in zip(obj_desc["orientation"], transforms[transform_idx])]
            if behavior_dict["name"] == "TransformSize":
                obj_desc["size"] = [x + 5 * abs(t) for x, t in zip(obj_desc["orientation"], transforms[transform_idx])]
                
                # Match the behavior of the Roblox tool.
                phi, theta, psi = obj_desc["orientation"]
                rot_mat = np.array(
                    [[math.cos(theta)*math.cos(psi), -math.cos(phi)*math.sin(psi) + math.sin(phi)*math.sin(theta)*math.cos(psi), math.sin(phi)*math.sin(psi)+math.cos(phi)*math.sin(theta)*math.cos(psi)], 
                    [math.cos(theta)*math.sin(psi), math.cos(phi)*math.cos(psi) + math.sin(phi)*math.sin(theta)*math.sin(psi), -math.sin(phi)*math.cos(psi) + math.cos(phi)*math.sin(theta)*math.sin(psi)], 
                    [-math.sin(theta), math.sin(phi)*math.cos(theta), math.cos(phi)*math.cos(theta)]], dtype=np.float32)
                oriented_pos = rot_mat.dot(transforms[transform_idx])
                obj_desc["position"] = [x + 2.5 * t for x, t in zip(obj_desc["position"], oriented_pos)]



    successes_per_pertubation, total_trials, levels_info, levels_json = \
        run_infer(
            args, logging=False, level_types=args.level_types, transform_params=transform_params,
            world_simulator=world_simulator, policies=loaded_policies, success_counts=success_counts, pre_coordinate_transform_func=pre_coordinate_transform
        )
    
    for i, success in enumerate(successes_per_pertubation):
        success_rate_str += "\n" + f" Succeeded in {success} / {total_trials}"
    sem.release()
    return successes_per_pertubation

if __name__ == "__main__":
    args = arg_parser.parse_args()

    main_infer(arg_parser.parse_args())
