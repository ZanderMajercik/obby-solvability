"""
Fine-tune a PPO checkpoint on one query level (RL-FT).
Stops when the maximum episode return reaches --solve-threshold (default 10).
scripts/bash/run_rl_finetune.sh runs it over the query set.

Usage:
  python scripts/finetune_ppo.py \
    --query-path query_data/Designer1/Query1.json \
    --ckpt-path ckpts/policy.pth \
    --num-worlds 1024 \
    --num-updates 200 \
    --gpu-sim
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import madrona_puzzle_bench
from madrona_puzzle_bench import SimFlags, RewardMode
from madrona_puzzle_bench_learn import (
    train, TrainConfig, PPOConfig, SimInterface, LearningState,
)

sys.path.insert(0, str(Path(__file__).parent))
from generateLevels import jsonDescriptionsToTensor
from parameters.params_transforms import TransformParameters, generateLimitationStepFunc
from policy import make_policy, setup_obs


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--query-path', type=str, required=True)
arg_parser.add_argument('--ckpt-path', type=str, required=True)
arg_parser.add_argument('--num-worlds', type=int, default=1024)
arg_parser.add_argument('--num-updates', type=int, default=500)
arg_parser.add_argument('--steps-per-update', type=int, default=40)
arg_parser.add_argument('--lr', type=float, default=1e-4)
arg_parser.add_argument('--gamma', type=float, default=0.998)
arg_parser.add_argument('--entropy-coef', type=float, default=0.01)
arg_parser.add_argument('--solve-threshold', type=float, default=10.0,
                        help='reward_max value at or above which we mark as solved.')
arg_parser.add_argument('--break-when-solved', action='store_true',
                        help='Stop training as soon as reward_max >= solve-threshold.')
arg_parser.add_argument('--gpu-sim', action='store_true')
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--disable-step-obs', action='store_true',
                        help='Zero step-count obs.')
arg_parser.add_argument('--disable-phys-obs', action='store_true',
                        help='Zero entity-physics obs.')
arg_parser.add_argument('--disable-lidar', action='store_true',
                        help='Zero the 128-ray lidar.')
arg_parser.add_argument('--seed', type=int, default=5)
arg_parser.add_argument('--log-every', type=int, default=10)
arg_parser.add_argument('--save-dir', type=str, default=None,
                        help='If set, save finetuned checkpoint here on solve or at end.')
arg_parser.add_argument('--result-path', type=str, default=None,
                        help='If set, write a JSON result file with solve_update, time_sec, solved.')


class FinetuneCallback:
    def __init__(self, num_worlds, steps_per_update, solve_threshold, break_when_solved, log_every, save_dir, policy):
        self.num_worlds = num_worlds
        self.steps_per_update = steps_per_update
        self.solve_threshold = solve_threshold
        self.log_every = log_every
        self.save_dir = save_dir
        self.policy = policy
        self.break_when_solved = break_when_solved
        self.solved = False
        self.solve_update = None
        self.solve_time_sec = None
        self.best_return = -float('inf')
        self.updates_done = 0
        self.start_time = time.time()
        # The training loop reads these attributes.
        self.world_weights = None
        self.logit_scales = None
        self.sil_weights = None

    def __call__(self, update_idx, update_time, update_results, learning_state):
        update_id = update_idx + 1
        self.updates_done = update_id

        with torch.no_grad():
            reward_max = update_results.rewards.max().cpu().item()
            reward_mean = update_results.rewards.mean().cpu().item()

            done_mask = (update_results.dones == 1.0)
            return_max = 0.0
            return_mean = 0.0
            if done_mask.sum() > 0:
                returns = update_results.returns[done_mask]
                return_max = returns.max().cpu().item()
                return_mean = returns.mean().cpu().item()

        self.best_return = max(self.best_return, reward_max)

        if update_id % self.log_every == 0 or update_id == 1:
            elapsed = time.time() - self.start_time
            fps = self.num_worlds * self.steps_per_update / update_time
            print(f"[update {update_id:4d}] {elapsed:6.1f}s  "
                  f"reward_max={reward_max:.3f}  return_max={return_max:.3f}  "
                  f"return_mean={return_mean:.3f}  fps={fps:.0f}")

        if reward_max >= self.solve_threshold and not self.solved:
            self.solved = True
            self.solve_update = update_id
            elapsed = time.time() - self.start_time
            self.solve_time_sec = elapsed
            print(f"\n*** SOLVED at update {update_id} (reward_max={reward_max:.3f} > {self.solve_threshold}) "
                  f"after {elapsed:.1f}s ***\n")
            if self.save_dir:
                Path(self.save_dir).mkdir(parents=True, exist_ok=True)
                save_path = Path(self.save_dir) / f"finetuned_solved_{update_id}.pth"
                learning_state.save(update_idx, save_path)
                print(f"Saved checkpoint to {save_path}")
            if self.break_when_solved:
                return False

        return True


def main():
    args = arg_parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dev = torch.device(f'cuda:{args.gpu_id}') if (args.gpu_sim and torch.cuda.is_available()) else torch.device('cpu')

    query_path = Path(args.query_path)
    with open(query_path) as f:
        query = json.load(f)
    level_json = query['level_json']
    level_json_str = json.dumps(level_json)

    print(f"Query: {query_path.name}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Worlds: {args.num_worlds}  Updates: {args.num_updates}  Solve threshold: {args.solve_threshold}")

    # The observation flags must match the checkpoint's training config.
    sim_flags = SimFlags.Default
    if args.disable_phys_obs:
        sim_flags |= SimFlags.DisablePhysObs
    if args.disable_step_obs:
        sim_flags |= SimFlags.DisableStepObs
    if args.disable_lidar:
        sim_flags |= SimFlags.DisableLidar

    sim = madrona_puzzle_bench.SimManager(
        exec_mode=madrona_puzzle_bench.madrona.ExecMode.CUDA if args.gpu_sim else madrona_puzzle_bench.madrona.ExecMode.CPU,
        gpu_id=args.gpu_id,
        num_worlds=args.num_worlds,
        rand_seed=args.seed,
        sim_flags=int(sim_flags),
        reward_mode=RewardMode.Dense1,
        episode_len=3000,
        levels_per_episode=1,
        button_width=1.3,
        door_width=20.0 / 3.0,
        reward_per_dist=0.05,
        slack_reward=-0.005,
    )
    sim.init()

    # Load the one level into every world.
    json_desc_tensor = sim.json_level_descriptions_tensor().to_torch()
    json_indices = sim.json_index_tensor().to_torch()
    # Fill all descriptor slots; every world points at slot 0.
    level_jsons = [level_json_str] * json_desc_tensor.shape[0]
    json_desc_tensor[:] = jsonDescriptionsToTensor(json_desc_tensor.shape, level_jsons, json_levels_physics=None).to(dev)
    json_indices[..., 0] = 0

    enum_counts = sim.enum_counts_tensor().to_torch().to(dev)
    obs, num_obs_features = setup_obs(sim, enum_counts)

    actions_num_buckets = [2, 8, 3, 2]
    policy = make_policy(num_obs_features, None, 256, False, actions_num_buckets)

    weights = LearningState.load_policy_weights(args.ckpt_path)
    key_load = policy.load_state_dict(weights, strict=False)
    if key_load.missing_keys or key_load.unexpected_keys:
        raise RuntimeError(f"policy checkpoint {args.ckpt_path} does not match the model: "
                           f"missing={key_load.missing_keys} unexpected={key_load.unexpected_keys}")
    policy = policy.to(dev)
    print(f"Loaded checkpoint: {args.ckpt_path}")

    # Identity transforms: no noise and no action delay.
    transform_params = TransformParameters.default_params(args.num_worlds, dev)

    # Reset the sim. checkpoint_resets stays zero, so agents always restart from the spawn.
    checkpoint_resets = sim.checkpoint_reset_tensor().to_torch()
    checkpoint_resets.fill_(0.0)
    resets = sim.reset_tensor().to_torch()
    resets.fill_(1.0)
    sim.step()
    resets.fill_(0.0)
    checkpoint_resets.fill_(0.0)

    _base_step_fn = generateLimitationStepFunc(
        sim, obs,
        transform_params.get_action_transform(),
        transform_params.get_obs_transform(),
        dev,
    )

    def step_fn():
        checkpoint_resets.fill_(0.0)  # always reset to the spawn, never to a checkpoint
        _base_step_fn()

    cb = FinetuneCallback(
        num_worlds=args.num_worlds,
        steps_per_update=args.steps_per_update,
        solve_threshold=args.solve_threshold,
        break_when_solved=args.break_when_solved,
        log_every=args.log_every,
        save_dir=args.save_dir,
        policy=policy,
    )

    def _write_result(result):
        # Write atomically (temp file + rename) so that a SIGTERM during the write cannot
        # leave a truncated result.json. -inf (no update finished) is not valid JSON, so write null.
        if result["best_reward_max"] == -float('inf'):
            result["best_reward_max"] = None
        path = Path(args.result_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, 'w') as f:
            json.dump(result, f, indent=2)
        os.replace(tmp, path)

    # On SIGTERM (e.g. from timeout), write the result before Python exits.
    def _sigterm_handler(signum, frame):
        elapsed = time.time() - cb.start_time
        result = {
            "query": str(args.query_path),
            "ckpt": str(args.ckpt_path),
            "solved": cb.solved,
            "solve_update": cb.solve_update,
            "solve_time_sec": cb.solve_time_sec,
            "best_reward_max": cb.best_return,
            "total_updates": cb.updates_done,
            "total_time_sec": round(elapsed, 2),
            "timed_out": True,
        }
        if args.result_path and not Path(args.result_path).exists():
            _write_result(result)
        sys.exit(0)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    print(f"\nStarting fine-tuning...\n")
    train(
        dev,
        SimInterface(
            step=step_fn,
            obs=obs,
            actions=sim.action_tensor().to_torch(),
            dones=sim.done_tensor().to_torch(),
            rewards=sim.reward_tensor().to_torch(),
            resets=sim.reset_tensor().to_torch(),
            checkpoints=sim.checkpoint_tensor().to_torch(),
            checkpoint_resets=sim.checkpoint_reset_tensor().to_torch(),
        ),
        TrainConfig(
            run_name='finetune',
            num_updates=args.num_updates,
            steps_per_update=args.steps_per_update,
            num_bptt_chunks=1,
            lr=args.lr,
            gamma=args.gamma,
            gae_lambda=0.95,
            ppo=PPOConfig(
                num_mini_batches=1,
                clip_coef=0.2,
                value_loss_coef=0.5,
                entropy_coef=args.entropy_coef,
                max_grad_norm=0.5,
                num_epochs=2,
                use_intrinsic_loss=False,
                value_loss_intrinsic_coef=0.5,
                intrinsic_loss_coef=0.5,
            ),
            mixed_precision=False,
            normalize_advantages=True,
            normalize_values=True,
        ),
        policy,
        cb,
        None,
    )

    elapsed = time.time() - cb.start_time
    if cb.solved:
        print(f"\nResult: SOLVED at update {cb.solve_update} ({cb.solve_time_sec:.2f}s)")
    else:
        print(f"\nResult: NOT SOLVED after {cb.updates_done} updates ({elapsed:.1f}s) — best reward_max={cb.best_return:.3f}")

    result = {
        "query": str(args.query_path),
        "ckpt": str(args.ckpt_path),
        "solved": cb.solved,
        "solve_update": cb.solve_update,
        "solve_time_sec": cb.solve_time_sec,
        "best_reward_max": cb.best_return,
        "total_updates": cb.updates_done,
        "total_time_sec": round(elapsed, 2),
    }
    if args.result_path:
        _write_result(result)
        print(f"Result saved to {args.result_path}")


if __name__ == '__main__':
    main()
