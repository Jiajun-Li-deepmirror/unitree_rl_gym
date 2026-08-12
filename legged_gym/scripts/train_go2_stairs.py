"""Training entry point for the go2_stairs task.

go2_stairs is deliberately NOT registered in legged_gym/envs/__init__.py (the user's
explicit choice, to keep this task self-contained in legged_gym/envs/go2_stairs/ without
touching any existing file) -- so the vanilla legged_gym/scripts/train.py can't reach it:
task_registry.make_env(name=...) only knows names that got registered there.

This script gets the same effect a different way: it instantiates GO2StairsCfg/
GO2StairsRobot directly (exactly like view_go2_stairs_terrain.py already does for
inspection) and hands the resulting env straight to task_registry.make_alg_runner(),
which accepts an already-built env and a train_cfg without needing a registered name.

On top of that, it drives OnPolicyRunner.learn() in chunks (aligned to
train_cfg.runner.save_interval) instead of one single call for the whole run, purely so it
can poll env.get_and_reset_episode_report() between chunks and persist a detailed,
per-(terrain_type, difficulty_row) breakdown that rsl_rl's own logging never sees --
this is the log to read first when trying to figure out what's actually happening deep
into an unattended run on the training machine (which terrain/difficulty is regressing,
whether the curriculum is actually advancing envs past the easiest row, etc).

Usage:
    python train_go2_stairs.py --headless --num_envs 4096
    python train_go2_stairs.py --headless --num_envs 4096 --max_iterations 3000 --resume
"""
import json
import os
import time

import isaacgym  # noqa: F401  (must be imported before torch)
from isaacgym import gymapi

from legged_gym.envs.go2_stairs.go2_stairs_config import GO2StairsCfg, GO2StairsCfgPPO
from legged_gym.envs.go2_stairs.go2_stairs_env import GO2StairsRobot
from legged_gym.utils import class_to_dict, get_args, set_seed, task_registry
from legged_gym.utils.helpers import parse_sim_params

import torch  # noqa: F401  (imported after isaacgym, matches the rest of this codebase)

# one robot per tile in the dev viewer; real training runs on far more -- see
# GO2StairsCfg.env.num_envs's own comment. Override with --num_envs to change.
DEFAULT_TRAIN_NUM_ENVS = 4096


def _write_report(log_dir, writer, iteration, tot_timesteps, report):
    """Appends one JSON line to <log_dir>/episode_report.jsonl (the detailed record meant
    to be read directly, e.g. `jq` or a small pandas script, when diagnosing a run after
    the fact) and mirrors the same numbers into TensorBoard under ByTerrain/... tags (for
    live monitoring alongside the rest of rsl_rl's own scalars). Deliberately two sinks,
    not one: the jsonl file is complete/structured and survives independent of TensorBoard
    ever being opened; TensorBoard is just a convenient live view of the same data.
    """
    record = {
        'iteration': iteration,
        'tot_timesteps': tot_timesteps,
        'wall_time': time.time(),
        **report,
    }
    with open(os.path.join(log_dir, 'episode_report.jsonl'), 'a') as f:
        f.write(json.dumps(record) + '\n')

    if writer is None:
        return
    for terrain_name, rows in report['by_terrain'].items():
        for row, stats in rows.items():
            prefix = f'ByTerrain/{terrain_name}/row{row}'
            writer.add_scalar(f'{prefix}/episodes', stats['episodes'], iteration)
            writer.add_scalar(f'{prefix}/mean_episode_length_s', stats['mean_episode_length_s'], iteration)
            for name, rate in stats['outcome_rate'].items():
                writer.add_scalar(f'{prefix}/outcome_rate/{name}', rate, iteration)
            for name, val in stats['mean_reward'].items():
                writer.add_scalar(f'{prefix}/mean_reward/{name}', val, iteration)
    for terrain_name, rows in report['env_distribution'].items():
        for row, count in rows.items():
            writer.add_scalar(f'EnvDistribution/{terrain_name}/row{row}', count, iteration)


def train(args):
    env_cfg = GO2StairsCfg()
    train_cfg = GO2StairsCfgPPO()

    env_cfg.env.num_envs = args.num_envs if args.num_envs is not None else DEFAULT_TRAIN_NUM_ENVS

    set_seed(args.seed if args.seed is not None else train_cfg.seed)
    sim_params = parse_sim_params(args, {"sim": class_to_dict(env_cfg.sim)})
    env = GO2StairsRobot(cfg=env_cfg, sim_params=sim_params, physics_engine=args.physics_engine,
                          sim_device=args.sim_device, headless=args.headless)

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, train_cfg=train_cfg, args=args)

    log_interval = train_cfg.runner.save_interval
    total_iterations = train_cfg.runner.max_iterations
    done_iterations = 0
    init_at_random_ep_len = True
    while done_iterations < total_iterations:
        chunk = min(log_interval, total_iterations - done_iterations)
        ppo_runner.learn(num_learning_iterations=chunk, init_at_random_ep_len=init_at_random_ep_len)
        init_at_random_ep_len = False  # only randomize episode phase once, at the very start
        done_iterations += chunk
        if ppo_runner.log_dir is not None:
            report = env.get_and_reset_episode_report()
            _write_report(ppo_runner.log_dir, ppo_runner.writer, done_iterations, ppo_runner.tot_timesteps, report)


if __name__ == '__main__':
    args = get_args()
    train(args)
