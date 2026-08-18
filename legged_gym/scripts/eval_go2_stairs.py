"""Evaluation entry point for the go2_stairs task.

Loads a checkpoint saved by train_go2_stairs.py and runs it deterministically (no
exploration noise, no domain randomization/observation noise) with
terrain.curriculum=False so envs spread across EVERY difficulty row from the start (see
GO2StairsRobot._get_env_origins) instead of only whatever row training's curriculum has
reached so far -- an evaluation is supposed to answer "how good is this checkpoint at
every terrain/difficulty", not just the ones training happened to be practicing recently.

Runs until every (terrain_type, difficulty_row) bucket has accumulated at least
--min_episodes_per_bucket episodes (falling back to --max_steps if some bucket is simply
too rare to hit that, e.g. a difficulty row on a terrain column with few assigned envs),
then writes a JSON report next to the loaded checkpoint and prints a summary table.

Usage:
    python eval_go2_stairs.py --num_envs 1000
    python eval_go2_stairs.py --num_envs 1000 --load_run Aug12_10-00-00_ --checkpoint 1500
"""
import json
import os
import time

import isaacgym  # noqa: F401  (must be imported before torch)
from isaacgym import gymapi

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.go2_stairs.go2_stairs_config import GO2StairsCfg, GO2StairsCfgPPO
from legged_gym.envs.go2_stairs.go2_stairs_env import GO2StairsRobot
from legged_gym.utils import class_to_dict, get_args, get_load_path, set_seed, task_registry
from legged_gym.utils.helpers import parse_sim_params

import torch  # noqa: F401  (imported after isaacgym, matches the rest of this codebase)

DEFAULT_EVAL_NUM_ENVS = 1000
DEFAULT_MIN_EPISODES_PER_BUCKET = 30
DEFAULT_MAX_STEPS = 20000  # hard stop even if some rare bucket never reaches the target


def evaluate(args, min_episodes_per_bucket=DEFAULT_MIN_EPISODES_PER_BUCKET, max_steps=DEFAULT_MAX_STEPS):
    env_cfg = GO2StairsCfg()
    train_cfg = GO2StairsCfgPPO()

    env_cfg.env.num_envs = args.num_envs if args.num_envs is not None else DEFAULT_EVAL_NUM_ENVS
    env_cfg.terrain.curriculum = False  # spread envs across every difficulty row, not just what training reached
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    set_seed(args.seed if args.seed is not None else train_cfg.seed)
    sim_params = parse_sim_params(args, {"sim": class_to_dict(env_cfg.sim)})
    env = GO2StairsRobot(cfg=env_cfg, sim_params=sim_params, physics_engine=args.physics_engine,
                          sim_device=args.sim_device, headless=args.headless)

    if args.max_iterations is not None:
        train_cfg.runner.max_iterations = args.max_iterations
    if args.load_run is not None:
        train_cfg.runner.load_run = args.load_run
    if args.checkpoint is not None:
        train_cfg.runner.checkpoint = args.checkpoint
    root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    resume_path = get_load_path(root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
    print(f"[eval_go2_stairs] Loading checkpoint: {resume_path}")

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, train_cfg=train_cfg, args=args)
    policy = ppo_runner.get_inference_policy(device=env.device)

    obs = env.get_observations()
    num_buckets = env_cfg.terrain.num_cols * env_cfg.terrain.num_rows

    # A freshly-created sim reliably reports a spurious contact-force spike on its very
    # first physics step (a known PhysX actor-creation artifact, not a real event -- every
    # env resets simultaneously with episode_length=0 right as the actors settle into their
    # just-created poses). Over a full training run this is a single negligible blip lost
    # among tens of thousands of iterations, but an eval loop is short and stops as soon as
    # its episode-count target is hit -- with --num_envs envs all resetting at once, that
    # one artifact alone can satisfy the whole per-bucket episode quota before any policy
    # actually gets to demonstrate anything, silently producing a report that's entirely
    # (or mostly) this one degenerate zero-length burst. Step past it and throw away
    # whatever accumulated during warmup before the real measurement loop starts.
    warmup_steps = 10
    for _ in range(warmup_steps):
        with torch.inference_mode():
            actions = policy(obs.detach())
        obs, _, _, _, _ = env.step(actions.detach())
    env.get_and_reset_episode_report()  # discard warmup-only data (the artifact + any luck)

    # Also require at least one full max_episode_length window of REAL steps before
    # stopping, even if the per-bucket episode quota is technically hit sooner -- otherwise
    # a burst of short/failed episodes early on can satisfy the quota while genuine
    # full-length successful episodes (which take up to max_episode_length steps to
    # complete) never get the chance to occur at all, biasing the report toward failures.
    min_steps = int(env.max_episode_length)
    step = 0
    while step < max_steps:
        with torch.inference_mode():
            actions = policy(obs.detach())
        obs, _, _, _, _ = env.step(actions.detach())
        step += 1
        if step % 500 == 0:
            counts = env.bucket_ep_count.cpu()
            covered = int((counts >= min_episodes_per_bucket).sum().item())
            print(f"[eval_go2_stairs] step {step}: {covered}/{num_buckets} buckets have "
                  f">= {min_episodes_per_bucket} episodes so far")
            if covered == num_buckets and step >= min_steps:
                break

    report = env.get_and_reset_episode_report()
    report['checkpoint'] = resume_path
    report['eval_steps'] = step
    report['wall_time'] = time.time()

    out_dir = os.path.dirname(resume_path)
    # deliberately avoids the substring "model" anywhere in this filename: get_load_path's
    # checkpoint=-1 resolution does `if 'model' in file` with no further filtering, so an
    # eval report sitting in the same run directory as the real model_*.pt checkpoints can
    # get misidentified as one and passed straight to torch.load -- happened once already.
    ckpt_num = os.path.basename(resume_path).replace('model_', '').replace('.pt', '')
    out_path = os.path.join(out_dir, f"evalreport_ckpt{ckpt_num}.json")
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"[eval_go2_stairs] Wrote detailed report to {out_path}")

    print("\n[eval_go2_stairs] Summary (terrain_type / row / episodes / mean_ep_len_s / outcome rates):")
    for terrain_name, rows in report['by_terrain'].items():
        for row, stats in sorted(rows.items()):
            rates = ", ".join(f"{k}={v:.2f}" for k, v in stats['outcome_rate'].items() if v > 0)
            print(f"  {terrain_name:14s} row{row}  n={stats['episodes']:5d}  "
                  f"len={stats['mean_episode_length_s']:6.2f}s  {rates}")


if __name__ == '__main__':
    args = get_args()
    evaluate(args)
