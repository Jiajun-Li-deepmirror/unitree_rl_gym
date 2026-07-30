import os
import sys

import isaacgym
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import numpy as np
import torch

# vx/vy/vyaw step applied on every key press, and the camera offset (world
# frame, relative to the robot) used while the follow-above camera is active.
VX_STEP = 0.1
VY_STEP = 0.1
VYAW_STEP = 0.1
CAMERA_OFFSET = np.array([-1.0, 0.0, 1.5])  # behind (-x) and above (+z) the robot

KEY_ACTIONS = {
    gymapi.KEY_W: "vx_plus",
    gymapi.KEY_S: "vx_minus",
    gymapi.KEY_A: "vy_plus",
    gymapi.KEY_D: "vy_minus",
    gymapi.KEY_LEFT_BRACKET: "vyaw_plus",
    gymapi.KEY_RIGHT_BRACKET: "vyaw_minus",
    gymapi.KEY_C: "toggle_follow_camera",
}


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    # commands are driven from the keyboard instead of being randomly resampled
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 1e9

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    has_viewer = env.viewer is not None
    if has_viewer:
        for key, action in KEY_ACTIONS.items():
            env.gym.subscribe_viewer_keyboard_event(env.viewer, key, action)
        print("Keyboard control: w/s = vx, a/d = vy, [/] = vyaw, c = toggle follow-above camera")

    commands = torch.zeros(3, dtype=torch.float, device=env.device)
    follow_camera = False

    for i in range(10*int(env.max_episode_length)):
        if has_viewer:
            if env.gym.query_viewer_has_closed(env.viewer):
                break
            # this drains the viewer's whole event queue, so replicate the
            # built-in QUIT / toggle_viewer_sync handling normally done inside
            # env.render() (it would otherwise see an already-empty queue)
            for evt in env.gym.query_viewer_action_events(env.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    env.enable_viewer_sync = not env.enable_viewer_sync
                if evt.value <= 0:
                    continue
                if evt.action == "vx_plus":
                    commands[0] = min(commands[0].item() + VX_STEP, env.command_ranges["lin_vel_x"][1])
                elif evt.action == "vx_minus":
                    commands[0] = max(commands[0].item() - VX_STEP, env.command_ranges["lin_vel_x"][0])
                elif evt.action == "vy_plus":
                    commands[1] = min(commands[1].item() + VY_STEP, env.command_ranges["lin_vel_y"][1])
                elif evt.action == "vy_minus":
                    commands[1] = max(commands[1].item() - VY_STEP, env.command_ranges["lin_vel_y"][0])
                elif evt.action == "vyaw_plus":
                    commands[2] = min(commands[2].item() + VYAW_STEP, env.command_ranges["ang_vel_yaw"][1])
                elif evt.action == "vyaw_minus":
                    commands[2] = max(commands[2].item() - VYAW_STEP, env.command_ranges["ang_vel_yaw"][0])
                elif evt.action == "toggle_follow_camera":
                    follow_camera = not follow_camera
                    if not follow_camera:
                        env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

            env.commands[:, 0] = commands[0]
            env.commands[:, 1] = commands[1]
            env.commands[:, 2] = commands[2]

        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        if has_viewer and follow_camera:
            robot_pos = env.root_states[0, :3].cpu().numpy()
            env.set_camera(robot_pos + CAMERA_OFFSET, robot_pos)


if __name__ == '__main__':
    args = get_args()
    play(args)
