# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Same as play.py, but with a single manually-controlled robot instead of a
# scripted command sequence: WASD drives vx/vy, [ and ] drive yaw rate, and
# the viewer camera chases the robot from behind and above instead of the
# scripted fly-by. Needs a viewer -- do not pass --headless.

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

from isaacgym import gymapi
from isaacgym.torch_utils import quat_apply

import numpy as np
import torch


# action name -> (gymapi key, which command axis it drives, sign)
KEY_BINDINGS = {
    "vx+": (gymapi.KEY_W, "lin_vel_x", +1),
    "vx-": (gymapi.KEY_S, "lin_vel_x", -1),
    "vy+": (gymapi.KEY_A, "lin_vel_y", +1),
    "vy-": (gymapi.KEY_D, "lin_vel_y", -1),
    "yaw+": (gymapi.KEY_LEFT_BRACKET, "ang_vel_yaw", +1),
    "yaw-": (gymapi.KEY_RIGHT_BRACKET, "ang_vel_yaw", -1),
}

CHASE_DISTANCE = 1.5  # m, behind the robot
CHASE_HEIGHT = 0.6    # m, above the robot


def update_chase_camera(env):
    """Keep the viewer camera behind and above the (single) robot, tracking
    its current position and heading every frame."""
    base_pos = env.root_states[0, :3].cpu().numpy()
    forward = quat_apply(env.base_quat[0:1], env.forward_vec[0:1])[0].cpu().numpy()
    cam_pos = base_pos - forward * CHASE_DISTANCE + np.array([0.0, 0.0, CHASE_HEIGHT])
    cam_target = base_pos + np.array([0.0, 0.0, 0.2])
    env.set_camera(cam_pos, cam_target)


def compute_keyboard_commands(env, key_state):
    """Turn the current held-key state into a (vx, vy, yaw) command. Holding a
    key commands the actual trained extreme for that axis/direction
    (env.command_ranges' min or max), not an arbitrary fixed speed -- e.g.
    go2_stairs trains lin_vel_y at exactly [0,0], so A/D won't do anything
    there; that's the trained policy's real capability, not a bug in this
    script. Opposing keys on the same axis held together cancel to 0."""
    direction = {"lin_vel_x": 0, "lin_vel_y": 0, "ang_vel_yaw": 0}
    for action, (_, axis, sign) in KEY_BINDINGS.items():
        if key_state.get(action, False):
            direction[axis] += sign
    cmd = {}
    for axis, d in direction.items():
        lo, hi = env.command_ranges[axis]
        target = hi if d > 0 else (lo if d < 0 else 0.0)
        cmd[axis] = float(np.clip(target, lo, hi))
    return cmd["lin_vel_x"], cmd["lin_vel_y"], cmd["ang_vel_yaw"]


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # single, manually-driven robot
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    # keyboard drives commands directly every frame: turn off heading-based
    # auto yaw (would fight the yaw keys) and push resampling far enough out
    # that the periodic random resample never fires during a play session.
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 1e9

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    if env.viewer is None:
        raise RuntimeError("play_keyboard.py needs a viewer for keyboard input -- don't pass --headless")

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    for action_name, (key, _, _) in KEY_BINDINGS.items():
        env.gym.subscribe_viewer_keyboard_event(env.viewer, key, action_name)
    key_state = {action_name: False for action_name in KEY_BINDINGS}

    print("Keyboard control: W/S = vx, A/D = vy, [ / ] = yaw rate. ESC or close window to quit.")
    print(f"Trained command ranges -- lin_vel_x={env.command_ranges['lin_vel_x']} "
          f"lin_vel_y={env.command_ranges['lin_vel_y']} ang_vel_yaw={env.command_ranges['ang_vel_yaw']}")

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    img_idx = 0

    for i in range(10*int(env.max_episode_length)):
        for evt in env.gym.query_viewer_action_events(env.viewer):
            if evt.action in key_state:
                key_state[evt.action] = evt.value > 0

        vx, vy, yaw = compute_keyboard_commands(env, key_state)
        env.commands[0, 0] = vx
        env.commands[0, 1] = vy
        env.commands[0, 2] = yaw

        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        update_chase_camera(env)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    args = get_args()
    play(args)
