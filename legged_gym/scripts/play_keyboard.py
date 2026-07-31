import os
import sys
from collections import deque

import isaacgym
from isaacgym import gymapi
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, get_load_path, task_registry

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# vx/vy/vyaw step applied on every key press, and the camera offset (world
# frame, relative to the robot) used while the follow-above camera is active.
VX_STEP = 0.1
VY_STEP = 0.1
VYAW_STEP = 0.1
CAMERA_OFFSET = np.array([-1.0, 0.0, 1.5])  # behind (-x) and above (+z) the robot

# live vx/vyaw command-vs-actual strip chart (scrolling window, like an ECG trace)
PLOT_WINDOW_S = 5.0
PLOT_VX_RANGE = 1.5
PLOT_VYAW_RANGE = 1.0

KEY_ACTIONS = {
    gymapi.KEY_W: "vx_plus",
    gymapi.KEY_S: "vx_minus",
    gymapi.KEY_A: "vy_plus",
    gymapi.KEY_D: "vy_minus",
    gymapi.KEY_LEFT_BRACKET: "vyaw_plus",
    gymapi.KEY_RIGHT_BRACKET: "vyaw_minus",
    gymapi.KEY_C: "toggle_follow_camera",
    gymapi.KEY_R: "reset",
}

DEPTH_WINDOW_NAME = "go2_stairs depth camera"


def _depth_to_gray(depth, near, far):
    """ Raw metric depth (meters) -> a grayscale uint8 image for cv2.imshow. """
    depth = np.nan_to_num(depth, posinf=far, neginf=near)
    depth = np.clip(depth, near, far)
    return (255 * (1 - (depth - near) / (far - near))).astype(np.uint8)  # near = bright


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
    # don't auto-reset the robot just because the nominal training episode length elapsed -
    # only reset on an actual failure (fall / bad contact), so an interactive run isn't cut
    # off mid-climb
    env_cfg.env.episode_length_s = 1e9
    if args.task == "go2_stairs":
        # single continuously-ascending U-shaped staircase, one robot, no curriculum grid
        env_cfg.terrain.u_shape_playground = True
    if hasattr(env_cfg, "camera"):
        env_cfg.camera.use_camera = args.use_camera

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # e.g. "Jul30_05-16-51_/model_800.pt" -> "Jul30_05-16-51_/800", for the plot window title
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
    run_name = os.path.basename(os.path.dirname(resume_path))
    checkpoint = os.path.splitext(os.path.basename(resume_path))[0].rsplit('_', 1)[-1]
    run_label = f"{run_name}/{checkpoint}"

    has_viewer = env.viewer is not None
    if has_viewer:
        for key, action in KEY_ACTIONS.items():
            env.gym.subscribe_viewer_keyboard_event(env.viewer, key, action)
        print("Keyboard control: w/s = vx, a/d = vy, [/] = vyaw, c = toggle follow-above camera, r = reset")

    # only set if the env actually created a camera (cfg has `camera` and use_camera was True)
    use_camera = getattr(env, "camera_handle", None) is not None
    if use_camera:
        cam_cfg = env_cfg.camera
        camera_step_interval = max(round(1.0 / (cam_cfg.fps * env.dt)), 1)
        cv2.namedWindow(DEPTH_WINDOW_NAME, cv2.WINDOW_NORMAL)

    commands = torch.zeros(3, dtype=torch.float, device=env.device)
    follow_camera = True

    # live strip chart: vx/vyaw command vs. actual, scrolling over the last PLOT_WINDOW_S seconds
    plot_len = max(int(PLOT_WINDOW_S / env.dt), 1)
    t_buf = deque(maxlen=plot_len)
    vx_cmd_buf = deque(maxlen=plot_len)
    vx_real_buf = deque(maxlen=plot_len)
    vyaw_cmd_buf = deque(maxlen=plot_len)
    vyaw_real_buf = deque(maxlen=plot_len)

    plt.ion()
    fig, (ax_vx, ax_vyaw) = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True)
    fig.canvas.manager.set_window_title(run_label)
    line_vx_cmd, = ax_vx.plot([], [], color='tab:orange', label='vx cmd')
    line_vx_real, = ax_vx.plot([], [], color='tab:blue', label='vx real')
    ax_vx.set_ylim(-PLOT_VX_RANGE, PLOT_VX_RANGE)
    ax_vx.set_ylabel('vx [m/s]')
    ax_vx.legend(loc='upper right')
    ax_vx.grid(True)

    line_vyaw_cmd, = ax_vyaw.plot([], [], color='tab:orange', label='vyaw cmd')
    line_vyaw_real, = ax_vyaw.plot([], [], color='tab:blue', label='vyaw real')
    ax_vyaw.set_ylim(-PLOT_VYAW_RANGE, PLOT_VYAW_RANGE)
    ax_vyaw.set_ylabel('vyaw [rad/s]')
    ax_vyaw.set_xlabel('time [s]')
    ax_vyaw.legend(loc='upper right')
    ax_vyaw.grid(True)
    fig.tight_layout()

    sim_time = 0.0

    try:
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
                        # leave the camera wherever the last follow update put it - just stop
                        # driving it every frame, instead of snapping back to the default pose
                        follow_camera = not follow_camera
                    elif evt.action == "reset":
                        env.reset_idx(torch.arange(env.num_envs, device=env.device))
                        commands[:] = 0.

                env.commands[:, 0] = commands[0]
                env.commands[:, 1] = commands[1]
                env.commands[:, 2] = commands[2]

            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())

            if has_viewer and follow_camera:
                robot_pos = env.root_states[0, :3].cpu().numpy()
                env.set_camera(robot_pos + CAMERA_OFFSET, robot_pos)

            if use_camera and i % camera_step_interval == 0:
                depth_img = env.get_camera_depth_image()
                cv2.imshow(DEPTH_WINDOW_NAME, _depth_to_gray(depth_img, cam_cfg.near_plane, cam_cfg.far_plane))
                cv2.waitKey(1)

            sim_time += env.dt
            t_buf.append(sim_time)
            vx_cmd_buf.append(commands[0].item())
            vx_real_buf.append(env.base_lin_vel[0, 0].item())
            vyaw_cmd_buf.append(commands[2].item())
            vyaw_real_buf.append(env.base_ang_vel[0, 2].item())

            if plt.fignum_exists(fig.number):
                line_vx_cmd.set_data(t_buf, vx_cmd_buf)
                line_vx_real.set_data(t_buf, vx_real_buf)
                line_vyaw_cmd.set_data(t_buf, vyaw_cmd_buf)
                line_vyaw_real.set_data(t_buf, vyaw_real_buf)
                ax_vx.set_xlim(sim_time - PLOT_WINDOW_S, sim_time)
                ax_vyaw.set_xlim(sim_time - PLOT_WINDOW_S, sim_time)
                plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        # Ctrl+C during a blocked C-extension call (cv2's GUI event loop, matplotlib's backend,
        # IsaacGym's viewer/physx) can otherwise take several presses to actually exit - clean up
        # what we can, then hard-exit immediately instead of waiting on normal interpreter shutdown.
        if use_camera:
            cv2.destroyAllWindows()
        plt.close(fig)
        os._exit(0)


if __name__ == '__main__':
    args = get_args()
    play(args)
