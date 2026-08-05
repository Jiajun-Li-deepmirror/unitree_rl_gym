import os
from collections import deque
from datetime import datetime

import isaacgym
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.algorithms.student_policy import StudentDepthPolicy

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def train_student(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    student_cfg = env_cfg.student

    # small num_envs - per-env camera rendering is expensive (see student_cfg.num_envs's comment)
    env_cfg.env.num_envs = args.num_envs if args.num_envs is not None else student_cfg.num_envs
    env_cfg.camera.use_camera = True
    env_cfg.camera.scale = student_cfg.camera_scale
    env_cfg.env.test = True

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # load the frozen teacher exactly like play.py does
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    teacher_policy = ppo_runner.get_inference_policy(device=env.device)
    for p in ppo_runner.alg.actor_critic.parameters():
        p.requires_grad_(False)

    num_proprio_obs = env.num_obs - env.num_height_points
    cam_cfg = env_cfg.camera
    input_height = max(1, round(cam_cfg.height * cam_cfg.scale))
    input_width = max(1, round(cam_cfg.width * cam_cfg.scale))

    student = StudentDepthPolicy(
        num_proprio_obs=num_proprio_obs,
        num_actions=env.num_actions,
        input_height=input_height,
        input_width=input_width,
        cnn_channels=student_cfg.cnn_channels,
        cnn_feature_dim=student_cfg.cnn_feature_dim,
        gru_hidden_dim=student_cfg.gru_hidden_dim,
        actor_hidden_dims=student_cfg.actor_hidden_dims,
    ).to(env.device)
    optimizer = torch.optim.Adam(student.parameters(), lr=student_cfg.learning_rate)

    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', student_cfg.experiment_name,
                            datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
    print(f"Distilling student -> {log_dir}")

    obs = env.get_observations()
    hidden = student.init_hidden(env.num_envs, env.device)
    cur_episode_length = torch.zeros(env.num_envs, device=env.device)
    episode_lengths = deque(maxlen=100)

    chunk_len = student_cfg.chunk_len
    for it in range(student_cfg.num_iterations + 1):
        if it < student_cfg.lr_warmup_iters:
            lr = student_cfg.learning_rate * (it + 1) / student_cfg.lr_warmup_iters
            for group in optimizer.param_groups:
                group['lr'] = lr

        optimizer.zero_grad()
        chunk_loss = torch.zeros((), device=env.device)
        for _ in range(chunk_len):
            depth = env.get_camera_depth_images()
            proprio = obs[:, :num_proprio_obs]
            student_action, hidden = student.forward_step(
                depth, proprio, hidden, cam_cfg.near_plane, cam_cfg.far_plane)
            with torch.no_grad():
                teacher_action = teacher_policy(obs)
            chunk_loss = chunk_loss + F.mse_loss(student_action, teacher_action)

            obs, _, rews, dones, infos = env.step(student_action.detach())
            # zero GRU hidden state for envs that just reset - obs is already the fresh
            # post-reset state (reset_idx runs before compute_observations), so this must
            # happen now, before the next iteration of this loop feeds that obs in
            hidden = student.reset_hidden(hidden, dones)

            cur_episode_length += 1
            done_ids = dones.nonzero(as_tuple=False).flatten()
            if len(done_ids) > 0:
                episode_lengths.extend(cur_episode_length[done_ids].cpu().tolist())
                cur_episode_length[done_ids] = 0.

        chunk_loss = chunk_loss / chunk_len
        chunk_loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), student_cfg.max_grad_norm)
        optimizer.step()
        hidden = hidden.detach()  # bounds backprop to this chunk (truncated BPTT)

        writer.add_scalar('Loss/action_mse', chunk_loss.item(), it)
        if episode_lengths:
            writer.add_scalar('Train/mean_episode_length', sum(episode_lengths) / len(episode_lengths), it)
        if it % 10 == 0:
            mean_len = sum(episode_lengths) / len(episode_lengths) if episode_lengths else float('nan')
            print(f"[iter {it}/{student_cfg.num_iterations}] action_mse={chunk_loss.item():.5f} "
                  f"mean_ep_len={mean_len:.1f}")

        if it % student_cfg.save_interval == 0:
            torch.save({
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iter': it,
            }, os.path.join(log_dir, f'model_{it}.pt'))


if __name__ == '__main__':
    args = get_args()
    train_student(args)
