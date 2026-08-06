import os
from collections import deque

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .height_encoder_runner import HeightEncoderOnPolicyRunner
from .student_policy import StudentDepthPolicy
from .student_distillation import StudentDistillation


class StudentDistillationRunner(HeightEncoderOnPolicyRunner):
    """ Distills a depth-camera student policy from the frozen teacher. Subclasses
        HeightEncoderOnPolicyRunner (not the bare rsl_rl OnPolicyRunner) because the go2_stairs
        teacher this student imitates is an ActorCriticHeightEncoder: HeightEncoderOnPolicyRunner's
        __init__ contains its own eval(policy_class_name) call in *its own module's* namespace,
        which is where ActorCriticHeightEncoder is actually importable from - eval() resolves
        against the globals of the module the call is textually written in, not the caller's, so
        subclassing the bare OnPolicyRunner here would NameError trying to resolve it.
        super().__init__() builds/holds the teacher (actor_critic + PPO) exactly as before, so
        `train_cfg.runner.resume=True` + task_registry.make_alg_runner's runner.load(resume_path)
        keeps loading the teacher checkpoint unmodified - then this class bolts the student
        network and its own algorithm object on top. rsl_rl itself is never edited: this class and
        StudentDistillation (the student-side analogue of rsl_rl.algorithms.PPO) both live in
        legged_gym/algorithms/, following the precedent set by
        ActorCriticHeightEncoder/HeightEncoderOnPolicyRunner.
    """

    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        super().__init__(env, train_cfg, log_dir, device)

        # the teacher is a fixed distillation target - never trained further
        for p in self.alg.actor_critic.parameters():
            p.requires_grad_(False)
        self.teacher_policy = self.get_inference_policy(device=self.device)

        student_cfg = env.cfg.student
        cam_cfg = env.cfg.camera
        self.num_proprio_obs = env.num_obs - env.num_height_points
        input_height = max(1, round(cam_cfg.height * cam_cfg.scale))
        input_width = max(1, round(cam_cfg.width * cam_cfg.scale))

        student = StudentDepthPolicy(
            num_proprio_obs=self.num_proprio_obs,
            num_actions=env.num_actions,
            input_height=input_height,
            input_width=input_width,
            cnn_channels=student_cfg.cnn_channels,
            cnn_feature_dim=student_cfg.cnn_feature_dim,
            gru_hidden_dim=student_cfg.gru_hidden_dim,
            actor_hidden_dims=student_cfg.actor_hidden_dims,
        ).to(self.device)
        self.student_alg = StudentDistillation(
            student,
            learning_rate=student_cfg.learning_rate,
            max_grad_norm=student_cfg.max_grad_norm,
            device=self.device,
        )

        self.chunk_len = student_cfg.chunk_len
        self.base_lr = student_cfg.learning_rate
        self.lr_warmup_iters = student_cfg.lr_warmup_iters
        self.save_interval = student_cfg.save_interval
        self.near_plane = cam_cfg.near_plane
        self.far_plane = cam_cfg.far_plane

    def learn_distill(self, num_learning_iterations):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        obs = self.env.get_observations()
        self.student_alg.init_hidden(self.env.num_envs)
        cur_episode_length = torch.zeros(self.env.num_envs, device=self.device)
        episode_lengths = deque(maxlen=100)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            if it < self.lr_warmup_iters:
                self.student_alg.set_learning_rate(self.base_lr * (it + 1) / self.lr_warmup_iters)

            chunk_loss = torch.zeros((), device=self.device)
            for _ in range(self.chunk_len):
                depth = self.env.get_camera_depth_images()
                proprio = obs[:, :self.num_proprio_obs]
                student_action = self.student_alg.act(depth, proprio, self.near_plane, self.far_plane)
                with torch.no_grad():
                    teacher_action = self.teacher_policy(obs)
                chunk_loss = chunk_loss + F.mse_loss(student_action, teacher_action)

                obs, _, rews, dones, infos = self.env.step(student_action.detach())
                # zero GRU hidden state for envs that just reset - obs is already the fresh
                # post-reset state (reset_idx runs before compute_observations), so this must
                # happen now, before the next iteration of this loop feeds that obs in
                self.student_alg.reset_hidden(dones)

                cur_episode_length += 1
                done_ids = dones.nonzero(as_tuple=False).flatten()
                if len(done_ids) > 0:
                    episode_lengths.extend(cur_episode_length[done_ids].cpu().tolist())
                    cur_episode_length[done_ids] = 0.

            chunk_loss = chunk_loss / self.chunk_len
            self.student_alg.update(chunk_loss)

            if self.writer is not None:
                self.writer.add_scalar('Loss/action_mse', chunk_loss.item(), it)
                if episode_lengths:
                    self.writer.add_scalar('Train/mean_episode_length',
                                            sum(episode_lengths) / len(episode_lengths), it)
            if it % 10 == 0:
                mean_len = sum(episode_lengths) / len(episode_lengths) if episode_lengths else float('nan')
                print(f"[iter {it}/{tot_iter}] action_mse={chunk_loss.item():.5f} mean_ep_len={mean_len:.1f}")

            if it % self.save_interval == 0:
                self.save_student(os.path.join(self.log_dir, f'model_{it}.pt'))

        self.current_learning_iteration += num_learning_iterations
        self.save_student(os.path.join(self.log_dir, f'model_{self.current_learning_iteration}.pt'))

    def save_student(self, path):
        # deliberately not named save()/load() - those are inherited unchanged from
        # OnPolicyRunner and target the teacher (self.alg.actor_critic); keeping the names
        # distinct avoids ambiguity about which policy a given checkpoint call touches
        torch.save({
            'model_state_dict': self.student_alg.student.state_dict(),
            'optimizer_state_dict': self.student_alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
        }, path)
