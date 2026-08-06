import os
from datetime import datetime

import isaacgym
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def train_student(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    student_cfg = env_cfg.student

    # small num_envs - per-env camera rendering is expensive (see student_cfg.num_envs's comment)
    env_cfg.env.num_envs = args.num_envs if args.num_envs is not None else student_cfg.num_envs
    env_cfg.camera.use_camera = True
    env_cfg.camera.scale = student_cfg.camera_scale
    env_cfg.env.test = True

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # load the frozen teacher exactly like play.py does; StudentDistillationRunner (registered
    # in legged_gym.algorithms) builds the teacher the same way OnPolicyRunner would - swapping
    # in the runner class here is what wires up the student on top of it
    train_cfg.runner.resume = True
    train_cfg.runner_class_name = 'StudentDistillationRunner'
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)

    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', student_cfg.experiment_name,
                            datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
    os.makedirs(log_dir, exist_ok=True)
    # make_alg_runner also created a log_dir, but under train_cfg.runner.experiment_name (the
    # teacher's) since that's also where it looks for the resume checkpoint - point the runner at
    # the student's own experiment_name/log_dir instead, before its writer gets lazily created
    ppo_runner.log_dir = log_dir
    print(f"Distilling student -> {log_dir}")

    ppo_runner.learn_distill(num_learning_iterations=student_cfg.num_iterations)


if __name__ == '__main__':
    args = get_args()
    train_student(args)
