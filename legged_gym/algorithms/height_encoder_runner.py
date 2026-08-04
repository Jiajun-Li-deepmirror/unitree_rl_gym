from rsl_rl.algorithms import PPO  # noqa: F401, resolved by eval() below
from rsl_rl.runners import OnPolicyRunner

from .height_actor_critic import ActorCriticHeightEncoder  # noqa: F401, resolved by eval() below


class HeightEncoderOnPolicyRunner(OnPolicyRunner):
    """ Same as OnPolicyRunner, except it builds the actor-critic from the env's proprio/height-scan
        split (env.num_obs - env.num_height_points) instead of a single flat num_obs, since
        ActorCriticHeightEncoder needs to know where to split the raw observation to run its
        (trainable) height-scan encoder as part of the network's forward pass. Everything else
        (rollout, PPO update, logging, save/load) is inherited unchanged from OnPolicyRunner.
    """

    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        assert env.num_privileged_obs is None, \
            "HeightEncoderOnPolicyRunner assumes the actor and critic share the same (proprio + height-scan) observations"

        num_proprio_obs = env.num_obs - env.num_height_points
        height_cfg = env.cfg.height_encoder
        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCriticHeightEncoder
        actor_critic = actor_critic_class(
            num_proprio_obs, env.num_height_points, env.num_actions,
            height_hidden_dims=height_cfg.hidden_dims, height_latent_dim=height_cfg.latent_dim,
            **self.policy_cfg,
        ).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env,
                               [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()
