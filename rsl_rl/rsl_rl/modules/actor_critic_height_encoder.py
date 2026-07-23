import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.actor_critic import ActorCritic, get_activation


class ActorCriticHeightEncoder(ActorCritic):
    """ActorCritic variant that encodes the privileged height-scan segment of the
    observation into a fixed-size latent before it reaches the actor/critic MLPs,
    instead of feeding the raw height-scan vector directly.

    The incoming observation is assumed to be [proprioception (num_prop_obs) ++
    height-scan (remainder)], which is exactly how LeggedRobot.compute_observations
    concatenates its obs_buf. The height-scan latent size (`latent_dim`) is the
    teacher-side half of the teacher/student distillation: a depth+GRU student
    encoder built later must output a latent of this same size so it can be
    swapped in for `height_encoder` without touching `actor`/`critic`.
    """
    is_recurrent = False

    def __init__(self,
                 num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 num_prop_obs,
                 encoder_hidden_dims=[128, 64],
                 latent_dim=32,
                 actor_hidden_dims=[512, 256, 128],
                 critic_hidden_dims=[512, 256, 128],
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):
        self.num_prop_obs = num_prop_obs
        self.latent_dim = latent_dim
        num_height_obs_actor = num_actor_obs - num_prop_obs
        num_height_obs_critic = num_critic_obs - num_prop_obs

        # actor/critic MLPs consume [proprioception ++ latent], not the raw obs
        super().__init__(num_prop_obs + latent_dim,
                          num_prop_obs + latent_dim,
                          num_actions,
                          actor_hidden_dims=actor_hidden_dims,
                          critic_hidden_dims=critic_hidden_dims,
                          activation=activation,
                          init_noise_std=init_noise_std,
                          **kwargs)

        act = get_activation(activation)
        self.actor_height_encoder = self._build_encoder(num_height_obs_actor, encoder_hidden_dims, latent_dim, act)
        self.critic_height_encoder = self._build_encoder(num_height_obs_critic, encoder_hidden_dims, latent_dim, act)

        print(f"Actor height encoder: {self.actor_height_encoder}")
        print(f"Critic height encoder: {self.critic_height_encoder}")

    @staticmethod
    def _build_encoder(input_dim, hidden_dims, latent_dim, activation):
        layers = [nn.Linear(input_dim, hidden_dims[0]), activation]
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[l], latent_dim))
            else:
                layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                layers.append(activation)
        return nn.Sequential(*layers)

    def _split(self, observations):
        prop = observations[..., :self.num_prop_obs]
        height = observations[..., self.num_prop_obs:]
        return prop, height

    def update_distribution(self, observations):
        prop, height = self._split(observations)
        latent = self.actor_height_encoder(height)
        mean = self.actor(torch.cat((prop, latent), dim=-1))
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        prop, height = self._split(observations)
        latent = self.actor_height_encoder(height)
        return self.actor(torch.cat((prop, latent), dim=-1))

    def evaluate(self, critic_observations, **kwargs):
        prop, height = self._split(critic_observations)
        latent = self.critic_height_encoder(height)
        return self.critic(torch.cat((prop, latent), dim=-1))
