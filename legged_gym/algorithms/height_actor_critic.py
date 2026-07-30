import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic import ActorCritic, get_activation


class ActorCriticHeightEncoder(ActorCritic):
    """ ActorCritic whose height-scan encoder is a real nn.Module submodule instead of a fixed,
        untrained MLP living in the env's observation pipeline. Raw observations are split into
        [proprio | height_scan]; the height scan is compressed through an MLP encoder into a
        small latent, and [proprio | latent] is what the actor/critic MLPs actually see. Since
        the encoder is registered on this module, it's included in actor_critic.parameters() and
        gets trained by the normal PPO gradient step in rsl_rl.algorithms.PPO.update().
    """
    is_recurrent = False

    def __init__(self, num_proprio_obs, num_height_points, num_actions,
                 height_hidden_dims=[128, 64], height_latent_dim=32, **kwargs):
        self.num_proprio_obs = num_proprio_obs
        self.num_height_points = num_height_points

        # actor/critic MLPs are sized on [proprio, latent] - build them via the base class first
        super().__init__(num_proprio_obs + height_latent_dim, num_proprio_obs + height_latent_dim,
                          num_actions, **kwargs)

        # only safe to register submodules after nn.Module.__init__() has run (inside super().__init__())
        activation = get_activation(kwargs.get('activation', 'elu'))
        dims = [num_height_points] + list(height_hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), activation]
        layers.append(nn.Linear(dims[-1], height_latent_dim))
        self.height_encoder = nn.Sequential(*layers)

    def _encode(self, observations):
        proprio = observations[:, :self.num_proprio_obs]
        height_scan = observations[:, self.num_proprio_obs:]
        latent = self.height_encoder(height_scan)
        return torch.cat((proprio, latent), dim=-1)

    def update_distribution(self, observations):
        # NOTE: don't also override act() to pre-encode and call super().act() - ActorCritic.act()
        # calls self.update_distribution(observations), which already dispatches here polymorphically;
        # encoding again in an act() override would double-encode (and feed the wrong slice width).
        super().update_distribution(self._encode(observations))

    def act_inference(self, observations):
        return super().act_inference(self._encode(observations))

    def evaluate(self, critic_observations, **kwargs):
        return super().evaluate(self._encode(critic_observations))
