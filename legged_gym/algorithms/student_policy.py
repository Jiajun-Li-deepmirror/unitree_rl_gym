import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic import get_activation


class StudentDepthPolicy(nn.Module):
    """ Depth-camera + GRU student policy, distilled from the privileged height-scan teacher
        (ActorCriticHeightEncoder) via DAgger - see legged_gym/scripts/train_student.py.

        Plain nn.Module, not an ActorCritic/ActorCriticRecurrent subclass: there's no reward,
        return, or advantage here, just a deterministic action regressed against the teacher's
        mean action at every visited state, so no critic is needed. The GRU is a bare nn.GRUCell
        driven one step at a time by the training script (not rsl_rl's Memory class, whose
        batched/masked mode assumes PPO-rollout-shaped, episode-aligned padding that a
        student-driven DAgger rollout - which can reset mid chunk - doesn't have).
    """

    def __init__(self, num_proprio_obs, num_actions, input_height, input_width,
                 cnn_channels=[16, 32, 32], cnn_feature_dim=128, gru_hidden_dim=128,
                 actor_hidden_dims=[256, 256], activation='elu', **kwargs):
        super().__init__()
        act = get_activation(activation)

        # strided-conv depth encoder; first layer uses a wider kernel since the input is a
        # single low-res depth frame, not a natural RGB image
        conv_layers = []
        in_channels = 1
        for i, out_channels in enumerate(cnn_channels):
            kernel_size = 5 if i == 0 else 3
            conv_layers += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2), act]
            in_channels = out_channels
        self.cnn_convs = nn.Sequential(*conv_layers)

        # probe the flattened conv output size once, from the actual (height, width) the camera
        # will produce, so cnn_head is fully built (and in the optimizer's param list) up front -
        # no lazy/first-forward layer creation that could silently miss the optimizer
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_height, input_width)
            flat_dim = self.cnn_convs(dummy).flatten(start_dim=1).shape[1]
        self.cnn_head = nn.Linear(flat_dim, cnn_feature_dim)

        self.gru_hidden_dim = gru_hidden_dim
        self.gru = nn.GRUCell(cnn_feature_dim + num_proprio_obs, gru_hidden_dim)

        dims = [gru_hidden_dim] + list(actor_hidden_dims)
        actor_layers = []
        for i in range(len(dims) - 1):
            actor_layers += [nn.Linear(dims[i], dims[i + 1]), act]
        actor_layers.append(nn.Linear(dims[-1], num_actions))
        self.actor = nn.Sequential(*actor_layers)

    def _encode_depth(self, depth_img, near, far):
        # depth_img: (N, H, W) positive meters, as returned by GO2Stairs.get_camera_depth_images
        depth = torch.clamp(depth_img, near, far)
        depth = (depth - near) / max(far - near, 1e-6)  # -> [0, 1]
        features = self.cnn_convs(depth.unsqueeze(1))
        return self.cnn_head(features.flatten(start_dim=1))

    def forward_step(self, depth_img, proprio, hidden, near, far):
        """ One env step. depth_img (N,H,W), proprio (N, num_proprio_obs), hidden
            (N, gru_hidden_dim). Returns (action, new_hidden).
        """
        depth_feature = self._encode_depth(depth_img, near, far)
        gru_input = torch.cat((depth_feature, proprio), dim=-1)
        new_hidden = self.gru(gru_input, hidden)
        action = self.actor(new_hidden)
        return action, new_hidden

    def init_hidden(self, num_envs, device):
        return torch.zeros(num_envs, self.gru_hidden_dim, device=device)

    def reset_hidden(self, hidden, dones):
        # zero hidden-state rows for envs that just reset, so the GRU doesn't carry memory across
        # an episode boundary. Out-of-place (mask-multiply), not `hidden[dones] = 0.` - hidden is
        # still part of this chunk's autograd graph (backward hasn't run yet), and in-place writes
        # to a tensor GRUCell's backward needs would corrupt it.
        keep = (~dones).to(hidden.dtype).unsqueeze(-1)
        return hidden * keep
