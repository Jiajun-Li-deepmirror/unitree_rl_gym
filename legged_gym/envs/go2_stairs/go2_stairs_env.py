import numpy as np
import torch
import torch.nn as nn

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import torch_rand_float

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw


class GO2Stairs(LeggedRobot):
    """ GO2 trained on stairs terrain. Adds trimesh terrain generation and height-scan
        sensing on top of the base LeggedRobot, and encodes the height scan to a small
        latent vector (via an MLP) before it is appended to the proprioceptive observation.
    """

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        else:
            raise ValueError(f"go2_stairs only supports terrain.mesh_type in ['plane', 'trimesh'], got '{mesh_type}'")
        self._create_envs()

    def _create_trimesh(self):
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -self.cfg.terrain.border_size
        tm_params.transform.p.y = -self.cfg.terrain.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        self._compute_edge_mask()

    def _compute_edge_mask(self):
        """ Marks heightfield cells that sit on a step edge (height jump to a neighboring
            cell exceeds rewards.edge_height_threshold), used to penalize feet landing right
            on the edge of a stair instead of solidly on a tread.
        """
        height_field = self.terrain.height_field_raw.astype(np.float32) * self.cfg.terrain.vertical_scale
        threshold = self.cfg.rewards.edge_height_threshold
        dx = np.abs(np.diff(height_field, axis=0, prepend=height_field[:1, :]))
        dy = np.abs(np.diff(height_field, axis=1, prepend=height_field[:, :1]))
        edge = (dx > threshold) | (dy > threshold)
        self.x_edge_mask = torch.tensor(edge, device=self.device, dtype=torch.bool)

    def _get_env_origins(self):
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            return super()._get_env_origins()
        self.custom_origins = True
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        max_init_level = self.cfg.terrain.max_init_terrain_level
        if not self.cfg.terrain.curriculum:
            max_init_level = self.cfg.terrain.num_rows - 1
        self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
        self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device),
                                        (self.num_envs / self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
        self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
        self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]

    def _init_height_points(self):
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)
        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self):
        points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) \
            + self.root_states[:, :3].unsqueeze(1)
        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = torch.clip(points[:, :, 0].view(-1), 0, self.height_samples.shape[0] - 2)
        py = torch.clip(points[:, :, 1].view(-1), 0, self.height_samples.shape[1] - 2)

        heights = torch.min(self.height_samples[px, py], self.height_samples[px + 1, py])
        heights = torch.min(heights, self.height_samples[px, py + 1])
        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _build_height_encoder(self):
        cfg = self.cfg.height_encoder
        dims = [self.num_height_points] + list(cfg.hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ELU()]
        layers.append(nn.Linear(dims[-1], cfg.latent_dim))
        return nn.Sequential(*layers).to(self.device)

    def _init_feet_state(self):
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_pos = self.rigid_body_states_view[:, self.feet_indices, :3]

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.feet_pos = self.rigid_body_states_view[:, self.feet_indices, :3]

    def _init_buffers(self):
        super()._init_buffers()
        self._init_feet_state()
        self.hip_indices = torch.tensor(
            [i for i, name in enumerate(self.dof_names) if "hip" in name],
            dtype=torch.long, device=self.device,
        )
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
            self.measured_heights = torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
            self.height_encoder = self._build_height_encoder()

    def _post_physics_step_callback(self):
        self._update_gait_phase()
        self.update_feet_state()
        # NOTE: computed from self.last_contacts before _reward_feet_air_time (which runs
        # later, in compute_reward) overwrites it with this step's contact state.
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        return super()._post_physics_step_callback()

    def _resample_commands(self, env_ids):
        """ Same lin_vel_x / lin_vel_y / ang_vel_yaw sampling as the base class (heading_command
            is always False for this task), but zeroing vx and vyaw independently below their own
            0.1 threshold instead of the base class's combined-xy-norm > 0.2 check.
        """
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(
            self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        self.commands[env_ids, 0] *= torch.abs(self.commands[env_ids, 0]) > 0.1
        self.commands[env_ids, 2] *= torch.abs(self.commands[env_ids, 2]) > 0.1

    def _update_gait_phase(self):
        """ Trotting gait phase for the diagonal leg pairs (FL+RR / FR+RL).
            feet_indices order is [FL, FR, RL, RR] (URDF body order filtered by foot_name).
        """
        cycle_time = self.cfg.rewards.cycle_time
        phase = (self.episode_length_buf * self.dt) % cycle_time / cycle_time
        phase_offsets = torch.tensor([0.0, 0.5, 0.5, 0.0], device=self.device)
        self.leg_phase = (phase.unsqueeze(1) + phase_offsets.unsqueeze(0)) % 1.0
        # only enforce the alternating gait when a non-zero lin/ang vel command is given
        self.gait_enabled = torch.any(self.commands[:, :3] != 0., dim=1)

    def compute_observations(self):
        height_scan = torch.clip(
            self.root_states[:, 2].unsqueeze(1) - self.cfg.rewards.base_height_target - self.measured_heights,
            -1, 1.,
        ) * self.obs_scales.height_measurements
        if self.add_noise:
            noise_scale = (self.cfg.noise.noise_scales.height_measurements
                           * self.cfg.noise.noise_level * self.obs_scales.height_measurements)
            height_scan += (2 * torch.rand_like(height_scan) - 1) * noise_scale
        with torch.no_grad():
            height_latent = self.height_encoder(height_scan)

        self.obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            height_latent,
        ), dim=-1)
        # add noise if needed (noise_scale_vec is zero-padded over the height-latent tail,
        # since noise there is already injected into the raw height scan above)
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _reward_base_height(self):
        # Penalize base height away from target, measured relative to the feet currently on the
        # ground (not the world z=0 plane, since the stairs terrain isn't flat) so this doesn't
        # depend on the height-scan implementation.
        contact = (self.contact_forces[:, self.feet_indices, 2] > 1.).float()
        contact_count = contact.sum(dim=1).clamp(min=1)
        stance_height = (self.feet_pos[:, :, 2] * contact).sum(dim=1) / contact_count
        base_height = self.root_states[:, 2] - stance_height
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_gait_phase(self):
        # reward feet contact state matching the expected stance/swing of the trot phase
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(len(self.feet_indices)):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1.
            res += ~(contact ^ is_stance)
        return res * self.gait_enabled

    def _reward_stand_still(self):
        # Penalize motion away from the default pose, but only when vx, vy AND vyaw are all zero
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (~self.gait_enabled)

    def _reward_feet_edge(self):
        # Penalize feet contacting the terrain right on a stair edge (unstable foothold),
        # only once the curriculum has advanced past the easiest terrain rows.
        feet_pos_xy = ((self.feet_pos[:, :, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()  # (num_envs, 4, 2)
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0] - 1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1] - 1)
        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]

        self.feet_at_edge = self.contact_filt & feet_at_edge
        rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        return rew.float()

    def _reward_hip_pos(self):
        # Penalize hip joints drifting from their default angle (keeps the stance width stable)
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)
