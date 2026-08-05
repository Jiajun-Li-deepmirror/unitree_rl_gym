import sys
import types

import numpy as np
import torch

from isaacgym import gymapi, gymtorch, gymutil, terrain_utils
from isaacgym.torch_utils import quat_apply, quat_from_euler_xyz, torch_rand_float

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi

# below this magnitude, vx/vyaw are snapped to exactly 0 wherever self.commands is written, so the
# policy never observes an ambiguous near-zero-but-nonzero command; play_keyboard.py applies the
# same threshold to its live keyboard-driven commands
VEL_DEADZONE = 0.1


class GO2Stairs(LeggedRobot):
    """ GO2 trained on stairs terrain: trimesh generation + height-scan sensing. """

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        if self.cfg.terrain.u_shape_playground:
            self.terrain = self._build_u_shape_terrain()
            self._create_trimesh()
        else:
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
        self._init_camera()

    def _build_u_shape_terrain(self):
        """ U-shaped switchback staircase: flight 1 climbs +x, turns at a landing, flight 2
            climbs further in -x. Bypasses the curriculum grid - single-robot showcase terrain
            for play_keyboard.py.
        """
        cfg = self.cfg.terrain
        u_cfg = cfg.u_shape
        horizontal_scale = cfg.horizontal_scale
        vertical_scale = cfg.vertical_scale
        border_px = round(cfg.border_size / horizontal_scale)

        step_width_px = max(round(u_cfg.step_width / horizontal_scale), 1)
        step_height_px = max(round(u_cfg.step_height / vertical_scale), 1)
        flight_width_px = max(round(u_cfg.flight_width / horizontal_scale), 1)
        platform_px = max(round(u_cfg.platform_size / horizontal_scale), 1)
        top_platform_px = max(round(u_cfg.top_platform_size / horizontal_scale), 1)
        # flight 1/2 share this x-range; run-up must be >= top_platform so it isn't overwritten
        run_up_px = max(platform_px, top_platform_px)

        x_extent = run_up_px + u_cfg.num_steps * step_width_px + platform_px
        y_extent = 2 * flight_width_px
        tot_rows = x_extent + 2 * border_px
        tot_cols = y_extent + 2 * border_px

        height_field_raw = np.zeros((tot_rows, tot_cols), dtype=np.int16)

        y0 = border_px
        y1 = y0 + flight_width_px

        # flight 1: climbs in +x, starting after the flat run-up (spawn area)
        x = border_px + run_up_px
        height = 0
        for _ in range(u_cfg.num_steps):
            x_end = x + step_width_px
            height += step_height_px
            height_field_raw[x:x_end, y0:y0 + flight_width_px] = height
            x = x_end

        # landing: flat turn at the top of flight 1, spans both flights
        landing_start = x
        height_field_raw[landing_start:landing_start + platform_px, y0:y0 + y_extent] = height

        # flight 2 climbs back in -x, one flight-width over in y, starting at the landing's near edge
        x = landing_start
        for _ in range(u_cfg.num_steps):
            x_start = x - step_width_px
            height += step_height_px
            height_field_raw[x_start:x, y1:y1 + flight_width_px] = height
            x = x_start

        # top platform: flat landing at the final height, carved out of the run-up region
        height_field_raw[x - top_platform_px:x, y1:y1 + flight_width_px] = height

        vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
            height_field_raw, horizontal_scale, vertical_scale, cfg.slope_treshold)

        terrain = types.SimpleNamespace()
        terrain.cfg = cfg
        terrain.height_field_raw = height_field_raw
        terrain.heightsamples = height_field_raw
        terrain.tot_rows, terrain.tot_cols = tot_rows, tot_cols
        terrain.vertices, terrain.triangles = vertices, triangles
        # spawn near flight 1's first step; NOTE: don't add border_px/y0 back in (mesh placement already cancels it) or spawn shifts by border_size
        spawn_margin_px = min(max(round(0.6 / horizontal_scale), 1), run_up_px)
        spawn_x = (run_up_px - spawn_margin_px) * horizontal_scale
        spawn_y = (flight_width_px / 2) * horizontal_scale
        terrain.env_origins = np.array([[[spawn_x, spawn_y, 0.0]]])  # (num_rows=1, num_cols=1, 3)
        return terrain

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
        """ Marks cells next to a height jump > edge_height_threshold, used to penalize feet on a
            stair edge. prepend-diff flags the high side, append-diff flags the low side, so both
            sides of every edge get caught.
        """
        height_field = self.terrain.height_field_raw.astype(np.float32) * self.cfg.terrain.vertical_scale
        threshold = self.cfg.rewards.edge_height_threshold
        dx_hi = np.abs(np.diff(height_field, axis=0, prepend=height_field[:1, :]))
        dx_lo = np.abs(np.diff(height_field, axis=0, append=height_field[-1:, :]))
        dy_hi = np.abs(np.diff(height_field, axis=1, prepend=height_field[:, :1]))
        dy_lo = np.abs(np.diff(height_field, axis=1, append=height_field[:, -1:]))
        edge = (dx_hi > threshold) | (dx_lo > threshold) | (dy_hi > threshold) | (dy_lo > threshold)
        self.x_edge_mask = torch.tensor(edge, device=self.device, dtype=torch.bool)

    def _get_env_origins(self):
        if self.cfg.terrain.u_shape_playground:
            self.custom_origins = True
            origin = torch.tensor(self.terrain.env_origins[0, 0], device=self.device, dtype=torch.float)
            self.env_origins = origin.unsqueeze(0).repeat(self.num_envs, 1)
            # no real curriculum grid on this terrain; not flat either, so lin_vel_z/orientation
            # get the relaxed (stairs) treatment throughout
            self.terrain_levels = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.is_flat_terrain = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return
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
        self.max_terrain_level = self.cfg.terrain.num_rows
        self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        self._init_flat_terrain_mask()

    def _init_flat_terrain_mask(self):
        """ Marks which envs sit on the dedicated flat column - terrain_proportions' last entry,
            a genuinely flat terrain type (see Terrain.make_terrain in legged_gym/utils/terrain.py) -
            by replicating the `choice` comparison Terrain.curiculum() used to pick that column's
            terrain. Used by _reward_lin_vel_z/_reward_orientation to only apply their full penalty
            there.
        """
        choice = self.terrain_types.float() / self.cfg.terrain.num_cols + 0.001
        proportions = self.cfg.terrain.terrain_proportions
        flat_lower = sum(proportions[:-1]) # cumulative share before the flat slot
        flat_upper = sum(proportions) # flat is always the last configured terrain type
        self.is_flat_terrain = (choice >= flat_lower) & (choice < flat_upper)

    def _update_terrain_curriculum(self, env_ids):
        """ Moves each env to a harder terrain row if it walked past half the tile, or an easier
            one if it fell well short of its commanded distance - standard legged_gym game-inspired
            terrain curriculum.
        """
        if not self.init_done:
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        move_up = distance > self.terrain.env_length / 2
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5) & ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def _reset_root_states(self, env_ids):
        """ Same as the base class, but with a uniformly random initial yaw instead of the fixed
            orientation from init_state.rot - stairs are approached from a random heading during
            training so the policy doesn't overfit to always starting square with the terrain.
            Skipped on the u_shape_playground: that terrain has one fixed climb direction, so
            random spawn yaw would just point the robot away from the staircase.
        """
        super()._reset_root_states(env_ids)
        if self.cfg.terrain.u_shape_playground:
            return
        yaw = torch_rand_float(-np.pi, np.pi, (len(env_ids), 1), device=self.device).squeeze(1)
        zeros = torch.zeros_like(yaw)
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                      gymtorch.unwrap_tensor(self.root_states),
                                                      gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def reset_idx(self, env_ids):
        run_curriculum = len(env_ids) > 0 and self.cfg.terrain.curriculum and not self.cfg.terrain.u_shape_playground
        if run_curriculum:
            self._update_terrain_curriculum(env_ids)
        super().reset_idx(env_ids)
        if run_curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())

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
        world_points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) \
            + self.root_states[:, :3].unsqueeze(1)
        points = world_points + self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).round().long()
        px = torch.clip(points[:, :, 0].view(-1), 0, self.height_samples.shape[0] - 1)
        py = torch.clip(points[:, :, 1].view(-1), 0, self.height_samples.shape[1] - 1)

        heights = self.height_samples[px, py]
        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _init_buffers(self):
        super()._init_buffers()
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)
        self.hip_indices = torch.tensor(
            [i for i, name in enumerate(self.dof_names) if "hip" in name],
            dtype=torch.long, device=self.device,
        )
        self.last_torques = torch.zeros_like(self.torques)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
            self.measured_heights = torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)

    def post_physics_step(self):
        super().post_physics_step()
        # updated after compute_reward (called inside super()), same timing as last_actions/last_dof_vel,
        # so _reward_delta_torques compares this step's torques against the previous step's
        self.last_torques[:] = self.torques[:]

    def _update_gait_phase(self):
        """ Trotting gait phase for the diagonal leg pairs (FL+RR / FR+RL).
            feet_indices order is [FL, FR, RL, RR] (URDF body order filtered by foot_name).
        """
        cycle_time = self.cfg.rewards.cycle_time
        phase = (self.episode_length_buf * self.dt) % cycle_time / cycle_time
        self.phase = phase  # stashed for compute_observations' sin/cos phase encoding
        phase_offsets = torch.tensor([0.0, 0.5, 0.5, 0.0], device=self.device)
        self.leg_phase = (phase.unsqueeze(1) + phase_offsets.unsqueeze(0)) % 1.0
        # only enforce the alternating gait when a non-zero lin/ang vel command is given
        self.gait_enabled = torch.any(self.commands[:, :3] != 0., dim=1)

    def _post_physics_step_callback(self):
        self._update_gait_phase()
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # NOTE: uses self.last_contacts before _reward_feet_air_time overwrites it this step
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        result = super()._post_physics_step_callback()

        # heading-command tracking, walking envs only - turns the sampled heading target
        # (commands[:,3]) into vyaw (commands[:,2]) every step, base-class style, but scoped so
        # stand-still/rotate-in-place envs keep their own directly-sampled vyaw untouched
        # (heading_command is False so the base class doesn't do this for every env itself)
        _, _, is_walking = self._command_mode()
        if torch.any(is_walking):
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            target_ang_vel = torch.clip(
                0.5 * wrap_to_pi(self.commands[:, 3] - heading),
                self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1],
            )
            target_ang_vel[torch.abs(target_ang_vel) < VEL_DEADZONE] = 0.
            self.commands[is_walking, 2] = target_ang_vel[is_walking]
        return result

    def _command_mode(self):
        """ Which command_proportions category is currently active, derived fresh from
            self.commands (not cached from _resample_commands, since most envs don't resample
            every step). Returns (is_standing_still, is_rotating_in_place, is_walking), each
            shaped (num_envs,) bool.
        """
        is_standing_still = torch.all(torch.abs(self.commands[:, :3]) < 1e-6, dim=1)
        is_rotating_in_place = (torch.abs(self.commands[:, 0]) < 1e-6) & (torch.abs(self.commands[:, 1]) < 1e-6) \
            & (torch.abs(self.commands[:, 2]) >= 1e-6)
        is_walking = ~is_standing_still & ~is_rotating_in_place
        return is_standing_still, is_rotating_in_place, is_walking

    def _resample_commands(self, env_ids):
        """ Draws each env's command from one of 3 modes, weighted by cfg.commands.command_proportions
            (ramped up over training - see cfg.commands.command_curriculum):
              - stand still: vx = vy = vyaw = 0
              - rotate in place: vx = vy = 0, vyaw sampled from ang_vel_yaw range
              - walk: vx/vy sampled from their ranges; vyaw is tracked every step from a sampled
                heading target (commands[:,3]) in _post_physics_step_callback, scoped to walking
                envs only
        """
        curr_cfg = self.cfg.commands.command_curriculum
        num_increments = 0
        if curr_cfg.enabled:
            current_iteration = self.common_step_counter // curr_cfg.num_steps_per_env
            num_increments = current_iteration // curr_cfg.increment_interval
        stand_w, rotate_w, walk_w = self.cfg.commands.command_proportions
        stand_w = stand_w + curr_cfg.increment * num_increments
        rotate_w = rotate_w + curr_cfg.increment * num_increments
        proportions = torch.cumsum(torch.tensor([stand_w, rotate_w, walk_w], device=self.device), dim=0)
        choice = torch.rand(len(env_ids), device=self.device) * proportions[-1]
        stand_ids = env_ids[choice < proportions[0]]
        rotate_ids = env_ids[(choice >= proportions[0]) & (choice < proportions[1])]
        walk_ids = env_ids[choice >= proportions[1]]

        self.commands[stand_ids, :4] = 0.

        if len(rotate_ids) > 0:
            self.commands[rotate_ids, 0] = 0.
            self.commands[rotate_ids, 1] = 0.
            vyaw = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1],
                (len(rotate_ids), 1), device=self.device).squeeze(1)
            vyaw[torch.abs(vyaw) < VEL_DEADZONE] = 0.
            self.commands[rotate_ids, 2] = vyaw
            self.commands[rotate_ids, 3] = 0.

        if len(walk_ids) > 0:
            vx = torch_rand_float(
                self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1],
                (len(walk_ids), 1), device=self.device).squeeze(1)
            vx[torch.abs(vx) < VEL_DEADZONE] = 0.
            self.commands[walk_ids, 0] = vx
            self.commands[walk_ids, 1] = torch_rand_float(
                self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1],
                (len(walk_ids), 1), device=self.device).squeeze(1)
            self.commands[walk_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0], self.command_ranges["heading"][1],
                (len(walk_ids), 1), device=self.device).squeeze(1)

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:12+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[12+self.num_actions:12+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[12+2*self.num_actions:12+3*self.num_actions] = 0. # previous actions
        tail_start = 12 + 3*self.num_actions
        noise_vec[tail_start:tail_start+2] = 0. # sin/cos gait phase
        noise_vec[tail_start+2:tail_start+5] = 0. # one-hot command mode
        noise_vec[tail_start+5:] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        return noise_vec

    def compute_observations(self):
        height_scan = torch.clip(
            self.root_states[:, 2].unsqueeze(1) - self.cfg.rewards.base_height_target - self.measured_heights,
            -1, 1.,
        ) * self.obs_scales.height_measurements

        # sin/cos gait phase, so the policy knows where it is in the trot cycle
        sin_phase = torch.sin(2 * torch.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * torch.pi * self.phase).unsqueeze(1)

        # one-hot [stand still, rotate in place, walking], derived fresh via _command_mode
        is_standing_still, is_rotating_in_place, is_walking = self._command_mode()
        command_mode = torch.stack((is_standing_still, is_rotating_in_place, is_walking), dim=1).float()

        self.obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            sin_phase,
            cos_phase,
            command_mode,
            height_scan,
        ), dim=-1)
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def render(self, sync_frame_time=True):
        """ Same as BaseTask.render(), but debug lines are (re)drawn right before draw_viewer()
            so height-scan/feet-edge markers never lag a step behind the current pose.
        """
        if not self.viewer:
            return
        if self.gym.query_viewer_has_closed(self.viewer):
            sys.exit()
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "QUIT" and evt.value > 0:
                sys.exit()
            elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                self.enable_viewer_sync = not self.enable_viewer_sync
        if self.device != 'cpu':
            self.gym.fetch_results(self.sim, True)
        if self.enable_viewer_sync:
            self.gym.step_graphics(self.sim)
            if self.cfg.terrain.measure_heights:
                self._draw_debug_vis()
            if self.camera_handle is not None:
                self._draw_camera_fov()
            self.gym.draw_viewer(self.viewer, self.sim, True)
            if sync_frame_time:
                self.gym.sync_frame_time(self.sim)
        else:
            self.gym.poll_viewer_events(self.viewer)

    def _init_camera(self):
        """ Mounts a depth camera on env 0 only - interactive use in play_keyboard.py, never
            enabled during training (cfg.camera.use_camera is always False otherwise).
        """
        self.camera_handle = None
        if not hasattr(self.cfg, "camera") or not self.cfg.camera.use_camera:
            return
        cam_cfg = self.cfg.camera
        camera_props = gymapi.CameraProperties()
        camera_props.width = max(1, round(cam_cfg.width * cam_cfg.scale))
        camera_props.height = max(1, round(cam_cfg.height * cam_cfg.scale))
        camera_props.horizontal_fov = cam_cfg.horizontal_fov
        camera_props.near_plane = cam_cfg.near_plane
        camera_props.far_plane = cam_cfg.far_plane
        self.camera_handle = self.gym.create_camera_sensor(self.envs[0], camera_props)
        body_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], cam_cfg.mount_body)
        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(*cam_cfg.mount_pos)
        # identity attach looks along the body's local +x; pitching around local y tilts the view down
        local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), cam_cfg.mount_pitch)
        self.gym.attach_camera_to_body(self.camera_handle, self.envs[0], body_handle, local_transform, gymapi.FOLLOW_TRANSFORM)

    def get_camera_depth_image(self):
        """ Renders and returns the depth image as a (height, width) array of positive meters
            (IMAGE_DEPTH itself returns negative distances). Only valid when use_camera is True.
        """
        self.gym.render_all_camera_sensors(self.sim)
        depth = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handle, gymapi.IMAGE_DEPTH)
        return -depth

    def _draw_camera_fov(self, color=(1.0, 0.6, 0.0)):
        """ Draws a small wireframe pyramid from the camera's position, sized to its actual
            horizontal_fov/aspect ratio - a visual sanity check of where it's looking.
        """
        cam_cfg = self.cfg.camera
        env_h = self.envs[0]
        transform = self.gym.get_camera_transform(self.sim, env_h, self.camera_handle)
        apex = transform.p
        fwd = transform.r.rotate(gymapi.Vec3(1, 0, 0))
        up = transform.r.rotate(gymapi.Vec3(0, 0, 1))
        right = transform.r.rotate(gymapi.Vec3(0, 1, 0))

        length = cam_cfg.fov_viz_length
        hfov = np.radians(cam_cfg.horizontal_fov)
        vfov = np.radians(cam_cfg.horizontal_fov) * cam_cfg.height / cam_cfg.width
        half_w = length * np.tan(hfov / 2)
        half_h = length * np.tan(vfov / 2)

        def corner(sx, sy):
            return gymapi.Vec3(
                apex.x + fwd.x * length + right.x * sx * half_w + up.x * sy * half_h,
                apex.y + fwd.y * length + right.y * sx * half_w + up.y * sy * half_h,
                apex.z + fwd.z * length + right.z * sx * half_w + up.z * sy * half_h,
            )

        corners = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)]
        verts = []
        for c in corners:
            verts += [apex.x, apex.y, apex.z, c.x, c.y, c.z]
        for i in range(4):
            a, b = corners[i], corners[(i + 1) % 4]
            verts += [a.x, a.y, a.z, b.x, b.y, b.z]
        vertices = np.array(verts, dtype=np.float32)
        colors = np.array(list(color) * 8, dtype=np.float32)
        self.gym.add_lines(self.viewer, env_h, 8, vertices, colors)

    def _draw_debug_vis(self):
        """ Draws a sphere at every height-scan point (red on a stair-edge cell, green otherwise)
            and a sphere at each foot (red if it's standing on a stair-edge cell), so the
            _reward_feet_edge lookup can be sanity-checked visually during play.
        """
        self.gym.clear_lines(self.viewer)

        world_points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) \
            + self.root_states[:, :3].unsqueeze(1)
        idx_points = world_points + self.terrain.cfg.border_size
        idx_points = (idx_points / self.terrain.cfg.horizontal_scale).round().long()
        px = torch.clip(idx_points[:, :, 0].view(-1), 0, self.height_samples.shape[0] - 1)
        py = torch.clip(idx_points[:, :, 1].view(-1), 0, self.height_samples.shape[1] - 1)

        heights = (self.height_samples[px, py].view(self.num_envs, -1) * self.terrain.cfg.vertical_scale).cpu().numpy()
        is_edge = self.x_edge_mask[px, py].view(self.num_envs, -1).cpu().numpy()
        world_xy = world_points[:, :, :2].cpu().numpy()

        green = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0, 1, 0))
        red = gymutil.WireframeSphereGeometry(0.03, 4, 4, None, color=(1, 0, 0))
        for i in range(self.num_envs):
            for j in range(self.num_height_points):
                pose = gymapi.Transform(gymapi.Vec3(float(world_xy[i, j, 0]), float(world_xy[i, j, 1]), float(heights[i, j])))
                geom = red if is_edge[i, j] else green
                gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[i], pose)

        feet_pos_xy = ((self.rigid_body_states[:, self.feet_indices, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0] - 1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1] - 1)
        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]].cpu().numpy()
        feet_xy = self.rigid_body_states[:, self.feet_indices, :2].cpu().numpy()
        feet_z = self.rigid_body_states[:, self.feet_indices, 2].cpu().numpy()

        foot_green = gymutil.WireframeSphereGeometry(0.03, 8, 8, None, color=(0, 1, 0))
        foot_red = gymutil.WireframeSphereGeometry(0.045, 8, 8, None, color=(1, 0, 0))
        for i in range(self.num_envs):
            for k in range(len(self.feet_indices)):
                pose = gymapi.Transform(gymapi.Vec3(float(feet_xy[i, k, 0]), float(feet_xy[i, k, 1]), float(feet_z[i, k])))
                geom = foot_red if feet_at_edge[i, k] else foot_green
                gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[i], pose)

    def _reward_feet_edge(self):
        feet_pos_xy = ((self.rigid_body_states[:, self.feet_indices, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()  # (num_envs, 4, 2)
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0]-1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1]-1)
        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]

        self.feet_at_edge = self.contact_filt & feet_at_edge
        rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        return rew.float()

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)

    def _reward_lin_vel_z(self):
        # full penalty only on flat terrain; halved everywhere else, since some vertical velocity
        # is expected/unavoidable while climbing stairs
        rew = torch.square(self.base_lin_vel[:, 2])
        rew[~self.is_flat_terrain] *= 0.5
        return rew

    def _reward_orientation(self):
        # only enforced on flat ground; stairs require the body to pitch to climb/descend
        rew = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        rew[~self.is_flat_terrain] = 0.
        return rew

    def _reward_base_height(self):
        # only enforced on flat ground; absolute world-z height isn't a meaningful target once
        # the terrain under the robot isn't at z=0 (stairs), so this is left off there
        rew = torch.square(self.root_states[:, 2] - self.cfg.rewards.base_height_target)
        rew[~self.is_flat_terrain] = 0.
        return rew

    def _reward_action_rate(self):
        # aligned with reference: L2 norm of the action delta, not sum of squares (base class default)
        return torch.norm(self.last_actions - self.actions, dim=1)

    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)

    def _reward_dof_error(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_feet_stumble(self):
        # penalize feet hitting vertical surfaces; same shape as base _reward_stumble but a 4x
        # (not 5x) horizontal/vertical contact-force ratio, matching the reference
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >
                          4 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1).float()

    def _gait_phase_match_count(self):
        # feet count currently matching expected trot phase - shared by _reward_gait_phase and
        # could double as a monitoring metric later
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(len(self.feet_indices)):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1.
            res += ~(contact ^ is_stance)
        return res

    def _reward_gait_phase(self):
        # rewarded while rotating in place or walking, on any terrain - the diagonal trot phase
        # itself is still the target gait on stairs, unlike feet_swing_height's fixed clearance
        # target which doesn't make sense once foot height depends on the terrain step
        _, is_rotating_in_place, is_walking = self._command_mode()
        return self._gait_phase_match_count() * (is_rotating_in_place | is_walking)

    def _reward_feet_swing_height(self):
        # penalize swing feet missing the target clearance above local terrain, while rotating
        # in place or walking, but only on flat ground (see _reward_gait_phase)
        feet_pos_xy = ((self.rigid_body_states[:, self.feet_indices, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.height_samples.shape[0] - 1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.height_samples.shape[1] - 1)
        terrain_height = self.height_samples[feet_pos_xy[..., 0], feet_pos_xy[..., 1]] * self.terrain.cfg.vertical_scale
        clearance = self.rigid_body_states[:, self.feet_indices, 2] - terrain_height

        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        # only penalize falling short of the target - clearing higher (e.g. a tall stair riser) is fine
        shortfall = torch.clamp(self.cfg.rewards.feet_swing_height_target - clearance, min=0.)
        swing_error = torch.square(shortfall) * (~contact)
        _, is_rotating_in_place, is_walking = self._command_mode()
        return torch.sum(swing_error, dim=1) * (is_rotating_in_place | is_walking) * self.is_flat_terrain

    def _reward_stand_still_contact(self):
        # bonus for keeping all 4 feet planted while standing still, on any terrain
        is_standing_still, _, _ = self._command_mode()
        all_feet_contact = torch.all(self.contact_forces[:, self.feet_indices, 2] > 1., dim=1)
        return all_feet_contact.float() * is_standing_still
