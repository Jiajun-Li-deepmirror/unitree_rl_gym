import math
import sys
import types

import numpy as np
import torch

from isaacgym import gymapi, gymtorch, gymutil, terrain_utils
from isaacgym.torch_utils import torch_rand_float

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw


class GO2Stairs(LeggedRobot):
    """ GO2 trained on stairs terrain. Adds trimesh terrain generation and height-scan sensing
        on top of the base LeggedRobot. The raw height scan is appended to the proprioceptive
        observation as-is; it's encoded into a latent by ActorCriticHeightEncoder (trained
        end-to-end with the policy), not by the env - see legged_gym/algorithms/.
    """

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
        """ A single continuously-ascending U-shaped (switchback) staircase: flight 1 climbs
            in +x, a landing turns the walking direction around, flight 2 climbs further
            going back in -x, one flight-width over in y. Bypasses the curriculum Terrain
            grid entirely - meant for a single robot (play_keyboard.py showcase run).
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
        # flight 1 and flight 2 share this x-range (mirrored, offset in y); the run-up before
        # flight 1 has to be at least as wide as the top platform carved out of it below flight 2
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

        # flight 2: continues climbing, going back in -x, one flight-width over in y - starts
        # flush against the landing's near edge (NOT its far edge) so it doesn't climb back
        # into and overwrite part of the landing; ends back at flight 1's own start x
        x = landing_start
        for _ in range(u_cfg.num_steps):
            x_start = x - step_width_px
            height += step_height_px
            height_field_raw[x_start:x, y1:y1 + flight_width_px] = height
            x = x_start

        # top platform: flat landing at the final height, carved out of the run_up_px region
        # (same x-range as the bottom spawn area, but on flight 2's y-strip)
        height_field_raw[x - top_platform_px:x, y1:y1 + flight_width_px] = height

        vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
            height_field_raw, horizontal_scale, vertical_scale, cfg.slope_treshold)

        terrain = types.SimpleNamespace()
        terrain.cfg = cfg
        terrain.height_field_raw = height_field_raw
        terrain.heightsamples = height_field_raw
        terrain.tot_rows, terrain.tot_cols = tot_rows, tot_cols
        terrain.vertices, terrain.triangles = vertices, triangles
        # spawn just in front of flight 1's first step (facing the stairs), not centered on
        # the run-up platform. NOTE: env_origins is a *world* position, added directly to
        # root_states - and the mesh itself is placed at world = pixel*horizontal_scale -
        # border_size (see _create_trimesh), which exactly cancels the border_px pixel
        # padding baked into every x/y index here. So this must NOT add border_px/y0 back in,
        # or the spawn point ends up shifted by a full border_size (default 25m) off the mesh.
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

    def _reset_root_states(self, env_ids):
        if not self.cfg.terrain.u_shape_playground:
            return super()._reset_root_states(env_ids)
        # same as the base class, but skips its +-1m random xy jitter: on this compact
        # showcase platform that jitter can dump the robot onto the first step or past the
        # platform edge, defeating the point of a fixed, deliberately placed spawn point
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _get_env_origins(self):
        if self.cfg.terrain.u_shape_playground:
            self.custom_origins = True
            origin = torch.tensor(self.terrain.env_origins[0, 0], device=self.device, dtype=torch.float)
            self.env_origins = origin.unsqueeze(0).repeat(self.num_envs, 1)
            # no real curriculum on this terrain; _reward_feet_edge just gates on level > 3
            self.terrain_levels = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
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

    def reset_idx(self, env_ids):
        run_curriculum = len(env_ids) > 0 and self.cfg.terrain.curriculum and not self.cfg.terrain.u_shape_playground
        if run_curriculum:
            self._update_terrain_curriculum(env_ids)
        super().reset_idx(env_ids)
        if run_curriculum:
            # so terrain difficulty progression is visible in tensorboard, same as
            # command curriculum's "max_command_x" a few lines up in the base class
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())

    def _update_terrain_curriculum(self, env_ids):
        """ Moves each env to a harder terrain row if it walked far enough this episode (past
            half the terrain tile length), or an easier one if it barely covered the ground its
            own commanded speed implied it should have. Must run BEFORE _reset_dofs/
            _reset_root_states (hence the reset_idx override above, calling this first) since it
            reads the pre-reset root_states and updates env_origins in place, which the reset
            then spawns the robot at.
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
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = torch.clip(points[:, :, 0].view(-1), 0, self.height_samples.shape[0] - 2)
        py = torch.clip(points[:, :, 1].view(-1), 0, self.height_samples.shape[1] - 2)

        heights = torch.min(self.height_samples[px, py], self.height_samples[px + 1, py])
        heights = torch.min(heights, self.height_samples[px, py + 1])
        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _draw_debug_vis(self):
        """ Draws a small sphere at every height-scan sample point: red if that point falls
            on a stair-edge cell (self.x_edge_mask, same mask _reward_feet_edge uses), green
            otherwise. Recomputed fresh from the CURRENT root/base state every call (rather
            than reusing values cached during this step's _get_heights()) so it can't ever
            draw one step's spheres against a different step's (post-reset) robot pose -
            that mismatch is what caused the visible flicker/jumping.
        """
        self.gym.clear_lines(self.viewer)

        world_points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) \
            + self.root_states[:, :3].unsqueeze(1)
        idx_points = world_points + self.terrain.cfg.border_size
        idx_points = (idx_points / self.terrain.cfg.horizontal_scale).long()
        px = torch.clip(idx_points[:, :, 0].view(-1), 0, self.height_samples.shape[0] - 2)
        py = torch.clip(idx_points[:, :, 1].view(-1), 0, self.height_samples.shape[1] - 2)

        heights = torch.min(self.height_samples[px, py], self.height_samples[px + 1, py])
        heights = torch.min(heights, self.height_samples[px, py + 1])
        heights = (heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale).cpu().numpy()
        is_edge = self.x_edge_mask[px, py].view(self.num_envs, -1).cpu().numpy()
        world_xy = world_points[:, :, :2].cpu().numpy()

        green = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0, 1, 0))
        red = gymutil.WireframeSphereGeometry(0.03, 4, 4, None, color=(1, 0, 0))
        for i in range(self.num_envs):
            for j in range(self.num_height_points):
                pose = gymapi.Transform(gymapi.Vec3(float(world_xy[i, j, 0]), float(world_xy[i, j, 1]), float(heights[i, j])))
                geom = red if is_edge[i, j] else green
                gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[i], pose)

    def _draw_camera_fov(self, color=(1.0, 0.6, 0.0)):
        """ Draws a small wireframe pyramid from the depth camera's current position out to a
            small rectangle at cam_cfg.fov_viz_length, sized to match its actual
            horizontal_fov/aspect ratio - a quick visual sanity check of where it's looking.
        """
        cam_cfg = self.cfg.camera
        env_h = self.envs[0]
        transform = self.gym.get_camera_transform(self.sim, env_h, self.camera_handle)
        apex = transform.p
        fwd = transform.r.rotate(gymapi.Vec3(1, 0, 0))
        up = transform.r.rotate(gymapi.Vec3(0, 0, 1))
        right = transform.r.rotate(gymapi.Vec3(0, 1, 0))

        length = cam_cfg.fov_viz_length
        hfov = math.radians(cam_cfg.horizontal_fov)
        vfov = math.radians(cam_cfg.horizontal_fov) * cam_cfg.height / cam_cfg.width
        half_w = length * math.tan(hfov / 2)
        half_h = length * math.tan(vfov / 2)

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

    def render(self, sync_frame_time=True):
        """ Same as BaseTask.render(), except the height-scan/camera-FOV debug lines are
            (re)drawn right before gym.draw_viewer() - not from post_physics_step(), a step
            earlier - since clearing/adding them anywhere else left a window where the viewer
            could capture a half-updated line buffer, which is what caused the flicker.
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

    def _init_feet_state(self):
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_pos = self.rigid_body_states_view[:, self.feet_indices, :3]

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.feet_pos = self.rigid_body_states_view[:, self.feet_indices, :3]

    def _init_camera(self):
        """ Mounts a depth camera on env 0 only (see cfg.camera) - interactive use in
            play_keyboard.py, never enabled during training (cfg.camera.use_camera is always
            False unless play_keyboard.py's --use_camera flag turns it on for num_envs=1).
        """
        self.camera_handle = None
        cam_cfg = self.cfg.camera
        if not cam_cfg.use_camera:
            return
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
        # identity attach looks along the body's local +x (forward); pitching around local y
        # tilts the view down toward the ground/stairs ahead
        local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), cam_cfg.mount_pitch)
        self.gym.attach_camera_to_body(self.camera_handle, self.envs[0], body_handle, local_transform, gymapi.FOLLOW_TRANSFORM)

    def get_camera_depth_image(self):
        """ Renders and returns the mounted depth camera's current image as a (height, width)
            numpy array of positive distances in meters (IMAGE_DEPTH itself returns negative
            distances). Only valid when cfg.camera.use_camera is True.
        """
        self.gym.render_all_camera_sensors(self.sim)
        depth = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handle, gymapi.IMAGE_DEPTH)
        return -depth

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
        # Raw height scan (not an encoded latent): the encoder is a trainable submodule of
        # ActorCriticHeightEncoder now, so it needs to see the actual scan to be able to learn
        # anything from it - see legged_gym/algorithms/height_actor_critic.py.
        height_scan = torch.clip(
            self.root_states[:, 2].unsqueeze(1) - self.cfg.rewards.base_height_target - self.measured_heights,
            -1, 1.,
        ) * self.obs_scales.height_measurements
        if self.add_noise:
            noise_scale = (self.cfg.noise.noise_scales.height_measurements
                           * self.cfg.noise.noise_level * self.obs_scales.height_measurements)
            height_scan += (2 * torch.rand_like(height_scan) - 1) * noise_scale

        self.obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            height_scan,
        ), dim=-1)
        # add noise if needed (noise_scale_vec is zero-padded over the height-scan tail,
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
