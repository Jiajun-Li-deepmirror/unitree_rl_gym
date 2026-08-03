import math
import sys
import types

import numpy as np
import torch

from isaacgym import gymapi, gymtorch, gymutil, terrain_utils
from isaacgym.torch_utils import torch_rand_float, quat_apply, quat_from_euler_xyz

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi


class GO2Stairs(LeggedRobot):
    """ GO2 trained on stairs terrain: trimesh generation + height-scan sensing. Height scan is
        raw (encoded by ActorCriticHeightEncoder, not here) - see legged_gym/algorithms/.
    """

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        if self.cfg.terrain.u_shape_playground:
            self.terrain = self._build_u_shape_terrain()
            self._create_trimesh()
        else:
            mesh_type = self.cfg.terrain.mesh_type
            if mesh_type == 'plane':
                self._create_ground_plane()
            elif mesh_type == 'trimesh':
                self.terrain = self._build_wave_stairs_terrain()
                self._create_trimesh()
            else:
                raise ValueError(f"go2_stairs only supports terrain.mesh_type in ['plane', 'trimesh'], got '{mesh_type}'")
        self._create_envs()
        self._init_camera()

    def _build_u_shape_terrain(self):
        """ U-shaped switchback staircase: flight 1 climbs +x, turns at a landing, flight 2
            climbs further in -x. Bypasses the curriculum grid - single-robot showcase terrain.
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

    def _build_wave_stairs_terrain(self):
        """ Curriculum-compatible grid of wave-shaped staircases: num_rows ridges chained
            back-to-back along x per column. Each ridge is a two-sided staircase - flat trough,
            climbs num_steps_per_side in +x, a short peak platform, then descends the same
            num_steps_per_side back to the SAME baseline (height 0) the trough sits at. Row
            picks difficulty via the same step_height formula as terrain.py's make_terrain
            (0.05 + 0.278*difficulty), for continuity with what was already being trained on.

            Every ridge starts and ends at height 0, so chaining them leaves no cliff at the
            difficulty-row boundary the way a flat top platform dropping straight back to 0
            would - just progressively taller waves, like a 1-axis pyramid (climbs toward a
            ridge line, not radially toward a point) repeated with growing amplitude. First half
            of columns are "hill" ridges (climb then descend, like an above-ground pyramid);
            second half are "pit" ridges (descend then climb back up, like a dug-out pyramid) -
            mirrors terrain.py make_terrain's up/down pyramid_stairs_terrain split.

            Doesn't reproduce terrain_proportions' other categories (slope/rough/discrete) -
            this is a dedicated stairs-only replacement, matching go2_stairs' own
            terrain_proportions being 100% stairs anyway.
        """
        cfg = self.cfg.terrain
        wave_cfg = cfg.wave_stairs
        horizontal_scale = cfg.horizontal_scale
        vertical_scale = cfg.vertical_scale
        num_rows, num_cols = cfg.num_rows, cfg.num_cols
        border_px = round(cfg.border_size / horizontal_scale)

        tile_wid_px = round(cfg.terrain_width / horizontal_scale)
        step_width_px = max(round(wave_cfg.step_width / horizontal_scale), 1)
        lead_in_px = max(round(wave_cfg.lead_in_size / horizontal_scale), 1)
        peak_platform_px = max(round(wave_cfg.peak_platform_size / horizontal_scale), 1)
        num_steps = wave_cfg.num_steps_per_side
        ridge_len_px = lead_in_px + 2 * num_steps * step_width_px + peak_platform_px

        tot_rows = num_rows * ridge_len_px + 2 * border_px
        tot_cols = num_cols * tile_wid_px + 2 * border_px
        height_field_raw = np.zeros((tot_rows, tot_cols), dtype=np.int16)
        env_origins = np.zeros((num_rows, num_cols, 3), dtype=np.float32)

        ridge_len = ridge_len_px * horizontal_scale
        # spawn a bit into the flat trough, not right at the ridge's own near boundary
        spawn_local_px = min(max(round(0.5 / horizontal_scale), 1), lead_in_px)

        for j in range(num_cols):
            sign = 1 if j < num_cols / 2 else -1  # +1: hill (up then down), -1: pit (down then up)
            col_y0 = border_px + j * tile_wid_px
            for i in range(num_rows):
                difficulty = i / num_rows
                step_height_px = max(round((0.05 + 0.278 * difficulty) / vertical_scale), 1)

                ridge_x0 = border_px + i * ridge_len_px
                x = ridge_x0 + lead_in_px
                height = 0
                for _ in range(num_steps):
                    x_end = x + step_width_px
                    height += sign * step_height_px
                    height_field_raw[x:x_end, col_y0:col_y0 + tile_wid_px] = height
                    x = x_end
                x_end = x + peak_platform_px
                height_field_raw[x:x_end, col_y0:col_y0 + tile_wid_px] = height
                x = x_end
                for _ in range(num_steps):
                    x_end = x + step_width_px
                    height -= sign * step_height_px
                    height_field_raw[x:x_end, col_y0:col_y0 + tile_wid_px] = height
                    x = x_end
                # height is back to 0 here, exactly matching the next ridge's own trough

                env_origins[i, j, 0] = i * ridge_len + spawn_local_px * horizontal_scale
                env_origins[i, j, 1] = (j + 0.5) * cfg.terrain_width
                env_origins[i, j, 2] = 0.0  # flat trough, untouched by the loop above

        vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
            height_field_raw, horizontal_scale, vertical_scale, cfg.slope_treshold)

        terrain = types.SimpleNamespace()
        terrain.cfg = cfg
        terrain.height_field_raw = height_field_raw
        terrain.heightsamples = height_field_raw
        terrain.tot_rows, terrain.tot_cols = tot_rows, tot_cols
        terrain.vertices, terrain.triangles = vertices, triangles
        terrain.env_origins = env_origins
        terrain.env_length = ridge_len  # so _update_terrain_curriculum's move_up threshold (half a ridge) scales with the new tile size
        terrain.env_width = cfg.terrain_width
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

    def _reset_root_states(self, env_ids):
        if not self.cfg.terrain.u_shape_playground:
            # skip the base class's +-1m xy jitter here too, so envs sharing the same
            # (terrain_level, terrain_type) truly spawn at the identical point instead of each
            # scattering randomly within a 2m box around it
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            # random spawn yaw in +-90deg - heading-command tracking (see _post_physics_step_callback)
            # still turns walking envs the rest of the way toward the climb direction
            yaw = torch_rand_float(-math.pi / 2, math.pi / 2, (len(env_ids), 1), device=self.device).squeeze(1)
            zeros = torch.zeros_like(yaw)
            self.root_states[env_ids, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)
            # zero spawn velocity (base class randomizes +-0.5) so resets restart cleanly at rest
            self.root_states[env_ids, 7:13] = 0.
            env_ids_int32 = env_ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
            return
        # skip the +-1m xy jitter here - could dump the robot onto/past a step on this compact platform
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
        self.env_origins[:] = self._terrain_spawn_origins(self.terrain_levels, self.terrain_types)

    def _terrain_spawn_origins(self, terrain_levels, terrain_types):
        """ Every env of a given (terrain_level, terrain_type) spawns at that ridge's own spawn
            point (terrain_origins, straight from GO2Stairs._build_wave_stairs_terrain).
        """
        origins = self.terrain_origins[terrain_levels, terrain_types].clone()
        return origins

    def reset_idx(self, env_ids):
        run_curriculum = len(env_ids) > 0 and self.cfg.terrain.curriculum and not self.cfg.terrain.u_shape_playground
        if run_curriculum:
            self._update_terrain_curriculum(env_ids)
        if len(env_ids) > 0:
            # reset to False; _resample_commands below will flip it back on if the new episode walks
            self.had_walking_command[env_ids] = False
        super().reset_idx(env_ids)
        if run_curriculum:
            # so terrain difficulty progression is visible in tensorboard
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if len(env_ids) > 0:
            self._log_monitoring_metrics(env_ids)

    def _log_monitoring_metrics(self, env_ids):
        """ Reports monitoring-only metrics to tensorboard (extras["episode"]) - no reward attached.
            Every key must be set on every call (0 if the mask is empty) since rsl_rl's log() assumes
            every ep_infos entry has the same key set, or it KeyErrors.
        """
        stand_mask = self.stand_still_height_err_count[env_ids] > 0
        mean_err = self.stand_still_height_err_sum[env_ids] / self.stand_still_height_err_count[env_ids].clamp(min=1)
        self.extras["episode"]["stand_still_base_height_error"] = (
            torch.mean(mean_err[stand_mask]) if stand_mask.any() else torch.zeros((), device=self.device))
        self.stand_still_height_err_sum[env_ids] = 0.
        self.stand_still_height_err_count[env_ids] = 0.

        rotate_mask = self.rotate_ang_vel_err_count[env_ids] > 0
        mean_err = self.rotate_ang_vel_err_sum[env_ids] / self.rotate_ang_vel_err_count[env_ids].clamp(min=1)
        self.extras["episode"]["rotate_in_place_ang_vel_error"] = (
            torch.mean(mean_err[rotate_mask]) if rotate_mask.any() else torch.zeros((), device=self.device))
        self.rotate_ang_vel_err_sum[env_ids] = 0.
        self.rotate_ang_vel_err_count[env_ids] = 0.

        # fraction of feet matching gait phase, counted only during rotate-in-place steps
        rotate_gait_mask = self.rotate_gait_match_count[env_ids] > 0
        mean_match = self.rotate_gait_match_sum[env_ids] / self.rotate_gait_match_count[env_ids].clamp(min=1)
        self.extras["episode"]["rotate_in_place_gait_match_frac"] = (
            torch.mean(mean_match[rotate_gait_mask]) if rotate_gait_mask.any() else torch.zeros((), device=self.device))
        self.rotate_gait_match_sum[env_ids] = 0.
        self.rotate_gait_match_count[env_ids] = 0.

    def check_termination(self):
        super().check_termination()
        if self.cfg.terrain.u_shape_playground:
            return
        # end the episode as soon as this ridge's climb is done, instead of letting the robot
        # keep walking forward into the next (harder) ridge's territory before its level updates -
        # exactly the same forward-progress check _update_terrain_curriculum uses for move_up.
        # Also end it if it's wandered off the sides of its own assigned tile in y. Neither is a
        # failure, so both get folded into time_out_buf too (exempts them from _reward_termination).
        forward_distance = self.root_states[:, 0] - self.env_origins[:, 0]
        finished_climb = (forward_distance > self.terrain.env_length / 2) & self.had_walking_command
        lateral_distance = torch.abs(self.root_states[:, 1] - self.env_origins[:, 1])
        off_tile = lateral_distance > self.terrain.env_width / 2
        early_stop = finished_climb | off_tile
        self.reset_buf |= early_stop
        self.time_out_buf |= early_stop

    def _first_step_height(self, terrain_levels):
        # height of just the first stair riser at this difficulty - must match
        # _build_wave_stairs_terrain's step_height formula
        difficulty = terrain_levels.float() / self.cfg.terrain.num_rows
        return 0.05 + 0.278 * difficulty

    def _update_terrain_curriculum(self, env_ids):
        """ Moves each env to a harder terrain row on forward (+x) progress past half the ridge
            length (terrain.env_length), or an easier one if it didn't even clear the first stair
            riser (height change < one step_height) - much more lenient than requiring full
            distance, so a slow-but-real attempt at a harder level isn't instantly bounced back
            down. Envs that never sampled a walking command this episode are excluded.
        """
        if not self.init_done:
            return
        forward_distance = self.root_states[env_ids, 0] - self.env_origins[env_ids, 0]
        height_change = torch.abs(self.root_states[env_ids, 2] - self.env_origins[env_ids, 2])
        had_walking_command = self.had_walking_command[env_ids]

        move_up = (forward_distance > self.terrain.env_length / 2) & had_walking_command
        didnt_clear_first_step = height_change < self._first_step_height(self.terrain_levels[env_ids])
        move_down = didnt_clear_first_step & ~move_up & had_walking_command

        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )
        self.env_origins[env_ids] = self._terrain_spawn_origins(self.terrain_levels[env_ids], self.terrain_types[env_ids])

    def _init_height_points(self):
        y = torch.tensor(self.cfg.height_scan.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.height_scan.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)
        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self):
        # nearest-cell (round) lookup - the old floor+min-of-neighbors under-reported height on descending stairs
        world_points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) \
            + self.root_states[:, :3].unsqueeze(1)
        points = world_points + self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).round().long()
        px = torch.clip(points[:, :, 0].view(-1), 0, self.height_samples.shape[0] - 1)
        py = torch.clip(points[:, :, 1].view(-1), 0, self.height_samples.shape[1] - 1)

        heights = self.height_samples[px, py]
        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _draw_debug_vis(self):
        """ Draws a sphere at every height-scan point: red on a stair-edge cell, green otherwise.
            Recomputed fresh from the current root/base state every call, so it never draws one
            step's spheres against a different (post-reset) pose.
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

        # also draw the 4 feet using _reward_feet_edge's exact lookup, so it can be checked visually
        feet_at_edge = self._feet_at_edge().cpu().numpy()
        feet_xy = self.feet_pos[:, :, :2].cpu().numpy()
        feet_z = self.feet_pos[:, :, 2].cpu().numpy()
        foot_green = gymutil.WireframeSphereGeometry(0.03, 8, 8, None, color=(0, 1, 0))
        foot_red = gymutil.WireframeSphereGeometry(0.045, 8, 8, None, color=(1, 0, 0))
        for i in range(self.num_envs):
            for k in range(feet_at_edge.shape[1]):
                pose = gymapi.Transform(gymapi.Vec3(float(feet_xy[i, k, 0]), float(feet_xy[i, k, 1]), float(feet_z[i, k])))
                geom = foot_red if feet_at_edge[i, k] else foot_green
                gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[i], pose)

    def _feet_at_edge(self):
        """ Which feet sit on an x_edge_mask cell, regardless of contact - shared by
            _reward_feet_edge and the debug-vis markers above.
        """
        feet_pos_xy = ((self.feet_pos[:, :, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).long()
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0] - 1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1] - 1)
        return self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]

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
        """ Same as BaseTask.render(), but debug lines are (re)drawn right before draw_viewer()
            instead of a step earlier in post_physics_step(), which left a stale line buffer that
            caused visible flicker.
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
        """ Mounts a depth camera on env 0 only - interactive use in play_keyboard.py, never
            enabled during training (cfg.camera.use_camera is always False otherwise).
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

        # monitoring-only accumulators for extras["episode"] (see reset_idx), no reward attached
        self.stand_still_height_err_sum = torch.zeros(self.num_envs, device=self.device)
        self.stand_still_height_err_count = torch.zeros(self.num_envs, device=self.device)
        self.rotate_ang_vel_err_sum = torch.zeros(self.num_envs, device=self.device)
        self.rotate_ang_vel_err_count = torch.zeros(self.num_envs, device=self.device)
        self.rotate_gait_match_sum = torch.zeros(self.num_envs, device=self.device)
        self.rotate_gait_match_count = torch.zeros(self.num_envs, device=self.device)

        # tracks whether this episode ever sampled a walking command (see _update_terrain_curriculum)
        self.had_walking_command = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _post_physics_step_callback(self):
        self._update_gait_phase()
        self.update_feet_state()
        # NOTE: uses self.last_contacts before _reward_feet_air_time overwrites it this step
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

        # resamples commands (may set a fresh heading target for envs just entering "walking")
        result = super()._post_physics_step_callback()

        # heading-command tracking, walking envs only - turns the sampled heading target
        # (commands[:,3]) into vyaw (commands[:,2]) every step, base-class style, but scoped so
        # stand-still/rotate-in-place envs keep their own directly-sampled vyaw untouched
        is_standing_still, is_rotating_in_place, is_walking = self._command_mode()
        if torch.any(is_walking):
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            target_ang_vel = torch.clip(
                0.5 * wrap_to_pi(self.commands[:, 3] - heading),
                self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1],
            )
            self.commands[is_walking, 2] = target_ang_vel[is_walking]

        base_height_error = torch.abs(self._stance_relative_base_height() - self.cfg.rewards.base_height_target)
        self.stand_still_height_err_sum += base_height_error * is_standing_still
        self.stand_still_height_err_count += is_standing_still.float()

        ang_vel_error = torch.abs(self.base_ang_vel[:, 2] - self.commands[:, 2])
        self.rotate_ang_vel_err_sum += ang_vel_error * is_rotating_in_place
        self.rotate_ang_vel_err_count += is_rotating_in_place.float()

        gait_match_frac = self._gait_phase_match_count() / len(self.feet_indices)
        self.rotate_gait_match_sum += gait_match_frac * is_rotating_in_place
        self.rotate_gait_match_count += is_rotating_in_place.float()

        return result

    def _command_mode(self):
        """ Which command_proportions category is active, derived fresh from self.commands - shared
            by the monitoring accumulators and the mode one-hot in compute_observations.
            Returns (is_standing_still, is_rotating_in_place, is_walking), each (num_envs,) bool.
        """
        is_standing_still = torch.all(torch.abs(self.commands[:, :3]) < 1e-6, dim=1)
        is_rotating_in_place = (torch.abs(self.commands[:, 0]) < 1e-6) & (torch.abs(self.commands[:, 1]) < 1e-6) \
            & (torch.abs(self.commands[:, 2]) >= 1e-6)
        is_walking = ~is_standing_still & ~is_rotating_in_place
        return is_standing_still, is_rotating_in_place, is_walking

    def _resample_commands(self, env_ids):
        """ Draws each env's command from command_proportions (switching to command_curriculum's
            late_proportions once past switch_iteration, see there): stand still, rotate in place
            (vyaw floored so it never decays to ~0), or normal walking. Walking samples a heading
            target (commands[:,3], centered on the climb direction) instead of vyaw directly -
            _post_physics_step_callback turns that into vyaw every step, base-class style, but
            scoped to walking envs only so stand-still/rotate-in-place keep their own direct vyaw.
        """
        curr_cfg = self.cfg.commands.command_curriculum
        current_iteration = self.common_step_counter // curr_cfg.num_steps_per_env
        active_proportions = (curr_cfg.late_proportions if current_iteration >= curr_cfg.switch_iteration
                               else self.cfg.commands.command_proportions)
        proportions = torch.cumsum(torch.tensor(active_proportions, device=self.device), dim=0)
        choice = torch.rand(len(env_ids), device=self.device) * proportions[-1]
        stand_ids = env_ids[choice < proportions[0]]
        rotate_ids = env_ids[(choice >= proportions[0]) & (choice < proportions[1])]
        normal_ids = env_ids[choice >= proportions[1]]

        self.commands[stand_ids, :3] = 0.

        if len(rotate_ids) > 0:
            vyaw_min = self.cfg.commands.rotate_in_place_ang_vel_min
            vyaw_max = self.command_ranges["ang_vel_yaw"][1]
            mag = torch_rand_float(vyaw_min, vyaw_max, (len(rotate_ids), 1), device=self.device).squeeze(1)
            sign = torch.sign(torch_rand_float(-1., 1., (len(rotate_ids), 1), device=self.device).squeeze(1))
            self.commands[rotate_ids, 0] = 0.
            self.commands[rotate_ids, 1] = 0.
            self.commands[rotate_ids, 2] = mag * sign

        if len(normal_ids) > 0:
            self.commands[normal_ids, 0] = torch_rand_float(
                self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(normal_ids), 1), device=self.device).squeeze(1)
            self.commands[normal_ids, 1] = torch_rand_float(
                self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(normal_ids), 1), device=self.device).squeeze(1)
            self.commands[normal_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(normal_ids), 1), device=self.device).squeeze(1)

            self.commands[normal_ids, 0] *= torch.abs(self.commands[normal_ids, 0]) > 0.1

        # heading (not a one-shot vyaw) drives walking now, so _post_physics_step_callback keeps
        # steering toward the climb direction every step regardless of the heading at sample
        # time - no need to also require "already facing the right way" here, just that a
        # walking command was sampled at all this episode.
        _, _, is_walking = self._command_mode()
        self.had_walking_command[env_ids] |= is_walking[env_ids]

    def _update_gait_phase(self):
        """ Trotting gait phase for the diagonal leg pairs (FL+RR / FR+RL).
            feet_indices order is [FL, FR, RL, RR] (URDF body order filtered by foot_name).
        """
        cycle_time = self.cfg.rewards.cycle_time
        phase = (self.episode_length_buf * self.dt) % cycle_time / cycle_time
        self.phase = phase  # stashed for compute_observations' sin/cos phase encoding below
        phase_offsets = torch.tensor([0.0, 0.5, 0.5, 0.0], device=self.device)
        self.leg_phase = (phase.unsqueeze(1) + phase_offsets.unsqueeze(0)) % 1.0
        # only enforce the alternating gait when a non-zero lin/ang vel command is given
        self.gait_enabled = torch.any(self.commands[:, :3] != 0., dim=1)

    def compute_observations(self):
        # raw height scan (not encoded) - the encoder is trainable, lives in ActorCriticHeightEncoder
        height_scan = torch.clip(
            self.root_states[:, 2].unsqueeze(1) - self.cfg.rewards.base_height_target - self.measured_heights,
            -1, 1.,
        ) * self.obs_scales.height_measurements
        if self.add_noise:
            noise_scale = (self.cfg.noise.noise_scales.height_measurements
                           * self.cfg.noise.noise_level * self.obs_scales.height_measurements)
            height_scan += (2 * torch.rand_like(height_scan) - 1) * noise_scale

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
        # noise_scale_vec is zero-padded over the height-scan tail (noise already added above)
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _stance_relative_base_height(self):
        # base height relative to stance feet (not world z=0), since stairs terrain isn't flat
        contact = (self.contact_forces[:, self.feet_indices, 2] > 1.).float()
        contact_count = contact.sum(dim=1).clamp(min=1)
        stance_height = (self.feet_pos[:, :, 2] * contact).sum(dim=1) / contact_count
        return self.root_states[:, 2] - stance_height

    def _reward_base_height(self):
        # penalize base height error, measured relative to stance feet not world z=0
        return torch.square(self._stance_relative_base_height() - self.cfg.rewards.base_height_target)

    def _reward_feet_air_time(self):
        # same as base class, but gated on gait_enabled (any vx/vy/vyaw) not just lin commands, so rotate-in-place still rewards lifting a foot
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)
        rew_airTime *= self.gait_enabled
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _gait_phase_match_count(self):
        # feet count currently matching expected trot phase - shared by _reward_gait_phase and monitoring
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(len(self.feet_indices)):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1.
            res += ~(contact ^ is_stance)
        return res

    def _reward_gait_phase(self):
        # only rewarded while rotating in place (not walking, so the stair-climbing gait can adapt freely)
        _, is_rotating_in_place, _ = self._command_mode()
        return self._gait_phase_match_count() * is_rotating_in_place

    def _reward_feet_swing_height(self):
        # penalize swing feet missing the target clearance above local terrain, while gaited
        feet_pos_xy = ((self.feet_pos[:, :, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()  # (num_envs, 4, 2)
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.height_samples.shape[0] - 1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.height_samples.shape[1] - 1)
        terrain_height = self.height_samples[feet_pos_xy[..., 0], feet_pos_xy[..., 1]] * self.terrain.cfg.vertical_scale
        clearance = self.feet_pos[:, :, 2] - terrain_height

        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        # only penalize falling short of the target - clearing higher (e.g. a tall stair riser) is fine
        shortfall = torch.clamp(self.cfg.rewards.feet_swing_height_target - clearance, min=0.)
        swing_error = torch.square(shortfall) * (~contact)
        # only enforced while rotating in place (not walking, so the stair-climbing gait can adapt freely)
        _, is_rotating_in_place, _ = self._command_mode()
        return torch.sum(swing_error, dim=1) * is_rotating_in_place

    def _reward_stand_still(self):
        # penalize motion away from the default pose, but only when vx, vy AND vyaw are all zero
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (~self.gait_enabled)

    def _reward_stand_still_contact(self):
        # bonus for keeping all 4 feet planted while standing still, on top of the pose penalty
        all_feet_contact = torch.all(self.contact_forces[:, self.feet_indices, 2] > 1., dim=1)
        return all_feet_contact.float() * (~self.gait_enabled)

    def _reward_dof_pos_deviation(self):
        # penalize joint deviation from default at all times, not just when standing still
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_feet_edge(self):
        # penalize feet on a stair edge, only past the easiest terrain rows
        self.feet_at_edge = self.contact_filt & self._feet_at_edge()
        rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        return rew.float()

    def _reward_hip_pos(self):
        # penalize hip joints drifting from their default angle (keeps the stance width stable)
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)
