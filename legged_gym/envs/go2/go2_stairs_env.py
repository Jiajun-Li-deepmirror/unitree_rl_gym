import logging
import os

import numpy as np
import torch

from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import quat_apply, torch_rand_float

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.helpers import class_to_dict


class GO2Stairs(LeggedRobot):

    DEBUG_LOG_INTERVAL_STEPS = 500

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self._debug_logger = None
        self._height_scan_debug_info = None
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._height_scan_debug_info = self._build_height_scan_debug_info()
        self.edge_mask = self._compute_edge_mask()

    def _compute_edge_mask(self):
        if self.height_samples is None:
            return None
        hs = self.height_samples
        threshold = max(1, int(round(self.cfg.rewards.edge_height_threshold_m / self.cfg.terrain.vertical_scale)))

        x_edge = torch.zeros_like(hs, dtype=torch.bool)
        x_diff = (hs[1:, :] - hs[:-1, :]).abs() > threshold
        x_edge[:-1, :] |= x_diff
        x_edge[1:, :] |= x_diff

        y_edge = torch.zeros_like(hs, dtype=torch.bool)
        y_diff = (hs[:, 1:] - hs[:, :-1]).abs() > threshold
        y_edge[:, :-1] |= y_diff
        y_edge[:, 1:] |= y_diff

        return x_edge | y_edge

    def _lookup_edge_mask(self, xy_world):
        if self.edge_mask is None:
            return None
        idx = ((xy_world + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()
        idx_x = torch.clip(idx[..., 0], 0, self.edge_mask.shape[0] - 1)
        idx_y = torch.clip(idx[..., 1], 0, self.edge_mask.shape[1] - 1)
        return self.edge_mask[idx_x, idx_y]

    def init_debug_logger(self, log_dir):
        """Set up a per-run debug log file at <log_dir>/debug.log. Call this once
        the training log_dir is known (log_dir isn't available at env creation
        time, only once OnPolicyRunner is built)."""
        os.makedirs(log_dir, exist_ok=True)
        logger = logging.getLogger(f"go2_stairs_debug.{id(self)}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        for h in list(logger.handlers):
            logger.removeHandler(h)
        handler = logging.FileHandler(os.path.join(log_dir, 'debug.log'))
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logger.addHandler(handler)
        self._debug_logger = logger
        if self._height_scan_debug_info is not None:
            self._debug_logger.info(self._height_scan_debug_info)

    def _build_height_scan_debug_info(self):
        hp = self.height_points[0]
        x = hp[:, 0].cpu().numpy()
        y = hp[:, 1].cpu().numpy()
        return (f"[height_scan] num_points={hp.shape[0]} "
                f"x_range=[{x.min():.3f},{x.max():.3f}]m "
                f"y_half_width_range=[{np.abs(y).min():.3f},{np.abs(y).max():.3f}]m "
                f"height_scan_cfg={class_to_dict(self.cfg.height_scan)} "
                f"depth_camera_cfg={class_to_dict(self.cfg.depth_camera)}")

    def _init_buffers(self):
        super()._init_buffers()
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # shape: num_envs, num_bodies, 13 (pos[0:3], quat[3:7], lin_vel[7:10], ang_vel[10:13])
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)

    def _create_envs(self):
        """Same as the base class, plus (only when cfg.depth_camera.use_camera):
        one depth camera per env, rigidly attached to the "base" body. This is
        prep for the depth+GRU student -- not wired into observations/reward
        anywhere yet. self.camera_handles is always defined (empty when the
        camera is off) so other code can check it unconditionally."""
        super()._create_envs()
        self.camera_handles = []
        if not self.cfg.depth_camera.use_camera:
            return

        cam = self.cfg.depth_camera
        # proportional downscale from the native datasheet resolution -- keeps the
        # width/height ratio (and thus the horizontal_fov/vertical_fov mapping
        # above) correct while cutting render cost/GPU memory per env.
        self.camera_image_width = max(1, int(round(cam.native_image_width * cam.resolution_scale)))
        self.camera_image_height = max(1, int(round(cam.native_image_height * cam.resolution_scale)))
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.camera_image_width
        camera_props.height = self.camera_image_height
        camera_props.horizontal_fov = cam.horizontal_fov
        camera_props.near_plane = cam.min_range
        camera_props.far_plane = cam.max_range
        camera_props.enable_tensors = True

        local_transform = gymapi.Transform()
        # z offset relative to the base body origin: mount_height is "camera
        # height above ground in nominal stance", and base_height_target (our own
        # reward's nominal standing height) is the least fragile reference we have
        # for how high the base origin sits above the ground -- init_state.pos[2]
        # is just the pre-settling spawn height, not the equilibrium standing
        # height, so it can't be used for this the way it might look like it should.
        local_transform.p = gymapi.Vec3(cam.mount_forward_offset, 0.0,
                                         cam.mount_height - self.cfg.rewards.base_height_target)
        # Matches the isaacgym docs' attach_camera_to_body example convention
        # (rotate about local +Y). NOT yet visually verified on this machine
        # (GPU was busy) -- once a viewer is available, confirm positive
        # camera_pitch_deg actually tilts the view down toward the ground and
        # flip the sign here if it instead tilts up.
        #
        # This is independent from the height-scan's geometry (see
        # _init_height_points): the height-scan is now a fixed near-field trapezoid
        # (not derived from this camera transform at all), so this pitch only
        # affects what the real depth image shows, e.g. trading close-up ground
        # detail for farther look-ahead -- tune once there's visual feedback on
        # what the depth image actually looks like.
        local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(cam.camera_pitch_deg))

        for i in range(self.num_envs):
            camera_handle = self.gym.create_camera_sensor(self.envs[i], camera_props)
            base_body_handle = self.gym.find_actor_rigid_body_handle(self.envs[i], self.actor_handles[i], "base")
            self.gym.attach_camera_to_body(camera_handle, self.envs[i], base_body_handle,
                                            local_transform, gymapi.FOLLOW_TRANSFORM)
            self.camera_handles.append(camera_handle)

    def render(self, sync_frame_time=True):
        super().render(sync_frame_time=sync_frame_time)
        if self.cfg.depth_camera.use_camera and not self.headless:
            self._show_depth_camera_view()

    def _show_depth_camera_view(self):
        """Render camera sensors and pop up env 0's depth image in a cv2 window.
        Only runs with a viewer (not headless): in headless mode base_task.py
        sets graphics_device_id=-1 (no graphics context at all), so camera
        sensors can't render there -- wiring depth into actual headless student
        training will need that addressed first. This is prep/visualization
        only, matching "don't add it to training yet"."""
        try:
            import cv2
        except ImportError:
            return
        # _draw_debug_vis's height-scan/foot spheres (drawn via gym.add_lines) are
        # actual scene-space geometry, not a viewer-only HUD overlay -- they linger
        # from the previous step's post_physics_step call and would otherwise show
        # up as phantom floating objects in the depth image. Clear them right before
        # the camera renders; the interactive viewer already drew this frame's debug
        # vis before we get here (see render()), so this doesn't affect what you see
        # in the viewer, only what the depth camera captures.
        self.gym.clear_lines(self.viewer)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        depth = gymtorch.wrap_tensor(
            self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], self.camera_handles[0], gymapi.IMAGE_DEPTH))
        cam = self.cfg.depth_camera
        # IMAGE_DEPTH gives negative distance-to-pixel in meters; flip to positive
        # and normalize [min_range, max_range] -> [0, 255] for display.
        d = torch.clamp(depth, min=-cam.max_range, max=-cam.min_range)
        d = -d
        normalized = (d - cam.min_range) / max(cam.max_range - cam.min_range, 1e-6) * 255.0
        img = normalized.cpu().numpy().astype(np.uint8)
        self.gym.end_access_image_tensors(self.sim)
        cv2.imshow("go2_stairs depth (env 0)", img)
        cv2.waitKey(1)

    def _post_physics_step_callback(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        super()._post_physics_step_callback()
        if self._debug_logger is not None and self.common_step_counter % self.DEBUG_LOG_INTERVAL_STEPS == 0:
            self._log_training_diagnostics()

    def _reward_no_fly(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        num_contact = torch.sum(contact.float(), dim=1)
        return (num_contact >= 2).float()

    def _reward_feet_edge(self):
        feet_at_edge = self._lookup_edge_mask(self.rigid_body_state[:, self.feet_indices, :2])
        if feet_at_edge is None:
            return torch.zeros(self.num_envs, device=self.device)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        feet_at_edge = contact & feet_at_edge
        return (self.terrain_levels > 3).float() * torch.sum(feet_at_edge, dim=-1).float()

    def _reward_base_height(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        feet_z = self.rigid_body_state[:, self.feet_indices, 2]
        num_contact = torch.sum(contact.float(), dim=1)
        ground_z = torch.sum(feet_z * contact, dim=1) / num_contact.clamp(min=1)
        # if no foot is in contact this instant, fall back to the last valid
        # estimate implicitly by zeroing the penalty rather than dividing by 0
        base_height = self.root_states[:, 2] - ground_z
        return torch.square(base_height - self.cfg.rewards.base_height_target) * (num_contact > 0)

    def _draw_debug_vis(self):
        """Same height-scan point cloud as the base class's visualization, plus:
          - each height-scan point turns red instead of yellow when it lands on
            an edge_mask cell, so the edge classification is visible directly
            over the local terrain patch each robot is scanning.
          - each foot gets its own marker: orange when _reward_feet_edge would
            currently penalize it (in contact AND on an edge cell), green
            otherwise. Uses the same _lookup_edge_mask() the reward itself
            uses, so the visualization can't silently drift from what's
            actually being penalized.
        """
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        normal_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        edge_geom = gymutil.WireframeSphereGeometry(0.03, 4, 4, None, color=(1, 0, 0))
        foot_ok_geom = gymutil.WireframeSphereGeometry(0.03, 6, 6, None, color=(0, 1, 0))
        foot_edge_geom = gymutil.WireframeSphereGeometry(0.045, 6, 6, None, color=(1, 0.3, 0))

        world_points = quat_apply(self.base_quat.repeat(1, self.num_height_points), self.height_points) \
                       + (self.root_states[:, :3]).unsqueeze(1)
        point_is_edge = self._lookup_edge_mask(world_points[..., :2])

        foot_xy = self.rigid_body_state[:, self.feet_indices, :2]
        foot_z = self.rigid_body_state[:, self.feet_indices, 2]
        foot_is_edge = self._lookup_edge_mask(foot_xy)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        foot_penalized = (contact & foot_is_edge) if foot_is_edge is not None else torch.zeros_like(contact)

        world_points_np = world_points.cpu().numpy()
        point_is_edge_np = point_is_edge.cpu().numpy() if point_is_edge is not None else None
        foot_xy_np = foot_xy.cpu().numpy()
        foot_z_np = foot_z.cpu().numpy()
        foot_penalized_np = foot_penalized.cpu().numpy()
        measured_heights_np = self.measured_heights.cpu().numpy()

        for i in range(self.num_envs):
            for j in range(self.num_height_points):
                x, y = world_points_np[i, j, 0], world_points_np[i, j, 1]
                z = measured_heights_np[i, j]
                geom = edge_geom if (point_is_edge_np is not None and point_is_edge_np[i, j]) else normal_geom
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[i], sphere_pose)

            for k in range(len(self.feet_indices)):
                x, y = foot_xy_np[i, k]
                z = foot_z_np[i, k]
                geom = foot_edge_geom if foot_penalized_np[i, k] else foot_ok_geom
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _reward_feet_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        foot_xy_vel = torch.norm(self.rigid_body_state[:, self.feet_indices, 7:9], dim=2)
        return torch.sum(contact * torch.square(foot_xy_vel), dim=1)

    def _log_training_diagnostics(self):
        vx = self.commands[:, 0]
        vx_zero_fraction = (vx.abs() < 1e-6).float().mean().item()
        heading_or_yaw = self.commands[:, 3] if self.cfg.commands.heading_command else self.commands[:, 2]
        self._debug_logger.info(
            f"[step {self.common_step_counter}] "
            f"terrain_levels mean={self.terrain_levels.float().mean().item():.2f} "
            f"std={self.terrain_levels.float().std().item():.2f} "
            f"lin_vel_x_range={self.command_ranges['lin_vel_x']} "
            f"vx_zero_fraction={vx_zero_fraction:.3f} "
            f"heading_sampled_range=[{heading_or_yaw.min().item():.2f},{heading_or_yaw.max().item():.2f}]"
        )

    def _init_height_points(self):
        """Fixed trapezoid, not a camera-FOV ground projection: short (near) edge
        at x=near_edge_x with half-width near_edge_half_width (the front-foot
        stance width), long (far) edge depth_range farther out, side edges
        angled by depth_camera.horizontal_fov/2 (the only thing still shared
        with the camera config -- see the height_scan config comment). Covers
        less area than the real camera can see (some of it outside the
        camera's FOV/range entirely) in exchange for much denser sampling
        where footstep decisions actually happen."""
        hs = self.cfg.height_scan
        half_hfov = np.deg2rad(self.cfg.depth_camera.horizontal_fov) / 2
        near_x = hs.near_edge_x
        far_x = near_x + hs.depth_range

        x_rows = np.linspace(near_x, far_x, hs.n_forward)

        xs, ys = [], []
        for x in x_rows:
            depth_from_near_edge = x - near_x
            y_half = hs.near_edge_half_width + depth_from_near_edge * np.tan(half_hfov)
            y_row = np.linspace(-y_half, y_half, hs.n_lateral)
            xs.extend([x] * hs.n_lateral)
            ys.extend(y_row.tolist())

        x_t = torch.tensor(xs, device=self.device, dtype=torch.float, requires_grad=False)
        y_t = torch.tensor(ys, device=self.device, dtype=torch.float, requires_grad=False)

        self.num_height_points = x_t.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = x_t
        points[:, :, 1] = y_t
        return points

    def _get_heights(self, env_ids=None):
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) \
                     + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply(self.base_quat.repeat(1, self.num_height_points), self.height_points) \
                     + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def update_command_curriculum(self, env_ids):
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = 0.
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    def _resample_commands(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0], self.command_ranges["heading"][1],
                (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1],
                (len(env_ids), 1), device=self.device).squeeze(1)

        # Hardware deadzone: exact vx cutoff, independent of heading/ang_vel channels.
        self.commands[env_ids, 0] *= (torch.abs(self.commands[env_ids, 0]) > 0.1)
