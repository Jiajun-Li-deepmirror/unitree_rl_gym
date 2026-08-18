import os

import numpy as np

# isaacgym must be imported before torch in this process
from isaacgym import gymapi, gymtorch, terrain_utils
from isaacgym.torch_utils import quat_apply, quat_from_angle_axis, to_torch, torch_rand_float

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math import wrap_to_pi


class GO2StairsRobot(LeggedRobot):
    """Go2 on stairs/staircase terrain.

    unitree_rl_gym's base ``LeggedRobot`` never wires up heightfield/trimesh
    terrain (``create_sim`` only ever builds a flat ground plane, and
    ``utils/terrain.py``'s ``Terrain`` class is unused dead code in this
    fork). So this class builds its own trimesh terrain from scratch:

      - flat ground (with a little height noise)
      - a short flat-up-flat-down-flat staircase, in 2-step and 3-step
        variants, open on both sides so it can be crossed at an angle
      - an enclosed, U-shaped switchback staircase (two 12-step flights
        joined by a landing, three walls, one open central shaft) meant
        to stand in for a real fire-escape-style stairwell

    Everything below is a new method added on this subclass; nothing in
    envs/base/*.py or envs/go2/*.py is touched.
    """

    # ------------------------------------------------------------------
    # sim / terrain creation
    # ------------------------------------------------------------------
    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'trimesh':
            self._build_terrain()
            self._create_trimesh()
        else:
            raise ValueError(f"GO2StairsRobot only supports terrain.mesh_type in ['plane', 'trimesh'], got '{mesh_type}'")
        self._create_envs()

    def _build_terrain(self):
        """Builds the full height-field grid: one column per terrain type, one row per difficulty."""
        cfg = self.cfg.terrain
        self.horizontal_scale = cfg.horizontal_scale
        self.vertical_scale = cfg.vertical_scale

        generators = [
            self._gen_flat,
            lambda difficulty: self._gen_short_stairs(difficulty, num_steps=2),
            lambda difficulty: self._gen_short_stairs(difficulty, num_steps=3),
            self._gen_u_staircase,
        ]
        assert len(generators) == cfg.num_cols == len(cfg.terrain_types), \
            "terrain.num_cols / terrain.terrain_types must match the number of generator functions"

        self.length_per_env_pixels = self._m2px(cfg.terrain_length)
        self.width_per_env_pixels = self._m2px(cfg.terrain_width)
        self.border = self._m2px(cfg.border_size)
        self.tot_rows = cfg.num_rows * self.length_per_env_pixels + 2 * self.border
        self.tot_cols = cfg.num_cols * self.width_per_env_pixels + 2 * self.border

        height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3), dtype=np.float32)

        for i in range(cfg.num_rows):
            difficulty = i / max(cfg.num_rows - 1, 1)
            for j in range(cfg.num_cols):
                tile, spawn_xyz = generators[j](difficulty)
                assert tile.shape == (self.length_per_env_pixels, self.width_per_env_pixels), \
                    f"terrain generator {cfg.terrain_types[j]} returned shape {tile.shape}, " \
                    f"expected {(self.length_per_env_pixels, self.width_per_env_pixels)}"
                row0 = self.border + i * self.length_per_env_pixels
                col0 = self.border + j * self.width_per_env_pixels
                height_field_raw[row0:row0 + self.length_per_env_pixels, col0:col0 + self.width_per_env_pixels] = tile
                env_origins[i, j] = [
                    i * cfg.terrain_length + spawn_xyz[0],
                    j * cfg.terrain_width + spawn_xyz[1],
                    spawn_xyz[2],
                ]

        self.height_field_raw = height_field_raw
        self.env_origins_grid = env_origins

    def _create_trimesh(self):
        """Adds one static triangle-mesh actor PER (row, col) tile, plus 4 for the border
        strips around the whole grid -- deliberately NOT one giant actor spanning the full
        grid (that was the original, simpler implementation).

        A single monolithic mesh here measurably breaks GPU PhysX: at num_envs>=2048 with
        this task's ~700-900k-triangle combined grid, the GPU broadphase logs
        "PxgDynamicsMemoryConfig::foundLostAggregatePairsCapacity ... the simulation will
        miss interactions", and those missed interactions are exactly what caused robots to
        gradually sink through the floor over ~15-20 steps on 3 of 4 terrain-type columns
        (confirmed empirically: identical setup on CPU PhysX -- no GPU broadphase at all --
        never showed this; a fresh, never-trained network already showed it on GPU, ruling
        out any training/reward cause; and raising sim.physx.default_buffer_size_multiplier
        by 8x had zero measurable effect, ruling that specific knob out as the fix). Splitting
        into small per-tile pieces means a robot's broadphase queries only ever need to
        consider its own local piece instead of the whole grid.

        Only the PHYSICS/collision representation is split this way -- self.height_samples
        (used for the CPU-side height-scan/_sample_terrain_height_at queries) stays exactly
        the single combined array it always was, since that code path never touches PhysX
        broadphase and splitting it would need every caller to know which piece to look in
        for no benefit.
        """
        cfg = self.cfg.terrain

        def add_piece(row0, row1, col0, col1):
            sub = self.height_field_raw[row0:row1, col0:col1]
            vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
                sub, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_treshold)
            tm_params = gymapi.TriangleMeshParams()
            tm_params.nb_vertices = vertices.shape[0]
            tm_params.nb_triangles = triangles.shape[0]
            # convert_heightfield_to_trimesh always builds LOCAL vertices starting at (0,0)
            # for whatever sub-array it's given, so the piece's own offset (row0, col0)
            # within the combined grid has to be added back in via the transform, on top of
            # the same -border_size shift the original single-mesh version used (world x=0
            # lines up with the first real tile, i.e. combined-grid row/col == self.border).
            tm_params.transform.p.x = row0 * cfg.horizontal_scale - cfg.border_size
            tm_params.transform.p.y = col0 * cfg.horizontal_scale - cfg.border_size
            tm_params.transform.p.z = 0.0
            tm_params.static_friction = cfg.static_friction
            tm_params.dynamic_friction = cfg.dynamic_friction
            tm_params.restitution = cfg.restitution
            self.gym.add_triangle_mesh(self.sim, vertices.flatten(order='C'), triangles.flatten(order='C'), tm_params)

        L, W, B = self.length_per_env_pixels, self.width_per_env_pixels, self.border
        tot_rows, tot_cols = self.height_field_raw.shape
        for i in range(cfg.num_rows):
            for j in range(cfg.num_cols):
                r0 = B + i * L
                c0 = B + j * W
                add_piece(r0, r0 + L, c0, c0 + W)
        # border strips (flat padding around the whole tile grid): bottom/top span the full
        # width including corners; left/right only span the tile rows, to avoid re-adding
        # the corners a second time
        add_piece(0, B, 0, tot_cols)
        add_piece(tot_rows - B, tot_rows, 0, tot_cols)
        add_piece(B, tot_rows - B, 0, B)
        add_piece(B, tot_rows - B, tot_cols - B, tot_cols)

        self.height_samples = torch.tensor(self.height_field_raw, device=self.device)

    # ------------------------------------------------------------------
    # pixel helpers
    # ------------------------------------------------------------------
    def _m2px(self, meters):
        """Meters -> pixels, asserting the conversion is exact (no silent rounding of stair geometry)."""
        px = meters / self.horizontal_scale
        px_rounded = int(round(px))
        assert abs(px - px_rounded) < 1e-6, \
            f"{meters} m is not an exact multiple of horizontal_scale={self.horizontal_scale} m"
        return px_rounded

    def _h2u(self, meters):
        """Meters of height -> int16 height-field units."""
        return int(round(meters / self.vertical_scale))

    def _step_height_at(self, difficulty):
        """Nominal per-step rise at this difficulty: linearly interpolates
        terrain.step_height_range, so the easiest row is a shorter, more
        forgiving riser and only the hardest row is the full real-world spec."""
        lo, hi = self.cfg.terrain.step_height_range
        return lo + difficulty * (hi - lo)

    def _u_staircase_y_offset(self):
        """Meters of flat padding on the u_staircase's near (low-y) side, centering its
        own (wall+corridor+shaft+corridor+wall) structure within the wider tile. Shared by
        _gen_u_staircase (terrain) and _init_buffers (waypoint chain) so they always agree."""
        cfg = self.cfg.terrain
        structure_width = 2 * cfg.wall_thickness + 2 * cfg.stair_width + cfg.shaft_width
        return (cfg.terrain_width - structure_width) / 2

    # ------------------------------------------------------------------
    # terrain generators -- each returns (height_field_tile[int16], (spawn_x, spawn_y, spawn_z) in tile-local meters)
    # ------------------------------------------------------------------
    def _gen_flat(self, difficulty):
        cfg = self.cfg.terrain
        tile = np.zeros((self.length_per_env_pixels, self.width_per_env_pixels), dtype=np.int16)
        noise_m = cfg.flat_max_height_noise * difficulty
        if noise_m > 0:
            noise_units = self._h2u(noise_m)
            tile += np.random.randint(-noise_units, noise_units + 1, size=tile.shape).astype(np.int16)
        spawn = (cfg.terrain_length / 2, cfg.terrain_width / 2, 0.0)
        return tile, spawn

    def _gen_short_stairs(self, difficulty, num_steps):
        """flat lead-in -> up num_steps -> flat platform -> down num_steps -> flat run-out,
        spanning the tile's FULL width (no walls, no side margins) so a randomized spawn
        yaw (see _create_envs/_reset_root_states) can't drift the robot off the stairs
        before it finishes crossing at an angle -- the whole tile width is walkable stair.

        The lead-in/run-out length is fixed; the platform absorbs whatever tile length is
        left over so the pattern fills the whole 6.0m tile instead of leaving a dead flat
        strip past the down-flight.
        """
        cfg = self.cfg.terrain
        tile = np.zeros((self.length_per_env_pixels, self.width_per_env_pixels), dtype=np.int16)
        noise_max = cfg.max_step_height_noise * difficulty
        step_height = self._step_height_at(difficulty)

        step_depth_px = self._m2px(cfg.step_depth)
        lead_in_px = self._m2px(cfg.short_stairs_lead_in)

        platform_px = self.length_per_env_pixels - 2 * lead_in_px - 2 * num_steps * step_depth_px
        assert platform_px > 0, (
            f"short_stairs (num_steps={num_steps}) doesn't fit in a {cfg.terrain_length}m tile: "
            f"2 x lead_in ({cfg.short_stairs_lead_in}m) + 2 x {num_steps} steps ({cfg.step_depth}m each) "
            f"already exceeds it -- shrink short_stairs_lead_in or grow terrain_length"
        )

        x = lead_in_px
        cum_height = 0.0
        for _ in range(num_steps):
            rise = step_height + np.random.uniform(-noise_max, noise_max)
            cum_height += rise
            tile[x:x + step_depth_px, :] = self._h2u(cum_height)
            x += step_depth_px

        tile[x:x + platform_px, :] = self._h2u(cum_height)
        x += platform_px

        down_height = cum_height
        for k in range(num_steps):
            if k == num_steps - 1:
                down_height = 0.0  # snap the last step so the run-out is exactly flat
            else:
                drop = step_height + np.random.uniform(-noise_max, noise_max)
                down_height = max(down_height - drop, 0.0)
            tile[x:x + step_depth_px, :] = self._h2u(down_height)
            x += step_depth_px

        # x now lands exactly on length_per_env_pixels (lead_in + steps + platform + steps == tile length)
        spawn = (cfg.short_stairs_lead_in / 2, cfg.terrain_width / 2, 0.0)
        return tile, spawn

    def _gen_u_staircase(self, difficulty):
        """Two 12-step flights side by side joined by a landing, 3 walls, open central shaft.

        Layout (x = direction of travel up flight A, y = lateral):
            y:  [wall | corridor A | shaft | corridor B | wall]
            x:  [entry platform | flight A (climbing) | landing] for corridor A / shaft
                [top exit platform | flight B (climbing back) | landing] for corridor B
        Flight A climbs as x increases; flight B climbs as x decreases (the switchback).
        The shaft is an open pit for the whole flight run and is bridged only under the landing.

        Both the outer/end-cap walls and the shaft's protective lip are
        written relative to the LOCAL floor height at each x (so they track
        the stair profile), and their size is difficulty-scaled in opposite
        directions:
          - walls: a low guard-board at difficulty 0 growing into a full
            enclosing wall by difficulty 1 (matches the earlier plan to not
            box the robot in on easy episodes)
          - shaft lip: a protective curb at difficulty 0 (so falling in
            isn't the dominant failure mode while it's still learning to
            climb) that narrows to nothing by difficulty 1, exposing the
            bare open shaft
        """
        cfg = self.cfg.terrain
        tile = np.zeros((self.length_per_env_pixels, self.width_per_env_pixels), dtype=np.int16)
        noise_max = cfg.max_step_height_noise * difficulty
        step_height = self._step_height_at(difficulty)
        guard_h = cfg.wall_height_range[0] + difficulty * (cfg.wall_height_range[1] - cfg.wall_height_range[0])
        curb_px = int(round(self._m2px(cfg.shaft_curb_width) * (1.0 - difficulty)))

        # the u_staircase's own structure (wall+corridor+shaft+corridor+wall) is narrower
        # than the tile now that terrain_width was widened for the short_stairs terrain's
        # sake -- center it, leaving flat padding on either side (harmless: it's outside
        # the walls either way)
        y_off_px = self._m2px(self._u_staircase_y_offset())
        wall_px = self._m2px(cfg.wall_thickness)
        corridor_px = self._m2px(cfg.stair_width)
        shaft_px = self._m2px(cfg.shaft_width)
        yA0 = y_off_px + wall_px
        yA1 = yA0 + corridor_px
        yS0 = yA1
        yS1 = yS0 + shaft_px
        yB0 = yS1
        yB1 = yB0 + corridor_px
        yWall2_1 = yB1 + wall_px
        assert yWall2_1 <= self.width_per_env_pixels, \
            f"u_staircase width layout ({yWall2_1}px) doesn't fit terrain_width ({self.width_per_env_pixels}px)"

        entry_px = self._m2px(cfg.landing_depth)
        step_px = self._m2px(cfg.step_depth)
        flight_px = cfg.u_staircase_num_steps * step_px
        landing_px = self._m2px(cfg.landing_depth)
        x_entry_end = entry_px
        x_flight_end = x_entry_end + flight_px
        x_landing_end = x_flight_end + landing_px
        assert x_landing_end == self.length_per_env_pixels, \
            f"u_staircase length layout ({x_landing_end}px) doesn't match terrain_length ({self.length_per_env_pixels}px)"

        # entry platform (corridor A), height 0, plus its outer wall and shaft lip
        tile[0:x_entry_end, yA0:yA1] = 0
        tile[0:x_entry_end, y_off_px:yA0] = self._h2u(0.0 + guard_h)
        if curb_px > 0:
            tile[0:x_entry_end, yS0:yS0 + curb_px] = self._h2u(0.0 + cfg.shaft_curb_height)

        # flight A: floor climbs from 0 to flightA_top in per-step blocks (real stair risers,
        # must stay stepped). The wall/shaft-lip bands alongside it are given a smooth
        # per-pixel RAMP instead of copying the same per-step blocks: those bands are only
        # a few pixels wide, so a stepped rise there stacks a second x-direction cliff right
        # next to the corridor's own x-cliff at every corner. convert_heightfield_to_trimesh's
        # slope-threshold vertex-shift correction (move_x/move_y/move_corners) then fires from
        # both directions on the same vertices and tears the mesh -- the jagged, gapped wall
        # geometry seen in the viewer. A smooth ramp has no x-cliff to correct, so it renders
        # as a clean sloped guard-rail (physically fine -- it doesn't need to mimic the stairs).
        cum = 0.0
        for k in range(cfg.u_staircase_num_steps):
            rise = step_height + np.random.uniform(-noise_max, noise_max)
            cum += rise
            x0 = x_entry_end + k * step_px
            x1 = x0 + step_px
            tile[x0:x1, yA0:yA1] = self._h2u(cum)
        flightA_top = cum
        wall_ramp_a = np.linspace(0.0, flightA_top, flight_px, endpoint=False)
        tile[x_entry_end:x_flight_end, y_off_px:yA0] = \
            np.round((wall_ramp_a + guard_h) / cfg.vertical_scale).astype(np.int16)[:, None]
        if curb_px > 0:
            tile[x_entry_end:x_flight_end, yS0:yS0 + curb_px] = \
                np.round((wall_ramp_a + cfg.shaft_curb_height) / cfg.vertical_scale).astype(np.int16)[:, None]

        # landing: bridges corridor A / shaft / corridor B, flat at flightA_top (already covers the shaft, no lip needed here)
        tile[x_flight_end:x_landing_end, yA0:yB1] = self._h2u(flightA_top)
        tile[x_flight_end:x_landing_end, y_off_px:yA0] = self._h2u(flightA_top + guard_h)
        tile[x_flight_end:x_landing_end, yB1:yWall2_1] = self._h2u(flightA_top + guard_h)

        # flight B: floor climbs from flightA_top (next to the landing) further up as x
        # decreases, in per-step blocks. Wall/shaft-lip bands again use a smooth ramp (see
        # flight A above for why) -- height decreases as x increases, from top_height at the
        # x_entry_end boundary down to flightA_top at the x_flight_end boundary.
        cum_b = flightA_top
        for k in range(cfg.u_staircase_num_steps):
            rise = step_height + np.random.uniform(-noise_max, noise_max)
            cum_b += rise
            x1 = x_flight_end - k * step_px
            x0 = x1 - step_px
            tile[x0:x1, yB0:yB1] = self._h2u(cum_b)
        top_height = cum_b
        wall_ramp_b = np.linspace(top_height, flightA_top, flight_px)
        tile[x_entry_end:x_flight_end, yB1:yWall2_1] = \
            np.round((wall_ramp_b + guard_h) / cfg.vertical_scale).astype(np.int16)[:, None]
        if curb_px > 0:
            tile[x_entry_end:x_flight_end, yS1 - curb_px:yS1] = \
                np.round((wall_ramp_b + cfg.shaft_curb_height) / cfg.vertical_scale).astype(np.int16)[:, None]

        # top exit platform (corridor B), flat at top_height, plus its outer wall and shaft lip
        tile[0:x_entry_end, yB0:yB1] = self._h2u(top_height)
        tile[0:x_entry_end, yB1:yWall2_1] = self._h2u(top_height + guard_h)
        if curb_px > 0:
            tile[0:x_entry_end, yS1 - curb_px:yS1] = self._h2u(top_height + cfg.shaft_curb_height)

        # open shaft: a deep pit everywhere except under the landing (bridged above) and under the two lips
        tile[0:x_flight_end, yS0 + curb_px:yS1 - curb_px] = self._h2u(-cfg.shaft_depth)

        # end-cap wall ("3rd wall") at the far edge, beyond the landing -- spans only the
        # u_staircase's own structure width (not the tile's flat side padding), and the
        # near/entry edge (x=0) is deliberately left open for the robot to walk in
        tile[x_landing_end - wall_px:x_landing_end, y_off_px:yWall2_1] = self._h2u(flightA_top + guard_h)

        spawn = (cfg.landing_depth / 2, self._u_staircase_y_offset() + cfg.wall_thickness + cfg.stair_width / 2, 0.0)
        return tile, spawn

    # ------------------------------------------------------------------
    # env placement (mirrors the terrain-curriculum logic that upstream
    # legged_gym has and this stripped-down fork does not)
    # ------------------------------------------------------------------
    def _get_env_origins(self):
        if self.cfg.terrain.mesh_type != 'trimesh':
            return super()._get_env_origins()

        cfg = self.cfg.terrain
        self.custom_origins = True
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        max_init_level = (cfg.num_rows - 1) if not cfg.curriculum else cfg.max_init_terrain_level
        self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
        self.terrain_types = torch.div(
            torch.arange(self.num_envs, device=self.device),
            (self.num_envs / cfg.num_cols), rounding_mode='floor').long()
        self.max_terrain_level = cfg.num_rows
        self.terrain_origins = torch.from_numpy(self.env_origins_grid).to(self.device).to(torch.float)
        self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]

        # world-frame bounds of each env's own assigned tile, used by
        # check_termination() to catch a robot wandering onto a
        # neighboring tile (wrong difficulty/geometry for its reward and,
        # on the u_staircase, its waypoint chain). MUST be refreshed
        # whenever terrain_levels changes (see _update_terrain_curriculum)
        # -- a promoted/demoted env moves to a different row, i.e. a
        # different x-range entirely, so stale bounds here would flag it
        # as "out of bounds" the instant it respawns in its new row.
        all_ids = torch.arange(self.num_envs, device=self.device)
        self.tile_x_min = torch.zeros(self.num_envs, device=self.device)
        self.tile_x_max = torch.zeros(self.num_envs, device=self.device)
        self.tile_y_min = torch.zeros(self.num_envs, device=self.device)
        self.tile_y_max = torch.zeros(self.num_envs, device=self.device)
        self._update_tile_bounds(all_ids)

    def _update_tile_bounds(self, env_ids):
        """(Re)computes tile_x_min/x_max/y_min/y_max for env_ids from their CURRENT
        terrain_levels/terrain_types. Called at startup for everyone, and again from
        _update_terrain_curriculum for whichever envs it just promoted/demoted."""
        cfg = self.cfg.terrain
        self.tile_x_min[env_ids] = self.terrain_levels[env_ids].float() * cfg.terrain_length
        self.tile_x_max[env_ids] = self.tile_x_min[env_ids] + cfg.terrain_length
        self.tile_y_min[env_ids] = self.terrain_types[env_ids].float() * cfg.terrain_width
        self.tile_y_max[env_ids] = self.tile_y_min[env_ids] + cfg.terrain_width

    def _init_buffers(self):
        super()._init_buffers()
        cfg = self.cfg.terrain

        self.hip_indices = torch.tensor(
            [i for i, n in enumerate(self.dof_names) if 'hip' in n], dtype=torch.long, device=self.device)

        self.u_staircase_col = cfg.terrain_types.index('u_staircase')
        self.on_u_staircase = self.terrain_types == self.u_staircase_col

        # waypoint chain along the u_staircase's walkable centerline, in
        # tile-local meters (identical for every difficulty row -- only
        # step HEIGHT varies with difficulty, not the xy footprint).
        #
        # Each straight segment between consecutive waypoints must be either
        # purely flat or exactly span one flight's rise -- never a mix of
        # the two, or the segment's implied slope is wrong (too shallow),
        # which either floats it above flat ground (harmless, invisible) or
        # buries it inside the rising stairs (very visible, wrong). That
        # ruled out a naive 4-point chain (entry-center -> flightA-top ->
        # landing-crossing -> exit-center): the entry/exit-center points sit
        # mid-platform rather than at the flat/rising boundary, and the
        # single crossing segment used to sit right at the void's edge
        # (x_flight_end) instead of safely inside the landing. Hence 8
        # points, one at every flat/rising boundary:
        #   wp0 entry platform center      (== spawn point)
        #   wp1 bottom of flight A          (flat -> rising boundary)
        #   wp2 top of flight A             (rising -> flat boundary, at the landing's near edge)
        #   wp3 mid-landing, corridor A y   (walk forward, away from the void's edge)
        #   wp4 mid-landing, corridor B y   (cross laterally, still mid-landing -- away from the void)
        #   wp5 back at the landing's near edge, corridor B y (about to start flight B)
        #   wp6 top of flight B             (flat -> flat boundary with the top platform)
        #   wp7 top exit platform center    (== goal)
        x_entry_end = cfg.landing_depth
        x_flight_end = x_entry_end + cfg.u_staircase_num_steps * cfg.step_depth
        mid_landing_x = x_flight_end + cfg.landing_depth / 2
        y_off = self._u_staircase_y_offset()  # centers the structure in the (now wider) tile
        corridor_a_y = y_off + cfg.wall_thickness + cfg.stair_width / 2
        corridor_b_y = y_off + cfg.wall_thickness + cfg.stair_width + cfg.shaft_width + cfg.stair_width / 2
        self.u_staircase_wp_local = to_torch([
            [x_entry_end / 2, corridor_a_y],
            [x_entry_end, corridor_a_y],
            [x_flight_end, corridor_a_y],
            [mid_landing_x, corridor_a_y],
            [mid_landing_x, corridor_b_y],
            [x_flight_end, corridor_b_y],
            [x_entry_end, corridor_b_y],
            [x_entry_end / 2, corridor_b_y],
        ], device=self.device)
        self.num_waypoints = self.u_staircase_wp_local.shape[0]
        self.waypoint_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # +1 = ascending (wp0->wp7, spawn at the bottom entry), -1 = descending (wp7->wp0,
        # spawn at the top exit) -- see _reset_u_staircase_waypoints. Meaningless/unused
        # for non-u_staircase envs.
        self.waypoint_dir = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        # this episode's actual starting waypoint index -- NOT always a true endpoint, see
        # _reset_u_staircase_waypoints' graduated-start curriculum. Needed by
        # _update_terrain_curriculum to measure progress relative to where THIS episode
        # actually started, not an assumed fixed 0/(num_waypoints-1).
        self.waypoint_start_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # set True for exactly one step, the step a waypoint is actually reached (not
        # every step it happens to already be sitting there) -- see
        # _post_physics_step_callback / _reward_waypoint_progress
        self.waypoint_advanced = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # dense potential-based shaping companion to the sparse waypoint_advanced bonus --
        # see _post_physics_step_callback for how these are maintained and
        # _reward_waypoint_dist_progress for why the sparse-only signal wasn't enough
        self.prev_waypoint_dist = torch.zeros(self.num_envs, device=self.device)
        self.waypoint_dist_delta = torch.zeros(self.num_envs, device=self.device)

        # rigid body state tensor (positions/velocities of EVERY body, not just the root) --
        # the base class never acquires this since it never needs per-foot world position;
        # we do, for the gait-phase swing-height reward and the feet_edge reward (both need
        # to know where each foot actually is, not just whether it's in contact)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, 13)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # gait clock: a single shared phase (trot pairs use a fixed offset per foot, see
        # _create_envs' foot_phase_offset), advanced only while |cmd| clears the dead
        # zone -- frozen while genuinely standing still, see _post_physics_step_callback
        self.gait_phase = torch.zeros(self.num_envs, device=self.device)

        self.height_points = self._init_height_points()
        self.num_height_points = self.height_points.shape[1]

        # per-episode outcome classification, set fresh every check_termination() call
        # and read right after (same step) by _accumulate_episode_stats -- see there
        self.fell_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.shaft_fall_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.oob_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reached_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._outcome_names = ['reached_goal', 'shaft_fall', 'oob_fail', 'fell', 'oob_benign', 'timeout']

    def _prepare_reward_function(self):
        """Overrides LeggedRobot._prepare_reward_function purely to size the detailed
        per-(terrain_type, difficulty_row) episode-logging accumulators (see
        get_and_reset_episode_report) right after self.episode_sums exists -- this runs
        AFTER _init_buffers in the base class's __init__ (episode_sums doesn't exist yet
        during _init_buffers), so it can't be done there.

        Kept as flat 1D tensors (bucket index = terrain_type*num_rows + row) purely so every
        scatter_add_ target in _accumulate_episode_stats is a plain contiguous tensor with no
        reshape/stride subtleties; get_and_reset_episode_report reshapes only when reading.
        """
        super()._prepare_reward_function()
        cfg = self.cfg.terrain
        self._reward_names = list(self.episode_sums.keys())
        num_buckets = cfg.num_cols * cfg.num_rows
        self.bucket_ep_count = torch.zeros(num_buckets, device=self.device)
        self.bucket_ep_len_sum = torch.zeros(num_buckets, device=self.device)
        self.bucket_outcome_count = torch.zeros(num_buckets * len(self._outcome_names), device=self.device)
        self.bucket_rew_sum = torch.zeros(num_buckets * len(self._reward_names), device=self.device)

    def _init_height_points(self):
        """(num_envs, num_points, 3) body-frame xyz offsets to sample terrain height at
        (z always 0 here; _get_heights rotates+translates these into world points)."""
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        num_points = grid_x.numel()
        points = torch.zeros(self.num_envs, num_points, 3, device=self.device)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _tile_of(self, world_xy):
        """(row, col) that world_xy (any leading shape, last dim 2) falls into. Deliberately
        UNCLAMPED: a point beyond the whole tile grid (out in the flat border margin next to
        an edge row/col) must come back as a row/col that can never equal a real own_row/
        own_col, so _height_scan_same_tile correctly masks it. Clamping here would instead
        snap it right back onto the edge tile's own index, making border-margin points at
        edge tiles indistinguishable from genuine same-tile points -- interior tiles never hit
        this because spilling past their bounds always lands on a real (different) neighbor."""
        cfg = self.cfg.terrain
        row = torch.floor(world_xy[..., 0] / cfg.terrain_length).long()
        col = torch.floor(world_xy[..., 1] / cfg.terrain_width).long()
        return row, col

    def _sample_terrain_height_at(self, world_xy):
        """Terrain height (world frame, meters) at arbitrary world xy points -- any leading
        shape, last dim 2. Does NOT do cross-tile masking (unlike _get_heights): callers
        that need it (feet_edge, gait swing height) only ever query points right at/near
        the robot's own feet, which are always within its own tile by construction."""
        cfg = self.cfg.terrain
        px_py = (world_xy + cfg.border_size) / cfg.horizontal_scale
        px = torch.clip(px_py[..., 0].long(), 0, self.height_samples.shape[0] - 2)
        py = torch.clip(px_py[..., 1].long(), 0, self.height_samples.shape[1] - 2)
        h1 = self.height_samples[px, py]
        h2 = self.height_samples[px + 1, py]
        h3 = self.height_samples[px, py + 1]
        return torch.min(torch.min(h1, h2), h3).float() * cfg.vertical_scale

    def _height_scan_world_xy(self, env_ids):
        """(len(env_ids), num_height_points, 2) world xy of the fixed body-frame height-scan
        grid (self.height_points), yaw-rotated + translated -- shared by _get_heights and the
        two debug-viz accessors below so all three agree on exactly which points are sampled."""
        P = self.num_height_points
        quat = self.base_quat[env_ids]
        quat_yaw = quat.clone()
        quat_yaw[:, 0] = 0.
        quat_yaw[:, 1] = 0.
        quat_yaw = quat_yaw / torch.norm(quat_yaw, dim=-1, keepdim=True)
        quat_yaw_exp = quat_yaw.unsqueeze(1).expand(-1, P, -1).reshape(-1, 4)
        local_pts = self.height_points[env_ids].reshape(-1, 3)
        rotated = quat_apply(quat_yaw_exp, local_pts).view(len(env_ids), P, 3)
        return (rotated + self.root_states[env_ids, :3].unsqueeze(1))[..., :2]

    def _height_scan_same_tile(self, env_ids, world_xy):
        """(len(env_ids), num_height_points) bool: whether each height-scan point resolves
        into the querying env's own (row, col) tile (True) or a neighboring one (False, would
        get masked in the observation -- see _get_heights)."""
        point_row, point_col = self._tile_of(world_xy)
        own_row = self.terrain_levels[env_ids].unsqueeze(1)
        own_col = self.terrain_types[env_ids].unsqueeze(1)
        return (point_row == own_row) & (point_col == own_col)

    def _get_heights(self, env_ids=None):
        """(len(env_ids), num_height_points) terrain height, sampled on the fixed body-
        frame grid (self.height_points), yaw-rotated + translated into world points.

        Points that resolve into a DIFFERENT (row, col) tile than the querying env's own are
        replaced with (own current base z - base_height_target), i.e. the ground height
        implied by standing at exactly the nominal target clearance -- NOT the base z itself.
        compute_observations encodes root_z - base_height_target - measured_heights, so this
        choice is what makes a masked point evaluate to exactly 0 (truly neutral, indistinguishable
        from ordinary flat ground directly underfoot); using root_z directly would instead give a
        constant -base_height_target bias on every masked point regardless of how the robot is
        actually standing, which is a real (if easy to miss) signal leak, not a neutral one.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        world_xy = self._height_scan_world_xy(env_ids)
        same_tile = self._height_scan_same_tile(env_ids, world_xy)
        heights = self._sample_terrain_height_at(world_xy)
        own_ground_z = self.root_states[env_ids, 2].unsqueeze(1) - self.cfg.rewards.base_height_target
        own_ground_z = own_ground_z.expand(-1, self.num_height_points)
        return torch.where(same_tile, heights, own_ground_z)

    def _waypoint_world(self, env_ids, idx=None):
        """World xy of each env's CURRENT waypoint (or `idx` if given), for u_staircase envs only."""
        cfg = self.cfg.terrain
        if idx is None:
            idx = self.waypoint_idx[env_ids].clamp(max=self.num_waypoints - 1)
        local = self.u_staircase_wp_local[idx]
        row = self.terrain_levels[env_ids].float()
        offset_x = row * cfg.terrain_length
        offset_y = torch.full_like(row, self.u_staircase_col * cfg.terrain_width)
        return local + torch.stack([offset_x, offset_y], dim=1)

    def _u_staircase_top_height(self, rows):
        """Nominal (unjittered) height of the u_staircase's top exit platform, for a
        tensor of difficulty rows. Shared by get_u_staircase_waypoints_world (debug viz)
        and _u_staircase_waypoint_z (spawn height for any waypoint) so they can't disagree."""
        cfg = self.cfg.terrain
        difficulty = rows.float() / max(cfg.num_rows - 1, 1)
        step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
        return 2 * cfg.u_staircase_num_steps * step_height

    def _u_staircase_waypoint_z(self, idx, rows):
        """Nominal world z for waypoint index `idx` (any shape, matching `rows`) on
        difficulty row `rows` -- wp0/wp1 (entry, bottom of flight A) are at 0; wp2-wp5 (top
        of flight A through the landing crossing back to its near edge) are at flightA_top;
        wp6/wp7 (top of flight B, top exit) are at top_height. Shared by
        _u_staircase_spawn_world (actual spawn height, now for an arbitrary graduated-start
        waypoint, not just the two true endpoints) and get_u_staircase_waypoints_world
        (debug viz) so they can't disagree."""
        top_height = self._u_staircase_top_height(rows)
        flightA_top = top_height / 2
        return torch.where(idx <= 1, torch.zeros_like(flightA_top),
                            torch.where(idx <= 5, flightA_top, top_height))

    def _reset_u_staircase_waypoints(self, env_ids):
        """Rolls a fresh ascend/descend direction for u_staircase envs among env_ids and
        (re)initializes waypoint_idx/waypoint_dir/waypoint_start_idx to match.

        The START waypoint is graduated by difficulty row, not always the true opposite
        end: the hardest row starts at the true bottom (ascend) or top (descend) -- the
        full real climb, matching the original design -- but easier rows start much closer
        to the goal, requiring only a short stretch of the chain. Training data showed the
        sparse-arrival + dense-distance rewards alone couldn't get PPO to commit to a full
        ~3m/12-step climb with no prior success to build on; starting near the goal lets it
        first learn "finish the last short stretch and succeed" (a quick, easy win that
        immediately promotes it), then each subsequent row demands progressively more of
        the chain, right up to the same full climb the hardest row always required.
        waypoint_start_idx is persisted (not just derived from waypoint_dir) because
        _update_terrain_curriculum needs to measure progress relative to THIS episode's
        actual start, not assume it was always a true endpoint.

        Non-u_staircase envs just get the harmless default (idx=0, dir=+1, start_idx=0;
        unused, since every consumer of these buffers is gated on on_u_staircase)."""
        self.waypoint_dir[env_ids] = 1
        self.waypoint_idx[env_ids] = 0
        self.waypoint_start_idx[env_ids] = 0
        u_mask = self.on_u_staircase[env_ids]
        if torch.any(u_mask):
            u_ids = env_ids[u_mask]
            n = len(u_ids)
            descending = torch.rand(n, device=self.device) < self.cfg.terrain.u_staircase_descend_prob
            self.waypoint_dir[u_ids] = torch.where(descending, -torch.ones(n, dtype=torch.long, device=self.device),
                                                    torch.ones(n, dtype=torch.long, device=self.device))

            cfg = self.cfg.terrain
            difficulty = self.terrain_levels[u_ids].float() / max(cfg.num_rows - 1, 1)
            # at least 1 waypoint's worth of travel even at the easiest row (a genuine,
            # if trivial, task) up to the full chain (num_waypoints-1) at the hardest
            required_dist = torch.clamp(torch.round((self.num_waypoints - 1) * difficulty), min=1).long()
            start_idx = torch.where(descending, required_dist, (self.num_waypoints - 1) - required_dist)
            start_idx = torch.clamp(start_idx, 0, self.num_waypoints - 1)
            self.waypoint_idx[u_ids] = start_idx
            self.waypoint_start_idx[u_ids] = start_idx

    def _u_staircase_spawn_world(self, env_ids):
        """World xyz to spawn u_staircase envs at, matching each one's just-rolled
        waypoint_idx (see _reset_u_staircase_waypoints, which must be called first and sets
        this to the difficulty-graduated START index -- not always a true endpoint)."""
        cfg = self.cfg.terrain
        idx = self.waypoint_idx[env_ids]
        local_xy = self.u_staircase_wp_local[idx]
        row = self.terrain_levels[env_ids].float()
        x = row * cfg.terrain_length + local_xy[:, 0]
        y = self.u_staircase_col * cfg.terrain_width + local_xy[:, 1]
        z = self._u_staircase_waypoint_z(idx, row)
        return torch.stack([x, y, z], dim=1)


    def get_u_staircase_waypoints_world(self):
        """(num_rows, num_waypoints, 3) world-xyz of the waypoint chain on every u_staircase
        difficulty row -- for external debug visualization (see view_go2_stairs_terrain.py).

        z uses the NOMINAL (unjittered) step height per row: wp0/wp1 (entry, bottom of
        flight A) are at 0; wp2-wp5 (top of flight A through the landing crossing back to
        its near edge) are at flightA_top; wp6/wp7 (top of flight B, top exit) are at
        top_height. This exactly matches the wp indices laid out in _init_buffers. The
        exact per-step jittered heights aren't retained after terrain generation, but the
        few-cm error from using the nominal height doesn't matter for a debug overlay --
        it only needs to not be buried inside the stairs.
        """
        cfg = self.cfg.terrain
        rows = torch.arange(cfg.num_rows, device=self.device).float()
        offset_x = rows.unsqueeze(1) * cfg.terrain_length  # (num_rows,1)
        offset_y = self.u_staircase_col * cfg.terrain_width
        local = self.u_staircase_wp_local.unsqueeze(0)  # (1,num_waypoints,2)
        world_xy = local.clone().repeat(cfg.num_rows, 1, 1)
        world_xy[:, :, 0] += offset_x
        world_xy[:, :, 1] += offset_y

        top_height = self._u_staircase_top_height(rows)
        flightA_top = top_height / 2
        zeros = torch.zeros_like(flightA_top)
        # wp0, wp1, wp2, wp3, wp4, wp5, wp6, wp7
        z = torch.stack([zeros, zeros, flightA_top, flightA_top, flightA_top, flightA_top, top_height, top_height], dim=1)

        return torch.cat([world_xy, z.unsqueeze(-1)], dim=-1)

    def get_height_scan_world(self, env_ids=None):
        """((len(env_ids), num_height_points, 3), (len(env_ids), num_height_points) bool) --
        world xyz of the height-scan points using the RAW (unmasked) sampled terrain height,
        plus whether each point is in the querying env's own tile, for external debug
        visualization (see view_go2_stairs_terrain.py).

        z is deliberately NOT self.measured_heights (what the policy actually gets, with
        cross-tile points replaced by the robot's own z -- see _get_heights). Reusing the
        masked value here would draw those spheres floating at the robot's OWN height
        regardless of how far away their xy actually is, which looks like broken/floating
        points -- this view answers "does the scan grid visually sit on whatever's really
        there", which needs the true height. The returned mask lets the caller color
        same-tile vs. cross-tile (would-be-masked) points differently instead of leaving
        them visually indistinguishable. See get_height_scan_obs_world for the companion
        view of what the policy actually receives.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        world_xy = self._height_scan_world_xy(env_ids)
        same_tile = self._height_scan_same_tile(env_ids, world_xy)
        z = self._sample_terrain_height_at(world_xy)
        return torch.cat([world_xy, z.unsqueeze(-1)], dim=-1), same_tile

    def get_height_scan_obs_world(self, env_ids=None):
        """(len(env_ids), num_height_points, 3) world xyz of the height-scan points using
        self.measured_heights -- the actual value the policy's observation vector encodes
        (cross-tile points collapsed to the robot's own current standing-clearance ground
        height, i.e. root z minus base_height_target -- see _get_heights). This is "what does
        the policy think is there": on flat ground, masked points render at the SAME world z
        as genuine flat terrain right under the robot, since both are meant to read as a
        neutral "ground is exactly where I'd expect" signal.

        self.measured_heights is refreshed every step in compute_observations, for all envs.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        world_xy = self._height_scan_world_xy(env_ids)
        z = self.measured_heights[env_ids]
        return torch.cat([world_xy, z.unsqueeze(-1)], dim=-1)

    def get_feet_edge_debug(self, env_ids=None):
        """For external debug visualization: ((len(env_ids), num_feet, 4, 3) world xyz of
        the 4 edge-check points around each foot, (len(env_ids), num_feet) bool for
        whether that foot is currently flagged as near-an-edge-while-in-contact) -- same
        computation as _reward_feet_edge, just also returning the points themselves."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        cfg = self.cfg.terrain
        foot_xy = self.rigid_body_states[env_ids][:, self.feet_indices, :2]  # (M,4,2)
        d = cfg.feet_edge_check_offset
        offsets = to_torch([[d, 0.], [-d, 0.], [0., d], [0., -d]], device=self.device)  # (4,2)
        pts = foot_xy.unsqueeze(2) + offsets.view(1, 1, 4, 2)  # (M,4feet,4pts,2)
        h = self._sample_terrain_height_at(pts)  # (M,4,4)
        edge = (h.max(dim=2).values - h.min(dim=2).values) > cfg.feet_edge_height_diff  # (M,4)
        in_contact = self.contact_forces[env_ids][:, self.feet_indices, 2] > 1.
        is_edge = edge & in_contact
        world_pts = torch.cat([pts, h.unsqueeze(-1)], dim=-1)  # (M,4feet,4pts,3)
        return world_pts, is_edge

    def _random_yaw_quat(self, env_ids):
        """Random yaw-only spawn quaternion for stairs_2step/3step envs -- so diagonal
        stair-crossing actually gets trained instead of always approaching dead-on.
        Identity (facing +x) for every other terrain type: u_staircase's corridor is
        narrow and walled, so a random approach angle there just means starting halfway
        into a wall; flat has no orientation-dependent structure to diversify against."""
        cfg = self.cfg.terrain
        quat = to_torch([0., 0., 0., 1.], device=self.device).repeat(len(env_ids), 1)
        short_stairs_cols = (cfg.terrain_types.index('stairs_2step'), cfg.terrain_types.index('stairs_3step'))
        is_short_stairs = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        for col in short_stairs_cols:
            is_short_stairs |= self.terrain_types[env_ids] == col
        idx = is_short_stairs.nonzero(as_tuple=False).flatten()
        if len(idx) > 0:
            n = len(idx)
            yaw = torch_rand_float(cfg.short_stairs_yaw_range[0], cfg.short_stairs_yaw_range[1], (n, 1), device=self.device).squeeze(1)
            axis = to_torch([0., 0., 1.], device=self.device).repeat(n, 1)
            quat[idx] = quat_from_angle_axis(yaw, axis)
        return quat

    def _sample_dead_zone(self, magnitude_range, n, positive_only=False):
        """Samples `n` values whose magnitude clears commands.dead_zone (never inside it)."""
        if n == 0:
            return torch.zeros(0, device=self.device)
        mag = torch_rand_float(magnitude_range[0], magnitude_range[1], (n, 1), device=self.device).squeeze(1)
        if positive_only:
            return mag
        sign = torch.where(torch.rand(n, device=self.device) < 0.5,
                            -torch.ones(n, device=self.device), torch.ones(n, device=self.device))
        return mag * sign

    def _resample_commands(self, env_ids):
        """Overrides LeggedRobot._resample_commands: dead-zone-aware stratified sampling
        on flat/short-stairs terrain (stand/walk/spin/combo buckets instead of plain i.i.d.
        uniform, which would waste most draws inside the robot's own +/-0.1 dead zone and
        under-sample the "stand still"/"pure spin" corners); on the u_staircase, only vx is
        sampled here -- heading/yaw rate are driven every step by the waypoint chase in
        _post_physics_step_callback. vy is always 0: the deployed policy only ever takes
        (vx, vyaw) commands.
        """
        if len(env_ids) == 0:
            return
        cmd_cfg = self.cfg.commands
        self.commands[env_ids, 1] = 0.0

        u_mask = self.on_u_staircase[env_ids]
        u_ids = env_ids[u_mask]
        generic_ids = env_ids[~u_mask]

        if len(u_ids) > 0:
            self.commands[u_ids, 0] = self._sample_dead_zone(cmd_cfg.u_staircase_vx_range, len(u_ids), positive_only=True)

        if len(generic_ids) > 0:
            n = len(generic_ids)
            weights = to_torch([cmd_cfg.stand_prob, cmd_cfg.walk_prob, cmd_cfg.spin_prob, cmd_cfg.combo_prob], device=self.device)
            bucket = torch.multinomial(weights, n, replacement=True)
            walk_mask = bucket == 1
            spin_mask = bucket == 2
            combo_mask = bucket == 3
            vx = torch.zeros(n, device=self.device)
            vyaw = torch.zeros(n, device=self.device)
            vx_mask = walk_mask | combo_mask
            vyaw_mask = spin_mask | combo_mask
            vx[vx_mask] = self._sample_dead_zone(cmd_cfg.lin_vel_x_range, int(vx_mask.sum().item()))
            vyaw[vyaw_mask] = self._sample_dead_zone(cmd_cfg.ang_vel_yaw_range, int(vyaw_mask.sum().item()))
            self.commands[generic_ids, 0] = vx
            self.commands[generic_ids, 2] = vyaw

    def _post_physics_step_callback(self):
        """Overrides LeggedRobot._post_physics_step_callback: same interval-based resample as
        the base class, but the heading->yaw-rate conversion only applies to u_staircase envs
        (chasing a waypoint that advances every step), not generic envs (whose vyaw was
        sampled directly by _resample_commands and shouldn't be overwritten). Also refreshes
        the rigid-body-state tensor (the base class never does -- see _init_buffers) and
        advances the gait-phase clock, both needed before reward/observation computation."""
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)

        # gait phase: advances at a fixed frequency whenever the commanded (vx, vyaw)
        # clears the dead zone, frozen while genuinely standing still (0 command) -- so
        # "spin in place" still cycles the legs (fixes body-twist-without-stepping),
        # while true standing is left to the gated base_height reward instead
        cmd_mag = torch.sqrt(self.commands[:, 0] ** 2 + self.commands[:, 2] ** 2)
        self.gait_moving = cmd_mag > 1e-3
        self.gait_phase = torch.where(
            self.gait_moving,
            (self.gait_phase + 2 * float(np.pi) * self.cfg.rewards.gait_freq * self.dt) % (2 * float(np.pi)),
            self.gait_phase)

        self.waypoint_advanced[:] = False
        self.waypoint_dist_delta[:] = 0.
        u_ids = self.on_u_staircase.nonzero(as_tuple=False).flatten()
        if len(u_ids) > 0:
            robot_xy = self.root_states[u_ids, :2]
            wp_world = self._waypoint_world(u_ids)
            dist_now = torch.norm(wp_world - robot_xy, dim=1)
            # dense potential-based shaping toward the CURRENT (not-yet-advanced) target --
            # see _reward_waypoint_dist_progress for why this is needed alongside the
            # sparse waypoint_advanced bonus below: that bonus only pays out on actually
            # arriving at a waypoint up to 3m/12 steps away, which turned out to give zero
            # gradient for "making some progress but not there yet", letting training data
            # show robots getting stuck (even sliding backward) partway up a flight with no
            # per-step incentive to keep climbing
            self.waypoint_dist_delta[u_ids] = self.prev_waypoint_dist[u_ids] - dist_now
            self.prev_waypoint_dist[u_ids] = dist_now

            reached = dist_now < self.cfg.commands.waypoint_reach_threshold
            # advances toward wp7 if ascending (dir=+1) or wp0 if descending (dir=-1);
            # clip is a no-op once at that end, so no separate "already at the goal" guard
            # is needed -- re-adding waypoint_dir there would just clip right back
            next_idx = torch.clip(self.waypoint_idx[u_ids] + self.waypoint_dir[u_ids], 0, self.num_waypoints - 1)
            self.waypoint_advanced[u_ids] = reached & (next_idx != self.waypoint_idx[u_ids])
            self.waypoint_idx[u_ids] = torch.where(reached, next_idx, self.waypoint_idx[u_ids])

            wp_world = self._waypoint_world(u_ids)  # re-fetch: idx may have just advanced
            # the distance potential also has to restart relative to the NEW target the
            # instant idx advances, or next step's delta would be a huge (and bogus)
            # negative spike from "close to the old target" to "far from the new one"
            self.prev_waypoint_dist[u_ids] = torch.where(reached, torch.norm(wp_world - robot_xy, dim=1),
                                                          self.prev_waypoint_dist[u_ids])
            delta = wp_world - robot_xy
            self.commands[u_ids, 3] = torch.atan2(delta[:, 1], delta[:, 0])

            forward = quat_apply(self.base_quat[u_ids], self.forward_vec[u_ids])
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            yaw_limit = self.cfg.commands.ang_vel_yaw_range[1]
            kp = self.cfg.commands.waypoint_heading_kp
            self.commands[u_ids, 2] = torch.clip(kp * wrap_to_pi(self.commands[u_ids, 3] - heading), -yaw_limit, yaw_limit)

    def compute_observations(self):
        """Overrides LeggedRobot.compute_observations: the base 48-dim proprio vector
        (unchanged, same layout/order) plus two features the base class has no notion
        of -- gait phase (sin, cos) and the height scan -- appended at the end. Not
        calling super() here since it applies noise internally sized for 48 dims; this
        builds the full (48+2+num_height_points) vector once and applies noise to all
        of it at the end via the correspondingly-sized _get_noise_scale_vec override.
        """
        self.measured_heights = self._get_heights()
        heights = torch.clip(
            self.root_states[:, 2].unsqueeze(1) - self.cfg.rewards.base_height_target - self.measured_heights,
            -1., 1.) * self.obs_scales.height_measurements

        self.obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            torch.sin(self.gait_phase).unsqueeze(1),
            torch.cos(self.gait_phase).unsqueeze(1),
            heights,
        ), dim=-1)
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _get_noise_scale_vec(self, cfg):
        """Overrides LeggedRobot._get_noise_scale_vec: same base-48 layout as upstream,
        plus gait phase (no sensor noise -- it's an internal clock, nothing to measure)
        and height-scan noise appended at the end, matching compute_observations' layout.
        Computes the height-point count directly from cfg rather than reading
        self.num_height_points, since this runs (via super()._init_buffers()) before this
        subclass's own _init_buffers code has set that up yet."""
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.  # commands
        noise_vec[12:12 + self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[12 + self.num_actions:12 + 2 * self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[12 + 2 * self.num_actions:12 + 3 * self.num_actions] = 0.  # previous actions
        idx = 12 + 3 * self.num_actions
        noise_vec[idx:idx + 2] = 0.  # gait phase sin/cos
        idx += 2
        n_height = len(cfg.terrain.measured_points_x) * len(cfg.terrain.measured_points_y)
        noise_vec[idx:idx + n_height] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        return noise_vec

    def check_termination(self):
        """Overrides LeggedRobot.check_termination: same contact-force/orientation checks,
        plus three additions this terrain set needs and the base class has no notion of --
        every legitimate floor here is z >= 0, so a robot that fell into the u_staircase
        shaft (which bottoms out at -shaft_depth) is caught by a simple height threshold;
        a robot that wandered outside its own assigned tile is terminated so its
        reward/waypoint bookkeeping (which assumes that tile's geometry) can't go stale;
        and a robot that actually reached the u_staircase's final waypoint ends its episode
        there (a success, not a failure -- folded into time_out_buf below) instead of
        running out the full episode_length_s with a heading-chase target that's already
        been reached (which was undefined/noisy: the reach threshold caps waypoint_idx at
        the last index, but nothing previously stopped the episode there, so it would just
        keep chasing a point it's already standing on for however long was left)."""
        # each condition is also stashed on a persistent per-env buffer (fell_buf/
        # shaft_fall_buf/oob_buf/reached_goal_buf), not just OR'd into reset_buf --
        # _accumulate_episode_stats reads these right after, in reset_idx, to classify
        # *why* each about-to-reset env is resetting for the detailed episode-outcome log.
        # Recomputing this same logic there from scratch would be redundant and risk
        # drifting out of sync with whatever reset_buf actually ends up meaning here.
        self.fell_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.fell_buf |= torch.logical_or(torch.abs(self.rpy[:, 1]) > 1.0, torch.abs(self.rpy[:, 0]) > 0.8)
        self.reset_buf = self.fell_buf.clone()

        self.shaft_fall_buf = self.root_states[:, 2] < self.cfg.terrain.fall_height_threshold
        self.reset_buf |= self.shaft_fall_buf

        margin = self.cfg.terrain.bounds_margin
        xy = self.root_states[:, :2]
        out_of_bounds = (xy[:, 0] < self.tile_x_min - margin) | (xy[:, 0] > self.tile_x_max + margin) | \
                        (xy[:, 1] < self.tile_y_min - margin) | (xy[:, 1] > self.tile_y_max + margin)
        self.oob_buf = out_of_bounds
        self.reset_buf |= out_of_bounds

        reached_goal = torch.zeros_like(self.reset_buf)
        u_ids = self.on_u_staircase.nonzero(as_tuple=False).flatten()
        if len(u_ids) > 0:
            wp_world = self._waypoint_world(u_ids)
            # goal is wp7 if ascending (dir=+1), wp0 if descending (dir=-1)
            goal_idx = torch.where(self.waypoint_dir[u_ids] == -1,
                                    torch.zeros_like(self.waypoint_idx[u_ids]),
                                    torch.full_like(self.waypoint_idx[u_ids], self.num_waypoints - 1))
            at_goal_idx = self.waypoint_idx[u_ids] == goal_idx
            close_enough = torch.norm(self.root_states[u_ids, :2] - wp_world, dim=1) < self.cfg.commands.waypoint_reach_threshold
            reached_goal[u_ids] = at_goal_idx & close_enough
        self.reached_goal_buf = reached_goal
        self.reset_buf |= reached_goal

        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.time_out_buf |= reached_goal  # success, not a failure -- exempt from any future termination penalty
        # leaving the tile is only hazard-adjacent on the u_staircase (walls/shaft nearby);
        # flat/short_stairs have open, flat margins on every side (no cliff, no wall), so
        # wandering past the tracked area there isn't a "failure" the way falling over is
        # -- it just means the episode's useful tracking window ended, same as a timeout
        self.time_out_buf |= (out_of_bounds & ~self.on_u_staircase)
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """Overrides LeggedRobot.reset_idx to (1) capture per-(terrain_type, row) episode
        stats for the detailed outcome log BEFORE anything below can change/zero the data
        it reads (terrain_levels, episode_length_buf, episode_sums, the outcome flags set
        by check_termination) -- see _accumulate_episode_stats; (2) run the curriculum
        promotion BEFORE the base class resets root state -- _update_terrain_curriculum
        needs the OLD (pre-reset) root_states to judge how far this episode got, and it
        must run before _reset_root_states so the (possibly just-changed) terrain_levels/
        env_origins are what the robot actually gets respawned at -- and (3) log the mean
        difficulty row into extras/episode, same convention upstream legged_gym uses, so a
        training dashboard can show whether the curriculum is actually advancing."""
        if len(env_ids) == 0:
            return
        self._accumulate_episode_stats(env_ids)
        self._update_terrain_curriculum(env_ids)
        super().reset_idx(env_ids)
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())

    def _accumulate_episode_stats(self, env_ids):
        """Buckets this batch of about-to-reset envs by (terrain_type, CURRENT difficulty
        row) and tallies outcome/episode-length/reward-component totals into the
        persistent bucket_* accumulators, for get_and_reset_episode_report() to read later.
        Must run before _update_terrain_curriculum (which may change terrain_levels for the
        NEXT episode) and before super().reset_idx() (which zeroes episode_sums/
        episode_length_buf) -- both would otherwise corrupt exactly the pre-reset info this
        is trying to capture.

        Outcome is classified with a fixed precedence (a reset can technically satisfy more
        than one condition in the same step, e.g. reached_goal on the exact step a push also
        knocks orientation over the fell threshold): reached_goal > shaft_fall > oob_fail >
        fell > oob_benign > plain timeout. reached_goal/shaft_fall/oob_fail are only ever
        true on the u_staircase; oob_benign only on flat/short_stairs (see check_termination).
        """
        cfg = self.cfg.terrain
        NR, NO, NRW = cfg.num_rows, len(self._outcome_names), len(self._reward_names)
        cols = self.terrain_types[env_ids]
        rows = self.terrain_levels[env_ids]
        bucket = cols * NR + rows

        reached = self.reached_goal_buf[env_ids]
        shaft = self.shaft_fall_buf[env_ids]
        oob = self.oob_buf[env_ids]
        fell = self.fell_buf[env_ids]
        on_u = self.on_u_staircase[env_ids]
        oob_fail = oob & on_u
        oob_benign = oob & ~on_u

        outcome = torch.full_like(bucket, 5)  # default: plain timeout
        outcome[oob_benign] = 4
        outcome[fell] = 3
        outcome[oob_fail] = 2
        outcome[shaft] = 1
        outcome[reached] = 0

        ones = torch.ones_like(bucket, dtype=torch.float)
        self.bucket_ep_count.scatter_add_(0, bucket, ones)
        ep_len_s = self.episode_length_buf[env_ids].float() * self.dt
        self.bucket_ep_len_sum.scatter_add_(0, bucket, ep_len_s)
        self.bucket_outcome_count.scatter_add_(0, bucket * NO + outcome, ones)
        for i, name in enumerate(self._reward_names):
            self.bucket_rew_sum.scatter_add_(0, bucket * NRW + i, self.episode_sums[name][env_ids])

    def get_and_reset_episode_report(self):
        """Snapshot + zero the per-(terrain_type, difficulty_row) episode-outcome
        accumulators since the last call. Meant to be polled periodically (e.g. every few
        dozen learning iterations) by the training/eval script, not every step -- each call
        reports "what happened in this recent window", which is what makes a time series of
        these useful for spotting exactly when a specific terrain/difficulty bucket started
        regressing, rather than one all-time-cumulative number that dilutes recent problems.

        Returns {terrain_type_name: {row: {...}}}, one entry per (type, row) bucket that had
        at least one episode this window -- buckets with zero episodes are omitted entirely
        (not reported as a 0% rate) so "no data yet" can't be misread as "always failing".
        Also includes 'env_distribution': a live (not windowed) snapshot of how many envs
        are currently assigned to each bucket, to tell "no episodes" (bucket has envs, just
        none finished yet) apart from "no envs" (curriculum hasn't promoted anyone there).
        """
        cfg = self.cfg.terrain
        NR, NO, NRW = cfg.num_rows, len(self._outcome_names), len(self._reward_names)
        ep_count = self.bucket_ep_count.cpu().numpy()
        ep_len_sum = self.bucket_ep_len_sum.cpu().numpy()
        outcome_count = self.bucket_outcome_count.view(cfg.num_cols, NR, NO).cpu().numpy()
        rew_sum = self.bucket_rew_sum.view(cfg.num_cols, NR, NRW).cpu().numpy()
        ep_count_2d = ep_count.reshape(cfg.num_cols, NR)
        ep_len_sum_2d = ep_len_sum.reshape(cfg.num_cols, NR)

        report = {}
        for c, terrain_name in enumerate(cfg.terrain_types):
            per_row = {}
            for r in range(NR):
                n = ep_count_2d[c, r]
                if n <= 0:
                    continue
                per_row[r] = {
                    'episodes': int(n),
                    'mean_episode_length_s': float(ep_len_sum_2d[c, r] / n),
                    'outcome_rate': {name: float(outcome_count[c, r, k] / n)
                                      for k, name in enumerate(self._outcome_names)},
                    'mean_reward': {name: float(rew_sum[c, r, i] / (n * self.max_episode_length_s))
                                    for i, name in enumerate(self._reward_names)},
                }
            if per_row:
                report[terrain_name] = per_row

        env_distribution = {}
        types_cpu = self.terrain_types.cpu().numpy()
        levels_cpu = self.terrain_levels.cpu().numpy()
        for c, terrain_name in enumerate(cfg.terrain_types):
            counts = {}
            for r in range(NR):
                n = int(((types_cpu == c) & (levels_cpu == r)).sum())
                if n > 0:
                    counts[r] = n
            if counts:
                env_distribution[terrain_name] = counts

        self.bucket_ep_count.zero_()
        self.bucket_ep_len_sum.zero_()
        self.bucket_outcome_count.zero_()
        self.bucket_rew_sum.zero_()
        return {'by_terrain': report, 'env_distribution': env_distribution}

    def _update_terrain_curriculum(self, env_ids):
        """Promotes/demotes each env's difficulty ROW (terrain_types/columns never change)
        based on how this just-finished episode went. Progress is measured differently per
        terrain type -- see terrain.curriculum_demote_waypoint_frac's docstring in the
        config for why u_staircase can't just use raw displacement like the others.

        This never runs while cfg.terrain.curriculum is False (e.g. the standalone terrain
        viewer sets it False specifically so difficulty rows stay put for inspection instead
        of drifting as robots fall/reset), and never before init_done (defensive, matches
        upstream legged_gym's own guard -- nothing should reset before the sim is fully up).
        """
        cfg = self.cfg.terrain
        if not cfg.curriculum or not self.init_done:
            return

        move_up = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        move_down = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        u_mask = self.on_u_staircase[env_ids]

        if torch.any(u_mask):
            u_ids = env_ids[u_mask]
            # progress = how far idx moved from its OWN actual start this episode, relative
            # to how far it NEEDED to move to finish (goal - start) -- NOT a fixed
            # (num_waypoints-1) denominator, since the graduated-start curriculum (see
            # _reset_u_staircase_waypoints) means easier rows start much closer to the goal
            # than the true opposite endpoint, and a fixed denominator would make those
            # nearly impossible to ever register as "reached the goal" (frac >= 1.0) even
            # after actually completing their (much shorter) required stretch. Uses
            # waypoint_dir/waypoint_start_idx as they stood for the episode that just ended
            # (this runs before _reset_root_states rolls fresh ones for the next episode).
            start_idx = self.waypoint_start_idx[u_ids].float()
            goal_idx = torch.where(self.waypoint_dir[u_ids] == -1,
                                    torch.zeros_like(start_idx),
                                    torch.full_like(start_idx, self.num_waypoints - 1))
            required = (goal_idx - start_idx).abs().clamp(min=1.)
            reached_frac = (self.waypoint_idx[u_ids].float() - start_idx).abs() / required
            move_up[u_mask] = reached_frac >= 1.0
            move_down[u_mask] = reached_frac < cfg.curriculum_demote_waypoint_frac

        g_mask = ~u_mask
        if torch.any(g_mask):
            g_ids = env_ids[g_mask]
            distance = torch.norm(self.root_states[g_ids, :2] - self.env_origins[g_ids, :2], dim=1)
            # "did about as well as commanded" target, same convention as upstream
            # legged_gym's rough-terrain curriculum: half the distance a perfect
            # tracker would have covered at its last-sampled forward speed
            target = torch.norm(self.commands[g_ids, :2], dim=1) * self.max_episode_length_s * 0.5
            g_move_up = distance > cfg.terrain_length / 2
            # only demote when the last-sampled command was ~straight (vyaw ~0): the
            # target formula above assumes straight-line motion, so a "combo" episode
            # (nonzero vx AND vyaw) legitimately covers less net displacement than target
            # even when correctly tracking the command -- it's turning, not underperforming
            # ("spin", vx=0, already has target=0 and so is unaffected either way)
            was_straight = self.commands[g_ids, 2].abs() < self.cfg.commands.dead_zone
            g_move_down = (distance < target) & ~g_move_up & was_straight
            move_up[g_mask] = g_move_up
            move_down[g_mask] = g_move_down

        self.terrain_levels[env_ids] = torch.clip(
            self.terrain_levels[env_ids] + move_up.long() - move_down.long(), 0, cfg.num_rows - 1)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self._update_tile_bounds(env_ids)

    def _reset_root_states(self, env_ids):
        """Same as LeggedRobot._reset_root_states, except: (1) the fixed +/-1m spawn jitter
        (applied on EVERY reset, not just the very first spawn) is replaced with
        cfg.terrain.spawn_xy_jitter -- the base class's +/-1m was flinging robots past
        the u_staircase's walls/into the shaft, and off the short-stairs' tread on every
        reset -- _create_envs already had this fix for the initial spawn, but every
        subsequent reset_idx() call goes through this method instead, which was still
        using the base class's hardcoded range; (2) spawn yaw is randomized for
        stairs_2step/3step envs (see _random_yaw_quat) instead of always facing +x; and
        (3) u_staircase envs get a freshly-rolled ascend/descend direction, spawning at
        the bottom entry or top exit to match. Facing stays +x (the _random_yaw_quat
        default) either way: wp0->wp1 (ascend's first leg, across the entry platform)
        and wp7->wp6 (descend's first leg, across the top exit platform) are BOTH +x --
        entry and exit platforms sit at the same (near) end of the tile and both flights
        start by heading away from it. (Earlier had this spawning descend facing -x,
        on the assumption it should mirror flight B's climbing direction -- wrong, that
        forgot descend starts on the flat platform, not already on flight B.)
        """
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            jitter = self.cfg.terrain.spawn_xy_jitter
            self.root_states[env_ids, :2] += torch_rand_float(-jitter, jitter, (len(env_ids), 2), device=self.device)
            self.root_states[env_ids, 3:7] = self._random_yaw_quat(env_ids)

            self._reset_u_staircase_waypoints(env_ids)
            u_mask = self.on_u_staircase[env_ids]
            if torch.any(u_mask):
                u_ids = env_ids[u_mask]
                self.root_states[u_ids, :3] = self.base_init_state[:3] + self._u_staircase_spawn_world(u_ids)
                self.root_states[u_ids, :2] += torch_rand_float(-jitter, jitter, (len(u_ids), 2), device=self.device)
                # prev_waypoint_dist needs the just-set spawn xy, so this can't happen
                # inside _reset_u_staircase_waypoints (which runs before spawn xy is set)
                wp_world = self._waypoint_world(u_ids)
                self.prev_waypoint_dist[u_ids] = torch.norm(wp_world - self.root_states[u_ids, :2], dim=1)
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device)
        self.gait_phase[env_ids] = 0.
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _create_envs(self):
        """Same as LeggedRobot._create_envs, except the fixed +/-1m spawn jitter
        (fine on open rough terrain, not fine next to a 1.4m-wide flight or a
        0.4m-wide shaft) is replaced with cfg.terrain.spawn_xy_jitter."""
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        jitter = self.cfg.terrain.spawn_xy_jitter
        spawn_quats = self._random_yaw_quat(torch.arange(self.num_envs, device=self.device)).cpu().numpy()
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-jitter, jitter, (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            start_pose.r = gymapi.Quat(*spawn_quats[i])

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        # trot gait: diagonal pairs move together. Go2's feet are named FL_foot/FR_foot/
        # RL_foot/RR_foot -- FL & RR share phase (offset 0), FR & RL are offset by pi.
        # Determined by name rather than assumed index order, since body index order
        # depends on the URDF/asset loader, not on how feet_names happens to be sorted.
        self.foot_phase_offset = torch.zeros(len(feet_names), device=self.device)
        for i, name in enumerate(feet_names):
            if name.startswith('FR') or name.startswith('RL'):
                self.foot_phase_offset[i] = float(np.pi)

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    # ------------------------------------------------------------------
    # reward functions -- Extreme Parkour's list adapted to this task
    # (tracking_lin_vel/tracking_ang_vel/lin_vel_z/ang_vel_xy/orientation/
    # dof_acc/collision/action_rate/torques/dof_pos_limits are all reused
    # as-is from LeggedRobot, just enabled via this task's reward scales)
    # plus the gait-phase and waypoint-progress rewards this task adds.
    # ------------------------------------------------------------------
    def _reward_hip_pos(self):
        # Penalize hip joints drifting from their default angle -- keeps the stance
        # natural instead of splaying the hips out to solve balance problems
        err = self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]
        return torch.sum(torch.square(err), dim=1)

    def _reward_dof_error(self):
        # Penalize ALL joints drifting from their default pose -- general "stay near a
        # natural pose" regularizer, softer/broader than hip_pos
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_feet_stumble(self):
        # Penalize a foot hitting a near-vertical surface (tripping on a stair's riser/
        # edge) -- identical logic to LeggedRobot._reward_stumble, just under the name
        # that matches this task's (and Extreme Parkour's) reward-scale key
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >
                          5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1).float()

    def _reward_feet_edge(self):
        # Penalize a foot that's in contact right where the local terrain height changes
        # sharply -- i.e. planted right on a stair's edge, at risk of slipping off it.
        # Checked by sampling terrain height a small offset to each side of the foot
        # (+-x, +-y) and looking at the spread, not by re-using the coarse body-frame
        # height-scan grid, which doesn't necessarily have a sample point right under
        # any given foot.
        cfg = self.cfg.terrain
        foot_xy = self.rigid_body_states[:, self.feet_indices, :2]  # (N,4,2)
        d = cfg.feet_edge_check_offset
        offsets = to_torch([[d, 0.], [-d, 0.], [0., d], [0., -d]], device=self.device)  # (4,2)
        pts = foot_xy.unsqueeze(2) + offsets.view(1, 1, 4, 2)  # (N,4,4,2)
        h = self._sample_terrain_height_at(pts)  # (N,4,4)
        edge = (h.max(dim=2).values - h.min(dim=2).values) > cfg.feet_edge_height_diff  # (N,4)
        in_contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        return torch.sum((edge & in_contact).float(), dim=1)

    def _reward_feet_swing_height(self):
        # Penalize a swing-phase foot for not clearing the LOCAL terrain (not world z --
        # this terrain set climbs stairs) by the target height. Gated to gait_moving:
        # while genuinely standing still the phase is frozen (see
        # _post_physics_step_callback) and every foot would otherwise be stuck
        # permanently on one side of the swing/stance split for no reason.
        foot_phase = self.gait_phase.unsqueeze(1) + self.foot_phase_offset.unsqueeze(0)
        swing = (torch.sin(foot_phase) > 0) & self.gait_moving.unsqueeze(1)
        foot_xy = self.rigid_body_states[:, self.feet_indices, :2]
        foot_z = self.rigid_body_states[:, self.feet_indices, 2]
        terrain_h = self._sample_terrain_height_at(foot_xy)
        clearance = foot_z - terrain_h
        err = torch.square(clearance - self.cfg.rewards.foot_swing_height_target)
        return torch.sum(err * swing.float(), dim=1)

    def _reward_gait_contact(self):
        # Reward each foot's actual contact state for matching what the gait clock
        # currently expects (in contact during its "stance" half-cycle, airborne during
        # "swing") -- directly targets the "twists the body instead of moving the feet"
        # failure mode that motivated adding a gait clock in the first place. Also gated
        # to gait_moving, same reasoning as feet_swing_height.
        foot_phase = self.gait_phase.unsqueeze(1) + self.foot_phase_offset.unsqueeze(0)
        expect_contact = torch.sin(foot_phase) <= 0
        actual_contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        match = (expect_contact == actual_contact).float()
        return torch.sum(match * self.gait_moving.unsqueeze(1).float(), dim=1)

    def _reward_waypoint_progress(self):
        # Sparse bonus on the u_staircase the instant a NEW waypoint is reached (not
        # every step it happens to already be sitting on one -- see
        # _post_physics_step_callback, which only sets this True for one step). Zero for
        # every other terrain type. A small assist through the awkward heading
        # transition at the landing crossing, per the original waypoint design.
        return self.waypoint_advanced.float()

    def _reward_waypoint_dist_progress(self):
        # Dense potential-based shaping companion to _reward_waypoint_progress: this pays
        # out every step in proportion to how much closer (or, if negative, farther) the
        # robot got to its CURRENT target waypoint since the last step -- see
        # _post_physics_step_callback for exactly how waypoint_dist_delta is maintained
        # (including the reset-to-zero the instant a waypoint is actually reached, so
        # switching targets doesn't create a bogus one-step spike either direction).
        #
        # Added after training data showed the sparse-only bonus wasn't enough: robots got
        # stuck partway up a 12-step/~3m flight (even sliding back down) with literally no
        # reward gradient for "making some progress but not there yet" -- the only
        # progress-linked signal was a single +0.5 that only pays out on fully arriving at
        # a waypoint up to 3m away. This fills that gap with a per-step signal that scales
        # with actual forward progress along the current leg, whatever its length.
        return self.waypoint_dist_delta

    def _reward_base_height(self):
        # Overrides LeggedRobot._reward_base_height: measured relative to the LOCAL
        # terrain height (not a fixed world z -- this terrain set climbs stairs), and
        # gated to only apply while standing (gait_moving False, i.e. |cmd| in the dead
        # zone) -- while actively walking/climbing, base height naturally varies with
        # the gait and the stairs and shouldn't be penalized for that.
        local_h = self._sample_terrain_height_at(self.root_states[:, :2])
        base_h = self.root_states[:, 2] - local_h
        err = torch.square(base_h - self.cfg.rewards.base_height_target)
        return err * (~self.gait_moving).float()
