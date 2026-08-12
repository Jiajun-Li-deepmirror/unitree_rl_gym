"""Stand-alone terrain viewer for the go2_stairs task.

No trained policy exists yet, so this intentionally bypasses
task_registry/play.py (which expects a checkpoint to load) and instead
instantiates GO2StairsRobot directly, steps it with zero actions, and
leaves the Isaac Gym viewer open so every (difficulty row, terrain type)
tile can be flown around and inspected.

Usage:
    python view_go2_stairs_terrain.py
    python view_go2_stairs_terrain.py --num_envs 20
"""
import isaacgym  # noqa: F401  (must be imported before torch)
from isaacgym import gymapi, gymutil

from legged_gym.envs.go2_stairs.go2_stairs_config import GO2StairsCfg
from legged_gym.envs.go2_stairs.go2_stairs_env import GO2StairsRobot
from legged_gym.utils.helpers import class_to_dict, get_args, parse_sim_params, set_seed

import numpy as np
import torch

_WAYPOINT_SPHERE = gymutil.WireframeSphereGeometry(0.08, 8, 8, None, color=(1.0, 0.15, 0.15))
_HEIGHT_SCAN_SPHERE = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0.1, 0.7, 1.0))
_EDGE_OK_SPHERE = gymutil.WireframeSphereGeometry(0.015, 4, 4, None, color=(0.2, 1.0, 0.2))
# deliberately NOT red -- red is already used for waypoint markers, and the two would be
# easy to mistake for each other at a glance
_EDGE_FLAGGED_SPHERE = gymutil.WireframeSphereGeometry(0.035, 6, 6, None, color=(1.0, 0.0, 1.0))


def _draw_u_staircase_waypoints(env):
    """Adds the u_staircase's waypoint centerline (one chain per difficulty row) as
    orange line segments with a red sphere at each waypoint.

    Each waypoint uses its own height (entry=0, top of flight A/across the landing,
    top exit) -- NOT a flat z, since the chain climbs two flights of stairs. A flat z
    would bury most of the line inside the stairs instead of showing the actual path.
    """
    waypoints = env.get_u_staircase_waypoints_world().cpu().numpy()  # (num_rows, num_wp, 3)
    lift = 0.05  # draw slightly above each waypoint's own floor height, not a flat world z
    for row in range(waypoints.shape[0]):
        pts = waypoints[row]
        num_segments = pts.shape[0] - 1
        verts = np.zeros((num_segments, 2, 3), dtype=np.float32)
        verts[:, 0, :] = pts[:-1]
        verts[:, 1, :] = pts[1:]
        verts[:, :, 2] += lift
        colors = np.tile(np.array([1.0, 0.6, 0.0], dtype=np.float32), (num_segments, 1))
        env.gym.add_lines(env.viewer, env.envs[0], num_segments, verts, colors)
        for k in range(pts.shape[0]):
            pose = gymapi.Transform(gymapi.Vec3(float(pts[k, 0]), float(pts[k, 1]), float(pts[k, 2]) + lift), r=None)
            gymutil.draw_lines(_WAYPOINT_SPHERE, env.gym, env.viewer, env.envs[0], pose)


def _draw_height_scan(env, focus_env_ids):
    """Adds a small cyan sphere at every height-scan point's OBS height -- exactly what
    self.measured_heights encodes for the policy right now, for a handful of "focus" envs
    only (all 20 x 187 points would be unreadable clutter). Cross-tile points (a different
    (row, col) tile than the robot's own) are collapsed to the robot's own current z here,
    same as in the actual observation -- see _get_heights.
    """
    pts = env.get_height_scan_obs_world(focus_env_ids).cpu().numpy()  # (F, P, 3)
    for f in range(pts.shape[0]):
        for p in range(pts.shape[1]):
            x, y, z = pts[f, p]
            pose = gymapi.Transform(gymapi.Vec3(float(x), float(y), float(z) + 0.01), r=None)
            gymutil.draw_lines(_HEIGHT_SCAN_SPHERE, env.gym, env.viewer, env.envs[0], pose)


def _draw_feet_edge(env, focus_env_ids):
    """Adds the 4 edge-check points around each focus env's foot: green normally, red
    (and bigger) the instant that foot is flagged as "in contact right on an edge" by
    _reward_feet_edge -- lets you confirm the flagging actually lines up with a real
    stair edge instead of firing on flat ground or missing an obvious one.
    """
    world_pts, is_edge = env.get_feet_edge_debug(focus_env_ids)
    world_pts = world_pts.cpu().numpy()  # (F, num_feet, 4, 3)
    is_edge = is_edge.cpu().numpy()  # (F, num_feet)
    for f in range(world_pts.shape[0]):
        for foot in range(world_pts.shape[1]):
            geom = _EDGE_FLAGGED_SPHERE if is_edge[f, foot] else _EDGE_OK_SPHERE
            for k in range(world_pts.shape[2]):
                x, y, z = world_pts[f, foot, k]
                pose = gymapi.Transform(gymapi.Vec3(float(x), float(y), float(z) + 0.01), r=None)
                gymutil.draw_lines(geom, env.gym, env.viewer, env.envs[0], pose)


def view_terrain(args):
    env_cfg = GO2StairsCfg()

    if args.num_envs is not None:
        env_cfg.env.num_envs = args.num_envs
    env_cfg.env.episode_length_s = 1000  # avoid timeout resets while inspecting
    env_cfg.terrain.curriculum = False  # spread envs across every difficulty row, not just row 0
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.noise.add_noise = False

    set_seed(args.seed if args.seed is not None else 1)
    sim_params = parse_sim_params(args, {"sim": class_to_dict(env_cfg.sim)})

    env = GO2StairsRobot(cfg=env_cfg, sim_params=sim_params, physics_engine=args.physics_engine,
                          sim_device=args.sim_device, headless=args.headless)

    cfg = env_cfg.terrain
    # drawn for every env -- with the default num_envs=20 this is ~4000 wireframe
    # spheres/frame (20 x 187 height-scan + 20 x 16 feet-edge), which is fine for a
    # one-off inspection session but will get sluggish if you crank --num_envs up a lot
    focus_env_ids = torch.arange(env.num_envs, device=env.device)

    print(f"[view_go2_stairs_terrain] {cfg.num_rows} difficulty rows x {cfg.num_cols} terrain "
          f"types {cfg.terrain_types}, tile size {cfg.terrain_length}m x {cfg.terrain_width}m, "
          f"{env.num_envs} robots spawned.")
    print("[view_go2_stairs_terrain] Robots hold their default stance (zero actions); "
          "falling into the u_staircase shaft is expected to trigger a reset. "
          "Fly the viewer camera (right-drag + WASD) to inspect each tile.")
    print("[view_go2_stairs_terrain] The u_staircase's waypoint centerline is drawn as "
          "orange segments with red spheres at each waypoint, one chain per difficulty row.")
    print(f"[view_go2_stairs_terrain] Height-scan points (cyan, showing exactly what the "
          f"policy's observation encodes) and feet-edge checks (green/magenta) are drawn "
          f"for all {env.num_envs} robots.")

    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    while True:
        env.step(actions)
        if env.viewer is not None:
            env.gym.clear_lines(env.viewer)
            _draw_u_staircase_waypoints(env)
            _draw_height_scan(env, focus_env_ids)
            _draw_feet_edge(env, focus_env_ids)


if __name__ == '__main__':
    view_terrain(get_args())
