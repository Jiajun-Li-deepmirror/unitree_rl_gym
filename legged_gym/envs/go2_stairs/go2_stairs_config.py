from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class GO2StairsCfg(LeggedRobotCfg):
    """Config for the Go2 stairs/staircase task.

    Deliberately inherits only the generic ``LeggedRobotCfg`` base, not
    ``GO2RoughCfg`` from go2_config.py: this file (plus go2_stairs_env.py)
    is meant to stay self-contained. Every Go2-specific number below
    (URDF path, joint names, PD gains, default pose, contact link names,
    soft dof-limit/base-height tuning) is copied verbatim from
    go2_config.py as of the time of writing -- if that file's numbers
    change later, these will not follow automatically.

    Terrain, command generation, termination/curriculum, observations (gait
    phase + height scan), and rewards are all implemented in
    go2_stairs_env.py's GO2StairsRobot. `normalization`/`noise` are still
    the plain LeggedRobotCfg defaults -- nothing terrain/task-specific has
    come up that needs changing there yet.
    """

    class env(LeggedRobotCfg.env):
        # one robot per (difficulty, terrain type) tile by default so a
        # quick look at the viewer shows the whole grid; override with
        # --num_envs for anything else
        num_envs = 20
        episode_length_s = 20
        # 48 (base proprio: lin_vel 3 + ang_vel 3 + gravity 3 + commands 3 +
        #     dof_pos 12 + dof_vel 12 + actions 12)
        # + 2  (gait phase sin/cos)
        # + 187 (height scan: len(measured_points_x)=17 * len(measured_points_y)=11,
        #        inherited from LeggedRobotCfg.terrain -- not overridden below)
        num_observations = 48 + 2 + 187

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'
        horizontal_scale = 0.05  # [m] divides every stair dimension below exactly
        vertical_scale = 0.005  # [m]
        border_size = 3.0  # [m] flat padding around the whole terrain grid
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True
        slope_treshold = 1.5  # slopes/steps steeper than this become vertical faces
        # measured_points_x/y: NOT overridden here -- inherit LeggedRobotCfg.terrain's
        # symmetric +/-0.8m x / +/-0.5m y grid (187 points) rather than Extreme Parkour's
        # forward-biased one, since our commands go backward and sideways-via-yaw just as
        # often as forward (EP's robot only ever runs forward)

        # feet_edge: how far around each foot (in each of +-x/+-y) to sample terrain
        # height when checking for a nearby edge, and how big a height difference
        # between those samples counts as "there's an edge right here"
        feet_edge_check_offset = 0.06  # [m]
        feet_edge_height_diff = 0.08  # [m]

        # ---- grid layout: one column per terrain type, one row per difficulty ----
        # difficulty here only controls per-step height unevenness -- the
        # nominal stair/staircase dimensions are fixed real-world targets,
        # not a curriculum axis
        terrain_types = ['flat', 'stairs_2step', 'stairs_3step', 'u_staircase']
        num_cols = 4
        num_rows = 5
        max_init_terrain_level = 0  # spawn everyone on the easiest row by default

        # every tile shares this footprint. length is sized to exactly fit
        # the u_staircase (entry 1.5 + 12 steps*0.25 + landing 1.5 = 6.0m),
        # its largest dimension. width is widened past the u_staircase's own
        # 3.4m (wall+corridor+shaft+corridor+wall) on purpose: on the
        # stairs_2step/3step terrain the whole width is now walkable stair
        # (see _gen_short_stairs) instead of a narrow 1.4m band with open
        # flat margins, specifically so a randomized spawn yaw can't drift
        # the robot off the stairs before it finishes crossing -- the
        # u_staircase's own structure just gets centered in the extra width
        # with flat padding on either side (see _gen_u_staircase)
        terrain_length = 6.0  # [m] along the robot's forward (+x) direction inside a tile
        terrain_width = 6.0  # [m] lateral extent of a tile

        # ---- real-world dimensions ----
        # step_depth/stair_width/landing_depth are fixed real-world targets
        # (they set the pixel layout below) and don't scale with difficulty.
        # step_height DOES scale with difficulty though -- same convention as
        # upstream legged_gym's own terrain curriculum -- so the easiest row
        # is a shorter, more forgiving riser and only the hardest row is the
        # full real-world 0.20m spec the robot actually has to handle.
        step_height_range = [0.10, 0.20]  # [m] nominal rise per step at (easiest, hardest) row
        step_depth = 0.25  # [m] nominal tread depth
        stair_width = 1.4  # [m] usable walking width of one u_staircase flight (real spec)
        landing_depth = 1.5  # [m] depth of the entry/mid/top platforms on the u_staircase
        u_staircase_num_steps = 12  # steps per flight
        # each new episode independently rolls ascend (spawn at the bottom entry,
        # chase wp0->wp7) vs descend (spawn at the top exit, chase wp7->wp0) -- same
        # geometry either way, just walking the waypoint chain in the other direction
        u_staircase_descend_prob = 0.5

        # 2/3-step terrain only (open, no walls, and now spans the tile's
        # full width -- see terrain_width above -- so diagonal approaches
        # stay possible without a risk of drifting off the stairs before
        # finishing the crossing). The lead-in/run-out length is fixed; the
        # mid platform absorbs whatever length is left in the tile so the
        # pattern fills the whole 6.0m tile instead of leaving a dead flat
        # strip at the end.
        short_stairs_lead_in = 1.0  # [m] flat approach before/after the stairs
        # random spawn yaw so diagonal crossing actually gets trained, not
        # just crossing straight-on. Safe now that the stairs span the
        # tile's full width (see terrain_width) -- a robot spawned at an
        # angle and given an independently-sampled (vx, vyaw) command may
        # still drift, but it drifts across MORE stairs, never off of them
        short_stairs_yaw_range = [-0.78, 0.78]  # [rad], ~+/-45 degrees

        # u_staircase only
        shaft_width = 0.4  # [m] width of the open central shaft between the two flights
        wall_thickness = 0.1  # [m]
        shaft_depth = 3.0  # [m] how far the open shaft "falls" below the corridor floor

        # difficulty-scaled protection, both measured relative to the LOCAL
        # floor height (so they track the stair profile instead of being a
        # flat-topped building wall):
        #  - outer/end-cap walls interpolate from a low guard-board at the
        #    easiest row up to a real enclosing wall at the hardest row
        #  - the shaft gets a protective lip at the easiest row (so early
        #    training isn't dominated by falling in) that narrows to nothing
        #    by the hardest row, exposing the bare open shaft
        wall_height_range = [0.12, 1.0]  # [m] guard-board height at (easiest, hardest) row
        shaft_curb_width = 0.1  # [m] lip width at the easiest row, shrinks to 0 at the hardest
        shaft_curb_height = 0.15  # [m] lip height above the local corridor floor

        # difficulty axis: per-step height jitter, scaled 0 -> max at the hardest row
        max_step_height_noise = 0.04  # [m]
        flat_max_height_noise = 0.03  # [m]

        # replaces the base class's hardcoded +/-1m random spawn offset
        # (way too big next to a 1.4m-wide flight or a 0.4m-wide shaft)
        spawn_xy_jitter = 0.1  # [m]

        # termination: every legitimate floor in this terrain set is at
        # z >= 0 (entry platforms start at 0, everything else is built up
        # from there) -- only the u_staircase shaft goes negative -- so a
        # single height threshold cleanly catches "fell into the shaft"
        # without needing a live height-scan query yet
        fall_height_threshold = -0.3  # [m] terminate if base z drops below this
        # terminate if the robot leaves the (row, col) tile it was assigned,
        # so reward/waypoint bookkeeping (which assumes that tile's geometry)
        # can't silently go stale, and so curriculum promotion can't be
        # fooled by displacement that happened outside the intended terrain
        bounds_margin = 0.5  # [m] slack added around the tile before it counts as "left"

        # curriculum promotion, evaluated once per env at the end of each
        # episode (see GO2StairsRobot._update_terrain_curriculum). Progress
        # is measured differently per terrain type: flat/short_stairs use
        # raw displacement from the tile's spawn point (nothing else to
        # measure against); u_staircase uses how far along its waypoint
        # chain the env got, since raw displacement is a bad proxy for a
        # switchback path (the top exit sits right next to the entry in x,
        # so a robot that fully climbed both flights would show almost no
        # net displacement even though it did the whole task)
        curriculum_demote_waypoint_frac = 0.25  # below this fraction of the u_staircase chain, drop a difficulty row

    class commands(LeggedRobotCfg.commands):
        # the real robot ignores |vx| < dead_zone and |vyaw| < dead_zone
        # (they behave identically to a zero command), so commands are
        # sampled to either land exactly on zero or clear the dead zone --
        # never inside it, which would just be wasted exploration
        dead_zone = 0.1  # [m/s] and [rad/s]
        lin_vel_x_range = [0.1, 1.5]  # [m/s] magnitude when nonzero (low end == dead_zone)
        ang_vel_yaw_range = [0.1, 2.0]  # [rad/s] magnitude when nonzero (low end == dead_zone); the high end also caps the waypoint-chase yaw-rate on the u_staircase

        # stratified sampling on flat/short-stairs terrain so "stand still"
        # and "pure spin" aren't crowded out by a 2D box's corners -- these
        # four should sum to 1
        stand_prob = 0.15  # vx=0, vyaw=0
        walk_prob = 0.30  # vx!=0, vyaw=0
        spin_prob = 0.25  # vx=0, vyaw!=0
        combo_prob = 0.30  # vx!=0, vyaw!=0

        # u_staircase: no "stand"/"spin" bucket -- the point of this terrain
        # is forward progress along the waypoint chain, so vx is always
        # nonzero and vyaw is entirely derived from waypoint-chasing (see
        # waypoint_heading_kp/ang_vel_yaw_range above), never sampled directly
        u_staircase_vx_range = [0.2, 0.8]  # [m/s]
        waypoint_reach_threshold = 0.3  # [m] distance at which the next waypoint becomes the target
        waypoint_heading_kp = 1.0  # P-gain converting heading error -> commanded yaw rate

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]  # x,y,z [m]  -- from go2_config.py
        default_joint_angles = {  # from go2_config.py
            'FL_hip_joint': 0.1,
            'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1,
            'RR_hip_joint': -0.1,

            'FL_thigh_joint': 0.8,
            'RL_thigh_joint': 1.,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 1.,

            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5,
        }

    class control(LeggedRobotCfg.control):  # from go2_config.py
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}  # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):  # from go2_config.py
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1

    class sim(LeggedRobotCfg.sim):
        class physx(LeggedRobotCfg.sim.physx):
            # base default (5) is sized for legged_gym's usual simple heightfields. Our
            # trimesh is much bigger/less regular (a full 5x4 grid of stairs/staircase
            # tiles, ~860k triangles after the horizontal_scale=0.05 pass) -- PhysX GPU
            # broadphase logs "PxgDynamicsMemoryConfig::foundLostAggregatePairsCapacity"
            # warnings ("the simulation will miss interactions") at the base default,
            # which manifests as robots silently sinking through the floor over several
            # steps (missed foot-ground contact pairs -> no ground reaction force that
            # step) on exactly the terrain/row combinations whose geometry happens to
            # stress the buffer hardest -- not a training/reward bug. Bumped well past
            # the capacities the warning itself reported needing (up to ~18M).
            default_buffer_size_multiplier = 40

    class rewards(LeggedRobotCfg.rewards):  # soft_dof_pos_limit/base_height_target from go2_config.py
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25  # now used as a LOCAL (relative-to-terrain) target, only
        # when the robot is standing still -- see GO2StairsRobot._reward_base_height

        # gait clock: phase advances at this frequency whenever |cmd| clears the dead
        # zone (frozen while genuinely standing still -- see GO2StairsRobot's gait_phase
        # update), so "spin in place" still cycles the legs instead of just twisting the
        # body, which was the whole point of adding this
        gait_freq = 2.0  # [Hz]
        foot_swing_height_target = 0.08  # [m] target foot clearance above the local terrain during swing

        class scales(LeggedRobotCfg.rewards.scales):
            # tracking (mirrors Extreme Parkour's tracking_goal_vel/tracking_yaw, but
            # against the commanded body-frame (vx, vyaw) directly rather than a world-
            # frame goal-direction -- see the design discussion for why: this task's
            # commands are externally driven, not autonomously chosen by the policy)
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.5
            # regularization (Extreme Parkour's list, adapted)
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -1.0
            dof_acc = -2.5e-7
            collision = -1.0
            action_rate = -0.1
            torques = -0.0002
            # was -10.0. With only_positive_rewards clipping the per-step total to >=0,
            # a term this large relative to the rest (~0.02-2 in magnitude) meant any step
            # where it fired swamped the whole sum, which then got floored to exactly 0 --
            # identical to "did nothing wrong at all" from the policy's perspective. That
            # erases the only signal telling it "stop pushing joints toward their limits",
            # which matches what training data showed: dof_pos crept steadily toward the
            # limits over the course of a rollout on 3 of 4 terrain types, and the
            # resulting shaft_fall rate (robot's legs buckling -> root sinks below the
            # fall-height floor) got WORSE, not better, over 450 iterations of training --
            # not normal early-training noise, which should trend down, not up. Brought in
            # line with the other regularization terms so it can't single-handedly zero
            # out the whole step's reward.
            dof_pos_limits = -1.0
            hip_pos = -0.5
            dof_error = -0.04
            feet_stumble = -1.0
            feet_edge = -1.0
            # gait phase -- replaces feet_air_time (set to 0 below): a step clock the
            # policy can see (obs) and be scored against is a much more direct fix for
            # "twists the body instead of moving the feet" than an air-time heuristic
            feet_air_time = 0.
            feet_swing_height = -2.0
            gait_contact = 0.2
            # u_staircase only: sparse bonus each time a new waypoint is reached, to help
            # early training across the awkward heading transition at the landing crossing
            # -- see the original waypoint design discussion. Was 0.5 -- raised 10x
            # alongside waypoint_dist_progress: if "survive safely without really climbing"
            # is already earning decent reward every step (see that scale's comment), the
            # one-off payout for committing to and finishing a risky multi-second climb
            # needs to be large enough to actually outweigh that safer alternative, not a
            # rounding error next to it.
            waypoint_progress = 5.0
            # u_staircase only: dense per-step reward for closing distance to the CURRENT
            # target waypoint (see _reward_waypoint_dist_progress) -- added after training
            # data showed the sparse bonus alone gave no gradient across a whole 12-step/
            # ~3m flight, letting robots get stuck (even slide backward) mid-climb.
            # Was 5.0 -- raised 10x after the first round of training data showed net
            # progress toward the goal over a full ~20s episode was ~0.008m (essentially
            # zero, not just slow): tracking_lin_vel was already near its max reward the
            # whole episode (matching COMMANDED body-frame forward velocity, satisfiable by
            # oscillating/stepping in place without net world-space displacement) and, at
            # the old scale, dwarfed this term's already-tiny signal, leaving PPO's gradient
            # dominated by "keep doing whatever gets tracking_lin_vel" with essentially no
            # pressure toward the much harder actually-climb-the-flight behavior.
            waypoint_dist_progress = 50.0
            # gated to standing-still episodes only, and measured relative to the local
            # terrain height, not a fixed world z -- see _reward_base_height
            base_height = -1.0

    class viewer(LeggedRobotCfg.viewer):
        # framed to see the whole terrain grid (5 rows x 4 cols of 6x3.4m
        # tiles, plus the 3m border) from above at an angle
        ref_env = 0
        pos = [-6, -6, 22]
        lookat = [15, 7, 0]


class GO2StairsCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        # was 0.01. Training data showed u_staircase stuck at a stable-but-not-progressing
        # local optimum (fall rate kept improving, near-zero net progress toward the goal
        # never budged even after a 10x reward-scale increase) -- consistent with a
        # risk-averse policy that found "survive safely without committing to a multi-
        # second climb" and had no incentive strong enough to explore the riskier
        # actually-climb behavior. A bit more entropy regularization encourages broader
        # exploration instead of collapsing onto the first safe-looking behavior.
        entropy_coef = 0.02

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'go2_stairs'
