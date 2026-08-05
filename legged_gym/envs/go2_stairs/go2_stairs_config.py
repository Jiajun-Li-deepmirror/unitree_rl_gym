from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# height-scan grid, in base frame (x fwd/back, y left/right). Same point counts as the base
# class default (17 x 11 = 187) but x starts at 0 (forward-only) instead of being centered on
# the base, since stairs are only ever climbed/descended in front of the robot.
MEASURED_POINTS_X = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
MEASURED_POINTS_Y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
NUM_HEIGHT_POINTS = len(MEASURED_POINTS_X) * len(MEASURED_POINTS_Y)

class GO2StairsCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_observations = 48 + 2 + 3 + NUM_HEIGHT_POINTS  # proprio (48) + gait sin/cos phase (2) + command-mode one-hot (3) + raw height-scan points

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.,   # [rad]
            'RL_hip_joint': 0.,   # [rad]
            'FR_hip_joint': 0. ,  # [rad]
            'RR_hip_joint': 0.,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh'
        curriculum = True
        measure_heights = True
        horizontal_scale = 0.05 # [m], finer than the base default so stair edges are resolved sharply
        vertical_scale = 0.05
        slope_treshold = 0.3
        measured_points_x = MEASURED_POINTS_X
        measured_points_y = MEASURED_POINTS_Y
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, flat]
        terrain_proportions = [0.0, 0.0, 0.4, 0.4, 0.0, 0.2]

        # single continuously-ascending U-shaped staircase, used by play_keyboard.py
        u_shape_playground = False
        class u_shape:
            num_steps = 12       # steps per flight
            step_width = 0.25     # [m] depth of each step
            step_height = 0.2   # [m] rise of each step
            flight_width = 2.0   # [m] width of each flight
            platform_size = 1.0  # [m] flat spawn run-up / mid-turn landing length
            top_platform_size = 1.5  # [m] flat landing at the top of flight 2

    class camera:
        use_camera = False
        # resolution/fps: Depth Output Resolution "Up to 544 x 640" @ "Up to 15 fps"
        width = 544
        height = 640
        scale = 0.5  # uniform scale factor applied to width/height (e.g. 0.5 to halve resolution)
        fps = 15
        near_plane = 1e-5
        far_plane = 5.0  # [m]
        horizontal_fov = 87.0  # [deg]
        mount_body = "Head_lower"  # front/chin mount point already present on the go2 URDF
        mount_pos = [0.05, 0.0, 0.15]  # local offset from mount_body's own origin [m]
        mount_pitch = 0.5236  # [rad] ~30 deg downward tilt, to see the ground/stairs ahead
        fov_viz_length = 0.1  # [m] length of the small FOV pyramid drawn in play_keyboard.py

    class commands( LeggedRobotCfg.commands ):
        curriculum = False
        heading_command = False
        command_proportions = [0.0, 0.0, 1.0]
        class command_curriculum:
            enabled = True            # set False to keep command_proportions fixed (e.g. for play/eval)
            increment = 0.05          # added to the stand & rotate weights each interval below
            increment_interval = 2000 # [iterations]
            num_steps_per_env = 24    # must match GO2StairsCfgPPO.runner.num_steps_per_env
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [0.0, 1.2]     # min max [m/s], walk mode only
            lin_vel_y = [0.0, 0.0]     # min max [m/s], walk mode only
            ang_vel_yaw = [-0.75, 0.75] # min max [rad/s]; rotate-in-place mode, and the clip bound for walk mode's heading-derived vyaw
            heading = [-3.14, 3.14]    # walk mode only

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        soft_torque_limit = 0.4
        base_height_target = 0.25
        max_contact_force = 40. # forces above this value are penalized
        tracking_sigma = 0.2 # tracking reward = exp(-error^2/sigma)
        edge_height_threshold = 0.03 # [m], min height jump between neighboring terrain cells to count as a stair edge
        cycle_time = 0.5 # [s], gait phase cycle period used for the obs sin/cos phase encoding
        feet_swing_height_target = 0.10 # [m], target swing-foot clearance above the terrain while rotating in place
        class scales( LeggedRobotCfg.rewards.scales ):
            # regularization
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -1.
            base_height = -1.0 # flat terrain only (see is_flat_terrain gating in go2_stairs_env.py)
            dof_acc = -2.5e-7
            collision = -10.
            action_rate = -0.1
            delta_torques = -1.0e-7
            torques = -0.00001
            dof_pos_limits = -10.0
            hip_pos = -0.5
            dof_error = -0.04

            feet_stumble = -1.
            feet_edge = -1.

            # rotate-in-place/walk only (see _command_mode gating in go2_stairs_env.py)
            gait_phase = 0.4
            # rotate-in-place/walk + flat terrain only (feet_swing_height_target isn't meaningful on stairs)
            feet_swing_height = -10.0

            # stand-still only, any terrain
            stand_still_contact = 0.2

    class height_encoder:
        # MLP that encodes the raw height-scan points into a latent appended to the observation
        # (trained jointly with the policy - see ActorCriticHeightEncoder in legged_gym/algorithms/)
        hidden_dims = [128, 64]
        latent_dim = 32

class GO2StairsCfgPPO( LeggedRobotCfgPPO ):
    # trains the height-scan encoder jointly with the policy (see legged_gym/algorithms/)
    runner_class_name = 'HeightEncoderOnPolicyRunner'
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCriticHeightEncoder'
        run_name = ''
        experiment_name = 'stairs_go2'
        max_iterations = 6000 # must match cfg.commands.command_curriculum's increment schedule
