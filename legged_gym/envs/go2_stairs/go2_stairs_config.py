from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

HEIGHT_LATENT_DIM = 32  # dim of the height-scan encoder's output latent (see height_encoder below)

# height-scan grid, in base frame (x fwd/back, y left/right) - edit these two lists to reshape it
MEASURED_POINTS_X = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]  # forward-only, same 17 points, 0.05 spacing
MEASURED_POINTS_Y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
# proprio(48) + raw height-scan points; encoder lives in ActorCriticHeightEncoder, not here
NUM_HEIGHT_POINTS = len(MEASURED_POINTS_X) * len(MEASURED_POINTS_Y)

class GO2StairsCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        # num_envs = 2048
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
        horizontal_scale = 0.05
        slope_treshold = 0.3
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.0, 0.0, 0.5, 0.5, 0.0]

        # single continuously-ascending U-shaped staircase, used by play_keyboard.py
        u_shape_playground = False
        class u_shape:
            num_steps = 12       # steps per flight
            step_width = 0.25     # [m] depth of each step
            step_height = 0.2   # [m] rise of each step
            flight_width = 2.0   # [m] width of each flight
            platform_size = 1.0  # [m] flat spawn run-up / mid-turn landing length
            top_platform_size = 1.5  # [m] flat landing at the top of flight 2

        class wave_stairs:
            num_steps_per_side = 8
            step_width = 0.25        # [m] depth of each step
            lead_in_size = 1.0       # [m] flat trough before the climb starts (also the descent's landing)
            peak_platform_size = 0.5  # [m] flat platform at each ridge's peak

    class commands( LeggedRobotCfg.commands ):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4
        resampling_time = 10. # time before commands are changed [s]
        # kept False so the base class doesn't apply heading->vyaw tracking to ALL envs every
        # step (that would override stand-still/rotate-in-place's own direct vyaw too) - GO2Stairs
        # implements the same tracking itself in _post_physics_step_callback, scoped to walking envs
        heading_command = False
        # mostly-walking early on, so stairs-climbing is the dominant skill being trained instead
        # of being diluted by gait_phase/feet_swing_height rewards firing during stand-still/
        # rotate-in-place - see command_curriculum below for the later switch.
        command_proportions = [0.05, 0.05, 0.9]
        rotate_in_place_ang_vel_min = 0.1 # [rad/s], floor on |vyaw| so "rotate" never degrades to ~0
        class command_curriculum:
            switch_iteration = 8000
            num_steps_per_env = 24
            late_proportions = [0.2, 0.2, 0.6]
        class ranges:
            lin_vel_x = [0.0, 1.2]   # min max [m/s] - forward walking disabled for this phase
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-0.75, 0.75] # min max [rad/s] - also the clip bound for heading-derived vyaw
            heading = [-0.5, 0.5] # centered on the climb direction (+x)

    class height_scan:
        # grid shape - see the module-level MEASURED_POINTS_X/Y comment above to edit it
        measured_points_x = MEASURED_POINTS_X
        measured_points_y = MEASURED_POINTS_Y

    class height_encoder:
        # MLP that encodes the raw height-scan points into a latent appended to the observation
        hidden_dims = [128, 64]
        latent_dim = HEIGHT_LATENT_DIM

    class camera:
        use_camera = False
        # resolution/fps: Depth Output Resolution "Up to 544 x 640" @ "Up to 15 fps"
        width = 544
        height = 640
        scale = 0.5  # uniform scale factor applied to width/height (e.g. 0.5 to halve resolution)
        fps = 15
        # near_plane = 0.19  # [m] Min-Z at max resolution
        near_plane = 1e-5
        # not in the datasheet (only "<2% accuracy @ 3m" is specced) - pick a sensible default
        far_plane = 5.0  # [m]
        horizontal_fov = 87.0  # [deg] not in the datasheet either - typical for this class of stereo module
        mount_body = "Head_lower"  # front/chin mount point already present on the go2 URDF
        mount_pos = [0.05, 0.0, 0.15]  # local offset from mount_body's own origin [m]
        mount_pitch = 0.5236  # [rad] ~30 deg downward tilt, to see the ground/stairs ahead
        fov_viz_length = 0.1  # [m] length of the small FOV pyramid drawn in play_keyboard.py

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
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        edge_height_threshold = 0.03 # [m], min height jump between neighboring terrain cells to count as a stair edge
        cycle_time = 0.5 # trot gait period [s]
        feet_swing_height_target = 0.10 # [m], target swing-foot clearance above the terrain while gaited
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.2
            torques = -0.0002
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -2.0
            feet_air_time = 1.0
            collision = -1.0
            stumble = -1.0
            action_rate = -0.01
            stand_still = -0.5
            dof_pos_limits = -10.0

            gait_phase = .4
            feet_swing_height = -10.0
            feet_edge = -1.0
            hip_pos = -1.0
            # stand_still_contact = 0.2
            # dof_pos_deviation = -0.1

class GO2StairsCfgPPO( LeggedRobotCfgPPO ):
    # trains the height-scan encoder jointly with the policy (see legged_gym/algorithms/)
    runner_class_name = 'HeightEncoderOnPolicyRunner'
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCriticHeightEncoder'
        run_name = ''
        experiment_name = 'stairs_go2'
        max_iterations = 10000
        save_interval = 100
