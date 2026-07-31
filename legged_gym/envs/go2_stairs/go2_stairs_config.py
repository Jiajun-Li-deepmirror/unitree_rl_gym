from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

HEIGHT_LATENT_DIM = 32  # dim of the height-scan encoder's output latent (see height_encoder below)

# Shape of the height-scan grid, in the base frame: x is forward(+)/backward(-), y is
# left(+)/right(-). Edit these two lists to reshape the scan - e.g. drop the negative entries
# in MEASURED_POINTS_X to stop sampling points behind the robot. Kept as module constants
# (mirrored onto GO2StairsCfg.height_scan below) rather than only inside the class, so
# NUM_HEIGHT_POINTS can be computed here before GO2StairsCfg itself is defined - edit these,
# not the copies under `class height_scan`, or num_observations will fall out of sync.
MEASURED_POINTS_X = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]  # forward-only, same 17 points, 0.05 spacing
MEASURED_POINTS_Y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
# proprio (48, same layout as go2) + raw height-scan points. The height-scan encoder MLP now lives
# inside ActorCriticHeightEncoder (see legged_gym/algorithms/height_actor_critic.py) so it's trained
# end-to-end by PPO, instead of being a fixed/untrained encoder baked into the env observation - so
# the env has to expose the *raw* height scan here, not an already-encoded latent.
NUM_HEIGHT_POINTS = len(MEASURED_POINTS_X) * len(MEASURED_POINTS_Y)

class GO2StairsCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        # num_envs = 2048
        num_observations = 48 + NUM_HEIGHT_POINTS  # proprio (48) + raw height-scan points

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
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.0, 0.0, 0.5, 0.5, 0.0]

        # if True, replaces the curriculum terrain grid with a single continuously-ascending
        # U-shaped (switchback) staircase - one flight up, a landing that turns around, a
        # second flight up. Used by play_keyboard.py for a one-robot showcase run.
        u_shape_playground = False
        class u_shape:
            num_steps = 12       # steps per flight
            step_width = 0.3     # [m] depth of each step
            step_height = 0.12   # [m] rise of each step
            flight_width = 2.0   # [m] width of each flight
            platform_size = 1.0  # [m] flat spawn run-up / mid-turn landing length
            top_platform_size = 1.5  # [m] flat landing at the top of flight 2

    class commands( LeggedRobotCfg.commands ):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4
        resampling_time = 10. # time before commands are changed [s]
        heading_command = False
        # how each resampled command is drawn: [stand still, rotate in place, normal velocity]
        command_proportions = [0.2, 0.2, 0.6]
        rotate_in_place_ang_vel_min = 0.1 # [rad/s], floor on |vyaw| so "rotate" never degrades to ~0
        class ranges:
            lin_vel_x = [0.0, 1.2]   # min max [m/s], slower than flat ground for stair climbing
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-0.75, 0.75] # min max [rad/s]
            heading = [-3.14, 3.14]

    class height_scan:
        # Shape of the height-scan grid - see the module-level MEASURED_POINTS_X/Y comment
        # above for how to edit this (e.g. to drop the points behind the robot).
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
        # not in the datasheet (only "<2% accuracy @ 3m" is specced) - pick a sensible usable
        # range and tune later against the real sensor if needed
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
            # feet_air_time = 1.0
            collision = -1.
            stumble = -1.0
            action_rate = -0.01
            stand_still = -0.5
            dof_pos_limits = -10.0

            gait_phase = 0.18
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
        run_name = 'mlp_ori_height'
        experiment_name = 'stairs_go2'
        max_iterations = 3000
        save_interval = 100
