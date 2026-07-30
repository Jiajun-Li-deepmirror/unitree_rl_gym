from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

HEIGHT_LATENT_DIM = 32  # dim of the MLP-encoded height-scan latent appended to the observation

class GO2StairsCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_observations = 48 + HEIGHT_LATENT_DIM  # proprio (48, same layout as go2) + height-scan latent

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

    class commands( LeggedRobotCfg.commands ):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4
        resampling_time = 10. # time before commands are changed [s]
        heading_command = False
        class ranges:
            lin_vel_x = [0.0, 1.2]   # min max [m/s], slower than flat ground for stair climbing
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-0.75, 0.75] # min max [rad/s]
            heading = [-3.14, 3.14]

    class height_encoder:
        # MLP that encodes the raw height-scan points into a latent appended to the observation
        hidden_dims = [128, 64]
        latent_dim = HEIGHT_LATENT_DIM

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
        
        cycle_time = 0.5 # trot gait period [s]
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.0002
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.
            feet_air_time = 1.0
            collision = -1.
            feet_stumble = -0.0
            action_rate = -0.01
            stand_still = -0.
            dof_pos_limits = -10.0

            # gait_phase = 0.18
            gait_phase = 0.

class GO2StairsCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'stairs_go2'
