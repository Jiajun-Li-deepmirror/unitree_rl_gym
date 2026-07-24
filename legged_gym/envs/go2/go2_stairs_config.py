from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO

class GO2StairsCfg( GO2RoughCfg ):
    class env( GO2RoughCfg.env ):
        num_observations = 235  # 48 base + 17*11=187 height-scan points (trapezoid layout, same point count as base)

    class terrain( GO2RoughCfg.terrain ):
        curriculum = True
        # [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.05, 0.05, 0.4, 0.4, 0.1]

    class commands( GO2RoughCfg.commands ):
        curriculum = False
        max_curriculum = 1.2
        heading_command = True
        class ranges( GO2RoughCfg.commands.ranges ):
            lin_vel_x = [0.0, 1.2]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [-0.75, 0.75]
            heading = [-3.14, 3.14]

    class rewards( GO2RoughCfg.rewards ):
        edge_height_threshold_m = 0.03
        class scales( GO2RoughCfg.rewards.scales ):
            feet_contact_forces = -0.01
            feet_slip = -0.04
            feet_stumble = -1.0
            feet_edge = -0.25
            no_fly = 0.1
            base_height = -4.0
            hip_default_pose = -0.2
            orientation = -2.0

    class height_scan:
        near_edge_x = 0.2            
        near_edge_half_width = 0.14  
        depth_range = 1.0            
        n_forward = 17               
        n_lateral = 11               

    class depth_camera:
        horizontal_fov = 58.0
        vertical_fov = 87.0
        min_range = 1e-6
        max_range = 3.5
        mount_forward_offset = 0.30
        mount_height_offset = 0.1
        camera_pitch_deg = 20.0

        use_camera = False
        native_image_width = 544
        native_image_height = 640
        resolution_scale = 0.5
        fps = 15

        noise_relative_std = 0.02

class GO2StairsCfgPPO( GO2RoughCfgPPO ):
    class policy( GO2RoughCfgPPO.policy ):
        num_prop_obs = 48
        encoder_hidden_dims = [128, 64]
        latent_dim = 32

    class runner( GO2RoughCfgPPO.runner ):
        # policy_class_name = 'ActorCritic'
        policy_class_name = 'ActorCriticHeightEncoder'
        run_name = ''
        experiment_name = 'stairs_go2'
