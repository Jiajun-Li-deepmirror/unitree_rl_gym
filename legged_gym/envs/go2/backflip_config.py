from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO

class BackflipCfg( GO2RoughCfg ):
    class env( GO2RoughCfg.env ):
        num_observations = 48 + 5  # 48 base proprioception + 5-dim one-hot stage indicator
        episode_length_s = 4  # a backflip is over in ~1-2s; short episodes for tight signal

    class terrain( GO2RoughCfg.terrain ):
        mesh_type = 'plane'  # flat ground only, no height-scan needed for this task
        measure_heights = False
        curriculum = False

    class commands( GO2RoughCfg.commands ):
        heading_command = False  # unused -- tracking_lin_vel/ang_vel scales are zeroed below

    class domain_rand( GO2RoughCfg.domain_rand ):
        push_robots = False  # a mid-flip push is destructive, not a useful robustness signal

    class rewards( GO2RoughCfg.rewards ):
        # go2's own natural standing height at its default joint angles (see
        # earlier forward-kinematics check in this project: ~0.300m), not an
        # arbitrary number -- Stand/Land target this.
        stand_height_target = 0.30
        sit_height_target = 0.15          # crouch depth before push-off, PLACEHOLDER
        sit_to_jump_height = 0.20         # Sit->Jump trigger threshold, PLACEHOLDER
        max_flip_height = 0.6             # cap on the Jump/Air height reward, PLACEHOLDER
        max_roll_rad = 0.7                # Air-stage roll safety cutoff (falling sideways), PLACEHOLDER
        stand_duration_s = 0.3            # Stand stage fixed duration before auto-advancing to Sit

        class scales( GO2RoughCfg.rewards.scales ):
            tracking_lin_vel = 0.
            tracking_ang_vel = 0.
            # all of the below are PLACEHOLDER starting weights -- see the plan's
            # "本次不做" note: expect to retune after watching the first few
            # training runs, same as go2_stairs's reward-tuning experience.
            stage_height = 2.0
            jump_height = 1.0
            flip_ang_vel = 1.0
            flip_balance = 1.0
            stand_orientation = 1.0
            lin_vel_penalty = 0.5
            style = -0.05

class BackflipCfgPPO( GO2RoughCfgPPO ):
    class runner( GO2RoughCfgPPO.runner ):
        run_name = ''
        experiment_name = 'backflip_go2'
