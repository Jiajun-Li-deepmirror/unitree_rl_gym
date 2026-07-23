import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float

from legged_gym.envs.base.legged_robot import LeggedRobot

STAGE_STAND, STAGE_SIT, STAGE_JUMP, STAGE_AIR, STAGE_LAND = 0, 1, 2, 3, 4
NUM_STAGES = 5


class Backflip(LeggedRobot):
    """Go2 in-place backflip via a 5-stage reward schedule (Stand -> Sit ->
    Jump -> Air -> Land), following the staged-reward approach from "Stage-Wise
    Reward Shaping for Acrobatic Robots" (arxiv 2409.15755) rather than trying
    to shape the whole flip with one undifferentiated reward set. self.stage
    (per-env int, see STAGE_* above) drives which reward terms are active each
    step and is also appended to the observation so the policy can condition
    its behavior on it -- see compute_observations."""

    def _init_buffers(self):
        super()._init_buffers()
        self.stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _update_stage(self):
        height = self.root_states[:, 2]
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        num_contact = torch.sum(contact.float(), dim=1)
        stand_steps = int(self.cfg.rewards.stand_duration_s / self.dt)

        stage = self.stage.clone()
        stage = torch.where((stage == STAGE_STAND) & (self.episode_length_buf > stand_steps),
                             torch.full_like(stage, STAGE_SIT), stage)
        stage = torch.where((stage == STAGE_SIT) & (height < self.cfg.rewards.sit_to_jump_height),
                             torch.full_like(stage, STAGE_JUMP), stage)
        stage = torch.where((stage == STAGE_JUMP) & (num_contact == 0),
                             torch.full_like(stage, STAGE_AIR), stage)
        stage = torch.where((stage == STAGE_AIR) & (num_contact > 0),
                             torch.full_like(stage, STAGE_LAND), stage)
        self.stage = stage

    def _post_physics_step_callback(self):
        self._update_stage()
        super()._post_physics_step_callback()

    def check_termination(self):
        """Same base-contact termination as the base class, but the hard-coded
        pitch>1.0rad limit is dropped entirely (a backflip needs pitch to swing
        through ~2*pi -- that limit would terminate the episode the instant it
        starts rotating) and replaced with a roll-only safety cutoff (falling
        sideways is the real failure mode here, not pitching)."""
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.reset_buf |= torch.abs(self.rpy[:, 0]) > self.cfg.rewards.max_roll_rad
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) > 0:
            self.stage[env_ids] = STAGE_STAND

    def _reset_dofs(self, env_ids):
        """Tight additive noise around the nominal stance instead of the base
        class's +/-50% multiplicative scatter -- a repeatable acrobatic launch
        needs a consistent starting pose, not the wide randomization that's
        useful for robustness in a walking task."""
        self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(
            -0.05, 0.05, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """Same as the base class, except base velocity is zeroed (a clean
        launch) instead of randomized +/-0.5 m/s and rad/s."""
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device)
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 7:13] = 0.
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def compute_observations(self):
        super().compute_observations()
        stage_onehot = torch.nn.functional.one_hot(self.stage, num_classes=NUM_STAGES).float()
        self.obs_buf = torch.cat((self.obs_buf, stage_onehot), dim=-1)

    def _reward_stage_height(self):
        height = self.root_states[:, 2]
        is_stand_or_land = (self.stage == STAGE_STAND) | (self.stage == STAGE_LAND)
        is_sit = self.stage == STAGE_SIT
        target = torch.where(is_sit, torch.full_like(height, self.cfg.rewards.sit_height_target),
                              torch.full_like(height, self.cfg.rewards.stand_height_target))
        active = is_stand_or_land | is_sit
        return -torch.abs(height - target) * active.float()

    def _reward_jump_height(self):
        height = self.root_states[:, 2]
        is_jump_or_air = (self.stage == STAGE_JUMP) | (self.stage == STAGE_AIR)
        capped = torch.clamp(height, max=self.cfg.rewards.max_flip_height)
        return capped * is_jump_or_air.float()

    def _reward_flip_ang_vel(self):
        """Reward backward pitch rotation during Jump/Air. Sign NOT yet
        empirically verified against go2's actual body-frame convention --
        flip it if the learned flip goes forward instead of backward."""
        is_jump_or_air = (self.stage == STAGE_JUMP) | (self.stage == STAGE_AIR)
        return (-self.base_ang_vel[:, 1]) * is_jump_or_air.float()

    def _reward_flip_balance(self):
        # a backflip should be pure pitch rotation -- penalize any roll
        # creeping in during the airborne stages.
        is_jump_or_air = (self.stage == STAGE_JUMP) | (self.stage == STAGE_AIR)
        return -torch.square(self.rpy[:, 0]) * is_jump_or_air.float()

    def _reward_stand_orientation(self):
        is_grounded_stage = (self.stage == STAGE_STAND) | (self.stage == STAGE_SIT) | (self.stage == STAGE_LAND)
        return -torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * is_grounded_stage.float()

    def _reward_lin_vel_penalty(self):
        is_grounded_stage = (self.stage == STAGE_STAND) | (self.stage == STAGE_SIT) | (self.stage == STAGE_LAND)
        penalty = torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1) + torch.square(self.base_ang_vel[:, 2])
        return -penalty * is_grounded_stage.float()

    def _reward_style(self):
        # keep joints close to the default pose throughout -- small weight on
        # purpose, this is a mild regularizer, not meant to fight the large
        # crouch/push-off/tuck departures from default the flip needs.
        return -torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
