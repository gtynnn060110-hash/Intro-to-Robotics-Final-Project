import numpy as np
from gymnasium import spaces

try:
    import mujoco
except Exception:  # pragma: no cover - mujoco may be unavailable in CI
    mujoco = None

from .unitree_a1_env import UnitreeA1Env


class UnitreeA1WalkEnv(UnitreeA1Env):
    """Gait-aware fixed-speed walking task.

    The original walking task only rewarded forward velocity and survival. That
    lets PPO exploit high-frequency joint chatter. This version exposes a gait
    clock and rewards a low-frequency diagonal trot: stance feet should stay in
    contact and not slip; swing feet should leave the ground with clearance.
    """

    LEG_ORDER = ("FR", "FL", "RR", "RL")
    LEG_PHASE_OFFSETS = np.array([0.0, np.pi, np.pi, 0.0], dtype=np.float64)

    def __init__(
        self,
        model_path: str,
        target_vx: float = 0.2,
        reset_noise: float = 0.0,
        terrain_friction: float = 1.5,
        terrain_height_scale: float | None = None,
        gait_frequency: float = 0.95,
        randomize_gait_phase: bool = True,
        swing_height: float = 0.045,
        stance_clearance: float = 0.012,
        foot_contact_weight: float = 1.2,
        swing_clearance_weight: float = 0.7,
        stance_slip_weight: float = 0.25,
        gait_symmetry_weight: float = 0.15,
        support_clearance_threshold: float = 0.035,
        min_support_contacts: float = 2.0,
        support_penalty_weight: float = 5.0,
        rear_air_penalty_weight: float = 8.0,
        stance_contact_penalty_weight: float = 2.0,
        overspeed_deadband: float = 0.03,
        overspeed_weight: float = 6.0,
        overspeed_quadratic_weight: float = 16.0,
        forward_reward_weight: float = 3.0,
        progress_reward_weight: float = 5.0,
        backward_penalty_weight: float = 8.0,
        low_speed_penalty_weight: float = 25.0,
        low_speed_fraction: float = 0.75,
        speed_reward_sharpness: float = 10.0,
        upright_reward_weight: float = 1.5,
        height_reward_weight: float = 1.2,
        height_reward_sharpness: float = 24.0,
        height_target_offset: float = 0.0,
        low_height_penalty_weight: float = 6.0,
        low_height_penalty_quadratic_weight: float = 18.0,
        lateral_penalty_weight: float = 1.0,
        yaw_penalty_weight: float = 0.25,
        pitch_tilt_penalty_weight: float = 6.0,
        roll_tilt_penalty_weight: float = 4.0,
        ang_vel_penalty_weight: float = 0.035,
        joint_vel_penalty_weight: float = 0.0025,
        pose_penalty_weight: float = 0.02,
        action_penalty_weight: float = 0.0025,
        smooth_penalty_weight: float = 0.006,
        gait_clock_obs: bool = True,
        use_trot_reference: bool = False,
        trot_frequency: float | None = None,
        trot_thigh_amplitude: float = 0.0,
        trot_calf_amplitude: float = 0.0,
        trot_stance_calf_amplitude: float = 0.0,
        **kwargs,
    ):
        kwargs.setdefault("task", "walk")
        kwargs.setdefault("frame_skip", 8)
        super().__init__(model_path=model_path, **kwargs)

        self.target_vx = float(target_vx)
        self.reset_noise = float(max(reset_noise, 0.0))
        self.terrain_friction = float(max(terrain_friction, 0.05))
        self._terrain_hfield = self._find_hfield("terrain")
        self._default_terrain_height_scale = (
            float(self.model.hfield_size[self._terrain_hfield, 2]) if self._terrain_hfield >= 0 else None
        )
        if terrain_height_scale is None:
            terrain_height_scale = self._default_terrain_height_scale
        self.terrain_height_scale = None if terrain_height_scale is None else float(max(terrain_height_scale, 1e-4))

        self.gait_frequency = float(max(gait_frequency, 0.05))
        self.randomize_gait_phase = bool(randomize_gait_phase)
        self.gait_phase = 0.0
        self.swing_height = float(max(swing_height, 0.0))
        self.stance_clearance = float(max(stance_clearance, 0.0))
        self.foot_contact_weight = float(max(foot_contact_weight, 0.0))
        self.swing_clearance_weight = float(max(swing_clearance_weight, 0.0))
        self.stance_slip_weight = float(max(stance_slip_weight, 0.0))
        self.gait_symmetry_weight = float(max(gait_symmetry_weight, 0.0))
        self.support_clearance_threshold = float(max(support_clearance_threshold, 0.0))
        self.min_support_contacts = float(max(min_support_contacts, 0.0))
        self.support_penalty_weight = float(max(support_penalty_weight, 0.0))
        self.rear_air_penalty_weight = float(max(rear_air_penalty_weight, 0.0))
        self.stance_contact_penalty_weight = float(max(stance_contact_penalty_weight, 0.0))

        self.overspeed_deadband = float(max(overspeed_deadband, 0.0))
        self.overspeed_weight = float(max(overspeed_weight, 0.0))
        self.overspeed_quadratic_weight = float(max(overspeed_quadratic_weight, 0.0))
        self.forward_reward_weight = float(max(forward_reward_weight, 0.0))
        self.progress_reward_weight = float(progress_reward_weight)
        self.backward_penalty_weight = float(max(backward_penalty_weight, 0.0))
        self.low_speed_penalty_weight = float(max(low_speed_penalty_weight, 0.0))
        self.low_speed_fraction = float(max(low_speed_fraction, 0.0))
        self.speed_reward_sharpness = float(max(speed_reward_sharpness, 1e-6))
        self.upright_reward_weight = float(max(upright_reward_weight, 0.0))
        self.height_reward_weight = float(max(height_reward_weight, 0.0))
        self.height_reward_sharpness = float(max(height_reward_sharpness, 1e-6))
        self.height_target_offset = float(height_target_offset)
        self.low_height_penalty_weight = float(max(low_height_penalty_weight, 0.0))
        self.low_height_penalty_quadratic_weight = float(max(low_height_penalty_quadratic_weight, 0.0))
        self.lateral_penalty_weight = float(max(lateral_penalty_weight, 0.0))
        self.yaw_penalty_weight = float(max(yaw_penalty_weight, 0.0))
        self.pitch_tilt_penalty_weight = float(max(pitch_tilt_penalty_weight, 0.0))
        self.roll_tilt_penalty_weight = float(max(roll_tilt_penalty_weight, 0.0))
        self.ang_vel_penalty_weight = float(max(ang_vel_penalty_weight, 0.0))
        self.joint_vel_penalty_weight = float(max(joint_vel_penalty_weight, 0.0))
        self.pose_penalty_weight = float(max(pose_penalty_weight, 0.0))
        self.action_penalty_weight = float(max(action_penalty_weight, 0.0))
        self.smooth_penalty_weight = float(max(smooth_penalty_weight, 0.0))

        # Backward-compatible names accepted by older scripts. The reference is
        # optional and disabled by default; the gait reward is the primary guide.
        self.use_trot_reference = bool(use_trot_reference)
        self.trot_frequency = self.gait_frequency if trot_frequency is None else float(max(trot_frequency, 0.05))
        self.trot_thigh_amplitude = float(max(trot_thigh_amplitude, 0.0))
        self.trot_calf_amplitude = float(max(trot_calf_amplitude, 0.0))
        self.trot_stance_calf_amplitude = float(max(trot_stance_calf_amplitude, 0.0))

        self.gait_clock_obs = bool(gait_clock_obs)
        if self.gait_clock_obs:
            base_dim = int(np.prod(self.observation_space.shape))
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(base_dim + 7,),
                dtype=np.float32,
            )

        self._ordered_foot_geom_ids = self._get_ordered_foot_geom_ids()
        self._foot_radii = np.array([self.model.geom_size[g, 0] for g in self._ordered_foot_geom_ids], dtype=np.float64)
        self.prev_foot_xy = np.zeros((len(self._ordered_foot_geom_ids), 2), dtype=np.float64)
        self.initial_x = 0.0
        self.prev_x = 0.0
        self._configure_terrain()

    def set_target_vx(self, target_vx: float):
        self.target_vx = float(target_vx)

    def set_reset_noise(self, reset_noise: float):
        self.reset_noise = float(max(reset_noise, 0.0))

    def set_terrain_friction(self, terrain_friction: float):
        self.terrain_friction = float(max(terrain_friction, 0.05))
        self._configure_terrain()

    def set_terrain_height_scale(self, terrain_height_scale: float):
        self.terrain_height_scale = float(max(terrain_height_scale, 1e-4))
        self._configure_terrain()

    def _find_hfield(self, name):
        if mujoco is None:
            return -1
        try:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_HFIELD, name)
        except Exception:
            return -1

    def _configure_terrain(self):
        for geom_id in self._robot_collision_geom_ids:
            self.model.geom_friction[geom_id, 0] = self.terrain_friction
        if self._terrain_geom >= 0:
            self.model.geom_friction[self._terrain_geom, 0] = self.terrain_friction
            self.model.geom_friction[self._terrain_geom, 1] = 0.01
            self.model.geom_friction[self._terrain_geom, 2] = 0.001
        if self._terrain_hfield >= 0 and self.terrain_height_scale is not None:
            self.model.hfield_size[self._terrain_hfield, 2] = self.terrain_height_scale

    def _get_ordered_foot_geom_ids(self):
        by_leg = {}
        for geom_id in self._foot_geom_ids:
            body_id = int(self.model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if not body_name:
                continue
            for leg in self.LEG_ORDER:
                if body_name.startswith(f"{leg}_"):
                    by_leg[leg] = geom_id
        return [by_leg[leg] for leg in self.LEG_ORDER if leg in by_leg]

    def _gait_clock_obs(self):
        if not self.gait_clock_obs:
            return np.zeros(0, dtype=np.float32)
        desired_contact, swing_amount = self._desired_gait_state(self.gait_phase)
        return np.concatenate(
            [
                np.array([np.sin(self.gait_phase), np.cos(self.gait_phase), self.target_vx], dtype=np.float32),
                desired_contact.astype(np.float32),
            ]
        )

    def _get_obs_raw(self):
        base_obs = super()._get_obs_raw()
        return np.concatenate([base_obs, self._gait_clock_obs()]).astype(np.float32)

    def _desired_gait_state(self, phase):
        leg_phase = phase + self.LEG_PHASE_OFFSETS[: len(self._ordered_foot_geom_ids)]
        swing_amount = np.maximum(0.0, np.sin(leg_phase))
        desired_contact = (swing_amount <= 0.0).astype(np.float64)
        return desired_contact, swing_amount

    def _foot_contacts(self):
        contacts = np.zeros(len(self._ordered_foot_geom_ids), dtype=np.float64)
        foot_set = set(self._ordered_foot_geom_ids)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1 = int(contact.geom1)
            g2 = int(contact.geom2)
            if self._terrain_geom not in (g1, g2):
                continue
            other = g2 if g1 == self._terrain_geom else g1
            if other in foot_set:
                contacts[self._ordered_foot_geom_ids.index(other)] = 1.0
        return contacts

    def _foot_state(self, dt):
        count = len(self._ordered_foot_geom_ids)
        positions = np.zeros((count, 3), dtype=np.float64)
        clearances = np.zeros(count, dtype=np.float64)
        for i, geom_id in enumerate(self._ordered_foot_geom_ids):
            pos = np.asarray(self.data.geom_xpos[geom_id], dtype=np.float64).copy()
            positions[i] = pos
            terrain_z = self._raycast_terrain_height(pos[:2], pos[2])
            if terrain_z is None:
                terrain_z = 0.0
            clearances[i] = float(pos[2] - terrain_z - self._foot_radii[i])
        foot_xy_vel = (positions[:, :2] - self.prev_foot_xy) / max(dt, 1e-8)
        self.prev_foot_xy = positions[:, :2].copy()
        return positions, clearances, foot_xy_vel, self._foot_contacts()

    def _trot_reference_offset(self):
        offset = np.zeros(self.n_joints, dtype=np.float32)
        if not self.use_trot_reference:
            return offset
        phase = float(self.gait_phase)
        for leg_idx, leg_start in enumerate((0, 3, 6, 9)):
            wave = float(np.sin(phase + self.LEG_PHASE_OFFSETS[leg_idx]))
            swing = max(0.0, wave)
            stance = max(0.0, -wave)
            offset[leg_start + 1] = self.trot_thigh_amplitude * wave
            offset[leg_start + 2] = -self.trot_calf_amplitude * swing + self.trot_stance_calf_amplitude * stance
        return offset

    def reset(self, *, seed=None, options=None):
        self._configure_terrain()
        super().reset(seed=seed, options={"task": "walk"})
        self._configure_terrain()
        self.current_task = "walk"
        self.steps = 0
        self.last_action.fill(0.0)
        if self.randomize_gait_phase:
            self.gait_phase = float(self.np_random.uniform(0.0, 2.0 * np.pi))
        else:
            self.gait_phase = 0.0

        if self.reset_noise > 0.0:
            rng = self.np_random
            roll = rng.normal(0.0, 0.04 * self.reset_noise)
            pitch = rng.normal(0.0, 0.04 * self.reset_noise)
            yaw = rng.normal(0.0, 0.12 * self.reset_noise)
            self.data.qpos[3:7] = self._quat_from_euler(roll, pitch, yaw)
            joint_noise = rng.normal(0.0, 0.04 * self.reset_noise, size=self.n_joints)
            self.data.qpos[-self.n_joints :] = np.clip(self.standing_ctrl + joint_noise, self.ctrl_low, self.ctrl_high)
            self.data.qvel[:6] = rng.normal(0.0, 0.03 * self.reset_noise, size=6)
            self.data.qvel[-self.n_joints :] = rng.normal(0.0, 0.05 * self.reset_noise, size=self.n_joints)
            self.data.ctrl[: self.n_joints] = self.standing_ctrl
            mujoco.mj_forward(self.model, self.data)
            self._raise_robot_above_terrain(clearance=0.02)

        self.initial_x = float(self.data.qpos[0])
        self.prev_x = self.initial_x
        if self._ordered_foot_geom_ids:
            self.prev_foot_xy = np.array([self.data.geom_xpos[g, :2] for g in self._ordered_foot_geom_ids])
        self._reset_progress_trackers()
        _, _, _, initial_upright, initial_height_error = self._posture_metrics()
        self.initial_upright = float(initial_upright)
        self.initial_height_error = float(initial_height_error)

        obs = self._normalize_obs(self._get_obs_raw(), update=True)
        return obs, self._walking_info({}, False, False, False, False)

    def step(self, action):
        phase_before = float(self.gait_phase)
        action = np.asarray(action, dtype=np.float32).reshape(self.n_joints,)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        prev_action = self.last_action.copy()

        target = self.standing_ctrl + self._trot_reference_offset() + self.action_scale * action
        target = np.clip(target, self.ctrl_low, self.ctrl_high)
        self.data.ctrl[: min(len(self.data.ctrl), len(target))] = target[: min(len(self.data.ctrl), len(target))]

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1

        dt = float(self.model.opt.timestep * self.frame_skip)
        self.gait_phase = float((self.gait_phase + 2.0 * np.pi * self.gait_frequency * dt) % (2.0 * np.pi))

        terrain_z, target_z, z, upright, _ = self._posture_metrics()
        body_z_axis = self._body_z_axis(self.data.qpos[3:7])
        pitch_tilt = float(body_z_axis[0])
        roll_tilt = float(body_z_axis[1])
        lin_vel = np.asarray(self.data.qvel[:3], dtype=np.float64)
        ang_vel = np.asarray(self.data.qvel[3:6], dtype=np.float64)
        joint_vel = np.asarray(self.data.qvel[-self.n_joints :], dtype=np.float64)
        joint_error = np.asarray(self.data.qpos[-self.n_joints :], dtype=np.float64) - self.standing_ctrl
        vx = float(lin_vel[0])
        vy = float(lin_vel[1])
        yaw_rate = float(ang_vel[2])
        distance = float(self.data.qpos[0] - self.initial_x)
        walk_target_z = target_z + self.height_target_offset
        walk_height_error = float(z - walk_target_z)

        _, clearances, foot_xy_vel, contacts = self._foot_state(dt)
        desired_contact, swing_amount = self._desired_gait_state(phase_before)
        stance_mask = desired_contact > 0.5
        swing_mask = ~stance_mask
        contact_reward = float(np.mean(contacts == desired_contact)) if len(contacts) else 0.0
        desired_clearance = self.stance_clearance + self.swing_height * swing_amount
        clearance_error = clearances - desired_clearance
        swing_reward = float(np.mean(np.exp(-120.0 * np.square(clearance_error[swing_mask])))) if np.any(swing_mask) else 0.0
        stance_slip = (
            float(np.mean(np.sum(np.square(foot_xy_vel[stance_mask]), axis=1) * contacts[stance_mask]))
            if np.any(stance_mask)
            else 0.0
        )
        support_state = np.maximum(contacts, (clearances <= self.support_clearance_threshold).astype(np.float64))
        support_count = float(np.sum(support_state))
        support_deficit = max(0.0, self.min_support_contacts - support_count)
        support_penalty = self.support_penalty_weight * support_deficit * support_deficit
        rear_contact_count = float(np.sum(support_state[2:4])) if len(support_state) >= 4 else support_count
        rear_air_penalty = self.rear_air_penalty_weight if rear_contact_count < 0.5 else 0.0
        stance_contact_miss = float(np.mean(1.0 - support_state[stance_mask])) if np.any(stance_mask) else 0.0
        stance_contact_penalty = self.stance_contact_penalty_weight * stance_contact_miss
        diagonal_a = float(abs(clearances[0] - clearances[3])) if len(clearances) >= 4 else 0.0
        diagonal_b = float(abs(clearances[1] - clearances[2])) if len(clearances) >= 4 else 0.0
        gait_symmetry_penalty = diagonal_a + diagonal_b

        forward_reward = self.forward_reward_weight * float(np.exp(-self.speed_reward_sharpness * (vx - self.target_vx) ** 2))
        progress_reward = self.progress_reward_weight * vx
        overspeed = max(0.0, vx - self.target_vx - self.overspeed_deadband)
        overspeed_penalty = self.overspeed_weight * overspeed + self.overspeed_quadratic_weight * overspeed * overspeed
        upright_reward = self.upright_reward_weight * float(np.clip((upright + 1.0) * 0.5, 0.0, 1.0))
        height_reward = self.height_reward_weight * float(np.exp(-self.height_reward_sharpness * walk_height_error * walk_height_error))
        low_height_margin = max(0.0, walk_target_z - z)
        low_height_penalty = self.low_height_penalty_weight * low_height_margin + self.low_height_penalty_quadratic_weight * low_height_margin * low_height_margin
        lateral_penalty = self.lateral_penalty_weight * vy * vy
        yaw_penalty = self.yaw_penalty_weight * yaw_rate * yaw_rate
        pitch_tilt_penalty = self.pitch_tilt_penalty_weight * pitch_tilt * pitch_tilt
        roll_tilt_penalty = self.roll_tilt_penalty_weight * roll_tilt * roll_tilt
        ang_vel_penalty = self.ang_vel_penalty_weight * float(np.dot(ang_vel, ang_vel))
        joint_vel_penalty = self.joint_vel_penalty_weight * float(np.mean(np.square(joint_vel)))
        pose_penalty = self.pose_penalty_weight * float(np.mean(np.square(joint_error)))
        action_penalty = self.action_penalty_weight * float(np.mean(np.square(action)))
        smooth_penalty = self.smooth_penalty_weight * float(np.mean(np.square(action - prev_action)))
        backward_penalty = self.backward_penalty_weight * max(0.0, -vx)
        low_speed = max(0.0, self.low_speed_fraction * self.target_vx - vx)
        low_speed_penalty = self.low_speed_penalty_weight * low_speed * low_speed

        fallen = bool(z < terrain_z + 0.10 or upright < 0.45)
        catastrophic = bool(z < terrain_z - 0.02 or upright < 0.10)
        stable_walk = bool(not fallen and upright > 0.75 and abs(walk_height_error) < 0.10 and abs(vy) < 0.35 and abs(yaw_rate) < 1.5)
        timeout = bool(self.steps >= self.max_episode_steps)
        fall_penalty = 8.0 if fallen else 0.0
        terminal_reward = -30.0 if catastrophic else -15.0 if fallen else 20.0 + 5.0 * distance if timeout and stable_walk else 0.0

        gait_contact_reward = self.foot_contact_weight * contact_reward
        gait_swing_reward = self.swing_clearance_weight * swing_reward
        stance_slip_penalty = self.stance_slip_weight * stance_slip
        symmetry_penalty = self.gait_symmetry_weight * gait_symmetry_penalty
        reward = (
            forward_reward
            + progress_reward
            + upright_reward
            + height_reward
            + gait_contact_reward
            + gait_swing_reward
            - lateral_penalty
            - yaw_penalty
            - pitch_tilt_penalty
            - roll_tilt_penalty
            - ang_vel_penalty
            - joint_vel_penalty
            - pose_penalty
            - action_penalty
            - smooth_penalty
            - backward_penalty
            - low_speed_penalty
            - low_height_penalty
            - overspeed_penalty
            - stance_slip_penalty
            - symmetry_penalty
            - support_penalty
            - rear_air_penalty
            - stance_contact_penalty
            - fall_penalty
            + terminal_reward
        )

        self.last_action = action.copy()
        self.prev_upright = float(upright)
        self.prev_abs_height_error = abs(walk_height_error)
        self.prev_x = float(self.data.qpos[0])

        reward_terms = {
            "reward_forward": forward_reward,
            "reward_progress": progress_reward,
            "reward_upright": upright_reward,
            "reward_height": height_reward,
            "reward_gait_contact": gait_contact_reward,
            "reward_gait_swing": gait_swing_reward,
            "penalty_stance_slip": stance_slip_penalty,
            "penalty_gait_symmetry": symmetry_penalty,
            "penalty_support": support_penalty,
            "penalty_rear_air": rear_air_penalty,
            "penalty_stance_contact": stance_contact_penalty,
            "penalty_lateral": lateral_penalty,
            "penalty_yaw": yaw_penalty,
            "penalty_pitch_tilt": pitch_tilt_penalty,
            "penalty_roll_tilt": roll_tilt_penalty,
            "penalty_ang_vel": ang_vel_penalty,
            "penalty_joint_vel": joint_vel_penalty,
            "penalty_pose": pose_penalty,
            "penalty_action": action_penalty,
            "penalty_smooth": smooth_penalty,
            "penalty_backward": backward_penalty,
            "penalty_low_speed": low_speed_penalty,
            "penalty_low_height": low_height_penalty,
            "penalty_overspeed": overspeed_penalty,
            "penalty_fall": fall_penalty,
            "reward_terminal": terminal_reward,
            "foot_contact_match": contact_reward,
            "swing_clearance_score": swing_reward,
            "stance_slip": stance_slip,
            "support_count": support_count,
            "rear_contact_count": rear_contact_count,
            "stance_contact_miss": stance_contact_miss,
            "pitch_tilt": pitch_tilt,
            "roll_tilt": roll_tilt,
        }
        obs = self._normalize_obs(self._get_obs_raw(), update=True)
        info = self._walking_info(reward_terms, stable_walk, fallen, catastrophic, timeout)
        return obs, float(reward), bool(fallen or catastrophic), timeout, info

    def _walking_info(self, reward_terms, stable_walk, fallen, catastrophic, timeout):
        terrain_z, target_z, z, upright, height_error = self._posture_metrics()
        lin_vel = np.asarray(self.data.qvel[:3], dtype=np.float64)
        ang_vel = np.asarray(self.data.qvel[3:6], dtype=np.float64)
        distance = float(self.data.qpos[0] - self.initial_x)
        elapsed = max(self.steps * self.model.opt.timestep * self.frame_skip, 1e-8)
        info = {
            "task": "walk",
            "target_vx": self.target_vx,
            "reset_noise": self.reset_noise,
            "terrain_friction": self.terrain_friction,
            "terrain_height_scale": float(self.terrain_height_scale) if self.terrain_height_scale is not None else 0.0,
            "gait_frequency": self.gait_frequency,
            "gait_phase": self.gait_phase,
            "z": float(z),
            "target_z": float(target_z),
            "upright": float(upright),
            "height_error": float(height_error),
            "walk_height_error": float(z - (target_z + self.height_target_offset)),
            "vx": float(lin_vel[0]),
            "vy": float(lin_vel[1]),
            "yaw_rate": float(ang_vel[2]),
            "pitch_tilt": float(self._body_z_axis(self.data.qpos[3:7])[0]),
            "roll_tilt": float(self._body_z_axis(self.data.qpos[3:7])[1]),
            "speed_error": float(lin_vel[0] - self.target_vx),
            "overspeed": float(max(0.0, lin_vel[0] - self.target_vx - self.overspeed_deadband)),
            "distance": distance,
            "mean_vx_episode": distance / elapsed,
            "stable_walk": bool(stable_walk),
            "fallen": bool(fallen),
            "catastrophic": bool(catastrophic),
            "timeout": bool(timeout),
            "survived": bool(timeout and not fallen and not catastrophic),
            "initial_upright": self.initial_upright,
            "initial_height_error": self.initial_height_error,
        }
        info.update(reward_terms)
        return info
