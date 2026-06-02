import gymnasium as gym
import math
import numpy as np

try:
    import mujoco
except Exception:  # pragma: no cover - mujoco may be unavailable in CI
    mujoco = None

from gymnasium import spaces
from .normalizer import Normalizer


class UnitreeA1Env(gym.Env):
    """Minimal Gymnasium wrapper for a Unitree A1 MuJoCo model.

    Notes:
    - Expects the model XML path passed as `model_path`.
    - Actions are 12 normalized joint target offsets in [-1,1].
    - The XML uses MuJoCo position actuators, so `data.ctrl` receives target joint angles.
    - Observation vector: trunk quaternion, trunk velocities, height error, joint pos/vel => 35 dims.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        model_path: str,
        kp: float = 120.0,
        kd: float = 1.0,
        frame_skip: int = 4,
        action_scale: float = 0.5,
        foot_clearance: float = 0.005,
        ik_iterations: int = 40,
        task: str = "stand",
        max_episode_steps: int = 1000,
        recovery_prob: float = 0.0,
        contact_solref=(0.004, 1.0),
        contact_solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
    ):
        if mujoco is None:
            raise RuntimeError("mujoco is required to use UnitreeA1Env (use mjpython)")

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.n_joints = 12
        self._home_key = None
        try:
            self._home_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        except Exception:
            self._home_key = None
        self._trunk_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self._terrain_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
        self._foot_geom_ids = self._get_foot_geom_ids()
        self._robot_collision_geom_ids = self._get_robot_collision_geom_ids()

        self.kp = float(kp)
        self.kd = float(kd)
        self.frame_skip = int(frame_skip)
        self.action_scale = float(action_scale)
        self.foot_clearance = float(foot_clearance)
        self.ik_iterations = int(ik_iterations)
        self.task = str(task)
        self.current_task = self.task
        self.max_episode_steps = int(max_episode_steps)
        self.recovery_prob = float(recovery_prob)
        self.contact_solref = np.array(contact_solref, dtype=np.float64)
        self.contact_solimp = np.array(contact_solimp, dtype=np.float64)
        self._configure_contact_params()
        self.default_ctrl = self._get_default_ctrl()
        self.standing_ctrl = self.default_ctrl.copy()
        self.ctrl_low, self.ctrl_high = self._get_ctrl_range()
        self.nominal_base_clearance = self._compute_nominal_base_clearance()
        self.last_action = np.zeros(self.n_joints, dtype=np.float32)
        self.steps = 0

        # action: normalized offsets from the terrain-aware standing pose
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32)

        # observation dims: quat(4) + ang vel(3) + lin vel(3) + height error(1) + 12 pos + 12 vel
        obs_dim = 4 + 3 + 3 + 1 + self.n_joints + self.n_joints
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.obs_normalizer = Normalizer(obs_dim)

        self._viewer = None

    def _get_joint_indices(self):
        # Heuristic: assume the last n_joints qpos/qvel entries correspond to actuated joints
        return slice(-self.n_joints, None)

    def _get_foot_geom_ids(self):
        foot_geom_ids = []
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name and body_name.endswith("_calf") and self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE:
                foot_geom_ids.append(geom_id)
        return foot_geom_ids

    def _get_robot_collision_geom_ids(self):
        geom_ids = []
        for geom_id in range(self.model.ngeom):
            if geom_id == self._terrain_geom:
                continue
            if self.model.geom_group[geom_id] == 3:
                geom_ids.append(geom_id)
        return geom_ids

    def _configure_contact_params(self):
        geom_ids = list(self._robot_collision_geom_ids)
        if self._terrain_geom >= 0:
            geom_ids.append(self._terrain_geom)

        for geom_id in geom_ids:
            self.model.geom_solref[geom_id] = self.contact_solref
            self.model.geom_solimp[geom_id] = self.contact_solimp

    def _get_default_ctrl(self):
        if self._home_key is not None and self._home_key >= 0 and self.model.nkey > self._home_key:
            return np.array(self.model.key_ctrl[self._home_key, : self.n_joints], dtype=np.float32)
        return np.zeros(self.n_joints, dtype=np.float32)

    def _get_ctrl_range(self):
        ctrl_range = np.array(self.model.actuator_ctrlrange[: self.n_joints], dtype=np.float32)
        limited = np.array(self.model.actuator_ctrllimited[: self.n_joints], dtype=bool)
        low = np.where(limited, ctrl_range[:, 0], -np.inf).astype(np.float32)
        high = np.where(limited, ctrl_range[:, 1], np.inf).astype(np.float32)
        return low, high

    def _raycast_terrain_height(self, xy, start_z):
        ray_start = np.array([xy[0], xy[1], start_z + 2.0], dtype=np.float64)
        ray_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        geomgroup = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        geomid = np.array([-1], dtype=np.int32)
        dist = mujoco.mj_ray(
            self.model,
            self.data,
            ray_start,
            ray_dir,
            geomgroup,
            1,
            self._trunk_body,
            geomid,
        )
        if dist <= 0:
            return None
        return float(ray_start[2] - dist)

    def _terrain_height_under_base(self):
        base_pos = self.data.qpos[:3]
        terrain_z = self._raycast_terrain_height(base_pos[:2], base_pos[2])
        if terrain_z is None:
            return 0.0
        return terrain_z

    @staticmethod
    def _quat_from_euler(roll, pitch, yaw):
        cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
        cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
        cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
        return np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _body_z_axis(quat):
        quat = np.asarray(quat, dtype=np.float64)
        quat = quat / max(np.linalg.norm(quat), 1e-8)
        w, x, y, z = quat
        return np.array(
            [
                2.0 * (x * z + w * y),
                2.0 * (y * z - w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
            dtype=np.float64,
        )

    def _upright_score(self):
        return float(self._body_z_axis(self.data.qpos[3:7])[2])

    def _compute_nominal_base_clearance(self):
        mujoco.mj_resetData(self.model, self.data)
        if self._home_key is not None and self._home_key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key)
        mujoco.mj_forward(self.model, self.data)

        base_pos = self.data.qpos[:3].copy()
        terrain_z = self._raycast_terrain_height(base_pos[:2], base_pos[2])
        if terrain_z is None:
            return float(base_pos[2])
        return float(base_pos[2] - terrain_z)

    def _compute_foot_targets(self):
        targets = []
        for geom_id in self._foot_geom_ids:
            foot_pos = self.data.geom_xpos[geom_id].copy()
            foot_radius = float(self.model.geom_size[geom_id, 0])
            terrain_z = self._raycast_terrain_height(foot_pos[:2], foot_pos[2])
            if terrain_z is not None:
                foot_pos[2] = terrain_z + foot_radius + self.foot_clearance
            targets.append(foot_pos)
        return targets

    def _geom_low_points(self, geom_id):
        geom_type = self.model.geom_type[geom_id]
        pos = self.data.geom_xpos[geom_id].copy()
        size = self.model.geom_size[geom_id]
        xmat = self.data.geom_xmat[geom_id].reshape(3, 3)

        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            return [pos - np.array([0.0, 0.0, size[0]])]

        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            corners = []
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    for sz in (-1.0, 1.0):
                        local = np.array([sx * size[0], sy * size[1], sz * size[2]])
                        corners.append(pos + xmat @ local)
            min_z = min(point[2] for point in corners)
            return [point for point in corners if point[2] <= min_z + 1e-6]

        if geom_type in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            radius = float(size[0])
            half_length = float(size[1])
            axis = xmat[:, 2]
            end_a = pos + half_length * axis
            end_b = pos - half_length * axis
            lower_end = end_a if end_a[2] < end_b[2] else end_b
            return [lower_end - np.array([0.0, 0.0, radius])]

        rbound = float(self.model.geom_rbound[geom_id])
        return [pos - np.array([0.0, 0.0, rbound])]

    def _raise_robot_above_terrain(self, clearance=0.01):
        max_lift = 0.0
        for geom_id in self._robot_collision_geom_ids:
            for point in self._geom_low_points(geom_id):
                terrain_z = self._raycast_terrain_height(point[:2], point[2])
                if terrain_z is None:
                    continue
                required_lift = terrain_z + clearance - float(point[2])
                max_lift = max(max_lift, required_lift)

        if max_lift > 0.0:
            self.data.qpos[2] += max_lift
            mujoco.mj_forward(self.model, self.data)
        return max_lift

    def _solve_standing_ik(self):
        if not self._foot_geom_ids:
            return self.default_ctrl.copy()

        targets = self._compute_foot_targets()
        qpos_idx = np.arange(self.model.nq - self.n_joints, self.model.nq)
        dof_idx = np.arange(self.model.nv - self.n_joints, self.model.nv)
        damping = 1e-3

        for _ in range(self.ik_iterations):
            errors = []
            jac_rows = []
            for geom_id, target in zip(self._foot_geom_ids, targets):
                errors.append(target - self.data.geom_xpos[geom_id])

                jacp = np.zeros((3, self.model.nv), dtype=np.float64)
                jacr = np.zeros((3, self.model.nv), dtype=np.float64)
                mujoco.mj_jacGeom(self.model, self.data, jacp, jacr, geom_id)
                jac_rows.append(jacp[:, dof_idx])

            error = np.concatenate(errors)
            if np.linalg.norm(error, ord=np.inf) < 1e-4:
                break

            jac = np.vstack(jac_rows)
            if not np.all(np.isfinite(jac)):
                break

            lhs = jac.T @ jac + damping * np.eye(jac.shape[1])
            rhs = jac.T @ error
            try:
                dq = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                dq = np.linalg.pinv(lhs) @ rhs

            if not np.all(np.isfinite(dq)):
                break

            dq = np.clip(0.5 * dq, -0.08, 0.08)
            self.data.qpos[qpos_idx] = np.clip(self.data.qpos[qpos_idx] + dq, self.ctrl_low, self.ctrl_high)
            mujoco.mj_forward(self.model, self.data)

        return self.data.qpos[qpos_idx].copy().astype(np.float32)

    def _reset_to_standing_pose(self):
        mujoco.mj_resetData(self.model, self.data)
        if self._home_key is not None and self._home_key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key)

        base_pos = self.data.qpos[:3].copy()
        terrain_z = self._raycast_terrain_height(base_pos[:2], base_pos[2])
        if terrain_z is not None:
            self.data.qpos[2] = terrain_z + self.nominal_base_clearance
        mujoco.mj_forward(self.model, self.data)

        self.standing_ctrl = self._solve_standing_ik()
        self.data.qpos[-self.n_joints :] = self.standing_ctrl
        self.data.ctrl[: self.n_joints] = self.standing_ctrl
        mujoco.mj_forward(self.model, self.data)

    def _apply_recovery_randomization(self):
        terrain_z = self._terrain_height_under_base()
        rng = self.np_random

        if rng.random() < 0.45:
            roll = rng.uniform(-math.pi, math.pi)
            pitch = rng.uniform(-1.1, 1.1)
            base_height = rng.uniform(0.10, 0.22)
        else:
            roll = rng.uniform(-0.9, 0.9)
            pitch = rng.uniform(-0.7, 0.7)
            base_height = rng.uniform(0.20, 0.34)
        yaw = rng.uniform(-math.pi, math.pi)

        self.data.qpos[2] = terrain_z + base_height
        self.data.qpos[3:7] = self._quat_from_euler(roll, pitch, yaw)

        joint_noise = rng.normal(0.0, 0.22, size=self.n_joints)
        self.data.qpos[-self.n_joints :] = np.clip(self.standing_ctrl + joint_noise, self.ctrl_low, self.ctrl_high)
        self.data.qvel[:6] = rng.normal(0.0, 0.25, size=6)
        self.data.qvel[-self.n_joints :] = rng.normal(0.0, 0.6, size=self.n_joints)
        self.data.ctrl[: self.n_joints] = self.standing_ctrl
        mujoco.mj_forward(self.model, self.data)
        self._raise_robot_above_terrain(clearance=0.03)

    def _get_obs_raw(self):
        qpos = np.array(self.data.qpos)
        qvel = np.array(self.data.qvel)
        quat = qpos[3:7].copy()
        ang_vel = qvel[3:6].copy()
        lin_vel = qvel[:3].copy()
        height_error = np.array([qpos[2] - (self._terrain_height_under_base() + self.nominal_base_clearance)])
        ji = self._get_joint_indices()
        joint_pos = qpos[ji].copy()
        joint_vel = qvel[ji].copy()
        return np.concatenate([quat, ang_vel, lin_vel, height_error, joint_pos, joint_vel]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.last_action.fill(0.0)
        options = options or {}
        task = options.get("task", self.task)
        self.current_task = task

        self._reset_to_standing_pose()
        use_recovery = task == "recovery" or self.np_random.random() < self.recovery_prob
        if use_recovery:
            self._apply_recovery_randomization()

        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs_raw()
        obs = self.obs_normalizer.normalize(obs, update=True)
        info = {
            "task": task,
            "recovery_reset": use_recovery,
            "z": float(self.data.qpos[2]),
            "upright": self._upright_score(),
        }
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self.n_joints,)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Zero action keeps the terrain-aware standing pose from the latest reset.
        target = self.standing_ctrl + self.action_scale * action
        target = np.clip(target, self.ctrl_low, self.ctrl_high)

        ji = self._get_joint_indices()
        qvel = np.array(self.data.qvel)
        curr_vel = qvel[ji]

        # XML position actuators apply their own PD internally from these angle targets.
        nctrl = min(len(self.data.ctrl), len(target))
        self.data.ctrl[:nctrl] = target[:nctrl]

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1

        obs_raw = self._get_obs_raw()
        obs = self.obs_normalizer.normalize(obs_raw, update=True)

        terrain_z = self._terrain_height_under_base()
        target_z = terrain_z + self.nominal_base_clearance
        z = float(self.data.qpos[2])
        upright = self._upright_score()
        height_error = z - target_z
        ang_vel = np.asarray(self.data.qvel[3:6])
        lin_vel = np.asarray(self.data.qvel[:3])
        joint_error = np.asarray(self.data.qpos[-self.n_joints :]) - self.standing_ctrl

        upright_reward = 1.5 * np.clip(upright, -1.0, 1.0)
        height_reward = 1.0 * np.exp(-12.0 * height_error * height_error)
        stillness_reward = 0.5 * np.exp(-1.5 * float(np.dot(lin_vel, lin_vel)))
        pose_penalty = 0.15 * float(np.mean(np.square(joint_error)))
        joint_vel_penalty = 0.02 * float(np.mean(np.square(curr_vel)))
        ang_vel_penalty = 0.03 * float(np.dot(ang_vel, ang_vel))
        action_penalty = 0.02 * float(np.mean(np.square(action)))
        smooth_penalty = 0.01 * float(np.mean(np.square(action - self.last_action)))
        reward = (
            upright_reward
            + height_reward
            + stillness_reward
            - pose_penalty
            - joint_vel_penalty
            - ang_vel_penalty
            - action_penalty
            - smooth_penalty
        )

        self.last_action = action.copy()

        fallen = bool(z < terrain_z + 0.06 or upright < -0.35)
        catastrophic = bool(z < terrain_z - 0.08)
        if self.current_task == "recovery":
            terminated = catastrophic
        else:
            terminated = fallen
        truncated = bool(self.steps >= self.max_episode_steps)
        if fallen:
            reward -= 2.0

        info = {
            "z": z,
            "target_z": target_z,
            "upright": upright,
            "height_error": height_error,
            "fallen": fallen,
            "catastrophic": catastrophic,
            "reward_upright": upright_reward,
            "reward_height": height_reward,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self, mode="human"):
        if mode == "human":
            if self._viewer is None:
                try:
                    import mujoco.viewer  # type: ignore

                    self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
                except Exception:
                    self._viewer = None
            if self._viewer is not None:
                try:
                    self._viewer.sync()
                except Exception:
                    pass
        elif mode == "rgb_array":
            try:
                return mujoco.render(self.model, self.data, width=640, height=480)
            except Exception:
                return None

    def close(self):
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
