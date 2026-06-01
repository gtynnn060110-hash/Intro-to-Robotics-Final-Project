import gymnasium as gym
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
    - Actions are 12 target joint angles (radians, normalized in [-1,1]).
    - A simple PD layer converts targets to torques applied via `sim.data.ctrl`.
    - Observation vector: trunk quaternion (4), trunk ang vel (3), 12 joint pos, 12 joint vel => 31 dims.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, model_path: str, kp: float = 120.0, kd: float = 1.0, frame_skip: int = 4):
        if mujoco is None:
            raise RuntimeError("mujoco is required to use UnitreeA1Env (use mjpython)")

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self._home_key = None
        try:
            self._home_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        except Exception:
            self._home_key = None

        self.kp = float(kp)
        self.kd = float(kd)
        self.frame_skip = int(frame_skip)

        self.n_joints = 12

        # action: normalized joint target angles in [-1,1] -> scaled later
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32)

        # observation dims: quaternion(4) + ang vel(3) + 12 pos + 12 vel
        obs_dim = 4 + 3 + self.n_joints + self.n_joints
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.obs_normalizer = Normalizer(obs_dim)

        self._viewer = None

    def _get_joint_indices(self):
        # Heuristic: assume the last n_joints qpos/qvel entries correspond to actuated joints
        return slice(-self.n_joints, None)

    def _get_obs_raw(self):
        qpos = np.array(self.data.qpos)
        qvel = np.array(self.data.qvel)
        # trunk quaternion usually at indices 3:7
        quat = qpos[3:7].copy()
        ang_vel = qvel[3:6].copy()
        ji = self._get_joint_indices()
        joint_pos = qpos[ji].copy()
        joint_vel = qvel[ji].copy()
        return np.concatenate([quat, ang_vel, joint_pos, joint_vel]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        if self._home_key is not None and self._home_key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key)
        mujoco.mj_forward(self.model, self.data)
        # small randomization optional
        obs = self._get_obs_raw()
        obs = self.obs_normalizer.normalize(obs, update=True)
        return obs, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self.n_joints,)
        # scale normalized actions to reasonable joint target range (±0.5 rad)
        target = 0.5 * action

        ji = self._get_joint_indices()
        qpos = np.array(self.data.qpos)
        qvel = np.array(self.data.qvel)
        curr_pos = qpos[ji]
        curr_vel = qvel[ji]

        # PD control -> torque
        tau = self.kp * (target - curr_pos) - self.kd * curr_vel

        # apply to controls (assumes actuator mapping is first N controls)
        nctrl = min(len(self.data.ctrl), len(tau))
        self.data.ctrl[:nctrl] = tau[:nctrl]

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs_raw = self._get_obs_raw()
        obs = self.obs_normalizer.normalize(obs_raw, update=True)

        # simple reward: encourage upright trunk (z position) and penalize large joint velocities
        z = float(self.data.qpos[2])
        vel_penalty = 0.01 * np.sum(np.square(curr_vel))
        reward = float(z) - vel_penalty

        terminated = False
        truncated = False
        info = {"z": z}
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
