"""Fine-tune a recovery checkpoint into a fixed-speed walking policy."""
import argparse
import os
import pickle
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from envs import UnitreeA1WalkEnv


INFO_KEYWORDS = (
    "target_vx",
    "reset_noise",
    "terrain_friction",
    "terrain_height_scale",
    "gait_frequency",
    "gait_phase",
    "upright",
    "height_error",
    "vx",
    "vy",
    "yaw_rate",
    "pitch_tilt",
    "roll_tilt",
    "speed_error",
    "overspeed",
    "reward_forward",
    "reward_progress",
    "reward_gait_contact",
    "reward_gait_swing",
    "penalty_overspeed",
    "penalty_low_speed",
    "penalty_stance_slip",
    "penalty_backward",
    "penalty_support",
    "penalty_rear_air",
    "penalty_stance_contact",
    "penalty_pitch_tilt",
    "penalty_roll_tilt",
    "distance",
    "mean_vx_episode",
    "stable_walk",
    "fallen",
    "catastrophic",
    "survived",
    "foot_contact_match",
    "swing_clearance_score",
    "stance_slip",
    "support_count",
    "rear_contact_count",
    "stance_contact_miss",
    "reward_teacher_action",
    "teacher_action_error",
    "teacher_reward_weight",
)


class TeacherActionRewardWrapper(gym.Wrapper):
    """Keep PPO close to a frozen BC/SFT walking policy with action reward."""

    def __init__(
        self,
        env,
        teacher_checkpoint,
        teacher_vecnormalize,
        reward_weight=5.0,
        action_error_scale=4.0,
        deterministic=True,
    ):
        super().__init__(env)
        self.reward_weight = float(max(reward_weight, 0.0))
        self.action_error_scale = float(max(action_error_scale, 1e-8))
        self.deterministic = bool(deterministic)
        self._last_raw_obs = None

        self.teacher = PPO.load(str(teacher_checkpoint), device="cpu")
        self.teacher.policy.set_training_mode(False)

        with open(teacher_vecnormalize, "rb") as f:
            vecnormalize = pickle.load(f)
        self._teacher_obs_mean = np.asarray(vecnormalize.obs_rms.mean, dtype=np.float32)
        self._teacher_obs_var = np.asarray(vecnormalize.obs_rms.var, dtype=np.float32)
        self._teacher_clip_obs = float(getattr(vecnormalize, "clip_obs", 10.0))
        self._teacher_epsilon = float(getattr(vecnormalize, "epsilon", 1e-8))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_raw_obs = np.asarray(obs, dtype=np.float32).copy()
        return obs, info

    def step(self, action):
        teacher_action = self._predict_teacher_action()
        action = np.asarray(action, dtype=np.float32)
        action_error = float(np.mean(np.square(action - teacher_action)))
        teacher_reward = self.reward_weight * float(np.exp(-self.action_error_scale * action_error))

        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["reward_teacher_action"] = teacher_reward
        info["teacher_action_error"] = action_error
        info["teacher_reward_weight"] = self.reward_weight
        self._last_raw_obs = np.asarray(obs, dtype=np.float32).copy()
        return obs, float(reward + teacher_reward), terminated, truncated, info

    def _predict_teacher_action(self):
        if self._last_raw_obs is None:
            raise RuntimeError("TeacherActionRewardWrapper.step() called before reset().")
        obs = self._normalize_teacher_obs(self._last_raw_obs)
        action, _ = self.teacher.predict(obs, deterministic=self.deterministic)
        return np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)

    def _normalize_teacher_obs(self, obs):
        if obs.shape != self._teacher_obs_mean.shape:
            raise ValueError(
                "Teacher observation stats do not match this environment: "
                f"obs={obs.shape}, teacher={self._teacher_obs_mean.shape}. "
                "Check that gait-clock/action settings match the BC run."
            )
        norm_obs = (obs - self._teacher_obs_mean) / np.sqrt(self._teacher_obs_var + self._teacher_epsilon)
        return np.clip(norm_obs, -self._teacher_clip_obs, self._teacher_clip_obs).astype(np.float32)


class WalkCurriculumCallback(BaseCallback):
    def __init__(
        self,
        start_vx,
        end_vx,
        curriculum_steps,
        reset_noise_start=0.0,
        reset_noise_end=0.0,
        terrain_friction_start=1.5,
        terrain_friction_end=1.5,
        terrain_height_scale_start=0.3,
        terrain_height_scale_end=0.3,
        update_interval=1000,
        verbose=0,
    ):
        super().__init__(verbose)
        self.start_vx = float(start_vx)
        self.end_vx = float(end_vx)
        self.curriculum_steps = max(int(curriculum_steps), 1)
        self.reset_noise_start = float(reset_noise_start)
        self.reset_noise_end = float(reset_noise_end)
        self.terrain_friction_start = float(terrain_friction_start)
        self.terrain_friction_end = float(terrain_friction_end)
        self.terrain_height_scale_start = float(terrain_height_scale_start)
        self.terrain_height_scale_end = float(terrain_height_scale_end)
        self.update_interval = max(int(update_interval), 1)
        self._last_update = -1

    def _on_training_start(self):
        self._set_curriculum(self.num_timesteps)
        self._last_update = self.num_timesteps

    def _on_step(self):
        if self.num_timesteps - self._last_update < self.update_interval:
            return True
        self._set_curriculum(self.num_timesteps)
        self._last_update = self.num_timesteps
        return True

    def _set_curriculum(self, num_timesteps):
        progress = min(float(num_timesteps) / float(self.curriculum_steps), 1.0)
        target_vx = self.start_vx + progress * (self.end_vx - self.start_vx)
        reset_noise = self.reset_noise_start + progress * (self.reset_noise_end - self.reset_noise_start)
        terrain_friction = self.terrain_friction_start + progress * (
            self.terrain_friction_end - self.terrain_friction_start
        )
        terrain_height_scale = self.terrain_height_scale_start + progress * (
            self.terrain_height_scale_end - self.terrain_height_scale_start
        )
        self.training_env.env_method("set_target_vx", float(target_vx))
        self.training_env.env_method("set_reset_noise", float(reset_noise))
        self.training_env.env_method("set_terrain_friction", float(terrain_friction))
        self.training_env.env_method("set_terrain_height_scale", float(terrain_height_scale))


class WalkMetricsCallback(BaseCallback):
    def __init__(self, log_interval=10_000, save_vecnormalize_freq=100_000, save_path=None, verbose=0):
        super().__init__(verbose)
        self.log_interval = int(log_interval)
        self.save_vecnormalize_freq = int(save_vecnormalize_freq)
        self.save_path = Path(save_path) if save_path else None
        self._episode_infos = []
        self._last_log = -1
        self._last_save = -1

    def _on_training_start(self):
        if self.save_path is not None:
            self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._episode_infos.append(dict(info))

        if self.num_timesteps - self._last_log >= self.log_interval:
            self._print_metrics(infos)
            self._last_log = self.num_timesteps

        if (
            self.save_path is not None
            and self.save_vecnormalize_freq > 0
            and self.num_timesteps - self._last_save >= self.save_vecnormalize_freq
        ):
            vec_normalize = self.model.get_vec_normalize_env()
            if vec_normalize is not None:
                vec_normalize.save(str(self.save_path / f"vecnormalize_{self.num_timesteps}_steps.pkl"))
            self._last_save = self.num_timesteps
        return True

    def _print_metrics(self, infos):
        recent = self._episode_infos[-40:]
        payload = {
            "steps": self.num_timesteps,
            "target_vx": _mean_info(infos, "target_vx"),
            "vx": _mean_info(infos, "vx"),
            "upright": _mean_info(infos, "upright"),
            "height_error": _mean_info(infos, "height_error"),
            "fallen": _mean_info(infos, "fallen"),
            "foot_contact": _mean_info(infos, "foot_contact_match"),
            "swing": _mean_info(infos, "swing_clearance_score"),
            "stance_slip": _mean_info(infos, "stance_slip"),
            "support_count": _mean_info(infos, "support_count"),
            "rear_contacts": _mean_info(infos, "rear_contact_count"),
            "rear_air_penalty": _mean_info(infos, "penalty_rear_air"),
            "stance_contact_miss": _mean_info(infos, "stance_contact_miss"),
            "pitch_tilt": _mean_info(infos, "pitch_tilt"),
            "roll_tilt": _mean_info(infos, "roll_tilt"),
            "teacher_reward": _mean_info(infos, "reward_teacher_action"),
            "teacher_action_error": _mean_info(infos, "teacher_action_error"),
            "low_speed_penalty": _mean_info(infos, "penalty_low_speed"),
            "pitch_penalty": _mean_info(infos, "penalty_pitch_tilt"),
            "roll_penalty": _mean_info(infos, "penalty_roll_tilt"),
        }
        if recent:
            payload.update(
                {
                    "ep_survive": float(np.mean([float(info.get("survived", False)) for info in recent])),
                    "ep_distance": float(np.mean([float(info.get("distance", 0.0)) for info in recent])),
                    "ep_mean_vx": float(np.mean([float(info.get("mean_vx_episode", 0.0)) for info in recent])),
                    "ep_reward": float(np.mean([float(info["episode"]["r"]) for info in recent if "episode" in info])),
                }
            )
        print(
            "[walk] "
            + " ".join(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}" for key, value in payload.items()),
            flush=True,
        )


def _mean_info(infos, key):
    values = [float(info[key]) for info in infos if key in info]
    if not values:
        return 0.0
    return float(np.mean(values))


def make_env(
    model_path,
    seed,
    max_episode_steps,
    target_vx,
    reset_noise,
    action_scale,
    overspeed_deadband,
    overspeed_weight,
    overspeed_quadratic_weight,
    forward_reward_weight,
    progress_reward_weight,
    backward_penalty_weight,
    low_speed_penalty_weight,
    low_speed_fraction,
    speed_reward_sharpness,
    upright_reward_weight,
    height_reward_weight,
    height_reward_sharpness,
    height_target_offset,
    low_height_penalty_weight,
    low_height_penalty_quadratic_weight,
    lateral_penalty_weight,
    yaw_penalty_weight,
    pitch_tilt_penalty_weight,
    roll_tilt_penalty_weight,
    ang_vel_penalty_weight,
    joint_vel_penalty_weight,
    pose_penalty_weight,
    action_penalty_weight,
    smooth_penalty_weight,
    terrain_friction,
    terrain_height_scale,
    frame_skip,
    gait_frequency,
    swing_height,
    stance_clearance,
    foot_contact_weight,
    swing_clearance_weight,
    stance_slip_weight,
    gait_symmetry_weight,
    support_clearance_threshold,
    min_support_contacts,
    support_penalty_weight,
    rear_air_penalty_weight,
    stance_contact_penalty_weight,
    gait_clock_obs,
    use_trot_reference,
    trot_frequency,
    trot_thigh_amplitude,
    trot_calf_amplitude,
    trot_stance_calf_amplitude,
    teacher_checkpoint,
    teacher_vecnormalize,
    teacher_reward_weight,
    teacher_action_error_scale,
    teacher_deterministic,
):
    def _init():
        env = UnitreeA1WalkEnv(
            model_path=model_path,
            target_vx=target_vx,
            reset_noise=reset_noise,
            terrain_friction=terrain_friction,
            terrain_height_scale=terrain_height_scale,
            max_episode_steps=max_episode_steps,
            action_scale=action_scale,
            overspeed_deadband=overspeed_deadband,
            overspeed_weight=overspeed_weight,
            overspeed_quadratic_weight=overspeed_quadratic_weight,
            forward_reward_weight=forward_reward_weight,
            progress_reward_weight=progress_reward_weight,
            backward_penalty_weight=backward_penalty_weight,
            low_speed_penalty_weight=low_speed_penalty_weight,
            low_speed_fraction=low_speed_fraction,
            speed_reward_sharpness=speed_reward_sharpness,
            upright_reward_weight=upright_reward_weight,
            height_reward_weight=height_reward_weight,
            height_reward_sharpness=height_reward_sharpness,
            height_target_offset=height_target_offset,
            low_height_penalty_weight=low_height_penalty_weight,
            low_height_penalty_quadratic_weight=low_height_penalty_quadratic_weight,
            lateral_penalty_weight=lateral_penalty_weight,
            yaw_penalty_weight=yaw_penalty_weight,
            pitch_tilt_penalty_weight=pitch_tilt_penalty_weight,
            roll_tilt_penalty_weight=roll_tilt_penalty_weight,
            ang_vel_penalty_weight=ang_vel_penalty_weight,
            joint_vel_penalty_weight=joint_vel_penalty_weight,
            pose_penalty_weight=pose_penalty_weight,
            action_penalty_weight=action_penalty_weight,
            smooth_penalty_weight=smooth_penalty_weight,
            frame_skip=frame_skip,
            gait_frequency=gait_frequency,
            swing_height=swing_height,
            stance_clearance=stance_clearance,
            foot_contact_weight=foot_contact_weight,
            swing_clearance_weight=swing_clearance_weight,
            stance_slip_weight=stance_slip_weight,
            gait_symmetry_weight=gait_symmetry_weight,
            support_clearance_threshold=support_clearance_threshold,
            min_support_contacts=min_support_contacts,
            support_penalty_weight=support_penalty_weight,
            rear_air_penalty_weight=rear_air_penalty_weight,
            stance_contact_penalty_weight=stance_contact_penalty_weight,
            gait_clock_obs=gait_clock_obs,
            use_trot_reference=use_trot_reference,
            trot_frequency=trot_frequency,
            trot_thigh_amplitude=trot_thigh_amplitude,
            trot_calf_amplitude=trot_calf_amplitude,
            trot_stance_calf_amplitude=trot_stance_calf_amplitude,
            normalize_obs=False,
        )
        teacher_enabled = bool(teacher_checkpoint and teacher_reward_weight > 0.0)
        if teacher_enabled:
            env = TeacherActionRewardWrapper(
                env,
                teacher_checkpoint=teacher_checkpoint,
                teacher_vecnormalize=teacher_vecnormalize,
                reward_weight=teacher_reward_weight,
                action_error_scale=teacher_action_error_scale,
                deterministic=teacher_deterministic,
            )
        env.reset(seed=seed)
        info_keywords = INFO_KEYWORDS
        if not teacher_enabled:
            info_keywords = tuple(key for key in INFO_KEYWORDS if not key.startswith("teacher_") and key != "reward_teacher_action")
        return Monitor(env, info_keywords=info_keywords)

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unitree_a1/scene.xml")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--vecnormalize-load", default=None)
    parser.add_argument("--teacher-run-dir", default=None)
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--teacher-vecnormalize", default=None)
    parser.add_argument("--teacher-reward-weight", type=float, default=5.0)
    parser.add_argument("--teacher-action-error-scale", type=float, default=4.0)
    parser.add_argument("--teacher-deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-dir", default="runs/walk_from_prev_fresh_v1")
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--target-vx", type=float, default=0.2)
    parser.add_argument("--start-vx", type=float, default=0.1)
    parser.add_argument("--curriculum-steps", type=int, default=500_000)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--reset-noise-start", type=float, default=0.0)
    parser.add_argument("--reset-noise-end", type=float, default=0.15)
    parser.add_argument("--terrain-friction-start", type=float, default=1.5)
    parser.add_argument("--terrain-friction-end", type=float, default=1.5)
    parser.add_argument("--terrain-height-scale-start", type=float, default=0.3)
    parser.add_argument("--terrain-height-scale-end", type=float, default=0.3)
    parser.add_argument("--action-scale", type=float, default=0.5)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--gait-frequency", type=float, default=0.95)
    parser.add_argument("--swing-height", type=float, default=0.045)
    parser.add_argument("--stance-clearance", type=float, default=0.012)
    parser.add_argument("--foot-contact-weight", type=float, default=1.2)
    parser.add_argument("--swing-clearance-weight", type=float, default=0.7)
    parser.add_argument("--stance-slip-weight", type=float, default=0.25)
    parser.add_argument("--gait-symmetry-weight", type=float, default=0.15)
    parser.add_argument("--support-clearance-threshold", type=float, default=0.035)
    parser.add_argument("--min-support-contacts", type=float, default=2.0)
    parser.add_argument("--support-penalty-weight", type=float, default=5.0)
    parser.add_argument("--rear-air-penalty-weight", type=float, default=8.0)
    parser.add_argument("--stance-contact-penalty-weight", type=float, default=2.0)
    parser.add_argument("--gait-clock-obs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overspeed-deadband", type=float, default=0.02)
    parser.add_argument("--overspeed-weight", type=float, default=8.0)
    parser.add_argument("--overspeed-quadratic-weight", type=float, default=20.0)
    parser.add_argument("--forward-reward-weight", type=float, default=3.0)
    parser.add_argument("--progress-reward-weight", type=float, default=0.8)
    parser.add_argument("--backward-penalty-weight", type=float, default=2.0)
    parser.add_argument("--low-speed-penalty-weight", type=float, default=80.0)
    parser.add_argument("--low-speed-fraction", type=float, default=0.9)
    parser.add_argument("--speed-reward-sharpness", type=float, default=25.0)
    parser.add_argument("--upright-reward-weight", type=float, default=1.5)
    parser.add_argument("--height-reward-weight", type=float, default=1.5)
    parser.add_argument("--height-reward-sharpness", type=float, default=30.0)
    parser.add_argument("--height-target-offset", type=float, default=0.0)
    parser.add_argument("--low-height-penalty-weight", type=float, default=0.0)
    parser.add_argument("--low-height-penalty-quadratic-weight", type=float, default=0.0)
    parser.add_argument("--lateral-penalty-weight", type=float, default=2.5)
    parser.add_argument("--yaw-penalty-weight", type=float, default=0.6)
    parser.add_argument("--pitch-tilt-penalty-weight", type=float, default=8.0)
    parser.add_argument("--roll-tilt-penalty-weight", type=float, default=5.0)
    parser.add_argument("--ang-vel-penalty-weight", type=float, default=0.06)
    parser.add_argument("--joint-vel-penalty-weight", type=float, default=0.003)
    parser.add_argument("--pose-penalty-weight", type=float, default=0.03)
    parser.add_argument("--action-penalty-weight", type=float, default=0.004)
    parser.add_argument("--smooth-penalty-weight", type=float, default=0.010)
    parser.add_argument("--use-trot-reference", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trot-frequency", type=float, default=1.35)
    parser.add_argument("--trot-thigh-amplitude", type=float, default=0.22)
    parser.add_argument("--trot-calf-amplitude", type=float, default=0.32)
    parser.add_argument("--trot-stance-calf-amplitude", type=float, default=0.08)
    parser.add_argument("--start-method", choices=["forkserver", "spawn", "fork"], default="fork")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=10_000)
    parser.add_argument("--checkpoint-save-freq", type=int, default=100_000)
    parser.add_argument("--norm-reward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reset-num-timesteps", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.teacher_run_dir:
        teacher_run_dir = Path(args.teacher_run_dir)
        if args.teacher_checkpoint is None:
            args.teacher_checkpoint = str(teacher_run_dir / "ppo_walk_bc_pretrained.zip")
        if args.teacher_vecnormalize is None:
            args.teacher_vecnormalize = str(teacher_run_dir / "vecnormalize.pkl")
    if args.teacher_checkpoint and args.teacher_vecnormalize is None:
        raise ValueError("--teacher-vecnormalize is required when --teacher-checkpoint is set.")
    if args.teacher_checkpoint is None:
        args.teacher_reward_weight = 0.0
    elif args.teacher_reward_weight <= 0.0:
        print("[walk] teacher checkpoint provided but teacher reward weight <= 0; teacher regularization is disabled.")
    else:
        if not Path(args.teacher_checkpoint).exists():
            raise FileNotFoundError(args.teacher_checkpoint)
        if not Path(args.teacher_vecnormalize).exists():
            raise FileNotFoundError(args.teacher_vecnormalize)
        if args.n_envs > 1 and args.start_method == "fork":
            args.start_method = "forkserver"
            print("[walk] teacher regularization is not fork-safe; using start_method=forkserver.", flush=True)
        print(
            "[walk] teacher regularization enabled "
            f"checkpoint={args.teacher_checkpoint} vecnormalize={args.teacher_vecnormalize} "
            f"weight={args.teacher_reward_weight:.3f} scale={args.teacher_action_error_scale:.3f}",
            flush=True,
        )

    torch.set_num_threads(max(int(args.torch_threads), 1))
    torch.set_num_interop_threads(1)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [
        make_env(
            model_path=args.model,
            seed=args.seed + i,
            max_episode_steps=args.max_episode_steps,
            target_vx=args.start_vx,
            reset_noise=args.reset_noise_start,
            action_scale=args.action_scale,
            overspeed_deadband=args.overspeed_deadband,
            overspeed_weight=args.overspeed_weight,
            overspeed_quadratic_weight=args.overspeed_quadratic_weight,
            forward_reward_weight=args.forward_reward_weight,
            progress_reward_weight=args.progress_reward_weight,
            backward_penalty_weight=args.backward_penalty_weight,
            low_speed_penalty_weight=args.low_speed_penalty_weight,
            low_speed_fraction=args.low_speed_fraction,
            speed_reward_sharpness=args.speed_reward_sharpness,
            upright_reward_weight=args.upright_reward_weight,
            height_reward_weight=args.height_reward_weight,
            height_reward_sharpness=args.height_reward_sharpness,
            height_target_offset=args.height_target_offset,
            low_height_penalty_weight=args.low_height_penalty_weight,
            low_height_penalty_quadratic_weight=args.low_height_penalty_quadratic_weight,
            lateral_penalty_weight=args.lateral_penalty_weight,
            yaw_penalty_weight=args.yaw_penalty_weight,
            pitch_tilt_penalty_weight=args.pitch_tilt_penalty_weight,
            roll_tilt_penalty_weight=args.roll_tilt_penalty_weight,
            ang_vel_penalty_weight=args.ang_vel_penalty_weight,
            joint_vel_penalty_weight=args.joint_vel_penalty_weight,
            pose_penalty_weight=args.pose_penalty_weight,
            action_penalty_weight=args.action_penalty_weight,
            smooth_penalty_weight=args.smooth_penalty_weight,
            terrain_friction=args.terrain_friction_start,
            terrain_height_scale=args.terrain_height_scale_start,
            frame_skip=args.frame_skip,
            gait_frequency=args.gait_frequency,
            swing_height=args.swing_height,
            stance_clearance=args.stance_clearance,
            foot_contact_weight=args.foot_contact_weight,
            swing_clearance_weight=args.swing_clearance_weight,
            stance_slip_weight=args.stance_slip_weight,
            gait_symmetry_weight=args.gait_symmetry_weight,
            support_clearance_threshold=args.support_clearance_threshold,
            min_support_contacts=args.min_support_contacts,
            support_penalty_weight=args.support_penalty_weight,
            rear_air_penalty_weight=args.rear_air_penalty_weight,
            stance_contact_penalty_weight=args.stance_contact_penalty_weight,
            gait_clock_obs=args.gait_clock_obs,
            use_trot_reference=args.use_trot_reference,
            trot_frequency=args.trot_frequency,
            trot_thigh_amplitude=args.trot_thigh_amplitude,
            trot_calf_amplitude=args.trot_calf_amplitude,
            trot_stance_calf_amplitude=args.trot_stance_calf_amplitude,
            teacher_checkpoint=args.teacher_checkpoint,
            teacher_vecnormalize=args.teacher_vecnormalize,
            teacher_reward_weight=args.teacher_reward_weight,
            teacher_action_error_scale=args.teacher_action_error_scale,
            teacher_deterministic=args.teacher_deterministic,
        )
        for i in range(args.n_envs)
    ]
    if args.n_envs > 1:
        env = SubprocVecEnv(env_fns, start_method=args.start_method)
    else:
        env = DummyVecEnv(env_fns)

    if args.vecnormalize_load:
        env = VecNormalize.load(args.vecnormalize_load, env)
        env.training = True
        env.norm_reward = bool(args.norm_reward)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=bool(args.norm_reward), clip_obs=10.0)

    if args.resume_from:
        model = PPO.load(args.resume_from, env=env, seed=args.seed, verbose=1)
        model.learning_rate = args.learning_rate
        model.lr_schedule = get_schedule_fn(args.learning_rate)
    else:
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
            n_steps=2048,
            batch_size=512,
            learning_rate=args.learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.15,
            ent_coef=0.01,
        )

    callbacks = CallbackList(
        [
            WalkCurriculumCallback(
                start_vx=args.start_vx,
                end_vx=args.target_vx,
                curriculum_steps=args.curriculum_steps,
                reset_noise_start=args.reset_noise_start,
                reset_noise_end=args.reset_noise_end,
                terrain_friction_start=args.terrain_friction_start,
                terrain_friction_end=args.terrain_friction_end,
                terrain_height_scale_start=args.terrain_height_scale_start,
                terrain_height_scale_end=args.terrain_height_scale_end,
                update_interval=args.log_interval,
            ),
            CheckpointCallback(
                save_freq=max(args.checkpoint_save_freq // args.n_envs, 1),
                save_path=str(checkpoint_dir),
                name_prefix="ppo_walk",
            ),
            WalkMetricsCallback(
                log_interval=args.log_interval,
                save_vecnormalize_freq=args.checkpoint_save_freq,
                save_path=checkpoint_dir,
            ),
        ]
    )

    try:
        model.learn(
            total_timesteps=args.total_steps,
            callback=callbacks,
            reset_num_timesteps=args.reset_num_timesteps,
        )
        model.save(str(run_dir / "ppo_walk_final"))
        env.save(str(run_dir / "vecnormalize.pkl"))
    finally:
        env.close()


if __name__ == "__main__":
    main()
