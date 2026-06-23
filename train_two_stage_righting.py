"""Two-stage righting curriculum training for Unitree A1.

This file is intentionally isolated from train.py and train_hard_bin.py.

Stage A (`--stage righting`) trains a policy to turn severe low-upright resets
into a righted state.  The success target is upright >= righting_target, not
full stable standing.

Stage B (`--stage stabilize`) resumes from Stage A and trains full recovery on
a mixture of ordinary recovery resets and oversampled hard resets.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from envs import UnitreeA1Env


class TwoStageRightingEnv(UnitreeA1Env):
    """Reward wrapper for righting-first recovery.

    The base dynamics and observation/action spaces are inherited unchanged.
    """

    def __init__(
        self,
        *args,
        stage="righting",
        difficulty_min=0.35,
        difficulty_max=0.45,
        hard_reset_prob=1.0,
        hard_upright_min=0.0,
        hard_upright_max=0.5,
        righting_target=0.62,
        righting_success_steps=5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.stage = str(stage)
        self.righting_target = float(righting_target)
        self.righting_success_steps_required = int(righting_success_steps)
        self.righting_steps = 0
        self.righted = False
        self.ever_righted = False
        self.extra_reward = 0.0
        self.set_recovery_difficulty_range(difficulty_min, difficulty_max)
        self.set_hard_reset_oversampling(hard_reset_prob, hard_upright_min, hard_upright_max)

    def reset(self, *, seed=None, options=None):
        self.righting_steps = 0
        self.righted = False
        self.ever_righted = False
        self.extra_reward = 0.0
        obs, info = super().reset(seed=seed, options=options)
        info["stage"] = self.stage
        info["righted"] = False
        info["ever_righted"] = False
        return obs, info

    def step(self, action):
        prev_upright = float(self.prev_upright)
        prev_abs_height_error = float(self.prev_abs_height_error)
        obs, base_reward, terminated, truncated, info = super().step(action)

        if self.stage == "righting":
            reward, terminated = self._righting_reward(
                terminated=terminated,
                truncated=truncated,
                info=info,
                prev_upright=prev_upright,
                prev_abs_height_error=prev_abs_height_error,
            )
        elif self.stage == "stabilize":
            reward = self._stabilize_reward(
                base_reward=base_reward,
                info=info,
                prev_upright=prev_upright,
                prev_abs_height_error=prev_abs_height_error,
            )
        else:
            reward = float(base_reward)

        info["stage"] = self.stage
        info["righted"] = bool(self.righted)
        info["ever_righted"] = bool(self.ever_righted)
        info["righting_steps"] = int(self.righting_steps)
        info["reward_base_recovery"] = float(base_reward)
        info["reward_extra_two_stage"] = float(self.extra_reward)
        return obs, float(reward), bool(terminated), truncated, info

    def _righted_now(self, info):
        upright = float(info["upright"])
        z = float(info["z"])
        target_z = float(info["target_z"])
        return bool(upright >= self.righting_target and z > target_z - 0.12)

    def _update_righted_state(self, info):
        if self._righted_now(info):
            self.righting_steps += 1
        elif float(info["upright"]) < self.righting_target - 0.10:
            self.righting_steps = 0
        else:
            self.righting_steps = max(0, self.righting_steps - 1)
        self.righted = bool(
            self.righting_steps >= self.righting_success_steps_required
            or bool(info.get("recovered", False))
        )
        self.ever_righted = bool(self.ever_righted or self.righted)

    def _righting_reward(self, terminated, truncated, info, prev_upright, prev_abs_height_error):
        upright = float(info["upright"])
        z = float(info["z"])
        target_z = float(info["target_z"])
        height_error = float(info["height_error"])
        upright_progress = upright - prev_upright
        height_progress = prev_abs_height_error - abs(height_error)
        lin_vel = np.asarray(self.data.qvel[:3])
        ang_vel = np.asarray(self.data.qvel[3:6])
        joint_vel = np.asarray(self.data.qvel[-self.n_joints :])
        time_fraction = min(float(self.steps) / max(float(self.max_episode_steps), 1.0), 1.0)

        target_margin = max(self.righting_target - self.initial_upright, 0.08)
        normalized_gain = (upright - self.initial_upright) / target_margin
        torso_lift = np.clip(z - (target_z - 0.16), -0.12, 0.22)
        low_height_margin = max(0.0, target_z - z - 0.08)

        reward = (
            5.0 * np.clip((upright + 1.0) * 0.5, 0.0, 1.0)
            + 9.0 * np.clip(normalized_gain, -0.5, 1.2)
            + 32.0 * np.clip(upright_progress, -0.05, 0.12)
            + 11.0 * np.clip(height_progress, -0.04, 0.09)
            + 2.0 * torso_lift
            + 2.5 * max(0.0, upright - 0.35)
            - 10.0 * low_height_margin
            - 0.010 * float(np.dot(ang_vel, ang_vel))
            - 0.002 * float(np.mean(np.square(joint_vel)))
            - 0.002 * float(np.dot(lin_vel[:2], lin_vel[:2]))
            - 0.025 * time_fraction
        )

        self._update_righted_state(info)
        if bool(info.get("catastrophic", False)):
            reward -= 35.0
            terminated = True
        elif bool(info.get("failure_timeout", False)):
            reward -= 22.0
            terminated = True
        elif self.righted:
            reward += 90.0 + 35.0 * (1.0 - time_fraction)
            terminated = True
        elif truncated:
            reward -= 18.0

        self.extra_reward = float(reward)
        return float(reward), bool(terminated)

    def _stabilize_reward(self, base_reward, info, prev_upright, prev_abs_height_error):
        upright = float(info["upright"])
        z = float(info["z"])
        target_z = float(info["target_z"])
        height_error = float(info["height_error"])
        upright_progress = upright - prev_upright
        height_progress = prev_abs_height_error - abs(height_error)
        low_start = bool(self.initial_upright < 0.55)
        stable = bool(info.get("stable", False))
        recovered = bool(info.get("recovered", False))
        terminal_bad = bool(
            info.get("failure_timeout", False)
            or info.get("catastrophic", False)
            or info.get("episode_timeout", False)
        )

        self._update_righted_state(info)
        extra = 0.0
        if low_start:
            low_height_margin = max(0.0, target_z - z - 0.06)
            extra += 36.0 * np.clip(upright_progress, -0.05, 0.12)
            extra += 12.0 * np.clip(height_progress, -0.04, 0.09)
            extra += 4.0 * np.clip(upright - self.initial_upright, -0.2, 0.8)
            extra += 4.0 if self.ever_righted else 0.0
            extra += 9.0 if stable else -1.0
            extra -= 16.0 * low_height_margin
            if recovered:
                extra += 170.0
            elif terminal_bad:
                extra -= 120.0
            reward = 0.45 * float(base_reward) + extra
        else:
            # Keep ordinary recovery close to the original objective so broad
            # sweep performance is less likely to be overwritten.
            reward = float(base_reward)

        self.extra_reward = float(extra)
        return float(reward)


def make_env(args, rank):
    def _init():
        env = TwoStageRightingEnv(
            args.model,
            task="recovery",
            max_episode_steps=args.max_episode_steps,
            recovery_difficulty=args.difficulty_end,
            normalize_obs=False,
            action_scale=args.action_scale,
            success_steps=args.success_steps,
            failure_steps=args.failure_steps,
            stage=args.stage,
            difficulty_min=args.difficulty_start,
            difficulty_max=args.difficulty_end,
            hard_reset_prob=args.hard_reset_prob,
            hard_upright_min=args.hard_upright_min,
            hard_upright_max=args.hard_upright_max,
            righting_target=args.righting_target,
            righting_success_steps=args.righting_success_steps,
        )
        env.reset(seed=args.seed + rank)
        return Monitor(env)

    return _init


class TrainMetricsCallback(BaseCallback):
    def __init__(self, csv_path, log_interval=20_000):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.log_interval = int(log_interval)
        self._last_log = 0
        self.rows = []
        self.fieldnames = [
            "timesteps",
            "episode_reward_mean_50",
            "episode_len_mean_50",
            "recovered_rate_50",
            "righted_rate_50",
            "ever_righted_rate_50",
            "initial_upright_mean_50",
            "hard_reset_rate_50",
        ]

    def _on_training_start(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def _on_step(self):
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is not None:
                self.rows.append(
                    {
                        "reward": float(episode.get("r", 0.0)),
                        "length": float(episode.get("l", 0.0)),
                        "recovered": float(bool(info.get("recovered", False))),
                        "righted": float(bool(info.get("righted", False))),
                        "ever_righted": float(bool(info.get("ever_righted", False))),
                        "initial_upright": float(info.get("initial_upright", 0.0)),
                        "hard_reset": float(bool(info.get("hard_reset_sampled", False))),
                    }
                )
                self.rows = self.rows[-500:]

        if self.num_timesteps - self._last_log < self.log_interval or not self.rows:
            return True

        recent = self.rows[-50:]
        row = {
            "timesteps": self.num_timesteps,
            "episode_reward_mean_50": float(np.mean([x["reward"] for x in recent])),
            "episode_len_mean_50": float(np.mean([x["length"] for x in recent])),
            "recovered_rate_50": float(np.mean([x["recovered"] for x in recent])),
            "righted_rate_50": float(np.mean([x["righted"] for x in recent])),
            "ever_righted_rate_50": float(np.mean([x["ever_righted"] for x in recent])),
            "initial_upright_mean_50": float(np.mean([x["initial_upright"] for x in recent])),
            "hard_reset_rate_50": float(np.mean([x["hard_reset"] for x in recent])),
        }
        with self.csv_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)
        print(
            "[train] "
            f"steps={self.num_timesteps} "
            f"rew50={row['episode_reward_mean_50']:.1f} "
            f"righted50={row['righted_rate_50']:.2f} "
            f"ever_righted50={row['ever_righted_rate_50']:.2f} "
            f"recovered50={row['recovered_rate_50']:.2f} "
            f"init_upright50={row['initial_upright_mean_50']:.2f}",
            flush=True,
        )
        self._last_log = self.num_timesteps
        return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unitree_a1/scene.xml")
    parser.add_argument("--stage", choices=["righting", "stabilize"], default="righting")
    parser.add_argument("--run-dir", default="runs/two_stage_righting_curriculum_v1/stage_a_righting")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--success-steps", type=int, default=15)
    parser.add_argument("--failure-steps", type=int, default=180)
    parser.add_argument("--action-scale", type=float, default=0.9)
    parser.add_argument("--difficulty-start", type=float, default=None)
    parser.add_argument("--difficulty-end", type=float, default=0.45)
    parser.add_argument("--hard-reset-prob", type=float, default=None)
    parser.add_argument("--hard-upright-min", type=float, default=0.0)
    parser.add_argument("--hard-upright-max", type=float, default=0.5)
    parser.add_argument("--righting-target", type=float, default=0.62)
    parser.add_argument("--righting-success-steps", type=int, default=5)
    parser.add_argument("--vec-normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--vecnormalize-load", default=None)
    parser.add_argument("--reset-num-timesteps", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--start-method", choices=["forkserver", "spawn", "fork"], default="forkserver")
    parser.add_argument("--checkpoint-save-freq", type=int, default=250_000)
    parser.add_argument("--log-interval", type=int, default=20_000)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()

    if args.difficulty_start is None:
        args.difficulty_start = 0.35 if args.stage == "righting" else 0.25
    if args.hard_reset_prob is None:
        args.hard_reset_prob = 1.0 if args.stage == "righting" else 0.45
    return args


def main():
    args = parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(max(1, min(args.torch_threads, 4)))

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    env_fns = [make_env(args, i) for i in range(args.n_envs)]
    if args.n_envs > 1:
        env = SubprocVecEnv(env_fns, start_method=args.start_method)
    else:
        env = DummyVecEnv(env_fns)

    if args.vec_normalize and args.vecnormalize_load:
        env = VecNormalize.load(args.vecnormalize_load, env)
        env.training = True
        env.norm_reward = False
    elif args.vec_normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    if args.resume_from:
        model = PPO.load(
            args.resume_from,
            env=env,
            seed=args.seed,
            verbose=1,
            custom_objects={
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "n_epochs": args.n_epochs,
            },
        )
        model.learning_rate = args.learning_rate
        model.lr_schedule = get_schedule_fn(args.learning_rate)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.15,
            ent_coef=0.006,
        )

    reset_num_timesteps = args.reset_num_timesteps
    if reset_num_timesteps is None:
        reset_num_timesteps = args.resume_from is None

    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=max(args.checkpoint_save_freq // args.n_envs, 1),
                save_path=str(run_dir / "checkpoints"),
                name_prefix=f"ppo_two_stage_{args.stage}",
            ),
            TrainMetricsCallback(run_dir / "train_metrics.csv", log_interval=args.log_interval),
        ]
    )

    try:
        model.learn(
            total_timesteps=args.total_steps,
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps,
        )
        model.save(str(run_dir / f"ppo_two_stage_{args.stage}_final"))
        if args.vec_normalize:
            env.save(str(run_dir / "vecnormalize.pkl"))
    finally:
        env.close()


if __name__ == "__main__":
    main()
