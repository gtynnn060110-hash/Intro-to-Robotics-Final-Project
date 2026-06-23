"""Isolated hard-bin recovery training for Unitree A1.

This script intentionally does not import or modify train.py.  It defines a
small reward wrapper around UnitreeA1Env for a two-stage hard-bin experiment:

Stage A (`--stage righting`) learns to raise initial_upright in [0.0, 0.5].
Stage B (`--stage full`) fine-tunes the same policy on full stable recovery
with hard reset oversampling mixed with ordinary recovery resets.
"""

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from envs import UnitreeA1Env


class HardBinRecoveryEnv(UnitreeA1Env):
    """UnitreeA1Env with isolated hard-bin reward shaping.

    The observation/action spaces stay identical to UnitreeA1Env.  Evaluation
    should still use the base recovered flag from UnitreeA1Env when measuring
    full recovery.
    """

    def __init__(
        self,
        *args,
        reward_mode="righting",
        hard_reset_prob=1.0,
        hard_upright_min=0.0,
        hard_upright_max=0.5,
        difficulty_min=0.45,
        difficulty_max=0.45,
        righting_target=0.62,
        righting_success_steps=6,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reward_mode = str(reward_mode)
        self.righting_target = float(righting_target)
        self.righting_success_steps_required = int(righting_success_steps)
        self.righting_steps = 0
        self.righted = False
        self.full_extra_reward = 0.0
        self.set_recovery_difficulty_range(difficulty_min, difficulty_max)
        self.set_hard_reset_oversampling(hard_reset_prob, hard_upright_min, hard_upright_max)

    def reset(self, *, seed=None, options=None):
        self.righting_steps = 0
        self.righted = False
        obs, info = super().reset(seed=seed, options=options)
        info["reward_mode"] = self.reward_mode
        info["righted"] = False
        info["righting_steps"] = 0
        return obs, info

    def step(self, action):
        prev_upright = float(self.prev_upright)
        prev_abs_height_error = float(self.prev_abs_height_error)
        obs, base_reward, terminated, truncated, info = super().step(action)

        if self.reward_mode == "righting":
            reward, terminated = self._righting_reward(
                base_reward=base_reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                prev_upright=prev_upright,
                prev_abs_height_error=prev_abs_height_error,
            )
        elif self.reward_mode == "full":
            reward = self._full_recovery_extra_reward(
                base_reward=base_reward,
                info=info,
                prev_upright=prev_upright,
                prev_abs_height_error=prev_abs_height_error,
            )
        else:
            reward = float(base_reward)

        info["reward_mode"] = self.reward_mode
        info["righted"] = bool(self.righted)
        info["righting_steps"] = int(self.righting_steps)
        info["reward_base_recovery"] = float(base_reward)
        info["reward_full_extra"] = float(self.full_extra_reward)
        return obs, float(reward), terminated, truncated, info

    def _righting_reward(self, base_reward, terminated, truncated, info, prev_upright, prev_abs_height_error):
        upright = float(info["upright"])
        height_error = float(info["height_error"])
        z = float(info["z"])
        target_z = float(info["target_z"])
        upright_progress = upright - prev_upright
        height_progress = prev_abs_height_error - abs(height_error)
        time_fraction = min(float(self.steps) / max(float(self.max_episode_steps), 1.0), 1.0)

        righting_margin = max(self.righting_target - self.initial_upright, 0.05)
        normalized_gain = (upright - self.initial_upright) / righting_margin
        target_bonus = max(0.0, upright - 0.30)
        low_height_margin = max(0.0, target_z - z - 0.06)
        lin_vel = np.asarray(self.data.qvel[:3])
        ang_vel = np.asarray(self.data.qvel[3:6])
        joint_vel = np.asarray(self.data.qvel[-self.n_joints :])

        reward = (
            3.0 * np.clip((upright + 1.0) * 0.5, 0.0, 1.0)
            + 6.0 * np.clip(normalized_gain, -0.4, 1.3)
            + 22.0 * np.clip(upright_progress, -0.04, 0.10)
            + 8.0 * np.clip(height_progress, -0.04, 0.08)
            + 2.5 * target_bonus
            - 18.0 * low_height_margin
            - 0.015 * float(np.dot(ang_vel, ang_vel))
            - 0.002 * float(np.mean(np.square(joint_vel)))
            - 0.004 * float(np.dot(lin_vel[:2], lin_vel[:2]))
            - 0.02 * time_fraction
        )

        currently_righted = bool(upright >= self.righting_target and z > target_z - 0.09)
        if currently_righted:
            self.righting_steps += 1
        elif upright < self.righting_target - 0.08:
            self.righting_steps = 0
        else:
            self.righting_steps = max(0, self.righting_steps - 1)

        self.righted = bool(self.righting_steps >= self.righting_success_steps_required or info.get("recovered", False))
        if bool(info.get("catastrophic", False)):
            reward -= 35.0
            terminated = True
        elif bool(info.get("failure_timeout", False)):
            reward -= 25.0
            terminated = True
        elif self.righted:
            reward += 70.0 + 25.0 * (1.0 - time_fraction)
            terminated = True
        elif truncated:
            reward -= 20.0

        info["righting_target"] = self.righting_target
        info["reward_righting_total"] = float(reward)
        return float(reward), bool(terminated)

    def _full_recovery_extra_reward(self, base_reward, info, prev_upright, prev_abs_height_error):
        upright = float(info["upright"])
        height_error = float(info["height_error"])
        z = float(info["z"])
        target_z = float(info["target_z"])
        upright_progress = upright - prev_upright
        height_progress = prev_abs_height_error - abs(height_error)
        low_start = self.initial_upright < 0.5

        extra = 0.0
        if low_start:
            stable = bool(info.get("stable", False))
            recovered = bool(info.get("recovered", False))
            terminal_bad = bool(
                info.get("failure_timeout", False)
                or info.get("catastrophic", False)
                or info.get("episode_timeout", False)
            )
            low_height_margin = max(0.0, target_z - z - 0.06)
            righting_gain = upright - self.initial_upright

            extra += 34.0 * np.clip(upright_progress, -0.04, 0.10)
            extra += 12.0 * np.clip(height_progress, -0.04, 0.08)
            extra += 3.0 * np.clip(righting_gain, -0.2, 0.8)
            extra += 2.0 if upright > 0.55 else 0.0
            extra += 8.0 if stable else -1.6
            extra -= 20.0 * low_height_margin

            if recovered:
                extra += 180.0
            elif terminal_bad:
                extra -= 140.0

            # The base dense reward is too high for long hard-start failures.
            reward = 0.35 * float(base_reward) + extra
        else:
            reward = float(base_reward)

        self.full_extra_reward = float(extra)
        return float(reward)


def make_train_env(args, seed):
    def _init():
        env = HardBinRecoveryEnv(
            args.model,
            task="recovery",
            max_episode_steps=args.max_episode_steps,
            recovery_difficulty=args.difficulty_end,
            normalize_obs=False,
            action_scale=args.action_scale,
            success_steps=args.success_steps,
            failure_steps=args.failure_steps,
            reward_mode=args.stage,
            hard_reset_prob=args.hard_reset_prob,
            hard_upright_min=args.hard_upright_min,
            hard_upright_max=args.hard_upright_max,
            difficulty_min=args.difficulty_start,
            difficulty_max=args.difficulty_end,
            righting_target=args.righting_target,
            righting_success_steps=args.righting_success_steps,
        )
        env.reset(seed=seed)
        return Monitor(env)

    return _init


def copy_vecnormalize(eval_env, train_vec_normalize):
    if train_vec_normalize is None:
        return eval_env
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=train_vec_normalize.clip_obs)
    eval_env.obs_rms = copy.deepcopy(train_vec_normalize.obs_rms)
    eval_env.ret_rms = copy.deepcopy(train_vec_normalize.ret_rms)
    eval_env.training = False
    eval_env.norm_reward = False
    return eval_env


def make_base_eval_env(
    model_path,
    action_scale,
    max_episode_steps,
    success_steps,
    failure_steps,
    difficulty,
    hard_reset_prob,
    hard_upright_min,
    hard_upright_max,
):
    def _init():
        env = UnitreeA1Env(
            model_path,
            task="recovery",
            max_episode_steps=max_episode_steps,
            recovery_difficulty=difficulty,
            normalize_obs=False,
            action_scale=action_scale,
            success_steps=success_steps,
            failure_steps=failure_steps,
        )
        env.set_hard_reset_oversampling(hard_reset_prob, hard_upright_min, hard_upright_max)
        return Monitor(env)

    return DummyVecEnv([_init])


def get_reset_info(vec_env):
    base = vec_env.venv if isinstance(vec_env, VecNormalize) else vec_env
    infos = getattr(base, "reset_infos", None)
    if infos:
        return dict(infos[0])
    return {}


def run_eval_episode(
    model,
    train_vec_normalize,
    model_path,
    seed,
    difficulty,
    hard_reset_prob,
    hard_upright_min,
    hard_upright_max,
    action_scale,
    max_episode_steps,
    success_steps,
    failure_steps,
):
    env = make_base_eval_env(
        model_path=model_path,
        action_scale=action_scale,
        max_episode_steps=max_episode_steps,
        success_steps=success_steps,
        failure_steps=failure_steps,
        difficulty=difficulty,
        hard_reset_prob=hard_reset_prob,
        hard_upright_min=hard_upright_min,
        hard_upright_max=hard_upright_max,
    )
    env = copy_vecnormalize(env, train_vec_normalize)
    env.seed(int(seed))
    obs = env.reset()
    reset_info = get_reset_info(env)
    final_info = {}
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(action)
            final_info = dict(infos[0])
            if bool(dones[0]):
                break
    finally:
        env.close()
    final_info.update({f"reset_{key}": value for key, value in reset_info.items()})
    if "initial_upright" not in final_info and "initial_upright" in reset_info:
        final_info["initial_upright"] = reset_info["initial_upright"]
    final_info["seed"] = int(seed)
    final_info["difficulty"] = float(difficulty)
    return final_info


def summarize_infos(infos):
    recovered = sum(int(bool(info.get("recovered", False))) for info in infos)
    failed = sum(
        int(bool(info.get("failure_timeout", False) or info.get("catastrophic", False)))
        for info in infos
    )
    righted = sum(int(float(info.get("upright", -1.0)) >= 0.62) for info in infos)
    return {
        "episodes": len(infos),
        "recovered": recovered,
        "failed": failed,
        "righted_final_upright_ge_062": righted,
        "mean_len": float(np.mean([info.get("episode", {}).get("l", 0.0) for info in infos])) if infos else 0.0,
        "mean_return": float(np.mean([info.get("episode", {}).get("r", 0.0) for info in infos])) if infos else 0.0,
        "mean_initial_upright": float(np.mean([info.get("initial_upright", 0.0) for info in infos])) if infos else 0.0,
    }


class TrainMetricsCallback(BaseCallback):
    def __init__(self, csv_path, log_interval=10_000, verbose=0):
        super().__init__(verbose)
        self.csv_path = Path(csv_path)
        self.log_interval = int(log_interval)
        self._last_log = 0
        self._episode_rows = []
        self._fieldnames = [
            "timesteps",
            "episode_reward_mean_50",
            "episode_len_mean_50",
            "recovered_rate_50",
            "righted_rate_50",
            "initial_upright_mean_50",
            "hard_reset_rate_50",
        ]

    def _on_training_start(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=self._fieldnames).writeheader()

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            episode = info.get("episode")
            if episode is not None:
                self._episode_rows.append(
                    {
                        "reward": float(episode.get("r", 0.0)),
                        "length": float(episode.get("l", 0.0)),
                        "recovered": float(bool(info.get("recovered", False))),
                        "righted": float(bool(info.get("righted", False))),
                        "initial_upright": float(info.get("initial_upright", 0.0)),
                        "hard_reset": float(bool(info.get("hard_reset_sampled", False))),
                    }
                )
                self._episode_rows = self._episode_rows[-500:]

        if self.num_timesteps - self._last_log < self.log_interval or not self._episode_rows:
            return True

        recent = self._episode_rows[-50:]
        row = {
            "timesteps": self.num_timesteps,
            "episode_reward_mean_50": float(np.mean([item["reward"] for item in recent])),
            "episode_len_mean_50": float(np.mean([item["length"] for item in recent])),
            "recovered_rate_50": float(np.mean([item["recovered"] for item in recent])),
            "righted_rate_50": float(np.mean([item["righted"] for item in recent])),
            "initial_upright_mean_50": float(np.mean([item["initial_upright"] for item in recent])),
            "hard_reset_rate_50": float(np.mean([item["hard_reset"] for item in recent])),
        }
        with self.csv_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._fieldnames).writerow(row)
        print(
            "[train] "
            f"steps={self.num_timesteps} "
            f"rew50={row['episode_reward_mean_50']:.1f} "
            f"righted50={row['righted_rate_50']:.2f} "
            f"recovered50={row['recovered_rate_50']:.2f} "
            f"init_upright50={row['initial_upright_mean_50']:.2f}",
            flush=True,
        )
        self._last_log = self.num_timesteps
        return True


class HardBinEvalCallback(BaseCallback):
    def __init__(
        self,
        save_path,
        model_path,
        action_scale,
        max_episode_steps,
        success_steps,
        failure_steps,
        eval_freq=200_000,
        n_eval_episodes=20,
        seed=70_000,
        hard_upright_min=0.0,
        hard_upright_max=0.5,
        verbose=0,
    ):
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.model_path = model_path
        self.action_scale = float(action_scale)
        self.max_episode_steps = int(max_episode_steps)
        self.success_steps = int(success_steps)
        self.failure_steps = int(failure_steps)
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.seed = int(seed)
        self.hard_upright_min = float(hard_upright_min)
        self.hard_upright_max = float(hard_upright_max)
        self._last_eval = -1
        self.best_score = -np.inf

    def _on_training_start(self):
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self):
        if self.eval_freq <= 0 or self.num_timesteps - self._last_eval < self.eval_freq:
            return True
        train_vec_normalize = self.model.get_vec_normalize_env()
        infos = []
        attempts = 0
        while len(infos) < self.n_eval_episodes and attempts < self.n_eval_episodes * 10:
            seed = self.seed + self.num_timesteps + attempts
            info = run_eval_episode(
                model=self.model,
                train_vec_normalize=train_vec_normalize,
                model_path=self.model_path,
                seed=seed,
                difficulty=0.45,
                hard_reset_prob=1.0,
                hard_upright_min=self.hard_upright_min,
                hard_upright_max=self.hard_upright_max,
                action_scale=self.action_scale,
                max_episode_steps=self.max_episode_steps,
                success_steps=self.success_steps,
                failure_steps=self.failure_steps,
            )
            upright0 = float(info.get("initial_upright", 99.0))
            if self.hard_upright_min <= upright0 <= self.hard_upright_max:
                infos.append(info)
            attempts += 1

        metrics = summarize_infos(infos)
        hard_rate = metrics["recovered"] / max(metrics["episodes"], 1)
        score = hard_rate - 0.05 * (metrics["failed"] / max(metrics["episodes"], 1))
        self._last_eval = self.num_timesteps
        print(
            "[hard-eval] "
            f"steps={self.num_timesteps} "
            f"recovered={metrics['recovered']}/{metrics['episodes']} "
            f"failed={metrics['failed']}/{metrics['episodes']} "
            f"righted={metrics['righted_final_upright_ge_062']}/{metrics['episodes']} "
            f"score={score:.3f} "
            f"best={self.best_score:.3f}",
            flush=True,
        )
        if score > self.best_score:
            self.best_score = score
            self.model.save(str(self.save_path / "best_model"))
            if train_vec_normalize is not None:
                train_vec_normalize.save(str(self.save_path / "best_vecnormalize.pkl"))
            with (self.save_path / "best_metrics.json").open("w") as f:
                json.dump({"timesteps": self.num_timesteps, "score": score, **metrics}, f, indent=2)
        return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unitree_a1/scene.xml")
    parser.add_argument("--stage", choices=["righting", "full"], default="righting")
    parser.add_argument("--run-dir", default="runs/hard_bin_stage")
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
    parser.add_argument("--righting-success-steps", type=int, default=6)
    parser.add_argument("--vec-normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--vecnormalize-load", default=None)
    parser.add_argument("--reset-num-timesteps", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--start-method", choices=["forkserver", "spawn", "fork"], default="forkserver")
    parser.add_argument("--checkpoint-save-freq", type=int, default=250_000)
    parser.add_argument("--log-interval", type=int, default=20_000)
    parser.add_argument("--eval-freq", type=int, default=250_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    args = parser.parse_args()

    if args.difficulty_start is None:
        args.difficulty_start = 0.45 if args.stage == "righting" else 0.25
    if args.hard_reset_prob is None:
        args.hard_reset_prob = 1.0 if args.stage == "righting" else 0.75
    return args


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    env_fns = [make_train_env(args, args.seed + i) for i in range(args.n_envs)]
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

    callbacks = [
        CheckpointCallback(
            save_freq=max(args.checkpoint_save_freq // args.n_envs, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix=f"ppo_hard_bin_{args.stage}",
        ),
        TrainMetricsCallback(run_dir / "train_metrics.csv", log_interval=args.log_interval),
        HardBinEvalCallback(
            save_path=run_dir / "best_eval",
            model_path=args.model,
            action_scale=args.action_scale,
            max_episode_steps=args.max_episode_steps,
            success_steps=args.success_steps,
            failure_steps=args.failure_steps,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            seed=args.seed + 70_000,
            hard_upright_min=args.hard_upright_min,
            hard_upright_max=args.hard_upright_max,
        ),
    ]

    try:
        model.learn(
            total_timesteps=args.total_steps,
            callback=CallbackList(callbacks),
            reset_num_timesteps=reset_num_timesteps,
        )
        model.save(str(run_dir / f"ppo_hard_bin_{args.stage}_final"))
        if args.vec_normalize:
            env.save(str(run_dir / "vecnormalize.pkl"))
    finally:
        env.close()


if __name__ == "__main__":
    main()
