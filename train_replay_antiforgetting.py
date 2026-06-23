"""Replay-balanced anti-forgetting fine-tuning for Unitree A1 recovery.

This script is intentionally isolated from train.py.  It fine-tunes a single
PPO policy from prev_fresh_final on a replay-balanced reset distribution:

- 60% ordinary recovery resets, difficulty sampled from [0.25, 0.45]
- 20% ordinary high difficulty resets, difficulty sampled from [0.50, 0.60]
- 20% hard-bin resets at difficulty 0.45 with initial_upright in [0.0, 0.5]

Anti-forgetting is implemented as a small KL distillation penalty against the
teacher policy on rollout observations, plus conservative learning rate and a
gradual action_scale schedule.
"""

import argparse
import copy
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from envs import UnitreeA1Env
from train_hard_bin import run_eval_episode


class ReplayBalancedRecoveryEnv(UnitreeA1Env):
    """UnitreeA1Env with per-reset replay bucket sampling."""

    def __init__(
        self,
        *args,
        general_prob=0.60,
        high_prob=0.20,
        hard_prob=0.20,
        general_min=0.25,
        general_max=0.45,
        high_min=0.50,
        high_max=0.60,
        hard_difficulty=0.45,
        hard_upright_min=0.0,
        hard_upright_max=0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        probs = np.array([general_prob, high_prob, hard_prob], dtype=np.float64)
        if probs.sum() <= 0:
            raise ValueError("At least one replay bucket probability must be positive")
        self.replay_probs = probs / probs.sum()
        self.general_min = float(general_min)
        self.general_max = float(general_max)
        self.high_min = float(high_min)
        self.high_max = float(high_max)
        self.hard_difficulty = float(hard_difficulty)
        self.hard_upright_min = float(hard_upright_min)
        self.hard_upright_max = float(hard_upright_max)
        self.replay_bucket = "general"

    def set_action_scale(self, value):
        self.action_scale = float(value)

    def _choose_replay_bucket(self):
        bucket = self.np_random.choice(["general", "high", "hard"], p=self.replay_probs)
        self.replay_bucket = str(bucket)
        if bucket == "general":
            self.set_recovery_difficulty_range(self.general_min, self.general_max)
            self.set_hard_reset_oversampling(0.0, self.hard_upright_min, self.hard_upright_max)
        elif bucket == "high":
            self.set_recovery_difficulty_range(self.high_min, self.high_max)
            self.set_hard_reset_oversampling(0.0, self.hard_upright_min, self.hard_upright_max)
        else:
            self.set_recovery_difficulty_range(self.hard_difficulty, self.hard_difficulty)
            self.set_hard_reset_oversampling(1.0, self.hard_upright_min, self.hard_upright_max)

    def reset(self, *, seed=None, options=None):
        self._choose_replay_bucket()
        obs, info = super().reset(seed=seed, options=options)
        info["replay_bucket"] = self.replay_bucket
        info["action_scale"] = float(self.action_scale)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["replay_bucket"] = self.replay_bucket
        info["action_scale"] = float(self.action_scale)
        return obs, reward, terminated, truncated, info


class ReplayAntiForgettingPPO(PPO):
    """PPO with a small teacher KL penalty to reduce policy forgetting."""

    def __init__(self, *args, distill_coef=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.distill_coef = float(distill_coef)
        self.teacher_policy = None

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["teacher_policy"]

    def set_teacher(self, teacher_model_path):
        if not teacher_model_path:
            self.teacher_policy = None
            return
        teacher = PPO.load(teacher_model_path, device=self.device)
        teacher.policy.set_training_mode(False)
        for param in teacher.policy.parameters():
            param.requires_grad_(False)
        self.teacher_policy = teacher.policy

    def _teacher_kl(self, observations):
        if self.teacher_policy is None or self.distill_coef <= 0.0:
            return observations.new_tensor(0.0)
        new_dist = self.policy.get_distribution(observations)
        with th.no_grad():
            teacher_dist = self.teacher_policy.get_distribution(observations)
        kl = th.distributions.kl_divergence(teacher_dist.distribution, new_dist.distribution)
        if kl.ndim > 1:
            kl = kl.sum(dim=1)
        return kl.mean()

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        teacher_kls = []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())

                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                teacher_kl = self._teacher_kl(rollout_data.observations)
                teacher_kls.append(float(teacher_kl.detach().cpu().item()))
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.distill_coef * teacher_kl
                )

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/teacher_kl", np.mean(teacher_kls) if teacher_kls else 0.0)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


class ActionScaleScheduleCallback(BaseCallback):
    def __init__(self, start, end, schedule_steps, update_interval=10_000):
        super().__init__()
        self.start = float(start)
        self.end = float(end)
        self.schedule_steps = max(int(schedule_steps), 1)
        self.update_interval = max(int(update_interval), 1)
        self.current_scale = self.start
        self._last_update = -1

    def _on_training_start(self):
        self._set_scale(self.start)

    def _on_step(self):
        if self.num_timesteps - self._last_update < self.update_interval:
            return True
        progress = min(float(self.num_timesteps) / float(self.schedule_steps), 1.0)
        scale = self.start + progress * (self.end - self.start)
        self._set_scale(scale)
        return True

    def _set_scale(self, scale):
        self.current_scale = float(scale)
        self.training_env.env_method("set_action_scale", self.current_scale)
        self._last_update = self.num_timesteps


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
            "general_rate_50",
            "high_rate_50",
            "hard_rate_50",
            "hard_bin_recovered_rate_50",
            "initial_upright_mean_50",
            "action_scale_mean_50",
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
                bucket = str(info.get("replay_bucket", "unknown"))
                hard_bin = float(info.get("initial_upright", 1.0)) < 0.5
                self.rows.append(
                    {
                        "reward": float(episode.get("r", 0.0)),
                        "length": float(episode.get("l", 0.0)),
                        "recovered": float(bool(info.get("recovered", False))),
                        "bucket": bucket,
                        "hard_bin": float(hard_bin),
                        "initial_upright": float(info.get("initial_upright", 0.0)),
                        "action_scale": float(info.get("action_scale", 0.0)),
                    }
                )
                self.rows = self.rows[-500:]
        if self.num_timesteps - self._last_log < self.log_interval or not self.rows:
            return True

        recent = self.rows[-50:]
        def mean_where(key, value):
            selected = [row["recovered"] for row in recent if row[key] == value]
            return float(np.mean(selected)) if selected else 0.0

        hard_rows = [row["recovered"] for row in recent if row["hard_bin"] > 0.5]
        row = {
            "timesteps": self.num_timesteps,
            "episode_reward_mean_50": float(np.mean([x["reward"] for x in recent])),
            "episode_len_mean_50": float(np.mean([x["length"] for x in recent])),
            "recovered_rate_50": float(np.mean([x["recovered"] for x in recent])),
            "general_rate_50": mean_where("bucket", "general"),
            "high_rate_50": mean_where("bucket", "high"),
            "hard_rate_50": mean_where("bucket", "hard"),
            "hard_bin_recovered_rate_50": float(np.mean(hard_rows)) if hard_rows else 0.0,
            "initial_upright_mean_50": float(np.mean([x["initial_upright"] for x in recent])),
            "action_scale_mean_50": float(np.mean([x["action_scale"] for x in recent])),
        }
        with self.csv_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)
        print(
            "[train] "
            f"steps={self.num_timesteps} "
            f"rew50={row['episode_reward_mean_50']:.1f} "
            f"rec50={row['recovered_rate_50']:.2f} "
            f"general={row['general_rate_50']:.2f} "
            f"high={row['high_rate_50']:.2f} "
            f"hard={row['hard_rate_50']:.2f} "
            f"scale={row['action_scale_mean_50']:.3f}",
            flush=True,
        )
        self._last_log = self.num_timesteps
        return True


class CompositeEvalCallback(BaseCallback):
    def __init__(
        self,
        save_path,
        model_path,
        action_scale_callback,
        max_episode_steps=1000,
        success_steps=15,
        failure_steps=180,
        eval_freq=250_000,
        seed=91_000,
        sweep_difficulties=(0.25, 0.35, 0.45, 0.50, 0.60),
        sweep_episodes=30,
        hard_episodes=100,
        hard_upright_min=0.0,
        hard_upright_max=0.5,
    ):
        super().__init__()
        self.save_path = Path(save_path)
        self.model_path = model_path
        self.action_scale_callback = action_scale_callback
        self.max_episode_steps = int(max_episode_steps)
        self.success_steps = int(success_steps)
        self.failure_steps = int(failure_steps)
        self.eval_freq = int(eval_freq)
        self.seed = int(seed)
        self.sweep_difficulties = [float(x) for x in sweep_difficulties]
        self.sweep_episodes = int(sweep_episodes)
        self.hard_episodes = int(hard_episodes)
        self.hard_upright_min = float(hard_upright_min)
        self.hard_upright_max = float(hard_upright_max)
        self._last_eval = -1
        self.best_score = -np.inf

    def _on_training_start(self):
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self):
        if self.eval_freq <= 0 or self.num_timesteps - self._last_eval < self.eval_freq:
            return True
        scale = float(self.action_scale_callback.current_scale)
        metrics = evaluate_composite(
            model=self.model,
            train_vec_normalize=self.model.get_vec_normalize_env(),
            model_path=self.model_path,
            seed=self.seed + self.num_timesteps,
            action_scale=scale,
            max_episode_steps=self.max_episode_steps,
            success_steps=self.success_steps,
            failure_steps=self.failure_steps,
            sweep_difficulties=self.sweep_difficulties,
            sweep_episodes=self.sweep_episodes,
            hard_episodes=self.hard_episodes,
            hard_upright_min=self.hard_upright_min,
            hard_upright_max=self.hard_upright_max,
        )
        self._last_eval = self.num_timesteps
        score = float(metrics["score"])
        print(
            "[composite-eval] "
            f"steps={self.num_timesteps} "
            f"scale={scale:.3f} "
            f"sweep={metrics['sweep_recovered']}/{metrics['sweep_episodes']} "
            f"hard={metrics['hard_recovered']}/{metrics['hard_episodes']} "
            f"score={score:.3f} "
            f"best={self.best_score:.3f}",
            flush=True,
        )
        metrics_path = self.save_path / f"metrics_{self.num_timesteps}.json"
        with metrics_path.open("w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        if score > self.best_score:
            self.best_score = score
            self.model.save(str(self.save_path / "best_model"))
            vec_normalize = self.model.get_vec_normalize_env()
            if vec_normalize is not None:
                vec_normalize.save(str(self.save_path / "best_vecnormalize.pkl"))
            with (self.save_path / "best_metrics.json").open("w") as f:
                json.dump(metrics, f, indent=2, sort_keys=True)
        return True


def make_train_env(args, rank):
    def _init():
        env = ReplayBalancedRecoveryEnv(
            args.model,
            task="recovery",
            max_episode_steps=args.max_episode_steps,
            recovery_difficulty=args.general_min,
            normalize_obs=False,
            action_scale=args.action_scale_start,
            success_steps=args.success_steps,
            failure_steps=args.failure_steps,
            general_prob=args.general_prob,
            high_prob=args.high_prob,
            hard_prob=args.hard_prob,
            general_min=args.general_min,
            general_max=args.general_max,
            high_min=args.high_min,
            high_max=args.high_max,
            hard_difficulty=args.hard_difficulty,
            hard_upright_min=args.hard_upright_min,
            hard_upright_max=args.hard_upright_max,
        )
        env.reset(seed=args.seed + rank)
        return Monitor(env)
    return _init


def evaluate_composite(
    model,
    train_vec_normalize,
    model_path,
    seed,
    action_scale,
    max_episode_steps,
    success_steps,
    failure_steps,
    sweep_difficulties,
    sweep_episodes,
    hard_episodes,
    hard_upright_min,
    hard_upright_max,
):
    sweep_rows = []
    for difficulty in sweep_difficulties:
        recovered = 0
        failed = 0
        for episode in range(sweep_episodes):
            info = run_eval_episode(
                model=model,
                train_vec_normalize=train_vec_normalize,
                model_path=model_path,
                seed=seed + 100_000 + int(round(1000 * difficulty)) + episode,
                difficulty=float(difficulty),
                hard_reset_prob=0.0,
                hard_upright_min=hard_upright_min,
                hard_upright_max=hard_upright_max,
                action_scale=action_scale,
                max_episode_steps=max_episode_steps,
                success_steps=success_steps,
                failure_steps=failure_steps,
            )
            recovered += int(bool(info.get("recovered", False)))
            failed += int(bool(info.get("failure_timeout", False) or info.get("catastrophic", False)))
        sweep_rows.append({"difficulty": float(difficulty), "episodes": sweep_episodes, "recovered": recovered, "failed": failed})

    hard_infos = []
    attempts = 0
    max_attempts = hard_episodes * 10
    while len(hard_infos) < hard_episodes and attempts < max_attempts:
        info = run_eval_episode(
            model=model,
            train_vec_normalize=train_vec_normalize,
            model_path=model_path,
            seed=seed + attempts,
            difficulty=0.45,
            hard_reset_prob=1.0,
            hard_upright_min=hard_upright_min,
            hard_upright_max=hard_upright_max,
            action_scale=action_scale,
            max_episode_steps=max_episode_steps,
            success_steps=success_steps,
            failure_steps=failure_steps,
        )
        initial_upright = float(info.get("initial_upright", 99.0))
        if hard_upright_min <= initial_upright <= hard_upright_max:
            hard_infos.append(info)
        attempts += 1

    hard_recovered = sum(int(bool(info.get("recovered", False))) for info in hard_infos)
    hard_failed = sum(int(bool(info.get("failure_timeout", False) or info.get("catastrophic", False))) for info in hard_infos)
    bins = {
        "[0.0,0.25)": {"episodes": 0, "recovered": 0},
        "[0.25,0.5)": {"episodes": 0, "recovered": 0},
    }
    for info in hard_infos:
        initial_upright = float(info.get("initial_upright", 0.0))
        key = "[0.0,0.25)" if initial_upright < 0.25 else "[0.25,0.5)"
        bins[key]["episodes"] += 1
        bins[key]["recovered"] += int(bool(info.get("recovered", False)))

    sweep_recovered = sum(row["recovered"] for row in sweep_rows)
    total_sweep_episodes = sum(row["episodes"] for row in sweep_rows)
    sweep_rate = sweep_recovered / max(total_sweep_episodes, 1)
    hard_rate = hard_recovered / max(len(hard_infos), 1)
    score = 0.7 * sweep_rate + 0.3 * hard_rate
    return {
        "seed": int(seed),
        "action_scale": float(action_scale),
        "sweep": sweep_rows,
        "sweep_episodes": int(total_sweep_episodes),
        "sweep_recovered": int(sweep_recovered),
        "sweep_success_rate": float(sweep_rate),
        "hard_episodes": int(len(hard_infos)),
        "hard_recovered": int(hard_recovered),
        "hard_failed": int(hard_failed),
        "hard_attempts": int(attempts),
        "hard_success_rate": float(hard_rate),
        "hard_bins": bins,
        "score": float(score),
    }


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unitree_a1/scene.xml")
    parser.add_argument("--run-dir", default="runs/replay_antiforgetting_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-steps", type=int, default=750_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--distill-coef", type=float, default=0.02)
    parser.add_argument("--teacher-model", default="runs/recovery_fresh_curriculum_025_to_045_reward_v2/ppo_recovery_stand_final.zip")
    parser.add_argument("--resume-from", default="runs/recovery_fresh_curriculum_025_to_045_reward_v2/ppo_recovery_stand_final.zip")
    parser.add_argument("--vecnormalize-load", default="runs/recovery_fresh_curriculum_025_to_045_reward_v2/vecnormalize.pkl")
    parser.add_argument("--reset-num-timesteps", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--success-steps", type=int, default=15)
    parser.add_argument("--failure-steps", type=int, default=180)
    parser.add_argument("--action-scale-start", type=float, default=0.5)
    parser.add_argument("--action-scale-end", type=float, default=0.65)
    parser.add_argument("--action-scale-steps", type=int, default=500_000)
    parser.add_argument("--general-prob", type=float, default=0.60)
    parser.add_argument("--high-prob", type=float, default=0.20)
    parser.add_argument("--hard-prob", type=float, default=0.20)
    parser.add_argument("--general-min", type=float, default=0.25)
    parser.add_argument("--general-max", type=float, default=0.45)
    parser.add_argument("--high-min", type=float, default=0.50)
    parser.add_argument("--high-max", type=float, default=0.60)
    parser.add_argument("--hard-difficulty", type=float, default=0.45)
    parser.add_argument("--hard-upright-min", type=float, default=0.0)
    parser.add_argument("--hard-upright-max", type=float, default=0.5)
    parser.add_argument("--vec-normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start-method", choices=["forkserver", "spawn", "fork"], default="forkserver")
    parser.add_argument("--checkpoint-save-freq", type=int, default=250_000)
    parser.add_argument("--log-interval", type=int, default=20_000)
    parser.add_argument("--eval-freq", type=int, default=250_000)
    parser.add_argument("--eval-sweep-episodes", type=int, default=30)
    parser.add_argument("--eval-hard-episodes", type=int, default=100)
    parser.add_argument("--eval-difficulties", default="0.25,0.35,0.45,0.50,0.60")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.torch_threads > 0:
        th.set_num_threads(args.torch_threads)
        th.set_num_interop_threads(max(1, args.torch_threads))

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    env_fns = [make_train_env(args, rank) for rank in range(args.n_envs)]
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
        model = ReplayAntiForgettingPPO.load(
            args.resume_from,
            env=env,
            seed=args.seed,
            verbose=1,
            device="auto",
            custom_objects={
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "n_epochs": args.n_epochs,
            },
        )
        model.learning_rate = args.learning_rate
        model.lr_schedule = get_schedule_fn(args.learning_rate)
        model.distill_coef = float(args.distill_coef)
    else:
        model = ReplayAntiForgettingPPO(
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
            clip_range=0.12,
            ent_coef=0.004,
            distill_coef=args.distill_coef,
        )
    model.set_teacher(args.teacher_model)

    action_scale_callback = ActionScaleScheduleCallback(
        start=args.action_scale_start,
        end=args.action_scale_end,
        schedule_steps=args.action_scale_steps,
        update_interval=args.log_interval,
    )
    callbacks = [
        action_scale_callback,
        CheckpointCallback(
            save_freq=max(args.checkpoint_save_freq // args.n_envs, 1),
            save_path=str(checkpoint_dir),
            name_prefix="ppo_replay_antiforgetting",
        ),
        TrainMetricsCallback(run_dir / "train_metrics.csv", log_interval=args.log_interval),
        CompositeEvalCallback(
            save_path=run_dir / "best_eval",
            model_path=args.model,
            action_scale_callback=action_scale_callback,
            max_episode_steps=args.max_episode_steps,
            success_steps=args.success_steps,
            failure_steps=args.failure_steps,
            eval_freq=args.eval_freq,
            seed=args.seed + 91_000,
            sweep_difficulties=parse_float_list(args.eval_difficulties),
            sweep_episodes=args.eval_sweep_episodes,
            hard_episodes=args.eval_hard_episodes,
            hard_upright_min=args.hard_upright_min,
            hard_upright_max=args.hard_upright_max,
        ),
    ]

    try:
        model.learn(
            total_timesteps=args.total_steps,
            callback=CallbackList(callbacks),
            reset_num_timesteps=args.reset_num_timesteps,
        )
        model.save(str(run_dir / "ppo_replay_antiforgetting_final"))
        if args.vec_normalize:
            env.save(str(run_dir / "vecnormalize.pkl"))
    finally:
        env.close()


if __name__ == "__main__":
    # Keep worker subprocesses from multiplying BLAS/PyTorch threads.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
