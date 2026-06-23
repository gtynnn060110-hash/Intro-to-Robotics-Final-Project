"""Train the Unitree A1 recovery policy used by prev_fresh_final.

This is the clean recovery-curriculum training entrypoint. It keeps the later
resume/hard-reset options available, but its defaults reproduce the original
flat-to-medium recovery run:

    difficulty 0.25 -> 0.45 over 4M PPO steps, mixed curriculum sampling,
    no hard-reset oversampling.

Run with `python train_recovery.py` for headless training.
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from envs import UnitreeA1Env


class WandbMetricsCallback(BaseCallback):
    def __init__(self, run, log_interval=1000, model_save_freq=50_000, model_save_path=None, verbose=0):
        super().__init__(verbose)
        self.run = run
        self.log_interval = int(log_interval)
        self.model_save_freq = int(model_save_freq)
        self.model_save_path = Path(model_save_path) if model_save_path else None
        self._episode_rewards = []

    def _on_training_start(self):
        if self.model_save_path is not None:
            self.model_save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self):
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])

        for info in infos:
            episode = info.get("episode")
            if episode is not None:
                self._episode_rewards.append(float(episode["r"]))

        should_log = self.num_timesteps % self.log_interval == 0
        if should_log:
            payload = {"time/total_timesteps": self.num_timesteps}
            if len(rewards) > 0:
                payload["rollout/reward_step_mean"] = float(np.mean(rewards))
            if len(dones) > 0:
                payload["rollout/done_rate"] = float(np.mean(dones))
            if self._episode_rewards:
                payload["rollout/ep_rew_mean"] = float(np.mean(self._episode_rewards[-20:]))

            scalar_keys = [
                "z",
                "target_z",
                "upright",
                "height_error",
                "fallen",
                "catastrophic",
                "stable",
                "recovered",
                "failure_timeout",
                "episode_timeout",
                "initial_upright",
                "initial_height_error",
                "hard_reset_sampled",
                "success_steps",
                "failure_steps",
                "failure_steps_limit",
                "recovery_difficulty",
                "recovery_difficulty_min",
                "recovery_difficulty_max",
                "reward_upright",
                "reward_height",
                "reward_low_height_penalty",
                "reward_time_penalty",
                "reward_progress",
                "reward_righting_progress",
                "reward_success",
                "reward_terminal",
            ]
            for key in scalar_keys:
                values = [info[key] for info in infos if key in info]
                if values:
                    payload[f"env/{key}"] = float(np.mean(values))

            self.run.log(payload, step=self.num_timesteps)

        should_save = (
            self.model_save_path is not None
            and self.model_save_freq > 0
            and self.num_timesteps % self.model_save_freq == 0
        )
        if should_save:
            self.model.save(str(self.model_save_path / f"model_{self.num_timesteps}_steps"))
            vec_normalize = self.model.get_vec_normalize_env()
            if vec_normalize is not None:
                vec_normalize.save(str(self.model_save_path / f"vecnormalize_{self.num_timesteps}_steps.pkl"))

        return True


class RecoveryCurriculumCallback(BaseCallback):
    def __init__(
        self,
        start,
        end,
        curriculum_steps,
        update_interval=1000,
        relative=False,
        mix_difficulty=False,
        hard_reset_prob=0.0,
        hard_reset_threshold=0.35,
        hard_reset_upright_min=0.0,
        hard_reset_upright_max=0.5,
        verbose=0,
    ):
        super().__init__(verbose)
        self.start = float(start)
        self.end = float(end)
        self.curriculum_steps = max(int(curriculum_steps), 1)
        self.update_interval = max(int(update_interval), 1)
        self.relative = bool(relative)
        self.mix_difficulty = bool(mix_difficulty)
        self.hard_reset_prob = float(hard_reset_prob)
        self.hard_reset_threshold = float(hard_reset_threshold)
        self.hard_reset_upright_min = float(hard_reset_upright_min)
        self.hard_reset_upright_max = float(hard_reset_upright_max)
        self._start_timestep = 0
        self._last_update = -1

    def _on_training_start(self):
        if self.relative:
            self._start_timestep = int(self.num_timesteps)
        self._set_difficulty(self._difficulty_at(self.num_timesteps))
        self._last_update = self.num_timesteps

    def _on_step(self):
        if self.num_timesteps - self._last_update < self.update_interval:
            return True
        self._set_difficulty(self._difficulty_at(self.num_timesteps))
        self._last_update = self.num_timesteps
        return True

    def _difficulty_at(self, num_timesteps):
        elapsed_timesteps = max(int(num_timesteps) - int(self._start_timestep), 0)
        progress = min(float(elapsed_timesteps) / float(self.curriculum_steps), 1.0)
        return self.start + progress * (self.end - self.start)

    def _set_difficulty(self, difficulty):
        if self.mix_difficulty:
            self.training_env.env_method("set_recovery_difficulty_range", self.start, float(difficulty))
        else:
            self.training_env.env_method("set_recovery_difficulty", float(difficulty))
        probability = self.hard_reset_prob if float(difficulty) >= self.hard_reset_threshold else 0.0
        self.training_env.env_method(
            "set_hard_reset_oversampling",
            probability,
            self.hard_reset_upright_min,
            self.hard_reset_upright_max,
        )


class RecoveryEvalCallback(BaseCallback):
    def __init__(
        self,
        model_path,
        save_path,
        difficulties,
        n_episodes=8,
        eval_freq=100_000,
        seed=100_000,
        max_episode_steps=800,
        verbose=0,
    ):
        super().__init__(verbose)
        self.model_path = model_path
        self.save_path = Path(save_path)
        self.difficulties = [float(value) for value in difficulties]
        self.n_episodes = int(n_episodes)
        self.eval_freq = int(eval_freq)
        self.seed = int(seed)
        self.max_episode_steps = int(max_episode_steps)
        self.best_score = -np.inf
        self._last_eval = -1

    def _on_training_start(self):
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self):
        if not self.difficulties or self.eval_freq <= 0:
            return True
        if self.num_timesteps - self._last_eval < self.eval_freq:
            return True

        metrics = self._evaluate()
        self._last_eval = self.num_timesteps
        recovered_rate = metrics["recovered"] / max(metrics["episodes"], 1)
        failure_rate = metrics["failed"] / max(metrics["episodes"], 1)
        hard_recovered_rate = metrics["hard_recovered"] / max(metrics["hard_episodes"], 1)
        score = 0.50 * recovered_rate + 0.50 * hard_recovered_rate - 0.10 * failure_rate
        print(
            "[eval] "
            f"steps={self.num_timesteps} "
            f"recovered={metrics['recovered']}/{metrics['episodes']} "
            f"hard_recovered={metrics['hard_recovered']}/{metrics['hard_episodes']} "
            f"failed={metrics['failed']}/{metrics['episodes']} "
            f"score={score:.3f} "
            f"best={self.best_score:.3f}",
            flush=True,
        )

        if score > self.best_score:
            self.best_score = score
            self.model.save(str(self.save_path / "best_model"))
            vec_normalize = self.model.get_vec_normalize_env()
            if vec_normalize is not None:
                vec_normalize.save(str(self.save_path / "best_vecnormalize.pkl"))
            (self.save_path / "best_metrics.txt").write_text(
                "\n".join(
                    [
                        f"steps={self.num_timesteps}",
                        f"score={score:.6f}",
                        f"recovered={metrics['recovered']}/{metrics['episodes']}",
                        f"hard_recovered={metrics['hard_recovered']}/{metrics['hard_episodes']}",
                        f"failed={metrics['failed']}/{metrics['episodes']}",
                        f"difficulties={','.join(str(value) for value in self.difficulties)}",
                    ]
                )
                + "\n"
            )
        return True

    def _evaluate(self):
        recovered = 0
        failed = 0
        episodes = 0
        hard_recovered = 0
        hard_episodes = 0
        for difficulty in self.difficulties:
            for episode in range(self.n_episodes):
                seed = self.seed + int(1000 * difficulty) + episode
                info = self._run_episode(difficulty, seed)
                episode_recovered = bool(info.get("recovered", False))
                recovered += int(episode_recovered)
                failed += int(bool(info.get("failure_timeout", False) or info.get("catastrophic", False)))
                if float(info.get("initial_upright", 1.0)) < 0.5:
                    hard_episodes += 1
                    hard_recovered += int(episode_recovered)
                episodes += 1
        return {
            "recovered": recovered,
            "failed": failed,
            "episodes": episodes,
            "hard_recovered": hard_recovered,
            "hard_episodes": hard_episodes,
        }

    def _run_episode(self, difficulty, seed):
        env = self._make_eval_env(difficulty, seed)
        obs = env.reset()
        info = {}
        try:
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, dones, infos = env.step(action)
                info = infos[0]
                if bool(dones[0]):
                    return info
        finally:
            env.close()

    def _make_eval_env(self, difficulty, seed):
        def _init():
            env = UnitreeA1Env(
                self.model_path,
                task="recovery",
                max_episode_steps=self.max_episode_steps,
                recovery_difficulty=difficulty,
                normalize_obs=False,
            )
            env.reset(seed=seed)
            return Monitor(env)

        env = DummyVecEnv([_init])
        train_vec_normalize = self.model.get_vec_normalize_env()
        if train_vec_normalize is None:
            return env

        eval_env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=train_vec_normalize.clip_obs)
        eval_env.obs_rms = copy.deepcopy(train_vec_normalize.obs_rms)
        eval_env.ret_rms = copy.deepcopy(train_vec_normalize.ret_rms)
        eval_env.training = False
        eval_env.norm_reward = False
        return eval_env


def make_env(model_path, seed, max_episode_steps, recovery_difficulty):
    def _init():
        env = UnitreeA1Env(
            model_path,
            task="recovery",
            max_episode_steps=max_episode_steps,
            recovery_difficulty=recovery_difficulty,
            normalize_obs=False,
        )
        env.reset(seed=seed)
        return Monitor(env)

    return _init


def parse_float_list(value):
    if not value:
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unitree_a1/scene.xml", help="Path to A1 scene XML")
    parser.add_argument("--total-steps", type=int, default=4_000_000, help="Total PPO training steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=800)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--recovery-difficulty-start", type=float, default=0.25)
    parser.add_argument("--recovery-difficulty-end", type=float, default=0.45)
    parser.add_argument("--curriculum-steps", type=int, default=4_000_000)
    parser.add_argument(
        "--curriculum-relative",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute curriculum progress relative to training start, useful for resumed curriculum stages.",
    )
    parser.add_argument(
        "--curriculum-mix-difficulty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample each recovery reset from [difficulty_start, current_curriculum_difficulty].",
    )
    parser.add_argument("--hard-reset-prob", type=float, default=0.0)
    parser.add_argument("--hard-reset-threshold", type=float, default=0.35)
    parser.add_argument("--hard-reset-upright-min", type=float, default=0.0)
    parser.add_argument("--hard-reset-upright-max", type=float, default=0.5)
    parser.add_argument("--vec-normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--start-method",
        choices=["forkserver", "spawn", "fork"],
        default="forkserver",
        help="Multiprocessing start method for SubprocVecEnv when n-envs > 1",
    )
    parser.add_argument("--run-dir", default="runs/recovery_fresh_curriculum_025_to_045_repro")
    parser.add_argument("--wandb-project", default="unitree-a1-recovery")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="disabled")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--wandb-log-interval", type=int, default=1000)
    parser.add_argument("--wandb-model-save-freq", type=int, default=100_000)
    parser.add_argument("--checkpoint-save-freq", type=int, default=100_000)
    parser.add_argument("--resume-from", default=None, help="Path to a saved PPO .zip checkpoint to continue training")
    parser.add_argument(
        "--vecnormalize-load",
        default=None,
        help="Path to a saved VecNormalize .pkl file to load before continuing training",
    )
    parser.add_argument(
        "--reset-num-timesteps",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reset SB3 timestep counter. Defaults to false when resuming and true for fresh training.",
    )
    parser.add_argument(
        "--eval-difficulties",
        default="0.35,0.45,0.5",
        help="Comma-separated recovery difficulties for periodic best-model evaluation, e.g. 0.45,0.50.",
    )
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--eval-freq", type=int, default=100_000)
    args = parser.parse_args()
    eval_difficulties = parse_float_list(args.eval_difficulties)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [
        make_env(
            args.model,
            args.seed + i,
            args.max_episode_steps,
            args.recovery_difficulty_start,
        )
        for i in range(args.n_envs)
    ]
    if args.n_envs > 1:
        env = SubprocVecEnv(env_fns, start_method=args.start_method)
        vec_env_type = "SubprocVecEnv"
    else:
        env = DummyVecEnv(env_fns)
        vec_env_type = "DummyVecEnv"
    if args.vec_normalize and args.vecnormalize_load:
        env = VecNormalize.load(args.vecnormalize_load, env)
        env.training = True
        env.norm_reward = False
    elif args.vec_normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    config = {
        "model_path": args.model,
        "total_steps": args.total_steps,
        "seed": args.seed,
        "max_episode_steps": args.max_episode_steps,
        "n_envs": args.n_envs,
        "vec_env": vec_env_type,
        "start_method": args.start_method if args.n_envs > 1 else None,
        "vec_normalize": args.vec_normalize,
        "recovery_difficulty_start": args.recovery_difficulty_start,
        "recovery_difficulty_end": args.recovery_difficulty_end,
        "curriculum_steps": args.curriculum_steps,
        "curriculum_relative": args.curriculum_relative,
        "curriculum_mix_difficulty": args.curriculum_mix_difficulty,
        "hard_reset_prob": args.hard_reset_prob,
        "hard_reset_threshold": args.hard_reset_threshold,
        "hard_reset_upright_min": args.hard_reset_upright_min,
        "hard_reset_upright_max": args.hard_reset_upright_max,
        "checkpoint_save_freq": args.checkpoint_save_freq,
        "resume_from": args.resume_from,
        "vecnormalize_load": args.vecnormalize_load,
        "reset_num_timesteps": args.reset_num_timesteps,
        "eval_difficulties": eval_difficulties,
        "eval_episodes": args.eval_episodes,
        "eval_freq": args.eval_freq,
        "algo": "PPO",
        "task": "recovery",
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
    }
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        mode=args.wandb_mode,
        config=config,
        monitor_gym=False,
        save_code=True,
    )
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    if args.resume_from:
        model = PPO.load(args.resume_from, env=env, seed=args.seed, verbose=1)
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
            learning_rate=args.learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.15,
            ent_coef=0.005,
        )
    reset_num_timesteps = args.reset_num_timesteps
    if reset_num_timesteps is None:
        reset_num_timesteps = args.resume_from is None

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_save_freq // args.n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_recovery_stand",
    )
    curriculum_callback = RecoveryCurriculumCallback(
        start=args.recovery_difficulty_start,
        end=args.recovery_difficulty_end,
        curriculum_steps=args.curriculum_steps,
        update_interval=args.wandb_log_interval,
        relative=args.curriculum_relative,
        mix_difficulty=args.curriculum_mix_difficulty,
        hard_reset_prob=args.hard_reset_prob,
        hard_reset_threshold=args.hard_reset_threshold,
        hard_reset_upright_min=args.hard_reset_upright_min,
        hard_reset_upright_max=args.hard_reset_upright_max,
    )
    wandb_callback = WandbMetricsCallback(
        run=run,
        log_interval=args.wandb_log_interval,
        model_save_path=str(run_dir / "wandb_models"),
        model_save_freq=args.wandb_model_save_freq,
        verbose=2,
    )
    callbacks = [curriculum_callback, checkpoint_callback, wandb_callback]
    if eval_difficulties:
        callbacks.append(
            RecoveryEvalCallback(
                model_path=args.model,
                save_path=run_dir / "best_eval",
                difficulties=eval_difficulties,
                n_episodes=args.eval_episodes,
                eval_freq=args.eval_freq,
                seed=args.seed + 50_000,
                max_episode_steps=args.max_episode_steps,
            )
        )

    try:
        model.learn(
            total_timesteps=args.total_steps,
            callback=CallbackList(callbacks),
            reset_num_timesteps=reset_num_timesteps,
        )
        model.save(str(run_dir / "ppo_recovery_stand_final"))
        if args.vec_normalize:
            env.save(str(run_dir / "vecnormalize.pkl"))
    finally:
        env.close()
        run.finish()


if __name__ == "__main__":
    main()
