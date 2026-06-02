"""Train Unitree A1 recovery standing with PPO.

Run with `mjpython train.py` on macOS.
"""
import argparse
from pathlib import Path

import numpy as np
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

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
                "reward_upright",
                "reward_height",
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

        return True


def make_env(model_path, seed, max_episode_steps):
    def _init():
        env = UnitreeA1Env(
            model_path,
            task="recovery",
            max_episode_steps=max_episode_steps,
        )
        env.reset(seed=seed)
        return Monitor(env)

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unitree_a1/scene.xml", help="Path to A1 scene XML")
    parser.add_argument("--total-steps", type=int, default=200_000, help="Total PPO training steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=800)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--run-dir", default="runs/recovery_stand")
    parser.add_argument("--wandb-project", default="unitree-a1-recovery")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--wandb-log-interval", type=int, default=1000)
    parser.add_argument("--wandb-model-save-freq", type=int, default=50_000)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv(
        [
            make_env(args.model, args.seed + i, args.max_episode_steps)
            for i in range(args.n_envs)
        ]
    )

    config = {
        "model_path": args.model,
        "total_steps": args.total_steps,
        "seed": args.seed,
        "max_episode_steps": args.max_episode_steps,
        "n_envs": args.n_envs,
        "algo": "PPO",
        "task": "recovery",
        "n_steps": 1024,
        "batch_size": 256,
        "learning_rate": 3e-4,
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
    policy_kwargs = dict(
    net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
        n_steps=2048,
        batch_size=512,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.005,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(10_000 // args.n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_recovery_stand",
    )
    wandb_callback = WandbMetricsCallback(
        run=run,
        log_interval=args.wandb_log_interval,
        model_save_path=str(run_dir / "wandb_models"),
        model_save_freq=max(args.wandb_model_save_freq // args.n_envs, 1),
        verbose=2,
    )

    try:
        model.learn(
            total_timesteps=args.total_steps,
            callback=CallbackList([checkpoint_callback, wandb_callback]),
        )
        model.save(str(run_dir / "ppo_recovery_stand_final"))
    finally:
        env.close()
        run.finish()


if __name__ == "__main__":
    main()
