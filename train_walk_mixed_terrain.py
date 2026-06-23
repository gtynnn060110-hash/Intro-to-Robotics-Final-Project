"""Fine-tune walking with mixed terrain replay to reduce forgetting."""
import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from envs import UnitreeA1WalkEnv
from train_walk import INFO_KEYWORDS, WalkMetricsCallback


@dataclass(frozen=True)
class TerrainScenario:
    weight: float
    friction_min: float
    friction_max: float
    height_min: float
    height_max: float
    reset_noise_min: float
    reset_noise_max: float


class RandomizedTerrainWrapper(gym.Wrapper):
    def __init__(self, env, scenarios, seed):
        super().__init__(env)
        total_weight = sum(max(scenario.weight, 0.0) for scenario in scenarios)
        if total_weight <= 0.0:
            raise ValueError("At least one terrain scenario must have positive weight.")
        self.scenarios = scenarios
        self.weights = np.array([max(scenario.weight, 0.0) / total_weight for scenario in scenarios])
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        scenario_index = int(self.rng.choice(len(self.scenarios), p=self.weights))
        scenario = self.scenarios[scenario_index]
        terrain_friction = self.rng.uniform(scenario.friction_min, scenario.friction_max)
        terrain_height = self.rng.uniform(scenario.height_min, scenario.height_max)
        reset_noise = self.rng.uniform(scenario.reset_noise_min, scenario.reset_noise_max)

        self.env.set_terrain_friction(float(terrain_friction))
        self.env.set_terrain_height_scale(float(terrain_height))
        self.env.set_reset_noise(float(reset_noise))
        obs, info = self.env.reset(seed=seed, options=options)
        info["scenario_id"] = scenario_index
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info


def make_env(
    model_path,
    seed,
    max_episode_steps,
    target_vx,
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
    scenarios,
):
    def _init():
        env = UnitreeA1WalkEnv(
            model_path=model_path,
            target_vx=target_vx,
            reset_noise=0.0,
            terrain_friction=1.5,
            terrain_height_scale=0.3,
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
            normalize_obs=False,
        )
        env = RandomizedTerrainWrapper(env, scenarios=scenarios, seed=seed)
        env.reset(seed=seed)
        return Monitor(env, info_keywords=INFO_KEYWORDS)

    return _init


def build_scenarios(args):
    return [
        TerrainScenario(
            weight=args.general_weight,
            friction_min=args.general_friction_min,
            friction_max=args.general_friction_max,
            height_min=args.general_height_min,
            height_max=args.general_height_max,
            reset_noise_min=args.general_noise_min,
            reset_noise_max=args.general_noise_max,
        ),
        TerrainScenario(
            weight=args.mid_weight,
            friction_min=args.mid_friction_min,
            friction_max=args.mid_friction_max,
            height_min=args.mid_height_min,
            height_max=args.mid_height_max,
            reset_noise_min=args.mid_noise_min,
            reset_noise_max=args.mid_noise_max,
        ),
        TerrainScenario(
            weight=args.hard_weight,
            friction_min=args.hard_friction_min,
            friction_max=args.hard_friction_max,
            height_min=args.hard_height_min,
            height_max=args.hard_height_max,
            reset_noise_min=args.hard_noise_min,
            reset_noise_max=args.hard_noise_max,
        ),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unitree_a1/scene.xml")
    parser.add_argument("--resume-from", required=True)
    parser.add_argument("--vecnormalize-load", required=True)
    parser.add_argument("--run-dir", default="runs/walk_mixed_terrain_v1")
    parser.add_argument("--total-steps", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=8e-6)
    parser.add_argument("--target-vx", type=float, default=0.2)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--action-scale", type=float, default=0.5)
    parser.add_argument("--overspeed-deadband", type=float, default=0.02)
    parser.add_argument("--overspeed-weight", type=float, default=8.0)
    parser.add_argument("--overspeed-quadratic-weight", type=float, default=20.0)
    parser.add_argument("--forward-reward-weight", type=float, default=3.0)
    parser.add_argument("--progress-reward-weight", type=float, default=5.0)
    parser.add_argument("--backward-penalty-weight", type=float, default=6.0)
    parser.add_argument("--low-speed-penalty-weight", type=float, default=40.0)
    parser.add_argument("--low-speed-fraction", type=float, default=1.0)
    parser.add_argument("--speed-reward-sharpness", type=float, default=20.0)

    parser.add_argument("--general-weight", type=float, default=0.25)
    parser.add_argument("--general-friction-min", type=float, default=0.50)
    parser.add_argument("--general-friction-max", type=float, default=1.50)
    parser.add_argument("--general-height-min", type=float, default=0.30)
    parser.add_argument("--general-height-max", type=float, default=0.40)
    parser.add_argument("--general-noise-min", type=float, default=0.00)
    parser.add_argument("--general-noise-max", type=float, default=0.15)

    parser.add_argument("--mid-weight", type=float, default=0.35)
    parser.add_argument("--mid-friction-min", type=float, default=0.20)
    parser.add_argument("--mid-friction-max", type=float, default=0.80)
    parser.add_argument("--mid-height-min", type=float, default=0.40)
    parser.add_argument("--mid-height-max", type=float, default=0.48)
    parser.add_argument("--mid-noise-min", type=float, default=0.00)
    parser.add_argument("--mid-noise-max", type=float, default=0.15)

    parser.add_argument("--hard-weight", type=float, default=0.40)
    parser.add_argument("--hard-friction-min", type=float, default=0.10)
    parser.add_argument("--hard-friction-max", type=float, default=0.50)
    parser.add_argument("--hard-height-min", type=float, default=0.50)
    parser.add_argument("--hard-height-max", type=float, default=0.50)
    parser.add_argument("--hard-noise-min", type=float, default=0.02)
    parser.add_argument("--hard-noise-max", type=float, default=0.15)

    parser.add_argument("--start-method", choices=["forkserver", "spawn", "fork"], default="fork")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=20_000)
    parser.add_argument("--checkpoint-save-freq", type=int, default=50_000)
    parser.add_argument("--reset-num-timesteps", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    torch.set_num_threads(max(int(args.torch_threads), 1))
    torch.set_num_interop_threads(1)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    scenarios = build_scenarios(args)

    env_fns = [
        make_env(
            model_path=args.model,
            seed=args.seed + i,
            max_episode_steps=args.max_episode_steps,
            target_vx=args.target_vx,
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
            scenarios=scenarios,
        )
        for i in range(args.n_envs)
    ]
    if args.n_envs > 1:
        env = SubprocVecEnv(env_fns, start_method=args.start_method)
    else:
        env = DummyVecEnv(env_fns)

    env = VecNormalize.load(args.vecnormalize_load, env)
    env.training = True
    env.norm_reward = False

    model = PPO.load(args.resume_from, env=env, seed=args.seed, verbose=1)
    model.learning_rate = args.learning_rate
    model.lr_schedule = get_schedule_fn(args.learning_rate)

    callbacks = CallbackList(
        [
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
