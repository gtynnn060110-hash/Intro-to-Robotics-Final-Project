"""Fixed-seed recovery benchmark.

This evaluates ordinary recovery difficulty sweeps only. It intentionally does
not run hard-bin filtering; every model is tested on the same seed bank.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs import UnitreeA1Env


DEFAULT_DIFFICULTIES = [0.25, 0.35, 0.45, 0.50, 0.60]


def parse_float_list(value):
    if isinstance(value, list):
        return [float(item) for item in value]
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def make_seed_bank(path, difficulties, episodes_per_difficulty, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for difficulty in difficulties:
        seeds = rng.choice(np.arange(1, 2_000_000_000, dtype=np.int64), size=episodes_per_difficulty, replace=False)
        for episode_idx, episode_seed in enumerate(seeds):
            rows.append(
                {
                    "difficulty": f"{difficulty:.2f}",
                    "episode_idx": int(episode_idx),
                    "seed": int(episode_seed),
                }
            )
    write_csv(path, rows, ["difficulty", "episode_idx", "seed"])
    return rows


def load_seed_bank(path):
    rows = []
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "difficulty": f"{float(row['difficulty']):.2f}",
                    "episode_idx": int(row["episode_idx"]),
                    "seed": int(row["seed"]),
                }
            )
    return rows


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_env(model_path, difficulty, action_scale, max_episode_steps, success_steps, failure_steps):
    def _init():
        env = UnitreeA1Env(
            model_path,
            task="recovery",
            max_episode_steps=max_episode_steps,
            recovery_difficulty=float(difficulty),
            normalize_obs=False,
            action_scale=action_scale,
            success_steps=success_steps,
            failure_steps=failure_steps,
        )
        env.set_hard_reset_oversampling(0.0)
        return Monitor(env)

    return DummyVecEnv([_init])


def load_eval_env(args, difficulty):
    env = make_env(
        args.env_model,
        difficulty,
        args.action_scale,
        args.max_episode_steps,
        args.success_steps,
        args.failure_steps,
    )
    if args.vecnormalize:
        env = VecNormalize.load(args.vecnormalize, env)
        env.training = False
        env.norm_reward = False
    return env


def reset_info(vec_env):
    base = vec_env.venv if isinstance(vec_env, VecNormalize) else vec_env
    infos = getattr(base, "reset_infos", None)
    if infos:
        return dict(infos[0])
    return {}


def eval_episode(model, env, seed, difficulty):
    env.seed(int(seed))
    obs = env.reset()
    initial_info = reset_info(env)
    final_info = {}
    total_reward = 0.0
    length = 0
    while True:
        if model is None:
            action = np.zeros((env.num_envs, *env.action_space.shape), dtype=np.float32)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        total_reward += float(rewards[0])
        length += 1
        final_info = dict(infos[0])
        if bool(dones[0]):
            break

    episode_info = final_info.get("episode", {})
    return {
        "seed": int(seed),
        "difficulty": f"{float(difficulty):.2f}",
        "initial_upright": float(final_info.get("initial_upright", initial_info.get("initial_upright", np.nan))),
        "initial_height_error": float(
            final_info.get("initial_height_error", initial_info.get("initial_height_error", np.nan))
        ),
        "recovered": int(bool(final_info.get("recovered", False))),
        "stable": int(bool(final_info.get("stable", False))),
        "failed": int(
            bool(
                final_info.get("failure_timeout", False)
                or final_info.get("catastrophic", False)
                or final_info.get("episode_timeout", False)
            )
        ),
        "fallen": int(bool(final_info.get("fallen", False))),
        "catastrophic": int(bool(final_info.get("catastrophic", False))),
        "episode_timeout": int(bool(final_info.get("episode_timeout", False))),
        "final_upright": float(final_info.get("upright", np.nan)),
        "final_height_error": float(final_info.get("height_error", np.nan)),
        "return": float(episode_info.get("r", total_reward)),
        "length": int(episode_info.get("l", length)),
    }


def summarize(rows):
    if not rows:
        return {
            "episodes": 0,
            "recovered": 0,
            "stable": 0,
            "failed": 0,
            "fallen": 0,
            "episode_timeout": 0,
            "recovery_rate": 0.0,
            "stable_rate": 0.0,
            "failed_rate": 0.0,
            "fallen_rate": 0.0,
            "episode_timeout_rate": 0.0,
            "mean_initial_upright": 0.0,
            "mean_final_upright": 0.0,
            "mean_return": 0.0,
            "mean_len": 0.0,
        }
    return {
        "episodes": len(rows),
        "recovered": int(sum(row["recovered"] for row in rows)),
        "stable": int(sum(row["stable"] for row in rows)),
        "failed": int(sum(row["failed"] for row in rows)),
        "fallen": int(sum(row["fallen"] for row in rows)),
        "episode_timeout": int(sum(row["episode_timeout"] for row in rows)),
        "recovery_rate": float(np.mean([row["recovered"] for row in rows])),
        "stable_rate": float(np.mean([row["stable"] for row in rows])),
        "failed_rate": float(np.mean([row["failed"] for row in rows])),
        "fallen_rate": float(np.mean([row["fallen"] for row in rows])),
        "episode_timeout_rate": float(np.mean([row["episode_timeout"] for row in rows])),
        "mean_initial_upright": float(np.mean([row["initial_upright"] for row in rows])),
        "mean_final_upright": float(np.mean([row["final_upright"] for row in rows])),
        "mean_return": float(np.mean([row["return"] for row in rows])),
        "mean_len": float(np.mean([row["length"] for row in rows])),
    }


def write_summaries(output_dir, episode_rows):
    per_difficulty = []
    model_names = sorted({row["model"] for row in episode_rows})
    difficulties = sorted({row["difficulty"] for row in episode_rows}, key=float)
    for model_name in model_names:
        model_rows = [row for row in episode_rows if row["model"] == model_name]
        for difficulty in difficulties:
            rows = [row for row in model_rows if row["difficulty"] == difficulty]
            per_difficulty.append({"model": model_name, "difficulty": difficulty, **summarize(rows)})

    aggregate = []
    for model_name in model_names:
        model_rows = [row for row in episode_rows if row["model"] == model_name]
        aggregate.append({"model": model_name, "difficulty": "all", **summarize(model_rows)})

    fields = [
        "model",
        "difficulty",
        "episodes",
        "recovered",
        "stable",
        "failed",
        "fallen",
        "episode_timeout",
        "recovery_rate",
        "stable_rate",
        "failed_rate",
        "fallen_rate",
        "episode_timeout_rate",
        "mean_initial_upright",
        "mean_final_upright",
        "mean_return",
        "mean_len",
    ]
    write_csv(output_dir / "summary_by_difficulty.csv", per_difficulty, fields)
    write_csv(output_dir / "summary_aggregate.csv", aggregate, fields)
    with (output_dir / "summary.json").open("w") as f:
        json.dump({"by_difficulty": per_difficulty, "aggregate": aggregate}, f, indent=2, sort_keys=True)
    return per_difficulty, aggregate


def parse_models(value):
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/recovery_seed_benchmark_v1")
    parser.add_argument("--seed-file", default=None)
    parser.add_argument("--generate-seeds", action="store_true")
    parser.add_argument("--seed-bank-seed", type=int, default=20260604)
    parser.add_argument("--episodes-per-difficulty", type=int, default=100)
    parser.add_argument("--difficulties", type=parse_float_list, default=DEFAULT_DIFFICULTIES)
    parser.add_argument("--models", required=True, help="JSON string/file: name, model, vecnormalize")
    parser.add_argument("--env-model", default="unitree_a1/scene.xml")
    parser.add_argument("--action-scale", type=float, default=0.5)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--success-steps", type=int, default=15)
    parser.add_argument("--failure-steps", type=int, default=180)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_file = Path(args.seed_file) if args.seed_file else output_dir / "seeds.csv"

    if args.generate_seeds or not seed_file.exists():
        seed_rows = make_seed_bank(seed_file, args.difficulties, args.episodes_per_difficulty, args.seed_bank_seed)
    else:
        seed_rows = load_seed_bank(seed_file)

    models = parse_models(args.models)
    with (output_dir / "benchmark_config.json").open("w") as f:
        json.dump({**vars(args), "seed_file": str(seed_file), "models": models}, f, indent=2, sort_keys=True)

    episode_rows = []
    episode_fields = [
        "model",
        "seed",
        "difficulty",
        "initial_upright",
        "initial_height_error",
        "recovered",
        "stable",
        "failed",
        "fallen",
        "catastrophic",
        "episode_timeout",
        "final_upright",
        "final_height_error",
        "return",
        "length",
    ]
    for spec in models:
        model_name = spec["name"]
        print(f"[model] {model_name}", flush=True)
        policy_type = spec.get("type", "ppo")
        if policy_type == "zero_action":
            model = None
        elif policy_type == "ppo":
            model = PPO.load(spec["model"])
        else:
            raise ValueError(f"Unsupported model type for {model_name}: {policy_type}")
        for difficulty in sorted({row["difficulty"] for row in seed_rows}, key=float):
            env_args = argparse.Namespace(**vars(args))
            env_args.vecnormalize = spec.get("vecnormalize") if policy_type == "ppo" else None
            env = load_eval_env(env_args, float(difficulty))
            difficulty_rows = [row for row in seed_rows if row["difficulty"] == difficulty]
            recovered = 0
            try:
                for item in difficulty_rows:
                    row = eval_episode(model, env, int(item["seed"]), float(difficulty))
                    row["model"] = model_name
                    episode_rows.append(row)
                    recovered += int(row["recovered"])
            finally:
                env.close()
            print(
                f"[difficulty] model={model_name} difficulty={float(difficulty):.2f} "
                f"recovered={recovered}/{len(difficulty_rows)}",
                flush=True,
            )

    write_csv(output_dir / "episodes.csv", episode_rows, episode_fields)
    per_difficulty, aggregate = write_summaries(output_dir, episode_rows)

    print("[aggregate]", flush=True)
    for row in aggregate:
        print(
            f"{row['model']}: recovered={row['recovered']}/{row['episodes']} "
            f"rate={row['recovery_rate']:.3f} mean_return={row['mean_return']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
