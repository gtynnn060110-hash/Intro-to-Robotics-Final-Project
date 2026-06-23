"""Evaluate hard-bin recovery checkpoints.

Outputs:
- hard_bin_episodes.csv: accepted difficulty=0.45, initial_upright in [0, 0.5]
- hard_bin_summary.json: aggregate hard-bin result
- difficulty_sweep_episodes.csv: ordinary recovery resets across difficulties
- difficulty_sweep_summary.csv: recovered counts by difficulty and initial_upright bin
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from train_hard_bin import make_base_eval_env, run_eval_episode


UPRIGHT_BINS = [
    (-1.0, 0.0, "[-1.0,0.0)"),
    (0.0, 0.25, "[0.0,0.25)"),
    (0.25, 0.5, "[0.25,0.5)"),
    (0.5, 0.75, "[0.5,0.75)"),
    (0.75, 1.01, "[0.75,1.0]"),
]


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def bin_name(initial_upright):
    value = float(initial_upright)
    for low, high, name in UPRIGHT_BINS:
        if low <= value < high:
            return name
    return "out_of_range"


def load_vecnormalize_source(args):
    if not args.vecnormalize:
        return None
    dummy = make_base_eval_env(
        model_path=args.env_model,
        action_scale=args.action_scale,
        max_episode_steps=args.max_episode_steps,
        success_steps=args.success_steps,
        failure_steps=args.failure_steps,
        difficulty=0.45,
        hard_reset_prob=0.0,
        hard_upright_min=args.hard_upright_min,
        hard_upright_max=args.hard_upright_max,
    )
    vec = VecNormalize.load(args.vecnormalize, dummy)
    vec.training = False
    vec.norm_reward = False
    return vec


def episode_row(info):
    episode = info.get("episode", {})
    return {
        "seed": int(info.get("seed", -1)),
        "difficulty": float(info.get("difficulty", 0.0)),
        "initial_upright": float(info.get("initial_upright", np.nan)),
        "bin": bin_name(info.get("initial_upright", np.nan)),
        "recovered": int(bool(info.get("recovered", False))),
        "stable": int(bool(info.get("stable", False))),
        "failed": int(bool(info.get("failure_timeout", False) or info.get("catastrophic", False))),
        "fallen": int(bool(info.get("fallen", False))),
        "final_upright": float(info.get("upright", np.nan)),
        "final_height_error": float(info.get("height_error", np.nan)),
        "length": int(episode.get("l", 0)),
        "return": float(episode.get("r", 0.0)),
    }


def summarize_rows(rows):
    if not rows:
        return {
            "episodes": 0,
            "recovered": 0,
            "stable": 0,
            "failed": 0,
            "fallen": 0,
            "mean_return": 0.0,
            "mean_len": 0.0,
            "mean_initial_upright": 0.0,
            "mean_final_upright": 0.0,
        }
    return {
        "episodes": len(rows),
        "recovered": int(sum(row["recovered"] for row in rows)),
        "stable": int(sum(row["stable"] for row in rows)),
        "failed": int(sum(row["failed"] for row in rows)),
        "fallen": int(sum(row["fallen"] for row in rows)),
        "mean_return": float(np.mean([row["return"] for row in rows])),
        "mean_len": float(np.mean([row["length"] for row in rows])),
        "mean_initial_upright": float(np.mean([row["initial_upright"] for row in rows])),
        "mean_final_upright": float(np.mean([row["final_upright"] for row in rows])),
    }


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_hard_bin(model, vec_source, args):
    infos = []
    attempts = 0
    max_attempts = args.hard_episodes * args.max_hard_attempts_per_episode
    while len(infos) < args.hard_episodes and attempts < max_attempts:
        seed = args.seed + attempts
        info = run_eval_episode(
            model=model,
            train_vec_normalize=vec_source,
            model_path=args.env_model,
            seed=seed,
            difficulty=0.45,
            hard_reset_prob=1.0,
            hard_upright_min=args.hard_upright_min,
            hard_upright_max=args.hard_upright_max,
            action_scale=args.action_scale,
            max_episode_steps=args.max_episode_steps,
            success_steps=args.success_steps,
            failure_steps=args.failure_steps,
        )
        initial_upright = float(info.get("initial_upright", 99.0))
        if args.hard_upright_min <= initial_upright <= args.hard_upright_max:
            infos.append(info)
        attempts += 1
    return [episode_row(info) for info in infos], attempts


def collect_difficulty_sweep(model, vec_source, args):
    rows = []
    for difficulty in args.sweep_difficulties:
        for episode in range(args.sweep_episodes):
            seed = args.seed + 100_000 + int(round(1000 * difficulty)) + episode
            info = run_eval_episode(
                model=model,
                train_vec_normalize=vec_source,
                model_path=args.env_model,
                seed=seed,
                difficulty=difficulty,
                hard_reset_prob=0.0,
                hard_upright_min=args.hard_upright_min,
                hard_upright_max=args.hard_upright_max,
                action_scale=args.action_scale,
                max_episode_steps=args.max_episode_steps,
                success_steps=args.success_steps,
                failure_steps=args.failure_steps,
            )
            rows.append(episode_row(info))
    return rows


def sweep_summary_rows(rows):
    summary_rows = []
    difficulties = sorted({row["difficulty"] for row in rows})
    for difficulty in difficulties:
        difficulty_rows = [row for row in rows if row["difficulty"] == difficulty]
        aggregate = summarize_rows(difficulty_rows)
        summary_rows.append({"difficulty": difficulty, "bin": "all", **aggregate})
        for _, _, name in UPRIGHT_BINS:
            bin_rows = [row for row in difficulty_rows if row["bin"] == name]
            if bin_rows:
                summary_rows.append({"difficulty": difficulty, "bin": name, **summarize_rows(bin_rows)})
    return summary_rows


def print_summary(hard_summary, hard_attempts, sweep_summary):
    print(
        "[hard-bin] "
        f"recovered={hard_summary['recovered']}/{hard_summary['episodes']} "
        f"stable={hard_summary['stable']}/{hard_summary['episodes']} "
        f"failed={hard_summary['failed']}/{hard_summary['episodes']} "
        f"attempts={hard_attempts} "
        f"mean_init={hard_summary['mean_initial_upright']:.3f} "
        f"mean_final={hard_summary['mean_final_upright']:.3f}",
        flush=True,
    )
    for row in sweep_summary:
        if row["bin"] == "all":
            print(
                "[sweep] "
                f"difficulty={row['difficulty']:.2f} "
                f"recovered={row['recovered']}/{row['episodes']} "
                f"failed={row['failed']}/{row['episodes']} "
                f"mean_init={row['mean_initial_upright']:.3f}",
                flush=True,
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="PPO checkpoint .zip")
    parser.add_argument("--vecnormalize", default=None, help="VecNormalize .pkl matching the checkpoint")
    parser.add_argument("--env-model", default="unitree_a1/scene.xml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=123_000)
    parser.add_argument("--hard-episodes", type=int, default=100)
    parser.add_argument("--max-hard-attempts-per-episode", type=int, default=10)
    parser.add_argument("--sweep-episodes", type=int, default=30)
    parser.add_argument(
        "--sweep-difficulties",
        type=parse_float_list,
        default=[0.25, 0.35, 0.45, 0.50, 0.60],
    )
    parser.add_argument("--action-scale", type=float, default=0.9)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--success-steps", type=int, default=15)
    parser.add_argument("--failure-steps", type=int, default=180)
    parser.add_argument("--hard-upright-min", type=float, default=0.0)
    parser.add_argument("--hard-upright-max", type=float, default=0.5)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = str(Path(args.model).resolve().parent / "hard_bin_eval")
    return args


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = PPO.load(args.model)
    vec_source = load_vecnormalize_source(args)

    hard_rows, hard_attempts = collect_hard_bin(model, vec_source, args)
    hard_summary = summarize_rows(hard_rows)
    hard_summary["attempts"] = hard_attempts
    hard_summary["difficulty"] = 0.45
    hard_summary["initial_upright_bin"] = f"[{args.hard_upright_min},{args.hard_upright_max}]"

    sweep_rows = collect_difficulty_sweep(model, vec_source, args)
    summary_rows = sweep_summary_rows(sweep_rows)

    write_csv(
        output_dir / "hard_bin_episodes.csv",
        hard_rows,
        [
            "seed",
            "difficulty",
            "initial_upright",
            "bin",
            "recovered",
            "stable",
            "failed",
            "fallen",
            "final_upright",
            "final_height_error",
            "length",
            "return",
        ],
    )
    with (output_dir / "hard_bin_summary.json").open("w") as f:
        json.dump(hard_summary, f, indent=2)
    write_csv(
        output_dir / "difficulty_sweep_episodes.csv",
        sweep_rows,
        [
            "seed",
            "difficulty",
            "initial_upright",
            "bin",
            "recovered",
            "stable",
            "failed",
            "fallen",
            "final_upright",
            "final_height_error",
            "length",
            "return",
        ],
    )
    write_csv(
        output_dir / "difficulty_sweep_summary.csv",
        summary_rows,
        [
            "difficulty",
            "bin",
            "episodes",
            "recovered",
            "stable",
            "failed",
            "fallen",
            "mean_return",
            "mean_len",
            "mean_initial_upright",
            "mean_final_upright",
        ],
    )
    with (output_dir / "eval_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    print_summary(hard_summary, hard_attempts, summary_rows)
    if vec_source is not None:
        vec_source.close()


if __name__ == "__main__":
    main()
