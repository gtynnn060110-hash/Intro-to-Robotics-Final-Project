"""Composite evaluation for replay anti-forgetting recovery policies."""

import argparse
import csv
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from train_hard_bin import make_base_eval_env, run_eval_episode
from train_replay_antiforgetting import parse_float_list


UPRIGHT_BINS = [
    (-1.0, 0.0, "[-1.0,0.0)"),
    (0.0, 0.25, "[0.0,0.25)"),
    (0.25, 0.5, "[0.25,0.5)"),
    (0.5, 0.75, "[0.5,0.75)"),
    (0.75, 1.01, "[0.75,1.0]"),
]


def bin_name(value):
    value = float(value)
    for low, high, name in UPRIGHT_BINS:
        if low <= value < high:
            return name
    return "out_of_range"


def load_vecnormalize(args):
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
        "initial_upright": float(info.get("initial_upright", 0.0)),
        "bin": bin_name(info.get("initial_upright", 0.0)),
        "recovered": int(bool(info.get("recovered", False))),
        "stable": int(bool(info.get("stable", False))),
        "failed": int(bool(info.get("failure_timeout", False) or info.get("catastrophic", False))),
        "fallen": int(bool(info.get("fallen", False))),
        "final_upright": float(info.get("upright", 0.0)),
        "final_height_error": float(info.get("height_error", 0.0)),
        "length": int(episode.get("l", 0)),
        "return": float(episode.get("r", 0.0)),
    }


def summarize(rows):
    if not rows:
        return {
            "episodes": 0,
            "recovered": 0,
            "stable": 0,
            "failed": 0,
            "fallen": 0,
            "mean_initial_upright": 0.0,
            "mean_final_upright": 0.0,
            "mean_return": 0.0,
        }
    return {
        "episodes": len(rows),
        "recovered": int(sum(row["recovered"] for row in rows)),
        "stable": int(sum(row["stable"] for row in rows)),
        "failed": int(sum(row["failed"] for row in rows)),
        "fallen": int(sum(row["fallen"] for row in rows)),
        "mean_initial_upright": sum(row["initial_upright"] for row in rows) / len(rows),
        "mean_final_upright": sum(row["final_upright"] for row in rows) / len(rows),
        "mean_return": sum(row["return"] for row in rows) / len(rows),
    }


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_sweep(model, vec_source, args):
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


def collect_hard_bin(model, vec_source, args):
    rows = []
    attempts = 0
    max_attempts = args.hard_episodes * args.max_hard_attempts_per_episode
    while len(rows) < args.hard_episodes and attempts < max_attempts:
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
            rows.append(episode_row(info))
        attempts += 1
    return rows, attempts


def sweep_summary(rows):
    output = []
    for difficulty in sorted({row["difficulty"] for row in rows}):
        difficulty_rows = [row for row in rows if row["difficulty"] == difficulty]
        output.append({"difficulty": difficulty, "bin": "all", **summarize(difficulty_rows)})
        for _, _, name in UPRIGHT_BINS:
            bin_rows = [row for row in difficulty_rows if row["bin"] == name]
            if bin_rows:
                output.append({"difficulty": difficulty, "bin": name, **summarize(bin_rows)})
    return output


def hard_bin_summary(rows, attempts):
    summary = summarize(rows)
    summary["attempts"] = attempts
    summary["difficulty"] = 0.45
    summary["initial_upright_bin"] = "hard"
    summary["bins"] = {}
    for _, _, name in UPRIGHT_BINS:
        if name in ("[0.0,0.25)", "[0.25,0.5)"):
            bin_rows = [row for row in rows if row["bin"] == name]
            summary["bins"][name] = summarize(bin_rows)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--vecnormalize", default=None)
    parser.add_argument("--env-model", default="unitree_a1/scene.xml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=81000)
    parser.add_argument("--action-scale", type=float, default=0.5)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--success-steps", type=int, default=15)
    parser.add_argument("--failure-steps", type=int, default=180)
    parser.add_argument("--hard-episodes", type=int, default=100)
    parser.add_argument("--max-hard-attempts-per-episode", type=int, default=10)
    parser.add_argument("--sweep-episodes", type=int, default=30)
    parser.add_argument("--sweep-difficulties", type=parse_float_list, default=[0.25, 0.35, 0.45, 0.50, 0.60])
    parser.add_argument("--hard-upright-min", type=float, default=0.0)
    parser.add_argument("--hard-upright-max", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = PPO.load(args.model)
    vec_source = load_vecnormalize(args)
    hard_rows, hard_attempts = collect_hard_bin(model, vec_source, args)
    sweep_rows = collect_sweep(model, vec_source, args)
    hard_summary = hard_bin_summary(hard_rows, hard_attempts)
    sweep_rows_summary = sweep_summary(sweep_rows)

    sweep_recovered = sum(row["recovered"] for row in sweep_rows)
    sweep_total = len(sweep_rows)
    hard_recovered = hard_summary["recovered"]
    hard_total = hard_summary["episodes"]
    score = 0.7 * (sweep_recovered / max(sweep_total, 1)) + 0.3 * (hard_recovered / max(hard_total, 1))

    fields = [
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
    ]
    write_csv(output_dir / "hard_bin_episodes.csv", hard_rows, fields)
    write_csv(output_dir / "difficulty_sweep_episodes.csv", sweep_rows, fields)
    write_csv(
        output_dir / "difficulty_sweep_summary.csv",
        sweep_rows_summary,
        [
            "difficulty",
            "bin",
            "episodes",
            "recovered",
            "stable",
            "failed",
            "fallen",
            "mean_initial_upright",
            "mean_final_upright",
            "mean_return",
        ],
    )
    summary = {
        "model": args.model,
        "vecnormalize": args.vecnormalize,
        "action_scale": args.action_scale,
        "seed": args.seed,
        "sweep_recovered": sweep_recovered,
        "sweep_episodes": sweep_total,
        "sweep_success_rate": sweep_recovered / max(sweep_total, 1),
        "hard_recovered": hard_recovered,
        "hard_episodes": hard_total,
        "hard_success_rate": hard_recovered / max(hard_total, 1),
        "score": score,
        "hard_summary": hard_summary,
        "sweep_summary": sweep_rows_summary,
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with (output_dir / "eval_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    print(
        "[composite] "
        f"sweep={sweep_recovered}/{sweep_total} "
        f"hard={hard_recovered}/{hard_total} "
        f"score={score:.3f}",
        flush=True,
    )
    for row in sweep_rows_summary:
        if row["bin"] == "all":
            print(f"[sweep] {row['difficulty']:.2f}: {row['recovered']}/{row['episodes']}", flush=True)
    for name, item in hard_summary["bins"].items():
        print(f"[hard-bin] {name}: {item['recovered']}/{item['episodes']}", flush=True)
    if vec_source is not None:
        vec_source.close()


if __name__ == "__main__":
    main()
