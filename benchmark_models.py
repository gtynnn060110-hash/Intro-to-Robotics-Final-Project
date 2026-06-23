"""Fixed-seed benchmark for recovery policies.

The benchmark uses the same seed bases for every model.  For each seed base it
runs:

- difficulty sweep: 0.25, 0.35, 0.45, 0.50, 0.60 with 30 episodes each
- hard-bin: difficulty 0.45, initial_upright in [0.0, 0.5] with 100 episodes

Composite score:
    0.7 * sweep_success_rate + 0.3 * hard_bin_success_rate
"""

import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from stable_baselines3 import PPO

from eval_replay_antiforgetting import collect_hard_bin, collect_sweep, load_vecnormalize, parse_float_list


DEFAULT_MODELS = [
    {
        "name": "prev_fresh_final",
        "model": "runs/recovery_fresh_curriculum_025_to_045_reward_v2/ppo_recovery_stand_final.zip",
        "vecnormalize": "runs/recovery_fresh_curriculum_025_to_045_reward_v2/vecnormalize.pkl",
        "action_scale": 0.5,
    },
    {
        "name": "current_hard_bin_best",
        "model": "runs/hard_bin_stage_b_full_v2/best_eval/best_model.zip",
        "vecnormalize": "runs/hard_bin_stage_b_full_v2/best_eval/best_vecnormalize.pkl",
        "action_scale": 0.8,
    },
    {
        "name": "replay_antiforgetting_fused_best",
        "model": "runs/replay_antiforgetting_v1/best_eval/best_model.zip",
        "vecnormalize": "runs/replay_antiforgetting_v1/best_eval/best_vecnormalize.pkl",
        "action_scale": 0.65,
    },
]


def parse_model_specs(value):
    if not value:
        return DEFAULT_MODELS
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(value)


def make_eval_args(args, model_spec, seed):
    return SimpleNamespace(
        model=model_spec["model"],
        vecnormalize=model_spec.get("vecnormalize"),
        env_model=args.env_model,
        output_dir=None,
        seed=int(seed),
        action_scale=float(model_spec["action_scale"]),
        max_episode_steps=args.max_episode_steps,
        success_steps=args.success_steps,
        failure_steps=args.failure_steps,
        hard_episodes=args.hard_episodes,
        max_hard_attempts_per_episode=args.max_hard_attempts_per_episode,
        sweep_episodes=args.sweep_episodes,
        sweep_difficulties=args.sweep_difficulties,
        hard_upright_min=args.hard_upright_min,
        hard_upright_max=args.hard_upright_max,
    )


def summarize_seed(model_name, seed, sweep_rows, hard_rows):
    row = {
        "model": model_name,
        "seed": int(seed),
        "sweep_recovered": int(sum(item["recovered"] for item in sweep_rows)),
        "sweep_episodes": int(len(sweep_rows)),
        "hard_recovered": int(sum(item["recovered"] for item in hard_rows)),
        "hard_episodes": int(len(hard_rows)),
    }
    row["sweep_success_rate"] = row["sweep_recovered"] / max(row["sweep_episodes"], 1)
    row["hard_success_rate"] = row["hard_recovered"] / max(row["hard_episodes"], 1)
    row["score"] = 0.7 * row["sweep_success_rate"] + 0.3 * row["hard_success_rate"]
    for difficulty in sorted({item["difficulty"] for item in sweep_rows}):
        rows = [item for item in sweep_rows if item["difficulty"] == difficulty]
        row[f"difficulty_{difficulty:.2f}"] = int(sum(item["recovered"] for item in rows))
    row["hard_bin_0_025"] = int(
        sum(item["recovered"] for item in hard_rows if 0.0 <= item["initial_upright"] < 0.25)
    )
    row["hard_bin_025_05"] = int(
        sum(item["recovered"] for item in hard_rows if 0.25 <= item["initial_upright"] <= 0.5)
    )
    row["hard_bin_0_025_episodes"] = int(
        sum(1 for item in hard_rows if 0.0 <= item["initial_upright"] < 0.25)
    )
    row["hard_bin_025_05_episodes"] = int(
        sum(1 for item in hard_rows if 0.25 <= item["initial_upright"] <= 0.5)
    )
    return row


def aggregate_rows(model_name, rows):
    total_sweep_recovered = int(sum(row["sweep_recovered"] for row in rows))
    total_sweep_episodes = int(sum(row["sweep_episodes"] for row in rows))
    total_hard_recovered = int(sum(row["hard_recovered"] for row in rows))
    total_hard_episodes = int(sum(row["hard_episodes"] for row in rows))
    scores = np.array([row["score"] for row in rows], dtype=np.float64)
    output = {
        "model": model_name,
        "seeds": len(rows),
        "sweep_recovered": total_sweep_recovered,
        "sweep_episodes": total_sweep_episodes,
        "sweep_success_rate": total_sweep_recovered / max(total_sweep_episodes, 1),
        "hard_recovered": total_hard_recovered,
        "hard_episodes": total_hard_episodes,
        "hard_success_rate": total_hard_recovered / max(total_hard_episodes, 1),
        "score_mean": float(scores.mean()) if len(scores) else 0.0,
        "score_std": float(scores.std(ddof=0)) if len(scores) else 0.0,
    }
    output["score_from_totals"] = 0.7 * output["sweep_success_rate"] + 0.3 * output["hard_success_rate"]
    for key in ["0.25", "0.35", "0.45", "0.50", "0.60"]:
        column = f"difficulty_{float(key):.2f}"
        output[f"difficulty_{key}_recovered"] = int(sum(row.get(column, 0) for row in rows))
        output[f"difficulty_{key}_episodes"] = int(sum(row["sweep_episodes"] for row in rows) / 5)
    output["hard_bin_0_025"] = int(sum(row["hard_bin_0_025"] for row in rows))
    output["hard_bin_0_025_episodes"] = int(sum(row["hard_bin_0_025_episodes"] for row in rows))
    output["hard_bin_025_05"] = int(sum(row["hard_bin_025_05"] for row in rows))
    output["hard_bin_025_05_episodes"] = int(sum(row["hard_bin_025_05_episodes"] for row in rows))
    return output


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/benchmark_fixed_seed_v1")
    parser.add_argument("--models", default=None, help="JSON string or JSON file with model specs")
    parser.add_argument("--seeds", type=parse_float_list, default=[61000, 71000, 81000, 91000, 101000])
    parser.add_argument("--env-model", default="unitree_a1/scene.xml")
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
    model_specs = parse_model_specs(args.models)

    with (output_dir / "benchmark_config.json").open("w") as f:
        json.dump({**vars(args), "models": model_specs}, f, indent=2, sort_keys=True)

    per_seed_rows = []
    aggregate = []
    for model_spec in model_specs:
        model_name = model_spec["name"]
        print(f"[model] {model_name}", flush=True)
        model = PPO.load(model_spec["model"])
        model_rows = []
        for seed in args.seeds:
            eval_args = make_eval_args(args, model_spec, int(seed))
            vec_source = load_vecnormalize(eval_args)
            hard_rows, hard_attempts = collect_hard_bin(model, vec_source, eval_args)
            sweep_rows = collect_sweep(model, vec_source, eval_args)
            if vec_source is not None:
                vec_source.close()
            row = summarize_seed(model_name, int(seed), sweep_rows, hard_rows)
            row["hard_attempts"] = hard_attempts
            row["action_scale"] = float(model_spec["action_scale"])
            model_rows.append(row)
            per_seed_rows.append(row)
            print(
                "[seed] "
                f"model={model_name} seed={int(seed)} "
                f"sweep={row['sweep_recovered']}/{row['sweep_episodes']} "
                f"hard={row['hard_recovered']}/{row['hard_episodes']} "
                f"score={row['score']:.3f}",
                flush=True,
            )
        aggregate.append(aggregate_rows(model_name, model_rows))

    write_csv(output_dir / "per_seed_summary.csv", per_seed_rows)
    write_csv(output_dir / "aggregate_summary.csv", aggregate)
    with (output_dir / "aggregate_summary.json").open("w") as f:
        json.dump(aggregate, f, indent=2, sort_keys=True)

    print("[aggregate]", flush=True)
    for row in aggregate:
        print(
            f"{row['model']}: "
            f"sweep={row['sweep_recovered']}/{row['sweep_episodes']} "
            f"hard={row['hard_recovered']}/{row['hard_episodes']} "
            f"score={row['score_from_totals']:.3f} "
            f"score_mean={row['score_mean']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
