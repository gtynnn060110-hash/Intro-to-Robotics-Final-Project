"""Evaluate two-stage righting + standing policies.

The evaluator can run either:
- one policy (`--mode single`) for direct comparison, or
- two policies (`--mode two-policy`) with a simple state machine:
  use righting while upright is low, switch to standing after the robot is
  righted enough.
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


def make_dummy_env(args, difficulty=0.45, hard_reset_prob=0.0):
    def _init():
        env = UnitreeA1Env(
            args.env_model,
            task="recovery",
            max_episode_steps=args.max_episode_steps,
            recovery_difficulty=difficulty,
            normalize_obs=False,
            action_scale=args.action_scale,
            success_steps=args.success_steps,
            failure_steps=args.failure_steps,
        )
        env.set_hard_reset_oversampling(
            hard_reset_prob,
            args.hard_upright_min,
            args.hard_upright_max,
        )
        return Monitor(env)

    return DummyVecEnv([_init])


def load_vecnormalize(path, args):
    if not path:
        return None
    dummy = make_dummy_env(args)
    vec = VecNormalize.load(path, dummy)
    vec.training = False
    vec.norm_reward = False
    return vec


def normalize_obs(obs, vec):
    obs_batch = np.asarray(obs, dtype=np.float32).reshape(1, -1)
    if vec is None:
        return obs_batch
    return vec.normalize_obs(obs_batch.copy())


def make_eval_env(args, difficulty, hard_reset_prob):
    env = UnitreeA1Env(
        args.env_model,
        task="recovery",
        max_episode_steps=args.max_episode_steps,
        recovery_difficulty=difficulty,
        normalize_obs=False,
        action_scale=args.action_scale,
        success_steps=args.success_steps,
        failure_steps=args.failure_steps,
    )
    env.set_hard_reset_oversampling(
        hard_reset_prob,
        args.hard_upright_min,
        args.hard_upright_max,
    )
    return Monitor(env)


def should_switch_to_standing(info, args):
    upright = float(info.get("upright", -1.0))
    z = float(info.get("z", 0.0))
    target_z = float(info.get("target_z", z))
    return bool(upright >= args.switch_upright and z >= target_z - args.switch_height_margin)


def run_episode(
    args,
    righting_model,
    righting_vec,
    standing_model,
    standing_vec,
    seed,
    difficulty,
    hard_reset_prob,
):
    env = make_eval_env(args, difficulty=difficulty, hard_reset_prob=hard_reset_prob)
    obs, reset_info = env.reset(seed=int(seed))
    info = dict(reset_info)
    initial_upright = float(info.get("initial_upright", np.nan))
    phase = "righting" if args.mode == "two-policy" and initial_upright < args.start_righting_below else "standing"
    if args.mode == "single":
        phase = "standing"

    phase_switches = 0
    righting_steps = 0
    standing_steps = 0
    ever_righted = False
    ever_switched = phase == "standing"
    episode_return = 0.0
    length = 0
    final_info = dict(info)

    try:
        while True:
            if args.mode == "two-policy":
                if phase == "righting" and should_switch_to_standing(info, args):
                    phase = "standing"
                    phase_switches += 1
                    ever_switched = True
                elif (
                    args.allow_switch_back
                    and phase == "standing"
                    and float(info.get("upright", 1.0)) < args.switch_back_upright
                ):
                    phase = "righting"
                    phase_switches += 1

            if phase == "righting":
                model = righting_model
                vec = righting_vec
                action_gain = args.righting_action_gain
                righting_steps += 1
            else:
                model = standing_model
                vec = standing_vec
                action_gain = args.standing_action_gain
                standing_steps += 1

            policy_obs = normalize_obs(obs, vec)
            action, _ = model.predict(policy_obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32).reshape(-1) * float(action_gain)
            action = np.clip(action, -1.0, 1.0)
            obs, reward, terminated, truncated, info = env.step(action)
            final_info = dict(info)
            episode_return += float(reward)
            length += 1

            if should_switch_to_standing(info, args):
                ever_righted = True
            if bool(terminated or truncated):
                break
    finally:
        env.close()

    final_info.update(
        {
            "seed": int(seed),
            "difficulty": float(difficulty),
            "initial_upright": initial_upright,
            "bin": bin_name(initial_upright),
            "phase_switches": int(phase_switches),
            "righting_steps_eval": int(righting_steps),
            "standing_steps_eval": int(standing_steps),
            "ever_righted_eval": bool(ever_righted),
            "ever_switched_eval": bool(ever_switched),
            "eval_return": float(episode_return),
            "eval_length": int(length),
        }
    )
    return final_info


def episode_row(info):
    return {
        "seed": int(info.get("seed", -1)),
        "difficulty": float(info.get("difficulty", 0.0)),
        "initial_upright": float(info.get("initial_upright", np.nan)),
        "bin": info.get("bin", bin_name(info.get("initial_upright", np.nan))),
        "recovered": int(bool(info.get("recovered", False))),
        "stable": int(bool(info.get("stable", False))),
        "failed": int(bool(info.get("failure_timeout", False) or info.get("catastrophic", False))),
        "fallen": int(bool(info.get("fallen", False))),
        "ever_righted_eval": int(bool(info.get("ever_righted_eval", False))),
        "ever_switched_eval": int(bool(info.get("ever_switched_eval", False))),
        "phase_switches": int(info.get("phase_switches", 0)),
        "righting_steps_eval": int(info.get("righting_steps_eval", 0)),
        "standing_steps_eval": int(info.get("standing_steps_eval", 0)),
        "final_upright": float(info.get("upright", np.nan)),
        "final_height_error": float(info.get("height_error", np.nan)),
        "length": int(info.get("eval_length", info.get("episode", {}).get("l", 0))),
        "return": float(info.get("eval_return", info.get("episode", {}).get("r", 0.0))),
    }


def summarize_rows(rows):
    if not rows:
        return {
            "episodes": 0,
            "recovered": 0,
            "stable": 0,
            "failed": 0,
            "fallen": 0,
            "ever_righted": 0,
            "ever_switched": 0,
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
        "ever_righted": int(sum(row["ever_righted_eval"] for row in rows)),
        "ever_switched": int(sum(row["ever_switched_eval"] for row in rows)),
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


def collect_hard_bin(args, righting_model, righting_vec, standing_model, standing_vec):
    rows = []
    attempts = 0
    max_attempts = args.hard_episodes * args.max_hard_attempts_per_episode
    while len(rows) < args.hard_episodes and attempts < max_attempts:
        seed = args.seed + attempts
        info = run_episode(
            args,
            righting_model,
            righting_vec,
            standing_model,
            standing_vec,
            seed=seed,
            difficulty=0.45,
            hard_reset_prob=1.0,
        )
        initial_upright = float(info.get("initial_upright", 99.0))
        if args.hard_upright_min <= initial_upright <= args.hard_upright_max:
            rows.append(episode_row(info))
        attempts += 1
    return rows, attempts


def collect_sweep(args, righting_model, righting_vec, standing_model, standing_vec):
    rows = []
    for difficulty in args.sweep_difficulties:
        for episode in range(args.sweep_episodes):
            seed = args.seed + 100_000 + int(round(1000 * difficulty)) + episode
            info = run_episode(
                args,
                righting_model,
                righting_vec,
                standing_model,
                standing_vec,
                seed=seed,
                difficulty=difficulty,
                hard_reset_prob=0.0,
            )
            rows.append(episode_row(info))
    return rows


def sweep_summary_rows(rows):
    summary_rows = []
    for difficulty in sorted({row["difficulty"] for row in rows}):
        difficulty_rows = [row for row in rows if row["difficulty"] == difficulty]
        summary_rows.append({"difficulty": difficulty, "bin": "all", **summarize_rows(difficulty_rows)})
        for _, _, name in UPRIGHT_BINS:
            bin_rows = [row for row in difficulty_rows if row["bin"] == name]
            if bin_rows:
                summary_rows.append({"difficulty": difficulty, "bin": name, **summarize_rows(bin_rows)})
    return summary_rows


def print_summary(hard_summary, hard_attempts, sweep_summary):
    print(
        "[hard-bin] "
        f"recovered={hard_summary['recovered']}/{hard_summary['episodes']} "
        f"righted={hard_summary['ever_righted']}/{hard_summary['episodes']} "
        f"switched={hard_summary['ever_switched']}/{hard_summary['episodes']} "
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
                f"righted={row['ever_righted']}/{row['episodes']} "
                f"failed={row['failed']}/{row['episodes']} "
                f"mean_init={row['mean_initial_upright']:.3f}",
                flush=True,
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["two-policy", "single"], default="two-policy")
    parser.add_argument("--righting-model", default=None)
    parser.add_argument("--righting-vecnormalize", default=None)
    parser.add_argument("--standing-model", required=True)
    parser.add_argument("--standing-vecnormalize", default=None)
    parser.add_argument("--env-model", default="unitree_a1/scene.xml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=200_000)
    parser.add_argument("--hard-episodes", type=int, default=100)
    parser.add_argument("--max-hard-attempts-per-episode", type=int, default=10)
    parser.add_argument("--sweep-episodes", type=int, default=30)
    parser.add_argument(
        "--sweep-difficulties",
        type=parse_float_list,
        default=[0.25, 0.35, 0.45, 0.50, 0.60],
    )
    parser.add_argument("--action-scale", type=float, default=0.9)
    parser.add_argument("--righting-action-gain", type=float, default=1.0)
    parser.add_argument("--standing-action-gain", type=float, default=1.0)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--success-steps", type=int, default=15)
    parser.add_argument("--failure-steps", type=int, default=180)
    parser.add_argument("--hard-upright-min", type=float, default=0.0)
    parser.add_argument("--hard-upright-max", type=float, default=0.5)
    parser.add_argument("--start-righting-below", type=float, default=0.62)
    parser.add_argument("--switch-upright", type=float, default=0.62)
    parser.add_argument("--switch-back-upright", type=float, default=0.48)
    parser.add_argument("--switch-height-margin", type=float, default=0.12)
    parser.add_argument("--allow-switch-back", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if args.mode == "two-policy" and not args.righting_model:
        parser.error("--righting-model is required for --mode two-policy")
    return args


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    standing_model = PPO.load(args.standing_model)
    standing_vec = load_vecnormalize(args.standing_vecnormalize, args)
    if args.mode == "two-policy":
        righting_model = PPO.load(args.righting_model)
        righting_vec = load_vecnormalize(args.righting_vecnormalize, args)
    else:
        righting_model = standing_model
        righting_vec = standing_vec

    hard_rows, hard_attempts = collect_hard_bin(
        args,
        righting_model,
        righting_vec,
        standing_model,
        standing_vec,
    )
    hard_summary = summarize_rows(hard_rows)
    hard_summary["attempts"] = hard_attempts
    hard_summary["difficulty"] = 0.45
    hard_summary["initial_upright_bin"] = f"[{args.hard_upright_min},{args.hard_upright_max}]"

    sweep_rows = collect_sweep(
        args,
        righting_model,
        righting_vec,
        standing_model,
        standing_vec,
    )
    sweep_summary = sweep_summary_rows(sweep_rows)

    episode_fields = [
        "seed",
        "difficulty",
        "initial_upright",
        "bin",
        "recovered",
        "stable",
        "failed",
        "fallen",
        "ever_righted_eval",
        "ever_switched_eval",
        "phase_switches",
        "righting_steps_eval",
        "standing_steps_eval",
        "final_upright",
        "final_height_error",
        "length",
        "return",
    ]
    summary_fields = [
        "difficulty",
        "bin",
        "episodes",
        "recovered",
        "stable",
        "failed",
        "fallen",
        "ever_righted",
        "ever_switched",
        "mean_return",
        "mean_len",
        "mean_initial_upright",
        "mean_final_upright",
    ]
    write_csv(output_dir / "hard_bin_episodes.csv", hard_rows, episode_fields)
    with (output_dir / "hard_bin_summary.json").open("w") as f:
        json.dump(hard_summary, f, indent=2)
    write_csv(output_dir / "difficulty_sweep_episodes.csv", sweep_rows, episode_fields)
    write_csv(output_dir / "difficulty_sweep_summary.csv", sweep_summary, summary_fields)
    with (output_dir / "eval_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    print_summary(hard_summary, hard_attempts, sweep_summary)
    for vec in (standing_vec, righting_vec):
        if vec is not None:
            vec.close()


if __name__ == "__main__":
    main()
