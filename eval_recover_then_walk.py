"""Composite recover-then-walk evaluation.

Each episode starts from a normal recovery reset. The recovery policy controls
the robot until the environment reports recovered. Without resetting MuJoCo
state, control then switches to the walking policy. Composite success means the
robot recovers and then walks at least the requested distance.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs import UnitreeA1Env, UnitreeA1WalkEnv


DEFAULT_DIFFICULTIES = [0.25, 0.35, 0.45, 0.50, 0.60]


def parse_float_list(value):
    if isinstance(value, list):
        return [float(item) for item in value]
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def load_seed_rows(path, difficulties=None, episodes_per_difficulty=None, seed=20260604):
    path = Path(path) if path else None
    if path and path.exists():
        rows = []
        with path.open(newline="") as f:
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

    rng = np.random.default_rng(seed)
    rows = []
    difficulties = difficulties or DEFAULT_DIFFICULTIES
    episodes_per_difficulty = int(episodes_per_difficulty or 30)
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
    return rows


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_recovery_dummy_env(args, difficulty=0.45):
    def _init():
        return UnitreeA1Env(
            args.model_xml,
            task="recovery",
            max_episode_steps=args.recovery_max_steps,
            recovery_difficulty=difficulty,
            normalize_obs=False,
            action_scale=args.recovery_action_scale,
            success_steps=args.success_steps,
            failure_steps=args.failure_steps,
        )

    return DummyVecEnv([_init])


def make_walk_dummy_env(args):
    def _init():
        return UnitreeA1WalkEnv(
            model_path=args.model_xml,
            target_vx=args.target_vx,
            reset_noise=0.0,
            terrain_friction=args.terrain_friction,
            terrain_height_scale=args.terrain_height_scale,
            max_episode_steps=args.walk_max_steps,
            action_scale=args.walk_action_scale,
            overspeed_deadband=args.overspeed_deadband,
            overspeed_weight=args.overspeed_weight,
            overspeed_quadratic_weight=args.overspeed_quadratic_weight,
            forward_reward_weight=args.forward_reward_weight,
            progress_reward_weight=args.progress_reward_weight,
            backward_penalty_weight=args.backward_penalty_weight,
            low_speed_penalty_weight=args.low_speed_penalty_weight,
            low_speed_fraction=args.low_speed_fraction,
            speed_reward_sharpness=args.speed_reward_sharpness,
            normalize_obs=False,
        )

    return DummyVecEnv([_init])


def load_vecnormalize(path, dummy_env):
    vec = VecNormalize.load(path, dummy_env)
    vec.training = False
    vec.norm_reward = False
    return vec


def normalize_obs(obs, vec):
    obs_batch = np.asarray(obs, dtype=np.float32).reshape(1, -1)
    return vec.normalize_obs(obs_batch.copy())


def policy_action(model, vec, obs):
    action, _ = model.predict(normalize_obs(obs, vec), deterministic=True)
    return np.asarray(action, dtype=np.float32).reshape(-1)


def make_composite_env(args, difficulty):
    env = UnitreeA1WalkEnv(
        model_path=args.model_xml,
        target_vx=args.target_vx,
        reset_noise=0.0,
        terrain_friction=args.terrain_friction,
        terrain_height_scale=args.terrain_height_scale,
        max_episode_steps=args.recovery_max_steps,
        recovery_difficulty=float(difficulty),
        normalize_obs=False,
        action_scale=args.recovery_action_scale,
        success_steps=args.success_steps,
        failure_steps=args.failure_steps,
        overspeed_deadband=args.overspeed_deadband,
        overspeed_weight=args.overspeed_weight,
        overspeed_quadratic_weight=args.overspeed_quadratic_weight,
        forward_reward_weight=args.forward_reward_weight,
        progress_reward_weight=args.progress_reward_weight,
        backward_penalty_weight=args.backward_penalty_weight,
        low_speed_penalty_weight=args.low_speed_penalty_weight,
        low_speed_fraction=args.low_speed_fraction,
        speed_reward_sharpness=args.speed_reward_sharpness,
    )
    env.set_hard_reset_oversampling(0.0)
    return env


def start_recovery(env, seed, difficulty, args):
    env.set_recovery_difficulty(float(difficulty))
    env.max_episode_steps = int(args.recovery_max_steps)
    env.action_scale = float(args.recovery_action_scale)
    env._configure_terrain()
    obs, info = UnitreeA1Env.reset(env, seed=int(seed), options={"task": "recovery"})
    env._configure_terrain()
    return obs, dict(info)


def start_walking_from_current_state(env, args):
    env.current_task = "walk"
    env.max_episode_steps = int(args.walk_max_steps)
    env.action_scale = float(args.walk_action_scale)
    env.steps = 0
    env.last_action.fill(0.0)
    env.initial_x = float(env.data.qpos[0])
    env.prev_x = env.initial_x
    env._reset_progress_trackers()
    env._configure_terrain()
    return env._get_obs_raw()


def run_episode(args, recovery_model, recovery_vec, walk_model, walk_vec, seed, difficulty):
    env = make_composite_env(args, difficulty)
    recovery_steps = 0
    walk_steps = 0
    recovery_return = 0.0
    walk_return = 0.0
    recovered = False
    switched = False
    walk_reached_distance = False
    failure_stage = ""
    recovery_info = {}
    walk_info = {}
    initial_info = {}

    try:
        obs, initial_info = start_recovery(env, seed, difficulty, args)
        info = dict(initial_info)

        while recovery_steps < args.recovery_max_steps:
            action = policy_action(recovery_model, recovery_vec, obs)
            obs, reward, terminated, truncated, info = UnitreeA1Env.step(env, action)
            recovery_return += float(reward)
            recovery_steps += 1
            recovery_info = dict(info)
            if bool(info.get("recovered", False)):
                recovered = True
                break
            if bool(info.get("catastrophic", False) or info.get("failure_timeout", False)):
                failure_stage = "recovery_failed"
                break
            if bool(terminated or truncated):
                failure_stage = "recovery_timeout"
                break

        if recovered:
            switched = True
            obs = start_walking_from_current_state(env, args)
            while walk_steps < args.walk_max_steps:
                action = policy_action(walk_model, walk_vec, obs)
                obs, reward, terminated, truncated, info = UnitreeA1WalkEnv.step(env, action)
                walk_return += float(reward)
                walk_steps += 1
                walk_info = dict(info)
                if float(info.get("distance", 0.0)) >= args.min_walk_distance:
                    walk_reached_distance = True
                    failure_stage = "success"
                    break
                if bool(info.get("fallen", False) or info.get("catastrophic", False) or terminated):
                    failure_stage = "walk_failed"
                    break
                if bool(truncated):
                    failure_stage = "walk_timeout_short"
                    break
        elif not failure_stage:
            failure_stage = "recovery_timeout"
    finally:
        env.close()

    final_info = walk_info if switched and walk_info else recovery_info
    walk_distance = float(walk_info.get("distance", 0.0)) if walk_info else 0.0
    elapsed_walk = max(walk_steps * env.model.opt.timestep * env.frame_skip, 1e-8)
    composite_success = bool(recovered and switched and walk_reached_distance)
    return {
        "seed": int(seed),
        "difficulty": f"{float(difficulty):.2f}",
        "initial_upright": float(initial_info.get("initial_upright", np.nan)),
        "initial_height_error": float(initial_info.get("initial_height_error", np.nan)),
        "recovered": int(recovered),
        "switched_to_walk": int(switched),
        "walk_reached_distance": int(walk_reached_distance),
        "composite_success": int(composite_success),
        "failure_stage": failure_stage,
        "recovery_steps": int(recovery_steps),
        "walk_steps": int(walk_steps),
        "total_steps": int(recovery_steps + walk_steps),
        "walk_distance": walk_distance,
        "walk_mean_vx": walk_distance / elapsed_walk if walk_steps > 0 else 0.0,
        "final_upright": float(final_info.get("upright", np.nan)),
        "final_height_error": float(final_info.get("height_error", np.nan)),
        "final_vx": float(final_info.get("vx", 0.0)),
        "fallen": int(bool(final_info.get("fallen", False))),
        "catastrophic": int(bool(final_info.get("catastrophic", False))),
        "recovery_return": float(recovery_return),
        "walk_return": float(walk_return),
        "total_return": float(recovery_return + walk_return),
    }


def summarize(rows):
    if not rows:
        return {
            "episodes": 0,
            "recovered": 0,
            "switched_to_walk": 0,
            "walk_reached_distance": 0,
            "composite_success": 0,
            "fallen": 0,
            "catastrophic": 0,
            "mean_walk_distance": 0.0,
            "mean_walk_vx": 0.0,
            "mean_recovery_steps": 0.0,
            "mean_walk_steps": 0.0,
            "mean_initial_upright": 0.0,
            "mean_final_upright": 0.0,
        }
    return {
        "episodes": len(rows),
        "recovered": int(sum(row["recovered"] for row in rows)),
        "switched_to_walk": int(sum(row["switched_to_walk"] for row in rows)),
        "walk_reached_distance": int(sum(row["walk_reached_distance"] for row in rows)),
        "composite_success": int(sum(row["composite_success"] for row in rows)),
        "fallen": int(sum(row["fallen"] for row in rows)),
        "catastrophic": int(sum(row["catastrophic"] for row in rows)),
        "recovery_rate": float(np.mean([row["recovered"] for row in rows])),
        "composite_success_rate": float(np.mean([row["composite_success"] for row in rows])),
        "mean_walk_distance": float(np.mean([row["walk_distance"] for row in rows])),
        "mean_walk_vx": float(np.mean([row["walk_mean_vx"] for row in rows])),
        "mean_recovery_steps": float(np.mean([row["recovery_steps"] for row in rows])),
        "mean_walk_steps": float(np.mean([row["walk_steps"] for row in rows])),
        "mean_initial_upright": float(np.mean([row["initial_upright"] for row in rows])),
        "mean_final_upright": float(np.mean([row["final_upright"] for row in rows])),
    }


def write_summaries(output_dir, rows):
    difficulties = sorted({row["difficulty"] for row in rows}, key=float)
    by_difficulty = [{"difficulty": difficulty, **summarize([row for row in rows if row["difficulty"] == difficulty])} for difficulty in difficulties]
    aggregate = {"difficulty": "all", **summarize(rows)}
    fields = [
        "difficulty",
        "episodes",
        "recovered",
        "switched_to_walk",
        "walk_reached_distance",
        "composite_success",
        "fallen",
        "catastrophic",
        "recovery_rate",
        "composite_success_rate",
        "mean_walk_distance",
        "mean_walk_vx",
        "mean_recovery_steps",
        "mean_walk_steps",
        "mean_initial_upright",
        "mean_final_upright",
    ]
    write_csv(output_dir / "summary_by_difficulty.csv", by_difficulty, fields)
    write_csv(output_dir / "summary_aggregate.csv", [aggregate], fields)
    with (output_dir / "summary.json").open("w") as f:
        json.dump({"by_difficulty": by_difficulty, "aggregate": aggregate}, f, indent=2, sort_keys=True)
    return by_difficulty, aggregate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery-model", default="runs/recovery_fresh_curriculum_025_to_045_reward_v2/ppo_recovery_stand_final.zip")
    parser.add_argument("--recovery-vecnormalize", default="runs/recovery_fresh_curriculum_025_to_045_reward_v2/vecnormalize.pkl")
    parser.add_argument("--walk-model", default="runs/walk_lowfriction_terrain_best/ppo_walk_lowfriction_terrain_best.zip")
    parser.add_argument("--walk-vecnormalize", default="runs/walk_lowfriction_terrain_best/vecnormalize_lowfriction_terrain_best.pkl")
    parser.add_argument("--model-xml", default="unitree_a1/scene.xml")
    parser.add_argument("--seed-file", default="runs/recovery_seed_benchmark_v3/seeds.csv")
    parser.add_argument("--output-dir", default="runs/recover_then_walk_eval_v1")
    parser.add_argument("--episodes-per-difficulty", type=int, default=30)
    parser.add_argument("--difficulties", type=parse_float_list, default=DEFAULT_DIFFICULTIES)
    parser.add_argument("--seed-bank-seed", type=int, default=20260604)
    parser.add_argument("--recovery-max-steps", type=int, default=1000)
    parser.add_argument("--walk-max-steps", type=int, default=500)
    parser.add_argument("--min-walk-distance", type=float, default=0.5)
    parser.add_argument("--success-steps", type=int, default=15)
    parser.add_argument("--failure-steps", type=int, default=180)
    parser.add_argument("--recovery-action-scale", type=float, default=0.5)
    parser.add_argument("--walk-action-scale", type=float, default=0.5)
    parser.add_argument("--target-vx", type=float, default=0.2)
    parser.add_argument("--terrain-friction", type=float, default=0.35)
    parser.add_argument("--terrain-height-scale", type=float, default=0.50)
    parser.add_argument("--overspeed-deadband", type=float, default=0.02)
    parser.add_argument("--overspeed-weight", type=float, default=8.0)
    parser.add_argument("--overspeed-quadratic-weight", type=float, default=20.0)
    parser.add_argument("--forward-reward-weight", type=float, default=3.0)
    parser.add_argument("--progress-reward-weight", type=float, default=1.5)
    parser.add_argument("--backward-penalty-weight", type=float, default=2.0)
    parser.add_argument("--low-speed-penalty-weight", type=float, default=0.0)
    parser.add_argument("--low-speed-fraction", type=float, default=0.6)
    parser.add_argument("--speed-reward-sharpness", type=float, default=4.0)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_rows = load_seed_rows(
        args.seed_file,
        difficulties=args.difficulties,
        episodes_per_difficulty=args.episodes_per_difficulty,
        seed=args.seed_bank_seed,
    )
    write_csv(output_dir / "seeds.csv", seed_rows, ["difficulty", "episode_idx", "seed"])
    with (output_dir / "eval_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    recovery_vec = load_vecnormalize(args.recovery_vecnormalize, make_recovery_dummy_env(args))
    walk_vec = load_vecnormalize(args.walk_vecnormalize, make_walk_dummy_env(args))
    recovery_model = PPO.load(args.recovery_model, device="cpu")
    walk_model = PPO.load(args.walk_model, device="cpu")

    rows = []
    fields = [
        "seed",
        "difficulty",
        "initial_upright",
        "initial_height_error",
        "recovered",
        "switched_to_walk",
        "walk_reached_distance",
        "composite_success",
        "failure_stage",
        "recovery_steps",
        "walk_steps",
        "total_steps",
        "walk_distance",
        "walk_mean_vx",
        "final_upright",
        "final_height_error",
        "final_vx",
        "fallen",
        "catastrophic",
        "recovery_return",
        "walk_return",
        "total_return",
    ]
    for difficulty in sorted({row["difficulty"] for row in seed_rows}, key=float):
        difficulty_rows = [row for row in seed_rows if row["difficulty"] == difficulty]
        successes = 0
        recovered = 0
        for item in difficulty_rows:
            row = run_episode(
                args,
                recovery_model,
                recovery_vec,
                walk_model,
                walk_vec,
                seed=int(item["seed"]),
                difficulty=float(difficulty),
            )
            rows.append(row)
            successes += int(row["composite_success"])
            recovered += int(row["recovered"])
        print(
            f"[difficulty] {float(difficulty):.2f} "
            f"recovered={recovered}/{len(difficulty_rows)} "
            f"composite_success={successes}/{len(difficulty_rows)}",
            flush=True,
        )

    write_csv(output_dir / "episodes.csv", rows, fields)
    by_difficulty, aggregate = write_summaries(output_dir, rows)
    print("[aggregate]", flush=True)
    print(
        f"recovered={aggregate['recovered']}/{aggregate['episodes']} "
        f"composite_success={aggregate['composite_success']}/{aggregate['episodes']} "
        f"mean_walk_distance={aggregate['mean_walk_distance']:.3f} "
        f"mean_walk_vx={aggregate['mean_walk_vx']:.3f}",
        flush=True,
    )

    recovery_vec.close()
    walk_vec.close()


if __name__ == "__main__":
    main()
