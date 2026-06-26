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
    if path is None:
        return None
    vec = VecNormalize.load(path, dummy_env)
    vec.training = False
    vec.norm_reward = False
    return vec


def resolve_checkpoint_path(path, default_filename):
    path = Path(path)
    if path.is_dir():
        for filename in (default_filename, "best_model.zip"):
            candidate = path / filename
            if candidate.exists():
                return str(candidate)
        path = path / default_filename
    return str(path)


def resolve_vecnormalize_path(path):
    if path is None:
        return None
    path = Path(path)
    if path.is_dir():
        for filename in ("vecnormalize.pkl", "best_vecnormalize.pkl"):
            candidate = path / filename
            if candidate.exists():
                return str(candidate)
        return None
    return str(path)


def normalize_obs(obs, vec):
    obs_batch = np.asarray(obs, dtype=np.float32).reshape(1, -1)
    return vec.normalize_obs(obs_batch.copy())


def adapt_obs_dim(obs, target_dim):
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    target_dim = int(target_dim)
    if obs.shape[0] == target_dim:
        return obs
    if obs.shape[0] > target_dim:
        return obs[:target_dim].copy()
    out = np.zeros(target_dim, dtype=np.float32)
    out[: obs.shape[0]] = obs
    return out


def policy_action(model, vec, obs):
    target_dim = int(np.prod(model.observation_space.shape))
    raw_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    if vec is None:
        policy_obs = adapt_obs_dim(raw_obs, target_dim).reshape(1, -1)
    else:
        vec_dim = int(np.prod(vec.observation_space.shape))
        if vec_dim == target_dim:
            policy_obs = normalize_obs(adapt_obs_dim(raw_obs, vec_dim), vec)
        elif vec_dim == raw_obs.shape[0]:
            policy_obs = adapt_obs_dim(normalize_obs(raw_obs, vec).reshape(-1), target_dim).reshape(1, -1)
        else:
            policy_obs = adapt_obs_dim(raw_obs, target_dim).reshape(1, -1)
    action, _ = model.predict(policy_obs, deterministic=True)
    return np.asarray(action, dtype=np.float32).reshape(-1)


def recovery_obs(env):
    """Observation with the recovery/standing shape, without walk gait-clock fields."""
    return UnitreeA1Env._get_obs_raw(env).astype(np.float32)


def walk_obs(env):
    return env._get_obs_raw().astype(np.float32)


def supervisor_metrics(env):
    terrain_z, target_z, z, upright, height_error = env._posture_metrics()
    lin_vel = np.asarray(env.data.qvel[:3], dtype=np.float64)
    ang_vel = np.asarray(env.data.qvel[3:6], dtype=np.float64)
    body_z_axis = env._body_z_axis(env.data.qpos[3:7])
    pitch_tilt = float(body_z_axis[0])
    roll_tilt = float(body_z_axis[1])

    support_count = 0.0
    if hasattr(env, "_ordered_foot_geom_ids") and env._ordered_foot_geom_ids:
        contacts = env._foot_contacts()
        clearances = []
        for i, geom_id in enumerate(env._ordered_foot_geom_ids):
            pos = np.asarray(env.data.geom_xpos[geom_id], dtype=np.float64)
            foot_radius = float(env._foot_radii[i]) if hasattr(env, "_foot_radii") else 0.0
            foot_terrain_z = env._raycast_terrain_height(pos[:2], pos[2])
            if foot_terrain_z is None:
                foot_terrain_z = 0.0
            clearances.append(float(pos[2] - foot_terrain_z - foot_radius))
        clearances = np.asarray(clearances, dtype=np.float64)
        support_state = np.maximum(
            contacts,
            (clearances <= float(getattr(env, "support_clearance_threshold", 0.035))).astype(np.float64),
        )
        support_count = float(np.sum(support_state))

    return {
        "terrain_z": float(terrain_z),
        "target_z": float(target_z),
        "z": float(z),
        "upright": float(upright),
        "height_error": float(height_error),
        "vx": float(lin_vel[0]),
        "vy": float(lin_vel[1]),
        "lin_xy_norm": float(np.linalg.norm(lin_vel[:2])),
        "ang_vel_norm": float(np.linalg.norm(ang_vel)),
        "yaw_rate": float(ang_vel[2]),
        "pitch_tilt": pitch_tilt,
        "roll_tilt": roll_tilt,
        "tilt_norm": float(np.linalg.norm([pitch_tilt, roll_tilt])),
        "support_count": support_count,
    }


def is_walk_ready(metrics, args):
    return bool(
        metrics["upright"] >= args.walk_ready_upright
        and abs(metrics["height_error"]) <= args.walk_ready_height_error
        and metrics["tilt_norm"] <= args.walk_ready_tilt
        and metrics["ang_vel_norm"] <= args.walk_ready_ang_vel
        and metrics["lin_xy_norm"] <= args.walk_ready_lin_vel
        and metrics["support_count"] >= args.walk_ready_support
    )


def needs_recovery(metrics, args):
    return bool(
        metrics["upright"] <= args.recover_upright
        or metrics["height_error"] <= -args.recover_low_height_margin
        or metrics["tilt_norm"] >= args.recover_tilt
        or metrics["ang_vel_norm"] >= args.recover_ang_vel
        or metrics["support_count"] <= args.recover_support
    )


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
    return recovery_obs(env), dict(info)


def enter_recovery_from_current_state(env, args):
    env.current_task = "recovery"
    env.max_episode_steps = int(args.recovery_max_steps)
    env.action_scale = float(args.recovery_action_scale)
    env.success_steps = 0
    env.failure_steps = 0
    env.prev_upright = float(env._upright_score())
    _, _, _, _, height_error = env._posture_metrics()
    env.prev_abs_height_error = abs(float(height_error))
    return recovery_obs(env)


def start_walking_from_current_state(env, args, initial_target_vx=0.0, last_recovery_action=None):
    env.current_task = "walk"
    env.max_episode_steps = int(args.walk_max_steps)
    env.action_scale = float(args.walk_action_scale)
    env.set_target_vx(float(initial_target_vx))
    env.steps = 0
    if last_recovery_action is None:
        env.last_action.fill(0.0)
    else:
        env.last_action = np.asarray(last_recovery_action, dtype=np.float32).reshape(env.n_joints).copy()
    env.gait_phase = float(args.walk_start_gait_phase)
    env.initial_x = float(env.data.qpos[0])
    env.prev_x = env.initial_x
    env._reset_progress_trackers()
    if env._ordered_foot_geom_ids:
        env.prev_foot_xy = np.array([env.data.geom_xpos[g, :2] for g in env._ordered_foot_geom_ids])
    env._configure_terrain()
    return walk_obs(env)


def run_episode(args, recovery_model, recovery_vec, walk_model, walk_vec, seed, difficulty):
    env = make_composite_env(args, difficulty)
    recovery_steps = 0
    stabilize_steps = 0
    ramp_steps_taken = 0
    walk_steps = 0
    recovery_return = 0.0
    stabilize_return = 0.0
    ramp_return = 0.0
    walk_return = 0.0
    recovered = False
    stabilized = False
    ramped_to_walk = False
    switched = False
    walk_reached_distance = False
    failure_stage = ""
    recovery_info = {}
    walk_info = {}
    initial_info = {}
    final_info = {}
    mode = "recover"
    good_steps = 0
    bad_steps = 0
    recovery_reentries = 0
    last_recovery_action = np.zeros(env.n_joints, dtype=np.float32)

    try:
        obs, initial_info = start_recovery(env, seed, difficulty, args)
        info = dict(initial_info)

        max_total_steps = int(args.recovery_max_steps + args.stabilize_max_steps + args.walk_ramp_steps + args.walk_max_steps)
        while (recovery_steps + stabilize_steps + ramp_steps_taken + walk_steps) < max_total_steps:
            if mode == "recover":
                if recovery_steps >= args.recovery_max_steps * max(1, 1 + recovery_reentries):
                    failure_stage = "recovery_timeout"
                    break
                action = policy_action(recovery_model, recovery_vec, obs)
                last_recovery_action = action.copy()
                _, reward, terminated, truncated, info = UnitreeA1Env.step(env, action)
                obs = recovery_obs(env)
                recovery_return += float(reward)
                recovery_steps += 1
                recovery_info = dict(info)
                final_info = dict(info)
                metrics = supervisor_metrics(env)
                good_steps = good_steps + 1 if is_walk_ready(metrics, args) else 0
                if bool(info.get("recovered", False)) or good_steps >= args.recover_to_stabilize_good_steps:
                    recovered = True
                    mode = "stabilize"
                    env.current_task = "stand"
                    env.max_episode_steps = int(args.stabilize_max_steps)
                    env.action_scale = float(args.recovery_action_scale)
                    good_steps = 0
                    bad_steps = 0
                    continue
                if bool(info.get("catastrophic", False) or info.get("failure_timeout", False)):
                    failure_stage = "recovery_failed"
                    break
                if bool(terminated or truncated):
                    failure_stage = "recovery_timeout"
                    break

            elif mode == "stabilize":
                if stabilize_steps >= args.stabilize_max_steps * max(1, 1 + recovery_reentries):
                    failure_stage = "stabilize_timeout"
                    break
                action = policy_action(recovery_model, recovery_vec, obs)
                last_recovery_action = action.copy()
                _, reward, terminated, truncated, info = UnitreeA1Env.step(env, action)
                obs = recovery_obs(env)
                stabilize_return += float(reward)
                stabilize_steps += 1
                recovery_info = dict(info)
                final_info = dict(info)
                metrics = supervisor_metrics(env)
                if needs_recovery(metrics, args):
                    mode = "recover"
                    obs = enter_recovery_from_current_state(env, args)
                    good_steps = 0
                    bad_steps = 0
                    continue
                if is_walk_ready(metrics, args):
                    good_steps += 1
                else:
                    good_steps = 0
                if good_steps >= args.stabilize_good_steps:
                    stabilized = True
                    switched = True
                    ramped_to_walk = args.walk_ramp_steps <= 0
                    mode = "walk" if ramped_to_walk else "walk_ramp"
                    obs = start_walking_from_current_state(
                        env,
                        args,
                        initial_target_vx=0.0,
                        last_recovery_action=last_recovery_action,
                    )
                    bad_steps = 0
                    continue
                if stabilize_steps >= args.stabilize_max_steps and args.force_walk_after_stabilize:
                    metrics = supervisor_metrics(env)
                    if not needs_recovery(metrics, args):
                        stabilized = True
                        switched = True
                        ramped_to_walk = args.walk_ramp_steps <= 0
                        mode = "walk" if ramped_to_walk else "walk_ramp"
                        obs = start_walking_from_current_state(
                            env,
                            args,
                            initial_target_vx=0.0,
                            last_recovery_action=last_recovery_action,
                        )
                        bad_steps = 0
                        continue
                if bool(info.get("catastrophic", False) or terminated or truncated):
                    failure_stage = "stabilize_failed"
                    break

            elif mode == "walk_ramp":
                ramp_index = ramp_steps_taken
                alpha = 1.0 if args.walk_ramp_steps <= 0 else min((ramp_index + 1) / float(args.walk_ramp_steps), 1.0)
                env.set_target_vx(args.target_vx * alpha)
                env.action_scale = float(args.walk_action_scale) * (
                    args.walk_ramp_min_action_scale + alpha * (1.0 - args.walk_ramp_min_action_scale)
                )
                walk_action = policy_action(walk_model, walk_vec, obs)
                action = (1.0 - alpha) * last_recovery_action + alpha * walk_action
                obs, reward, terminated, truncated, info = UnitreeA1WalkEnv.step(env, action)
                ramp_return += float(reward)
                ramp_steps_taken += 1
                walk_info = dict(info)
                final_info = dict(info)
                metrics = supervisor_metrics(env)
                if needs_recovery(metrics, args):
                    bad_steps += 1
                else:
                    bad_steps = 0
                if bad_steps >= args.recover_bad_steps:
                    if recovery_reentries < args.max_recovery_reentries:
                        recovery_reentries += 1
                        mode = "recover"
                        obs = enter_recovery_from_current_state(env, args)
                        bad_steps = 0
                        continue
                    failure_stage = "ramp_unstable"
                    break
                if bool(info.get("fallen", False) or info.get("catastrophic", False) or terminated):
                    failure_stage = "ramp_failed"
                    break
                if bool(truncated):
                    failure_stage = "ramp_timeout"
                    break
                if ramp_steps_taken >= args.walk_ramp_steps:
                    ramped_to_walk = True
                    mode = "walk"
                    env.set_target_vx(args.target_vx)
                    env.action_scale = float(args.walk_action_scale)
                    bad_steps = 0
                    continue

            elif mode == "walk":
                env.set_target_vx(args.target_vx)
                env.action_scale = float(args.walk_action_scale)
                action = policy_action(walk_model, walk_vec, obs)
                obs, reward, terminated, truncated, info = UnitreeA1WalkEnv.step(env, action)
                walk_return += float(reward)
                walk_steps += 1
                walk_info = dict(info)
                final_info = dict(info)
                if float(info.get("distance", 0.0)) >= args.min_walk_distance:
                    walk_reached_distance = True
                    failure_stage = "success"
                    break
                metrics = supervisor_metrics(env)
                if needs_recovery(metrics, args):
                    bad_steps += 1
                else:
                    bad_steps = 0
                if bad_steps >= args.recover_bad_steps:
                    if recovery_reentries < args.max_recovery_reentries:
                        recovery_reentries += 1
                        mode = "recover"
                        obs = enter_recovery_from_current_state(env, args)
                        bad_steps = 0
                        continue
                    failure_stage = "walk_unstable_need_recovery"
                    break
                if bool(info.get("fallen", False) or info.get("catastrophic", False) or terminated):
                    failure_stage = "walk_failed"
                    break
                if bool(truncated or walk_steps >= args.walk_max_steps):
                    failure_stage = "walk_timeout_short"
                    break
            else:
                failure_stage = f"unknown_mode_{mode}"
                break

        if not failure_stage:
            failure_stage = f"{mode}_timeout"
    finally:
        env.close()

    walk_distance = float(walk_info.get("distance", 0.0)) if walk_info else 0.0
    elapsed_walk = max((walk_steps + ramp_steps_taken) * env.model.opt.timestep * env.frame_skip, 1e-8)
    composite_success = bool(recovered and stabilized and ramped_to_walk and walk_reached_distance)
    return {
        "seed": int(seed),
        "difficulty": f"{float(difficulty):.2f}",
        "initial_upright": float(initial_info.get("initial_upright", np.nan)),
        "initial_height_error": float(initial_info.get("initial_height_error", np.nan)),
        "recovered": int(recovered),
        "stabilized": int(stabilized),
        "ramped_to_walk": int(ramped_to_walk),
        "switched_to_walk": int(switched),
        "walk_reached_distance": int(walk_reached_distance),
        "composite_success": int(composite_success),
        "failure_stage": failure_stage,
        "recovery_steps": int(recovery_steps),
        "stabilize_steps": int(stabilize_steps),
        "ramp_steps": int(ramp_steps_taken),
        "walk_steps": int(walk_steps),
        "total_steps": int(recovery_steps + stabilize_steps + ramp_steps_taken + walk_steps),
        "recovery_reentries": int(recovery_reentries),
        "walk_distance": walk_distance,
        "walk_mean_vx": walk_distance / elapsed_walk if walk_steps > 0 else 0.0,
        "final_upright": float(final_info.get("upright", np.nan)),
        "final_height_error": float(final_info.get("height_error", np.nan)),
        "final_vx": float(final_info.get("vx", 0.0)),
        "fallen": int(bool(final_info.get("fallen", False))),
        "catastrophic": int(bool(final_info.get("catastrophic", False))),
        "recovery_return": float(recovery_return),
        "stabilize_return": float(stabilize_return),
        "ramp_return": float(ramp_return),
        "walk_return": float(walk_return),
        "total_return": float(recovery_return + stabilize_return + ramp_return + walk_return),
    }


def summarize(rows):
    if not rows:
        return {
            "episodes": 0,
            "recovered": 0,
            "stabilized": 0,
            "ramped_to_walk": 0,
            "switched_to_walk": 0,
            "walk_reached_distance": 0,
            "composite_success": 0,
            "fallen": 0,
            "catastrophic": 0,
            "mean_walk_distance": 0.0,
            "mean_walk_vx": 0.0,
            "mean_recovery_steps": 0.0,
            "mean_stabilize_steps": 0.0,
            "mean_ramp_steps": 0.0,
            "mean_walk_steps": 0.0,
            "mean_recovery_reentries": 0.0,
            "mean_initial_upright": 0.0,
            "mean_final_upright": 0.0,
        }
    return {
        "episodes": len(rows),
        "recovered": int(sum(row["recovered"] for row in rows)),
        "stabilized": int(sum(row["stabilized"] for row in rows)),
        "ramped_to_walk": int(sum(row["ramped_to_walk"] for row in rows)),
        "switched_to_walk": int(sum(row["switched_to_walk"] for row in rows)),
        "walk_reached_distance": int(sum(row["walk_reached_distance"] for row in rows)),
        "composite_success": int(sum(row["composite_success"] for row in rows)),
        "fallen": int(sum(row["fallen"] for row in rows)),
        "catastrophic": int(sum(row["catastrophic"] for row in rows)),
        "recovery_rate": float(np.mean([row["recovered"] for row in rows])),
        "stabilize_rate": float(np.mean([row["stabilized"] for row in rows])),
        "ramp_rate": float(np.mean([row["ramped_to_walk"] for row in rows])),
        "composite_success_rate": float(np.mean([row["composite_success"] for row in rows])),
        "mean_walk_distance": float(np.mean([row["walk_distance"] for row in rows])),
        "mean_walk_vx": float(np.mean([row["walk_mean_vx"] for row in rows])),
        "mean_recovery_steps": float(np.mean([row["recovery_steps"] for row in rows])),
        "mean_stabilize_steps": float(np.mean([row["stabilize_steps"] for row in rows])),
        "mean_ramp_steps": float(np.mean([row["ramp_steps"] for row in rows])),
        "mean_walk_steps": float(np.mean([row["walk_steps"] for row in rows])),
        "mean_recovery_reentries": float(np.mean([row["recovery_reentries"] for row in rows])),
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
        "stabilized",
        "ramped_to_walk",
        "switched_to_walk",
        "walk_reached_distance",
        "composite_success",
        "fallen",
        "catastrophic",
        "recovery_rate",
        "stabilize_rate",
        "ramp_rate",
        "composite_success_rate",
        "mean_walk_distance",
        "mean_walk_vx",
        "mean_recovery_steps",
        "mean_stabilize_steps",
        "mean_ramp_steps",
        "mean_walk_steps",
        "mean_recovery_reentries",
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
    parser.add_argument("--recovery-model", default="runs/recovery_stand/ppo_recovery_stand_final.zip")
    parser.add_argument("--recovery-vecnormalize", default="vecnormalize.pkl")
    parser.add_argument("--walk-model", default="runs/walk_after_bc_teacher_v1/ppo_walk_final.zip")
    parser.add_argument("--walk-vecnormalize", default="runs/walk_after_bc_teacher_v1/vecnormalize.pkl")
    parser.add_argument("--model-xml", default="unitree_a1/scene.xml")
    parser.add_argument("--seed-file", default="runs/recovery_seed_benchmark_v3/seeds.csv")
    parser.add_argument("--output-dir", default="runs/recover_then_walk_eval_v1")
    parser.add_argument("--episodes-per-difficulty", type=int, default=30)
    parser.add_argument("--difficulties", type=parse_float_list, default=DEFAULT_DIFFICULTIES)
    parser.add_argument("--seed-bank-seed", type=int, default=20260604)
    parser.add_argument("--recovery-max-steps", type=int, default=1000)
    parser.add_argument("--recover-to-stabilize-good-steps", type=int, default=8)
    parser.add_argument("--stabilize-max-steps", type=int, default=160)
    parser.add_argument("--stabilize-good-steps", type=int, default=15)
    parser.add_argument("--walk-ramp-steps", type=int, default=60)
    parser.add_argument("--walk-ramp-min-action-scale", type=float, default=0.35)
    parser.add_argument("--walk-start-gait-phase", type=float, default=0.0)
    parser.add_argument("--max-recovery-reentries", type=int, default=1)
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
    parser.add_argument("--walk-ready-upright", type=float, default=0.90)
    parser.add_argument("--walk-ready-height-error", type=float, default=0.085)
    parser.add_argument("--walk-ready-tilt", type=float, default=0.35)
    parser.add_argument("--walk-ready-ang-vel", type=float, default=1.5)
    parser.add_argument("--walk-ready-lin-vel", type=float, default=0.45)
    parser.add_argument("--walk-ready-support", type=float, default=3.0)
    parser.add_argument("--recover-upright", type=float, default=0.62)
    parser.add_argument("--recover-low-height-margin", type=float, default=0.18)
    parser.add_argument("--recover-tilt", type=float, default=0.85)
    parser.add_argument("--recover-ang-vel", type=float, default=5.0)
    parser.add_argument("--recover-support", type=float, default=0.0)
    parser.add_argument("--recover-bad-steps", type=int, default=12)
    parser.add_argument("--force-walk-after-stabilize", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.recovery_model = resolve_checkpoint_path(args.recovery_model, "ppo_recovery_stand_final.zip")
    args.walk_model = resolve_checkpoint_path(args.walk_model, "ppo_walk_final.zip")
    args.recovery_vecnormalize = resolve_vecnormalize_path(args.recovery_vecnormalize)
    args.walk_vecnormalize = resolve_vecnormalize_path(args.walk_vecnormalize)

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
        "stabilized",
        "ramped_to_walk",
        "switched_to_walk",
        "walk_reached_distance",
        "composite_success",
        "failure_stage",
        "recovery_steps",
        "stabilize_steps",
        "ramp_steps",
        "walk_steps",
        "total_steps",
        "recovery_reentries",
        "walk_distance",
        "walk_mean_vx",
        "final_upright",
        "final_height_error",
        "final_vx",
        "fallen",
        "catastrophic",
        "recovery_return",
        "stabilize_return",
        "ramp_return",
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
        f"stabilized={aggregate['stabilized']}/{aggregate['episodes']} "
        f"ramped={aggregate['ramped_to_walk']}/{aggregate['episodes']} "
        f"composite_success={aggregate['composite_success']}/{aggregate['episodes']} "
        f"mean_walk_distance={aggregate['mean_walk_distance']:.3f} "
        f"mean_walk_vx={aggregate['mean_walk_vx']:.3f}",
        flush=True,
    )

    recovery_vec.close()
    walk_vec.close()


if __name__ == "__main__":
    main()
