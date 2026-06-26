"""Evaluate fixed-speed walking policies."""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs import UnitreeA1WalkEnv
from train_walk import INFO_KEYWORDS


def make_eval_env(
    model_path,
    seed,
    max_episode_steps,
    target_vx,
    reset_noise,
    action_scale,
    terrain_friction,
    terrain_height_scale,
    overspeed_deadband,
    overspeed_weight,
    overspeed_quadratic_weight,
    forward_reward_weight,
    progress_reward_weight,
    backward_penalty_weight,
    low_speed_penalty_weight,
    low_speed_fraction,
    speed_reward_sharpness,
    upright_reward_weight,
    height_reward_weight,
    height_reward_sharpness,
    height_target_offset,
    low_height_penalty_weight,
    low_height_penalty_quadratic_weight,
    lateral_penalty_weight,
    yaw_penalty_weight,
    ang_vel_penalty_weight,
    joint_vel_penalty_weight,
    pose_penalty_weight,
    action_penalty_weight,
    smooth_penalty_weight,
    frame_skip,
    gait_frequency,
    swing_height,
    stance_clearance,
    foot_contact_weight,
    swing_clearance_weight,
    stance_slip_weight,
    gait_symmetry_weight,
    gait_clock_obs,
    use_trot_reference,
    trot_frequency,
    trot_thigh_amplitude,
    trot_calf_amplitude,
    trot_stance_calf_amplitude,
):
    def _init():
        env = UnitreeA1WalkEnv(
            model_path=model_path,
            target_vx=target_vx,
            reset_noise=reset_noise,
            terrain_friction=terrain_friction,
            terrain_height_scale=terrain_height_scale,
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
            upright_reward_weight=upright_reward_weight,
            height_reward_weight=height_reward_weight,
            height_reward_sharpness=height_reward_sharpness,
            height_target_offset=height_target_offset,
            low_height_penalty_weight=low_height_penalty_weight,
            low_height_penalty_quadratic_weight=low_height_penalty_quadratic_weight,
            lateral_penalty_weight=lateral_penalty_weight,
            yaw_penalty_weight=yaw_penalty_weight,
            ang_vel_penalty_weight=ang_vel_penalty_weight,
            joint_vel_penalty_weight=joint_vel_penalty_weight,
            pose_penalty_weight=pose_penalty_weight,
            action_penalty_weight=action_penalty_weight,
            smooth_penalty_weight=smooth_penalty_weight,
            frame_skip=frame_skip,
            gait_frequency=gait_frequency,
            swing_height=swing_height,
            stance_clearance=stance_clearance,
            foot_contact_weight=foot_contact_weight,
            swing_clearance_weight=swing_clearance_weight,
            stance_slip_weight=stance_slip_weight,
            gait_symmetry_weight=gait_symmetry_weight,
            gait_clock_obs=gait_clock_obs,
            use_trot_reference=use_trot_reference,
            trot_frequency=trot_frequency,
            trot_thigh_amplitude=trot_thigh_amplitude,
            trot_calf_amplitude=trot_calf_amplitude,
            trot_stance_calf_amplitude=trot_stance_calf_amplitude,
            normalize_obs=False,
        )
        env.reset(seed=seed)
        info_keywords = tuple(
            key for key in INFO_KEYWORDS if not key.startswith("teacher_") and key != "reward_teacher_action"
        )
        return Monitor(env, info_keywords=info_keywords)

    return _init


def load_eval_env(args, seed):
    env = DummyVecEnv(
        [
            make_eval_env(
                model_path=args.model_xml,
                seed=seed,
                max_episode_steps=args.max_episode_steps,
                target_vx=args.target_vx,
                reset_noise=args.reset_noise,
                action_scale=args.action_scale,
                terrain_friction=args.terrain_friction,
                terrain_height_scale=args.terrain_height_scale,
                overspeed_deadband=args.overspeed_deadband,
                overspeed_weight=args.overspeed_weight,
                overspeed_quadratic_weight=args.overspeed_quadratic_weight,
                forward_reward_weight=args.forward_reward_weight,
                progress_reward_weight=args.progress_reward_weight,
                backward_penalty_weight=args.backward_penalty_weight,
                low_speed_penalty_weight=args.low_speed_penalty_weight,
                low_speed_fraction=args.low_speed_fraction,
                speed_reward_sharpness=args.speed_reward_sharpness,
                upright_reward_weight=args.upright_reward_weight,
                height_reward_weight=args.height_reward_weight,
                height_reward_sharpness=args.height_reward_sharpness,
                height_target_offset=args.height_target_offset,
                low_height_penalty_weight=args.low_height_penalty_weight,
                low_height_penalty_quadratic_weight=args.low_height_penalty_quadratic_weight,
                lateral_penalty_weight=args.lateral_penalty_weight,
                yaw_penalty_weight=args.yaw_penalty_weight,
                ang_vel_penalty_weight=args.ang_vel_penalty_weight,
                joint_vel_penalty_weight=args.joint_vel_penalty_weight,
                pose_penalty_weight=args.pose_penalty_weight,
                action_penalty_weight=args.action_penalty_weight,
                smooth_penalty_weight=args.smooth_penalty_weight,
                frame_skip=args.frame_skip,
                gait_frequency=args.gait_frequency,
                swing_height=args.swing_height,
                stance_clearance=args.stance_clearance,
                foot_contact_weight=args.foot_contact_weight,
                swing_clearance_weight=args.swing_clearance_weight,
                stance_slip_weight=args.stance_slip_weight,
                gait_symmetry_weight=args.gait_symmetry_weight,
                gait_clock_obs=args.gait_clock_obs,
                use_trot_reference=args.use_trot_reference,
                trot_frequency=args.trot_frequency,
                trot_thigh_amplitude=args.trot_thigh_amplitude,
                trot_calf_amplitude=args.trot_calf_amplitude,
                trot_stance_calf_amplitude=args.trot_stance_calf_amplitude,
            )
        ]
    )
    env = VecNormalize.load(args.vecnormalize, env)
    env.training = False
    env.norm_reward = False
    return env


def run_episode(model, env):
    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    last_info = {}
    upright_values = []
    height_errors = []
    vx_values = []
    vy_values = []
    yaw_rate_values = []
    speed_errors = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
        done = bool(dones[0])
        total_reward += float(reward[0])
        steps += 1
        last_info = dict(infos[0])
        if "upright" in last_info:
            upright_values.append(float(last_info["upright"]))
        if "height_error" in last_info:
            height_errors.append(float(last_info["height_error"]))
        if "vx" in last_info:
            vx_values.append(float(last_info["vx"]))
        if "vy" in last_info:
            vy_values.append(float(last_info["vy"]))
        if "yaw_rate" in last_info:
            yaw_rate_values.append(float(last_info["yaw_rate"]))
        if "speed_error" in last_info:
            speed_errors.append(float(last_info["speed_error"]))
    return {
        "reward": total_reward,
        "steps": steps,
        "survived": bool(last_info.get("survived", False)),
        "fallen": bool(last_info.get("fallen", False)),
        "catastrophic": bool(last_info.get("catastrophic", False)),
        "distance": float(last_info.get("distance", 0.0)),
        "mean_vx": float(last_info.get("mean_vx_episode", 0.0)),
        "mean_step_vx": float(np.mean(vx_values)) if vx_values else 0.0,
        "final_vx": float(last_info.get("vx", 0.0)),
        "mean_abs_vy": float(np.mean(np.abs(vy_values))) if vy_values else abs(float(last_info.get("vy", 0.0))),
        "final_vy": float(last_info.get("vy", 0.0)),
        "mean_abs_yaw_rate": (
            float(np.mean(np.abs(yaw_rate_values))) if yaw_rate_values else abs(float(last_info.get("yaw_rate", 0.0)))
        ),
        "final_yaw_rate": float(last_info.get("yaw_rate", 0.0)),
        "mean_upright": float(np.mean(upright_values)) if upright_values else float(last_info.get("upright", 0.0)),
        "final_upright": float(last_info.get("upright", 0.0)),
        "mean_height_error": float(np.mean(height_errors)) if height_errors else float(last_info.get("height_error", 0.0)),
        "height_error": float(last_info.get("height_error", 0.0)),
        "mean_abs_speed_error": (
            float(np.mean(np.abs(speed_errors))) if speed_errors else abs(float(last_info.get("speed_error", 0.0)))
        ),
        "speed_error": float(last_info.get("speed_error", 0.0)),
    }


def summarize(rows):
    count = len(rows)
    if count == 0:
        return {}
    return {
        "episodes": count,
        "survived": int(sum(row["survived"] for row in rows)),
        "fallen": int(sum(row["fallen"] for row in rows)),
        "survive_rate": float(np.mean([row["survived"] for row in rows])),
        "fall_rate": float(np.mean([row["fallen"] for row in rows])),
        "mean_reward": float(np.mean([row["reward"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_distance": float(np.mean([row["distance"] for row in rows])),
        "mean_forward_velocity": float(np.mean([row["mean_vx"] for row in rows])),
        "mean_step_vx": float(np.mean([row["mean_step_vx"] for row in rows])),
        "mean_final_vx": float(np.mean([row["final_vx"] for row in rows])),
        "mean_abs_vy": float(np.mean([row["mean_abs_vy"] for row in rows])),
        "mean_abs_yaw_rate": float(np.mean([row["mean_abs_yaw_rate"] for row in rows])),
        "mean_upright": float(np.mean([row["mean_upright"] for row in rows])),
        "mean_final_upright": float(np.mean([row["final_upright"] for row in rows])),
        "mean_height_error": float(np.mean([row["mean_height_error"] for row in rows])),
        "mean_abs_height_error": float(np.mean([abs(row["mean_height_error"]) for row in rows])),
        "mean_abs_speed_error": float(np.mean([row["mean_abs_speed_error"] for row in rows])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--vecnormalize", required=True)
    parser.add_argument("--model-xml", default="unitree_a1/scene.xml")
    parser.add_argument("--output-dir", default="runs/walk_eval")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=70_000)
    parser.add_argument("--target-vx", type=float, default=0.2)
    parser.add_argument("--reset-noise", type=float, default=0.0)
    parser.add_argument("--action-scale", type=float, default=0.5)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--gait-frequency", type=float, default=1.15)
    parser.add_argument("--swing-height", type=float, default=0.055)
    parser.add_argument("--stance-clearance", type=float, default=0.012)
    parser.add_argument("--foot-contact-weight", type=float, default=1.2)
    parser.add_argument("--swing-clearance-weight", type=float, default=1.0)
    parser.add_argument("--stance-slip-weight", type=float, default=0.25)
    parser.add_argument("--gait-symmetry-weight", type=float, default=0.15)
    parser.add_argument("--gait-clock-obs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--terrain-friction", type=float, default=1.5)
    parser.add_argument("--terrain-height-scale", type=float, default=0.3)
    parser.add_argument("--overspeed-deadband", type=float, default=0.02)
    parser.add_argument("--overspeed-weight", type=float, default=8.0)
    parser.add_argument("--overspeed-quadratic-weight", type=float, default=20.0)
    parser.add_argument("--forward-reward-weight", type=float, default=3.0)
    parser.add_argument("--progress-reward-weight", type=float, default=1.5)
    parser.add_argument("--backward-penalty-weight", type=float, default=2.0)
    parser.add_argument("--low-speed-penalty-weight", type=float, default=0.0)
    parser.add_argument("--low-speed-fraction", type=float, default=0.6)
    parser.add_argument("--speed-reward-sharpness", type=float, default=4.0)
    parser.add_argument("--upright-reward-weight", type=float, default=1.5)
    parser.add_argument("--height-reward-weight", type=float, default=1.5)
    parser.add_argument("--height-reward-sharpness", type=float, default=30.0)
    parser.add_argument("--height-target-offset", type=float, default=0.0)
    parser.add_argument("--low-height-penalty-weight", type=float, default=0.0)
    parser.add_argument("--low-height-penalty-quadratic-weight", type=float, default=0.0)
    parser.add_argument("--lateral-penalty-weight", type=float, default=1.0)
    parser.add_argument("--yaw-penalty-weight", type=float, default=0.20)
    parser.add_argument("--ang-vel-penalty-weight", type=float, default=0.04)
    parser.add_argument("--joint-vel-penalty-weight", type=float, default=0.003)
    parser.add_argument("--pose-penalty-weight", type=float, default=0.03)
    parser.add_argument("--action-penalty-weight", type=float, default=0.004)
    parser.add_argument("--smooth-penalty-weight", type=float, default=0.010)
    parser.add_argument("--use-trot-reference", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trot-frequency", type=float, default=1.35)
    parser.add_argument("--trot-thigh-amplitude", type=float, default=0.22)
    parser.add_argument("--trot-calf-amplitude", type=float, default=0.32)
    parser.add_argument("--trot-stance-calf-amplitude", type=float, default=0.08)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(max(int(args.torch_threads), 1))
    torch.set_num_interop_threads(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    model = None
    env = None
    try:
        env = load_eval_env(args, args.seed)
        model = PPO.load(args.model, env=env, device="cpu")
        for episode in range(args.episodes):
            env.seed(args.seed + episode)
            row = run_episode(model, env)
            row["episode"] = episode
            row["seed"] = args.seed + episode
            rows.append(row)
    finally:
        if env is not None:
            env.close()

    summary = summarize(rows)
    summary.update(
        {
            "model": args.model,
            "vecnormalize": args.vecnormalize,
            "target_vx": args.target_vx,
            "reset_noise": args.reset_noise,
            "action_scale": args.action_scale,
            "terrain_friction": args.terrain_friction,
            "terrain_height_scale": args.terrain_height_scale,
            "overspeed_deadband": args.overspeed_deadband,
            "overspeed_weight": args.overspeed_weight,
            "overspeed_quadratic_weight": args.overspeed_quadratic_weight,
            "forward_reward_weight": args.forward_reward_weight,
            "progress_reward_weight": args.progress_reward_weight,
            "backward_penalty_weight": args.backward_penalty_weight,
            "low_speed_penalty_weight": args.low_speed_penalty_weight,
            "low_speed_fraction": args.low_speed_fraction,
            "speed_reward_sharpness": args.speed_reward_sharpness,
            "upright_reward_weight": args.upright_reward_weight,
            "height_reward_weight": args.height_reward_weight,
            "height_reward_sharpness": args.height_reward_sharpness,
            "height_target_offset": args.height_target_offset,
            "low_height_penalty_weight": args.low_height_penalty_weight,
            "low_height_penalty_quadratic_weight": args.low_height_penalty_quadratic_weight,
            "lateral_penalty_weight": args.lateral_penalty_weight,
            "yaw_penalty_weight": args.yaw_penalty_weight,
            "ang_vel_penalty_weight": args.ang_vel_penalty_weight,
            "joint_vel_penalty_weight": args.joint_vel_penalty_weight,
            "pose_penalty_weight": args.pose_penalty_weight,
            "action_penalty_weight": args.action_penalty_weight,
            "smooth_penalty_weight": args.smooth_penalty_weight,
            "frame_skip": args.frame_skip,
            "gait_frequency": args.gait_frequency,
            "swing_height": args.swing_height,
            "stance_clearance": args.stance_clearance,
            "foot_contact_weight": args.foot_contact_weight,
            "swing_clearance_weight": args.swing_clearance_weight,
            "stance_slip_weight": args.stance_slip_weight,
            "gait_symmetry_weight": args.gait_symmetry_weight,
            "gait_clock_obs": args.gait_clock_obs,
            "use_trot_reference": args.use_trot_reference,
            "trot_frequency": args.trot_frequency,
            "trot_thigh_amplitude": args.trot_thigh_amplitude,
            "trot_calf_amplitude": args.trot_calf_amplitude,
            "trot_stance_calf_amplitude": args.trot_stance_calf_amplitude,
            "max_episode_steps": args.max_episode_steps,
        }
    )

    with (output_dir / "episodes.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()) if rows else ["episode"])
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
