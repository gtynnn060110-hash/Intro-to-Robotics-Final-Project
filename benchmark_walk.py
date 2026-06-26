"""Benchmark walking checkpoints across scenes and friction settings."""
import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
from stable_baselines3 import PPO

from eval_walk import load_eval_env, run_episode, summarize


DEFAULT_CHECKPOINTS = [
    {
        "name": "bc_dog_trot_v3",
        "model": "runs/walk_bc_dog_trot_v3/ppo_walk_bc_pretrained.zip",
        "vecnormalize": "runs/walk_bc_dog_trot_v3/vecnormalize.pkl",
    },
    {
        "name": "ppo_after_bc_teacher_v1",
        "model": "runs/walk_after_bc_teacher_v1/ppo_walk_final.zip",
        "vecnormalize": "runs/walk_after_bc_teacher_v1/vecnormalize.pkl",
    },
]

DEFAULT_TERRAINS = [
    {"name": "wave_h030", "model_xml": "unitree_a1/scene.xml", "height_scale": 0.30},
    {"name": "irregular_h030", "model_xml": "unitree_a1/scene_irregular.xml", "height_scale": 0.30},
    {"name": "irregular_h040", "model_xml": "unitree_a1/scene_irregular.xml", "height_scale": 0.40},
]


def parse_json_or_file(value, default):
    if value is None:
        return default
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(value)


def parse_float_list(value):
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def make_eval_args(args, checkpoint, terrain, friction):
    return SimpleNamespace(
        model=checkpoint["model"],
        vecnormalize=checkpoint["vecnormalize"],
        model_xml=terrain["model_xml"],
        output_dir=None,
        episodes=args.episodes,
        seed=args.seed,
        target_vx=args.target_vx,
        reset_noise=args.reset_noise,
        action_scale=args.action_scale,
        frame_skip=args.frame_skip,
        gait_frequency=args.gait_frequency,
        swing_height=args.swing_height,
        stance_clearance=args.stance_clearance,
        foot_contact_weight=args.foot_contact_weight,
        swing_clearance_weight=args.swing_clearance_weight,
        stance_slip_weight=args.stance_slip_weight,
        gait_symmetry_weight=args.gait_symmetry_weight,
        gait_clock_obs=args.gait_clock_obs,
        terrain_friction=float(friction),
        terrain_height_scale=float(terrain["height_scale"]),
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
        use_trot_reference=args.use_trot_reference,
        trot_frequency=args.trot_frequency,
        trot_thigh_amplitude=args.trot_thigh_amplitude,
        trot_calf_amplitude=args.trot_calf_amplitude,
        trot_stance_calf_amplitude=args.trot_stance_calf_amplitude,
        max_episode_steps=args.max_episode_steps,
        torch_threads=args.torch_threads,
    )


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def add_summary_metadata(summary, checkpoint, terrain, friction, episodes):
    summary.update(
        {
            "checkpoint_name": checkpoint["name"],
            "checkpoint": checkpoint["model"],
            "vecnormalize": checkpoint["vecnormalize"],
            "terrain_name": terrain["name"],
            "model_xml": terrain["model_xml"],
            "terrain_height_scale": float(terrain["height_scale"]),
            "terrain_friction": float(friction),
            "episodes": int(episodes),
        }
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/walk_benchmark_checkpoint_scene_friction_v1")
    parser.add_argument("--checkpoints", default=None, help="JSON string or JSON file; defaults to current two walking runs")
    parser.add_argument("--terrains", default=None, help="JSON string or JSON file; defaults to wave/irregular terrain set")
    parser.add_argument("--frictions", default="1.0,0.8,0.7")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=120000)
    parser.add_argument("--target-vx", type=float, default=0.2)
    parser.add_argument("--reset-noise", type=float, default=0.0)
    parser.add_argument("--action-scale", type=float, default=0.5)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--gait-frequency", type=float, default=1.2)
    parser.add_argument("--swing-height", type=float, default=0.045)
    parser.add_argument("--stance-clearance", type=float, default=0.012)
    parser.add_argument("--foot-contact-weight", type=float, default=1.2)
    parser.add_argument("--swing-clearance-weight", type=float, default=1.0)
    parser.add_argument("--stance-slip-weight", type=float, default=0.25)
    parser.add_argument("--gait-symmetry-weight", type=float, default=0.15)
    parser.add_argument("--gait-clock-obs", action=argparse.BooleanOptionalAction, default=True)
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

    checkpoints = parse_json_or_file(args.checkpoints, DEFAULT_CHECKPOINTS)
    terrains = parse_json_or_file(args.terrains, DEFAULT_TERRAINS)
    frictions = parse_float_list(args.frictions)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_config.json").write_text(
        json.dumps(
            {
                "checkpoints": checkpoints,
                "terrains": terrains,
                "frictions": frictions,
                "episodes": args.episodes,
                "seed": args.seed,
                "target_vx": args.target_vx,
                "gait_frequency": args.gait_frequency,
            },
            indent=2,
            sort_keys=True,
        )
    )

    per_episode_rows = []
    summary_rows = []
    for checkpoint in checkpoints:
        print(f"[checkpoint] {checkpoint['name']}", flush=True)
        model = PPO.load(checkpoint["model"], device="cpu")
        for terrain in terrains:
            for friction in frictions:
                eval_args = make_eval_args(args, checkpoint, terrain, friction)
                env = None
                rows = []
                try:
                    env = load_eval_env(eval_args, args.seed)
                    for episode in range(args.episodes):
                        env.seed(args.seed + episode)
                        row = run_episode(model, env)
                        row.update(
                            {
                                "checkpoint_name": checkpoint["name"],
                                "terrain_name": terrain["name"],
                                "terrain_friction": float(friction),
                                "terrain_height_scale": float(terrain["height_scale"]),
                                "episode": int(episode),
                                "seed": int(args.seed + episode),
                            }
                        )
                        rows.append(row)
                        per_episode_rows.append(row)
                finally:
                    if env is not None:
                        env.close()

                summary = add_summary_metadata(summarize(rows), checkpoint, terrain, friction, args.episodes)
                summary_rows.append(summary)
                print(
                    "[summary] "
                    f"checkpoint={checkpoint['name']} terrain={terrain['name']} friction={float(friction):.2f} "
                    f"survive={summary['survived']}/{summary['episodes']} "
                    f"vx={summary['mean_forward_velocity']:.3f} "
                    f"reward={summary['mean_reward']:.1f} "
                    f"height_abs={summary['mean_abs_height_error']:.3f}",
                    flush=True,
                )

    write_csv(output_dir / "per_episode.csv", per_episode_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2, sort_keys=True))
    print(f"[done] wrote {output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
