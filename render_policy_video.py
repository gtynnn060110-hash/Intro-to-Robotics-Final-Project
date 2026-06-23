"""Render MuJoCo policy demos to MP4 for slides."""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs import UnitreeA1Env, UnitreeA1WalkEnv


def load_vecnormalize(path, env_fn):
    if not path:
        return None
    vec = VecNormalize.load(path, DummyVecEnv([env_fn]))
    vec.training = False
    vec.norm_reward = False
    return vec


def normalize_obs(obs, vec):
    obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
    if vec is None:
        return obs
    return vec.normalize_obs(obs.copy())


def make_env(args):
    if args.task == "walk":
        env = UnitreeA1WalkEnv(
            model_path=args.model_xml,
            target_vx=args.target_vx,
            reset_noise=args.reset_noise,
            terrain_friction=args.terrain_friction,
            terrain_height_scale=args.terrain_height_scale,
            max_episode_steps=args.steps,
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
            overspeed_deadband=args.overspeed_deadband,
            overspeed_weight=args.overspeed_weight,
            overspeed_quadratic_weight=args.overspeed_quadratic_weight,
            forward_reward_weight=args.forward_reward_weight,
            progress_reward_weight=args.progress_reward_weight,
            backward_penalty_weight=args.backward_penalty_weight,
            low_speed_penalty_weight=args.low_speed_penalty_weight,
            low_speed_fraction=args.low_speed_fraction,
            speed_reward_sharpness=args.speed_reward_sharpness,
            use_trot_reference=args.use_trot_reference,
            trot_frequency=args.trot_frequency,
            trot_thigh_amplitude=args.trot_thigh_amplitude,
            trot_calf_amplitude=args.trot_calf_amplitude,
            trot_stance_calf_amplitude=args.trot_stance_calf_amplitude,
            normalize_obs=False,
        )
        obs, info = env.reset(seed=args.seed)
    else:
        env = UnitreeA1Env(
            args.model_xml,
            task="recovery",
            recovery_difficulty=args.recovery_difficulty,
            max_episode_steps=args.steps,
            action_scale=args.action_scale,
            success_steps=args.success_steps,
            failure_steps=args.failure_steps,
            normalize_obs=False,
        )
        obs, info = env.reset(seed=args.seed, options={"task": "recovery"})
    return env, obs, info


def make_vec_env_fn(args):
    def _init():
        env, _, _ = make_env(args)
        return Monitor(env)

    return _init


def set_camera(camera, env, args):
    x, y, z = env.data.qpos[:3]
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [float(x) + args.camera_x_offset, float(y), float(z) + args.camera_z_offset]
    camera.distance = args.camera_distance
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation


def render_video(args):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    env, obs, info = make_env(args)
    vec = load_vecnormalize(args.vecnormalize, make_vec_env_fn(args))
    model = None if args.zero_action else PPO.load(args.checkpoint, device="cpu")
    env.model.vis.global_.offwidth = max(int(env.model.vis.global_.offwidth), int(args.width))
    env.model.vis.global_.offheight = max(int(env.model.vis.global_.offheight), int(args.height))
    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    dt = env.model.opt.timestep * env.frame_skip
    render_every = max(int(round(1.0 / max(args.fps * args.slowdown * dt, 1e-8))), 1)

    last_info = dict(info)
    frames_written = 0
    stable_hold_count = 0
    hold_zero = False
    try:
        with imageio.get_writer(str(output), fps=args.fps, codec="libx264", quality=8, macro_block_size=16) as writer:
            for step in range(args.steps):
                if args.hold_zero_after_stable and not hold_zero:
                    upright = float(last_info.get("upright", 0.0))
                    height_error = abs(float(last_info.get("height_error", 999.0)))
                    if upright >= args.hold_upright and height_error <= args.hold_height_error:
                        stable_hold_count += 1
                    else:
                        stable_hold_count = 0
                    hold_zero = stable_hold_count >= args.hold_stable_steps

                if args.zero_action or hold_zero:
                    action = np.zeros(env.action_space.shape, dtype=np.float32)
                else:
                    policy_obs = normalize_obs(obs, vec)
                    action, _ = model.predict(policy_obs, deterministic=True)
                    action = np.asarray(action, dtype=np.float32).reshape(env.action_space.shape)

                obs, _, _, _, last_info = env.step(action)

                if step % render_every == 0:
                    set_camera(camera, env, args)
                    renderer.update_scene(env.data, camera=camera)
                    writer.append_data(renderer.render())
                    frames_written += 1
    finally:
        renderer.close()
        env.close()
        if vec is not None:
            vec.close()

    print(
        f"saved={output} frames={frames_written} "
        f"final_upright={float(last_info.get('upright', 0.0)):.3f} "
        f"distance={float(last_info.get('distance', 0.0)):.3f} "
        f"recovered={bool(last_info.get('recovered', False))} "
        f"survived={bool(last_info.get('survived', False))} "
        f"hold_zero={hold_zero}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["recovery", "walk"], required=True)
    parser.add_argument("--model-xml", default="unitree_a1/scene.xml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--vecnormalize", default=None)
    parser.add_argument("--zero-action", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--slowdown", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
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
    parser.add_argument("--recovery-difficulty", type=float, default=0.5)
    parser.add_argument("--success-steps", type=int, default=15)
    parser.add_argument("--failure-steps", type=int, default=180)
    parser.add_argument("--target-vx", type=float, default=0.2)
    parser.add_argument("--reset-noise", type=float, default=0.0)
    parser.add_argument("--terrain-friction", type=float, default=1.5)
    parser.add_argument("--terrain-height-scale", type=float, default=0.3)
    parser.add_argument("--overspeed-deadband", type=float, default=0.02)
    parser.add_argument("--overspeed-weight", type=float, default=8.0)
    parser.add_argument("--overspeed-quadratic-weight", type=float, default=20.0)
    parser.add_argument("--forward-reward-weight", type=float, default=3.0)
    parser.add_argument("--progress-reward-weight", type=float, default=5.0)
    parser.add_argument("--backward-penalty-weight", type=float, default=6.0)
    parser.add_argument("--low-speed-penalty-weight", type=float, default=40.0)
    parser.add_argument("--low-speed-fraction", type=float, default=1.0)
    parser.add_argument("--speed-reward-sharpness", type=float, default=20.0)
    parser.add_argument("--use-trot-reference", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trot-frequency", type=float, default=1.35)
    parser.add_argument("--trot-thigh-amplitude", type=float, default=0.22)
    parser.add_argument("--trot-calf-amplitude", type=float, default=0.32)
    parser.add_argument("--trot-stance-calf-amplitude", type=float, default=0.08)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--camera-x-offset", type=float, default=0.25)
    parser.add_argument("--camera-z-offset", type=float, default=0.15)
    parser.add_argument("--hold-zero-after-stable", action="store_true")
    parser.add_argument("--hold-upright", type=float, default=0.90)
    parser.add_argument("--hold-height-error", type=float, default=0.12)
    parser.add_argument("--hold-stable-steps", type=int, default=8)
    args = parser.parse_args()

    if not args.zero_action and not args.checkpoint:
        parser.error("--checkpoint is required unless --zero-action is set")
    render_video(args)


if __name__ == "__main__":
    main()
