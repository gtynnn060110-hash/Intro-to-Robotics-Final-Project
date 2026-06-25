"""Minimal demo to launch `UnitreeA1Env` using mjpython.

Run with `mjpython demo_env.py` on macOS or any Python where `mujoco` is available.
"""
import time
import argparse
import os
import math

import numpy as np

try:
    import mujoco
except Exception:
    mujoco = None

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs import UnitreeA1Env, UnitreeA1WalkEnv


def current_obs(env, vecnormalize=None):
    obs = env._get_obs_raw()
    if vecnormalize is not None:
        return vecnormalize.normalize_obs(np.asarray(obs, dtype=np.float32).reshape(1, -1))
    if hasattr(env, "_normalize_obs"):
        return env._normalize_obs(obs, update=False)
    if getattr(env, "obs_normalizer", None) is not None:
        return env.obs_normalizer.normalize(obs, update=False)
    return obs


def make_demo_env(args):
    if args.task == "walk":
        return UnitreeA1WalkEnv(
            model_path=args.model,
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
            stance_slip_weight=args.stance_slip_weight,
            support_clearance_threshold=args.support_clearance_threshold,
            min_support_contacts=args.min_support_contacts,
            support_penalty_weight=args.support_penalty_weight,
            rear_air_penalty_weight=args.rear_air_penalty_weight,
            stance_contact_penalty_weight=args.stance_contact_penalty_weight,
            lateral_penalty_weight=args.lateral_penalty_weight,
            yaw_penalty_weight=args.yaw_penalty_weight,
            pitch_tilt_penalty_weight=args.pitch_tilt_penalty_weight,
            roll_tilt_penalty_weight=args.roll_tilt_penalty_weight,
            use_trot_reference=args.use_trot_reference,
            normalize_obs=False,
        )
    return UnitreeA1Env(
        args.model,
        task=args.task,
        recovery_difficulty=args.recovery_level,
        max_episode_steps=args.steps,
        action_scale=args.action_scale,
        normalize_obs=False,
    )


def load_vecnormalize(path, args):
    if path is None:
        return None

    def _init():
        return Monitor(make_demo_env(args))

    vec = VecNormalize.load(path, DummyVecEnv([_init]))
    vec.training = False
    vec.norm_reward = False
    return vec


def quat_from_yaw(yaw):
    return np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)], dtype=np.float64)


def rotmat_from_quat(quat):
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / max(np.linalg.norm(quat), 1e-8)
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def run_yaw_slip_diagnosis(model_path, steps, seed, terrain_friction, stance_forward_offset):
    for yaw in (0.0, math.pi):
        env = UnitreeA1Env(
            model_path,
            task="stand",
            max_episode_steps=steps + 10,
            terrain_friction=terrain_friction,
            stance_forward_offset=stance_forward_offset,
        )
        start_xy, start_z, start_terrain, start_forward = setup_yaw_case(env, yaw, seed)

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        for _ in range(steps):
            env.step(action)

        end_xy = env.data.qpos[:2].copy()
        end_z = float(env.data.qpos[2])
        end_terrain = env._raycast_terrain_height(end_xy, end_z)
        world_disp = end_xy - start_xy
        body_forward_disp = float(np.dot(world_disp, start_forward[:2]))
        terrain_delta = float(end_terrain - start_terrain)
        body_vel = rotmat_from_quat(env.data.qpos[3:7]).T @ env.data.qvel[:3]

        print(f"\n=== yaw={math.degrees(yaw):.1f} deg ===")
        print(f"start_xy={np.round(start_xy, 5)} end_xy={np.round(end_xy, 5)}")
        print(f"world_disp_xy={np.round(world_disp, 5)}")
        print(f"body_forward_disp={body_forward_disp:.5f}  # positive means toward robot front")
        print(
            f"terrain_start={start_terrain:.5f} terrain_end={end_terrain:.5f} "
            f"terrain_delta={terrain_delta:.5f}"
        )
        print(f"z_start={start_z:.5f} z_end={end_z:.5f}")
        print(f"final_world_vel={np.round(env.data.qvel[:3], 5)}")
        print(f"final_body_vel={np.round(body_vel, 5)}")
        env.close()


def setup_yaw_case(env, yaw, seed):
    env.reset(seed=seed)
    mujoco.mj_resetData(env.model, env.data)
    if getattr(env, "_home_key", None) is not None and env._home_key >= 0:
        mujoco.mj_resetDataKeyframe(env.model, env.data, env._home_key)

    env.data.qpos[3:7] = quat_from_yaw(yaw)
    base_pos = env.data.qpos[:3].copy()
    terrain_z = env._raycast_terrain_height(base_pos[:2], base_pos[2])
    if terrain_z is not None:
        env.data.qpos[2] = terrain_z + env.nominal_base_clearance
    mujoco.mj_forward(env.model, env.data)

    env.standing_ctrl = env._solve_standing_ik()
    env.data.qpos[-env.n_joints :] = env.standing_ctrl
    env.data.qvel[:] = 0.0
    env.data.ctrl[: env.n_joints] = env.standing_ctrl
    mujoco.mj_forward(env.model, env.data)
    if hasattr(env, "_reset_progress_trackers"):
        env._reset_progress_trackers()

    start_xy = env.data.qpos[:2].copy()
    start_z = float(env.data.qpos[2])
    start_terrain = env._raycast_terrain_height(start_xy, start_z)
    start_forward = rotmat_from_quat(env.data.qpos[3:7])[:, 0]
    return start_xy, start_z, start_terrain, start_forward


def print_yaw_case_result(env, yaw, start_xy, start_z, start_terrain, start_forward):
    end_xy = env.data.qpos[:2].copy()
    end_z = float(env.data.qpos[2])
    end_terrain = env._raycast_terrain_height(end_xy, end_z)
    world_disp = end_xy - start_xy
    body_forward_disp = float(np.dot(world_disp, start_forward[:2]))
    terrain_delta = float(end_terrain - start_terrain)
    body_vel = rotmat_from_quat(env.data.qpos[3:7]).T @ env.data.qvel[:3]

    print(f"\n=== yaw={math.degrees(yaw):.1f} deg ===")
    print(f"start_xy={np.round(start_xy, 5)} end_xy={np.round(end_xy, 5)}")
    print(f"world_disp_xy={np.round(world_disp, 5)}")
    print(f"body_forward_disp={body_forward_disp:.5f}  # positive means toward robot front")
    print(
        f"terrain_start={start_terrain:.5f} terrain_end={end_terrain:.5f} "
        f"terrain_delta={terrain_delta:.5f}"
    )
    print(f"z_start={start_z:.5f} z_end={end_z:.5f}")
    print(f"final_world_vel={np.round(env.data.qvel[:3], 5)}")
    print(f"final_body_vel={np.round(body_vel, 5)}")


def write_video(frames, output_path, fps):
    if not frames:
        raise RuntimeError("No frames were captured for video export.")

    output_path = os.fspath(output_path)
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    try:
        import imageio.v2 as imageio

        with imageio.get_writer(output_path, fps=fps, macro_block_size=1) as writer:
            for frame in frames:
                writer.append_data(frame)
        return
    except Exception:
        pass

    try:
        try:
            from moviepy import ImageSequenceClip
        except Exception:
            from moviepy.editor import ImageSequenceClip

        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(output_path, fps=fps, codec="libx264", audio=False)
    except Exception as exc:
        raise RuntimeError("Could not export video. Install imageio[ffmpeg] or moviepy.") from exc


def run_yaw_slip_video_diagnosis(
    model_path,
    steps,
    seed,
    terrain_friction,
    stance_forward_offset,
    output_path,
    fps,
    width,
    height,
):
    env = UnitreeA1Env(
        model_path,
        task="stand",
        max_episode_steps=steps + 10,
        terrain_friction=terrain_friction,
        stance_forward_offset=stance_forward_offset,
    )
    env.model.vis.global_.offwidth = max(int(env.model.vis.global_.offwidth), int(width))
    env.model.vis.global_.offheight = max(int(env.model.vis.global_.offheight), int(height))
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    frames = []
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    capture_interval = max(1, int(round(1.0 / max(env.model.opt.timestep * env.frame_skip * fps, 1e-8))))
    hold_frames = max(1, fps // 2)

    try:
        for yaw in (0.0, math.pi):
            case = setup_yaw_case(env, yaw, seed)
            print(f"\nRecording yaw={math.degrees(yaw):.1f} deg for {steps} steps...")
            for _ in range(hold_frames):
                renderer.update_scene(env.data)
                frames.append(renderer.render().copy())

            for step_idx in range(steps):
                env.step(action)
                if step_idx % capture_interval == 0:
                    renderer.update_scene(env.data)
                    frames.append(renderer.render().copy())

            print_yaw_case_result(env, yaw, *case)
            for _ in range(hold_frames):
                renderer.update_scene(env.data)
                frames.append(renderer.render().copy())

        write_video(frames, output_path, fps)
        print(f"\n视频已导出: {output_path}")
    finally:
        renderer.close()
        env.close()


def run_yaw_slip_render_diagnosis(model_path, steps, seed, terrain_friction, stance_forward_offset):
    import mujoco.viewer

    env = UnitreeA1Env(
        model_path,
        task="stand",
        max_episode_steps=steps + 10,
        terrain_friction=terrain_friction,
        stance_forward_offset=stance_forward_offset,
    )
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    print("可视化诊断：先显示 yaw=0，再显示 yaw=180。每段结束后会打印位移和地形高度变化。")

    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            for yaw in (0.0, math.pi):
                case = setup_yaw_case(env, yaw, seed)
                print(f"\nRunning yaw={math.degrees(yaw):.1f} deg for {steps} steps...")
                for _ in range(steps):
                    if not viewer.is_running():
                        return
                    env.step(action)
                    viewer.sync()
                    time.sleep(env.model.opt.timestep * env.frame_skip)
                print_yaw_case_result(env, yaw, *case)
                time.sleep(1.0)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unitree_a1/scene.xml", help="Path to A1 model XML")
    parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps")
    parser.add_argument("--task", choices=["stand", "recovery", "walk"], default="stand")
    parser.add_argument("--checkpoint", default=None, help="Optional PPO checkpoint .zip to inspect")
    parser.add_argument("--vecnormalize", default=None, help="Optional VecNormalize .pkl saved with the PPO checkpoint")
    parser.add_argument("--seed", type=int, default=None, help="Reset seed for reproducible recovery poses")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of deterministic policy output")
    parser.add_argument("--no-auto-reset", action="store_true", help="Keep stepping after termination/truncation")
    parser.add_argument("--recovery-level", type=float, default=1.0, help="Recovery reset difficulty in [0, 1]")
    parser.add_argument("--target-vx", type=float, default=0.2, help="Target walking velocity for --task walk")
    parser.add_argument("--reset-noise", type=float, default=0.0, help="Reset perturbation for --task walk")
    parser.add_argument("--terrain-height-scale", type=float, default=0.3, help="Terrain hfield height scale for --task walk")
    parser.add_argument("--action-scale", type=float, default=0.5, help="Joint target action scale")
    parser.add_argument("--frame-skip", type=int, default=4, help="MuJoCo steps per policy step")
    parser.add_argument("--gait-frequency", type=float, default=0.95)
    parser.add_argument("--swing-height", type=float, default=0.045)
    parser.add_argument("--stance-clearance", type=float, default=0.012)
    parser.add_argument("--stance-slip-weight", type=float, default=0.25)
    parser.add_argument("--support-clearance-threshold", type=float, default=0.035)
    parser.add_argument("--min-support-contacts", type=float, default=2.0)
    parser.add_argument("--support-penalty-weight", type=float, default=5.0)
    parser.add_argument("--rear-air-penalty-weight", type=float, default=8.0)
    parser.add_argument("--stance-contact-penalty-weight", type=float, default=2.0)
    parser.add_argument("--lateral-penalty-weight", type=float, default=2.5)
    parser.add_argument("--yaw-penalty-weight", type=float, default=0.6)
    parser.add_argument("--pitch-tilt-penalty-weight", type=float, default=8.0)
    parser.add_argument("--roll-tilt-penalty-weight", type=float, default=5.0)
    parser.add_argument("--use-trot-reference", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--diagnose-yaw-slip", action="store_true", help="Compare zero-action drift at yaw=0 and yaw=180")
    parser.add_argument("--render", action="store_true", help="Render when used with diagnostic modes")
    parser.add_argument("--record-video", default=None, help="Export yaw-slip diagnostic video to an .mp4 path")
    parser.add_argument("--video-fps", type=int, default=30, help="FPS for --record-video")
    parser.add_argument("--video-width", type=int, default=1280, help="Video width for --record-video")
    parser.add_argument("--video-height", type=int, default=720, help="Video height for --record-video")
    parser.add_argument("--terrain-friction", type=float, default=0.12, help="Sliding friction for terrain/robot contacts")
    parser.add_argument(
        "--stance-forward-offset",
        type=float,
        default=0.0,
        help="Forward foot-target offset used by standing IK, in meters",
    )
    args = parser.parse_args()

    if mujoco is None:
        print("mujoco not importable. Use `mjpython` or install mujoco before running this demo.")
        return
    if args.diagnose_yaw_slip:
        if args.record_video is not None:
            run_yaw_slip_video_diagnosis(
                args.model,
                args.steps,
                args.seed,
                args.terrain_friction,
                args.stance_forward_offset,
                args.record_video,
                args.video_fps,
                args.video_width,
                args.video_height,
            )
        elif args.render:
            run_yaw_slip_render_diagnosis(
                args.model,
                args.steps,
                args.seed,
                args.terrain_friction,
                args.stance_forward_offset,
            )
        else:
            run_yaw_slip_diagnosis(
                args.model,
                args.steps,
                args.seed,
                args.terrain_friction,
                args.stance_forward_offset,
            )
        return
    if args.checkpoint is not None and not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.vecnormalize is not None and not os.path.exists(args.vecnormalize):
        raise FileNotFoundError(f"VecNormalize file not found: {args.vecnormalize}")

    env = make_demo_env(args)
    vecnormalize = load_vecnormalize(args.vecnormalize, args)
    policy = PPO.load(args.checkpoint, device="cpu") if args.checkpoint is not None else None
    obs, info = env.reset(seed=args.seed)
    print("Reset obs shape:", obs.shape)
    print("Reset info:", info)
    if policy is None:
        print("Control mode: zero action")
    else:
        print(f"Control mode: PPO checkpoint={args.checkpoint}")
        print(f"Action mode: {'stochastic' if args.stochastic else 'deterministic'}")

    renderer = None
    video_frames = []
    video_capture_interval = 1
    if args.record_video is not None:
        env.model.vis.global_.offwidth = max(int(env.model.vis.global_.offwidth), int(args.video_width))
        env.model.vis.global_.offheight = max(int(env.model.vis.global_.offheight), int(args.video_height))
        renderer = mujoco.Renderer(env.model, height=args.video_height, width=args.video_width)
        dt = float(env.model.opt.timestep * env.frame_skip)
        video_capture_interval = max(1, int(round(1.0 / max(dt * args.video_fps, 1e-8))))
        renderer.update_scene(env.data)
        video_frames.append(renderer.render().copy())
        print(
            f"Recording video to {args.record_video} "
            f"({args.video_width}x{args.video_height}@{args.video_fps}fps)",
            flush=True,
        )

    try:
        for i in range(args.steps):
            if policy is None:
                action = env.action_space.sample() * 0.0  # zero/stand action
            else:
                obs = current_obs(env, vecnormalize)
                action, _ = policy.predict(obs, deterministic=not args.stochastic)
                action = np.asarray(action, dtype=np.float32).reshape(env.action_space.shape)
            obs, reward, terminated, truncated, info = env.step(action)
            if i % 50 == 0:
                print(
                    f"step={i} reward={reward:.3f} z={info.get('z', None):.3f} "
                    f"upright={info.get('upright', 0.0):.3f} "
                    f"height_error={info.get('height_error', 0.0):.3f} "
                    f"vx={info.get('vx', 0.0):.3f} "
                    f"pitch={info.get('pitch_tilt', 0.0):.3f} "
                    f"roll={info.get('roll_tilt', 0.0):.3f} "
                    f"stance_slip={info.get('stance_slip', 0.0):.3f} "
                    f"support={info.get('support_count', 0.0):.1f} "
                    f"rear={info.get('rear_contact_count', 0.0):.1f} "
                    f"contact={info.get('foot_contact_match', 0.0):.3f} "
                    f"fallen={info.get('fallen', False)}"
                )
            if renderer is not None and i % video_capture_interval == 0:
                renderer.update_scene(env.data)
                video_frames.append(renderer.render().copy())
            if args.render:
                try:
                    env.render(mode="human")
                except Exception:
                    pass
                time.sleep(1.0 / 60.0)
            if (terminated or truncated) and not args.no_auto_reset:
                print(f"episode ended at step={i} terminated={terminated} truncated={truncated}")
                obs, info = env.reset(seed=args.seed)
                print("Reset info:", info)
                if renderer is not None:
                    renderer.update_scene(env.data)
                    video_frames.append(renderer.render().copy())
    finally:
        if renderer is not None:
            try:
                write_video(video_frames, args.record_video, args.video_fps)
                print(f"视频已导出: {args.record_video}")
            finally:
                renderer.close()
        if vecnormalize is not None:
            vecnormalize.close()
        env.close()


if __name__ == "__main__":
    main()
