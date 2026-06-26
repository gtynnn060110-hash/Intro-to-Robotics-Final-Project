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
from eval_recover_then_walk import (
    adapt_obs_dim,
    enter_recovery_from_current_state,
    is_walk_ready,
    needs_recovery,
    recovery_obs,
    resolve_checkpoint_path,
    start_recovery,
    start_walking_from_current_state,
    supervisor_metrics,
    walk_obs,
)


def current_obs(env, vecnormalize=None):
    obs = env._get_obs_raw()
    if vecnormalize is not None:
        return vecnormalize.normalize_obs(np.asarray(obs, dtype=np.float32).reshape(1, -1))
    if hasattr(env, "_normalize_obs"):
        return env._normalize_obs(obs, update=False)
    if getattr(env, "obs_normalizer", None) is not None:
        return env.obs_normalizer.normalize(obs, update=False)
    return obs


def normalize_for_policy(policy, obs, vecnormalize=None):
    target_dim = int(np.prod(policy.observation_space.shape))
    raw_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    if vecnormalize is None:
        return adapt_obs_dim(raw_obs, target_dim).reshape(1, -1)
    vec_dim = int(np.prod(vecnormalize.observation_space.shape))
    if vec_dim == target_dim:
        obs_for_vec = adapt_obs_dim(raw_obs, vec_dim).reshape(1, -1)
        return vecnormalize.normalize_obs(obs_for_vec.copy())
    if vec_dim == raw_obs.shape[0]:
        normalized = vecnormalize.normalize_obs(raw_obs.reshape(1, -1).copy()).reshape(-1)
        return adapt_obs_dim(normalized, target_dim).reshape(1, -1)
    return adapt_obs_dim(raw_obs, target_dim).reshape(1, -1)


def policy_action(policy, obs, vecnormalize=None, stochastic=False):
    policy_obs = normalize_for_policy(policy, obs, vecnormalize)
    action, _ = policy.predict(policy_obs, deterministic=not stochastic)
    return np.asarray(action, dtype=np.float32).reshape(-1)


def resolve_run_vecnormalize(model_path, explicit_vecnormalize):
    if explicit_vecnormalize:
        path = os.fspath(explicit_vecnormalize)
        if os.path.isdir(path):
            for filename in ("vecnormalize.pkl", "best_vecnormalize.pkl"):
                candidate = os.path.join(path, filename)
                if os.path.exists(candidate):
                    return candidate
            return None
        return path
    model_path = os.fspath(model_path)
    model_dir = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
    for filename in ("vecnormalize.pkl", "best_vecnormalize.pkl"):
        candidate = os.path.join(model_dir, filename)
        if os.path.exists(candidate):
            return candidate
    return None


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


def load_vecnormalize_with_factory(path, env_factory):
    if path is None:
        return None
    vec = VecNormalize.load(path, DummyVecEnv([lambda: Monitor(env_factory())]))
    vec.training = False
    vec.norm_reward = False
    return vec


def make_video_camera(args):
    if not args.video_follow_robot:
        return None
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = float(args.video_camera_distance)
    camera.azimuth = float(args.video_camera_azimuth)
    camera.elevation = float(args.video_camera_elevation)
    return camera


def update_video_camera(env, camera, args):
    if camera is None:
        return None
    trunk_body = getattr(env, "_trunk_body", -1)
    if trunk_body is not None and trunk_body >= 0:
        lookat = np.asarray(env.data.xpos[trunk_body], dtype=np.float64).copy()
    else:
        lookat = np.asarray(env.data.qpos[:3], dtype=np.float64).copy()
    lookat[2] += float(args.video_camera_z_offset)
    camera.lookat[:] = lookat
    camera.distance = float(args.video_camera_distance)
    camera.azimuth = float(args.video_camera_azimuth)
    camera.elevation = float(args.video_camera_elevation)
    return camera


def render_video_frame(renderer, env, args, camera):
    if camera is not None:
        update_video_camera(env, camera, args)
        renderer.update_scene(env.data, camera=camera)
    else:
        renderer.update_scene(env.data)
    return renderer.render().copy()


def make_recover_walk_env(args):
    return UnitreeA1WalkEnv(
        model_path=args.model,
        target_vx=args.target_vx,
        reset_noise=0.0,
        terrain_friction=args.terrain_friction,
        terrain_height_scale=args.terrain_height_scale,
        max_episode_steps=args.recovery_max_steps,
        recovery_difficulty=args.recovery_level,
        action_scale=args.recovery_action_scale,
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


def make_recovery_vec_env(args):
    return UnitreeA1Env(
        args.model,
        task="recovery",
        recovery_difficulty=args.recovery_level,
        max_episode_steps=args.recovery_max_steps,
        action_scale=args.recovery_action_scale,
        normalize_obs=False,
        frame_skip=args.frame_skip,
    )


def make_walk_vec_env(args):
    return UnitreeA1WalkEnv(
        model_path=args.model,
        target_vx=args.target_vx,
        reset_noise=0.0,
        terrain_friction=args.terrain_friction,
        terrain_height_scale=args.terrain_height_scale,
        max_episode_steps=args.walk_max_steps,
        action_scale=args.walk_action_scale,
        frame_skip=args.frame_skip,
        gait_frequency=args.gait_frequency,
        swing_height=args.swing_height,
        stance_clearance=args.stance_clearance,
        normalize_obs=False,
    )


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


def euler_from_quat(quat):
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / max(np.linalg.norm(quat), 1e-8)
    w, x, y, z = quat
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def align_robot_yaw(env, target_yaw=0.0):
    """Rotate only heading for demos, preserving roll/pitch and body-frame velocity."""
    old_quat = np.asarray(env.data.qpos[3:7], dtype=np.float64).copy()
    old_rot = rotmat_from_quat(old_quat)
    roll, pitch, old_yaw = euler_from_quat(old_quat)
    new_quat = env._quat_from_euler(roll, pitch, float(target_yaw))
    new_rot = rotmat_from_quat(new_quat)

    body_lin_vel = old_rot.T @ np.asarray(env.data.qvel[:3], dtype=np.float64)
    body_ang_vel = old_rot.T @ np.asarray(env.data.qvel[3:6], dtype=np.float64)
    env.data.qpos[3:7] = new_quat
    env.data.qvel[:3] = new_rot @ body_lin_vel
    env.data.qvel[3:6] = new_rot @ body_ang_vel
    mujoco.mj_forward(env.model, env.data)
    return old_yaw


def heading_normalized_walk_obs(env):
    """Walk observation in a yaw-aligned policy frame, without changing physics."""
    obs = walk_obs(env).copy()
    if obs.shape[0] < 10:
        return obs

    quat = np.asarray(env.data.qpos[3:7], dtype=np.float64)
    roll, pitch, yaw = euler_from_quat(quat)
    obs[0:4] = env._quat_from_euler(roll, pitch, 0.0).astype(np.float32)

    cy, sy = math.cos(-yaw), math.sin(-yaw)
    yaw_inv = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    obs[4:7] = (yaw_inv @ np.asarray(env.data.qvel[3:6], dtype=np.float64)).astype(np.float32)
    obs[7:10] = (yaw_inv @ np.asarray(env.data.qvel[:3], dtype=np.float64)).astype(np.float32)
    return obs


def apply_initial_pose_override(env, args):
    if args.initial_pose == "recovery":
        return

    yaw = float(args.initial_heading_yaw)
    if args.initial_pose == "stand":
        roll = 0.0
        pitch = 0.0
        base_height = env.nominal_base_clearance + args.initial_height_offset
    elif args.initial_pose == "tilt":
        roll = math.radians(args.initial_roll_deg)
        pitch = math.radians(args.initial_pitch_deg)
        base_height = env.nominal_base_clearance + args.initial_height_offset
    elif args.initial_pose == "side_left":
        roll = math.radians(85.0)
        pitch = 0.0
        base_height = max(0.16, env.nominal_base_clearance - 0.10 + args.initial_height_offset)
    elif args.initial_pose == "side_right":
        roll = math.radians(-85.0)
        pitch = 0.0
        base_height = max(0.16, env.nominal_base_clearance - 0.10 + args.initial_height_offset)
    elif args.initial_pose == "upside_down":
        roll = math.radians(180.0)
        pitch = 0.0
        base_height = max(0.16, env.nominal_base_clearance - 0.13 + args.initial_height_offset)
    else:
        raise ValueError(f"Unsupported initial pose: {args.initial_pose}")

    terrain_z = env._terrain_height_under_base()
    env.data.qpos[2] = terrain_z + base_height
    env.data.qpos[3:7] = env._quat_from_euler(roll, pitch, yaw)
    env.data.qpos[-env.n_joints :] = np.clip(env.standing_ctrl, env.ctrl_low, env.ctrl_high)
    env.data.qvel[:] = 0.0
    env.data.ctrl[: env.n_joints] = env.standing_ctrl
    mujoco.mj_forward(env.model, env.data)
    env._raise_robot_above_terrain(clearance=0.025)
    env._reset_progress_trackers()


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


def run_recover_walk_demo(args):
    recovery_model_path = resolve_checkpoint_path(args.recovery_model, "ppo_recovery_stand_final.zip")
    walk_model_path = resolve_checkpoint_path(args.walk_model, "ppo_walk_final.zip")
    recovery_vec_path = resolve_run_vecnormalize(args.recovery_model, args.recovery_vecnormalize)
    walk_vec_path = resolve_run_vecnormalize(args.walk_model, args.walk_vecnormalize)

    if not os.path.exists(recovery_model_path):
        raise FileNotFoundError(f"Recovery checkpoint not found: {recovery_model_path}")
    if not os.path.exists(walk_model_path):
        raise FileNotFoundError(f"Walk checkpoint not found: {walk_model_path}")
    if recovery_vec_path is not None and not os.path.exists(recovery_vec_path):
        raise FileNotFoundError(f"Recovery VecNormalize not found: {recovery_vec_path}")
    if walk_vec_path is not None and not os.path.exists(walk_vec_path):
        raise FileNotFoundError(f"Walk VecNormalize not found: {walk_vec_path}")

    env = make_recover_walk_env(args)
    recovery_policy = PPO.load(recovery_model_path, device="cpu")
    walk_policy = PPO.load(walk_model_path, device="cpu")
    recovery_vec = load_vecnormalize_with_factory(recovery_vec_path, lambda: make_recovery_vec_env(args))
    walk_vec = load_vecnormalize_with_factory(walk_vec_path, lambda: make_walk_vec_env(args))

    print("Composite visual demo: RECOVER -> STABILIZE -> WALK_RAMP -> WALK")
    print(f"Recovery policy: {recovery_model_path}")
    print(f"Recovery vecnormalize: {recovery_vec_path or 'None (same run dir did not contain vecnormalize.pkl)'}")
    print(f"Walk policy: {walk_model_path}")
    print(f"Walk vecnormalize: {walk_vec_path or 'None'}")
    print("Render hint: Ctrl + left mouse can disturb/drag the robot in the MuJoCo viewer.")

    obs, info = start_recovery(env, args.seed if args.seed is not None else 0, args.recovery_level, args)
    apply_initial_pose_override(env, args)
    if args.align_initial_heading:
        old_yaw = align_robot_yaw(env, args.initial_heading_yaw)
        obs = recovery_obs(env)
        print(
            f"initial heading aligned before demo: yaw {math.degrees(old_yaw):.1f} -> "
            f"{math.degrees(args.initial_heading_yaw):.1f} deg"
        )
    if args.initial_pose != "recovery":
        obs = recovery_obs(env)
        metrics = supervisor_metrics(env)
        print(
            f"initial pose override: {args.initial_pose} "
            f"upright={metrics['upright']:.3f} h_err={metrics['height_error']:.3f} "
            f"support={metrics['support_count']:.1f}"
        )
    mode = "recover"
    if args.initial_pose == "stand" and args.start_walk_if_ready and is_walk_ready(supervisor_metrics(env), args):
        mode = "walk_ramp" if args.walk_ramp_steps > 0 else "walk"
        obs = start_walking_from_current_state(env, args, initial_target_vx=0.0, last_recovery_action=None)
    good_steps = 0
    bad_steps = 0
    recovery_steps = 0
    stabilize_steps = 0
    ramp_steps = 0
    walk_steps = 0
    recovery_reentries = 0
    last_recovery_action = np.zeros(env.n_joints, dtype=np.float32)
    last_info = dict(info)

    renderer = None
    video_camera = None
    video_frames = []
    video_capture_interval = 1
    if args.record_video is not None:
        env.model.vis.global_.offwidth = max(int(env.model.vis.global_.offwidth), int(args.video_width))
        env.model.vis.global_.offheight = max(int(env.model.vis.global_.offheight), int(args.video_height))
        renderer = mujoco.Renderer(env.model, height=args.video_height, width=args.video_width)
        video_camera = make_video_camera(args)
        dt = float(env.model.opt.timestep * env.frame_skip)
        video_capture_interval = max(1, int(round(1.0 / max(dt * args.video_fps, 1e-8))))
        video_frames.append(render_video_frame(renderer, env, args, video_camera))

    try:
        for i in range(args.steps):
            if mode == "recover":
                action = policy_action(recovery_policy, obs, recovery_vec, stochastic=args.stochastic)
                last_recovery_action = action.copy()
                _, reward, terminated, truncated, info = UnitreeA1Env.step(env, action)
                obs = recovery_obs(env)
                recovery_steps += 1
                last_info = dict(info)
                metrics = supervisor_metrics(env)
                good_steps = good_steps + 1 if is_walk_ready(metrics, args) else 0
                if bool(info.get("recovered", False)) or good_steps >= args.recover_to_stabilize_good_steps:
                    mode = "stabilize"
                    env.current_task = "stand"
                    env.max_episode_steps = int(args.stabilize_max_steps)
                    env.action_scale = float(args.recovery_action_scale)
                    good_steps = 0
                    bad_steps = 0
                elif bool(info.get("catastrophic", False) or info.get("failure_timeout", False) or terminated or truncated):
                    if args.no_auto_reset:
                        pass
                    else:
                        obs, info = start_recovery(env, args.seed if args.seed is not None else 0, args.recovery_level, args)
                        mode = "recover"
                        recovery_steps = stabilize_steps = ramp_steps = walk_steps = 0
                        recovery_reentries = 0
                        last_info = dict(info)

            elif mode == "stabilize":
                action = policy_action(recovery_policy, obs, recovery_vec, stochastic=args.stochastic)
                last_recovery_action = action.copy()
                _, reward, terminated, truncated, info = UnitreeA1Env.step(env, action)
                obs = recovery_obs(env)
                stabilize_steps += 1
                last_info = dict(info)
                metrics = supervisor_metrics(env)
                if needs_recovery(metrics, args):
                    mode = "recover"
                    obs = enter_recovery_from_current_state(env, args)
                    good_steps = 0
                else:
                    good_steps = good_steps + 1 if is_walk_ready(metrics, args) else 0
                if good_steps >= args.stabilize_good_steps:
                    mode = "walk_ramp" if args.walk_ramp_steps > 0 else "walk"
                    if args.align_walk_heading:
                        old_yaw = align_robot_yaw(env, args.walk_heading_yaw)
                        print(
                            f"align walk heading: yaw {math.degrees(old_yaw):.1f} -> "
                            f"{math.degrees(args.walk_heading_yaw):.1f} deg"
                        )
                    obs = start_walking_from_current_state(
                        env,
                        args,
                        initial_target_vx=0.0,
                        last_recovery_action=last_recovery_action,
                    )
                    bad_steps = 0
                elif stabilize_steps >= args.stabilize_max_steps:
                    metrics = supervisor_metrics(env)
                    if args.force_walk_after_stabilize and not needs_recovery(metrics, args):
                        mode = "walk_ramp" if args.walk_ramp_steps > 0 else "walk"
                        if args.align_walk_heading:
                            old_yaw = align_robot_yaw(env, args.walk_heading_yaw)
                            print(
                                f"force walk after stabilize: yaw {math.degrees(old_yaw):.1f} -> "
                                f"{math.degrees(args.walk_heading_yaw):.1f} deg"
                            )
                        obs = start_walking_from_current_state(
                            env,
                            args,
                            initial_target_vx=0.0,
                            last_recovery_action=last_recovery_action,
                        )
                        bad_steps = 0
                    else:
                        mode = "recover"
                        obs = enter_recovery_from_current_state(env, args)
                        good_steps = 0

            elif mode == "walk_ramp":
                alpha = min((ramp_steps + 1) / float(max(args.walk_ramp_steps, 1)), 1.0)
                env.set_target_vx(args.target_vx * alpha)
                env.action_scale = float(args.walk_action_scale) * (
                    args.walk_ramp_min_action_scale + alpha * (1.0 - args.walk_ramp_min_action_scale)
                )
                policy_obs = heading_normalized_walk_obs(env) if args.walk_policy_heading_frame else obs
                walk_action = policy_action(walk_policy, policy_obs, walk_vec, stochastic=args.stochastic)
                action = (1.0 - alpha) * last_recovery_action + alpha * walk_action
                obs, reward, terminated, truncated, info = UnitreeA1WalkEnv.step(env, action)
                ramp_steps += 1
                last_info = dict(info)
                metrics = supervisor_metrics(env)
                bad_steps = bad_steps + 1 if needs_recovery(metrics, args) else 0
                if bad_steps >= args.recover_bad_steps:
                    if recovery_reentries < args.max_recovery_reentries:
                        recovery_reentries += 1
                        mode = "recover"
                        obs = enter_recovery_from_current_state(env, args)
                        bad_steps = 0
                    else:
                        mode = "walk"
                elif bool(info.get("fallen", False) or info.get("catastrophic", False) or terminated):
                    mode = "recover"
                    obs = enter_recovery_from_current_state(env, args)
                elif ramp_steps >= args.walk_ramp_steps:
                    mode = "walk"
                    env.set_target_vx(args.target_vx)
                    env.action_scale = float(args.walk_action_scale)
                    bad_steps = 0

            else:  # walk
                env.set_target_vx(args.target_vx)
                env.action_scale = float(args.walk_action_scale)
                policy_obs = heading_normalized_walk_obs(env) if args.walk_policy_heading_frame else obs
                action = policy_action(walk_policy, policy_obs, walk_vec, stochastic=args.stochastic)
                obs, reward, terminated, truncated, info = UnitreeA1WalkEnv.step(env, action)
                walk_steps += 1
                last_info = dict(info)
                metrics = supervisor_metrics(env)
                bad_steps = bad_steps + 1 if needs_recovery(metrics, args) else 0
                if bad_steps >= args.recover_bad_steps:
                    if recovery_reentries < args.max_recovery_reentries:
                        recovery_reentries += 1
                        mode = "recover"
                        obs = enter_recovery_from_current_state(env, args)
                        bad_steps = 0
                    elif not args.no_auto_reset:
                        obs, info = start_recovery(env, args.seed if args.seed is not None else 0, args.recovery_level, args)
                        mode = "recover"
                        recovery_steps = stabilize_steps = ramp_steps = walk_steps = 0
                        recovery_reentries = 0
                        last_info = dict(info)
                elif bool(info.get("fallen", False) or info.get("catastrophic", False) or terminated):
                    mode = "recover"
                    obs = enter_recovery_from_current_state(env, args)

            if i % args.print_interval == 0:
                metrics = supervisor_metrics(env)
                print(
                    f"step={i} mode={mode} upright={metrics['upright']:.3f} "
                    f"h_err={metrics['height_error']:.3f} vx={metrics['vx']:.3f} "
                    f"support={metrics['support_count']:.1f} good={good_steps} bad={bad_steps} "
                    f"reentries={recovery_reentries} recovered={last_info.get('recovered', False)}"
                )
            if renderer is not None and i % video_capture_interval == 0:
                video_frames.append(render_video_frame(renderer, env, args, video_camera))
            if args.render:
                try:
                    env.render(mode="human")
                except Exception:
                    pass
                time.sleep(1.0 / 60.0)
    finally:
        if renderer is not None:
            try:
                write_video(video_frames, args.record_video, args.video_fps)
                print(f"视频已导出: {args.record_video}")
            finally:
                renderer.close()
        if recovery_vec is not None:
            recovery_vec.close()
        if walk_vec is not None:
            walk_vec.close()
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unitree_a1/scene.xml", help="Path to A1 model XML")
    parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps")
    parser.add_argument("--task", choices=["stand", "recovery", "walk", "recover_walk"], default="stand")
    parser.add_argument("--checkpoint", default=None, help="Optional PPO checkpoint .zip to inspect")
    parser.add_argument("--vecnormalize", default=None, help="Optional VecNormalize .pkl saved with the PPO checkpoint")
    parser.add_argument("--recovery-model", default="runs/recovery_stand", help="Recovery checkpoint .zip or run directory")
    parser.add_argument(
        "--recovery-vecnormalize",
        default=None,
        help="Recovery VecNormalize .pkl or run directory. Default: only auto-load vecnormalize.pkl from --recovery-model directory.",
    )
    parser.add_argument("--walk-model", default="runs/walk_after_bc_teacher_v1", help="Walk checkpoint .zip or run directory")
    parser.add_argument(
        "--walk-vecnormalize",
        default="runs/walk_after_bc_teacher_v1",
        help="Walk VecNormalize .pkl or run directory. Default: same run directory as the walk checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Reset seed for reproducible recovery poses")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of deterministic policy output")
    parser.add_argument("--no-auto-reset", action="store_true", help="Keep stepping after termination/truncation")
    parser.add_argument("--recovery-level", type=float, default=1.0, help="Recovery reset difficulty in [0, 1]")
    parser.add_argument(
        "--initial-pose",
        choices=["recovery", "stand", "tilt", "side_left", "side_right", "upside_down"],
        default="recovery",
        help="Override the recover_walk initial pose after reset.",
    )
    parser.add_argument("--initial-roll-deg", type=float, default=18.0)
    parser.add_argument("--initial-pitch-deg", type=float, default=10.0)
    parser.add_argument("--initial-height-offset", type=float, default=0.0)
    parser.add_argument(
        "--start-walk-if-ready",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For --initial-pose stand, enter walking ramp immediately if the pose is walk-ready.",
    )
    parser.add_argument("--recovery-max-steps", type=int, default=1000)
    parser.add_argument("--recover-to-stabilize-good-steps", type=int, default=8)
    parser.add_argument("--stabilize-max-steps", type=int, default=160)
    parser.add_argument("--stabilize-good-steps", type=int, default=15)
    parser.add_argument("--walk-ramp-steps", type=int, default=60)
    parser.add_argument("--walk-ramp-min-action-scale", type=float, default=0.35)
    parser.add_argument("--walk-start-gait-phase", type=float, default=0.0)
    parser.add_argument(
        "--align-walk-heading",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For the composite demo, physically align body yaw before entering the walking policy.",
    )
    parser.add_argument(
        "--walk-policy-heading-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Yaw-normalize observations for the walking policy without rotating the simulated robot.",
    )
    parser.add_argument(
        "--align-initial-heading",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For recover_walk demo, align yaw once before the first rendered step.",
    )
    parser.add_argument("--initial-heading-yaw", type=float, default=0.0, help="Initial demo yaw in radians.")
    parser.add_argument("--walk-heading-yaw", type=float, default=0.0, help="Target yaw in radians used by --align-walk-heading")
    parser.add_argument("--max-recovery-reentries", type=int, default=1)
    parser.add_argument("--walk-max-steps", type=int, default=500)
    parser.add_argument("--recovery-action-scale", type=float, default=0.5)
    parser.add_argument("--walk-action-scale", type=float, default=0.5)
    parser.add_argument("--success-steps", type=int, default=15)
    parser.add_argument("--failure-steps", type=int, default=180)
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
    parser.add_argument(
        "--force-walk-after-stabilize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enter WALK_RAMP after stabilize timeout if the robot is not in a recovery-needed state.",
    )
    parser.add_argument("--print-interval", type=int, default=50)
    parser.add_argument("--use-trot-reference", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--diagnose-yaw-slip", action="store_true", help="Compare zero-action drift at yaw=0 and yaw=180")
    parser.add_argument("--render", action="store_true", help="Render when used with diagnostic modes")
    parser.add_argument("--record-video", default=None, help="Export yaw-slip diagnostic video to an .mp4 path")
    parser.add_argument("--video-fps", type=int, default=30, help="FPS for --record-video")
    parser.add_argument("--video-width", type=int, default=1280, help="Video width for --record-video")
    parser.add_argument("--video-height", type=int, default=720, help="Video height for --record-video")
    parser.add_argument(
        "--video-follow-robot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep exported video centered on the robot trunk.",
    )
    parser.add_argument("--video-camera-distance", type=float, default=2.2)
    parser.add_argument("--video-camera-azimuth", type=float, default=135.0)
    parser.add_argument("--video-camera-elevation", type=float, default=-18.0)
    parser.add_argument("--video-camera-z-offset", type=float, default=0.15)
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
    if args.task == "recover_walk":
        run_recover_walk_demo(args)
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
    video_camera = None
    video_frames = []
    video_capture_interval = 1
    if args.record_video is not None:
        env.model.vis.global_.offwidth = max(int(env.model.vis.global_.offwidth), int(args.video_width))
        env.model.vis.global_.offheight = max(int(env.model.vis.global_.offheight), int(args.video_height))
        renderer = mujoco.Renderer(env.model, height=args.video_height, width=args.video_width)
        video_camera = make_video_camera(args)
        dt = float(env.model.opt.timestep * env.frame_skip)
        video_capture_interval = max(1, int(round(1.0 / max(dt * args.video_fps, 1e-8))))
        video_frames.append(render_video_frame(renderer, env, args, video_camera))
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
                video_frames.append(render_video_frame(renderer, env, args, video_camera))
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
                    video_frames.append(render_video_frame(renderer, env, args, video_camera))
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
