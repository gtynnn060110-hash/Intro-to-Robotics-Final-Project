"""Behavior-cloning warm start from quadruped motion clips.

The first version of this script used a hand-written sinusoidal trot teacher.
That was useful for checking the pipeline, but it produced an animation-like
prior rather than a real gait prior. This version reads DeepMimic-style motion
clips from Google's motion_imitation project and trains the PPO actor mean to
imitate their periodic joint targets.

Run with mjpython on macOS:

    mjpython pretrain_walk_bc.py --motion dog_trot --run-dir runs/walk_bc_trot

Then continue with PPO:

    mjpython train_walk.py \
      --resume-from runs/walk_bc_trot/ppo_walk_bc_pretrained.zip \
      --vecnormalize-load runs/walk_bc_trot/vecnormalize.pkl \
      --run-dir runs/walk_after_bc
"""
import argparse
import json
import os
import urllib.error
import urllib.request
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

try:
    import mujoco
except Exception:  # pragma: no cover
    mujoco = None

from envs import UnitreeA1WalkEnv


MOTION_URLS = {
    "dog_trot": "https://raw.githubusercontent.com/erwincoumans/motion_imitation/master/motion_imitation/data/motions/dog_trot.txt",
    "dog_pace": "https://raw.githubusercontent.com/erwincoumans/motion_imitation/master/motion_imitation/data/motions/dog_pace.txt",
    "dog_walk": "https://raw.githubusercontent.com/erwincoumans/motion_imitation/master/motion_imitation/data/motions/dog_walk.txt",
    "inplace_steps": "https://raw.githubusercontent.com/erwincoumans/motion_imitation/master/motion_imitation/data/motions/inplace_steps.txt",
}


class MotionClip:
    """Periodic root pose + 12-joint clip in DeepMimic JSON format."""

    def __init__(self, frames, frame_duration, retarget_mode="centered", motion_scale=0.75):
        frames = np.asarray(frames, dtype=np.float64)
        if frames.ndim != 2 or frames.shape[1] < 19:
            raise ValueError(f"Expected frames with at least 19 values, got {frames.shape}.")
        self.frames = frames
        self.frame_duration = float(frame_duration)
        self.duration = self.frame_duration * len(self.frames)
        self.root_pos = self.frames[:, 0:3]
        self.root_quat = self.frames[:, 3:7]
        self.joints = self.frames[:, 7:19]
        self.mean_joints = np.mean(self.joints, axis=0)
        self.retarget_mode = str(retarget_mode)
        self.motion_scale = float(motion_scale)

    @classmethod
    def from_file(cls, path, retarget_mode="centered", motion_scale=0.75):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        frame_duration = float(payload.get("FrameDuration", 1.0 / 60.0))
        frames = payload.get("Frames")
        if not frames:
            raise ValueError(f"Motion file has no Frames: {path}")
        return cls(frames, frame_duration, retarget_mode=retarget_mode, motion_scale=motion_scale)

    def sample_joints(self, phase):
        """Return 12 reference joint angles for phase in radians."""
        phase01 = (float(phase) % (2.0 * np.pi)) / (2.0 * np.pi)
        frame_pos = phase01 * len(self.joints)
        i0 = int(np.floor(frame_pos)) % len(self.joints)
        i1 = (i0 + 1) % len(self.joints)
        alpha = frame_pos - np.floor(frame_pos)
        joints = (1.0 - alpha) * self.joints[i0] + alpha * self.joints[i1]
        return joints.astype(np.float64)

    def target_ctrl(self, standing_ctrl, phase, ctrl_low, ctrl_high):
        joints = self.sample_joints(phase)
        if self.retarget_mode == "raw":
            target = joints
        elif self.retarget_mode == "centered":
            # Motion clips are animal/retargeted reference poses, not guaranteed
            # to share our MuJoCo A1's nominal standing offset. Preserve their
            # cyclic variation but center it around the terrain-aware A1 stance.
            target = standing_ctrl + self.motion_scale * (joints - self.mean_joints)
        elif self.retarget_mode == "blend":
            target = (1.0 - self.motion_scale) * standing_ctrl + self.motion_scale * joints
        else:
            raise ValueError(f"Unknown retarget mode: {self.retarget_mode}")
        return np.clip(target, ctrl_low, ctrl_high).astype(np.float32)


def resolve_motion_file(args):
    if args.motion_file:
        path = Path(args.motion_file)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    if args.motion not in MOTION_URLS:
        raise ValueError(f"Unknown motion '{args.motion}'. Choices: {', '.join(sorted(MOTION_URLS))}")

    cache_dir = Path(args.motion_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{args.motion}.txt"
    if path.exists():
        return path

    url = MOTION_URLS[args.motion]
    print(f"[bc] downloading motion clip {args.motion} -> {path}", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            path.write_bytes(response.read())
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            "Could not download the motion clip. Re-run with --motion-file pointing "
            f"to a local DeepMimic motion txt, or allow network access. URL: {url}"
        ) from exc
    return path


def make_env(args):
    return UnitreeA1WalkEnv(
        model_path=args.model,
        target_vx=args.target_vx,
        terrain_friction=args.terrain_friction,
        terrain_height_scale=args.terrain_height_scale,
        max_episode_steps=args.max_episode_steps,
        action_scale=args.action_scale,
        frame_skip=args.frame_skip,
        gait_frequency=args.gait_frequency,
        swing_height=args.swing_height,
        stance_clearance=args.stance_clearance,
        randomize_gait_phase=False,
        normalize_obs=False,
    )


def expert_action(env, clip, phase):
    target = clip.target_ctrl(env.standing_ctrl, phase, env.ctrl_low, env.ctrl_high)
    action = (target - env.standing_ctrl) / max(float(env.action_scale), 1e-8)
    return np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)


def reset_to_motion_phase(env, clip, phase, seed):
    env.reset(seed=seed)
    env.gait_phase = float(phase % (2.0 * np.pi))
    target = clip.target_ctrl(env.standing_ctrl, env.gait_phase, env.ctrl_low, env.ctrl_high)
    env.data.qpos[-env.n_joints :] = target
    env.data.qvel[:] = 0.0
    env.data.ctrl[: env.n_joints] = target
    mujoco.mj_forward(env.model, env.data)
    env._raise_robot_above_terrain(clearance=0.02)
    env.prev_foot_xy = np.array([env.data.geom_xpos[g, :2] for g in env._ordered_foot_geom_ids])
    env.last_action[:] = expert_action(env, clip, env.gait_phase)
    return env._get_obs_raw().copy()


def collect_dataset(args, clip):
    env = make_env(args)
    rng = np.random.default_rng(args.seed)
    observations = []
    actions = []
    try:
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        reset_to_motion_phase(env, clip, phase, args.seed)
        rollout_step = 0

        for i in range(args.samples):
            if rollout_step >= args.rollout_len:
                phase = float(rng.uniform(0.0, 2.0 * np.pi)) if args.random_start_phase else 0.0
                reset_to_motion_phase(env, clip, phase, args.seed + i + 1)
                rollout_step = 0

            obs = env._get_obs_raw().copy()
            action = expert_action(env, clip, env.gait_phase)
            observations.append(obs)
            actions.append(action)

            noisy_action = action
            if args.expert_action_noise > 0.0:
                noisy_action = np.clip(
                    action + rng.normal(0.0, args.expert_action_noise, size=action.shape),
                    env.action_space.low,
                    env.action_space.high,
                ).astype(np.float32)
            _, _, terminated, truncated, _ = env.step(noisy_action)
            rollout_step += 1

            if terminated or truncated:
                phase = float(rng.uniform(0.0, 2.0 * np.pi)) if args.random_start_phase else 0.0
                reset_to_motion_phase(env, clip, phase, args.seed + i + 1000)
                rollout_step = 0
    finally:
        env.close()

    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def make_vecnormalize(args, observations):
    def _init():
        return Monitor(make_env(args))

    vec = VecNormalize(DummyVecEnv([_init]), norm_obs=True, norm_reward=True, clip_obs=10.0)
    vec.obs_rms.update(observations)
    vec.training = False
    vec.norm_reward = False
    return vec


def policy_mean_actions(policy, obs_tensor):
    features = policy.extract_features(obs_tensor)
    if isinstance(features, tuple):
        latent_pi, _ = policy.mlp_extractor(*features)
    else:
        latent_pi, _ = policy.mlp_extractor(features)
    return policy.action_net(latent_pi)


def train_bc(model, observations, actions, epochs, batch_size, learning_rate, seed):
    device = model.policy.device
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate)
    obs = torch.as_tensor(observations, dtype=torch.float32, device=device)
    act = torch.as_tensor(actions, dtype=torch.float32, device=device)
    n = len(observations)

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        losses = []
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            pred = policy_mean_actions(model.policy, obs[idx])
            loss = torch.mean(torch.square(pred - act[idx]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"[bc] epoch={epoch:03d} mse={np.mean(losses):.6f}", flush=True)


def save_run_config(run_dir, args, motion_path, clip, observations, actions):
    summary = {
        "motion_path": str(motion_path),
        "motion_frames": int(len(clip.frames)),
        "motion_frame_duration": clip.frame_duration,
        "motion_duration": clip.duration,
        "retarget_mode": clip.retarget_mode,
        "motion_scale": clip.motion_scale,
        "samples": int(len(observations)),
        "obs_dim": int(observations.shape[1]),
        "action_dim": int(actions.shape[1]),
        "action_mean_abs": float(np.mean(np.abs(actions))),
        "action_max_abs": float(np.max(np.abs(actions))),
        "args": vars(args),
    }
    with open(run_dir / "bc_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unitree_a1/scene.xml")
    parser.add_argument("--run-dir", default="runs/walk_bc_motion")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=30000)
    parser.add_argument("--rollout-len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--motion", default="dog_trot", choices=sorted(MOTION_URLS))
    parser.add_argument("--motion-file", default=None)
    parser.add_argument("--motion-cache-dir", default="data/motions")
    parser.add_argument("--retarget-mode", default="centered", choices=("centered", "raw", "blend"))
    parser.add_argument("--motion-scale", type=float, default=0.75)
    parser.add_argument("--random-start-phase", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expert-action-noise", type=float, default=0.0)
    parser.add_argument("--target-vx", type=float, default=0.2)
    parser.add_argument("--terrain-friction", type=float, default=1.0)
    parser.add_argument("--terrain-height-scale", type=float, default=0.3)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--action-scale", type=float, default=0.5)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--gait-frequency", type=float, default=1.2)
    parser.add_argument("--swing-height", type=float, default=0.045)
    parser.add_argument("--stance-clearance", type=float, default=0.012)
    args = parser.parse_args()

    if mujoco is None:
        raise RuntimeError("mujoco is required; run with mjpython on macOS.")

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    motion_path = resolve_motion_file(args)
    clip = MotionClip.from_file(motion_path, retarget_mode=args.retarget_mode, motion_scale=args.motion_scale)
    print(
        f"[bc] loaded motion={motion_path} frames={len(clip.frames)} "
        f"duration={clip.duration:.3f}s retarget={args.retarget_mode}",
        flush=True,
    )
    print("[bc] collecting expert rollout dataset...", flush=True)
    observations, actions = collect_dataset(args, clip)
    save_run_config(run_dir, args, motion_path, clip, observations, actions)
    vec = make_vecnormalize(args, observations)
    norm_obs = vec.normalize_obs(observations.copy())

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    model = PPO(
        "MlpPolicy",
        vec,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=args.seed,
        n_steps=2048,
        batch_size=512,
        learning_rate=args.learning_rate,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.01,
    )
    train_bc(model, norm_obs, actions, args.epochs, args.batch_size, args.learning_rate, args.seed)

    model.save(str(run_dir / "ppo_walk_bc_pretrained"))
    vec.save(str(run_dir / "vecnormalize.pkl"))
    vec.close()
    print(f"[bc] saved policy={run_dir / 'ppo_walk_bc_pretrained.zip'}", flush=True)
    print(f"[bc] saved vecnormalize={run_dir / 'vecnormalize.pkl'}", flush=True)
    print(f"[bc] saved summary={run_dir / 'bc_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
