"""Minimal demo to launch `UnitreeA1Env` using mjpython.

Run with `mjpython demo_env.py` on macOS or any Python where `mujoco` is available.
"""
import time
import argparse
import os

try:
    import mujoco
except Exception:
    mujoco = None

from stable_baselines3 import PPO

from envs import UnitreeA1Env


def current_obs(env):
    obs = env._get_obs_raw()
    return env.obs_normalizer.normalize(obs, update=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unitree_a1/scene.xml", help="Path to A1 model XML")
    parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps")
    parser.add_argument("--task", choices=["stand", "recovery"], default="stand")
    parser.add_argument("--checkpoint", default=None, help="Optional PPO checkpoint .zip to inspect")
    parser.add_argument("--seed", type=int, default=None, help="Reset seed for reproducible recovery poses")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of deterministic policy output")
    parser.add_argument("--no-auto-reset", action="store_true", help="Keep stepping after termination/truncation")
    args = parser.parse_args()

    if mujoco is None:
        print("mujoco not importable. Use `mjpython` or install mujoco before running this demo.")
        return
    if args.checkpoint is not None and not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    env = UnitreeA1Env(args.model, task=args.task)
    policy = PPO.load(args.checkpoint, env=env) if args.checkpoint is not None else None
    obs, info = env.reset(seed=args.seed)
    print("Reset obs shape:", obs.shape)
    print("Reset info:", info)
    if policy is None:
        print("Control mode: zero action")
    else:
        print(f"Control mode: PPO checkpoint={args.checkpoint}")
        print(f"Action mode: {'stochastic' if args.stochastic else 'deterministic'}")

    try:
        for i in range(args.steps):
            if policy is None:
                action = env.action_space.sample() * 0.0  # zero/stand action
            else:
                obs = current_obs(env)
                action, _ = policy.predict(obs, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, info = env.step(action)
            if i % 50 == 0:
                print(
                    f"step={i} reward={reward:.3f} z={info.get('z', None):.3f} "
                    f"upright={info.get('upright', 0.0):.3f} "
                    f"height_error={info.get('height_error', 0.0):.3f} "
                    f"fallen={info.get('fallen', False)}"
                )
            # render human viewer if available
            try:
                env.render(mode="human")
            except Exception:
                pass
            time.sleep(1.0 / 60.0)
            if (terminated or truncated) and not args.no_auto_reset:
                print(f"episode ended at step={i} terminated={terminated} truncated={truncated}")
                obs, info = env.reset(seed=args.seed)
                print("Reset info:", info)
    finally:
        env.close()


if __name__ == "__main__":
    main()
