"""Minimal demo to launch `UnitreeA1Env` using mjpython.

Run with `mjpython demo_env.py` on macOS or any Python where `mujoco` is available.
"""
import time
import argparse

try:
    import mujoco
except Exception:
    mujoco = None

from envs import UnitreeA1Env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unitree_a1/scene.xml", help="Path to A1 model XML")
    parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps")
    args = parser.parse_args()

    if mujoco is None:
        print("mujoco not importable. Use `mjpython` or install mujoco before running this demo.")
        return

    env = UnitreeA1Env(args.model)
    obs, _ = env.reset()
    print("Reset obs shape:", obs.shape)

    try:
        for i in range(args.steps):
            action = env.action_space.sample() * 0.0  # zero/stand action
            obs, reward, terminated, truncated, info = env.step(action)
            if i % 50 == 0:
                print(f"step={i} reward={reward:.3f} z={info.get('z', None)}")
            # render human viewer if available
            try:
                env.render(mode="human")
            except Exception:
                pass
            time.sleep(1.0 / 60.0)
            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
