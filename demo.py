"""Interactive MuJoCo demo aligned with the UnitreeA1Env reset/control logic.

Run with `mjpython demo.py` on macOS.
"""
import argparse
import os
import time

import numpy as np
import mujoco.viewer
from stable_baselines3 import PPO

from envs import UnitreeA1Env


def current_obs(env):
    obs = env._get_obs_raw()
    return env.obs_normalizer.normalize(obs, update=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unitree_a1/scene.xml", help="Path to A1 scene XML")
    parser.add_argument("--task", choices=["stand", "recovery"], default="stand")
    parser.add_argument("--checkpoint", default=None, help="Optional PPO checkpoint .zip to inspect")
    parser.add_argument("--seed", type=int, default=None, help="Reset seed for reproducible recovery poses")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of deterministic policy output")
    parser.add_argument("--auto-reset", action="store_true", help="Reset automatically when the env terminates/truncates")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"找不到模型文件，请检查路径是否正确: {args.model}")
    if args.checkpoint is not None and not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"找不到 checkpoint，请检查路径是否正确: {args.checkpoint}")

    env = UnitreeA1Env(args.model, task=args.task)
    obs, info = env.reset(seed=args.seed)
    policy = PPO.load(args.checkpoint, env=env) if args.checkpoint is not None else None
    action = np.zeros(env.action_space.shape, dtype=np.float32)

    print(f"机器狗加载成功，当前 task={args.task}。")
    if policy is None:
        print("控制模式：zero action，用于检查默认站立/恢复目标。")
    else:
        print(f"控制模式：PPO checkpoint = {args.checkpoint}")
        print(f"动作模式：{'stochastic' if args.stochastic else 'deterministic'}")
    print(f"Reset info: {info}")
    print("运行方式：mjpython demo.py")
    print("交互提示：双击机身后右键旋转视角，Ctrl + 左键可以拖拽机身观察物理响应。")
    print("checkpoint 模式默认不会自动 reset，方便手动扰动后观察 policy 恢复。")

    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            while viewer.is_running():
                viewer.sync()
                if policy is not None:
                    obs = current_obs(env)
                    action, _ = policy.predict(obs, deterministic=not args.stochastic)
                obs, _, terminated, truncated, _ = env.step(action)
                if args.auto_reset and (terminated or truncated):
                    obs, info = env.reset(seed=args.seed)
                viewer.sync()
                time.sleep(env.model.opt.timestep * env.frame_skip)
    finally:
        env.close()


if __name__ == "__main__":
    main()
