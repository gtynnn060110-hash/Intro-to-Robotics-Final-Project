# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

北京大学《机器人学概论》期末大作业：基于课程强化学习的四足机器人低摩擦非平整地形自适应运动控制。

Goal: Train a Unitree A1 quadruped robot in MuJoCo simulation to walk robustly on rough/icy terrain using deep RL (PPO via stable-baselines3) with curriculum learning.

## Environment Setup

```bash
conda activate robot_proj   # Python 3.10, Apple Silicon (ARM64)
```

Dependencies: `mujoco` (3.x native, NOT mujoco-py), `gymnasium`, `stable-baselines3`, `wandb`, `moviepy`

Robot model lives in `final_project/unitree_a1/` (local copy for terrain edits).

## Running Scripts

**Critical macOS requirement:** Use `mjpython` instead of `python` for any script that imports `mujoco` or launches a viewer — otherwise OpenGL rendering will fail.

```bash
mjpython demo.py              # Run the physics demo with interactive viewer
mjpython train.py             # Run RL training (when implemented)
```

## Architecture

The project follows a standard RL pipeline:

1. **MuJoCo model** (`final_project/unitree_a1/scene.xml`) — physics description of the A1 robot with ground plane, lighting, and contacts (local copy for terrain edits)
2. **Gymnasium environment** (to be built) — wraps `mujoco.MjModel`/`MjData` into a `gymnasium.Env` with:
   - Action space: 12 continuous joint target angles (PD-controlled)
   - Observation space: proprioceptive state (trunk quaternion, angular velocity, 12× joint angle + velocity)
   - Reward: curriculum-based, starting with stand-up reward (trunk height + joint velocity penalty)
3. **PPO training loop** — `stable-baselines3` PPO trains against the gymnasium env
4. **Logging** — `wandb` for metrics and video visualization

## Key Design Decisions

- **PD control layer**: Actions are target joint angles, not raw torques. The environment applies a PD controller internally before calling `mj_step`.
- **Curriculum RL**: Terrain difficulty and friction coefficient are scheduled to increase as the agent improves — start flat/normal, progress to rough/icy.
- **Observation normalization**: Proprioceptive observations should be normalized before feeding to the policy network.
