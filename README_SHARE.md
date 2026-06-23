# Unitree A1 walking and recovery scripts

This package contains the important source files for the walking and recovery
training experiments. It intentionally excludes `runs/`, checkpoints, videos,
logs, `__pycache__`, and local MuJoCo logs so the archive stays small enough to
send.

## Environment

Use the project conda environment if available:

```bash
conda activate robot_proj
```

Required Python packages:

```bash
pip install mujoco gymnasium stable-baselines3 wandb moviepy imageio
```

On macOS, use `mjpython` for scripts that import MuJoCo or render:

```bash
mjpython train_recovery.py
mjpython train_walk.py
```

On Linux/headless machines, normal `python` usually works:

```bash
python train_recovery.py
python train_walk.py
```

## Main files

- `envs/unitree_a1_env.py`: recovery/standing Gymnasium environment.
- `envs/unitree_a1_walk_env.py`: walking environment built on the A1 env.
- `train_recovery.py`: recovery policy training entrypoint.
- `train_hard_bin.py`: harder recovery/righting curriculum variant.
- `train_two_stage_righting.py`: two-stage righting training variant.
- `train_replay_antiforgetting.py`: recovery replay/anti-forgetting variant.
- `train_walk.py`: fixed-speed walking policy training entrypoint.
- `train_walk_mixed_terrain.py`: mixed-terrain walking fine-tuning.
- `eval_hard_bin.py`, `eval_two_stage_righting.py`,
  `eval_replay_antiforgetting.py`: matching evaluation scripts for recovery variants.
- `eval_recovery_seed_benchmark.py`: recovery benchmark over seeds.
- `eval_walk.py`: walking evaluation.
- `eval_recover_then_walk.py`: recovery-then-walking gated evaluation.
- `benchmark_models.py`: compare saved recovery checkpoints.
- `render_policy_video.py`: render a trained policy to video.
- `unitree_a1/`: MuJoCo model XML, mesh assets, terrain texture, terrain generator.

## Common commands

Short smoke run:

```bash
python train_recovery.py --total-steps 10000 --n-envs 1 --wandb-mode disabled --run-dir runs/recovery_smoke
python train_walk.py --total-steps 10000 --n-envs 1 --run-dir runs/walk_smoke
```

Recovery training:

```bash
python train_recovery.py \
  --model unitree_a1/scene.xml \
  --total-steps 4000000 \
  --n-envs 8 \
  --wandb-mode disabled \
  --run-dir runs/recovery_fresh_curriculum
```

Walking training:

```bash
python train_walk.py \
  --model unitree_a1/scene.xml \
  --total-steps 1000000 \
  --n-envs 8 \
  --run-dir runs/walk_from_recovery
```

Resume/fine-tune walking from a recovery checkpoint:

```bash
python train_walk.py \
  --resume-from runs/recovery_fresh_curriculum/ppo_recovery_stand_final.zip \
  --vecnormalize-load runs/recovery_fresh_curriculum/vecnormalize.pkl \
  --run-dir runs/walk_from_recovery
```

Evaluate walking:

```bash
python eval_walk.py \
  --model-path runs/walk_from_recovery/ppo_walk_final.zip \
  --vecnormalize-path runs/walk_from_recovery/vecnormalize.pkl
```

Render video:

```bash
python render_policy_video.py \
  --policy runs/walk_from_recovery/ppo_walk_final.zip \
  --vecnormalize runs/walk_from_recovery/vecnormalize.pkl \
  --task walk \
  --output runs/demo_videos/walk.mp4
```

## If sending trained models too

This archive does not include trained models. To let someone run evaluation
without retraining, send each policy with its matching VecNormalize file:

- recovery: `ppo_recovery_stand_final.zip` + `vecnormalize.pkl`
- walking: `ppo_walk_final.zip` + `vecnormalize.pkl`

Keep each pair from the same `runs/<experiment>/` directory.
