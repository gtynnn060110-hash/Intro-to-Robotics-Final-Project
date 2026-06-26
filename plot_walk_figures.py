"""Manual plotting script for walking benchmark results.

Run:

    mjpython plot_walk_figures.py

This script is intentionally plain and easy to edit. It generates three groups
of profile plots:

1. One plot per terrain. In each terrain plot, six lines compare:

    2 checkpoints x 3 friction coefficients

2. One plot per friction coefficient. In each friction plot, six lines compare:

    2 checkpoints x 3 terrains

3. One plot per checkpoint. In each checkpoint plot, nine lines compare:

    3 terrains x 3 friction coefficients

The x-axis is a small set of normalized evaluation metrics, so the line shape
shows the overall behavior profile of each condition.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================
# CONFIG: edit here first
# =========================

BENCHMARK_DIR = Path("runs/walk_benchmark_checkpoint_scene_friction_v1")
INPUT_CSV = BENCHMARK_DIR / "summary.csv"
OUTPUT_DIR = BENCHMARK_DIR / "figures_custom"

CHECKPOINT_ORDER = ["bc_dog_trot_v3", "ppo_after_bc_teacher_v1"]
CHECKPOINT_LABEL = {
    "bc_dog_trot_v3": "SFT",
    "ppo_after_bc_teacher_v1": "SFT+PPO",
}

TERRAIN_ORDER = ["wave_h030", "irregular_h030", "irregular_h040"]
TERRAIN_LABEL = {
    "wave_h030": "Wave h=0.30",
    "irregular_h030": "Irregular h=0.30",
    "irregular_h040": "Irregular h=0.40",
}

FRICTION_ORDER = [1.0, 0.8, 0.7]
FRICTION_LABEL = {
    1.0: "mu=1.0",
    0.8: "mu=0.8",
    0.7: "mu=0.7",
}

# Line color = checkpoint or terrain depending on the figure.
CHECKPOINT_COLOR = {
    "bc_dog_trot_v3": "#4C78A8",
    "ppo_after_bc_teacher_v1": "#F58518",
}
TERRAIN_COLOR = {
    "wave_h030": "#54A24B",
    "irregular_h030": "#B279A2",
    "irregular_h040": "#E45756",
}

# Line style/marker = controlled variable level.
FRICTION_STYLE = {
    1.0: ("-", "o"),
    0.8: ("--", "s"),
    0.7: (":", "^"),
}
CHECKPOINT_STYLE = {
    "bc_dog_trot_v3": ("-", "o"),
    "ppo_after_bc_teacher_v1": ("--", "s"),
}

# Metrics shown on the x-axis of each terrain figure.
# All values are normalized to [0, 1], where larger is better.
PROFILE_METRICS = [
    ("survive_rate", "Survival"),
    ("velocity_score", "Velocity"),
    ("reward_per_step_norm", "Reward/step"),
    ("lateral_stability_score", "Lateral"),
]

# Normalization constants. These are deliberately visible so you can tune them.
TARGET_VX = 0.20
MAX_ABS_VY = 0.30


def load_data():
    df = pd.read_csv(INPUT_CSV)

    # Reward regularization:
    # Raw episode reward is divided by episode length, so early termination does
    # not automatically look worse only because fewer steps were accumulated.
    df["reward_per_step"] = df["mean_reward"] / df["mean_steps"].clip(lower=1.0)
    r_min = df["reward_per_step"].min()
    r_max = df["reward_per_step"].max()
    df["reward_per_step_norm"] = (df["reward_per_step"] - r_min) / max(r_max - r_min, 1e-9)

    # Metric scores. Larger is better for all scores below.
    df["velocity_score"] = (df["mean_forward_velocity"] / TARGET_VX).clip(lower=0.0, upper=1.0)
    df["lateral_stability_score"] = 1.0 - (df["mean_abs_vy"] / MAX_ABS_VY).clip(
        lower=0.0, upper=1.0
    )

    df["checkpoint_label"] = df["checkpoint_name"].map(CHECKPOINT_LABEL).fillna(df["checkpoint_name"])
    df["terrain_label"] = df["terrain_name"].map(TERRAIN_LABEL).fillna(df["terrain_name"])
    return df


def select(df, checkpoint=None, terrain=None, friction=None):
    out = df
    if checkpoint is not None:
        out = out[out["checkpoint_name"] == checkpoint]
    if terrain is not None:
        out = out[out["terrain_name"] == terrain]
    if friction is not None:
        out = out[np.isclose(out["terrain_friction"], friction)]
    return out.copy()


def put_legend_above(ax, ncol, fontsize):
    ax.legend(
        frameon=False,
        ncol=ncol,
        fontsize=fontsize,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        borderaxespad=0.0,
        handlelength=2.2,
        columnspacing=1.1,
    )


def plot_profile_for_terrain(df, terrain):
    """One terrain per figure, six lines = 2 checkpoints x 3 frictions."""
    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    x = np.arange(len(PROFILE_METRICS))
    x_labels = [label for _, label in PROFILE_METRICS]

    for checkpoint in CHECKPOINT_ORDER:
        for friction in FRICTION_ORDER:
            row = select(df, checkpoint=checkpoint, terrain=terrain, friction=friction)
            if row.empty:
                continue
            row = row.iloc[0]
            values = [float(row[metric]) for metric, _ in PROFILE_METRICS]
            linestyle, marker = FRICTION_STYLE[friction]
            label = f"{CHECKPOINT_LABEL[checkpoint]}, {FRICTION_LABEL[friction]}"

            ax.plot(
                x,
                values,
                color=CHECKPOINT_COLOR[checkpoint],
                linestyle=linestyle,
                marker=marker,
                linewidth=2.0,
                markersize=5.5,
                label=label,
            )

    ax.set_title(TERRAIN_LABEL.get(terrain, terrain), pad=58)
    ax.set_ylabel("normalized score, higher is better")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, axis="y", alpha=0.25)
    put_legend_above(ax, ncol=3, fontsize=8.2)

    fig.tight_layout()
    filename = f"profile_{terrain}.png"
    fig.savefig(OUTPUT_DIR / filename, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return filename


def plot_profile_for_friction(df, friction):
    """One friction per figure, six lines = 2 checkpoints x 3 terrains."""
    fig, ax = plt.subplots(figsize=(8.6, 5.0))

    x = np.arange(len(PROFILE_METRICS))
    x_labels = [label for _, label in PROFILE_METRICS]

    for terrain in TERRAIN_ORDER:
        for checkpoint in CHECKPOINT_ORDER:
            row = select(df, checkpoint=checkpoint, terrain=terrain, friction=friction)
            if row.empty:
                continue
            row = row.iloc[0]
            values = [float(row[metric]) for metric, _ in PROFILE_METRICS]
            linestyle, marker = CHECKPOINT_STYLE[checkpoint]
            label = f"{TERRAIN_LABEL[terrain]}, {CHECKPOINT_LABEL[checkpoint]}"

            ax.plot(
                x,
                values,
                color=TERRAIN_COLOR[terrain],
                linestyle=linestyle,
                marker=marker,
                linewidth=2.0,
                markersize=5.5,
                label=label,
            )

    ax.set_title(f"Fixed friction: {FRICTION_LABEL.get(friction, friction)}", pad=58)
    ax.set_ylabel("normalized score, higher is better")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, axis="y", alpha=0.25)
    put_legend_above(ax, ncol=3, fontsize=7.9)

    fig.tight_layout()
    filename = f"profile_friction_{str(friction).replace('.', 'p')}.png"
    fig.savefig(OUTPUT_DIR / filename, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return filename


def plot_profile_for_checkpoint(df, checkpoint):
    """One checkpoint per figure, nine lines = 3 terrains x 3 frictions."""
    fig, ax = plt.subplots(figsize=(9.0, 5.3))

    x = np.arange(len(PROFILE_METRICS))
    x_labels = [label for _, label in PROFILE_METRICS]

    for terrain in TERRAIN_ORDER:
        for friction in FRICTION_ORDER:
            row = select(df, checkpoint=checkpoint, terrain=terrain, friction=friction)
            if row.empty:
                continue
            row = row.iloc[0]
            values = [float(row[metric]) for metric, _ in PROFILE_METRICS]
            linestyle, marker = FRICTION_STYLE[friction]
            label = f"{TERRAIN_LABEL[terrain]}, {FRICTION_LABEL[friction]}"

            ax.plot(
                x,
                values,
                color=TERRAIN_COLOR[terrain],
                linestyle=linestyle,
                marker=marker,
                linewidth=2.0,
                markersize=5.2,
                label=label,
            )

    ax.set_title(f"Fixed checkpoint: {CHECKPOINT_LABEL.get(checkpoint, checkpoint)}", pad=72)
    ax.set_ylabel("normalized score, higher is better")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, axis="y", alpha=0.25)
    put_legend_above(ax, ncol=3, fontsize=7.4)

    fig.tight_layout()
    filename = f"profile_checkpoint_{checkpoint}.png"
    fig.savefig(OUTPUT_DIR / filename, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return filename


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    saved = []
    for terrain in TERRAIN_ORDER:
        saved.append(plot_profile_for_terrain(df, terrain))
    for friction in FRICTION_ORDER:
        saved.append(plot_profile_for_friction(df, friction))
    for checkpoint in CHECKPOINT_ORDER:
        saved.append(plot_profile_for_checkpoint(df, checkpoint))

    # Save a copy with the regularized reward and score columns for inspection.
    df.to_csv(OUTPUT_DIR / "summary_with_profile_scores.csv", index=False)

    print(f"Saved figures to: {OUTPUT_DIR}")
    for filename in saved:
        print(f"  - {filename}")
    print("Saved table: summary_with_profile_scores.csv")


if __name__ == "__main__":
    main()
