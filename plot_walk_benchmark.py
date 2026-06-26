"""Post-process walking benchmark results and generate controlled-variable plots."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CHECKPOINT_LABELS = {
    "bc_dog_trot_v3": "SFT Training",
    "ppo_after_bc_teacher_v1": "PPO Fine-Tuning",
}

TERRAIN_LABELS = {
    "wave_h030": "Wave h=0.30",
    "irregular_h030": "Irregular h=0.30",
    "irregular_h040": "Irregular h=0.40",
}


def minmax(series):
    lo = float(series.min())
    hi = float(series.max())
    if abs(hi - lo) < 1e-12:
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def enrich_summary(summary):
    df = summary.copy()
    df["checkpoint_label"] = df["checkpoint_name"].map(CHECKPOINT_LABELS).fillna(df["checkpoint_name"])
    df["terrain_label"] = df["terrain_name"].map(TERRAIN_LABELS).fillna(df["terrain_name"])
    df["reward_per_step"] = df["mean_reward"] / df["mean_steps"].clip(lower=1.0)
    df["reward_per_step_norm"] = minmax(df["reward_per_step"])
    df["speed_ratio"] = df["mean_forward_velocity"] / 0.2
    df["condition"] = df["terrain_label"] + ", mu=" + df["terrain_friction"].map(lambda x: f"{x:.1f}")
    return df


def aggregate_per_episode(per_episode):
    df = per_episode.copy()
    df["checkpoint_label"] = df["checkpoint_name"].map(CHECKPOINT_LABELS).fillna(df["checkpoint_name"])
    df["terrain_label"] = df["terrain_name"].map(TERRAIN_LABELS).fillna(df["terrain_name"])
    df["reward_per_step"] = df["reward"] / df["steps"].clip(lower=1.0)
    grouped = (
        df.groupby(["checkpoint_name", "checkpoint_label", "terrain_name", "terrain_label", "terrain_friction"])
        .agg(
            reward_per_step_std=("reward_per_step", "std"),
            mean_vx_std=("mean_vx", "std"),
            height_error_abs_std=("mean_height_error", lambda x: np.std(np.abs(x), ddof=1)),
            survive_std=("survived", "std"),
        )
        .reset_index()
    )
    return grouped.fillna(0.0)


def line_grid(df, metric, ylabel, output, ylim=None, invert_y=False):
    terrains = list(TERRAIN_LABELS.keys())
    fig, axes = plt.subplots(1, len(terrains), figsize=(12, 3.4), sharey=True)
    for ax, terrain in zip(axes, terrains):
        sub = df[df["terrain_name"] == terrain].sort_values("terrain_friction", ascending=False)
        for checkpoint, group in sub.groupby("checkpoint_label", sort=False):
            group = group.sort_values("terrain_friction", ascending=False)
            ax.plot(
                group["terrain_friction"],
                group[metric],
                marker="o",
                linewidth=2,
                label=checkpoint,
            )
        ax.set_title(TERRAIN_LABELS.get(terrain, terrain))
        ax.set_xlabel("friction coefficient")
        ax.grid(True, alpha=0.25)
        ax.invert_xaxis()
        if ylim is not None:
            ax.set_ylim(*ylim)
        if invert_y:
            ax.invert_yaxis()
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def checkpoint_bar(df, output):
    pivot_metrics = [
        ("survive_rate", "Survival rate"),
        ("mean_forward_velocity", "Mean velocity (m/s)"),
        ("reward_per_step_norm", "Normalized reward/step"),
        ("mean_abs_height_error", "Abs. height error (m)"),
    ]
    grouped = df.groupby("checkpoint_label").agg(
        survive_rate=("survive_rate", "mean"),
        mean_forward_velocity=("mean_forward_velocity", "mean"),
        reward_per_step_norm=("reward_per_step_norm", "mean"),
        mean_abs_height_error=("mean_abs_height_error", "mean"),
    )

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    colors = ["#8da0cb", "#66c2a5"]
    for ax, (metric, title) in zip(axes, pivot_metrics):
        values = grouped[metric]
        ax.bar(values.index, values.values, color=colors[: len(values)])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Average across all terrain/friction conditions", y=1.04)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def condition_delta(df, output):
    teacher = df[df["checkpoint_name"] == "ppo_after_bc_teacher_v1"].copy()
    bc = df[df["checkpoint_name"] == "bc_dog_trot_v3"].copy()
    merged = teacher.merge(
        bc,
        on=["terrain_name", "terrain_label", "terrain_friction", "terrain_height_scale"],
        suffixes=("_teacher", "_bc"),
    )
    merged["velocity_gain"] = merged["mean_forward_velocity_teacher"] - merged["mean_forward_velocity_bc"]
    merged["survival_gain"] = merged["survive_rate_teacher"] - merged["survive_rate_bc"]
    merged["reward_step_gain"] = merged["reward_per_step_teacher"] - merged["reward_per_step_bc"]
    merged["condition"] = (
        merged["terrain_label"]
        + "\nmu="
        + merged["terrain_friction"].map(lambda x: f"{x:.1f}")
    )
    merged = merged.sort_values(["terrain_name", "terrain_friction"], ascending=[True, False])

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for ax, metric, title in [
        (axes[0], "velocity_gain", "Velocity gain (m/s)"),
        (axes[1], "survival_gain", "Survival gain"),
        (axes[2], "reward_step_gain", "Reward/step gain"),
    ]:
        ax.bar(merged["condition"], merged[metric], color="#fc8d62")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].tick_params(axis="x", rotation=35)
    fig.suptitle("Teacher PPO improvement over BC under matched conditions", y=0.995)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def latex_table(df, output):
    rows = []
    for _, row in df.sort_values(["checkpoint_name", "terrain_name", "terrain_friction"], ascending=[True, True, False]).iterrows():
        rows.append(
            f"{row['checkpoint_label']} & {row['terrain_label']} & {row['terrain_friction']:.1f} & "
            f"{int(row['survived'])}/{int(row['episodes'])} & "
            f"{row['mean_forward_velocity']:.3f} & "
            f"{row['reward_per_step']:.2f} & "
            f"{row['reward_per_step_norm']:.2f} & "
            f"{row['mean_abs_height_error']:.3f} \\\\"
        )
    output.write_text("\n".join(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="runs/walk_benchmark_checkpoint_scene_friction_v1")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    summary = pd.read_csv(input_dir / "summary.csv")
    per_episode = pd.read_csv(input_dir / "per_episode.csv")

    enriched = enrich_summary(summary)
    episode_stats = aggregate_per_episode(per_episode)
    enriched = enriched.merge(
        episode_stats,
        on=["checkpoint_name", "checkpoint_label", "terrain_name", "terrain_label", "terrain_friction"],
        how="left",
    )

    enriched.to_csv(input_dir / "summary_regularized.csv", index=False)
    latex_table(enriched, input_dir / "latex_table_rows.txt")

    line_grid(
        enriched,
        "mean_forward_velocity",
        "mean velocity (m/s)",
        input_dir / "fig_velocity_by_friction.png",
        ylim=(-0.08, 0.22),
    )
    line_grid(
        enriched,
        "survive_rate",
        "survival rate",
        input_dir / "fig_survival_by_friction.png",
        ylim=(0.0, 1.08),
    )
    line_grid(
        enriched,
        "reward_per_step_norm",
        "normalized reward/step",
        input_dir / "fig_reward_norm_by_friction.png",
        ylim=(0.0, 1.08),
    )
    line_grid(
        enriched,
        "mean_abs_height_error",
        "abs. height error (m)",
        input_dir / "fig_height_error_by_friction.png",
        ylim=(0.0, max(0.16, float(enriched["mean_abs_height_error"].max()) * 1.1)),
    )
    checkpoint_bar(enriched, input_dir / "fig_checkpoint_overall.png")
    condition_delta(enriched, input_dir / "fig_teacher_gain_by_condition.png")
    print(f"[done] wrote regularized summary and figures to {input_dir}")


if __name__ == "__main__":
    main()
