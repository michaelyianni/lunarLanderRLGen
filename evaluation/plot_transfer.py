import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import matplotlib.pyplot as plt
from evaluation.plot import ALGORITHM_COLOURS, SMOOTHING_WINDOW
from evaluation.metrics import smooth_rewards


# --- Config ---

TRANSFER_LOG_PATH = "results/logs/transfer/transfer_results.json"
FROZEN_LOG_PATHS  = {
    "gravity": "results/logs/gravity/gravity_results.json",
    "wind"   : "results/logs/wind/wind_results.json",
    "noise"  : "results/logs/noise/noise_results.json",
}

# DQN/PPO/A2C for recovery curves; all four for bar/box plots
ALGORITHMS_CURVES = ["DQN", "PPO", "A2C"]
ALGORITHMS_ALL    = ["DQN", "PPO", "A2C", "Ensemble"]
ENVIRONMENTS      = ["gravity", "wind", "noise"]

ENV_LABELS = {
    "gravity": "Gravity (g=−13.0)",
    "wind"   : "Wind (power=15.0)",
    "noise"  : "Noise (std=0.075)",
}

SAVE_DIR = "results/plots/transfer"



#  Plot 1 - Recovery Curves (one file per environment)


def plot_recovery_curves():
    """
    Produces one recovery curve figure per environment (3 total).
    Each figure has one row with three panels — one per algorithm (DQN/PPO/A2C).
    Each panel shows fine-tuned (solid) vs scratch (dashed) smoothed reward.
    The Ensemble has no training curve and is intentionally excluded here.

    Saved to:
        results/plots/transfer/recovery_curves_gravity.png
        results/plots/transfer/recovery_curves_wind.png
        results/plots/transfer/recovery_curves_noise.png
    """
    os.makedirs(SAVE_DIR, exist_ok=True)

    for env_name in ENVIRONMENTS:
        fig, axes = plt.subplots(1, len(ALGORITHMS_CURVES),
                                 figsize=(16, 5), sharey=True)

        for col, algorithm in enumerate(ALGORITHMS_CURVES):
            ax     = axes[col]
            colour = ALGORITHM_COLOURS.get(algorithm, "steelblue")

            for mode, linestyle, label_suffix in [
                ("finetune", "-",  "Fine-tuned"),
                ("scratch",  "--", "Scratch"),
            ]:
                curve_path = (f"results/logs/transfer/{env_name}/"
                              f"{algorithm.lower()}_{env_name}_{mode}_curve.json")
                try:
                    with open(curve_path, "r") as f:
                        curve = json.load(f)
                    smoothed = smooth_rewards(
                        curve["episode_rewards"], window=SMOOTHING_WINDOW
                    )
                    ax.plot(smoothed, color=colour, linestyle=linestyle,
                            linewidth=2.0, label=label_suffix)
                except FileNotFoundError:
                    ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                            ha="center", va="center", fontsize=10, color="grey")

            ax.axhline(y=200, color="red", linestyle=":", linewidth=1.2,
                       alpha=0.7, label="Solved threshold (200)")
            ax.set_title(algorithm, fontsize=12)
            ax.set_xlabel("Episode", fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9)

            if col == 0:
                ax.set_ylabel("Smoothed Reward", fontsize=10)

        fig.suptitle(
            f"Transfer Learning — Recovery Curves\n"
            f"{ENV_LABELS[env_name]}  "
            f"(Solid = Fine-tuned from baseline, Dashed = Scratch)",
            fontsize=12
        )
        plt.tight_layout()

        save_path = os.path.join(SAVE_DIR, f"recovery_curves_{env_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Recovery curves saved to {save_path}")



#  Plot 2 - Final Performance Comparison (Success Rate)

def plot_final_performance():
    """
    Grouped bar chart comparing fine-tuned vs scratch vs frozen success rate.
    One chart per environment, one bar group per algorithm (including Ensemble).
    Frozen Ensemble is read from the transfer log (loaded from ensemble results).
    """
    with open(TRANSFER_LOG_PATH, "r") as f:
        transfer_log = json.load(f)

    frozen_data = {}
    for env_name, path in FROZEN_LOG_PATHS.items():
        with open(path, "r") as f:
            frozen_data[env_name] = json.load(f)

    fig, axes = plt.subplots(1, len(ENVIRONMENTS), figsize=(18, 5), sharey=True)

    for col, env_name in enumerate(ENVIRONMENTS):
        ax      = axes[col]
        x       = np.arange(len(ALGORITHMS_ALL))
        width   = 0.25
        offsets = [-width, 0, width]

        modes_labels = [
            ("finetuned", "Fine-tuned", 0.85),
            ("scratch",   "Scratch",    0.55),
            ("frozen",    "Frozen",     0.40),
        ]

        for offset, (mode, label, alpha) in zip(offsets, modes_labels):
            success_rates = []
            for algorithm in ALGORITHMS_ALL:
                if algorithm == "Ensemble":
                    # All three Ensemble modes are in the transfer log
                    try:
                        stats = (transfer_log["results"]
                                 [env_name]["Ensemble"][mode]["stats"])
                        success_rates.append(stats["success_rate"] * 100)
                    except KeyError:
                        success_rates.append(0.0)
                elif mode == "frozen":
                    try:
                        stats = (frozen_data[env_name]["results"]
                                 [algorithm]["stats"])
                        success_rates.append(stats["success_rate"] * 100)
                    except KeyError:
                        success_rates.append(0.0)
                else:
                    try:
                        stats = (transfer_log["results"]
                                 [env_name][algorithm][mode]["stats"])
                        success_rates.append(stats["success_rate"] * 100)
                    except KeyError:
                        success_rates.append(0.0)

            colours = [ALGORITHM_COLOURS.get(a, "steelblue") for a in ALGORITHMS_ALL]
            ax.bar(x + offset, success_rates, width=width, label=label,
                   color=colours, alpha=alpha, edgecolor="black", linewidth=0.7)

        ax.axhline(y=100, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(ENV_LABELS[env_name], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(ALGORITHMS_ALL, fontsize=9)
        ax.set_ylim(0, 115)
        ax.grid(axis="y", alpha=0.3)
        if col == 0:
            ax.set_ylabel("Success Rate (%)", fontsize=10)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="grey", alpha=0.85, label="Fine-tuned"),
        plt.Rectangle((0, 0), 1, 1, color="grey", alpha=0.55, label="Scratch"),
        plt.Rectangle((0, 0), 1, 1, color="grey", alpha=0.40, label="Frozen"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)

    plt.suptitle("Transfer Learning — Fine-tuned vs Scratch vs Frozen", fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, "transfer_comparison.png")
    os.makedirs(SAVE_DIR, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Transfer comparison saved to {save_path}")


#  Plot 3 - Reward Distributions (Box Plots)

def plot_transfer_distributions():
    """
    One figure per environment (3 total), each showing reward distributions
    as box plots with jittered scatter overlay.
    Groups of three boxes per algorithm: [Frozen | Fine-tuned | Scratch].
    Includes the Ensemble agent as a fourth group.

    Saved to:
        results/plots/transfer/transfer_distributions_gravity.png
        results/plots/transfer/transfer_distributions_wind.png
        results/plots/transfer/transfer_distributions_noise.png
    """
    with open(TRANSFER_LOG_PATH, "r") as f:
        transfer_log = json.load(f)

    frozen_data = {}
    for env_name, path in FROZEN_LOG_PATHS.items():
        with open(path, "r") as f:
            frozen_data[env_name] = json.load(f)

    group_modes = [
        ("frozen",    "Frozen"),
        ("finetuned", "Fine-tuned"),
        ("scratch",   "Scratch"),
    ]

    for env_name in ENVIRONMENTS:
        fig, ax = plt.subplots(figsize=(14, 6))

        box_data   = []
        positions  = []
        colours    = []
        xtick_pos  = []
        xtick_lbls = []

        pos = 1
        for algorithm in ALGORITHMS_ALL:
            group_start = pos
            colour = ALGORITHM_COLOURS.get(algorithm, "steelblue")

            for mode, _ in group_modes:
                if algorithm == "Ensemble":
                    # All Ensemble modes come from the transfer log
                    try:
                        rewards = (transfer_log["results"]
                                   [env_name]["Ensemble"][mode]["rewards"])
                    except KeyError:
                        rewards = [0.0]
                elif mode == "frozen":
                    try:
                        rewards = (frozen_data[env_name]["results"]
                                   [algorithm]["rewards"])
                    except KeyError:
                        rewards = [0.0]
                else:
                    try:
                        rewards = (transfer_log["results"]
                                   [env_name][algorithm][mode]["rewards"])
                    except KeyError:
                        rewards = [0.0]

                box_data.append(rewards)
                positions.append(pos)
                colours.append(colour)
                pos += 1

            group_centre = (group_start + pos - 1) / 2
            xtick_pos.append(group_centre)
            xtick_lbls.append(algorithm)

            pos += 1.2  # gap between algorithm groups

        # --- Draw box plots ---
        bp = ax.boxplot(
            box_data,
            positions=positions,
            patch_artist=True,
            notch=False,
            widths=0.6,
            medianprops=dict(color="black", linewidth=2),
            manage_ticks=False,
        )

        # Colour and style each box
        for i, (patch, colour) in enumerate(zip(bp["boxes"], colours)):
            patch.set_facecolor(colour)
            mode_idx = i % len(group_modes)
            mode_name = group_modes[mode_idx][0]
            if mode_name == "frozen":
                patch.set_alpha(0.60)
                patch.set_hatch("//")
            elif mode_name == "finetuned":
                patch.set_alpha(0.90)
            else:  # scratch
                patch.set_alpha(0.60)

        # --- Jittered scatter overlay ---
        for i, (rewards, pos_x) in enumerate(zip(box_data, positions)):
            x_jitter = np.random.normal(pos_x, 0.08, size=len(rewards))
            ax.scatter(x_jitter, rewards, alpha=0.25, s=10, color=colours[i])

        # --- Solved threshold ---
        ax.axhline(y=200, color="red", linestyle="--", linewidth=1.2,
                   label="Solved threshold (200)")

        # --- Axes formatting ---
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_lbls, fontsize=11)
        ax.set_ylabel("Total Episode Reward", fontsize=11)
        ax.set_title(
            f"Transfer Learning — Reward Distributions\n"
            f"{ENV_LABELS[env_name]} (100 Episodes)",
            fontsize=12
        )
        ax.grid(axis="y", alpha=0.3)

        # --- Legends - both placed outside the plot area below the axes ---
        mode_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor="grey",
                           alpha=0.30, hatch="//", label="Frozen"),
            plt.Rectangle((0, 0), 1, 1, facecolor="grey",
                           alpha=0.90, label="Fine-tuned"),
            plt.Rectangle((0, 0), 1, 1, facecolor="grey",
                           alpha=0.60, label="Scratch"),
        ]
        alg_handles = [
            plt.Rectangle((0, 0), 1, 1,
                           facecolor=ALGORITHM_COLOURS.get(a, "steelblue"),
                           alpha=0.75, label=a)
            for a in ALGORITHMS_ALL
        ] + [
            plt.Line2D([0], [0], color="red", linestyle="--",
                       linewidth=1.2, label="Solved threshold (200)")
        ]

        # Mode legend - bottom left
        legend_mode = fig.legend(
            handles=mode_handles,
            title="Mode",
            loc="lower left",
            bbox_to_anchor=(0.01, 0.05),
            ncol=3,
            fontsize=9,
            framealpha=0.9,
        )
        fig.add_artist(legend_mode)

        # Algorithm + threshold legend - bottom right
        fig.legend(
            handles=alg_handles,
            loc="lower right",
            bbox_to_anchor=(0.99, 0.05),
            ncol=len(alg_handles),
            fontsize=9,
            framealpha=0.9,
        )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.18)   # make room for both legends below the axis

        save_path = os.path.join(SAVE_DIR, f"transfer_distributions_{env_name}.png")
        os.makedirs(SAVE_DIR, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Distribution plot saved to {save_path}")


# Entry Point

def main():
    plot_recovery_curves()
    plot_final_performance()
    plot_transfer_distributions()


if __name__ == "__main__":
    main()