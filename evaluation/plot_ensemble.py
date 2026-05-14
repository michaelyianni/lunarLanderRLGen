import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import matplotlib.pyplot as plt
from evaluation.plot import ALGORITHM_COLOURS


# --- Config ---
ENSEMBLE_LOG_PATH = "results/logs/ensemble/ensemble_results.json"
BASELINE_LOG_PATH = "results/logs/baseline/baseline_results.json"
GRAVITY_LOG_PATH  = "results/logs/gravity/gravity_results.json"
WIND_LOG_PATH     = "results/logs/wind/wind_results.json"
NOISE_LOG_PATH    = "results/logs/noise/noise_results.json"
SAVE_PATH         = "results/plots/ensemble/ensemble_comparison.png"

GRAVITY    = -13.0
WIND_POWER = 15.0
NOISE_STD  = 0.075

ENVIRONMENTS = ["standard", "gravity", "wind", "noise"]
ENV_LABELS   = {
    "standard": "Standard",
    "gravity" : f"Gravity (g={GRAVITY})",
    "wind"    : f"Wind (power={WIND_POWER})",
    "noise"   : f"Noise (std={NOISE_STD})",
}

# Maps each environment to its corresponding log file path
ENV_LOG_PATHS = {
    "standard": BASELINE_LOG_PATH,
    "gravity" : GRAVITY_LOG_PATH,
    "wind"    : WIND_LOG_PATH,
    "noise"   : NOISE_LOG_PATH,
}

ALGORITHMS = ["DQN", "PPO", "A2C", "Ensemble"]

DIST_SAVE_DIR = "results/plots/ensemble"


def load_results() -> dict:
    """
    Load and merge results for all algorithms across all environments.
    DQN, PPO, A2C results are read from their respective per-environment
    log files. Ensemble results are read from ensemble_results.json.

    Returns:
        dict: Nested dict keyed by environment -> algorithm -> {stats, rewards}.
    """
    env_logs = {}
    for env_name, log_path in ENV_LOG_PATHS.items():
        with open(log_path, "r") as f:
            env_logs[env_name] = json.load(f)

    with open(ENSEMBLE_LOG_PATH, "r") as f:
        ensemble_log = json.load(f)

    merged = {}
    for env_name in ENVIRONMENTS:
        merged[env_name] = {}

        # DQN, PPO, A2C - stats and rewards from environment-specific log
        log = env_logs[env_name]
        for alg in ["DQN", "PPO", "A2C"]:
            try:
                merged[env_name][alg] = {
                    "stats"  : log["results"][alg]["stats"],
                    "rewards": log["results"][alg]["rewards"],
                }
            except KeyError:
                pass

        # Ensemble - from ensemble_results.json
        try:
            merged[env_name]["Ensemble"] = {
                "stats"  : ensemble_log["results"][env_name]["stats"],
                "rewards": ensemble_log["results"][env_name]["rewards"],
            }
        except KeyError:
            pass

    return merged


def plot_comparison(merged: dict, save_path: str = None):
    """
    Produce two side-by-side grouped bar charts:
      1. Mean reward per environment, grouped by environment, one bar per algorithm.
      2. Success rate per environment, same layout.

    Args:
        merged    (dict): Output from load_results().
        save_path (str):  If provided, saves the figure to this path.
    """
    n_envs  = len(ENVIRONMENTS)
    n_algs  = len(ALGORITHMS)
    x       = np.arange(n_envs)
    width   = 0.18
    offsets = np.linspace(-(n_algs - 1) / 2 * width, (n_algs - 1) / 2 * width, n_algs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for i, alg in enumerate(ALGORITHMS):
        mean_rewards  = []
        std_rewards   = []
        success_rates = []

        for env_name in ENVIRONMENTS:
            entry = merged.get(env_name, {}).get(alg)
            stats = entry["stats"] if entry else None
            if stats is not None:
                mean_rewards.append(stats["mean"])
                std_rewards.append(stats["std"])
                success_rates.append(stats["success_rate"] * 100)
            else:
                mean_rewards.append(0.0)
                std_rewards.append(0.0)
                success_rates.append(0.0)

        colour = ALGORITHM_COLOURS.get(alg, "steelblue")

        ax1.bar(x + offsets[i], mean_rewards, width=width, yerr=std_rewards,
                label=alg, color=colour, capsize=4, alpha=0.85,
                edgecolor="black", linewidth=0.7)

        ax2.bar(x + offsets[i], success_rates, width=width,
                label=alg, color=colour, alpha=0.85,
                edgecolor="black", linewidth=0.7)

    # --- Mean reward chart ---
    ax1.axhline(y=200, color="red", linestyle="--", linewidth=1.2,
                label="Solved threshold (200)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([ENV_LABELS[e] for e in ENVIRONMENTS], fontsize=10)
    ax1.set_ylabel("Mean Reward", fontsize=11)
    ax1.set_title("Baseline Performance — Mean Reward", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # --- Success rate chart ---
    ax2.axhline(y=100, color="red", linestyle="--", linewidth=1.2,
                label="100% success")
    ax2.set_xticks(x)
    ax2.set_xticklabels([ENV_LABELS[e] for e in ENVIRONMENTS], fontsize=10)
    ax2.set_ylabel("Success Rate (%)", fontsize=11)
    ax2.set_ylim(0, 115)
    ax2.set_title("Baseline Performance — Success Rate", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_distributions(merged: dict, save_dir: str = DIST_SAVE_DIR):
    """
    Produce one box plot distribution figure per environment (4 total).
    Each figure matches the style of plot_baseline_distributions() in plot.py
    but includes the Ensemble agent as a fourth box alongside DQN, PPO, A2C.
    Jittered scatter points are overlaid on each box.

    Saved to:
        results/plots/ensemble/ensemble_distributions_standard.png
        results/plots/ensemble/ensemble_distributions_gravity.png
        results/plots/ensemble/ensemble_distributions_wind.png
        results/plots/ensemble/ensemble_distributions_noise.png

    Args:
        merged   (dict): Output from load_results().
        save_dir (str):  Directory to save plots into.
    """
    os.makedirs(save_dir, exist_ok=True)

    for env_name in ENVIRONMENTS:
        # Collect rewards and colours for each algorithm present
        algorithms  = []
        rewards_all = []
        colours     = []

        for alg in ALGORITHMS:
            entry = merged.get(env_name, {}).get(alg)
            if entry is not None and "rewards" in entry:
                algorithms.append(alg)
                rewards_all.append(entry["rewards"])
                colours.append(ALGORITHM_COLOURS.get(alg, "steelblue"))

        if not algorithms:
            print(f"  No data for {env_name} — skipping.")
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        bp = ax.boxplot(
            rewards_all,
            patch_artist=True,
            notch=False,
            widths=0.4,
            medianprops=dict(color="black", linewidth=2),
        )

        # Apply algorithm colours to boxes
        for patch, colour in zip(bp["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.75)

        # Jittered scatter overlay - consistent with baseline distribution plots
        for i, (alg, rwd_list) in enumerate(zip(algorithms, rewards_all), start=1):
            x_jitter = np.random.normal(i, 0.05, size=len(rwd_list))
            ax.scatter(x_jitter, rwd_list, alpha=0.25, s=12,
                       color=ALGORITHM_COLOURS.get(alg, "steelblue"))

        # Solved threshold
        ax.axhline(y=200, color="red", linestyle="--", linewidth=1.2,
                   label="Solved threshold (200)")

        ax.set_xticks(range(1, len(algorithms) + 1))
        ax.set_xticklabels(algorithms, fontsize=12)
        ax.set_ylabel("Total Episode Reward", fontsize=11)
        ax.set_title(
            f"Baseline Performance — Reward Distributions\n"
            f"{ENV_LABELS[env_name]} (100 Episodes)",
            fontsize=12
        )
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        save_path = os.path.join(save_dir, f"ensemble_distributions_{env_name}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Distribution plot saved to {save_path}")
        plt.close(fig)


def main():
    merged = load_results()
    plot_comparison(merged, save_path=SAVE_PATH)
    plot_distributions(merged, save_dir=DIST_SAVE_DIR)


if __name__ == "__main__":
    main()