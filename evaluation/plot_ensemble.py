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
SAVE_PATH         = "results/plots/ensemble/ensemble_comparison.png"

GRAVITY          = -13.0
WIND_POWER       = 15.0
NOISE_STD        = 0.075

ENVIRONMENTS = ["standard", "gravity", "wind", "noise"]
ENV_LABELS   = {
    "standard": "Standard",
    "gravity" : f"Gravity\n(g={GRAVITY})",
    "wind"    : f"Wind\n(power={WIND_POWER})",
    "noise"   : f"Noise\n(std={NOISE_STD})",
}

ALGORITHMS = ["DQN", "PPO", "A2C", "Ensemble"]


def load_results(ensemble_path: str, baseline_path: str) -> dict:
    """
    Load and merge ensemble and baseline result logs into a unified structure.

    Args:
        ensemble_path (str): Path to ensemble_results.json.
        baseline_path (str): Path to baseline_results.json.

    Returns:
        dict: Nested dict keyed by environment → algorithm → stats dict.
    """
    with open(ensemble_path, "r") as f:
        ensemble_log = json.load(f)
    with open(baseline_path, "r") as f:
        baseline_log = json.load(f)

    # Ensemble results cover all four environments
    merged = {}
    for env_name in ENVIRONMENTS:
        merged[env_name] = {}

        # Baseline contains DQN/PPO/A2C for the standard environment only
        if env_name == "standard":
            for alg in ["DQN", "PPO", "A2C"]:
                merged[env_name][alg] = baseline_log["results"][alg]["stats"]

        # Ensemble results for all environments
        if env_name in ensemble_log["results"]:
            merged[env_name]["Ensemble"] = ensemble_log["results"][env_name]["stats"]

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
    n_envs   = len(ENVIRONMENTS)
    n_algs   = len(ALGORITHMS)
    x        = np.arange(n_envs)
    width    = 0.18
    offsets  = np.linspace(-(n_algs - 1) / 2 * width, (n_algs - 1) / 2 * width, n_algs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for i, alg in enumerate(ALGORITHMS):
        mean_rewards  = []
        std_rewards   = []
        success_rates = []

        for env_name in ENVIRONMENTS:
            stats = merged.get(env_name, {}).get(alg)
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
    ax1.set_title("Ensemble vs Baselines — Mean Reward", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # --- Success rate chart ---
    ax2.axhline(y=100, color="red", linestyle="--", linewidth=1.2,
                label="100% success")
    ax2.set_xticks(x)
    ax2.set_xticklabels([ENV_LABELS[e] for e in ENVIRONMENTS], fontsize=10)
    ax2.set_ylabel("Success Rate (%)", fontsize=11)
    ax2.set_ylim(0, 115)
    ax2.set_title("Ensemble vs Baselines — Success Rate", fontsize=12)
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


def main():
    merged = load_results(ENSEMBLE_LOG_PATH, BASELINE_LOG_PATH)
    plot_comparison(merged, save_path=SAVE_PATH)


if __name__ == "__main__":
    main()
