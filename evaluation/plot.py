import json
import os
import matplotlib.pyplot as plt
import numpy as np
from evaluation.metrics import smooth_rewards, compute_convergence_episode


# --- Consistent style across all plots ---
ALGORITHM_COLOURS = {
    "DQN"     : "#1f77b4",   # Blue
    "PPO"     : "#ff7f0e",   # Orange
    "A2C"     : "#2ca02c",   # Green
    "Ensemble": "#9467bd",   # Purple
}

SMOOTHING_WINDOW = 50


def load_curve(path: str) -> dict:
    """
    Load a learning curve JSON file saved during training.

    Args:
        path (str): Path to the JSON file.

    Returns:
        dict with keys 'episode_rewards' and 'episode_lengths'.
    """
    with open(path, "r") as f:
        return json.load(f)


def plot_single_learning_curve(curve_path: str, algorithm: str, save_path: str = None):
    """
    Plots the learning curve (reward vs episodes) for a single algorithm.
    Shows both the raw noisy rewards and the smoothed trend line.

    Args:
        curve_path (str): Path to the learning curve JSON file.
        algorithm  (str): Algorithm name (e.g. "DQN") used for labels/colours.
        save_path  (str): If provided, saves the figure to this path.
    """
    data     = load_curve(curve_path)
    rewards  = data["episode_rewards"]
    episodes = list(range(1, len(rewards) + 1))
    smoothed = smooth_rewards(rewards, SMOOTHING_WINDOW)
    colour   = ALGORITHM_COLOURS.get(algorithm, "steelblue")

    convergence = compute_convergence_episode(rewards)
    
    print(f"{algorithm} convergence episode: {convergence if convergence != -1 else 'Not converged'}")

    fig, ax = plt.subplots(figsize=(10, 5))

    # Raw rewards (faint)
    ax.plot(episodes, rewards, color=colour, alpha=0.2, linewidth=0.8, label="Raw reward")

    # Smoothed trend
    ax.plot(episodes, smoothed, color=colour, linewidth=2.0, label=f"Smoothed (window={SMOOTHING_WINDOW})")

    # Solved threshold line
    ax.axhline(y=200, color="red", linestyle="--", linewidth=1.2, label="Solved threshold (200)")

    # Convergence marker
    if convergence != -1:
        ax.axvline(x=convergence, color="grey", linestyle=":", linewidth=1.2,
                   label=f"Convergence episode ≈ {convergence}")

    ax.set_title(f"{algorithm} — Learning Curve (Standard Environment)", fontsize=13)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Total Reward", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_comparison_learning_curves(curve_paths: dict, environment: str = "Standard", save_path: str = None):
    """
    Plots smoothed learning curves for multiple algorithms on the same axes.
    Used for direct comparison between DQN, PPO, and A2C.

    Args:
        curve_paths (dict): {"DQN": path, "PPO": path, "A2C": path}
        environment (str):  Environment name used in the plot title.
        save_path   (str):  If provided, saves the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    for algorithm, path in curve_paths.items():
        if path is None:
            # Ensemble (and any agent without a learning curve) is skipped silently
            continue
        data     = load_curve(path)
        rewards  = data["episode_rewards"]
        episodes = list(range(1, len(rewards) + 1))
        smoothed = smooth_rewards(rewards, SMOOTHING_WINDOW)
        colour   = ALGORITHM_COLOURS.get(algorithm, None)

        ax.plot(episodes, smoothed, color=colour, linewidth=2.0, label=algorithm)

    ax.axhline(y=200, color="red", linestyle="--", linewidth=1.2, label="Solved threshold (200)")
    ax.set_title(f"Learning Curve Comparison — {environment} Environment", fontsize=13)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Total Reward (Smoothed)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_final_performance_bar(log_paths: dict, environment: str = "Standard", save_path: str = None):
    """
    Bar chart comparing mean final reward and success rate across algorithms.
    Uses the training log JSON files (not learning curve files).

    Args:
        log_paths   (dict): {"DQN": path, "PPO": path, "A2C": path}
        environment (str):  Environment name used in the plot title.
        save_path   (str):  If provided, saves the figure to this path.
    """
    algorithms   = []
    mean_rewards = []
    std_rewards  = []
    success_rates = []

    for algorithm, path in log_paths.items():
        with open(path, "r") as f:
            log = json.load(f)
        algorithms.append(algorithm)
        mean_rewards.append(log["mean_reward"])
        std_rewards.append(log["std_reward"])
        success_rates.append(log["success_rate"] * 100)

    x      = np.arange(len(algorithms))
    colours = [ALGORITHM_COLOURS.get(a, "steelblue") for a in algorithms]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Mean reward bar chart
    bars = ax1.bar(x, mean_rewards, yerr=std_rewards, color=colours,
                   capsize=6, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax1.axhline(y=200, color="red", linestyle="--", linewidth=1.2, label="Solved threshold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, fontsize=11)
    ax1.set_ylabel("Mean Reward", fontsize=11)
    ax1.set_title(f"Mean Reward — {environment} Environment", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Success rate bar chart
    ax2.bar(x, success_rates, color=colours, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax2.axhline(y=100, color="red", linestyle="--", linewidth=1.2, label="100% success")
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms, fontsize=11)
    ax2.set_ylabel("Success Rate (%)", fontsize=11)
    ax2.set_ylim(0, 110)
    ax2.set_title(f"Success Rate — {environment} Environment", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)
    
def plot_baseline_distributions(results: dict, save_path: str = None):
    """
    Box plot showing the reward distribution across episodes for each
    algorithm in the standard environment. Used to establish baseline performance.

    Args:
        results   (dict): {"DQN": [rewards], "PPO": [rewards], "A2C": [rewards]}
        save_path (str):  If provided, saves the figure to this path.
    """
    algorithms = list(results.keys())
    rewards    = [results[a] for a in algorithms]
    colours    = [ALGORITHM_COLOURS.get(a, "steelblue") for a in algorithms]

    fig, ax = plt.subplots(figsize=(9, 6))

    bp = ax.boxplot(
        rewards,
        patch_artist=True,
        notch=False,
        widths=0.4,
        medianprops=dict(color="black", linewidth=2),
    )

    # Apply algorithm colours to boxes
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)

    # Overlay individual episode rewards as a scatter (shows raw spread)
    for i, (alg, rwd_list) in enumerate(results.items(), start=1):
        x_jitter = np.random.normal(i, 0.05, size=len(rwd_list))
        ax.scatter(x_jitter, rwd_list, alpha=0.25, s=12,
                   color=ALGORITHM_COLOURS.get(alg, "steelblue"))

    ax.axhline(y=200, color="red", linestyle="--", linewidth=1.2,
               label="Solved threshold (200)")

    ax.set_xticks(range(1, len(algorithms) + 1))
    ax.set_xticklabels(algorithms, fontsize=12)
    ax.set_ylabel("Total Episode Reward", fontsize=11)
    ax.set_title("Baseline Performance — Standard Environment (100 Episodes)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)


def _save_or_show(fig, save_path: str = None):
    """Helper: save figure to disk or display it interactively."""
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()