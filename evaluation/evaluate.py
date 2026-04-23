import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.plot import (
    plot_single_learning_curve,
    plot_comparison_learning_curves,
    plot_final_performance_bar,
)

# --- File paths ---
# Learning curve data (from training callbacks)
DQN_CURVE = "results/logs/dqn_learning_curve.json"
PPO_CURVE = "results/logs/ppo_learning_curve.json"
A2C_CURVE = "results/logs/a2c_learning_curve.json"

# Training summary logs
DQN_LOG   = "results/logs/dqn_training_log.json"
PPO_LOG   = "results/logs/ppo_training_log.json"
A2C_LOG   = "results/logs/a2c_training_log.json"


def run_dqn_only():
    """Run plots using only DQN results (for use before PPO/A2C are trained)."""
    print("Plotting DQN learning curve...")
    plot_single_learning_curve(
        curve_path=DQN_CURVE,
        algorithm="DQN",
        save_path="results/plots/dqn_learning_curve.png"
    )


def run_full_comparison():
    """Run all comparison plots once all three agents are trained."""
    print("Plotting comparison learning curves...")
    plot_comparison_learning_curves(
        curve_paths={"DQN": DQN_CURVE, "PPO": PPO_CURVE, "A2C": A2C_CURVE},
        environment="Standard",
        save_path="results/plots/comparison_learning_curves.png"
    )

    print("Plotting final performance bar charts...")
    plot_final_performance_bar(
        log_paths={"DQN": DQN_LOG, "PPO": PPO_LOG, "A2C": A2C_LOG},
        environment="Standard",
        save_path="results/plots/comparison_performance_bar.png"
    )


if __name__ == "__main__":
    # Right now only DQN is trained — run the single agent plot
    # run_dqn_only()

    # Uncomment the line below once PPO and A2C are also trained:
    run_full_comparison()