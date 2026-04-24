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
DQN_CURVE = "results/logs/training/dqn_learning_curve.json"
PPO_CURVE = "results/logs/training/ppo_learning_curve.json"
A2C_CURVE = "results/logs/training/a2c_learning_curve.json"

# Training summary logs
DQN_LOG   = "results/logs/training/dqn_training_log.json"
PPO_LOG   = "results/logs/training/ppo_training_log.json"
A2C_LOG   = "results/logs/training/a2c_training_log.json"


def run_dqn_only():
    """Run plots using only DQN results."""
    print("Plotting DQN learning curve...")
    plot_single_learning_curve(
        curve_path=DQN_CURVE,
        algorithm="DQN",
        save_path="results/plots/training/dqn_learning_curve.png"
    )
    
def run_a2c_only():
    """Run plots using only A2C results."""
    print("Plotting A2C learning curve...")
    plot_single_learning_curve(
        curve_path=A2C_CURVE,
        algorithm="A2C",
        save_path="results/plots/training/a2c_learning_curve.png"
    )
    
def run_ppo_only():
    """Run plots using only PPO results."""
    print("Plotting PPO learning curve...")
    plot_single_learning_curve(
        curve_path=PPO_CURVE,
        algorithm="PPO",
        save_path="results/plots/training/ppo_learning_curve.png"
    )


def run_full_comparison():
    """Run all comparison plots once all three agents are trained."""
    print("Plotting comparison learning curves...")
    plot_comparison_learning_curves(
        curve_paths={"DQN": DQN_CURVE, "PPO": PPO_CURVE, "A2C": A2C_CURVE},
        environment="Standard",
        save_path="results/plots/training/comparison_learning_curves.png"
    )

    # print("Plotting final performance bar charts...")
    # plot_final_performance_bar(
    #     log_paths={"DQN": DQN_LOG, "PPO": PPO_LOG, "A2C": A2C_LOG},
    #     environment="Standard",
    #     save_path="results/plots/training/comparison_performance_bar.png"
    # )


if __name__ == "__main__":
    
    # run_dqn_only()
    # run_ppo_only()
    # run_a2c_only()

    run_full_comparison()