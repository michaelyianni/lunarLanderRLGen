import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
from environments.gravity_env import GravityEnv
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.a2c_agent import A2CAgent
from evaluation.plot import plot_baseline_distributions, plot_final_performance_bar


# --- Config ---
N_EVAL_EPISODES = 100
SEED            = 42
GRAVITY         = -13.0

MODEL_PATHS = {
    "DQN": "results/models/dqn_standard",
    "PPO": "results/models/ppo_standard",
    "A2C": "results/models/a2c_standard",
}

LOG_PATH  = "results/logs/gravity/gravity_results.json"
PLOT_PATH = "results/plots/gravity/gravity_distributions.png"
BAR_PATH  = "results/plots/gravity/gravity_performance_bar.png"


def evaluate_agent(agent, env, n_episodes: int) -> list:
    """
    Run a frozen agent for n_episodes and return the list of total rewards.

    Args:
        agent:       A loaded DQNAgent, PPOAgent, or A2CAgent.
        env:         The GravityEnv instance.
        n_episodes:  Number of episodes to run.

    Returns:
        rewards (list): Total reward for each episode.
    """
    rewards = []

    for episode in range(n_episodes):
        obs, _               = env.reset()
        total_reward         = 0.0
        terminated, truncated = False, False

        while not (terminated or truncated):
            action = agent.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)

        rewards.append(total_reward)

        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{n_episodes} — Reward: {total_reward:.2f}")

    return rewards


def compute_stats(rewards: list) -> dict:
    """Compute summary statistics for a list of episode rewards."""
    arr = np.array(rewards)
    return {
        "mean"         : float(np.mean(arr)),
        "std"          : float(np.std(arr)),
        "min"          : float(np.min(arr)),
        "max"          : float(np.max(arr)),
        "median"       : float(np.median(arr)),
        "success_rate" : float(np.sum(arr >= 200) / len(arr)),
    }


def main():
    agent_classes = {"DQN": DQNAgent, "PPO": PPOAgent, "A2C": A2CAgent}
    all_rewards   = {}
    all_stats     = {}

    for name, AgentClass in agent_classes.items():
        print(f"\nEvaluating {name} in gravity environment (g={GRAVITY})...")
        env   = GravityEnv(gravity=GRAVITY)
        agent = AgentClass(env=env, seed=SEED)
        agent.load(MODEL_PATHS[name])

        rewards = evaluate_agent(agent, env, N_EVAL_EPISODES)
        stats   = compute_stats(rewards)
        env.close()

        all_rewards[name] = rewards
        all_stats[name]   = stats

        print(f"  Mean Reward  : {stats['mean']:.2f} +/- {stats['std']:.2f}")
        print(f"  Median       : {stats['median']:.2f}")
        print(f"  Min / Max    : {stats['min']:.2f} / {stats['max']:.2f}")
        print(f"  Success Rate : {stats['success_rate'] * 100:.1f}%")

    # --- Summary table ---
    print("\n" + "=" * 55)
    print(f"{'Algorithm':<12} {'Mean':>8} {'Std':>8} {'Median':>8} {'Success%':>10}")
    print("=" * 55)
    for name, stats in all_stats.items():
        print(f"{name:<12} {stats['mean']:>8.2f} {stats['std']:>8.2f} "
              f"{stats['median']:>8.2f} {stats['success_rate']*100:>9.1f}%")
    print("=" * 55)

    # --- Save results log ---
    log = {
        "environment"    : "gravity",
        "gravity"        : GRAVITY,
        "n_eval_episodes": N_EVAL_EPISODES,
        "seed"           : SEED,
        "results"        : {
            name: {"stats": all_stats[name], "rewards": all_rewards[name]}
            for name in all_stats
        },
    }
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=4)
    print(f"\nLog saved to {LOG_PATH}")

    # --- Plots ---
    plot_baseline_distributions(
        results=all_rewards,
        save_path=PLOT_PATH
    )

    temp_log_paths = {}
    for name, stats in all_stats.items():
        path = f"results/logs/gravity/{name.lower()}_gravity_log.json"
        with open(path, "w") as f:
            json.dump({
                "algorithm"   : name,
                "environment" : "gravity",
                "mean_reward" : stats["mean"],
                "std_reward"  : stats["std"],
                "success_rate": stats["success_rate"],
            }, f, indent=4)
        temp_log_paths[name] = path

    plot_final_performance_bar(
        log_paths=temp_log_paths,
        environment="Increased Gravity",
        save_path=BAR_PATH,
    )


if __name__ == "__main__":
    main()