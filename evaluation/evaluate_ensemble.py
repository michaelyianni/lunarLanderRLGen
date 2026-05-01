import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
from environments.base_env import LunarLanderEnv
from environments.gravity_env import GravityEnv
from environments.wind_env import WindEnv
from environments.noise_env import NoiseEnv
from agents.ensemble_agent import EnsembleAgent


# --- Config ---
N_EVAL_EPISODES  = 100
SEED             = 42
GRAVITY          = -13.0
WIND_POWER       = 15.0
TURBULENCE_POWER = 1.5
NOISE_STD        = 0.075

MODEL_PATHS = {
    "DQN": "results/models/dqn_standard",
    "PPO": "results/models/ppo_standard",
    "A2C": "results/models/a2c_standard",
}

LOG_PATH = "results/logs/ensemble/ensemble_results.json"


def evaluate_agent(agent, env, n_episodes: int) -> list:
    """
    Run a frozen agent for n_episodes and return the list of total rewards.

    Args:
        agent:       A loaded EnsembleAgent (or any agent with a predict() method).
        env:         The environment instance to evaluate in.
        n_episodes:  Number of episodes to run.

    Returns:
        rewards (list): Total reward for each episode.
    """
    rewards = []

    for episode in range(n_episodes):
        obs, _                = env.reset()
        total_reward          = 0.0
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


def _build_ensemble(env_standard) -> EnsembleAgent:
    """
    Helper: construct the EnsembleAgent using the standard environment for all
    three constituent agents (models are environment-agnostic at inference time).

    Args:
        env_standard: A LunarLanderEnv instance used to load the DQN model.

    Returns:
        EnsembleAgent: Ready-to-use ensemble.
    """
    envs = {
        "DQN": env_standard,
        "PPO": LunarLanderEnv(),
        "A2C": LunarLanderEnv(),
    }
    return EnsembleAgent.load(paths=MODEL_PATHS, envs=envs)


def main():
    all_rewards = {}
    all_stats   = {}

    # --- Standard environment ---
    print("\nEvaluating Ensemble in standard environment...")
    env_std   = LunarLanderEnv()
    ensemble  = _build_ensemble(env_std)
    rewards   = evaluate_agent(ensemble, env_std, N_EVAL_EPISODES)
    stats     = compute_stats(rewards)
    env_std.close()
    all_rewards["standard"] = rewards
    all_stats["standard"]   = stats

    # --- Gravity environment ---
    print(f"\nEvaluating Ensemble in gravity environment (g={GRAVITY})...")
    env_grav  = GravityEnv(gravity=GRAVITY)
    rewards   = evaluate_agent(ensemble, env_grav, N_EVAL_EPISODES)
    stats     = compute_stats(rewards)
    env_grav.close()
    all_rewards["gravity"] = rewards
    all_stats["gravity"]   = stats

    # --- Wind environment ---
    print(f"\nEvaluating Ensemble in wind environment (wind={WIND_POWER})...")
    env_wind  = WindEnv(wind_power=WIND_POWER, turbulence_power=TURBULENCE_POWER)
    rewards   = evaluate_agent(ensemble, env_wind, N_EVAL_EPISODES)
    stats     = compute_stats(rewards)
    env_wind.close()
    all_rewards["wind"] = rewards
    all_stats["wind"]   = stats

    # --- Noise environment ---
    print(f"\nEvaluating Ensemble in noise environment (noise_std={NOISE_STD})...")
    env_noise = NoiseEnv(noise_std=NOISE_STD, seed=SEED)
    rewards   = evaluate_agent(ensemble, env_noise, N_EVAL_EPISODES)
    stats     = compute_stats(rewards)
    env_noise.close()
    all_rewards["noise"] = rewards
    all_stats["noise"]   = stats

    # --- Print summary table ---
    print("\n" + "=" * 55)
    print(f"{'Environment':<16} {'Mean':>8} {'Std':>8} {'Median':>8} {'Success%':>10}")
    print("=" * 55)
    for env_name, stats in all_stats.items():
        print(f"{'Ensemble (' + env_name + ')':<16} {stats['mean']:>8.2f} {stats['std']:>8.2f} "
              f"{stats['median']:>8.2f} {stats['success_rate']*100:>9.1f}%")
    print("=" * 55)

    # --- Save results log ---
    log = {
        "algorithm"      : "Ensemble",
        "n_eval_episodes": N_EVAL_EPISODES,
        "seed"           : SEED,
        "environments"   : {
            "standard" : {"gravity": -10.0},
            "gravity"  : {"gravity": GRAVITY},
            "wind"     : {"wind_power": WIND_POWER, "turbulence_power": TURBULENCE_POWER},
            "noise"    : {"noise_std": NOISE_STD},
        },
        "results": {
            env_name: {"stats": all_stats[env_name], "rewards": all_rewards[env_name]}
            for env_name in all_stats
        },
    }
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=4)
    print(f"\nEnsemble log saved to {LOG_PATH}")


if __name__ == "__main__":
    main()
