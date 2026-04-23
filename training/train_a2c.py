import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import json
from environments.base_env import LunarLanderEnv
from agents.a2c_agent import A2CAgent
from training.callbacks import EpisodeRewardCallback


TOTAL_TIMESTEPS  = 500_000
EVAL_EPISODES    = 20
SEED             = 42
MODEL_SAVE_PATH  = "results/models/a2c_standard"
LOG_SAVE_PATH    = "results/logs/a2c_training_log.json"
CURVE_SAVE_PATH  = "results/logs/a2c_learning_curve.json"


def evaluate(agent: A2CAgent, env: LunarLanderEnv, n_episodes: int = EVAL_EPISODES):
    """
    Evaluate a trained agent over a number of episodes.

    Args:
        agent      (A2CAgent):       The trained agent to evaluate.
        env        (LunarLanderEnv): The environment to evaluate in.
        n_episodes (int):            Number of evaluation episodes.

    Returns:
        mean_reward  (float): Mean total reward across episodes.
        std_reward   (float): Std deviation of rewards.
        success_rate (float): Proportion of episodes with reward >= 200.
    """
    rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        terminated, truncated = False, False

        while not (terminated or truncated):
            action = agent.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

        rewards.append(total_reward)

    mean_reward  = float(np.mean(rewards))
    std_reward   = float(np.std(rewards))
    success_rate = float(np.sum(np.array(rewards) >= 200) / n_episodes)

    return mean_reward, std_reward, success_rate


def main():
    # --- Training ---
    train_env = LunarLanderEnv()
    agent = A2CAgent(env=train_env, seed=SEED)

    callback = EpisodeRewardCallback()
    agent.train(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    agent.save(MODEL_SAVE_PATH)
    train_env.close()

    # --- Evaluation ---
    eval_env = LunarLanderEnv()
    mean_reward, std_reward, success_rate = evaluate(agent, eval_env)
    eval_env.close()

    # --- Results ---
    print("\n--- A2C Evaluation Results (Standard Environment) ---")
    print(f"Mean Reward  : {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Success Rate : {success_rate * 100:.1f}%")
    print(f"Episodes     : {len(callback.episode_rewards)}")

    # --- Save training log ---
    log = {
        "algorithm"       : "A2C",
        "environment"     : "standard",
        "total_timesteps" : TOTAL_TIMESTEPS,
        "eval_episodes"   : EVAL_EPISODES,
        "seed"            : SEED,
        "mean_reward"     : mean_reward,
        "std_reward"      : std_reward,
        "success_rate"    : success_rate,
    }
    os.makedirs(os.path.dirname(LOG_SAVE_PATH), exist_ok=True)
    with open(LOG_SAVE_PATH, "w") as f:
        json.dump(log, f, indent=4)
    print(f"Log saved to {LOG_SAVE_PATH}")

    # --- Save learning curve data ---
    curve = {
        "episode_rewards" : callback.episode_rewards,
        "episode_lengths" : callback.episode_lengths,
    }
    with open(CURVE_SAVE_PATH, "w") as f:
        json.dump(curve, f, indent=4)
    print(f"Learning curve data saved to {CURVE_SAVE_PATH}")


if __name__ == "__main__":
    main()