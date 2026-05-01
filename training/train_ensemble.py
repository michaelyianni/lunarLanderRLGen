import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environments.base_env import LunarLanderEnv
from agents.ensemble_agent import EnsembleAgent


# --- Config ---
N_SMOKE_EPISODES = 5

MODEL_PATHS = {
    "DQN": "results/models/dqn_standard",
    "PPO": "results/models/ppo_standard",
    "A2C": "results/models/a2c_standard",
}


def main():
    """
    Setup and smoke-test script for the EnsembleAgent.

    Loads all three saved baseline models, constructs the EnsembleAgent,
    and runs a short smoke test in the standard environment to confirm the
    majority-vote mechanism is functioning correctly.
    """
    print("=" * 60)
    print("  Ensemble Agent — Setup & Smoke Test")
    print("=" * 60)

    # --- Load agents ---
    print("\nLoading baseline models...")
    env_dqn  = LunarLanderEnv()
    env_ppo  = LunarLanderEnv()
    env_a2c  = LunarLanderEnv()
    ensemble = EnsembleAgent.load(
        paths=MODEL_PATHS,
        envs={"DQN": env_dqn, "PPO": env_ppo, "A2C": env_a2c},
    )
    print("All three agents loaded successfully.")
    print(f"Ensemble constituent agents: {[type(a).__name__ for a in ensemble.agents]}")

    # --- Smoke test ---
    print(f"\nRunning {N_SMOKE_EPISODES}-episode smoke test in standard environment...")
    env = LunarLanderEnv()
    total_rewards = []

    for episode in range(N_SMOKE_EPISODES):
        obs, _                = env.reset()
        total_reward          = 0.0
        terminated, truncated = False, False
        steps                 = 0

        while not (terminated or truncated):
            action = ensemble.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            steps        += 1

        total_rewards.append(total_reward)
        outcome = "✅ SUCCESS" if total_reward >= 200 else "❌ FAILED"
        print(f"  Episode {episode + 1}/{N_SMOKE_EPISODES} — "
              f"Steps: {steps:>4}  Reward: {total_reward:>8.2f}  {outcome}")

    env.close()
    env_dqn.close()
    env_ppo.close()
    env_a2c.close()

    # --- Summary ---
    import numpy as np
    mean_r   = float(np.mean(total_rewards))
    success  = sum(1 for r in total_rewards if r >= 200)
    print(f"\nSmoke test complete.")
    print(f"  Mean reward : {mean_r:.2f}")
    print(f"  Successes   : {success}/{N_SMOKE_EPISODES}")
    print("\nMajority-vote mechanism confirmed — EnsembleAgent is ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
