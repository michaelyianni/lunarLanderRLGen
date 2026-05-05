import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np

from environments.gravity_env import GravityEnv
from environments.wind_env    import WindEnv
from environments.noise_env   import NoiseEnv

from agents.dqn_agent     import DQNAgent
from agents.ppo_agent     import PPOAgent
from agents.a2c_agent     import A2CAgent
from agents.ensemble_agent import EnsembleAgent


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

N_EVAL_EPISODES = 100
SEED            = 42

ALGORITHMS   = ["DQN", "PPO", "A2C"]
ENVIRONMENTS = ["gravity", "wind", "noise"]
MODES        = ["finetuned", "scratch"]

AGENT_CLASSES = {
    "DQN": DQNAgent,
    "PPO": PPOAgent,
    "A2C": A2CAgent,
}

# Frozen ensemble results are already computed — read from existing log
ENSEMBLE_FROZEN_LOG = "results/logs/ensemble/ensemble_results.json"

LOG_PATH = "results/logs/transfer/transfer_results.json"


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def make_env(env_name: str):
    """Instantiate the appropriate modified environment."""
    if env_name == "gravity":
        return GravityEnv(gravity=-13.0)
    elif env_name == "wind":
        return WindEnv(wind_power=15.0, turbulence_power=1.5)
    elif env_name == "noise":
        return NoiseEnv(noise_std=0.075, seed=SEED)


def evaluate_agent(agent, env, n_episodes: int) -> list:
    """Run an agent for n_episodes and return the list of total rewards."""
    rewards = []
    for _ in range(n_episodes):
        obs, _       = env.reset()
        total_reward = 0.0
        terminated   = truncated = False
        while not (terminated or truncated):
            action                                = agent.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward                         += float(reward)
        rewards.append(total_reward)
    return rewards


def compute_stats(rewards: list) -> dict:
    """Compute summary statistics for a list of episode rewards."""
    arr = np.array(rewards)
    return {
        "mean"        : float(np.mean(arr)),
        "std"         : float(np.std(arr)),
        "median"      : float(np.median(arr)),
        "min"         : float(np.min(arr)),
        "max"         : float(np.max(arr)),
        "success_rate": float(np.mean(arr >= 200)),
    }


def build_ensemble(env_name: str, mode: str) -> tuple:
    """
    Assemble an EnsembleAgent from three constituent models.
    Each constituent is loaded from its fine-tuned or scratch transfer model.

    Args:
        env_name (str): Environment name ("gravity", "wind", or "noise").
        mode     (str): "finetuned" or "scratch".

    Returns:
        tuple: (EnsembleAgent, list of constituent envs to close after evaluation)
    """
    constituent_envs   = []
    constituent_agents = []

    for alg_name, AgentClass in AGENT_CLASSES.items():
        env        = make_env(env_name)
        agent      = AgentClass(env=env, seed=SEED)
        model_path = (f"results/models/transfer/{env_name}/"
                      f"{alg_name.lower()}_{env_name}_{mode}")
        agent.load(model_path)
        constituent_agents.append(agent)
        constituent_envs.append(env)

    ensemble = EnsembleAgent(agents=constituent_agents)
    return ensemble, constituent_envs


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Transfer Learning — Evaluation")
    print("=" * 60)

    # keyed: env_name → algorithm → mode → {stats, rewards}
    all_results = {}

    # ── Individual algorithms (DQN, PPO, A2C) ───────────────────
    for env_name in ENVIRONMENTS:
        all_results[env_name] = {}

        for algorithm in ALGORITHMS:
            all_results[env_name][algorithm] = {}

            for mode in MODES:
                model_path = (f"results/models/transfer/{env_name}/"
                              f"{algorithm.lower()}_{env_name}_{mode}")
                print(f"\n  Evaluating {algorithm} [{mode}] in {env_name}...")

                env        = make_env(env_name)
                AgentClass = AGENT_CLASSES[algorithm]
                agent      = AgentClass(env=env, seed=SEED)
                agent.load(model_path)

                rewards = evaluate_agent(agent, env, N_EVAL_EPISODES)
                stats   = compute_stats(rewards)
                env.close()

                all_results[env_name][algorithm][mode] = {
                    "stats"  : stats,
                    "rewards": rewards,
                }

                print(f"    Mean: {stats['mean']:.2f}  "
                      f"Std: {stats['std']:.2f}  "
                      f"Success: {stats['success_rate']*100:.1f}%")

    # ── Ensemble (fine-tuned and scratch) ────────────────────────
    for env_name in ENVIRONMENTS:
        all_results[env_name]["Ensemble"] = {}

        for mode in MODES:
            print(f"\n  Evaluating Ensemble [{mode}] in {env_name}...")

            # Build ensemble from constituent transfer models
            ensemble, constituent_envs = build_ensemble(env_name, mode)

            # Evaluation environment (separate from constituent envs)
            eval_env = make_env(env_name)
            rewards  = evaluate_agent(ensemble, eval_env, N_EVAL_EPISODES)
            stats    = compute_stats(rewards)
            eval_env.close()

            # Close constituent envs
            for env in constituent_envs:
                env.close()

            all_results[env_name]["Ensemble"][mode] = {
                "stats"  : stats,
                "rewards": rewards,
            }

            print(f"    Mean: {stats['mean']:.2f}  "
                  f"Std: {stats['std']:.2f}  "
                  f"Success: {stats['success_rate']*100:.1f}%")

    # ── Load frozen Ensemble results from existing log ───────────
    print("\n  Loading frozen Ensemble results from ensemble log...")
    try:
        with open(ENSEMBLE_FROZEN_LOG, "r") as f:
            ensemble_log = json.load(f)
        for env_name in ENVIRONMENTS:
            frozen_stats   = ensemble_log["results"][env_name]["stats"]
            frozen_rewards = ensemble_log["results"][env_name]["rewards"]
            all_results[env_name]["Ensemble"]["frozen"] = {
                "stats"  : frozen_stats,
                "rewards": frozen_rewards,
            }
            print(f"    ✔ Frozen Ensemble loaded for {env_name}  "
                  f"(Success: {frozen_stats['success_rate']*100:.1f}%)")
    except (FileNotFoundError, KeyError) as e:
        print(f"    WARNING: Could not load frozen ensemble results — {e}")

    # ── Summary table ────────────────────────────────────────────
    all_algorithms = ALGORITHMS + ["Ensemble"]
    all_modes      = MODES + ["frozen"]

    print("\n" + "=" * 75)
    print(f"{'Env':<10} {'Algorithm':<10} {'Mode':<12} "
          f"{'Mean':>8} {'Std':>8} {'Success%':>10}")
    print("=" * 75)
    for env_name in ENVIRONMENTS:
        for algorithm in all_algorithms:
            for mode in all_modes:
                try:
                    s = all_results[env_name][algorithm][mode]["stats"]
                    print(f"{env_name:<10} {algorithm:<10} {mode:<12} "
                          f"{s['mean']:>8.2f} {s['std']:>8.2f} "
                          f"{s['success_rate']*100:>9.1f}%")
                except KeyError:
                    pass
    print("=" * 75)

    # ── Save ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump({
            "n_eval_episodes": N_EVAL_EPISODES,
            "seed"           : SEED,
            "results"        : all_results,
        }, f, indent=4)
    print(f"\nTransfer results saved to {LOG_PATH}")


if __name__ == "__main__":
    main()