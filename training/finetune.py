import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np

from environments.gravity_env import GravityEnv
from environments.wind_env    import WindEnv
from environments.noise_env   import NoiseEnv

from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.a2c_agent import A2CAgent
from training.callbacks import EpisodeRewardCallback


# --- Config ---

FINETUNE_TIMESTEPS = 200_000   # Budget for fine-tuning and scratch runs
EVAL_EPISODES      = 20
SEED               = 42

BASELINE_MODEL_PATHS = {
    "DQN": "results/models/dqn_standard",
    "PPO": "results/models/ppo_standard",
    "A2C": "results/models/a2c_standard",
}

AGENT_CLASSES = {
    "DQN": DQNAgent,
    "PPO": PPOAgent,
    "A2C": A2CAgent,
}


def make_env(env_name: str):
    """Instantiate a modified environment by name."""
    if env_name == "gravity":
        return GravityEnv(gravity=-13.0)
    elif env_name == "wind":
        return WindEnv(wind_power=15.0, turbulence_power=1.5)
    elif env_name == "noise":
        return NoiseEnv(noise_std=0.075, seed=SEED)
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def run_finetune(algorithm: str, env_name: str):
    """
    Load the baseline model, swap the environment, and continue training.
    Saves the fine-tuned model and its learning curve.
    """
    print(f"\n  Fine-tuning {algorithm} -> {env_name}...")

    env        = make_env(env_name)
    AgentClass = AGENT_CLASSES[algorithm]
    agent      = AgentClass(env=env, seed=SEED)

    # Load baseline weights — this is the key transfer learning step
    agent.load(BASELINE_MODEL_PATHS[algorithm])

    # Swap environment so the loaded model trains in the modified env
    agent.model.set_env(env.env)

    callback = EpisodeRewardCallback()
    agent.train(total_timesteps=FINETUNE_TIMESTEPS, callback=callback)

    # Save fine-tuned model
    model_path = f"results/models/transfer/{env_name}/{algorithm.lower()}_{env_name}_finetuned"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    agent.save(model_path)

    # Save learning curve
    curve_path = f"results/logs/transfer/{env_name}/{algorithm.lower()}_{env_name}_finetune_curve.json"
    os.makedirs(os.path.dirname(curve_path), exist_ok=True)
    with open(curve_path, "w") as f:
        json.dump({
            "algorithm"       : algorithm,
            "environment"     : env_name,
            "mode"            : "finetune",
            "total_timesteps" : FINETUNE_TIMESTEPS,
            "episode_rewards" : callback.episode_rewards,
            "episode_lengths" : callback.episode_lengths,
        }, f, indent=4)

    env.close()
    print(f"    SUCCESS: Saved to {model_path}")


def run_scratch(algorithm: str, env_name: str):
    """
    Train a fresh agent from random initialisation in the modified environment.
    This is the control - used to assess whether pre-training actually helps.
    """
    print(f"\n  Scratch training {algorithm} -> {env_name}...")

    env        = make_env(env_name)
    AgentClass = AGENT_CLASSES[algorithm]
    agent      = AgentClass(env=env, seed=SEED)   # No load - fresh weights

    callback = EpisodeRewardCallback()
    agent.train(total_timesteps=FINETUNE_TIMESTEPS, callback=callback)

    model_path = f"results/models/transfer/{env_name}/{algorithm.lower()}_{env_name}_scratch"
    agent.save(model_path)

    curve_path = f"results/logs/transfer/{env_name}/{algorithm.lower()}_{env_name}_scratch_curve.json"
    os.makedirs(os.path.dirname(curve_path), exist_ok=True)
    with open(curve_path, "w") as f:
        json.dump({
            "algorithm"       : algorithm,
            "environment"     : env_name,
            "mode"            : "scratch",
            "total_timesteps" : FINETUNE_TIMESTEPS,
            "episode_rewards" : callback.episode_rewards,
            "episode_lengths" : callback.episode_lengths,
        }, f, indent=4)

    env.close()
    print(f"    SUCCESS: Saved to {model_path}")


def main():
    algorithms   = ["DQN", "PPO", "A2C"]
    environments = ["gravity", "wind", "noise"]

    print("=" * 60)
    print("  Transfer Learning — Fine-tuning & Scratch Controls")
    print(f"  Timesteps per run : {FINETUNE_TIMESTEPS:,}")
    print(f"  Total runs        : {len(algorithms) * len(environments) * 2}")
    print("=" * 60)

    for env_name in environments:
        for algorithm in algorithms:
            run_finetune(algorithm, env_name)
            run_scratch(algorithm, env_name)
    
    
    # Or run a single algorithm
    
    # algorithm = "A2C"
    
    # for env_name in environments:
        
    #     run_finetune(algorithm, env_name)
    #     run_scratch(algorithm, env_name)

    print("\n SUCCESS: All fine-tuning and scratch runs complete.")


if __name__ == "__main__":
    main()