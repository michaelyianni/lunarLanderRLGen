import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time

from environments.base_env import LunarLanderEnv
from environments.gravity_env import GravityEnv
from environments.wind_env import WindEnv
from environments.noise_env import NoiseEnv

from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.a2c_agent import A2CAgent
from agents.ensemble_agent import EnsembleAgent


# --- Config --- (Change these variables to watch different agents/environments)

ALGORITHM   = "DQN"       # "DQN", "PPO", "A2C", "A2CLONG", or "Ensemble"
ENVIRONMENT = "standard"  # "standard", "gravity", "wind", or "noise"
N_EPISODES  = 100           # Number of episodes to watch
SEED        = 42

# Model paths per algorithm
MODEL_PATHS = {
    "DQN": "results/models/dqn_standard",
    "PPO": "results/models/ppo_standard",
    "A2C": "results/models/a2c_standard",
    "A2CLONG": "results/models/a2c_LONG",
}

# Agent classes per algorithm
AGENT_CLASSES = {
    "DQN": DQNAgent,
    "PPO": PPOAgent,
    "A2C": A2CAgent,
    "A2CLONG": A2CAgent
}


# --- Helpers ---

def make_environment(environment: str) -> LunarLanderEnv:
    """
    Creates the appropriate environment with render_mode='human'.
    Extended here in future to support modified environments.

    Args:
        environment (str): Environment name.

    Returns:
        LunarLanderEnv: The rendered environment instance.
    """
    if environment == "standard":
        return LunarLanderEnv(render_mode="human")
    elif environment == "gravity":
        return GravityEnv(render_mode="human", gravity=-13.0)
    elif environment == "wind":
        return WindEnv(render_mode="human", wind_power=15, turbulence_power=1.5)
    elif environment == "noise":
        return NoiseEnv(render_mode="human", noise_std=0.075, seed=SEED)
    else:
        raise ValueError(f"Unknown environment: '{environment}'. "
                         f"Choose from: 'standard', 'gravity', 'wind', 'noise'.")


def load_agent(algorithm: str, env: LunarLanderEnv) -> object:
    """
    Loads the appropriate trained agent from disk.

    Args:
        algorithm (str):    Algorithm name ("DQN", "PPO", "A2C", "A2CLONG", or "Ensemble").
        env (LunarLanderEnv): The environment instance.

    Returns:
        The loaded agent.
    """
    
    if algorithm == "Ensemble":
        print("  Loading constituent agents for Ensemble...")
        agents = []
        for name, AgentClass in AGENT_CLASSES.items():
            constituent_env = make_environment(ENVIRONMENT)
            agent = AgentClass(env=constituent_env, seed=SEED)
            agent.load(MODEL_PATHS[name])
            agents.append(agent)
            print(f"    SUCCESS: {name} loaded")
        return EnsembleAgent(agents=agents)
    
    if algorithm not in AGENT_CLASSES:
        raise ValueError(f"Unknown algorithm: '{algorithm}'. "
                         f"Choose from: {list(AGENT_CLASSES.keys())}")

    AgentClass = AGENT_CLASSES[algorithm]
    model_path = MODEL_PATHS[algorithm]
    agent      = AgentClass(env=env, seed=SEED)
    agent.load(model_path)
    return agent


def watch(algorithm: str, environment: str, n_episodes: int):
    """
    Loads a trained agent and renders it running in the environment.
    Prints a per-step observation summary and episode results to the console.

    Args:
        algorithm   (str): Algorithm to watch ("DQN", "PPO", "A2C", "A2CLONG", or "Ensemble").
        environment (str): Environment to run in.
        n_episodes  (int): Number of episodes to watch.
    """
    print("=" * 55)
    print(f"  Watching: {algorithm} in [{environment}] environment")
    print(f"  Episodes: {n_episodes}")
    print("=" * 55)

    env   = make_environment(environment)
    agent = load_agent(algorithm, env)

    for episode in range(1, n_episodes + 1):
        obs, _     = env.reset()
        total_reward = 0.0
        steps        = 0
        terminated   = False
        truncated    = False

        print(f"\nEpisode {episode}/{n_episodes} - starting...")

        while not (terminated or truncated):
            action               = agent.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward        += float(reward)
            steps               += 1

        # Episode outcome
        outcome = "LANDED" if total_reward >= 200 else "FAILED"
        print(f"  {outcome}  |  Steps: {steps:>4}  |  Total Reward: {total_reward:>8.2f}")

        # Brief pause between episodes so the window doesn't close instantly
        if episode < n_episodes:
            time.sleep(1.0)

    env.close()
    print("\nDone.")


# Entry Point

if __name__ == "__main__":
    watch(
        algorithm=ALGORITHM,
        environment=ENVIRONMENT,
        n_episodes=N_EPISODES,
    )