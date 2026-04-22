import os
from stable_baselines3 import DQN
from environments.base_env import LunarLanderEnv


class DQNAgent:
    """
    DQN Agent for the LunarLander environment.
    Wraps stable-baselines3's DQN with a consistent interface
    shared across all agents in this project.
    """

    def __init__(self, env: LunarLanderEnv, seed: int = 42):
        """
        Args:
            env  (LunarLanderEnv): The environment instance to train on.
            seed (int): Random seed for reproducibility.
        """
        self.env = env
        self.seed = seed
        self.model = DQN(
            policy="MlpPolicy",         # Multi-Layer Perceptron policy - standard feedforward neural network
            env=self.env.env,
            learning_rate=1e-3,         # Standard learning rate for DQN
            buffer_size=50_000,         # Size of the replay buffer - the memory that stores past experiences for training
            learning_starts=1_000,      # Number of random steps before training starts - diversifies experiences
            batch_size=64,              # Number of transitions sampled from the replay buffer per training update
            gamma=0.99,                 # Discount factor for future rewards
            train_freq=4,               # Frequency of training updates (every 4 steps) - balances learning and interaction
            target_update_interval=250, # Frequency of target network updates
            verbose=0,
            seed=seed,
        )

    def train(self, total_timesteps: int = 500_000, callback=None):
        """
        Train the DQN agent.

        Args:
            total_timesteps (int): Total number of environment steps to train for.
            callback: Optional callback function to be called at each step.
        """
        print(f"Training DQN for {total_timesteps:,} timesteps...")
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        print("Training complete.")

    def save(self, path: str):
        """
        Save the trained model to disk.

        Args:
            path (str): File path to save to (without .zip extension).
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}.zip")

    def load(self, path: str):
        """
        Load a previously saved model from disk.

        Args:
            path (str): File path to load from (without .zip extension).
        """
        self.model = DQN.load(path, env=self.env.env)
        print(f"Model loaded from {path}.zip")

    def predict(self, obs):
        """
        Predict an action for a given observation (no exploration).

        Args:
            obs: Observation from the environment.

        Returns:
            action: The action chosen by the agent.
        """
        action, _ = self.model.predict(obs, deterministic=True)
        return action