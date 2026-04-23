import os
from stable_baselines3 import PPO
from environments.base_env import LunarLanderEnv


class PPOAgent:
    """
    PPO Agent for the LunarLander environment.
    Wraps stable-baselines3's PPO with a consistent interface
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
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env.env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=0,
            seed=seed,
        )

    def train(self, total_timesteps: int = 500_000, callback=None):
        """
        Train the PPO agent.

        Args:
            total_timesteps (int): Total number of environment steps to train for.
            callback:              Optional SB3 callback (e.g. EpisodeRewardCallback).
        """
        print(f"Training PPO for {total_timesteps:,} timesteps...")
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
        
        # Verify the file was saved correctly
        zip_path = path + ".zip"
        if os.path.exists(zip_path) and os.path.getsize(zip_path) > 1000:
            print(f"Model saved successfully to {zip_path} ({os.path.getsize(zip_path) / 1024:.1f} KB)")
        else:
            print(f"WARNING: Model file at {zip_path} appears missing or corrupt — please re-run training.")

    def load(self, path: str):
        """
        Load a previously saved model from disk.

        Args:
            path (str): File path to load from (without .zip extension).
        """
        self.model = PPO.load(path, env=self.env.env)
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