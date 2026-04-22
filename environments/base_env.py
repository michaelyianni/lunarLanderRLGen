import gymnasium as gym
import numpy as np


class LunarLanderEnv:
    """
    Base wrapper around the Gymnasium LunarLander-v3 environment.
    Provides a consistent interface for all agents in this project.
    """

    def __init__(self, render_mode=None, **kwargs):
        """
        Args:
            render_mode (str): "human" to visualise, None for training.
            **kwargs: Additional keyword arguments passed to gym.make().
        """
        self.env = gym.make("LunarLander-v3", render_mode=render_mode, **kwargs)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        """Reset the environment and return the initial observation."""
        obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        """
        Take a step in the environment.

        Returns:
            obs        - next observation
            reward     - reward received
            terminated - True if the episode ended (crash or land)
            truncated  - True if the episode was cut short (e.g. time limit)
            info       - diagnostic info dict
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment (only works if render_mode='human')."""
        self.env.render()

    def close(self):
        """Close the environment and release resources."""
        self.env.close()

    @property
    def n_observations(self):
        """Number of values in the observation vector."""
        if hasattr(self.observation_space, 'shape') and self.observation_space.shape:
            return self.observation_space.shape[0]
        return 1

    @property
    def n_actions(self):
        """Number of discrete actions available."""
        if hasattr(self.action_space, 'shape') and self.action_space.shape:
            return self.action_space.shape[0]
        return self.action_space.shape