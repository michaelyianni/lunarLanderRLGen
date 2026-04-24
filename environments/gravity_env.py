import gymnasium as gym
from environments.base_env import LunarLanderEnv


class GravityEnv(LunarLanderEnv):
    """
    LunarLander environment with increased gravitational force.
    Inherits all behaviour from LunarLanderEnv — only the gravity
    parameter passed to gym.make() is changed.

    Default gravity : -10.0
    Modified gravity: -13.0
    """

    def __init__(self, render_mode=None, gravity: float = -13.0):
        """
        Args:
            render_mode (str):   "human" to visualise, None otherwise.
            gravity     (float): Gravitational acceleration. Default -13.0
                                 (slightly increased from the standard LunarLander gravity).
        """
        self.gravity = gravity
        self.env = gym.make(
            "LunarLander-v3",
            render_mode=render_mode,
            gravity=-11.9,
        )
        # Override gravity directly on the unwrapped environment
        self.env.unwrapped.gravity = gravity
        
        self.observation_space = self.env.observation_space
        self.action_space      = self.env.action_space