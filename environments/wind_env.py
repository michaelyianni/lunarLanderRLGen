import gymnasium as gym
from environments.base_env import LunarLanderEnv


class WindEnv(LunarLanderEnv):
    """
    LunarLander environment with wind disturbance enabled.
    Inherits all behaviour from LunarLanderEnv — wind parameters
    are passed to gym.make().

    Wind is disabled by default in LunarLander-v3.
    Here we enable it with a meaningful wind_power and turbulence_power
    to stress-test agent robustness.

    Default wind_power       : 0.0  (disabled)
    Modified wind_power      : 15.0
    Modified turbulence_power: 1.5
    """

    def __init__(
        self,
        render_mode=None,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
    ):
        """
        Args:
            render_mode       (str):   "human" to visualise, None otherwise.
            wind_power        (float): Magnitude of wind force. Default 15.0.
            turbulence_power  (float): Magnitude of turbulence. Default 1.5.
        """
        self.wind_power       = wind_power
        self.turbulence_power = turbulence_power
        self.env = gym.make(
            "LunarLander-v3",
            render_mode=render_mode,
            enable_wind=True,
            wind_power=wind_power,
            turbulence_power=turbulence_power,
        )
        self.observation_space = self.env.observation_space
        self.action_space      = self.env.action_space