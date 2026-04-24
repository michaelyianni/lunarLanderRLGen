import numpy as np
from environments.base_env import LunarLanderEnv


class NoiseEnv(LunarLanderEnv):
    """
    LunarLander environment with Gaussian sensor noise injected into
    the observation vector. The lander physics are unchanged — only
    the observations returned to the agent are corrupted.

    This simulates imperfect or unreliable sensors, testing whether
    agents can still make good decisions under perceptual uncertainty.

    Noise is applied to the 6 continuous observations only:
        [0] x position
        [1] y position
        [2] x velocity
        [3] y velocity
        [4] angle
        [5] angular velocity

    The two boolean leg-contact sensors (indices 6 and 7) are left
    unchanged, as noise on binary contact signals is not meaningful.

    Noise model: Gaussian, mean=0, std=noise_std (default 0.1)
    """

    def __init__(self, render_mode=None, noise_std: float = 0.1, seed: int = None):
        """
        Args:
            render_mode (str):   "human" to visualise, None otherwise.
            noise_std   (float): Standard deviation of Gaussian noise
                                 applied to continuous observations.
                                 Default 0.1.
            seed        (int):   Optional seed for the noise RNG,
                                 for reproducibility.
        """
        super().__init__(render_mode=render_mode)
        self.noise_std = noise_std
        self.rng       = np.random.default_rng(seed)

    def _apply_noise(self, obs: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian noise to the continuous observation values.
        Indices 0–5 are corrupted; indices 6–7 (leg contacts) are unchanged.

        Args:
            obs (np.ndarray): Original clean observation from the environment.

        Returns:
            np.ndarray: Noisy observation.
        """
        noisy_obs = obs.copy()
        noise     = self.rng.normal(loc=0.0, scale=self.noise_std, size=6)
        noisy_obs[:6] += noise
        return noisy_obs

    def reset(self):
        """Reset the environment and return a noisy initial observation."""
        obs, info = self.env.reset()
        return self._apply_noise(obs), info

    def step(self, action):
        """
        Step the environment and return a noisy observation.

        Returns:
            noisy_obs  - corrupted observation
            reward     - unmodified reward (physics unchanged)
            terminated - episode end flag
            truncated  - episode truncation flag
            info       - diagnostic info dict
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._apply_noise(obs), reward, terminated, truncated, info