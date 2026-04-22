from stable_baselines3.common.callbacks import BaseCallback


class EpisodeRewardCallback(BaseCallback):
    """
    Records total reward and episode length at the end of each episode
    during training. Used to plot learning curves (reward vs episodes).
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self._current_episode_reward = float(0.0)
        self._current_episode_length = 0

    def _on_step(self) -> bool:
        """Called after every environment step during training."""
        # Accumulate reward and steps for the current episode
        self._current_episode_reward += float(self.locals["rewards"][0])
        self._current_episode_length += 1

        # Check if the episode has ended (terminated or truncated)
        terminated = self.locals["dones"][0]
        if terminated:
            self.episode_rewards.append(self._current_episode_reward)
            self.episode_lengths.append(self._current_episode_length)
            self._current_episode_reward = float(0.0)
            self._current_episode_length = 0

        return True  # Returning False would stop training early