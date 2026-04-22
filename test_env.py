from environments.base_env import LunarLanderEnv


def run_random_agent(n_episodes=3, render=True):
    """
    Runs a random agent in the LunarLander environment.
    Used to verify the environment wrapper is working correctly.

    Args:
        n_episodes (int): Number of episodes to run.
        render (bool): Whether to display the environment visually.
    """
    render_mode = "human" if render else None
    env = LunarLanderEnv(render_mode=render_mode)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")
    print(f"No. observations:  {env.n_observations}")
    print(f"No. actions:       {env.n_actions}")
    print("-" * 40)

    for episode in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1

        print(f"Episode {episode + 1}: Steps = {steps:>4}, Total Reward = {total_reward:>8.2f}")

    env.close()
    print("-" * 40)
    print("Environment test complete.")


if __name__ == "__main__":
    run_random_agent(n_episodes=3, render=True)