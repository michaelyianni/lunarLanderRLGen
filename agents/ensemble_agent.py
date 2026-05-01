import collections

from environments.base_env import LunarLanderEnv
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.a2c_agent import A2CAgent


class EnsembleAgent:
    """
    Ensemble Agent that combines DQN, PPO, and A2C via majority-vote action selection.

    The ensemble is not trained directly; it aggregates the predictions of three
    pre-trained baseline agents and returns the action chosen by the majority.
    In the case of a three-way tie (all agents vote differently), PPO's action
    is used as the tiebreaker, as PPO is the most robust generaliser based on
    experimental results.
    """

    def __init__(self, agents: list):
        """
        Args:
            agents (list): List of loaded agent instances.
                           Expected order: [DQNAgent, PPOAgent, A2CAgent].
                           PPO must be at index 1 for the tiebreaker to work correctly.
        """
        self.agents = agents

    def predict(self, obs):
        """
        Predict an action by majority vote across all constituent agents.

        Each agent casts one vote for its preferred action. The action with the
        most votes is returned. If all three agents vote for different actions
        (a three-way tie), PPO's vote is used as the tiebreaker.

        Args:
            obs: Observation from the environment.

        Returns:
            action: The majority-vote action.
        """
        votes = [agent.predict(obs) for agent in self.agents]
        counter = collections.Counter(int(v) for v in votes)

        most_common = counter.most_common()
        top_count = most_common[0][1]

        # Check for a tie among top candidates
        tied_actions = [action for action, count in most_common if count == top_count]

        if len(tied_actions) == 1:
            # Clear majority
            return most_common[0][0]

        # Tiebreaker: use PPO's vote (index 1 in self.agents)
        ppo_vote = int(votes[1])
        if ppo_vote in tied_actions:
            return ppo_vote

        # Fallback (should not occur with three agents): return the first tied action
        return tied_actions[0]

    @classmethod
    def load(cls, paths: dict, envs: dict):
        """
        Convenience class method that loads all three baseline agents and returns
        a ready-to-use EnsembleAgent.

        Args:
            paths (dict): Model file paths keyed by algorithm name.
                          Expected keys: "DQN", "PPO", "A2C".
                          Example: {"DQN": "results/models/dqn_standard", ...}
            envs  (dict): Environment instances keyed by algorithm name.
                          Expected keys: "DQN", "PPO", "A2C".

        Returns:
            EnsembleAgent: A fully loaded ensemble ready for prediction.
        """
        dqn_agent = DQNAgent(env=envs["DQN"])
        dqn_agent.load(paths["DQN"])

        ppo_agent = PPOAgent(env=envs["PPO"])
        ppo_agent.load(paths["PPO"])

        a2c_agent = A2CAgent(env=envs["A2C"])
        a2c_agent.load(paths["A2C"])

        # Order matters: PPO must be at index 1 for the tiebreaker
        return cls(agents=[dqn_agent, ppo_agent, a2c_agent])
