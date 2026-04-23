import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from stable_baselines3 import DQN
from environments.base_env import LunarLanderEnv

TEST_PATH = "results/models/version_test"

print("Creating untrained DQN model...")
env = LunarLanderEnv()
model = DQN(policy="MlpPolicy", env=env.env, verbose=0)

print("Saving...")
model.save(TEST_PATH)

print("Loading...")
loaded = DQN.load(TEST_PATH, env=env.env)

print("\n✅ Save/load SUCCESS — version combination is compatible. Safe to retrain.")
env.close()

# Clean up test file
os.remove(TEST_PATH + ".zip")