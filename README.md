# LunarLander RL Generalisation Study

An investigation into how different reinforcement learning algorithms (DQN, PPO, A2C) generalise to modified environments in the Gymnasium LunarLander task.

## Research Question
How do different reinforcement learning algorithms generalise to modified environments in the LunarLander task?

## Algorithms
- Deep Q-Network (DQN)
- Proximal Policy Optimisation (PPO)
- Advantage Actor-Critic (A2C)

## Environment Modifications
- Increased gravity
- Wind disturbance
- Sensor noise

## Metrics
- Learning efficiency (episodes to convergence)
- Performance (average reward, landing success rate)
- Learning behaviour (reward vs episodes curves)

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Project Structure
- `agents/` - RL agent implementations
- `environments/` - Base and modified environment wrappers
- `training/` - Training scripts per algorithm
- `evaluation/` - Evaluation scripts and metrics
- `results/` - Saved models, logs, and plots
- `notebooks/` - Exploratory analysis notebooks
- `report/` - Final coursework report