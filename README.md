# LunarLander RL Generalisation Study

- RS1: How do different reinforcement learning algorithms generalise to modified environments in the LunarLander task?
- RS2: How does transfer learning from a standard training environment affect the re-convergence speed and final performance of different reinforcement learning algorithms when deployed in modified LunarLander environments?

## Algorithms
- Deep Q-Network (DQN)
- Proximal Policy Optimisation (PPO)
- Advantage Actor-Critic (A2C)
- Ensemble (combining the three above via majority vote)

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

## Documentation

- **[Final Report: Generalisation and Adaptation in Reinforcement Learning](./Generalisation%20and%20Adaptation%20in%20Reinforcement%20Learning.pdf)** - Comprehensive analysis of RL generalization findings
