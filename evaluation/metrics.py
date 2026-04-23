import numpy as np


def smooth_rewards(rewards: list, window: int = 50) -> list:
    """
    Applies a rolling mean to a reward series to smooth the learning curve.
    Useful for visualising the trend without noisy episode-to-episode variance.

    Args:
        rewards (list): Raw episode rewards from training.
        window  (int):  Number of episodes to average over.

    Returns:
        smoothed (list): Smoothed reward series (same length as input).
    """
    smoothed = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(rewards[start:i + 1]))
    return smoothed


def compute_convergence_episode(rewards: list, threshold: float = 200.0, window: int = 50) -> int:
    """
    Estimates the episode at which the agent converges by finding the first
    point where the smoothed reward stays above a threshold to the end of training.

    Args:
        rewards   (list):  Raw episode rewards from training.
        threshold (float): Reward value considered 'solved' (default 200).
        window    (int):   Smoothing window size.

    Returns:
        episode (int): Episode number of convergence, or -1 if never converged.
    """
    smoothed = smooth_rewards(rewards, window)
    for i, r in enumerate(smoothed):
        if all(v >= threshold for v in smoothed[i:]):
            return i
    return -1


def compute_summary_stats(rewards: list) -> dict:
    """
    Computes summary statistics for a list of episode rewards.

    Args:
        rewards (list): List of episode rewards.

    Returns:
        dict: mean, std, min, max, and median reward.
    """
    return {
        "mean"   : float(np.mean(rewards)),
        "std"    : float(np.std(rewards)),
        "min"    : float(np.min(rewards)),
        "max"    : float(np.max(rewards)),
        "median" : float(np.median(rewards)),
    }