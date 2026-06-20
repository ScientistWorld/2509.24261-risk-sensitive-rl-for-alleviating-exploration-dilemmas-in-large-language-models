"""Problem specifications for the RS-GRPO reproduction.

This module is method-agnostic: it defines the reward landscape and initial
policy used by the benchmark, but it does not implement any optimizer.
"""

from __future__ import annotations

import math


def bimodal_bandit_rewards(n_arms: int = 100) -> list[float]:
    """Return the 100-arm bimodal reward landscape from the paper's repo."""
    rewards = []
    for arm in range(n_arms):
        global_peak = 0.9 * math.exp(-((arm - 40) ** 2) / (2 * 5**2))
        local_peak = 0.5 * math.exp(-((arm - 75) ** 2) / (2 * 5**2))
        rewards.append(0.1 + global_peak + local_peak)
    return rewards


def local_optimum_initial_logits(n_arms: int = 100) -> list[float]:
    """Initial softmax logits sharply biased toward the suboptimal peak."""
    probs = [math.exp(-0.5 * ((arm - 75) / 5.0) ** 2) + 0.1 for arm in range(n_arms)]
    total = sum(probs)
    probs = [p / total for p in probs]
    logits = [math.log(p + 1e-8) for p in probs]
    mean_logit = sum(logits) / len(logits)
    return [logit - mean_logit for logit in logits]


def bandit_metadata() -> dict[str, int]:
    rewards = bimodal_bandit_rewards()
    return {
        "n_arms": len(rewards),
        "global_arm": max(range(len(rewards)), key=lambda i: rewards[i]),
        "local_arm": 75,
    }
