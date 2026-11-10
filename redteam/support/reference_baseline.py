"""Standalone risk-neutral policy-gradient baseline for the bandit benchmark."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import random
from pathlib import Path

from data.bandit_spec import bandit_metadata, bimodal_bandit_rewards, local_optimum_initial_logits


def stable_softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exp_logits = [math.exp(x - max_logit) for x in logits]
    total = sum(exp_logits)
    return [x / total for x in exp_logits]


def sample_categorical(policy: list[float], rng: random.Random) -> int:
    cumulative = []
    running = 0.0
    for prob in policy:
        running += prob
        cumulative.append(running)
    return min(bisect.bisect_left(cumulative, rng.random()), len(policy) - 1)


def policy_gradient_update(
    logits: list[float],
    actions: list[int],
    rewards: list[float],
    learning_rate: float,
) -> list[float]:
    policy = stable_softmax(logits)
    mean_reward = sum(rewards) / len(rewards)
    advantages = [reward - mean_reward for reward in rewards]
    grad = [0.0 for _ in logits]

    for action, advantage in zip(actions, advantages):
        for i, prob in enumerate(policy):
            grad[i] -= advantage * prob
        grad[action] += advantage

    scale = learning_rate / len(actions)
    return [logit + scale * g for logit, g in zip(logits, grad)]


def run_baseline(steps: int, batch_size: int, learning_rate: float, seed: int) -> dict:
    rng = random.Random(seed)
    rewards_by_arm = bimodal_bandit_rewards()
    logits = local_optimum_initial_logits(len(rewards_by_arm))
    meta = bandit_metadata()
    reward_history: list[float] = []
    buffer_actions: list[int] = []
    buffer_rewards: list[float] = []

    for _ in range(steps):
        policy = stable_softmax(logits)
        action = sample_categorical(policy, rng)
        reward = rewards_by_arm[action]
        buffer_actions.append(action)
        buffer_rewards.append(reward)
        reward_history.append(reward)

        if len(buffer_actions) == batch_size:
            logits = policy_gradient_update(logits, buffer_actions, buffer_rewards, learning_rate)
            buffer_actions = []
            buffer_rewards = []

    if buffer_actions:
        logits = policy_gradient_update(logits, buffer_actions, buffer_rewards, learning_rate)

    final_policy = stable_softmax(logits)
    return {
        "config": {
            "beta": 0.0,
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
        },
        "metadata": meta,
        "reward_by_arm": rewards_by_arm,
        "final_policy": final_policy,
        "best_arm": max(range(len(final_policy)), key=lambda i: final_policy[i]),
        "expected_reward": sum(p * r for p, r in zip(final_policy, rewards_by_arm)),
        "optimal_action_probability": final_policy[meta["global_arm"]],
        "local_action_probability": final_policy[meta["local_arm"]],
        "mean_reward_last_1000": sum(reward_history[-1000:]) / min(1000, len(reward_history)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/bandit_baseline.json"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run_baseline(args.steps, args.batch_size, args.learning_rate, args.seed)
    args.output.write_text(json.dumps({"bandit_escape": {"beta_0": result}}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
