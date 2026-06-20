"""Risk-Sensitive GRPO reproduction on the paper's bandit setting."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from data.bandit_spec import (
    bandit_metadata,
    bimodal_bandit_rewards,
    local_optimum_initial_logits,
)


@dataclass
class RunConfig:
    beta: float
    steps: int = 120_000
    batch_size: int = 16
    learning_rate: float = 0.1
    seed: int = 42


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


def rs_grpo_advantage(rewards: list[float], beta: float) -> list[float]:
    """Empirical risk-sensitive advantage from Eq. 8."""
    mean_reward = sum(rewards) / len(rewards)
    if beta == 0:
        return [reward - mean_reward for reward in rewards]

    scaled = [beta * reward for reward in rewards]
    max_scaled = max(scaled)
    soft_value = (math.log(sum(math.exp(x - max_scaled) for x in scaled) / len(scaled)) + max_scaled) / beta
    return [(math.exp(beta * (reward - soft_value)) - 1.0) / beta for reward in rewards]


def policy_gradient_update(
    logits: list[float],
    actions: list[int],
    rewards: list[float],
    beta: float,
    learning_rate: float,
) -> list[float]:
    policy = stable_softmax(logits)
    advantages = rs_grpo_advantage(rewards, beta)
    grad = [0.0 for _ in logits]

    for action, advantage in zip(actions, advantages):
        for i, prob in enumerate(policy):
            grad[i] -= advantage * prob
        grad[action] += advantage

    scale = learning_rate / len(actions)
    return [logit + scale * g for logit, g in zip(logits, grad)]


def run_bandit(config: RunConfig) -> dict:
    rng = random.Random(config.seed + int(config.beta * 1000))
    rewards_by_arm = bimodal_bandit_rewards()
    logits = local_optimum_initial_logits(len(rewards_by_arm))
    meta = bandit_metadata()

    reward_history: list[float] = []
    entropy_history: list[float] = []
    optimal_prob_history: list[float] = []
    local_prob_history: list[float] = []
    buffer_actions: list[int] = []
    buffer_rewards: list[float] = []

    for _ in range(config.steps):
        policy = stable_softmax(logits)
        action = sample_categorical(policy, rng)
        reward = rewards_by_arm[action]

        buffer_actions.append(action)
        buffer_rewards.append(reward)
        reward_history.append(reward)
        entropy_history.append(-sum(p * math.log(p + 1e-12) for p in policy))
        optimal_prob_history.append(policy[meta["global_arm"]])
        local_prob_history.append(policy[meta["local_arm"]])

        if len(buffer_actions) == config.batch_size:
            logits = policy_gradient_update(
                logits, buffer_actions, buffer_rewards, config.beta, config.learning_rate
            )
            buffer_actions = []
            buffer_rewards = []

    if buffer_actions:
        logits = policy_gradient_update(
            logits, buffer_actions, buffer_rewards, config.beta, config.learning_rate
        )

    final_policy = stable_softmax(logits)
    expected_reward = sum(p * r for p, r in zip(final_policy, rewards_by_arm))
    best_arm = max(range(len(final_policy)), key=lambda i: final_policy[i])

    return {
        "config": asdict(config),
        "metadata": meta,
        "reward_by_arm": rewards_by_arm,
        "final_policy": final_policy,
        "best_arm": best_arm,
        "expected_reward": expected_reward,
        "optimal_action_probability": final_policy[meta["global_arm"]],
        "local_action_probability": final_policy[meta["local_arm"]],
        "mean_reward_last_1000": sum(reward_history[-1000:]) / min(1000, len(reward_history)),
        "reward_history": reward_history,
        "entropy_history": entropy_history,
        "optimal_prob_history": optimal_prob_history,
        "local_prob_history": local_prob_history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--betas", nargs="+", type=float, default=[0, 2, 4, 8])
    parser.add_argument("--steps", type=int, default=120_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/bandit_results.json"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    for beta in args.betas:
        results[f"beta_{format(beta, 'g')}"] = run_bandit(
            RunConfig(
                beta=beta,
                steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
            )
        )
    args.output.write_text(json.dumps({"bandit_escape": results}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
