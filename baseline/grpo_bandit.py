"""Standalone risk-neutral GRPO baseline agent for the bandit benchmark.

Advantage of each sampled action: its reward minus the mean reward of its
group of N = 16 samples (the paper's Eq. 3 advantage). The evaluator samples
the actions and applies the softmax policy-gradient step.

Run: ``python baseline/grpo_bandit.py --list`` / ``--serve grpo_baseline``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.agent_protocol import Agent, main  # noqa: E402


class GRPOAgent(Agent):
    def advantages(self, rewards: list[float]) -> list[float]:
        mean_reward = sum(rewards) / len(rewards)
        return [r - mean_reward for r in rewards]


if __name__ == "__main__":
    main({"grpo_baseline": GRPOAgent})
