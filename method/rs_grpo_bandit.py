"""Risk-Sensitive GRPO (RS-GRPO) agent for the bandit exploration benchmark.

Implements the paper's empirical risk-sensitive advantage (Eq. 8):

    A_i = (1 / beta) * ( exp(beta * r_i) / mean_j exp(beta * r_j) - 1 ),

computed within each group of N = 16 sampled actions, with beta = 0 giving the
standard mean-centred GRPO advantage r_i - mean_j r_j. The evaluator owns the
sampling and applies the paper's softmax policy-gradient step with learning
rate 0.1 (Section 5 / Appendix E). Eq. 8 advantages already sum to zero over
the group, so the evaluator's centring leaves them unchanged.

Variants: beta in {0, 2, 4, 8, 16} (the paper's ablation grid plus the
upstream bandit script's beta = 16).

Run: ``python method/rs_grpo_bandit.py --list`` / ``--serve rs_grpo_beta8``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.agent_protocol import Agent, main  # noqa: E402

BETAS = {
    "grpo_beta0": 0.0,
    "rs_grpo_beta2": 2.0,
    "rs_grpo_beta4": 4.0,
    "rs_grpo_beta8": 8.0,
    "rs_grpo_beta16": 16.0,
}


def rs_grpo_advantage(rewards: list[float], beta: float) -> list[float]:
    """Empirical risk-sensitive advantage from Eq. 8 (beta = 0: GRPO)."""
    n = len(rewards)
    if beta == 0:
        mean_reward = sum(rewards) / n
        return [r - mean_reward for r in rewards]
    scaled = [beta * r for r in rewards]
    top = max(scaled)
    soft_value = (math.log(sum(math.exp(x - top) for x in scaled) / n) + top) / beta
    return [(math.exp(beta * (r - soft_value)) - 1.0) / beta for r in rewards]


class RSGRPOAgent(Agent):
    def __init__(self, beta: float) -> None:
        self.beta = beta

    def advantages(self, actions: list[int], rewards: list[float]) -> list[float]:
        return rs_grpo_advantage(rewards, self.beta)


def _factory(beta: float):
    return lambda: RSGRPOAgent(beta)


if __name__ == "__main__":
    main({name: _factory(beta) for name, beta in BETAS.items()})
