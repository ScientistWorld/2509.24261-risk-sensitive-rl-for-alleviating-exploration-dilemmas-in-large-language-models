"""Public generators for the bandit exploration benchmark (method-agnostic).

Two instance families are defined here. Scientists may use this module to
create as many development instances as they like. The evaluators in
``eval/train`` and ``eval/test`` carry their own private copies of these
generators, so editing this file never changes how a submission is scored.

``paper_instance``
    The exact 100-arm bimodal reward landscape and sharply biased initial
    policy of the paper's Section 5 bandit, with the arm labels shuffled by a
    random permutation (the problem is unchanged for a tabular policy, but the
    location of the optimum is not known in advance).

``scaled_instance``
    The same two reward peaks and the same kind of biased initial policy,
    embedded in a much larger action space (2**18 arms). The initial policy
    puts most of its mass on the local peak, a small and instance-dependent
    mass on the global peak, similar mass on many low-reward "plausible"
    decoy arms, and a long tail on everything else.

Every instance is a plain dict:

``rewards``         list[float], deterministic reward of every arm
``initial_logits``  list[float], logits of the fixed initial softmax policy
``optimal_arm``     arm with the largest reward
``local_arm``       arm at the centre of the initially favoured local peak
``global_peak``     arms whose reward is at least ``GLOBAL_PEAK_THRESHOLD``
``local_peak``      arms of the local peak neighbourhood (offset |d| <= 5)
"""

from __future__ import annotations

import math
import random

GLOBAL_PEAK_THRESHOLD = 0.9
PAPER_N_ARMS = 100
SCALED_N_ARMS = 2 ** 18
PEAK_HALF_WIDTH = 10  # each scaled peak spans offsets -10..10 (21 arms)
LOCAL_NEIGHBOURHOOD = 5
P_GLOBAL_RANGE = (0.001, 0.15)  # prior mass on the 21-arm global peak


def _bell(d: float, width: float = 5.0) -> float:
    return math.exp(-(d * d) / (2.0 * width * width))


def paper_rewards() -> list[float]:
    """Reward of the paper's 100 arms, in the paper's own arm order."""
    return [0.1 + 0.9 * _bell(x - 40) + 0.5 * _bell(x - 75) for x in range(PAPER_N_ARMS)]


def paper_initial_logits() -> list[float]:
    """Initial logits of the paper's bandit, in the paper's own arm order."""
    probs = [math.exp(-0.5 * ((x - 75) / 5.0) ** 2) + 0.1 for x in range(PAPER_N_ARMS)]
    total = sum(probs)
    logits = [math.log(p / total + 1e-8) for p in probs]
    mean = sum(logits) / len(logits)
    return [v - mean for v in logits]


def paper_instance(rng: random.Random) -> dict:
    """The paper's Section 5 bandit with arm labels randomly permuted."""
    rewards0 = paper_rewards()
    logits0 = paper_initial_logits()
    order = list(range(PAPER_N_ARMS))
    rng.shuffle(order)  # instance arm i is paper arm order[i]
    position = {paper_arm: i for i, paper_arm in enumerate(order)}
    rewards = [rewards0[order[i]] for i in range(PAPER_N_ARMS)]
    logits = [logits0[order[i]] for i in range(PAPER_N_ARMS)]
    return {
        "family": "paper",
        "rewards": rewards,
        "initial_logits": logits,
        "optimal_arm": position[40],
        "local_arm": position[75],
        "global_peak": sorted(i for i, r in enumerate(rewards) if r >= GLOBAL_PEAK_THRESHOLD),
        "local_peak": sorted(position[x] for x in range(70, 81)),
    }


def scaled_instance(
    rng: random.Random, n_arms: int = SCALED_N_ARMS, p_global: float | None = None
) -> dict:
    """The paper's two reward peaks inside a large action space."""
    offsets = list(range(-PEAK_HALF_WIDTH, PEAK_HALF_WIDTH + 1))
    p_local = rng.uniform(0.45, 0.65)
    drawn = math.exp(rng.uniform(math.log(P_GLOBAL_RANGE[0]), math.log(P_GLOBAL_RANGE[1])))
    p_global = drawn if p_global is None else p_global
    p_decoy = rng.uniform(0.15, 0.25)
    p_tail = 1.0 - p_local - p_global - p_decoy

    per_arm = p_global / len(offsets)
    n_decoy = max(1, int(round(p_decoy / per_arm)))
    special = rng.sample(range(n_arms), 2 * len(offsets) + n_decoy)
    local_arms = special[: len(offsets)]
    global_arms = special[len(offsets): 2 * len(offsets)]
    decoys = special[2 * len(offsets):]

    rewards = [0.1] * n_arms
    for d, arm in zip(offsets, local_arms):
        rewards[arm] = 0.1 + 0.5 * _bell(d)
    for d, arm in zip(offsets, global_arms):
        rewards[arm] = 0.1 + 0.9 * _bell(d)

    probs = [0.0] * n_arms
    bell_w = [_bell(d) for d in offsets]
    bell_total = sum(bell_w)
    for w, arm in zip(bell_w, local_arms):
        probs[arm] = p_local * w / bell_total
    for group, mass in ((global_arms, p_global), (decoys, p_decoy)):
        weights = [math.exp(0.3 * rng.gauss(0.0, 1.0)) for _ in group]
        total = sum(weights)
        for w, arm in zip(weights, group):
            probs[arm] = mass * w / total
    taken = set(special)
    tail_w = [0.0] * n_arms
    for arm in range(n_arms):
        if arm not in taken:
            tail_w[arm] = math.exp(rng.gauss(0.0, 1.0))
    tail_total = sum(tail_w)
    for arm in range(n_arms):
        if arm not in taken:
            probs[arm] = p_tail * tail_w[arm] / tail_total
    logits = [math.log(p) for p in probs]
    mean = sum(logits) / n_arms
    logits = [v - mean for v in logits]
    centre = offsets.index(0)
    return {
        "family": "scaled",
        "rewards": rewards,
        "initial_logits": logits,
        "optimal_arm": global_arms[centre],
        "local_arm": local_arms[centre],
        "global_peak": sorted(a for a in global_arms if rewards[a] >= GLOBAL_PEAK_THRESHOLD),
        "local_peak": sorted(
            a for d, a in zip(offsets, local_arms) if abs(d) <= LOCAL_NEIGHBOURHOOD
        ),
    }
