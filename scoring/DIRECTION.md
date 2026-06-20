# Research Direction

## Research Problem

This benchmark studies exploration-aware policy-gradient updates when an
initial policy is sharply biased toward a suboptimal region. The goal is to
improve the optimizer's ability to discover high-reward low-probability actions.

## Core Contribution

Work should remain focused on policy-gradient weighting, objectives, advantage
estimation, or closely related exploration mechanisms for the same fixed
stochastic policy setting.

## Approach Scope

Scientists may change the objective, advantage estimator, normalization,
scheduling, or update rule, as long as the method still learns from sampled
actions and rewards under the fixed benchmark.

## Out of Bounds

- Directly setting the policy to the known optimal action.
- Using extra oracle information beyond sampled actions and observed rewards.
- Replacing the benchmark with supervised fitting to the full reward landscape.
- Changing the reward landscape or initial policy to make the task easier.
