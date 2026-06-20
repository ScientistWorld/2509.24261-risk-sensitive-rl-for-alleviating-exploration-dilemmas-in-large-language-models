# Paper's Method

## Key Contribution

The paper introduces Risk-Sensitive GRPO, replacing GRPO's mean-centered
advantage with an advantage derived from an exponential-utility objective.

## Approach

For each prompt or rollout group, the method samples multiple responses and
computes rewards. Instead of using `reward - mean(reward)`, it computes a
soft maximum value
`beta^-1 * log(mean(exp(beta * reward)))` and assigns each sample the
risk-sensitive advantage `(exp(beta * (reward - soft_value)) - 1) / beta`.
The rest of the policy-gradient update is unchanged; `beta=0` recovers standard
GRPO.

## Main Claims

The primary claim is that risk-seeking optimization improves exploration:
policies can escape local optima induced by sharply biased initial
distributions. On reasoning benchmarks, this should improve pass@k while
maintaining or improving pass@1. A supporting claim is shown in a 100-armed
bandit: risk-neutral policy gradient remains trapped near a local peak, while
sufficiently risk-seeking updates move to the global peak.
