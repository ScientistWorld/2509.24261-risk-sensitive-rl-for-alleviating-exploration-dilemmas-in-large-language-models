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

## In This Gym

`method/rs_grpo_bandit.py` implements Eq. 8 as an agent (see
`data/agent_protocol.py`) with beta in {0, 2, 4, 8, 16}: the paper's ablation
grid plus the released bandit script's 16. The evaluator samples groups of
N = 16 from the softmax policy and applies the paper's policy-gradient step
(learning rate 0.1) with the returned advantages. Eq. 8 advantages sum to
zero within a group, so the evaluator's centring leaves them unchanged.
`rs_grpo_beta8` (the paper's RS-GRPO setting in its pass@k figure) is the
reference every new method is ranked against. `baseline/grpo_bandit.py` is
the standalone risk-neutral GRPO baseline.
