# Problem Definition

## Research Question

How can reinforcement learning improve generative policies on tasks where the
best outcomes are initially low probability, without simply concentrating
probability mass around the policy's existing favorite outputs?

## Why It Matters

For reasoning models, improving only average one-sample accuracy can reduce
solution diversity and hurt best-of-k performance. A good optimizer should
discover high-reward alternatives that the initial policy rarely samples.

## Setting Used Here

A many-armed bandit stands in for a generative policy. The initial softmax
policy concentrates on a good-but-suboptimal mode; the best actions are rare
under it. The policy is updated by a fixed policy-gradient step from groups
of sampled actions, and the question is how each sample in a group should be
weighted (its advantage) given only the group's rewards. Rewards are shown
with a hidden per-group offset, and in the large-scale experiment each
instance has its own unknown reward scale and its own unknown height of the
suboptimal mode. A rule cannot rely on knowing in advance what reward counts
as good.

## Success Criteria

A successful system moves probability toward globally high-reward actions
from a biased, sharply peaked initial policy, including when those actions
are unlikely at the start and the reward scale is unknown. It should:

- improve best-achievable and expected reward;
- converge precisely on the best actions;
- avoid collapsing onto the suboptimal mode;
- do all of this within the fixed interaction budget.
