# Research Direction

## Research Problem

This benchmark studies policy-gradient updates when the initial policy is
sharply biased toward a suboptimal region and the best outcomes are rare
under it. The goal is to improve the update's ability to discover and then
concentrate on high-reward, initially unlikely actions, quickly and
reliably.

## Core Contribution

Work should focus on how sampled outcomes are weighted in the policy
gradient: the advantage given to each sample of a group, how it depends on
the group's rewards, normalisation, risk attitude, schedules across training,
and memory of rewards seen in earlier groups.

## Approach Scope

Any rule that maps the rewards of a group (and the method's own history of
rewards) to advantages is in scope. That includes risk-sensitive objectives,
pass@k-style or best-of-k weighting, adaptive or scheduled parameters, and
learned baselines over reward histories.

## Out of Bounds

- Reading, importing or reverse-engineering anything under `eval/`, or
  reconstructing hidden instances from seed material.
- Changing the instances, the budget, the learning rate, the update rule or
  the advantage cap; editing the evaluators.
- Writing score files or printing results in place of the evaluator.
- Tuning a method to the visible slice's specific instances instead of the
  instance distribution.
