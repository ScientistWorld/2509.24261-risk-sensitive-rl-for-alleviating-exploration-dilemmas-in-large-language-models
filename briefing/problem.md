# Problem Definition

## Research Question

How can reinforcement learning improve generative policies on tasks where the
best outcomes are initially low probability, without simply concentrating
probability mass around the policy's existing favorite outputs?

## Why It Matters

For reasoning models, improving only average one-sample accuracy can reduce
solution diversity and hurt best-of-k performance. A good optimizer should
discover high-reward alternatives that the initial policy rarely samples.

## Success Criteria

A successful system should move probability toward globally high-reward actions
from a biased, sharply peaked initial policy. It should improve best-achievable
or expected reward while avoiding collapse onto a known suboptimal local mode.
