# Experiment Overview

Both experiments ask the same question: starting from a policy that strongly
prefers a good-but-suboptimal mode, does an update rule move the policy to
the best actions, and how fast? A method only decides how much weight each
sampled action gets, given the rewards of its group (shown with a hidden
per-group offset, so only differences within a group are informative).
Sampling, the policy update and all scoring are done by the evaluator on
instances the method has not seen.

## bandit_escape

The paper-scale version of the problem: 100 actions, one strongly favoured
local optimum (reward 0.6) and a global optimum (reward 1.0) that the initial
policy rarely samples. It establishes whether an update rule escapes the
initial bias at all, and how quickly. Rules that only reinforce what the
initial policy already likes stay at the local optimum.

## bandit_escape_scaled

The same question in a much larger action space, with a reward scale that is
not known in advance. Every episode has its own background reward, its own
gap between the background and the best actions, and its own height of the
suboptimal mode. The best actions are rare under the initial policy and are
surrounded by many equally likely poor actions. Trying actions one by one
cannot cover the space within the budget. It establishes whether an update
rule escapes the initial bias reliably when it cannot know beforehand which
reward level is good, and how quickly and precisely it then converges.
