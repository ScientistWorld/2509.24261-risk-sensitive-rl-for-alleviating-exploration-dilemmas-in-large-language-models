# Experiment Overview

Both experiments ask the same question: starting from a policy that strongly
prefers a good-but-suboptimal mode, does an update rule move the policy to
the best actions, and how fast? A method only decides how much weight each
sampled action gets, given the rewards of its group. Sampling, the policy
update and all scoring are done by the evaluator on instances the method has
not seen.

## bandit_escape

The paper-scale version of the problem: 100 actions, one strongly favoured
local optimum (reward 0.6) and a global optimum (reward 1.0) that the initial
policy rarely samples. It establishes whether an update rule escapes the
initial bias at all, and how quickly. Rules that only reinforce what the
initial policy already likes stay at the local optimum.

## bandit_escape_scaled

The same question in a much larger action space. The global optimum is rarer
under the initial policy (it varies across episodes, down to very unlikely),
it is surrounded by many equally likely but poor actions, and trying actions
one by one cannot cover the space within the budget. It establishes how far
an update rule's ability to escape extends as the good actions become less
likely at the start, the regime where the gap between methods is largest.
Many episodes are hard for every method tested so far.
