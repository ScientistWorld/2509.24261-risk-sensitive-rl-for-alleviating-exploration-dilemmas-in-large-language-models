# Evaluation

## Setting

A K-armed bandit with deterministic rewards in [0, 1] and a softmax policy
over the arms. The initial policy is fixed and sharply peaked on a suboptimal
"local" peak (reward 0.6); a "global" peak (reward 1.0) carries little initial
probability. Each episode is 7,500 groups of N = 16 actions (120,000
interactions).

The evaluator owns everything outside the method:

1. It samples each group of 16 actions from the current policy.
2. It sends the method the 16 rewards of the group (not which arms were
   played) and receives 16 advantages.
3. It checks the advantages (finite, |A| <= 8), centres them within the
   group, and applies the fixed softmax policy-gradient step with learning
   rate 0.1 to the sampled arms.
4. After the last group it scores the final policy against the hidden
   instance.

A method is therefore an *advantage rule*, which may keep any state it likes
across groups. The interface is documented in `data/agent_protocol.py`; the
reference agents in `method/` and `baseline/` are examples.

## Experiments

- `bandit_escape` (weight 0.4): the paper-scale 100-arm problem, with the arm
  labels shuffled in every episode (24 episodes per slice).
- `bandit_escape_scaled` (weight 0.6): the same two reward peaks inside
  2^18 arms. The global peak's initial probability varies from 0.1% to 15%
  across episodes, it is hidden among similarly likely low-reward arms, and
  the remaining mass is spread over a long tail (48 episodes per slice).

## Metrics (per experiment, averaged over episodes)

- `expected_reward` (higher is better, primary): reward of the final policy.
- `mean_reward_last_1000` (higher): mean reward of the last 1,000 sampled
  actions.
- `online_mean_reward` (higher): mean reward of all 120,000 sampled actions,
  which measures how fast the method gets there.
- `optimal_action_probability` (higher): final probability of the best arm.
- `global_peak_probability` (higher): final probability of arms with reward
  >= 0.9.
- `global_peak_success_rate` (higher): fraction of episodes whose final policy
  puts at least half its mass on those arms.
- `local_action_probability` (lower): final probability of the local peak's
  centre arm.
- `local_peak_probability` (lower): final probability left in the local peak
  neighbourhood.

Every number is computed by the evaluator from its own sampling record and
the final policy; nothing a method prints is read. A method that crashes,
violates the protocol, or returns a non-finite or over-cap advantage in any
episode is scored as failed: worst value (0 for higher-is-better metrics, 1
for lower-is-better) on every metric of both experiments.

## Running

- `bash scripts/evaluate_train.sh` scores every agent file on disk on the
  visible slice and writes `scoring/scores_train.json`.
- `bash scripts/evaluate.sh` does the same and writes `scoring/scores.json`.
  It accepts development flags such as `--methods name --max-episodes 2`.
- The hidden slice (`scripts/evaluate_test.sh`) uses different instances and
  sampling seeds from the same distribution and the same interface.

Every `*.py` file directly in `method/` or `baseline/` whose name does not
start with `_` is an agent file and is scored. Helpers belong in subpackages
or underscore-prefixed files.

A run of all reference agents takes about 40 minutes per slice on one CPU
core.
