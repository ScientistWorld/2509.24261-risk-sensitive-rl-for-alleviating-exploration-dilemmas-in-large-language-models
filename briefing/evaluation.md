# Evaluation

## Metrics

- `expected_reward`: final policy probability-weighted reward over all actions.
  Higher is better.
- `mean_reward_last_1000`: average sampled reward over the final 1000
  interactions. Higher is better.
- `optimal_action_probability`: final probability assigned to the single best
  action. Higher is better.
- `global_peak_probability`: final probability assigned to arms in the global
  reward peak region. Higher is better.
- `global_peak_success_rate`: fraction of independent runs that reach the
  global peak with high expected reward. Higher is better.
- `local_action_probability`: final probability assigned to the suboptimal
  initially favored action. Lower is better.
- `local_peak_probability`: final probability retained in the initially favored
  local neighborhood. Lower is better.

## Baselines and Targets

The provided reference baseline is expected to remain near the suboptimal peak
in the 100-arm benchmark, with final reward around 0.6. Successful exploratory
optimization should reach the global peak, with final reward near 1.0 and high
probability on the optimal action.

## Evaluation Protocol

Run the fixed 100-arm benchmark with the same initial policy, batch size 16, and
learning rate 0.1. `scripts/evaluate.sh` reads saved artifacts from
`checkpoints/` and writes `scoring/scores.json`; it does not rerun training.
