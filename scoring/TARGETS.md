# Evaluation Targets

All targets are measured by `bash scripts/evaluate.sh` (visible slice) and
`bash scripts/evaluate_test.sh` (hidden slice). Both run every agent file in
`method/` and `baseline/`.

## Primary: bandit_escape, bandit_escape_scaled

Can the update rule move a sharply biased initial policy from the local
reward peak to the global one?
- **Metric**: `expected_reward` (higher is better).
- **Supporting**: `global_peak_probability`, `optimal_action_probability`,
  `mean_reward_last_1000` (higher is better).

## Constraint: bandit_escape, bandit_escape_scaled

Does the final policy actually leave the initially favoured local mode,
rather than improving only by spreading probability around it?
- **Metric**: `local_action_probability` and `local_peak_probability` (lower
  is better).

## Robustness: bandit_escape, bandit_escape_scaled

Is the escape reliable across independent episodes (shuffled arms and
different sampling seeds, and on the scaled experiment different instances
whose global peak ranges from likely to very unlikely)?
- **Metric**: `global_peak_success_rate` (higher is better).

## Efficiency: bandit_escape, bandit_escape_scaled

How quickly does the rule get there within the fixed 120,000 interactions?
Too much risk-seeking slows convergence.
- **Metric**: `online_mean_reward` (higher is better).
