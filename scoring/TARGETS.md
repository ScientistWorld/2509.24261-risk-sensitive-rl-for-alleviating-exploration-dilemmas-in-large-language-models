# Evaluation Targets

## Primary: bandit_escape

Tests whether an optimizer can move a sharply biased initial policy from a
suboptimal local reward peak to the global reward peak.
- **Metric**: `expected_reward` (higher is better)
- **Run**: `bash scripts/evaluate.sh checkpoints/bandit_results.json`

## Constraint: bandit_escape

Tests whether the final policy actually leaves the initially favored local
mode, rather than improving only by spreading probability diffusely.
- **Metric**: `local_action_probability` (lower is better)
- **Run**: `bash scripts/evaluate.sh checkpoints/bandit_results.json`

## Robustness: bandit_escape

Tests whether the escape behavior is stable across independent stochastic
rollout seeds, rather than being caused by one favorable sample path.
- **Metric**: `global_peak_success_rate` (higher is better)
- **Run**: `bash scripts/evaluate.sh checkpoints/bandit_results.json`
