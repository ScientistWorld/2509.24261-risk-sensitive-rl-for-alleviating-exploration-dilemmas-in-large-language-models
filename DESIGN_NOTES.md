# Design Notes

## Checked

- Read `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side. The train script calls `eval/train/` and writes `scoring/scores_train.json`; the test script calls `eval/test/` and writes `scoring/scores_test.json`.
- Read both slice evaluators. They score saved method artifacts from `checkpoints/bandit_results.json` and do not import from `method/` or rerun training.
- Grepped for train/test cross-imports and paper-agreement metric names; none were found in the split evaluators or `scoring/reference.json`.
- Confirmed `scoring/scores_train.json` and `scoring/scores_test.json` do not exist.
- Ran `bash -n` on both slice scripts successfully.

## Changed

- Added `optimal_action_probability` to both train and test slices. This separately checks convergence to the single globally optimal arm instead of only broad high-reward-region mass.
- Added `local_peak_probability` to both train and test slices. This closes a shortcut where a method lowers probability on the exact initialized local arm while keeping mass in the same suboptimal local mode.
- Required at least four saved runs per method before either slice scores. The previous one-run fallback made train and test score the same run, eliminating the held-out signal.
- Added required `higher_is_better`, `coefficient`, and `description` metadata for all train/test metrics in `scoring/reference.json`.

## Runtime Concerns For Runner Iteration

- The default method script writes ten seeds per beta, so the new four-run minimum should pass for the reproduction path. A future method that emits fewer than four runs per method will fail evaluation by design; this is intentional to avoid a degenerate split.
- `local_peak_probability` uses the documented local arm from the artifact metadata and a fixed +/-5 arm neighborhood. This matches the current 100-arm benchmark shape; if a future method changes the benchmark geometry, the evaluator should reject that artifact rather than silently rescore a different problem.
- I could not use the prompt's paper path because it was absent on this node; the local `paper/2509.24261.md` and existing workspace target descriptions were used for metric alignment.
