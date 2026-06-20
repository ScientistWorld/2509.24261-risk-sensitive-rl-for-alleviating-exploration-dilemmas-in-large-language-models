# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-20 00:00] - none
- Read the paper and identified RS-GRPO's core computation: replacing GRPO's advantage with the risk-sensitive exponential-utility advantage.
- Implemented the paper's 100-arm bandit environment, GRPO baseline, RS-GRPO method, and artifact-based evaluator.

### [2026-06-20 00:20] - method_runs
- Verified that `scripts/method.sh` executes end-to-end on a smoke run and writes `scoring/scores.json`.
- A local 120,000-step check reproduced the expected bandit behavior; submitting the same pipeline as a CPU job for official scoring.

### [2026-06-20 09:27] - core_claim
- Audited the inherited workspace and confirmed the implementation directly uses the paper's RS-GRPO exponential-utility advantage rather than a surrogate.
- `python validate.py --compare` passes, with beta 0 and 2 remaining near the local optimum and beta 4 and 8 concentrating on the global optimum in the Section 5 bandit.
- Submitting a CPU scheduler run to regenerate the artifacts through `scripts/reproduce.sh` and validate the packaged workflow.

### [2026-06-20 09:30] - core_claim
- Job `4532f67a-4ac` failed before running the experiment because Python could not import the workspace `data` package from script-file execution.
- Fixed all standard shell entrypoints to export `/home/user` on `PYTHONPATH` and verified the method/evaluator import path with a tiny `tmp/` smoke run.
- Resubmitting the same CPU reproduction job.

## Stop Justification
