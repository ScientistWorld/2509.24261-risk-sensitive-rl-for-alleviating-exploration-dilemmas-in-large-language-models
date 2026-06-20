# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-06-20 09:34] - core_claim
- Implemented the paper's risk-sensitive advantage estimator from Eq. 8 in a 100-armed bandit policy-gradient loop.
- Reproduced the Section 5 / Figure 3 escape result: beta 0 and beta 2 remain near the suboptimal local peak, while beta 4 and beta 8 move to the global reward peak.
- Verified artifact-based evaluation with `python validate.py --compare`; `eval/` and `data/` do not import from `method/`.

### [2026-06-20 09:49] - core_claim_plus
- Scheduler job `2273f1d7-39c` reran the Section 5 bandit reproduction across ten seeds per beta.
- Multi-seed scoring shows beta 4 and beta 8 reach the global peak in every seed, while beta 0 and beta 2 remain trapped near the local optimum.
- `python validate.py --compare` passes in the container with the artifact-based evaluator and the expanded robustness metric.

## Stop Justification
- Completed at milestone `core_claim_plus`.
