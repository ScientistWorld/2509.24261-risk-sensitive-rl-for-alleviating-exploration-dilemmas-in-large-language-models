# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-20 09:28] - core_claim
- Scheduler job reproduced the Section 5 100-arm bandit mechanism: risk-neutral policy gradient stayed near the suboptimal local peak while sufficiently risk-seeking RS-GRPO moved to the global peak.
- `scripts/evaluate.sh` generated `scoring/scores.json` from saved artifacts and `python validate.py --compare` passed.

### [2026-06-20 10:05] - core_claim
- Audited the implementation against the paper and public RS-GRPO repository.
- Fixed diagnostic history saving and aligned beta comparisons to use the same rollout seed protocol; submitting a fresh ten-seed CPU verification job.

## Stop Justification
- Not stopping in this turn; a fresh verification job is being submitted after fidelity fixes.
