# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-06-20 09:28] - core_claim
- Scheduler job reproduced the Section 5 100-arm bandit mechanism: risk-neutral policy gradient stayed near the suboptimal local peak while sufficiently risk-seeking RS-GRPO moved to the global peak.
- `scripts/evaluate.sh` generated `scoring/scores.json` from saved artifacts and `python validate.py --compare` passed.

### [2026-06-20 10:05] - core_claim
- Audited the implementation against the paper and public RS-GRPO repository.
- Fixed diagnostic history saving and aligned beta comparisons to use the same rollout seed protocol; submitting a fresh ten-seed CPU verification job.

### [2026-06-20 10:15] - core_claim_plus
- Audited the continued workspace and confirmed the implementation uses the paper's risk-sensitive advantage from Eq. 8 on the Section 5 100-arm bandit, not a surrogate.
- Verified that `eval/` reads saved artifacts without importing `method/`, `scripts/evaluate.sh` works standalone, and `python validate.py --compare` passes.
- Restored `core_claim_plus` because the verified run covers the central bandit mechanism across beta values and ten rollout seeds.

## Stop Justification
- Completed at milestone `core_claim_plus`.
