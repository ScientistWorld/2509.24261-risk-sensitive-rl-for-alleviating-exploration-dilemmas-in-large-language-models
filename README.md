# Overview

- **Paper ID:** 2509.24261
- **Title:** Risk-Sensitive RL for Alleviating Exploration Dilemmas in Large Language Models
- **Domain:** Reinforcement learning for language-model reasoning
- **TL;DR:** The paper modifies GRPO's advantage calculation with a risk-seeking objective so RL fine-tuning explores low-probability high-reward solutions instead of only sharpening existing modes.

## Short Summary

The paper argues that RLVR for reasoning models often improves pass@1 by
collapsing onto already likely solutions, which can hurt pass@k and diversity.
It proposes RS-GRPO, a drop-in replacement for GRPO's advantage based on an
exponential-utility objective. The authors show theoretically and in a 100-arm
bandit that the risk-sensitive update can escape a local optimum that traps
risk-neutral policy gradient. Large-scale math experiments report improved
pass@32 with comparable pass@1.

## Key Results

- In the paper's bandit experiment, risk-neutral training stays near reward 0.6,
  while sufficiently risk-seeking settings converge to reward 1.0.
- On Qwen2.5-Math-1.5B with deepmath103k, the reported average pass@32 improves
  from 37.5 for GRPO to 42.0 for RS-GRPO, with similar pass@1.
- The recommended comparison setting uses `beta=2` for pass@k baselines and
  `beta=8` in the broader pass@k curves.

---

# Problem Definition

## Research Question

How can reinforcement learning improve generative policies on tasks where the
best outcomes are initially low probability, without simply concentrating
probability mass around the policy's existing favorite outputs?

## Why It Matters

For reasoning models, improving only average one-sample accuracy can reduce
solution diversity and hurt best-of-k performance. A good optimizer should
discover high-reward alternatives that the initial policy rarely samples.

## Success Criteria

A successful system should move probability toward globally high-reward actions
from a biased, sharply peaked initial policy. It should improve best-achievable
or expected reward while avoiding collapse onto a known suboptimal local mode.

---

# Paper's Method

## Key Contribution

The paper introduces Risk-Sensitive GRPO, replacing GRPO's mean-centered
advantage with an advantage derived from an exponential-utility objective.

## Approach

For each prompt or rollout group, the method samples multiple responses and
computes rewards. Instead of using `reward - mean(reward)`, it computes a
soft maximum value
`beta^-1 * log(mean(exp(beta * reward)))` and assigns each sample the
risk-sensitive advantage `(exp(beta * (reward - soft_value)) - 1) / beta`.
The rest of the policy-gradient update is unchanged; `beta=0` recovers standard
GRPO.

## Main Claims

The primary claim is that risk-seeking optimization improves exploration:
policies can escape local optima induced by sharply biased initial
distributions. On reasoning benchmarks, this should improve pass@k while
maintaining or improving pass@1. A supporting claim is shown in a 100-armed
bandit: risk-neutral policy gradient remains trapped near a local peak, while
sufficiently risk-seeking updates move to the global peak.

---

# Evaluation

## Metrics

- `expected_reward`: final policy probability-weighted reward over all actions.
  Higher is better.
- `mean_reward_last_1000`: average sampled reward over the final 1000
  interactions. Higher is better.
- `global_peak_probability`: final probability assigned to arms in the global
  reward peak region. Higher is better.
- `local_action_probability`: final probability assigned to the suboptimal
  initially favored action. Lower is better.

## Baselines and Targets

The risk-neutral reference is expected to remain near the suboptimal peak in the
100-arm benchmark, with final reward around 0.6. Successful exploratory
optimization should reach the global peak, with final reward near 1.0 and high
probability on the optimal action.

## Evaluation Protocol

Run the fixed 100-arm benchmark from the paper's Section 5 with the same initial
policy, batch size 16, and learning rate 0.1. `scripts/evaluate.sh` reads saved
artifacts from `checkpoints/` and writes `scoring/scores.json`; it does not
rerun training.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 16m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

<!-- Write your progress here -->

</details>

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 14m | **GPU**: 0.0h
- **Jobs**: 2 total (1 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

### [2026-06-20 09:33] - core_claim
- Job `a8ca55da-260` completed successfully in the packaged container.
- `scripts/reproduce.sh` regenerated `checkpoints/bandit_results.json` and `scoring/scores.json`, then `python validate.py --compare` passed.
- Reproduced the Section 5 bandit core claim: risk-neutral and low-beta training remain near the local optimum, while beta 4 and beta 8 converge to the global optimum.

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 8m | **GPU**: 0.0h
- **Jobs**: 4 total (2 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-20 09:34] - core_claim
- Implemented the paper's risk-sensitive advantage estimator from Eq. 8 in a 100-armed bandit policy-gradient loop.
- Reproduced the Section 5 / Figure 3 escape result: beta 0 and beta 2 remain near the suboptimal local peak, while beta 4 and beta 8 move to the global reward peak.
- Verified artifact-based evaluation with `python validate.py --compare`; `eval/` and `data/` do not import from `method/`.

### [2026-06-20 09:49] - core_claim_plus
- Scheduler job `2273f1d7-39c` reran the Section 5 bandit reproduction across ten seeds per beta.
- Multi-seed scoring shows beta 4 and beta 8 reach the global peak in every seed, while beta 0 and beta 2 remain trapped near the local optimum.
- `python validate.py --compare` passes in the container with the artifact-based evaluator and the expanded robustness metric.

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 9m | **GPU**: 0.0h
- **Jobs**: 4 total (2 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 4 total (2 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 4 total (2 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 1m | **GPU**: 0.0h
- **Jobs**: 5 total (3 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>


---

# Reproduction Milestones

**Current: core_claim_plus**

## Stop Justification
- Completed at milestone `core_claim`.
