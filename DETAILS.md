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

## This Gym

The gym reproduces the paper's Section 5 bandit (100 arms, hidden arm
permutations) and extends it to a 2^18-arm version. In that version every
instance has its own hidden reward scale and local-peak height, and the
optimum is less likely under the initial policy. This is where a fixed
risk-sensitivity, which the paper names as a limitation, can stay at a local
optimum. Rewards reach the method with a hidden per-group offset. Methods are
advantage rules run by the evaluator on hidden instances. The LLM pass@k
experiments are out of scope (CPU-only budget).

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

## Setting Used Here

A many-armed bandit stands in for a generative policy. The initial softmax
policy concentrates on a good-but-suboptimal mode; the best actions are rare
under it. The policy is updated by a fixed policy-gradient step from groups
of sampled actions, and the question is how each sample in a group should be
weighted (its advantage) given only the group's rewards. Rewards are shown
with a hidden per-group offset, and in the large-scale experiment each
instance has its own unknown reward scale and its own unknown height of the
suboptimal mode. A rule cannot rely on knowing in advance what reward counts
as good.

## Success Criteria

A successful system moves probability toward globally high-reward actions
from a biased, sharply peaked initial policy, including when those actions
are unlikely at the start and the reward scale is unknown. It should:

- improve best-achievable and expected reward;
- converge precisely on the best actions;
- avoid collapsing onto the suboptimal mode;
- do all of this within the fixed interaction budget.

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

## In This Gym

`method/rs_grpo_bandit.py` implements Eq. 8 as an agent (see
`data/agent_protocol.py`) with beta in {0, 2, 4, 8, 16}: the paper's ablation
grid plus the released bandit script's 16. The evaluator samples groups of
N = 16 from the softmax policy and applies the paper's policy-gradient step
(learning rate 0.1) with the returned advantages. Eq. 8 advantages sum to
zero within a group, so the evaluator's centring leaves them unchanged, and
they are invariant to the hidden per-group reward offset. On the scaled
experiment the reward scale varies per instance, so a fixed beta acts as a
weaker or stronger risk preference from one instance to the next (the paper
lists choosing beta adaptively as an open problem, Appendix A).
`rs_grpo_beta8` (the paper's RS-GRPO setting in its pass@k figure) is the
reference every new method is ranked against. `baseline/grpo_bandit.py` is
the standalone risk-neutral GRPO baseline.

---

# Evaluation

## Setting

A K-armed bandit with deterministic rewards and a softmax policy over the
arms. The initial policy is fixed and sharply peaked on a suboptimal "local"
peak; a "global" peak (the best arms) carries little initial probability.
Each episode is 7,500 groups of N = 16 actions (120,000 interactions).

The evaluator owns everything outside the method:

1. It samples each group of 16 actions from the current policy.
2. It sends the method the 16 rewards of the group (not which arms were
   played), all shifted by one hidden offset drawn uniformly from [-2, 2]
   for that group. Rewards can therefore be compared only within a group.
   It receives 16 advantages.
3. It checks the advantages (finite, |A| <= 8), centres them within the
   group, and applies the fixed softmax policy-gradient step with learning
   rate 0.1 to the sampled arms.
4. After the last group it scores the final policy against the hidden
   instance.

A method is therefore an *advantage rule*, which may keep any state it likes
across groups. The interface is documented in `data/agent_protocol.py`; the
reference agents in `method/` and `baseline/` are examples.

## Experiments

- `bandit_escape` (weight 0.4): the paper-scale 100-arm problem (background
  0.1, local peak 0.6, global peak 1.0), with the arm labels shuffled in every
  episode (24 episodes per slice).
- `bandit_escape_scaled` (weight 0.6): two reward peaks inside 2^18 arms.
  Every episode has its own hidden reward scale:
  - a background reward;
  - a span from the background to the best arm (log-uniform in 0.3-1.0);
  - a local-peak height (uniform in 30-85% of the span).

  The global peak's initial probability varies from 3% to 30% across
  episodes (log-uniform). It is hidden among similarly likely
  background-reward arms, and the remaining mass is spread over a long tail.
  The three parameters are stratified over the 96 episodes of a slice.
  Metrics are reported in the episode's normalised units (background 0, best
  arm 1). `data/bandit_instances.py` is a visible copy of both generators.

## Metrics (per experiment, averaged over episodes)

- `expected_reward` (higher is better, primary): reward of the final policy.
- `mean_reward_last_1000` (higher): mean reward of the last 1,000 sampled
  actions.
- `online_mean_reward` (higher): mean reward of all 120,000 sampled actions,
  which measures how fast the method gets there.
- `optimal_action_probability` (higher): final probability of the best arm.
- `global_peak_probability` (higher): final probability of the global-peak
  arms whose reward is at least 0.9 (of the span, on the scaled experiment).
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
- The hidden slice (`scripts/evaluate_test.sh`) uses different instances,
  offsets and sampling seeds from the same distribution and the same
  interface.

Every `*.py` file directly in `method/` or `baseline/` whose name does not
start with `_` is an agent file and is scored. Helpers belong in subpackages
or underscore-prefixed files.

A run of the six reference agents takes roughly 10-20 minutes per slice on
one CPU core.

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

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 2m | **GPU**: 0.0h
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

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
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

### [2026-06-20 10:18] - core_claim_plus
- Final verification kept the milestone at `core_claim_plus`.
- Fixed method-agnostic wording in documentation and made the top-level evaluator preserve arbitrary future method keys.
- Re-ran `scripts/evaluate.sh`, `python validate.py --compare`, and a custom-key evaluator smoke test.

</details>

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 8m | **GPU**: 0.0h
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

### [2026-06-20 10:18] - core_claim_plus
- Final verification kept the milestone at `core_claim_plus`.
- Fixed method-agnostic wording in documentation and made the top-level evaluator preserve arbitrary future method keys.
- Re-ran `scripts/evaluate.sh`, `python validate.py --compare`, and a custom-key evaluator smoke test.

### [2026-06-20 10:22] - core_claim_plus
- Re-audited the paper with a parallel extractor: the reproducible implementation remains scoped to the paper's Section 5 bandit because full LLM RS-GRPO would require VeRL/vLLM, 1.5B+ models, and benchmark-scale sampling beyond the current 3 GPU-hour budget.
- Hardened all bandit evaluators to reject artifacts whose reward landscape, metadata, or final policy shape do not match the fixed benchmark in `data/bandit_spec.py`.
- Local artifact scoring, train/test slice scoring, bytecode compilation, and `python validate.py --compare` pass after the evaluator hardening.

### [2026-06-20 10:26] - core_claim_plus
- Final verification reviewed folder structure, method/problem separation, scoring coefficients, train/test slices, download portability, and evaluator independence.
- Confirmed `python validate.py --compare` passes and the scored metrics reward the benchmark's scientific quantities rather than matching published values.
- Removed generated bytecode caches and kept the milestone at `core_claim_plus`.

</details>

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 8m | **GPU**: 0.0h
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

### [2026-06-20 10:18] - core_claim_plus
- Final verification kept the milestone at `core_claim_plus`.
- Fixed method-agnostic wording in documentation and made the top-level evaluator preserve arbitrary future method keys.
- Re-ran `scripts/evaluate.sh`, `python validate.py --compare`, and a custom-key evaluator smoke test.

### [2026-06-20 10:22] - core_claim_plus
- Re-audited the paper with a parallel extractor: the reproducible implementation remains scoped to the paper's Section 5 bandit because full LLM RS-GRPO would require VeRL/vLLM, 1.5B+ models, and benchmark-scale sampling beyond the current 3 GPU-hour budget.
- Hardened all bandit evaluators to reject artifacts whose reward landscape, metadata, or final policy shape do not match the fixed benchmark in `data/bandit_spec.py`.
- Local artifact scoring, train/test slice scoring, bytecode compilation, and `python validate.py --compare` pass after the evaluator hardening.

### [2026-06-20 10:26] - core_claim_plus
- Final verification reviewed folder structure, method/problem separation, scoring coefficients, train/test slices, download portability, and evaluator independence.
- Confirmed `python validate.py --compare` passes and the scored metrics reward the benchmark's scientific quantities rather than matching published values.
- Removed generated bytecode caches and kept the milestone at `core_claim_plus`.

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim_plus` | **Status**: error
- **Working time**: 0m | **GPU**: 0.0h
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

### [2026-06-20 10:18] - core_claim_plus
- Final verification kept the milestone at `core_claim_plus`.
- Fixed method-agnostic wording in documentation and made the top-level evaluator preserve arbitrary future method keys.
- Re-ran `scripts/evaluate.sh`, `python validate.py --compare`, and a custom-key evaluator smoke test.

### [2026-06-20 10:22] - core_claim_plus
- Re-audited the paper with a parallel extractor: the reproducible implementation remains scoped to the paper's Section 5 bandit because full LLM RS-GRPO would require VeRL/vLLM, 1.5B+ models, and benchmark-scale sampling beyond the current 3 GPU-hour budget.
- Hardened all bandit evaluators to reject artifacts whose reward landscape, metadata, or final policy shape do not match the fixed benchmark in `data/bandit_spec.py`.
- Local artifact scoring, train/test slice scoring, bytecode compilation, and `python validate.py --compare` pass after the evaluator hardening.

### [2026-06-20 10:26] - core_claim_plus
- Final verification reviewed folder structure, method/problem separation, scoring coefficients, train/test slices, download portability, and evaluator independence.
- Confirmed `python validate.py --compare` passes and the scored metrics reward the benchmark's scientific quantities rather than matching published values.
- Removed generated bytecode caches and kept the milestone at `core_claim_plus`.

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 51m | **GPU**: 0.0h
- **Jobs**: 7 total (5 completed, 2 failed)

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

### [2026-06-20 10:18] - core_claim_plus
- Final verification kept the milestone at `core_claim_plus`.
- Fixed method-agnostic wording in documentation and made the top-level evaluator preserve arbitrary future method keys.
- Re-ran `scripts/evaluate.sh`, `python validate.py --compare`, and a custom-key evaluator smoke test.

### [2026-06-20 10:22] - core_claim_plus
- Re-audited the paper with a parallel extractor: the reproducible implementation remains scoped to the paper's Section 5 bandit because full LLM RS-GRPO would require VeRL/vLLM, 1.5B+ models, and benchmark-scale sampling beyond the current 3 GPU-hour budget.
- Hardened all bandit evaluators to reject artifacts whose reward landscape, metadata, or final policy shape do not match the fixed benchmark in `data/bandit_spec.py`.
- Local artifact scoring, train/test slice scoring, bytecode compilation, and `python validate.py --compare` pass after the evaluator hardening.

### [2026-06-20 10:26] - core_claim_plus
- Final verification reviewed folder structure, method/problem separation, scoring coefficients, train/test slices, download portability, and evaluator independence.
- Confirmed `python validate.py --compare` passes and the scored metrics reward the benchmark's scientific quantities rather than matching published values.
- Removed generated bytecode caches and kept the milestone at `core_claim_plus`.


### [2026-11-11] - core_claim_plus
- Hardening iteration: replaced the artifact-scored pipeline with evaluator-run agents on hidden instances. The evaluator samples every action, applies the paper's softmax policy-gradient step, and computes every metric; methods return only advantages from the group's rewards (see HARDENING.md).
- Hidden test reproduces the claim on the paper-scale bandit (GRPO and beta 2 at about 0.60; beta >= 4 at 0.96-0.99, success 1.0) and on a new 2^18-arm scaled experiment where only risk-seeking runs escape (success: GRPO 0, beta 8 0.29).
- Reference is now rs_grpo_beta8. The audit reran the stronger variant (+2.67 normalized) and the shortcuts (forged output ties; NaN, oversized, crash and helper files score worst; ground-truth edits change nothing) through the hidden test.

</details>


---

# Reproduction Milestones

**Current: core_claim_plus**

## Stop Justification
- Completed at milestone `core_claim_plus`.
