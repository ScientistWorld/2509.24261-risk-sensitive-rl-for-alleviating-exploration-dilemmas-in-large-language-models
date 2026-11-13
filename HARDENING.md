# Hardening report (final iteration)

This iteration fixes the one weakness `REVIEW.md` listed after the previous
hardening:

> A five-line success-threshold advantage rule that uses the public reward
> scale scores +6.2 over the reference, more than twice the best honest
> variant (+2.67).

It then reruns the reference, the baselines and the full shortcut audit from
the final code.

## What changed

**1. Hidden per-group reward offsets (both experiments).** Each group's 16
rewards reach the method shifted by one hidden offset, drawn U(-2, 2) per
group. Rewards are therefore comparable only within a group.

- This removes absolute thresholds such as "reward >= 0.9 is a success".
- It also removes "reward value = arm identity" across groups (previously left
  shortcut 3).
- The paper's advantage (Eq. 8) and GRPO are exactly invariant to such a
  shift. On the paper-scale experiment every reference number equals the
  previous gym's to within 1e-16 (floating-point rounding).

**2. Hidden per-instance reward scale and success level (scaled experiment,
`bandit_escape_scaled`).** Each instance draws three things:

- a background reward;
- a span from background to the best arm, log-uniform in [0.3, 1.0];
- a local-peak height, uniform in [30%, 85%] of the span.

The success level is therefore not knowable in advance (fixes previously left
shortcut 4). Metrics are reported in each instance's normalised units:
background 0, best arm 1.

This is also a regime the paper names as its own limitation: a fixed β whose
effective strength depends on the reward scale. Appendix A calls choosing β
adaptively an open problem.

**3. Recalibrated optimum rarity on the scaled experiment.** The global peak's
initial mass is now log-uniform in [3%, 30%], previously [0.1%, 15%].

- When the optimum is extremely rare, any rule that refuses to reinforce
  anything until it sees a large jump beats the memoryless reference. That
  includes a fixed threshold tuned to the public distribution of scales.
- A calibration sweep showed this. It covered 8 candidate distributions × 23
  advantage rules, on paired episodes (`redteam/hardening/calibration/`).
  - In every candidate distribution whose optimum could be rarer than 1%,
    the best fixed-threshold rule stayed +1.3 to +4.6 above the reference,
    even with hidden scales.
  - Hidden offsets alone, with the old fixed scale, still left it at +4.4.
  - Only with the optimum at ≥ 3% did the threshold rule fall below the
    reference, while honest scale-free methods kept a large margin.

**4. Faster, equally exact sampler.** The softmax sampler is now a segment
tree, giving O(log n) sampling and updates. Each parent sum is recomputed from
its children, so no rounding error drifts. It samples the same cumulative
order as before. This let the scaled experiment grow to 96 stratified
episodes per slice, up from 48.

**5. Audit and documentation.**

- The audit agents are rewritten for the new protocol.
- The audit runs as per-agent chunks of the unchanged hidden-test evaluator,
  so it can resume after scheduler restarts.
- `briefing/`, the `scoring/*.md` files, `data/bandit_instances.py` (the
  visible generator copy) and `data/agent_protocol.py` now describe the
  offsets and hidden scales.
- The paper's reported numbers in `scoring/reference.json` are unchanged.

Unchanged:

- The paper's method (`method/rs_grpo_bandit.py`) and the GRPO baseline.
- The reference (`rs_grpo_beta8`).
- The update rule, the budget (7,500 groups of 16), the learning rate, the
  advantage cap, the metrics and their coefficients.
- The weights: `bandit_escape` 0.4, `bandit_escape_scaled` 0.6.

## How the numbers were produced

- **Reference runs.** Job `68b4155d-842` ran the reference set on both slices
  with no arguments (`scripts/evaluate_{train,test}.sh`).
- **Audit.** Job `ee3b5bf9-f72` ran every audit agent through the unchanged
  hidden-test evaluator:
  - the visible ground truth in `data/` was edited first;
  - every agent was installed as an ordinary method file;
  - a forged score file was pre-placed where the test writes.
- **Ranking.** "Rank" follows the gym's rule. For each experiment and metric:
  improvement over `rs_grpo_beta8` in the better direction ÷ the metric's SD
  across the audited valid rows × |coefficient| × experiment weight.
- **Noise.** Uncertainties come from a paired bootstrap over episodes: every
  method uses the same resampled hidden episodes.
- **Evidence:**
  - `redteam/hardening/out/` (`analysis.json`, `analysis.log`,
    `audit_scores_*.json`, `compare.log`);
  - the calibration sweep in `redteam/hardening/calibration/`;
  - the previous audit in `redteam/hardening/out_prev/`.

## Do better results score higher?

Yes, on both slices and well beyond the test's noise.

**Hidden test, after.**

| Method | E1 ER | E1 online | E1 opt-arm | E2 ER | E2 success | E2 online | E2 opt-arm | Rank (paired-bootstrap SD) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Honest: memory aspiration | 0.998 | 0.890 | 0.997 | 0.991 | 1.000 | 0.842 | 0.634 | **+5.83** (0.47) |
| Honest: best-seen elitism | 0.986 | 0.963 | 0.210 | 0.987 | 1.000 | 0.959 | 0.354 | **+5.33** (0.51) |
| `rs_grpo_beta16` (paper grid) | 0.963 | 0.657 | 0.753 | 0.837 | 0.708 | 0.585 | 0.228 | +0.91 |
| **Reference `rs_grpo_beta8`** | 0.980 | 0.775 | 0.492 | 0.811 | 0.562 | 0.631 | 0.153 | 0 |
| `rs_grpo_beta4` | 0.987 | 0.745 | 0.616 | 0.780 | 0.469 | 0.616 | 0.114 | -0.88 |
| `rs_grpo_beta2` | 0.596 | 0.547 | 0.000 | 0.746 | 0.365 | 0.595 | 0.082 | -5.82 |
| Weaker: GRPO (`grpo_baseline`) | 0.598 | 0.554 | 0.000 | 0.696 | 0.271 | 0.556 | 0.053 | **-6.92** (0.43) |

E1 is `bandit_escape`, E2 `bandit_escape_scaled`. ER is expected reward;
opt-arm is the optimal-arm probability.

**Before (previous gym, from the previous `HARDENING.md`).** The honest
aspiration variant scored +2.67, beta 16 -0.42, beta 4 -0.68, beta 2 -5.92 and
GRPO -6.81.

The honest variants keep only the group's rewards and a memory of earlier
groups. They need no knowledge of the reward scale, the success level or the
generator (`redteam/hardening/agents/honest_*.py`). Both reach the optimum on
every hidden episode.

They still sit below the bound on two metrics:

- **Precision:** E2 optimal-arm probability is 0.63 at best.
- **Speed:** online reward is 0.96 at best.

Each trades one of these for the other, so a method that is both fast and
precise has room above them.

**Paired per-episode differences vs the reference** (hidden test, mean ± SE):

| Comparison | Experiment | Expected reward | Success |
|---|---|---:|---:|
| Memory aspiration − reference | E2 (n = 96) | +0.180 ± 0.020 | +0.44 ± 0.05 |
| Memory aspiration − reference | E1 (n = 24) | +0.019 ± 0.002 | — |
| Reference − GRPO | E1 | +0.382 ± 0.002 | +1.00 |
| Reference − GRPO | E2 | +0.115 ± 0.021 | +0.29 ± 0.05 |

**Test noise.**

- Bootstrap of the reference's own hidden score: SD 0.48, 5–95% range
  [-0.78, +0.82].
- The reference's visible-slice result on the same scale: +0.58.

**Comparisons against that noise:**

- **Paper's effect:** reference vs GRPO is 6.9, about 16 SD.
- **Honest improvements:** +5.3 and +5.8, each more than 10 paired SD.

**Train/test agreement.** The visible slice gives the same ordering:

| Method | Visible-slice rank |
|---|---:|
| Memory aspiration | +5.04 |
| Best-seen elitism | +4.63 |
| β16 | +0.12 |
| Reference | 0 |
| Ported threshold (t0.3) | -1.08 |
| β4 | -1.25 |
| Success-only (verbatim) | -5.72 |
| β2 | -5.76 |
| GRPO | -6.85 |

Reference E2 numbers, visible vs hidden: expected reward 0.833 vs 0.811,
success 0.60 vs 0.56.

## Shortcuts: fix, and score before and after

"Before" scores are the previous gym's hidden test, in the same normalised
rank units. "After" scores are this audit's hidden test.

| Shortcut | Fix | Before | After |
|---|---|---|---|
| **Success-threshold rule using the public scale** (`variant_success_only`, REVIEW.md) | Per-group offsets make absolute thresholds meaningless; per-instance hidden span and local height make no fixed threshold separate the two peaks. | **+6.20** | Verbatim: **-5.92** (E2 expected reward 0.709, success 0.29; E1 stuck at 0.595). |
| **Same rule ported to the offsets**, with the best fixed threshold for the public scale distribution | Hidden span 0.3–1 and local height 30–85% of the span, with the optimum at ≥ 3% so the reference escapes often. | n/a (would have tied the old rule) | Best threshold t0.3: **-0.94** (paired-bootstrap SD 0.68, 5–95% [-2.09, +0.19]). The other thresholds score -1.09 to -3.54. |
| Previous honest variant, which assumed the known best reward 1.0 (`old_aspiration_known_scale`) | Same as above: the scale is no longer known. | +2.67 (counted as honest then) | -6.77 |
| Reward value as arm identity across groups | Per-group offsets. | Left (no score) | Not expressible: values are comparable only within a group. |
| Forged self-reported results in every reply and on stderr | Evaluator computes every metric. | 0 (tie) | 0, bit-identical to the reference. |
| Visible ground-truth edit in `data/` | `eval/test/` has its own generator. | No effect | No effect: all 12 reference rows bit-identical to the clean run (`compare.log`). |
| Forged score file left where the test writes | Evaluator deletes its output first. | Did not survive | Did not survive. |
| Crash / NaN / over-cap advantage / non-protocol file in `method/` | Run is failed, worst value on every metric of both experiments. | Worst (-8.30) | Worst (-10.09 in the all-rows cohort). |

**Visible answers, enumeration, duplicated seeds, stale checkpoints** stay
closed by the previous iteration's design:

- hidden instances generated only in `eval/test/`;
- no arm identities shown to the method;
- 2^18 arms, which the 120,000-interaction budget cannot enumerate;
- every seed chosen by the evaluator.

## Shortcuts and weaknesses left, and why

1. **Short memory-based rules score far above the reference.**
   `honest_best_seen` is about five lines. It remembers the best
   within-group-anchored reward and gives +8 near it and -1 elsewhere, and it
   scores +5.3.

   I classify it as an honest method, not a shortcut:
   - it uses no public constant, no scale and no generator knowledge;
   - it is a generic elitist rule;
   - "memory of rewards seen in earlier groups" is explicitly in scope in
     `scoring/DIRECTION.md`.

   Its gain is the paper's own weakness in this setting: a memoryless,
   group-relative advantage keeps reinforcing the local mode in groups where
   the optimum is absent.

   It reaches the success bound (1.0) on both experiments and expected
   reward 0.99. It does not reach the bound on optimal-arm probability (0.21
   on E1, 0.35 on E2), and the dense memory-aspiration variant outranks it.
   Still, a reviewer who counts any short rule above the paper's method as a
   flaw will see this one. I could not find a task setting in which
   group-memory rules do not beat a memoryless advantage while the paper's
   claim still holds.

2. **The ported threshold rule is below the reference, but not far.** Its
   upper 95% bootstrap bound is +0.19. Tuned to the public scale
   distribution, it ranks -0.94 on average, about 1.4 SD below.
   - Lowering the optimum's minimum mass further raises it above the
     reference (calibration: +1.2 at ≥ 1%, +2.6 at ≥ 0.03%).
   - Raising it further brings the honest methods to the bound.

   3% is the compromise.
3. **`rs_grpo_beta16` ranks +0.91 above the reference on the hidden test**,
   but only +0.12 on the visible slice. This is the paper's own β grid; a
   larger β partly compensates for smaller hidden reward spans. I kept β = 8,
   the paper's setting for its main pass@k results, as the reference.
4. **Reading `eval/test/` from disk at test time** is still physically
   possible: same user, same filesystem. It is out of bounds in
   `scoring/DIRECTION.md` and left to the shortcut grader.

## Reference score before and after

**Before:** `rs_grpo_beta8` on the previous hidden test.

- E1: expected reward 0.980, success 1.0.
- E2: expected reward 0.699, success 0.29, at the old fixed scale with the
  optimum at 0.1–15%.

**After:**

- **E1: unchanged** (to within 1e-16). Expected reward 0.980, online 0.775,
  optimal-arm 0.492, success 1.0. The offsets do not change Eq. 8, and the
  sampler draws the same arms.
- **E2.** The experiment is redefined, so its numbers are not comparable with
  before:
  - expected reward 0.811 (normalised units), online 0.631, optimal-arm
    0.153;
  - global-peak probability 0.553, success 0.562;
  - local-peak probability 0.419.

**The paper's claim reproduces in the paper's direction on both slices.**

- On E1, beta 0 and 2 stay at the local reward of about 0.60; beta ≥ 4
  reaches 0.96–0.99, with success 1.0.
- On E2, success rises with beta:

  | Method | E2 success |
  |---|---:|
  | GRPO | 0.27 |
  | β2 | 0.36 |
  | β4 | 0.47 |
  | β8 | 0.56 |
  | β16 | 0.71 |

## Headroom check

**Where the reference sits at or near the bound:**

- **E1:** at the bound of `global_peak_success_rate` (1.0). Near the bound on
  expected reward, last-1000 reward and global-peak probability (0.98), and on
  both local probabilities (0.001 and 0.008).
- **E2:** no metric is at the bound. Expected reward is 0.81, success 0.56
  and optimal-arm probability 0.15.

**Room above the reference, shown by the honest variants:**

- E2 success: 0.56 → 1.0.
- E2 optimal-arm probability: 0.15 → 0.63.
- E2 online reward: 0.63 → 0.96.
- E1 online reward: 0.78 → 0.96.
- E1 optimal-arm probability: 0.49 → 1.0.

**Room above the honest variants.** They split speed against precision:

- Memory aspiration has online 0.84 and optimal-arm 0.63.
- Best-seen elitism has online 0.96 and optimal-arm 0.35.

So the best combination is not yet reached on E2. That is the trade-off of
the paper's Lemma 4.

**Baselines:**

- GRPO is clearly below the reference on both experiments (-6.9).
- β2 is below the reference on E1 and slightly below it on E2.
- The ported threshold rule is below the reference.

**Brute force:** enumerating arms is not expressible (no arm identities), and
2^18 arms exceed the budget.

## Departures from the paper, and things not fixed

- **Bandit only.** Only the Section 5 bandit mechanism is covered; the LLM
  pass@k experiments are out of scope (CPU-only budget).
- **Budget.** 120,000 interactions per episode; the released script uses
  400,000.
- **E2 is this gym's own experiment,** with hidden scales, offsets, 2^18 arms
  and a recalibrated optimum rarity. It probes the paper's stated limitation,
  a fixed β under an unknown reward scale. The paper reports no numbers for
  it. `scoring/reference.json` keeps the paper's Section 5 numbers under
  `_paper_reported_section5`, and the gym's visible-slice reference
  reproduction under `results` (not a target).
- **Weights and coefficients are unchanged**, and no metric was added or
  removed.
- **Runtime.** On one CPU core the hidden test of the six reference agents
  takes about 25 minutes, and the audit about 70 minutes.

## Reproduce

- `bash scripts/run.sh` (about 2.5 h on one core) runs, in order: the
  reference set on both slices, `validate.py --compare`, the audit (restoring
  every file it touches) and the analysis. It resumes from stage markers in
  `logs/job_state/`.
- `python3 redteam/hardening/calibration/analyze.py` re-summarises the
  calibration sweep. The sweep itself is
  `bash redteam/hardening/calibration/run_calibration.sh`, about 45 min.
