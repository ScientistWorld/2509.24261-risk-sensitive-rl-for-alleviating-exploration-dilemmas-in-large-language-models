# Hardening report (iteration 11)

## Summary

The previous gym scored a saved artifact. That artifact contained the method's
own final policy and reward history, computed on a public 100-arm instance.
`REDTEAM.md` showed that visible answers, enumeration, forged reward tails,
edited ground truth, duplicated seeds and unscored methods all paid.

I replaced the artifact pipeline instead of patching it.

**The hidden test now runs the method itself.** It does this on bandit
instances that exist only inside `eval/test/bandit_eval.py`, generated from
secret seed material:

- The evaluator samples every group of 16 actions from the current policy.
- The method sees only the group's 16 rewards, never which arms were played,
  as in RLVR where a sampled response essentially never recurs.
- The method returns 16 advantages. The evaluator checks them (finite,
  |A| <= 8), centres them, and applies the paper's softmax policy-gradient
  step (lr 0.1, 7,500 groups = 120,000 interactions).
- The evaluator computes every metric from its own sampling record and the
  final policy.

A method is therefore exactly what the paper contributes: an advantage rule
(Eq. 8 is "a drop-in replacement for the advantage calculation").

**Why the cap never binds the paper's method.** Eq. 8 advantages always lie
in (-1/beta, min((e^beta - 1)/beta, (N - 1)/beta)], so they never exceed
about 5.4 for any beta. GRPO advantages are at most 1.

**Two scored experiments:**

- `bandit_escape` (weight 0.4): the paper's exact Section 5 bandit (100 arms,
  same rewards, same initial logits), with the arm labels permuted in each of
  24 hidden episodes.
- `bandit_escape_scaled` (weight 0.6): the same two reward peaks inside
  2^18 arms, over 48 hidden episodes. The initial policy keeps 45-65% of its
  mass on the local peak. It puts a log-uniform 0.1-15% on the 21-arm global
  peak, hidden among similarly likely low-reward decoys, and spreads the rest
  over a long tail. This is the regime where the paper says its method can
  stay at a local optimum (Sec. 6.2). It is the experiment with headroom.

Train and test share no code. The test imports nothing from `data/`,
`method/` or `eval/train/`. `scripts/evaluate_test.sh` takes no arguments and
scores every top-level, non-underscore `*.py` file in `method/` and
`baseline/`.

**Reference method.** `rs_grpo_beta8`, the paper's RS-GRPO setting for its
pass@k curves (Fig. 4) and the largest beta of its ablation grid. The previous
reference was `rs_grpo_beta4`. It is recorded as `_reference_method` in
`scoring/reference.json`. Beta 2, 4 and 16 are scored as the paper's ablation.

**How the numbers here were produced.** Two CPU jobs ran everything,
`0182d678-1d5` and `1f9a741c-44e`; each used 1 core.

- **Job 2 (final protocol).** It reran the reference set on both slices, then
  ran the audit through the unchanged `scripts/evaluate_test.sh` with no
  arguments, every agent installed as an ordinary method file.
- **Normalized scores.** These follow the ranking rule. For each experiment
  and metric, the improvement over `rs_grpo_beta8` is taken in the metric's
  better direction, divided by the metric's standard deviation across the
  audited cohort, and multiplied by |coefficient| × experiment weight. The
  cohort is the 9 valid rows unless stated. Code: `redteam/hardening/analyze.py`;
  output: `redteam/hardening/out/analysis.json`.

## Do better results score higher?

**Before (old gym, from REDTEAM.md).** The stronger variant beat the
reference by +0.345 normalized units against a seed spread of 0.247. That
variant was a tabular critic with natural policy gradient and early entropy.
The weaker variant (risk sensitivity removed) scored -8.94. But the test was
near saturation: the stronger variant reached about 1.0 on every metric.

**After (hidden test).**

| Method | E1 expected reward | E1 online | E2 expected reward | E2 success | E2 optimal-arm prob. | Normalized |
|---|---:|---:|---:|---:|---:|---:|
| Honest stronger: aspiration advantage | 0.998 | 0.891 | 0.752 | 0.479 | 0.292 | **+2.67** |
| Reference `rs_grpo_beta8` | 0.980 | 0.775 | 0.699 | 0.292 | 0.082 | 0 |
| `rs_grpo_beta16` | 0.963 | 0.657 | 0.691 | 0.292 | 0.100 | -0.42 |
| `rs_grpo_beta4` | 0.987 | 0.745 | 0.673 | 0.188 | 0.103 | -0.68 |
| `rs_grpo_beta2` | 0.596 | 0.547 | 0.629 | 0.083 | 0.041 | -5.92 |
| Weaker: GRPO (`grpo_baseline` = `grpo_beta0`) | 0.598 | 0.554 | 0.597 | 0.000 | 0.000 | **-6.81** |

The honest stronger variant is `redteam/hardening/agents/honest_stronger.py`.
It applies the same exponential utility as Eq. 8 (beta 8), but measures each
sample against the known best attainable reward 1.0 instead of the group's
soft value. It uses only the group's rewards.

Its gain is larger than the test's noise. Paired per-episode differences on
the same hidden instances and sampling seeds (mean ± standard error):

- **E1:** expected reward +0.018 ± 0.002; online reward +0.116 ± 0.003;
  optimal-arm probability +0.42 ± 0.10.
- **E2:** expected reward +0.052 ± 0.025; success +0.19 ± 0.06; optimal-arm
  probability +0.21 ± 0.06. Online reward is -0.012 ± 0.020, within noise.

**Test noise.**

- **Bootstrap over episodes.** Resampling the reference's 72 hidden episodes
  gives a normalized SD of 0.80, with 5–95% bounds of -1.29 to +1.32.
- **Visible-slice replicate.** The reference's visible-slice result (different
  instances and seeds), placed on the same cohort scale, is -1.26.

**Comparisons against that noise:**

- **Paper's effect:** reference vs GRPO is 6.8, far larger than the noise.
- **Stronger variant:** +2.67, also larger.
- **Paper's own beta variants:** beta 4 and 16 are within noise of beta 8, as
  Lemma 4 predicts for too-large beta.

**Train/test agreement.** The method ordering is the same on both slices. The
visible slice draws slightly harder E2 instances, so every honest method is
about 0.04–0.06 lower there. Reference E2 expected reward is 0.656 visible vs
0.699 hidden; the stronger variant 0.707 vs 0.752.

## Shortcuts: fix, and score before and after

"Before" scores are from `REDTEAM.md` (old gym). In that report Δ means the
normalized difference from the old reference within its 22-row cohort.
"After" scores are from the final audit run. The evidence lives in
`redteam/hardening/out/`:

- `audit_scores_test.json`: test scores of every audited agent.
- `audit_checks.json`: the identity and forged-file checks.
- `analysis.json`: the normalized scores and paired comparisons.

| Shortcut | Fix | Before | After |
|---|---|---|---|
| Visible answers (argmax of the public landscape) | Hidden instances: arm permutation and scaled-instance layout come from secret seed material in `eval/test/`. The method never sees arm identities or initial logits. | Works, Δ +0.245 | Not expressible. The v1 answer-lookup agent scored 1.000 on train but tied the reference on test (E1 0.980, E2 0.776); it cannot run under the final protocol. |
| Brute-force enumeration | Evaluator samples all actions and applies the update; advantages capped. 2^18-arm scaled experiment. Final protocol hides arm identities. | Works (100 queries), Δ +0.245 | With arm identities visible (job 1), a count-based "push down seen arms, then commit" agent reached the bound on both experiments: expected reward 1.000 and success 1.000 on all 48 scaled episodes. That is what led to the reward-only protocol. In the final protocol it cannot be expressed. Its reward-only analogue, success-only, is reported below as an honest variant. |
| Forged reward tail, forged summaries, impossible values | Every metric is computed by the evaluator. Metric values outside [0, 1] fail the run. | Works, Δ +1.739 | `shortcut_forged_output` (reference advantages plus forged fields in every reply and on stderr) is identical to the reference: normalized 0, paired difference exactly 0. |
| Ground-truth edit or relabel in `data/` | `eval/test/` holds its own private generator; nothing is read from `data/`. | Works, Δ +4.688 / +0.245 | After rescaling and relabelling `data/bandit_instances.py` and `data/bandit_spec.py` before the test, all 12 reference rows were bit-identical to the clean run. |
| Duplicated seeds, cherry-picked runs | Evaluator chooses every instance and sampling seed. Each episode runs in a fresh process. | Works, Δ +0.206 | Not expressible. |
| Stale checkpoint, train-filtered method list, disabled baseline | Test runs every agent file on disk; reads no checkpoints or train scores; reruns `baseline/grpo_bandit.py`. | Tie (pays) | A non-protocol file left in `method/` is scored as failed, worst everywhere (normalized -8.30 in the all-rows cohort), not skipped. |
| Non-finite output | NaN or inf advantage fails the run. | Accepted (exit 0) | `shortcut_nan_advantage`: failed, worst on all 16 metric entries. |
| Empty, null or missing outputs (test exited, nothing written) | Any agent failure gives the worst value on every metric of every experiment; the score file is always written. | No score, missing values cost nothing | `shortcut_crash`: failed, worst everywhere (-8.30). |
| Instant commit by huge steps (new) | Advantage cap \|A\| <= 8. | n/a | `shortcut_oversized_advantage` (reference × 100): failed, worst everywhere. |
| Forged or stale score file left where the test writes | Evaluator deletes the output at start and writes it atomically at the end. | Conditional | A forged `forged_method` entry written before the run did not survive. |

## Shortcuts left, and why

1. **A simple success-threshold rule outranks the paper's method.**
   `variant_success_only` gives +8 to samples with reward >= 0.9 and -1 to
   everything else. It never reinforces the local peak, so the initial
   policy keeps sampling until the optimum turns up.
   - Hidden test: normalized +6.20. E2 expected reward 0.890 and success
     0.896, against 0.699 / 0.292 for the reference. E1 expected reward is
     0.966, below the reference's 0.980, and optimal-arm probability 0.167
     vs 0.492.
   - It is not at the bound: it fails about 10% of hidden scaled episodes,
     and there its expected reward drops to about 0.27.
   - I widened E2's difficulty range from 1–15% to 0.1–15% so that no simple
     rule reaches the bound, but I did not suppress it. It is an honest,
     genuinely better advantage rule for a single-instance deterministic
     bandit. Any rule that doesn't erode the initial policy before sampling
     the optimum benefits, and it relies on the publicly known reward scale.
   - The consequence: a future agent that copies this rule outranks the
     reference without new research. The fix would be to hide the reward
     scale or target level. That changes the paper's [0, 1] reward setting,
     which its beta values assume, so I left it.
2. **Reading `eval/test/` from disk at test time.** An agent process runs as
   the same user on the same filesystem, so I cannot hide the evaluator's
   source or seed material from it. This is out of bounds in
   `scoring/DIRECTION.md` and left to the shortcut grader.
3. **Reward value as arm identity.** Each arm of the two peaks has a distinct
   deterministic reward, so a method can recognise "the same arm" by its
   reward and keep reinforcing it. This stays inside advantage design and
   cannot enumerate the background, which all shares one reward value. Left.
4. **Knowledge of the reward scale** (maximum 1.0, success threshold 0.9) is
   public and used by both variants above. Left, as domain knowledge.

## Reference score before and after

- **Before:** old test, `rs_grpo_beta4`. Expected reward 0.990, last-1000
  reward 0.989, optimal-arm probability 0.789, success 1.0.
- **After:** `rs_grpo_beta8` on the hidden test.
  - E1: expected reward 0.980, online reward 0.775, optimal-arm probability
    0.492, success 1.0.
  - E2: expected reward 0.699, online 0.598, optimal-arm probability 0.082,
    success 0.292.
  - `rs_grpo_beta4` under the same test: E1 0.987, E2 0.673.

The paper's claim reproduces in the paper's direction on both experiments
and both slices:

- Beta 0 and 2 stay at the local reward of about 0.60 on E1.
- Beta >= 4 reaches 0.96–0.99 with success 1.0.
- On E2 only risk-seeking runs escape. Success is 0 for GRPO, 0.08 for
  beta 2, 0.19 for beta 4 and 0.29 for beta 8 and 16.

## Headroom check

**E1 (paper scale).** The reference sits at the bound of
`global_peak_success_rate` (1.0). It is near the bound on `expected_reward`,
`mean_reward_last_1000` and `global_peak_probability` (0.98), and on
`local_action_probability` and `local_peak_probability` (0.001 and 0.008).
Headroom remains on:

- `online_mean_reward`: 0.775; the stronger variant reaches 0.891.
- `optimal_action_probability`: 0.49; the stronger variant reaches 0.92.

**E2 (added).** Added for headroom, at weight 0.6, from where the paper's
method falls short: scale, and an optimum that is very unlikely under the
initial policy. No metric is at its bound for the reference, or for any
method tested; the best is 0.896 success. The reference and both baselines
(GRPO, beta 2) were run on it. GRPO is clearly below the reference (0.597 vs
0.699) on a scale that leaves room above it.

**Brute force.** Enumerating arms cannot be expressed under the protocol, and
2^18 arms far exceed the 120,000-interaction budget.

## Weights

E2 weight 0.6, E1 weight 0.4 (previously one experiment, weight 1.0).

- E1 is the paper's own experiment and the reproduction anchor, but the
  reference is nearly saturated there.
- E2 measures the paper's central claim (escaping when good actions are
  unlikely at the start) where methods actually differ.

Metric coefficients are the previous magnitudes, all made positive, since
only magnitudes count. `online_mean_reward` (0.5) is added for convergence
speed (Lemma 4, Fig. 3 curves).

## Departures from the paper, and things not fixed

- **Bandit only.** Only the Section 5 bandit mechanism is covered; the LLM
  pass@k experiments are out of scope (CPU-only budget).
- **Budget.** 120,000 interactions per episode; the released script uses
  400,000.
- **Evaluator owns the update.** The update is applied as
  theta[a] += lr/N · (A - mean A) on the sampled arms. This equals the
  paper's softmax policy gradient whenever the advantages sum to zero, which
  Eq. 8 and GRPO always do. Sampling uses Python's `random` instead of NumPy.
- **Success definition.** Success is now "at least half the final mass on
  the global peak". The previous rule also required expected reward >= 0.9.
- **Metrics now computed by the evaluator.** `mean_reward_last_1000` is
  computed from the evaluator's own sampling record, and `online_mean_reward`
  is new.
- **E2 results in `reference.json`** are the reference reproduction's
  visible-slice numbers, labelled as such. The paper reports nothing at this
  scale; its Section 5 numbers are kept under `_paper_reported_section5`. E1
  keeps the paper's reported numbers unchanged.
- **Slice difficulty.** E2 difficulty is drawn independently per slice and
  not stratified, so the visible slice happens to be a little harder (see
  train/test agreement above). The ordering of methods is unaffected.
- **Hidden-test runtime.** About 40 minutes for the six reference agents on
  one CPU core (about 1 s per E1 episode and 3–7 s per E2 episode per method).
  It scores what the method produces per episode and does no pretraining,
  since methods need none.
- **Archived evidence that no longer runs.** The earlier red-team harness
  (`redteam/run_case.py`, `redteam/support/`) targets the old artifact
  pipeline. The job-1 agents in `redteam/hardening/agents_v1/` use the old
  protocol, which exposed arm identities. Both are kept as evidence only.

## Reproduce

- `bash scripts/run.sh`: reference set on both slices, then `validate.py`.
- `bash redteam/hardening/run_audit.sh`: the audit. It restores every file
  it touches.
- `python3 redteam/hardening/analyze.py`: the analysis.
