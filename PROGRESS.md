# Progress

## What Works

Implemented the paper's risk-sensitive policy-gradient advantage and the
Section 5 100-arm bandit reproduction. The method writes artifacts under
`checkpoints/`, and `scripts/evaluate.sh` scores those artifacts without
importing from `method/`. This reaches `core_claim_plus` because the reproduced
experiment directly tests the paper's central mechanism and now verifies the
effect across ten seeds per beta: risk-seeking advantages help a sharply biased
policy escape a local reward optimum.

Continuation audit on 2026-06-20 confirmed that the implementation is faithful
for the Section 5 bandit but does not cover the paper's large LLM RLVR
pass@k/pass@1 experiments. To strengthen the supported claim without replacing
the algorithm, the method and evaluator now support multi-seed aggregation for
the same paper bandit protocol.

## Results

The current `scoring/scores.json` reproduces the qualitative Figure 3 pattern:
risk-neutral training remained near the suboptimal reward peak around 0.6,
while beta 4 and beta 8 moved to the high-reward global peak around 0.97.
`python validate.py --compare` passes and reports that the artifact-based
evaluation preserves the expected separation between trapped and escaping
policies.

The packaged CPU job `a8ca55da-260` completed successfully and produced:

- `grpo_beta0`: expected reward 0.598, global-peak probability 0.000
- `rs_grpo_beta2`: expected reward 0.596, global-peak probability 0.001
- `rs_grpo_beta4`: expected reward 0.976, global-peak probability 0.989
- `rs_grpo_beta8`: expected reward 0.972, global-peak probability 0.984

The multi-seed job `da5a6748-fb3` completed the 40 bandit runs but failed
validation because `scores.json` included diagnostic standard deviations and
seed counts that were not declared benchmark metrics. The evaluator now writes
only reference metrics. Retry job `2273f1d7-39c` reran the same multi-seed
script in the container and passed `python validate.py --compare`, giving:

- `grpo_beta0`: expected reward 0.598, global-peak success rate 0.0
- `rs_grpo_beta2`: expected reward 0.596, global-peak success rate 0.0
- `rs_grpo_beta4`: expected reward 0.988, global-peak success rate 1.0
- `rs_grpo_beta8`: expected reward 0.979, global-peak success rate 1.0

## Remaining

The core Section 5 bandit claim is reproduced and scheduler-verified, including
the multi-seed robustness extension. The large LLM math pass@k results remain
outside this CPU reproduction.

## Issues

The released repository only contains the lightweight bandit and advantage
snippet, not a full VeRL training configuration for the math experiments. The
bandit experiment is therefore the first reproducible core mechanism.

Job `4532f67a-4ac` failed before executing the experiment because
`python method/rs_grpo_bandit.py` used `/home/user/method` as the import root
inside the container, so `from data.bandit_spec` failed. The standard shell
entrypoints now export `PYTHONPATH=/home/user:${PYTHONPATH:-}` before invoking
Python. A tiny `tmp/` smoke run verified that the method and evaluator imports
work through this path.

The retry job `a8ca55da-260` confirmed the fix.

On 2026-06-20, a fidelity audit against the public RS-GRPO repository found
that the released project provides `codes/bandit.py` and a VeRL advantage
snippet, but no full math-benchmark training stack. The local implementation
was adjusted to use the same rollout seed for every beta setting, matching the
released bandit comparison protocol, and `--save-history` now correctly writes
diagnostic trajectories. Job `2273f1d7-39c` verified the patched default
ten-seed run inside the container, and a continuation audit restored the
milestone to `core_claim_plus`.

## Deviations from Paper

The initial reproduction targets the paper's Section 5 bandit experiment rather
than the full LLM RLVR training. The algorithm is not replaced: the implemented
update is the paper's risk-sensitive advantage from Eq. 8 with beta=0 as the
GRPO baseline.
