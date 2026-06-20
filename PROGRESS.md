# Progress

## What Works

Implemented the paper's risk-sensitive policy-gradient advantage and the
Section 5 100-arm bandit reproduction. The method writes artifacts under
`checkpoints/`, and `scripts/evaluate.sh` scores those artifacts without
importing from `method/`.

## Results

The current `scoring/scores.json` reproduces the qualitative Figure 3 pattern:
risk-neutral training remained near the suboptimal reward peak around 0.6,
while beta 4 and beta 8 moved to the high-reward global peak around 0.97.
`python validate.py --compare` passes and reports that the artifact-based
evaluation preserves the expected separation between trapped and escaping
policies.

## Remaining

The core Section 5 bandit claim is reproduced. The next useful extension is to
run additional random seeds or implement a scaled LLM RLVR experiment if enough
compute and a suitable open model/dataset are available. The large LLM math
pass@k results remain outside this CPU reproduction.

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

## Deviations from Paper

The initial reproduction targets the paper's Section 5 bandit experiment rather
than the full LLM RLVR training. The algorithm is not replaced: the implemented
update is the paper's risk-sensitive advantage from Eq. 8 with beta=0 as the
GRPO baseline.
