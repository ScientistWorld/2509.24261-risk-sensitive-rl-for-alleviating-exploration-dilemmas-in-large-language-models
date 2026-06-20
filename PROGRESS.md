# Progress

## What Works

Implemented the paper's risk-sensitive policy-gradient advantage and the
Section 5 100-arm bandit reproduction. The method writes artifacts under
`checkpoints/`, and `scripts/evaluate.sh` scores those artifacts without
importing from `method/`.

## Results

A local 120,000-step check reproduced the qualitative Figure 3 pattern:
risk-neutral training remained near the suboptimal reward peak around 0.6,
while beta 4 and beta 8 moved to the high-reward global peak around 0.97.
The submitted job will regenerate the official `scoring/scores.json`.

## Remaining

Run the full benchmark job, inspect `scoring/scores.json`, and update the
milestone if the risk-sensitive runs reproduce the reported escape behavior.
The large LLM math pass@k results remain outside this initial CPU reproduction.

## Issues

The released repository only contains the lightweight bandit and advantage
snippet, not a full VeRL training configuration for the math experiments. The
bandit experiment is therefore the first reproducible core mechanism.

## Deviations from Paper

The initial reproduction targets the paper's Section 5 bandit experiment rather
than the full LLM RLVR training. The algorithm is not replaced: the implemented
update is the paper's risk-sensitive advantage from Eq. 8 with beta=0 as the
GRPO baseline.
