#!/bin/bash
# Evaluate the held-out test slice using the same artifact contract as evaluate.sh.

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

mkdir -p scoring

INPUT="${1:-checkpoints/bandit_results.json}"

python eval/test/evaluate_bandit_slice.py \
  --input "$INPUT" \
  --output scoring/scores_test.json \
  --train-scores scoring/scores_train.json
