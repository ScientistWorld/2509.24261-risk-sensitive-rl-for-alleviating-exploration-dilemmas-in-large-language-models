#!/bin/bash
# Evaluate the visible train slice using the same artifact contract as evaluate.sh.

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

mkdir -p scoring

INPUT="${1:-checkpoints/bandit_results.json}"

python eval/train/evaluate_bandit_slice.py --input "$INPUT" --output scoring/scores_train.json
