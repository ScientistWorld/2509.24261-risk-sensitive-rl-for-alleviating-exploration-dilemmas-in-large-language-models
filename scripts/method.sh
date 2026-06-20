#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

STEPS="${RS_GRPO_STEPS:-120000}"

python method/rs_grpo_bandit.py \
  --betas 0 2 4 8 \
  --steps "$STEPS" \
  --batch-size 16 \
  --learning-rate 0.1 \
  --seed 42 \
  --output checkpoints/bandit_results.json

bash scripts/evaluate.sh checkpoints/bandit_results.json
