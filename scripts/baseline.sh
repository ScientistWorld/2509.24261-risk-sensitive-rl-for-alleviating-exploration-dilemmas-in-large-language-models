#!/bin/bash
# Run the standalone risk-neutral GRPO baseline on the visible slice.
set -euo pipefail
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
python baseline/grpo_bandit.py --list
python eval/train/bandit_eval.py --methods grpo_baseline \
  --output logs/baseline_scores_train.json --details logs/baseline_details_train.json
