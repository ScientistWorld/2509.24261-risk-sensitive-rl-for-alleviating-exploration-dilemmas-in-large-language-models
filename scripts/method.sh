#!/bin/bash
# Run the paper's method (RS-GRPO, beta in {0,2,4,8,16}) on the visible slice.
# The method is an agent: the evaluator samples from the policy and applies
# the update, the agent supplies the paper's Eq. 8 advantages.
set -euo pipefail
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
python method/rs_grpo_bandit.py --list
python eval/train/bandit_eval.py \
  --methods grpo_beta0,rs_grpo_beta2,rs_grpo_beta4,rs_grpo_beta8,rs_grpo_beta16 \
  --output logs/method_scores_train.json --details logs/method_details_train.json
