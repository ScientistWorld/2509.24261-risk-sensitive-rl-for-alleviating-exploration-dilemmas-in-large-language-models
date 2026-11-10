#!/bin/bash
# Reproduce the gym's reference numbers end to end (CPU only, ~40 min on 1 core).
# Every agent on disk (paper's method variants and the GRPO baseline) is run
# by the evaluators on the visible and, when present, the hidden slice.
set -euo pipefail
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
bash scripts/evaluate_train.sh
# scoring/scores.json is the visible-slice result (what scripts/evaluate.sh writes).
cp scoring/scores_train.json scoring/scores.json
if [ -f eval/test/bandit_eval.py ]; then
  bash scripts/evaluate_test.sh
fi
