#!/bin/bash
set -euo pipefail

cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"

bash scripts/evaluate_train.sh checkpoints/bandit_results.json
bash scripts/evaluate_test.sh checkpoints/bandit_results.json
