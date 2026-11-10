#!/bin/bash
# Standard evaluation: score every agent on disk on the visible slice and
# write scoring/scores.json (same numbers as scoring/scores_train.json).
#
# Agents are files method/*.py and baseline/*.py (names not starting with
# "_") that speak the protocol in data/agent_protocol.py. The evaluator owns
# the bandit instances, samples every action, applies the fixed policy update
# and computes every metric itself; nothing written by a method is trusted.
#
# Optional development flags are passed through, e.g.
#   bash scripts/evaluate.sh --methods my_method --max-episodes 2
set -euo pipefail
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
mkdir -p scoring
python eval/train/bandit_eval.py --output scoring/scores.json "$@"
