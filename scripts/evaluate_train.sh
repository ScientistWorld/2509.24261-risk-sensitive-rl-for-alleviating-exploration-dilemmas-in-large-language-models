#!/bin/bash
# Score every agent on disk (method/*.py and baseline/*.py whose names do not
# start with "_") on the visible slice and write scoring/scores_train.json.
# Takes no arguments. The evaluator runs each agent through the protocol in
# data/agent_protocol.py; it never trusts a number written by a method.
set -euo pipefail
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
mkdir -p scoring
python eval/train/bandit_eval.py --output scoring/scores_train.json
