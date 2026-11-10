#!/bin/bash
# Score every agent on disk (method/*.py and baseline/*.py whose names do not
# start with "_") on the hidden slice and write scoring/scores_test.json.
# Takes no arguments. Same agent interface as the visible slice.
set -euo pipefail
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
mkdir -p scoring
python eval/test/bandit_eval.py --output scoring/scores_test.json
