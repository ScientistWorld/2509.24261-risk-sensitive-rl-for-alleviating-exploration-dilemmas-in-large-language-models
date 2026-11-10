#!/bin/bash
# Job entry point. Reproduces the reference numbers end to end, then (this
# iteration only) reruns the hardening audit through the hidden test.
set -uo pipefail
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
mkdir -p logs
echo "== reproduce (visible + hidden slice) =="
bash scripts/reproduce.sh 2>&1 | tee logs/reproduce.log
echo "reproduce exit: ${PIPESTATUS[0]}"
python validate.py --compare 2>&1 | tail -40 || true
echo "== hardening audit =="
bash redteam/hardening/run_audit.sh
echo "audit exit: $?"
