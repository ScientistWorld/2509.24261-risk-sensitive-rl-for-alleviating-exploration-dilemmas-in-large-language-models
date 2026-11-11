#!/bin/bash
# Job entry point: reproduce the reference numbers end to end (CPU, 1 core,
# ~80 min) and validate the workspace. The hardening audit is
# redteam/hardening/run_audit.sh (run it after this script when needed).
set -euo pipefail
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
mkdir -p logs
bash scripts/reproduce.sh 2>&1 | tee logs/reproduce.log
python validate.py --compare
