#!/bin/bash
# Job entry point (CPU, 1 core, ~1.5 h): reproduce the reference numbers end
# to end on both slices, record the scaled experiment's visible-slice numbers
# in scoring/reference.json, validate the workspace, then rerun the hardening
# audit (honest variants and shortcuts through the unchanged hidden test,
# restoring every file it touches) and its analysis.
set -euo pipefail
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
mkdir -p logs
bash scripts/reproduce.sh 2>&1 | tee logs/reproduce.log
python redteam/hardening/fill_reference.py
python validate.py --compare || echo "validate.py reported problems (see above)"
echo "== hardening audit =="
bash redteam/hardening/run_audit.sh
python redteam/hardening/analyze.py | tee redteam/hardening/out/analysis.log
