#!/bin/bash
# Job entry point (CPU, 1 core, ~2.5 h): reproduce the reference numbers end
# to end on both slices, record the scaled experiment's visible-slice numbers
# in scoring/reference.json, validate the workspace, then rerun the hardening
# audit (honest variants and shortcuts through the unchanged hidden test,
# restoring every file it touches) and its analysis.
#
# Each stage leaves a marker in logs/job_state/ when it completes, so a
# restarted job resumes at the first unfinished stage. Delete that folder
# (or run with FRESH=1) to rerun everything.
set -euo pipefail
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
STATE=logs/job_state
[ "${FRESH:-0}" = "1" ] && rm -rf "$STATE"
mkdir -p logs "$STATE"

stage() {  # stage NAME COMMAND...
  local name=$1; shift
  if [ -f "$STATE/$name.done" ]; then echo "== $name: already done, skipping =="; return 0; fi
  echo "== $name =="
  "$@"
  touch "$STATE/$name.done"
}

stage train bash -c 'bash scripts/evaluate_train.sh 2>&1 | tee logs/reproduce_train.log; cp scoring/scores_train.json scoring/scores.json'
stage test bash -c 'bash scripts/evaluate_test.sh 2>&1 | tee logs/reproduce_test.log'
python redteam/hardening/fill_reference.py
python validate.py --compare || echo "validate.py reported problems (see above)"
stage audit bash redteam/hardening/run_audit.sh
python redteam/hardening/analyze.py | tee redteam/hardening/out/analysis.log
