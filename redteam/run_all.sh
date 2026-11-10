#!/bin/bash
set -euo pipefail
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
while read -r name; do
  bash "redteam/$name/run.sh"
done < redteam/cases.txt
