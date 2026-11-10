#!/bin/bash
set -euo pipefail
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
python3 redteam/run_case.py shortcut_fixed_data_relabel
