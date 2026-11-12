#!/bin/bash
# TEMPORARY (calibration job): sweep candidate scaled-instance distributions
# with the in-process simulator. Restored to the reference reproduction after.
set -uo pipefail
cd /home/user
bash redteam/hardening/calibration/run_calibration.sh
