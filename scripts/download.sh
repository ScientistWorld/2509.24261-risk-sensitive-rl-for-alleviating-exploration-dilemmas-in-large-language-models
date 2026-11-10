#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0 GB
# Estimated disk usage after extraction: 0 GB
#
# This environment has no external datasets or model weights: every bandit
# instance is generated procedurally by data/bandit_instances.py (visible
# copy) and by the evaluators in eval/ at scoring time.

set -e

cd /home/user

# Nothing to fetch; keep the gitignored download folder present.
mkdir -p data/downloads
