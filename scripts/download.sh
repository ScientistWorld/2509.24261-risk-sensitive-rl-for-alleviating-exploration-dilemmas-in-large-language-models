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
# This reproduction uses a procedural benchmark, so there are no remote
# resources to fetch.

set -e

cd /home/user

# The runnable core reproduction uses the paper's procedural bandit setting.
# No external datasets or pretrained weights are required.
mkdir -p data/downloads
