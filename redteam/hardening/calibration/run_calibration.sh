#!/bin/bash
# Calibration sweep (CPU, 1 core, ~1.5 h): every candidate instance
# distribution x every rule, paired episodes. Resumable: a config whose output
# file is complete is skipped.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p out
for cfg in E1 C0 C1 C2 C3 C4 C5 C6; do
  eps=32
  [ "$cfg" = "E1" ] && eps=24
  out="out/${cfg}.jsonl"
  if [ -f "$out" ] && [ -f "out/${cfg}.done" ]; then
    echo "skip $cfg"; continue
  fi
  python3 sim.py "$cfg" "$eps" 7500 > "$out" 2> "out/${cfg}.log" && touch "out/${cfg}.done"
  echo "$cfg exit $?"
done
