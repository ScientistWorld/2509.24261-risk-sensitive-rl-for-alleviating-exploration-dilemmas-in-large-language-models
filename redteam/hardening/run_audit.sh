#!/bin/bash
# Hardening audit: rerun the honest variants and the shortcuts through the
# unchanged hidden test (scripts/evaluate_test.sh, no arguments), then restore
# every file it touched. Expects a clean reference run in scoring/ first.
set -uo pipefail
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
H=redteam/hardening
mkdir -p "$H/out"
cp scoring/scores_test.json "$H/out/clean_scores_test.json"
cp scoring/scores_train.json "$H/out/clean_scores_train.json"
cp logs/eval_test_details.json "$H/out/clean_test_details.json"
cp logs/eval_train_details.json "$H/out/clean_train_details.json"
# (a) a forged score file left where the test writes
python3 -c 'import json; json.dump({"experiments":{e:{"results":{"forged_method":{"expected_reward":1.0,"global_peak_success_rate":1.0}}} for e in ["bandit_escape","bandit_escape_scaled"]}}, open("scoring/scores_test.json","w"))'
# (b) edited visible ground truth
python3 "$H/edit_ground_truth.py"
# (c) audit agents installed as ordinary method files
for f in "$H"/agents/*.py; do cp "$f" "method/audit_$(basename "$f")"; done
bash scripts/evaluate_test.sh > "$H/out/audit_test.log" 2>&1; echo "audit test exit code $?" | tee -a "$H/out/audit_test.log"
cp scoring/scores_test.json "$H/out/audit_scores_test.json"
cp logs/eval_test_details.json "$H/out/audit_test_details.json"
python eval/train/bandit_eval.py \
  --methods honest_stronger_aspiration,variant_success_only \
  --output "$H/out/audit_scores_train.json" --details "$H/out/audit_train_details.json" \
  > "$H/out/audit_train.log" 2>&1; echo "audit train exit code $?" | tee -a "$H/out/audit_train.log"
# restore
rm -f method/audit_*.py
git checkout -- data/bandit_instances.py data/bandit_spec.py
cp "$H/out/clean_scores_test.json" scoring/scores_test.json
cp "$H/out/clean_test_details.json" logs/eval_test_details.json
python3 "$H/compare.py" | tee "$H/out/compare.log"
