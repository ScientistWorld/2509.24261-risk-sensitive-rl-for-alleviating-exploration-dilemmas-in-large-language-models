#!/bin/bash
# Hardening audit: rerun the reference agents, the honest variants and the
# shortcuts through the unchanged hidden-test evaluator (eval/test/), with the
# visible ground truth in data/ edited and every audit agent installed as an
# ordinary method file; then restore every file it touched.
#
# The run is split into one evaluator call per agent (--methods KEY), each
# with its own resume marker, because the scheduler may restart the job.
# Instances, offsets and sampling seeds depend only on (experiment, episode),
# and every agent runs in its own process, so a per-agent call scores an agent
# exactly as a single no-argument run would. The first call also writes where
# the test writes (scoring/scores_test.json), over a forged pre-existing file.
# Expects a clean reference run in scoring/ and logs/ first. AUDIT_DEV_ARGS
# (e.g. "--max-episodes 1 --num-batches 50") is for local smoke tests only.
set -uo pipefail
cd /home/user
export PYTHONPATH="/home/user:${PYTHONPATH:-}"
H=redteam/hardening
C="$H/out/chunks"
mkdir -p "$C"
# A resumed audit keeps the clean copies taken before the first attempt.
if [ ! -f "$H/out/clean_scores_test.json" ]; then
  cp scoring/scores_test.json "$H/out/clean_scores_test.json"
  cp scoring/scores_train.json "$H/out/clean_scores_train.json"
  cp logs/eval_test_details.json "$H/out/clean_test_details.json"
  cp logs/eval_train_details.json "$H/out/clean_train_details.json"
fi
rm -f method/audit_*.py
git checkout -- data/bandit_instances.py data/bandit_spec.py
# (b) edited visible ground truth
python3 "$H/edit_ground_truth.py"
# (c) audit agents installed as ordinary method files
for f in "$H"/agents/*.py; do cp "$f" "method/audit_$(basename "$f")"; done

# (a) failing agents, scored where the test writes, over a forged score file
if [ ! -f "$C/fail.done" ]; then
  python3 -c 'import json; json.dump({"experiments":{e:{"results":{"forged_method":{"expected_reward":1.0,"global_peak_success_rate":1.0}}} for e in ["bandit_escape","bandit_escape_scaled"]}}, open("scoring/scores_test.json","w"))'
  if python eval/test/bandit_eval.py ${AUDIT_DEV_ARGS:-} \
      --methods shortcut_crash,shortcut_nan_advantage,shortcut_oversized_advantage,audit_shortcut_hidden_helper \
      --details "$C/fail_details.json" > "$C/fail.log" 2>&1; then
    cp scoring/scores_test.json "$C/fail.json" && touch "$C/fail.done"
  fi
  echo "fail chunk done: $([ -f "$C/fail.done" ] && echo yes || echo no)"
fi

TEST_KEYS="rs_grpo_beta8 grpo_baseline grpo_beta0 rs_grpo_beta2 rs_grpo_beta4 rs_grpo_beta16
  honest_memory_aspiration honest_best_seen old_aspiration_known_scale variant_success_only
  shortcut_forged_output shortcut_threshold_t0.2 shortcut_threshold_t0.25 shortcut_threshold_t0.3
  shortcut_threshold_t0.4 shortcut_threshold_t0.5"
for key in $TEST_KEYS; do
  [ -f "$C/$key.done" ] && continue
  python eval/test/bandit_eval.py ${AUDIT_DEV_ARGS:-} --methods "$key" --output "$C/$key.json" \
    --details "$C/${key}_details.json" > "$C/$key.log" 2>&1 && touch "$C/$key.done"
  echo "test chunk $key done: $([ -f "$C/$key.done" ] && echo yes || echo no)"
done

TRAIN_KEYS="honest_memory_aspiration honest_best_seen shortcut_threshold_t0.3 variant_success_only"
for key in $TRAIN_KEYS; do
  [ -f "$C/train_$key.done" ] && continue
  python eval/train/bandit_eval.py ${AUDIT_DEV_ARGS:-} --methods "$key" --output "$C/train_$key.json" \
    --details "$C/train_${key}_details.json" > "$C/train_$key.log" 2>&1 && touch "$C/train_$key.done"
  echo "train chunk $key done: $([ -f "$C/train_$key.done" ] && echo yes || echo no)"
done

python3 "$H/merge_chunks.py" "fail $TEST_KEYS" "$TRAIN_KEYS"

# restore
rm -f method/audit_*.py
git checkout -- data/bandit_instances.py data/bandit_spec.py
cp "$H/out/clean_scores_test.json" scoring/scores_test.json
cp "$H/out/clean_test_details.json" logs/eval_test_details.json
python3 "$H/compare.py" | tee "$H/out/compare.log"
