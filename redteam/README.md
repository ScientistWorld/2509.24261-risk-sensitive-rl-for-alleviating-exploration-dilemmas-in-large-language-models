# Retained scoring audit

Read ../REDTEAM.md for conclusions and score interpretation. analysis.json and
results.csv preserve all metrics and explicit normalization cohorts. Each case
contains run.sh, method.py, run_result.json, the actual candidate artifact, and
scores_test.json when the test left one. Failed cases also have
failure_details.json. Missing test-score files are intentional evidence of the
failure-accounting defect; they are not zero-valued scores.

run_case.py installs the common support/candidate.py plus the per-case wrapper,
runs the method to completion, calls evaluate_train.sh, then calls the unchanged
scripts/evaluate_test.sh without arguments. Non-audit paths are restored in a
finally block. Shortcuts never import hidden code or read hidden score files.
The reference comparison metadata is used only by analyze.py.

All names prefixed shortcut_ are shortcuts, including exact-reference ties and
unsuccessful controls. A nonzero test exit with a surviving forged score is
explicitly conditional evidence, not a claimed successful score.
