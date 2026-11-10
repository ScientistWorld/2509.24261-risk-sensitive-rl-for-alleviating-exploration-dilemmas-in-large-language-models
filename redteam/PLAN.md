Audit scope: hidden test scoring quality and major scientist-plausible shortcuts.
Every case has a run.sh which installs its method and invokes both visible eval
and the hidden test, without arguments, in separate completed processes. The
harness records failures and keeps any surviving score file; it does not turn
failures into successes. It restores the original non-audit files after each
case. Candidate code never reads hidden evaluators, hidden seeds, hidden scores,
or the paper. Output-only shortcuts reuse the original method's trained JSON,
as authorized by the audit. reference_seed0 independently reruns the current
beta4 implementation. Its scores do not match the saved reference artifact;
REDTEAM.md documents this unresolved provenance issue. To independently train
a new artifact from an empty workspace (do not overwrite the saved evidence):
PYTHONPATH=. python3 redteam/support/reference_method.py --betas 0 2 4 8 --seeds 0 1 2 3 4 5 6 7 8 9 --output checkpoints/new_reference.json
The support/reference.json and original_scores_test.json are auditor comparison
inputs only, not candidate inputs. No candidate changes the test code, Python
environment, or running evaluator process.

Quality check: 3 independent batches of 10 reference seeds (0..9, 10..19,20..29);
an equal-budget sample-only critic/natural-gradient improvement; and removing
risk sensitivity as an ablation. Failures and nonfinite output are reported
separately from numeric ranking. An eventual standardized audit-cohort score is
only a diagnostic: future ranks depend on the future cohort's metric variances.
