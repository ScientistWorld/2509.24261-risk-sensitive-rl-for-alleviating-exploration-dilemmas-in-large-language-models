Job 711885e3-8b5 completed all cases, but method wrappers were invoked with -m
and copied over a same-size source file within the same second. Python reused
the bytecode wrapper for shortcut_nonfinite_policy when running the following
shortcut_constant_uniform case. That control is invalidated and rerun. The
harness now executes the wrapper as a source script; it does not alter the
Python environment or the evaluator. All cases are rerun for consistent evidence.

The first reference reruns passed an integer beta=4 while the original CLI
passes float beta=4.0. The second pass matches the CLI argument type to test
whether that explains the same-seed mismatch. No claim is made that it does.
