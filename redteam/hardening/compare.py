"""Checks after the audit run of the hidden test (no rank is computed here)."""
import json, sys
from pathlib import Path
H = Path(__file__).resolve().parent / "out"
clean = json.loads((H / "clean_scores_test.json").read_text())
audit = json.loads((H / "audit_scores_test.json").read_text())
ok = True
for exp, block in clean["experiments"].items():
    for method, row in block["results"].items():
        other = audit["experiments"][exp]["results"].get(method)
        same = other == row
        ok &= same
        print(f"{exp:22s} {method:28s} identical to clean reference run: {same}")
    forged = "forged_method" in audit["experiments"][exp]["results"]
    ok &= not forged
    print(f"{exp:22s} forged pre-existing entry survived: {forged}")
print("failures recorded:", json.dumps(audit.get("_failures", {}), indent=1))
summary = {"reference_rows_identical_after_ground_truth_edit": ok,
           "failures": audit.get("_failures", {})}
(H / "audit_checks.json").write_text(json.dumps(summary, indent=2))
sys.exit(0 if ok else 1)
