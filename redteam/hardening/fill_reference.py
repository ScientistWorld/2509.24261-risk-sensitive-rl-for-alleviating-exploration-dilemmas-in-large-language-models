"""Record the reference reproduction's visible-slice numbers for the scaled
experiment in scoring/reference.json (the paper reports none at this scale)."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
ref_path = ROOT / "scoring" / "reference.json"
ref = json.loads(ref_path.read_text())
train = json.loads((ROOT / "scoring" / "scores_train.json").read_text())
exp = "bandit_escape_scaled"
rows = train["experiments"][exp]["results"]
for method, entry in ref["experiments"][exp]["results"].items():
    kind = entry["type"]
    new = {"type": kind}
    new.update({k: round(v, 4) for k, v in rows[method].items()})
    ref["experiments"][exp]["results"][method] = new
ref_path.write_text(json.dumps(ref, indent=2) + "\n")
print("filled", exp, "results from scores_train.json")
