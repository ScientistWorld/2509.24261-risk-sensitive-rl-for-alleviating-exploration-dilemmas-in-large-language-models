"""Merge the per-agent audit chunks into one score file and one details file
per slice (out/audit_scores_{test,train}.json, out/audit_{test,train}_details.json),
the same shape a single evaluator run writes."""
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
C = OUT / "chunks"


def merge(names, prefix, slice_name):
    scores = {"experiments": {}, "_failures": {}, "_chunks": []}
    details = {"slice": slice_name, "failures": {}, "episodes": {}}
    for name in names:
        score_path = C / f"{prefix}{name}.json"
        detail_path = C / f"{prefix}{name}_details.json"
        if not score_path.exists() or not detail_path.exists():
            print(f"missing chunk {prefix}{name}")
            continue
        s = json.loads(score_path.read_text())
        d = json.loads(detail_path.read_text())
        scores["_chunks"].append(f"{prefix}{name}")
        for exp, block in s.get("experiments", {}).items():
            dst = scores["experiments"].setdefault(exp, {"results": {}})
            dst["results"].update(block.get("results", {}))
        scores["_failures"].update(s.get("_failures", {}))
        details["failures"].update(d.get("failures", {}))
        details["episodes"].update(d.get("episodes", {}))
    return scores, details


def main():
    test_names = sys.argv[1].split()
    train_names = sys.argv[2].split()
    s, d = merge(test_names, "", "test")
    (OUT / "audit_scores_test.json").write_text(json.dumps(s, indent=2))
    (OUT / "audit_test_details.json").write_text(json.dumps(d))
    s, d = merge(train_names, "train_", "train")
    (OUT / "audit_scores_train.json").write_text(json.dumps(s, indent=2))
    (OUT / "audit_train_details.json").write_text(json.dumps(d))
    print("merged", len(test_names), "test chunks and", len(train_names), "train chunks")


if __name__ == "__main__":
    main()
