"""Light analysis of the hardening audit (reads saved per-episode details only).

1. Normalised rank score of every method on the hidden test, per the gym's
   ranking rule: sum over experiments (weight) and metrics (|coefficient|) of
   the improvement over the reference in the metric's better direction,
   divided by the metric's standard deviation across the ranked cohort.
2. Paired per-episode differences (same hidden instances, same sampling
   seeds) with standard errors, for the comparisons the hardening report needs.
3. Seed/instance noise of the reference: its normalised score on the visible
   slice (independent instances and seeds) versus the hidden slice, and the
   bootstrap spread of its hidden-slice score over episodes.
"""
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"
REF = json.loads((ROOT / "scoring" / "reference.json").read_text())
REFERENCE = REF["_reference_method"]
EXPS = REF["experiments"]


def load(name):
    return json.loads((OUT / name).read_text())


def aggregate(details, method):
    out = {}
    if method in details["failures"]:
        return None
    for exp, runs in details["episodes"][method].items():
        out[exp] = {m: sum(r[m] for r in runs) / len(runs) for m in runs[0]}
    return out


def rank_scores(table, reference):
    """table[method][exp][metric] -> normalised score relative to reference."""
    methods = list(table)
    scores = {m: 0.0 for m in methods}
    for exp, spec in EXPS.items():
        for metric, mdef in spec["metrics"].items():
            values = [table[m][exp][metric] for m in methods]
            mean = sum(values) / len(values)
            sd = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) or 1.0
            sign = 1.0 if mdef["higher_is_better"] else -1.0
            for m in methods:
                gain = sign * (table[m][exp][metric] - table[reference][exp][metric])
                scores[m] += spec["weight"] * abs(mdef["coefficient"]) * gain / sd
    return scores


def paired(details, a, b, exp, metric):
    xa = details["episodes"][a][exp]
    xb = details["episodes"][b][exp]
    d = [p[metric] - q[metric] for p, q in zip(xa, xb)]
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    return mean, sd / math.sqrt(n), n


def main():
    audit = load("audit_test_details.json")
    worst = {exp: {m: (0.0 if d["higher_is_better"] else 1.0) for m, d in spec["metrics"].items()}
             for exp, spec in EXPS.items()}
    table = {}
    for method in audit["episodes"]:
        agg = aggregate(audit, method)
        table[method] = agg if agg is not None else worst
    report = {"reference": REFERENCE}
    report["rank_all_rows"] = rank_scores(table, REFERENCE)
    valid = {m: v for m, v in table.items() if m not in audit["failures"]}
    report["rank_valid_rows"] = rank_scores(valid, REFERENCE)

    ported = sorted(m for m in valid if m.startswith("shortcut_threshold_t"))
    best_ported = max(ported, key=lambda m: report["rank_valid_rows"][m]) if ported else None
    report["best_ported_threshold"] = best_ported
    comparisons = [
        ("honest_memory_aspiration", REFERENCE),
        ("honest_best_seen", REFERENCE),
        (REFERENCE, "grpo_baseline"),
        (REFERENCE, "rs_grpo_beta4"),
        (REFERENCE, "rs_grpo_beta16"),
        ("rs_grpo_beta4", "grpo_baseline"),
        ("shortcut_forged_output", REFERENCE),
    ] + ([(best_ported, REFERENCE)] if best_ported else [])
    comparisons = [(a, b) for a, b in comparisons if a in valid and b in valid]
    report["paired_test"] = {}
    for a, b in comparisons:
        for exp in EXPS:
            for metric in ("expected_reward", "global_peak_success_rate", "online_mean_reward",
                           "optimal_action_probability"):
                mean, se, n = paired(audit, a, b, exp, metric)
                report["paired_test"][f"{a} - {b} | {exp} | {metric}"] = [round(mean, 4), round(se, 4), n]

    # Paired bootstrap of rank differences: resample episodes (the same
    # episodes for every method), recompute the cohort table and the ranks.
    rng = random.Random(1)
    n_eps = {exp: len(audit["episodes"][REFERENCE][exp]) for exp in EXPS}
    diffs = {a: [] for a, b in comparisons if b == REFERENCE}
    diffs["grpo_baseline"] = []
    for _ in range(200):
        picks = {exp: [rng.randrange(n) for _ in range(n)] for exp, n in n_eps.items()}
        t = {}
        for m in valid:
            t[m] = {}
            for exp in EXPS:
                runs = audit["episodes"][m][exp]
                sel = [runs[i] for i in picks[exp]]
                t[m][exp] = {k: sum(r[k] for r in sel) / len(sel) for k in sel[0]}
        rk = rank_scores(t, REFERENCE)
        for m in diffs:
            diffs[m].append(rk[m])
    report["paired_bootstrap_rank_vs_reference"] = {}
    for m, v in diffs.items():
        mu = sum(v) / len(v)
        sd = math.sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1))
        v.sort()
        report["paired_bootstrap_rank_vs_reference"][m] = {
            "mean": round(mu, 3), "sd": round(sd, 3), "p05": round(v[10], 3), "p95": round(v[189], 3)}

    # Reference noise: visible vs hidden slice (independent instances and seeds),
    # scored on the same cohort scale as the hidden-slice ranking.
    clean_train = load("clean_train_details.json")
    clean_test = load("clean_test_details.json")
    ref_train = aggregate(clean_train, REFERENCE)
    ref_test = aggregate(clean_test, REFERENCE)
    noise_table = dict(valid)
    noise_table["_reference_visible_slice"] = ref_train
    report["reference_visible_vs_hidden"] = rank_scores(noise_table, REFERENCE)["_reference_visible_slice"]

    rng = random.Random(0)
    boots = []
    for _ in range(300):
        sample = {}
        for exp, runs in clean_test["episodes"][REFERENCE].items():
            pick = [runs[rng.randrange(len(runs))] for _ in runs]
            sample[exp] = {m: sum(r[m] for r in pick) / len(pick) for m in runs[0]}
        t = dict(valid)
        t["_boot"] = sample
        boots.append(rank_scores(t, REFERENCE)["_boot"])
    boots.sort()
    report["reference_bootstrap_over_episodes"] = {
        "sd": round(math.sqrt(sum(b * b for b in boots) / len(boots) - (sum(boots) / len(boots)) ** 2), 4),
        "p05": round(boots[15], 4), "p95": round(boots[284], 4)}

    (OUT / "analysis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in report["rank_valid_rows"].items()}, indent=1))
    print("all rows:", json.dumps({k: round(v, 3) for k, v in report["rank_all_rows"].items()}))
    for k, v in report["paired_test"].items():
        if "expected_reward" in k or "success" in k:
            print(f"{k:95s} {v[0]:+.4f} ± {v[1]:.4f} (n={v[2]})")
    print("reference visible-slice score on hidden-cohort scale:", round(report["reference_visible_vs_hidden"], 3))
    print("reference bootstrap:", report["reference_bootstrap_over_episodes"])
    print("paired bootstrap of rank vs reference:", json.dumps(report["paired_bootstrap_rank_vs_reference"], indent=1))


if __name__ == "__main__":
    main()
