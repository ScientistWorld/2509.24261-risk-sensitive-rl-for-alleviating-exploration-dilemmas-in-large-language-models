"""Summarise the calibration sweep: per-rule means per distribution, paired
differences to rs8, and the gym's rank rule (E1 weight 0.4 + scaled weight
0.6; coefficient x improvement over rs8 in the better direction / std of the
metric across the cohort of rules)."""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).parent / "out"
COEF = {"expected_reward": 1.0, "mean_reward_last_1000": 0.5, "online_mean_reward": 0.5,
        "optimal_action_probability": 0.4, "global_peak_probability": 0.8,
        "global_peak_success_rate": 0.6, "local_action_probability": 0.4,
        "local_peak_probability": 0.5}
HIB = {k: not k.startswith("local") for k in COEF}
W = {"E1": 0.4, "E2": 0.6}


def load(name):
    rows = defaultdict(dict)
    info = {}
    for line in open(OUT / f"{name}.jsonl"):
        d = json.loads(line)
        rows[d["rule"]][d["ep"]] = d["m"]
        info[d["ep"]] = d["info"]
    return rows, info


def means(rows):
    return {r: {k: sum(e[k] for e in eps.values()) / len(eps) for k in COEF} for r, eps in rows.items()}


def rank(e1, e2, ref="rs8", cohort=None):
    cohort = cohort or list(e2)
    total = defaultdict(float)
    for exp, mm in (("E1", e1), ("E2", e2)):
        for k, c in COEF.items():
            vals = [mm[r][k] for r in cohort]
            mu = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / max(1, len(vals) - 1)) or 1.0
            for r in mm:
                d = mm[r][k] - mm[ref][k]
                if not HIB[k]:
                    d = -d
                total[r] += W[exp] * c * d / sd
    return dict(total)


def paired(rows, a, b, k):
    ds = [rows[a][e][k] - rows[b][e][k] for e in rows[b]]
    n = len(ds)
    mu = sum(ds) / n
    se = math.sqrt(sum((d - mu) ** 2 for d in ds) / (n - 1) / n)
    return mu, se


def main():
    e1rows, _ = load("E1")
    e1 = means(e1rows)
    cfgs = sys.argv[1:] or ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
    print("E1 means")
    for r, m in sorted(e1.items(), key=lambda x: -x[1]["expected_reward"]):
        print(f"  {r:12s} ER={m['expected_reward']:.3f} onl={m['online_mean_reward']:.3f} "
              f"opt={m['optimal_action_probability']:.3f} succ={m['global_peak_success_rate']:.2f}")
    for cfg in cfgs:
        if not (OUT / f"{cfg}.jsonl").exists():
            continue
        rows, info = load(cfg)
        e2 = means(rows)
        cohort = [r for r in e2 if not r.startswith("anch")] + ["anch0.3"]
        rk = rank(e1, e2, cohort=cohort)
        print(f"\n== {cfg}  (episodes {len(rows['rs8'])})")
        for r in sorted(e2, key=lambda r: -rk[r]):
            m = e2[r]
            er, se = paired(rows, r, "rs8", "expected_reward") if r != "rs8" else (0, 0)
            print(f"  {r:12s} rank={rk[r]:+6.2f}  ER={m['expected_reward']:.3f} (d {er:+.3f}±{se:.3f}) "
                  f"succ={m['global_peak_success_rate']:.2f} onl={m['online_mean_reward']:.3f} "
                  f"opt={m['optimal_action_probability']:.3f} locP={m['local_peak_probability']:.2f}")


if __name__ == "__main__":
    main()
