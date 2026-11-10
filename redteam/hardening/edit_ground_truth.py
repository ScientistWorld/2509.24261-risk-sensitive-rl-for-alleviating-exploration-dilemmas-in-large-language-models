"""Edit the visible ground truth in data/ (rescale and relabel rewards, drop
the success threshold). The hidden test must not change at all."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
p = ROOT / "data" / "bandit_instances.py"
s = p.read_text()
s2 = (s.replace("rewards[arm] = 0.1 + 0.5 * _bell(d)", "rewards[arm] = 10.0 + 0.5 * _bell(d)")
       .replace("0.1 + 0.9 * _bell(x - 40) + 0.5 * _bell(x - 75)", "10.0 * (0.1 + 0.9 * _bell(x - 75) + 0.5 * _bell(x - 40))")
       .replace("GLOBAL_PEAK_THRESHOLD = 0.9", "GLOBAL_PEAK_THRESHOLD = 0.0"))
assert s2 != s
p.write_text(s2)
q = ROOT / "data" / "bandit_spec.py"
t = q.read_text()
t2 = t.replace("rewards.append(0.1 + global_peak + local_peak)", "rewards.append(10.0 * (0.1 + global_peak + local_peak))")
assert t2 != t
q.write_text(t2)
print("edited data/bandit_instances.py and data/bandit_spec.py")
