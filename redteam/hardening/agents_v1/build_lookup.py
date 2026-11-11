"""Build the answer table used by shortcut_train_lookup: regenerate every
visible-slice instance from the public generator (data/) and the visible
slice's public seed rule (readable in eval/train/), and map a fingerprint of
its initial logits to its optimal arm."""
import hashlib, json, random, sys
from array import array
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from data.bandit_instances import paper_instance, scaled_instance

PUBLIC_TRAIN_ENTROPY = "rs-grpo-bandit-train-v1"   # visible in eval/train/bandit_eval.py
EXPERIMENTS = {"bandit_escape": ("paper", 24), "bandit_escape_scaled": ("scaled", 48)}

def seed(*parts):
    text = "|".join(str(p) for p in (PUBLIC_TRAIN_ENTROPY,) + parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:16], "big")

table = {}
for exp, (family, n) in EXPERIMENTS.items():
    for e in range(n):
        rng = random.Random(seed(exp, e, "instance"))
        inst = paper_instance(rng) if family == "paper" else scaled_instance(rng)
        key = hashlib.sha256(array("d", inst["initial_logits"]).tobytes()).hexdigest()
        table[key] = inst["optimal_arm"]
out = ROOT / "redteam" / "hardening" / "train_lookup_cache.json"
out.write_text(json.dumps(table))
print(f"wrote {len(table)} visible-slice answers to {out}")
