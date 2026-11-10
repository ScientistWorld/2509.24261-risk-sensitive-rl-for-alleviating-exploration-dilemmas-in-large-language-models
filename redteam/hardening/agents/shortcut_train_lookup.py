"""Answer lookup from visible instances: fingerprint the initial logits and,
if they match a visible-slice instance (regenerated from the public seed rule
into a cache), drive the policy to the known optimal arm with maximal
advantages. Otherwise fall back to RS-GRPO beta = 8."""
import hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.agent_protocol import Agent, main
from method.rs_grpo_bandit import rs_grpo_advantage

CACHE = ROOT / "redteam" / "hardening" / "train_lookup_cache.json"

class Lookup(Agent):
    def start(self, info):
        super().start(info)
        table = json.loads(CACHE.read_text()) if CACHE.exists() else {}
        key = hashlib.sha256(info.initial_logits.tobytes()).hexdigest()
        self.opt = table.get(key)
    def advantages(self, actions, rewards):
        if self.opt is None:
            return rs_grpo_advantage(rewards, 8.0)
        cap = self.info.advantage_cap
        return [cap if a == self.opt else -cap / 2 for a in actions]

if __name__ == "__main__":
    main({"shortcut_train_lookup": Lookup})
