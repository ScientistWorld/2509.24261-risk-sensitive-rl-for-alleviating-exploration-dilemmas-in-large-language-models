"""Brute-force-style search inside the protocol: push every already-seen arm
down (count-based) to visit as many distinct arms as possible during the
first 2000 batches, remember the best arm, then push only that arm up."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.agent_protocol import Agent, main

EXPLORE_BATCHES = 2000

class Enumerate(Agent):
    def start(self, info):
        super().start(info)
        self.t = 0; self.count = {}; self.best_arm = None; self.best = -1.0
    def advantages(self, actions, rewards):
        cap = self.info.advantage_cap
        for a, r in zip(actions, rewards):
            if r > self.best:
                self.best, self.best_arm = r, a
        if self.t < EXPLORE_BATCHES:
            out = [cap if self.count.get(a, 0) == 0 else -min(cap, float(self.count[a])) for a in actions]
        else:
            out = [cap if a == self.best_arm else -cap / 2 for a in actions]
        for a in actions:
            self.count[a] = self.count.get(a, 0) + 1
        self.t += 1
        return out

if __name__ == "__main__":
    main({"shortcut_enumerate": Enumerate})
