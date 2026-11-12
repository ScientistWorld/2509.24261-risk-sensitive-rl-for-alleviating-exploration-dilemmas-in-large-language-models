"""Honest variant: best-seen elitism. Anchors each group's rewards at the group
minimum, remembers the best anchored value seen in the episode, and gives the
maximal advantage to samples within 5% of it and -1 to all others. Scale- and
offset-free (no knowledge of the reward scale or success level), but it
binarises the reward, so it converges less precisely than a dense rule.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.agent_protocol import Agent, main

TOL = 0.05


class BestSeen(Agent):
    def start(self, info):
        super().start(info)
        self.best = 0.0

    def advantages(self, rewards):
        lo = min(rewards)
        x = [r - lo for r in rewards]
        self.best = max(self.best, max(x))
        if self.best <= 0.0:
            return [0.0] * len(rewards)
        cap = self.info.advantage_cap
        return [cap if v >= (1.0 - TOL) * self.best else -1.0 for v in x]


if __name__ == "__main__":
    main({"honest_best_seen": BestSeen})
