"""Honest stronger variant: exponential utility (as in Eq. 8) measured against
the best reward level seen so far in the episode instead of the group's soft
value. Scale- and offset-free: rewards are anchored at the group minimum and
divided by the best anchored value seen (memory across groups), so it needs no
knowledge of the reward scale, the success level or the instance generator.

    x_i = r_i - min_j r_j,  M = max over all groups so far of max_i x_i
    e_i = exp(beta * (x_i - M) / M),  A_i = C * (e_i - mean_j e_j) / beta

beta = 16, C = 32 (A within the cap). Groups whose samples are all far below
the best level seen get nearly zero advantages, so the policy is not pushed
towards a mode once something better has been seen; a group containing a new
best gets a full-size advantage for it.
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.agent_protocol import Agent, main

BETA, SCALE = 16.0, 32.0


class MemoryAspiration(Agent):
    def start(self, info):
        super().start(info)
        self.best = 0.0

    def advantages(self, rewards):
        lo = min(rewards)
        x = [r - lo for r in rewards]
        self.best = max(self.best, max(x))
        if self.best <= 0.0:
            return [0.0] * len(rewards)
        e = [math.exp(BETA * (v - self.best) / self.best) for v in x]
        mean = sum(e) / len(e)
        cap = self.info.advantage_cap
        return [max(-cap, min(cap, SCALE * (v - mean) / BETA)) for v in e]


if __name__ == "__main__":
    main({"honest_memory_aspiration": MemoryAspiration})
