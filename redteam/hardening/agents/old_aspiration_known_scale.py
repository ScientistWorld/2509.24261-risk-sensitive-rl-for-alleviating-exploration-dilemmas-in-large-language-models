"""The previous iteration's honest stronger variant, run verbatim under the
new protocol: risk-seeking advantage against a fixed aspiration.

Same exponential utility as Eq. 8 (beta = 8), but each sample is compared
with the known best attainable reward (1.0) instead of the group's soft value:
A_i = C * (exp(beta (r_i - 1)) - mean_j exp(beta (r_j - 1))) / beta, C = 16.
Groups without a near-optimal sample therefore barely reinforce the local
mode, while a group that contains one gives it a full-size advantage. Uses
only the sampled rewards of the current group. It assumes the public reward
scale (best reward 1.0), which per-group offsets and hidden per-instance
scales now make unknowable.
"""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.agent_protocol import Agent, main

BETA, SCALE, ASPIRATION = 8.0, 16.0, 1.0

class Aspiration(Agent):
    def advantages(self, rewards):
        e = [math.exp(BETA * (r - ASPIRATION)) for r in rewards]
        mean = sum(e) / len(e)
        cap = self.info.advantage_cap
        return [max(-cap, min(cap, SCALE * (x - mean) / BETA)) for x in e]

if __name__ == "__main__":
    main({"old_aspiration_known_scale": Aspiration})
