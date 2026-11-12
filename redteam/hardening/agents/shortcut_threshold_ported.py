"""The success-threshold shortcut (variant_success_only) ported to the
per-group offsets: +8 for samples whose reward exceeds the group minimum by a
fixed threshold tau, -1 otherwise, so groups without such a sample leave the
policy unchanged. On the paper-scale experiment (100 arms, public scale) it
uses tau = 0.75; on the scaled experiment tau is a constant chosen from the
public distribution of reward scales. One variant per tau."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.agent_protocol import Agent, main

TAUS = (0.2, 0.25, 0.3, 0.4, 0.5)
TAU_PAPER = 0.75


class Threshold(Agent):
    def __init__(self, tau):
        self.tau = tau

    def advantages(self, rewards):
        lo = min(rewards)
        tau = TAU_PAPER if self.info.n_arms == 100 else self.tau
        cap = self.info.advantage_cap
        return [cap if r - lo >= tau else -1.0 for r in rewards]


if __name__ == "__main__":
    main({f"shortcut_threshold_t{t:g}": (lambda t=t: Threshold(t)) for t in TAUS})
