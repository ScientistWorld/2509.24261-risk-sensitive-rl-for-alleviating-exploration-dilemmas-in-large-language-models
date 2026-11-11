"""Reward-only analogue of 'sample from the prior, then commit': reinforce only
samples that reach the success threshold (reward >= 0.9) with the maximal
advantage, give everything else -1. Groups without a success leave the policy
unchanged (after centring), so the initial policy keeps sampling until the
optimum is found. A simple advantage rule, not the paper's method."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.agent_protocol import Agent, main

class SuccessOnly(Agent):
    def advantages(self, rewards):
        cap = self.info.advantage_cap
        return [cap if r >= 0.9 else -1.0 for r in rewards]

if __name__ == "__main__":
    main({"variant_success_only": SuccessOnly})
