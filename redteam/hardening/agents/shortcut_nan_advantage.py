"""Non-finite output: reference advantages, but NaN from batch 10 on."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.agent_protocol import Agent, main
from method.rs_grpo_bandit import rs_grpo_advantage

class NaNAgent(Agent):
    def start(self, info):
        super().start(info); self.t = 0
    def advantages(self, rewards):
        self.t += 1
        adv = rs_grpo_advantage(rewards, 8.0)
        if self.t > 10:
            adv[0] = float("nan")
        return adv

if __name__ == "__main__":
    main({"shortcut_nan_advantage": NaNAgent})
