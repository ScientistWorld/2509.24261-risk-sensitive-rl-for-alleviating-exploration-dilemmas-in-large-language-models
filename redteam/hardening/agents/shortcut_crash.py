"""Method that crashes partway through its first episode."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.agent_protocol import Agent, main
from method.rs_grpo_bandit import rs_grpo_advantage

class Crash(Agent):
    def start(self, info):
        super().start(info); self.t = 0
    def advantages(self, actions, rewards):
        self.t += 1
        if self.t == 50:
            raise RuntimeError("simulated failure")
        return rs_grpo_advantage(rewards, 8.0)

if __name__ == "__main__":
    main({"shortcut_crash": Crash})
