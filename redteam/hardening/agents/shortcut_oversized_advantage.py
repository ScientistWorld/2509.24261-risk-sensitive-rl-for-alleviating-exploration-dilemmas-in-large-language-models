"""Instant-commit attempt: reference advantages multiplied by 100, which would
let a single sample move the policy almost arbitrarily far."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.agent_protocol import Agent, main
from method.rs_grpo_bandit import rs_grpo_advantage

class Oversized(Agent):
    def advantages(self, actions, rewards):
        return [100.0 * a for a in rs_grpo_advantage(rewards, 8.0)]

if __name__ == "__main__":
    main({"shortcut_oversized_advantage": Oversized})
