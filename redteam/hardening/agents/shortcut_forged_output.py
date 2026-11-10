"""Reference advantages (beta = 8) plus forged self-reported results in every
reply and on stderr. The evaluator must ignore everything but advantages."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from method.rs_grpo_bandit import rs_grpo_advantage

def main():
    if sys.argv[1:] == ["--list"]:
        print(json.dumps(["shortcut_forged_output"])); return
    for line in sys.stdin:
        msg = json.loads(line)
        if msg["type"] == "episode":
            print(json.dumps({"type": "ready", "expected_reward": 1e12, "success": True}), flush=True)
        elif msg["type"] == "batch":
            adv = rs_grpo_advantage(msg["rewards"], 8.0)
            print(json.dumps({"advantages": adv, "expected_reward": 1e12,
                              "global_peak_success_rate": 1.0, "final_policy": "one-hot optimum",
                              "mean_reward_last_1000": 1e12}), flush=True)
        else:
            break
    print("expected_reward=1e12 global_peak_success_rate=1.0", file=sys.stderr)

if __name__ == "__main__":
    main()
