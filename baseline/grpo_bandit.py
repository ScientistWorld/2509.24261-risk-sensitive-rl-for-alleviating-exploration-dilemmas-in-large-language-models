"""Risk-neutral GRPO baseline for the paper's bandit setting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from method.rs_grpo_bandit import RunConfig, run_bandit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120_000)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/bandit_baseline.json"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run_bandit(RunConfig(beta=0.0, steps=args.steps))
    args.output.write_text(json.dumps({"bandit_escape": {"beta_0": result}}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
