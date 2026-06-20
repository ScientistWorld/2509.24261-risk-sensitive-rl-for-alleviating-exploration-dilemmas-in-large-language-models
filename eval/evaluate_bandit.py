"""Evaluate saved RS-GRPO bandit artifacts without importing method code."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


METHOD_NAMES = {
    "beta_0": "grpo_beta0",
    "beta_2": "rs_grpo_beta2",
    "beta_4": "rs_grpo_beta4",
    "beta_8": "rs_grpo_beta8",
}

RUN_KEY_RE = re.compile(r"^(beta_[^_]+)(?:_seed_[-0-9]+)?$")


def summarize_run(run: dict) -> dict[str, float]:
    rewards = run["reward_by_arm"]
    final_policy = run["final_policy"]
    local_arm = int(run["metadata"]["local_arm"])
    global_arm = int(run["metadata"]["global_arm"])
    expected_reward = sum(p * r for p, r in zip(final_policy, rewards))
    global_peak_probability = sum(
        prob for prob, reward in zip(final_policy, rewards) if reward >= 0.9
    )
    local_peak_probability = sum(
        prob for arm, prob in enumerate(final_policy) if abs(arm - local_arm) <= 5
    )
    return {
        "expected_reward": float(expected_reward),
        "mean_reward_last_1000": float(run["mean_reward_last_1000"]),
        "optimal_action_probability": float(final_policy[global_arm]),
        "global_peak_probability": float(global_peak_probability),
        "local_action_probability": float(final_policy[local_arm]),
        "local_peak_probability": float(local_peak_probability),
        "global_peak_success_rate": float(
            expected_reward >= 0.9 and global_peak_probability >= 0.5
        ),
    }


def aggregate_runs(summaries: list[dict[str, float]]) -> dict[str, float]:
    metrics = summaries[0].keys()
    aggregate = {}
    for metric in metrics:
        values = [summary[metric] for summary in summaries]
        aggregate[metric] = float(sum(values) / len(values))
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("checkpoints/bandit_results.json"),
        help="Artifact written by method/rs_grpo_bandit.py or baseline/grpo_bandit.py",
    )
    parser.add_argument("--output", type=Path, default=Path("scoring/scores.json"))
    args = parser.parse_args()

    data = json.loads(args.input.read_text())
    runs = data["bandit_escape"]
    scores = {"experiments": {"bandit_escape": {"results": {}}}}

    grouped: dict[str, list[dict[str, float]]] = {}
    for key, run in runs.items():
        match = RUN_KEY_RE.match(key)
        base_key = match.group(1) if match else key
        method_name = METHOD_NAMES.get(base_key)
        if method_name is None:
            continue
        grouped.setdefault(method_name, []).append(summarize_run(run))

    for method_name, summaries in grouped.items():
        scores["experiments"]["bandit_escape"]["results"][method_name] = aggregate_runs(summaries)

    if not scores["experiments"]["bandit_escape"]["results"]:
        known = ", ".join(sorted(METHOD_NAMES))
        raise SystemExit(f"No recognized bandit runs found in {args.input}; expected keys like: {known}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
