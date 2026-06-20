"""Evaluate saved bandit artifacts without importing method code."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from data.bandit_spec import bandit_metadata, bimodal_bandit_rewards


REFERENCE_METHOD_NAMES = {
    "beta_0": "grpo_beta0",
    "beta_2": "rs_grpo_beta2",
    "beta_4": "rs_grpo_beta4",
    "beta_8": "rs_grpo_beta8",
}

RUN_KEY_RE = re.compile(r"^(?P<base>.+?)(?:_seed_(?P<seed>-?\d+))?$")


def validate_run(key: str, run: dict) -> None:
    """Reject artifacts that do not use the fixed benchmark definition."""
    expected_rewards = bimodal_bandit_rewards()
    expected_meta = bandit_metadata()

    rewards = run.get("reward_by_arm")
    if rewards != expected_rewards:
        raise ValueError(f"{key}: reward_by_arm does not match data/bandit_spec.py")

    metadata = run.get("metadata", {})
    for field in ("n_arms", "global_arm", "local_arm"):
        if int(metadata.get(field, -1)) != int(expected_meta[field]):
            raise ValueError(f"{key}: metadata.{field} does not match data/bandit_spec.py")

    final_policy = run.get("final_policy")
    if not isinstance(final_policy, list) or len(final_policy) != len(expected_rewards):
        raise ValueError(f"{key}: final_policy must contain {len(expected_rewards)} probabilities")
    if any((not isinstance(prob, (int, float))) or prob < 0 for prob in final_policy):
        raise ValueError(f"{key}: final_policy contains a non-numeric or negative probability")
    if abs(sum(final_policy) - 1.0) > 1e-6:
        raise ValueError(f"{key}: final_policy probabilities must sum to 1")


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


def parse_run_key(key: str) -> tuple[str, int | None]:
    match = RUN_KEY_RE.match(key)
    if not match:
        return key, None
    base = match.group("base")
    seed_text = match.group("seed")
    return base, int(seed_text) if seed_text is not None else None


def canonical_method_name(base_key: str) -> str:
    return REFERENCE_METHOD_NAMES.get(base_key, base_key)


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
        validate_run(key, run)
        base_key, _ = parse_run_key(key)
        method_name = canonical_method_name(base_key)
        grouped.setdefault(method_name, []).append(summarize_run(run))

    for method_name, summaries in grouped.items():
        scores["experiments"]["bandit_escape"]["results"][method_name] = aggregate_runs(summaries)

    if not scores["experiments"]["bandit_escape"]["results"]:
        raise SystemExit(f"No bandit runs found in {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
