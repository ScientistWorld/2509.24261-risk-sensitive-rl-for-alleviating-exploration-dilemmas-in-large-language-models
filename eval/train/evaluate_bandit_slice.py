"""Score the visible train slice of saved bandit artifacts.

This evaluator is intentionally method-agnostic: it reads the JSON artifact
produced by a method and never imports from method/ or baseline/.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REFERENCE_METHOD_NAMES = {
    "beta_0": "grpo_beta0",
    "beta_2": "rs_grpo_beta2",
    "beta_4": "rs_grpo_beta4",
    "beta_8": "rs_grpo_beta8",
}

RUN_KEY_RE = re.compile(r"^(?P<base>.+?)(?:_seed_(?P<seed>-?\d+))?$")
MIN_RUNS_PER_METHOD = 4


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


def group_runs(runs: dict) -> dict[str, list[tuple[int | None, str, dict]]]:
    grouped: dict[str, list[tuple[int | None, str, dict]]] = {}
    for key, run in runs.items():
        base_key, seed = parse_run_key(key)
        method_name = canonical_method_name(base_key)
        grouped.setdefault(method_name, []).append((seed, key, run))
    return grouped


def select_train_runs(run_items: list[tuple[int | None, str, dict]]) -> list[dict]:
    ordered = sorted(run_items, key=lambda item: (item[0] is None, item[0] or 0, item[1]))
    if len(ordered) < MIN_RUNS_PER_METHOD:
        raise ValueError(
            f"Need at least {MIN_RUNS_PER_METHOD} saved runs per method for a "
            "non-degenerate train/test split."
        )
    return [run for index, (_, _, run) in enumerate(ordered) if index % 2 == 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("checkpoints/bandit_results.json"),
        help="Saved bandit artifact produced by a method implementation.",
    )
    parser.add_argument("--output", type=Path, default=Path("scoring/scores_train.json"))
    args = parser.parse_args()

    data = json.loads(args.input.read_text())
    grouped = group_runs(data["bandit_escape"])

    results = {}
    split_counts = {}
    for method_name in sorted(grouped):
        try:
            selected = select_train_runs(grouped[method_name])
        except ValueError as exc:
            raise SystemExit(f"{method_name}: {exc}") from exc
        if not selected:
            continue
        summaries = [summarize_run(run) for run in selected]
        results[method_name] = aggregate_runs(summaries)
        split_counts[method_name] = {
            "selected_runs": len(selected),
            "available_runs": len(grouped[method_name]),
        }

    if not results:
        raise SystemExit(f"No bandit runs found in {args.input}")

    scores = {
        "experiments": {
            "bandit_escape": {
                "slice": "train",
                "split_rule": "Per-method saved runs sorted by seed/key; even indices are train.",
                "split_counts": split_counts,
                "results": results,
            }
        }
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
