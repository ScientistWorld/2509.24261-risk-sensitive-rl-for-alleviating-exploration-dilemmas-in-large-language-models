"""Visible (train) evaluator for the bandit exploration benchmark.

Runs every agent found on disk (``method/*.py`` and ``baseline/*.py``, names
not starting with ``_``) through the agent protocol documented in
``data/agent_protocol.py`` on instances generated here, and scores the final
policies. Everything that is scored is computed by this file: the instances,
the sampled actions, the rewards, the policy updates and every metric. The
agent only returns advantages, which are checked (finite, |A| <= cap) and
centred before the fixed softmax policy-gradient step is applied.

This module is self-contained on purpose: it imports nothing from
``data/``, ``method/``, ``baseline/`` or ``eval/test/``.

Usage: ``python eval/train/bandit_eval.py`` (no arguments scores every method
on disk on the full slice). Optional flags exist for quick development runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import select
import subprocess
import sys
import tempfile
import time
from array import array
from bisect import bisect_right
from itertools import accumulate
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLICE = "train"
DEFAULT_OUTPUT = ROOT / "scoring" / "scores_train.json"
DEFAULT_DETAILS = ROOT / "logs" / "eval_train_details.json"
# Public seed material for the visible slice. The hidden slice uses its own.
SLICE_ENTROPY = "rs-grpo-bandit-train-v1"

AGENT_DIRS = ("method", "baseline")
NUM_BATCHES = 7500  # 120,000 interactions
BATCH_SIZE = 16
LEARNING_RATE = 0.1
ADVANTAGE_CAP = 8.0
REPLY_TIMEOUT = 300.0
LIST_TIMEOUT = 120.0

GLOBAL_PEAK_THRESHOLD = 0.9
SUCCESS_MASS = 0.5
PAPER_N_ARMS = 100
SCALED_N_ARMS = 2 ** 18
PEAK_HALF_WIDTH = 10
LOCAL_NEIGHBOURHOOD = 5
P_GLOBAL_RANGE = (0.01, 0.15)

EXPERIMENTS = {
    "bandit_escape": {"family": "paper", "episodes": 24},
    "bandit_escape_scaled": {"family": "scaled", "episodes": 48},
}

HIGHER_IS_BETTER = {
    "expected_reward": True,
    "mean_reward_last_1000": True,
    "online_mean_reward": True,
    "optimal_action_probability": True,
    "global_peak_probability": True,
    "global_peak_success_rate": True,
    "local_action_probability": False,
    "local_peak_probability": False,
}
# Worst attainable value of every metric (rewards lie in [0, 1]).
WORST = {name: (0.0 if better else 1.0) for name, better in HIGHER_IS_BETTER.items()}

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")


class EpisodeFailure(Exception):
    pass


# --------------------------------------------------------------------------
# Instance generation (private copy; must not depend on editable data/).
# --------------------------------------------------------------------------

def _bell(d: float, width: float = 5.0) -> float:
    return math.exp(-(d * d) / (2.0 * width * width))


def _paper_rewards() -> list[float]:
    return [0.1 + 0.9 * _bell(x - 40) + 0.5 * _bell(x - 75) for x in range(PAPER_N_ARMS)]


def _paper_initial_logits() -> list[float]:
    probs = [math.exp(-0.5 * ((x - 75) / 5.0) ** 2) + 0.1 for x in range(PAPER_N_ARMS)]
    total = sum(probs)
    logits = [math.log(p / total + 1e-8) for p in probs]
    mean = sum(logits) / len(logits)
    return [v - mean for v in logits]


def paper_instance(rng: random.Random) -> dict:
    rewards0 = _paper_rewards()
    logits0 = _paper_initial_logits()
    order = list(range(PAPER_N_ARMS))
    rng.shuffle(order)
    position = {paper_arm: i for i, paper_arm in enumerate(order)}
    rewards = [rewards0[order[i]] for i in range(PAPER_N_ARMS)]
    logits = [logits0[order[i]] for i in range(PAPER_N_ARMS)]
    return {
        "family": "paper",
        "rewards": rewards,
        "initial_logits": logits,
        "optimal_arm": position[40],
        "local_arm": position[75],
        "global_peak": sorted(i for i, r in enumerate(rewards) if r >= GLOBAL_PEAK_THRESHOLD),
        "local_peak": sorted(position[x] for x in range(70, 81)),
    }


def scaled_instance(rng: random.Random, n_arms: int = SCALED_N_ARMS) -> dict:
    offsets = list(range(-PEAK_HALF_WIDTH, PEAK_HALF_WIDTH + 1))
    p_local = rng.uniform(0.45, 0.65)
    p_global = math.exp(rng.uniform(math.log(P_GLOBAL_RANGE[0]), math.log(P_GLOBAL_RANGE[1])))
    p_decoy = rng.uniform(0.15, 0.25)
    p_tail = 1.0 - p_local - p_global - p_decoy

    per_arm = p_global / len(offsets)
    n_decoy = max(1, int(round(p_decoy / per_arm)))
    special = rng.sample(range(n_arms), 2 * len(offsets) + n_decoy)
    local_arms = special[: len(offsets)]
    global_arms = special[len(offsets): 2 * len(offsets)]
    decoys = special[2 * len(offsets):]

    rewards = [0.1] * n_arms
    for d, arm in zip(offsets, local_arms):
        rewards[arm] = 0.1 + 0.5 * _bell(d)
    for d, arm in zip(offsets, global_arms):
        rewards[arm] = 0.1 + 0.9 * _bell(d)

    probs = [0.0] * n_arms
    bell_w = [_bell(d) for d in offsets]
    bell_total = sum(bell_w)
    for w, arm in zip(bell_w, local_arms):
        probs[arm] = p_local * w / bell_total
    for group, mass in ((global_arms, p_global), (decoys, p_decoy)):
        weights = [math.exp(0.3 * rng.gauss(0.0, 1.0)) for _ in group]
        total = sum(weights)
        for w, arm in zip(weights, group):
            probs[arm] = mass * w / total
    taken = set(special)
    tail_w = [0.0] * n_arms
    for arm in range(n_arms):
        if arm not in taken:
            tail_w[arm] = math.exp(rng.gauss(0.0, 1.0))
    tail_total = sum(tail_w)
    for arm in range(n_arms):
        if arm not in taken:
            probs[arm] = p_tail * tail_w[arm] / tail_total
    logits = [math.log(p) for p in probs]
    mean = sum(logits) / n_arms
    logits = [v - mean for v in logits]
    centre = offsets.index(0)
    return {
        "family": "scaled",
        "rewards": rewards,
        "initial_logits": logits,
        "optimal_arm": global_arms[centre],
        "local_arm": local_arms[centre],
        "global_peak": sorted(a for a in global_arms if rewards[a] >= GLOBAL_PEAK_THRESHOLD),
        "local_peak": sorted(
            a for d, a in zip(offsets, local_arms) if abs(d) <= LOCAL_NEIGHBOURHOOD
        ),
    }


def _seed(*parts: object) -> int:
    text = "|".join(str(p) for p in (SLICE_ENTROPY,) + parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:16], "big")


def make_instance(experiment: str, episode: int) -> dict:
    rng = random.Random(_seed(experiment, episode, "instance"))
    family = EXPERIMENTS[experiment]["family"]
    return paper_instance(rng) if family == "paper" else scaled_instance(rng)


# --------------------------------------------------------------------------
# Softmax policy with sparse updates and exact block sums.
# --------------------------------------------------------------------------

class SoftmaxPolicy:
    def __init__(self, logits: list[float]) -> None:
        self.n = len(logits)
        self.block = max(16, math.isqrt(self.n))
        self.n_blocks = (self.n + self.block - 1) // self.block
        self.theta = list(logits)
        self._rebuild()

    def _rebuild(self) -> None:
        self.shift = max(self.theta)
        self.weights = [math.exp(t - self.shift) for t in self.theta]
        b = self.block
        self.block_sums = [sum(self.weights[i * b:(i + 1) * b]) for i in range(self.n_blocks)]

    def sample(self, rng: random.Random, k: int) -> list[int]:
        cumulative = list(accumulate(self.block_sums))
        total = cumulative[-1]
        out = []
        for _ in range(k):
            u = rng.random() * total
            blk = min(bisect_right(cumulative, u), self.n_blocks - 1)
            u -= cumulative[blk] - self.block_sums[blk]
            lo = blk * self.block
            inner = list(accumulate(self.weights[lo:lo + self.block]))
            j = min(bisect_right(inner, u), len(inner) - 1)
            while j > 0 and self.weights[lo + j] == 0.0:
                j -= 1
            out.append(lo + j)
        return out

    def apply(self, deltas: dict[int, float]) -> None:
        rebuild = False
        for arm, delta in deltas.items():
            self.theta[arm] += delta
            if self.theta[arm] - self.shift > 50.0:
                rebuild = True
        if rebuild:
            self._rebuild()
            return
        dirty = set()
        for arm in deltas:
            self.weights[arm] = math.exp(self.theta[arm] - self.shift)
            dirty.add(arm // self.block)
        b = self.block
        for blk in dirty:
            self.block_sums[blk] = sum(self.weights[blk * b:(blk + 1) * b])
        if sum(self.block_sums) < 1e-200:
            self._rebuild()

    def probabilities(self) -> list[float]:
        top = max(self.theta)
        w = [math.exp(t - top) for t in self.theta]
        z = math.fsum(w)
        return [x / z for x in w]


# --------------------------------------------------------------------------
# Agent discovery and protocol.
# --------------------------------------------------------------------------

class Method:
    def __init__(self, key: str, path: Path, variant: str | None, error: str | None = None):
        self.key = key
        self.path = path
        self.variant = variant
        self.error = error


def _agent_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def discover_methods() -> list[Method]:
    methods: list[Method] = []
    used: set[str] = set()
    for folder in AGENT_DIRS:
        directory = ROOT / folder
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.py")):
            if path.name.startswith("_"):
                continue
            names, error = _list_variants(path)
            if error is not None:
                key = path.stem if path.stem not in used else f"{folder}.{path.stem}"
                used.add(key)
                methods.append(Method(key, path, None, error))
                continue
            for name in names:
                key = name if name not in used else f"{path.stem}.{name}"
                used.add(key)
                methods.append(Method(key, path, name))
    return methods


def _list_variants(path: Path) -> tuple[list[str], str | None]:
    try:
        proc = subprocess.run(
            [sys.executable, str(path), "--list"], cwd=ROOT, env=_agent_env(),
            capture_output=True, text=True, timeout=LIST_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return [], "--list timed out"
    if proc.returncode != 0:
        return [], f"--list exited {proc.returncode}: {proc.stderr.strip()[-300:]}"
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    try:
        names = json.loads(lines[-1]) if lines else None
    except json.JSONDecodeError:
        names = None
    if not isinstance(names, list) or not names:
        return [], "--list did not print a non-empty JSON list of names"
    if not all(isinstance(n, str) and NAME_RE.match(n) for n in names):
        return [], "--list returned an invalid method name"
    if len(set(names)) != len(names):
        return [], "--list returned duplicate names"
    return names, None


class AgentProcess:
    def __init__(self, method: Method, log_handle) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, str(method.path), "--serve", method.variant],
            cwd=ROOT, env=_agent_env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=log_handle, bufsize=0,
        )
        self.fd = self.proc.stdout.fileno()
        self.buffer = b""

    def send(self, obj: dict) -> None:
        try:
            self.proc.stdin.write((json.dumps(obj) + "\n").encode())
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise EpisodeFailure(f"agent stopped reading: {exc}") from exc

    def receive(self) -> dict:
        deadline = time.monotonic() + REPLY_TIMEOUT
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EpisodeFailure("agent reply timed out")
            ready, _, _ = select.select([self.fd], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(self.fd, 1 << 16)
            if not chunk:
                raise EpisodeFailure(f"agent exited (code {self.proc.poll()})")
            self.buffer += chunk
            if len(self.buffer) > (8 << 20):
                raise EpisodeFailure("agent reply too long")
        line, self.buffer = self.buffer.split(b"\n", 1)
        try:
            message = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise EpisodeFailure(f"agent wrote a non-JSON line: {line[:80]!r}") from exc
        if not isinstance(message, dict):
            raise EpisodeFailure("agent reply is not a JSON object")
        return message

    def close(self) -> None:
        try:
            self.send({"type": "end"})
        except EpisodeFailure:
            pass
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()


def _check_advantages(message: dict) -> list[float]:
    values = message.get("advantages")
    if not isinstance(values, list) or len(values) != BATCH_SIZE:
        raise EpisodeFailure(f"reply must contain {BATCH_SIZE} advantages")
    out = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise EpisodeFailure("advantage is not a number")
        v = float(v)
        if not math.isfinite(v):
            raise EpisodeFailure("advantage is not finite")
        if abs(v) > ADVANTAGE_CAP:
            raise EpisodeFailure(f"advantage {v} exceeds the cap {ADVANTAGE_CAP}")
        out.append(v)
    return out


def run_episode(method: Method, inst: dict, logits_path: str, experiment: str,
                episode: int, num_batches: int, log_handle) -> dict:
    rewards = inst["rewards"]
    policy = SoftmaxPolicy(inst["initial_logits"])
    rng = random.Random(_seed(experiment, episode, "sampling"))
    agent = AgentProcess(method, log_handle)
    online = 0.0
    recent: list[float] = []
    try:
        agent.send({
            "type": "episode", "n_arms": len(rewards), "batch_size": BATCH_SIZE,
            "num_batches": num_batches, "learning_rate": LEARNING_RATE,
            "advantage_cap": ADVANTAGE_CAP, "initial_logits_file": logits_path,
            "seed": _seed(experiment, episode, "agent") % (2 ** 31),
        })
        if agent.receive().get("type") != "ready":
            raise EpisodeFailure("agent did not answer the episode message with ready")
        for t in range(num_batches):
            actions = policy.sample(rng, BATCH_SIZE)
            observed = [rewards[a] for a in actions]
            online += math.fsum(observed)
            recent.extend(observed)
            if len(recent) > 4000:
                recent = recent[-1000:]
            agent.send({"type": "batch", "index": t, "actions": actions, "rewards": observed})
            adv = _check_advantages(agent.receive())
            mean = math.fsum(adv) / BATCH_SIZE
            deltas: dict[int, float] = {}
            for a, v in zip(actions, adv):
                deltas[a] = deltas.get(a, 0.0) + LEARNING_RATE * (v - mean) / BATCH_SIZE
            policy.apply(deltas)
        agent.close()
    except EpisodeFailure:
        agent.kill()
        raise
    except Exception as exc:  # evaluator-side safety net: never score a broken run
        agent.kill()
        raise EpisodeFailure(f"{type(exc).__name__}: {exc}") from exc

    probs = policy.probabilities()
    tail = recent[-1000:]
    gp = math.fsum(probs[a] for a in inst["global_peak"])
    metrics = {
        "expected_reward": math.fsum(p * r for p, r in zip(probs, rewards)),
        "mean_reward_last_1000": math.fsum(tail) / len(tail),
        "online_mean_reward": online / (num_batches * BATCH_SIZE),
        "optimal_action_probability": probs[inst["optimal_arm"]],
        "global_peak_probability": gp,
        "global_peak_success_rate": 1.0 if gp >= SUCCESS_MASS else 0.0,
        "local_action_probability": probs[inst["local_arm"]],
        "local_peak_probability": math.fsum(probs[a] for a in inst["local_peak"]),
    }
    for name, value in metrics.items():
        if not math.isfinite(value) or value < -1e-9 or value > 1.0 + 1e-6:
            raise EpisodeFailure(f"impossible metric value {name}={value}")
    return metrics


# --------------------------------------------------------------------------
# Main loop.
# --------------------------------------------------------------------------

def _write_json(path: Path, data: dict, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=indent))
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--methods", default=None,
                        help="Comma-separated subset of method keys (development only).")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Cap episodes per experiment (development only).")
    parser.add_argument("--num-batches", type=int, default=NUM_BATCHES,
                        help="Override the interaction budget (development only).")
    args = parser.parse_args()

    # A stale or forged score file must never survive a run of the evaluator.
    if args.output.exists():
        args.output.unlink()

    methods = discover_methods()
    if args.methods:
        wanted = set(args.methods.split(","))
        methods = [m for m in methods if m.key in wanted]
    if not methods:
        raise SystemExit("no agent files found in method/ or baseline/")

    log_dir = ROOT / "logs" / f"agents_{SLICE}"
    log_dir.mkdir(parents=True, exist_ok=True)
    logs = {m.key: open(log_dir / f"{m.key}.log", "wb") for m in methods}
    failures = {m.key: f"discovery: {m.error}" for m in methods if m.error}
    episodes: dict[str, dict[str, list[dict]]] = {m.key: {e: [] for e in EXPERIMENTS} for m in methods}
    counts: dict[str, int] = {}
    started = time.time()

    with tempfile.TemporaryDirectory(prefix=f"bandit_{SLICE}_") as tmp:
        for experiment, spec in EXPERIMENTS.items():
            count = spec["episodes"] if args.max_episodes is None else min(spec["episodes"], args.max_episodes)
            counts[experiment] = count
            for episode in range(count):
                inst = make_instance(experiment, episode)
                logits_path = os.path.join(tmp, f"{experiment}_{episode}.f64")
                with open(logits_path, "wb") as handle:
                    array("d", inst["initial_logits"]).tofile(handle)
                for method in methods:
                    if method.key in failures:
                        continue
                    try:
                        metrics = run_episode(method, inst, logits_path, experiment, episode,
                                              args.num_batches, logs[method.key])
                    except EpisodeFailure as exc:
                        failures[method.key] = f"{experiment} episode {episode}: {exc}"
                        print(f"[{SLICE}] FAILED {method.key}: {failures[method.key]}", flush=True)
                        continue
                    episodes[method.key][experiment].append(metrics)
                os.unlink(logits_path)
                print(f"[{SLICE}] {experiment} episode {episode + 1}/{count} done "
                      f"({time.time() - started:.0f}s)", flush=True)
    for handle in logs.values():
        handle.close()

    scores: dict = {"experiments": {}}
    for experiment in EXPERIMENTS:
        results = {}
        for method in methods:
            runs = episodes[method.key][experiment]
            if method.key in failures or not runs:
                results[method.key] = dict(WORST)
            else:
                results[method.key] = {
                    name: math.fsum(r[name] for r in runs) / len(runs) for name in HIGHER_IS_BETTER
                }
        scores["experiments"][experiment] = {
            "_slice": SLICE,
            "_episodes": counts.get(experiment, 0),
            "results": results,
        }
    scores["_failures"] = failures
    scores["_protocol"] = {
        "num_batches": args.num_batches, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "advantage_cap": ADVANTAGE_CAP,
        "partial_run": bool(args.methods or args.max_episodes or args.num_batches != NUM_BATCHES),
    }
    _write_json(args.output, scores)
    _write_json(args.details, {"slice": SLICE, "failures": failures, "episodes": episodes}, None)
    print(f"Wrote {args.output} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
