"""Client side of the bandit evaluators' agent protocol (method-agnostic).

An *agent file* is any ``*.py`` file directly inside ``method/`` or
``baseline/`` whose name does not start with ``_``. The evaluators discover
every such file and score every variant it lists; a file that does not answer
``--list`` is scored as a failed method. Put helper modules in a subpackage or
give them a leading underscore.

The evaluator owns the bandit, the sampling of actions and the policy update.
An agent only decides the *advantage* of every sample in a group, from the
group's rewards (as in RLVR, it is never told which arm produced a reward).
Every group's rewards are shifted by one hidden offset, drawn uniformly from
[-2, 2] per group, so rewards are comparable only within a group::

    evaluator -> agent   {"type": "episode", "n_arms": K, "batch_size": N,
                          "num_batches": T, "learning_rate": lr,
                          "advantage_cap": c, "seed": s}
    agent -> evaluator   {"type": "ready"}
    repeated T times:
    evaluator -> agent   {"type": "batch", "index": t, "rewards": [N floats]}
    agent -> evaluator   {"advantages": [N finite floats, each |A| <= c]}
    evaluator -> agent   {"type": "end"}

The i-th advantage belongs to the i-th reward. After each reply the evaluator
centres the advantages within the group and applies the softmax
policy-gradient step ``theta[a_i] += lr / N * (A_i - mean(A))`` to the arms
that were sampled; the next group is drawn from the updated policy. Each
message is one line of JSON. Write logs to stderr: anything else on stdout
breaks the protocol and fails the run. One process serves one episode.

Usage::

    from data.agent_protocol import Agent, main

    class MyAgent(Agent):
        def advantages(self, rewards):
            mean = sum(rewards) / len(rewards)
            return [r - mean for r in rewards]

    if __name__ == "__main__":
        main({"my_method": MyAgent})
"""

from __future__ import annotations

import json
import sys
from typing import Callable, Mapping


class EpisodeInfo:
    """Episode settings sent by the evaluator."""

    def __init__(self, message: dict) -> None:
        self.n_arms = int(message["n_arms"])
        self.batch_size = int(message["batch_size"])
        self.num_batches = int(message["num_batches"])
        self.learning_rate = float(message["learning_rate"])
        self.advantage_cap = float(message["advantage_cap"])
        self.seed = int(message["seed"])


class Agent:
    """Base class: override ``advantages`` (and optionally ``start``)."""

    def start(self, info: EpisodeInfo) -> None:
        self.info = info

    def advantages(self, rewards: list[float]) -> list[float]:
        raise NotImplementedError


def _send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def serve(factory: Callable[[], Agent]) -> None:
    agent: Agent | None = None
    for line in sys.stdin:
        if not line.strip():
            continue
        message = json.loads(line)
        kind = message.get("type")
        if kind == "episode":
            agent = factory()
            agent.start(EpisodeInfo(message))
            _send({"type": "ready"})
        elif kind == "batch":
            if agent is None:
                raise RuntimeError("batch received before episode")
            values = agent.advantages(list(message["rewards"]))
            _send({"advantages": [float(v) for v in values]})
        elif kind == "end":
            break


def main(variants: Mapping[str, Callable[[], Agent]], argv: list[str] | None = None) -> None:
    """Entry point for agent files: ``--list`` or ``--serve NAME``."""
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["--list"]:
        print(json.dumps(sorted(variants)))
        return
    if len(args) == 2 and args[0] == "--serve":
        if args[1] not in variants:
            raise SystemExit(f"unknown variant {args[1]!r}")
        serve(variants[args[1]])
        return
    raise SystemExit("usage: <agent>.py --list | --serve NAME")
