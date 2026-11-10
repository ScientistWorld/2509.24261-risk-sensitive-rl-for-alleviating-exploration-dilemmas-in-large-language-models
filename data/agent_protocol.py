"""Client side of the bandit evaluators' agent protocol (method-agnostic).

An *agent file* is any ``*.py`` file directly inside ``method/`` or
``baseline/`` whose name does not start with ``_``. The evaluators discover
every such file and score every variant it lists. Put helper modules in a
subpackage or give them a leading underscore.

The evaluator owns the bandit, the sampling of actions and the policy update.
An agent only decides the *advantage* of every sampled action::

    evaluator -> agent   {"type": "episode", "n_arms": K, "batch_size": N,
                          "num_batches": T, "learning_rate": lr,
                          "advantage_cap": c, "initial_logits_file": path,
                          "seed": s}
    agent -> evaluator   {"type": "ready"}
    repeated T times:
    evaluator -> agent   {"type": "batch", "index": t,
                          "actions": [N ints], "rewards": [N floats]}
    agent -> evaluator   {"advantages": [N finite floats, each |A| <= c]}
    evaluator -> agent   {"type": "end"}

After each reply the evaluator centres the advantages within the batch and
applies the softmax policy-gradient step
``theta[a_n] += lr / N * (A_n - mean(A))`` to the arms that were sampled; the
next batch is drawn from the updated policy. ``initial_logits_file`` holds the
K initial logits as native-endian float64 (``array('d').fromfile``). Each
message is one line of JSON. Write logs to stderr: anything else on stdout
breaks the protocol and fails the run.

Usage::

    from data.agent_protocol import Agent, main

    class MyAgent(Agent):
        def advantages(self, actions, rewards):
            mean = sum(rewards) / len(rewards)
            return [r - mean for r in rewards]

    if __name__ == "__main__":
        main({"my_method": MyAgent})
"""

from __future__ import annotations

import json
import sys
from array import array
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
        self._logits_file = message["initial_logits_file"]
        self._logits: array | None = None

    @property
    def initial_logits(self) -> array:
        """The fixed initial logits (loaded on first access)."""
        if self._logits is None:
            values = array("d")
            with open(self._logits_file, "rb") as handle:
                values.fromfile(handle, self.n_arms)
            self._logits = values
        return self._logits


class Agent:
    """Base class: override ``advantages`` (and optionally ``start``)."""

    def start(self, info: EpisodeInfo) -> None:
        self.info = info

    def advantages(self, actions: list[int], rewards: list[float]) -> list[float]:
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
            values = agent.advantages(list(message["actions"]), list(message["rewards"]))
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
