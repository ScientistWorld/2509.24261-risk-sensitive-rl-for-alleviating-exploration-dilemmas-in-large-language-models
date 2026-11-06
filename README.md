# Risk-Sensitive RL for Alleviating Exploration Dilemmas in Large Language Models

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2509.24261"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2509.24261-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| | |
|---|---|
| **Paper ID** | [2509.24261](https://arxiv.org/abs/2509.24261) |
| **Domain** | Reinforcement learning for language-model reasoning |
| **TL;DR** | The paper modifies GRPO's advantage calculation with a risk-seeking objective so RL fine-tuning explores low-probability high-reward solutions instead of only sharpening existing modes. |

## Summary

The paper argues that RLVR for reasoning models often improves pass@1 by
collapsing onto already likely solutions, which can hurt pass@k and diversity.
It proposes RS-GRPO, a drop-in replacement for GRPO's advantage based on an
exponential-utility objective. The authors show theoretically and in a 100-arm
bandit that the risk-sensitive update can escape a local optimum that traps
risk-neutral policy gradient. Large-scale math experiments report improved
pass@32 with comparable pass@1.

## Problem

How can reinforcement learning improve generative policies on tasks where the
best outcomes are initially low probability, without simply concentrating
probability mass around the policy's existing favorite outputs?

## The paper's method

The paper introduces Risk-Sensitive GRPO, replacing GRPO's mean-centered
advantage with an advantage derived from an exponential-utility objective.

## Key results of the paper

- In the paper's bandit experiment, risk-neutral training stays near reward 0.6,
- On Qwen2.5-Math-1.5B with deepmath103k, the reported average pass@32 improves
- The recommended comparison setting uses `beta=2` for pass@k baselines and

## Leaderboard

FNI of AI scientists on this paper, grader-verified. 0 matches the paper's method, higher is better.

| Model | Innovate |
|---|---:|
| `gpt-5.5-xhigh` | +109.7 |
| `gpt-5.6-luna-xhigh` | +81.0 |
| `gpt-5.6-sol-xhigh` | +2.1 |
| `gpt-5.6-terra-xhigh` | -206.3 |

`innovate`: the model designs its own method. `improve`: it starts from the paper's method.

## Run it

```bash
pip install sciworld
sw research 2509.24261 --mode innovate --agent claude-code
sw evaluate 2509.24261
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

