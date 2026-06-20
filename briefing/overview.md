# Overview

- **Paper ID:** 2509.24261
- **Title:** Risk-Sensitive RL for Alleviating Exploration Dilemmas in Large Language Models
- **Domain:** Reinforcement learning for language-model reasoning
- **TL;DR:** The paper modifies GRPO's advantage calculation with a risk-seeking objective so RL fine-tuning explores low-probability high-reward solutions instead of only sharpening existing modes.

## Short Summary

The paper argues that RLVR for reasoning models often improves pass@1 by
collapsing onto already likely solutions, which can hurt pass@k and diversity.
It proposes RS-GRPO, a drop-in replacement for GRPO's advantage based on an
exponential-utility objective. The authors show theoretically and in a 100-arm
bandit that the risk-sensitive update can escape a local optimum that traps
risk-neutral policy gradient. Large-scale math experiments report improved
pass@32 with comparable pass@1.

## Key Results

- In the paper's bandit experiment, risk-neutral training stays near reward 0.6,
  while sufficiently risk-seeking settings converge to reward 1.0.
- On Qwen2.5-Math-1.5B with deepmath103k, the reported average pass@32 improves
  from 37.5 for GRPO to 42.0 for RS-GRPO, with similar pass@1.
- The recommended comparison setting uses `beta=2` for pass@k baselines and
  `beta=8` in the broader pass@k curves.
