# Experiment Overview

## bandit_escape

Tests whether a policy optimizer can overcome a misleading initial preference
for a local reward peak in a fixed 100-action decision problem. The experiment
is intended to distinguish genuine exploration of better low-probability
actions from methods that only reinforce the initially likely region.

The default reproduction runs multiple independent stochastic seeds for each
risk setting and aggregates the final policies. This checks that the threshold
behavior is robust rather than a lucky trajectory from one random seed.
