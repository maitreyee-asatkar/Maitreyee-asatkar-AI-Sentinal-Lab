# Interpretability Experiment 04 — Sparse Autoencoder

## Objective

Decompose transformer activations into a sparse set of learned features and investigate whether particular features correlate with adversarial prompts.

## Pipeline

1. Collect clean and adversarial activations from the same model layer.
2. Train or load an SAE for that activation space.
3. Encode activations into sparse feature vectors.
4. Compare feature activation rates between clean and adversarial examples.
5. Rank features by effect size.
6. Inspect the highest-signal features.
7. Optionally test candidate features with activation steering.

## Metrics

- reconstruction MSE
- mean absolute reconstruction error
- feature sparsity
- clean feature activation rate
- adversarial feature activation rate
- clean/adversarial effect size

## Important limitation

A feature that correlates with an attack is not automatically a human-interpretable or causal concept. Feature labels require additional inspection, and causal claims require intervention experiments.

## Security research value

This creates an evidence chain from an externally observable attack to internal model representations, then to candidate sparse features that can be tested causally.
