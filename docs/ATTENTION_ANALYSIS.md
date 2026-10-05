# Interpretability Experiment 02 — Attention Analysis

## Objective

Identify transformer attention heads whose behavior changes most between a clean security prompt and a prompt-injection variant.

## Measurement

For each layer/head pair we calculate:

- mean attention mass for the clean prompt
- mean attention mass for the adversarial prompt
- absolute difference

The results are ranked to identify candidate heads for deeper investigation.

## Follow-up

For the highest-delta heads, the token-level attention matrix can be exported. This allows us to inspect which tokens receive attention and whether security-relevant instruction tokens become unusually influential.

## Important limitation

Mean attention change is a screening metric, not proof that a head causes the security behavior. Later activation-steering experiments are required for causal testing.
