# Interpretability Experiment 03 — Activation Steering

## Objective
Test whether a candidate internal representation is causally related to security-relevant behavior.

## Hypothesis
If adversarial prompts produce a reproducible activation direction, moving a model's internal representation along the opposite direction may reduce attack success.

## Experimental design
1. Collect clean and adversarial activations.
2. Compute a normalized direction: clean activation minus adversarial activation.
3. Run baseline inference.
4. Run controlled interventions using several coefficients.
5. Measure attack success rate, refusal rate, normal-task success, output consistency, and intervention magnitude.
6. Repeat across multiple prompts and held-out prompts.

## Interpretation
A single successful intervention is not enough to establish causality. Evidence is stronger when the effect replicates across prompts, survives coefficient sweeps, persists on held-out prompts, and does not substantially damage benign task performance.

## Safety
This repository is intended for authorized AI security research. The steering module is an experimental analysis component, not a production model modification system.
