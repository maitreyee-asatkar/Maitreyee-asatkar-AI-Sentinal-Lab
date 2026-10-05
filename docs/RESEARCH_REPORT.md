# AI Sentinel Lab — Interpretability Research Report

## Research question
Do adversarial instructions produce reproducible changes in transformer internal representations that can be localized to attention heads and sparse features?

## Evidence chain
1. Behavioral security evaluation identifies attack cases.
2. Residual-stream analysis measures layer-level activation differences.
3. Attention analysis ranks heads by change between conditions.
4. Token-level export identifies attention patterns for selected heads.
5. NNsight captures internal activations and supports controlled intervention.
6. SAE analysis decomposes activations into sparse candidate features.
7. Activation steering tests whether candidate representations have a causal relationship with security behavior.

## Standards for a credible result
Replicate across multiple prompts and evaluate held-out examples. Report both security outcomes and benign-task performance.
The project distinguishes correlation, localization, and causal intervention rather than presenting attention or feature correlation as proof of causality.
