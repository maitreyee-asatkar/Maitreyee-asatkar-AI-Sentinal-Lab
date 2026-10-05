# Phase 2 — Real Model Security Evaluation

The project now supports a lazy-loaded Hugging Face/PyTorch model adapter.

For each attack case:
1. Send the attack prompt to a local model.
2. Capture the response.
3. Evaluate prompt and response.
4. Record model ID and findings.
5. Compare results across model versions.

The interpretability layer will capture hidden-state statistics for clean versus adversarial prompts before moving to attention, residual-stream, SAE, and activation-steering experiments.

The current string-based evaluator is a regression harness, not a complete model safety classifier.
