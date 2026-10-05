# Running the first interpretability experiment

## Local execution
Install the base dependencies, then install `requirements-interpretability.txt`.

Run:

`python -m interpretability.run_research --model gpt2`

The command compares a clean prompt with an adversarial prompt and writes `artifacts/residual_probe.json`.

## Output
The artifact contains per-layer pooled residual statistics:
- mean delta
- mean absolute delta
- L2 delta

The highest mean-absolute-delta layers become candidates for attention-head and activation-steering follow-up.

## Reproducibility
Keep the model identifier, prompts, package versions, and output artifact together when reporting results. Results should be rerun on held-out prompt pairs before being presented as general findings.
