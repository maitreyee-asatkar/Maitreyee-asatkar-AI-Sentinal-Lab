# Model Security Benchmark

The benchmark runner turns the static security corpus into a repeatable model evaluation.

## Run

Install optional model dependencies:

    pip install -r requirements-model.txt

Select a local Hugging Face model:

    export AI_SENTINEL_MODEL=distilgpt2

Then:

    python -m evaluator.benchmark

Results are written to artifacts/model_benchmark.json.

## Metrics to add next

- attack success rate
- refusal rate
- false-positive rate
- severity-weighted risk score
- response leakage rate
- clean-vs-adversarial behavioral delta

The benchmark separates attack generation from model assessment, so models can be swapped without rewriting the security corpus.
