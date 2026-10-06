# AI Sentinel Lab

AI Sentinel Lab is a local-first AI/LLM security testing platform for red teaming, threat modeling, security regression testing, and model interpretability research.

## What this project demonstrates

- Prompt injection and jailbreak testing
- AI security regression testing
- OWASP LLM Top 10 mapping
- Python + FastAPI
- Hugging Face + PyTorch model integration
- Interpretability experiment scaffolding
- Dockerized security tooling
- GitHub Actions CI
- AI threat modeling

## Architecture

```
Attack Corpus
     |
     v
Security Evaluator ---> Hugging Face / PyTorch Model
     |                         |
     |                         v
     |                  Generated Response
     |                         |
     +-------------------------+
     |
     v
Risk + OWASP Finding
     |
     v
Security Regression Record
```

## Run the MVP

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

Run tests:

```bash
pytest -q
```

## Optional real-model stack

```bash
pip install -r requirements-model.txt
export AI_SENTINEL_MODEL=distilgpt2
```

The model adapter is lazy-loaded so the MVP security harness works without downloading a model.

## Interpretability research

The research workflow runs four complementary experiments against GPT-2 by default:

1. **Residual-stream comparison** — measures how clean vs. adversarial activations diverge across transformer layers.
2. **Attention-head analysis** — ranks heads using descriptive attention changes; this does not establish causality.
3. **NNsight activation analysis** — compares layer-10 activations with sequence-length-safe pooling; this is descriptive.
4. **Causal attention-head ablation** — zeroes candidate heads at `blocks.{layer}.attn.hook_z` and measures whether the adversarial output distribution moves toward the paired clean distribution.\n5. **Activation steering** — constructs a clean-minus-adversarial residual direction and tests security restoration against clean-utility drift across multiple coefficients.

The causal experiment now uses **five matched clean/adversarial prompt pairs** and reports both per-pair measurements and an aggregate ranking. A positive restoration fraction means the tested ablation moved the adversarial output distribution toward the clean distribution for that pair.

### Reproduce the research locally

```bash
pip install -r requirements-model.txt
pip install -r requirements-interpretability.txt

python -m interpretability.run_research --model gpt2 --output artifacts/residual_probe.json
python -m interpretability.attention_heads --model gpt2 --output artifacts/attention_heads.json --layers 8 9 10 11
python -m interpretability.nnsight_activation --model gpt2 --layer 10 --output artifacts/nnsight_activation.json
python -m interpretability.head_ablation --model gpt2 --output artifacts/head_ablation.json\npython -m interpretability.activation_steering --model gpt2 --layer 10 --output artifacts/activation_steering.json\npython -m interpretability.dashboard
```

Results from GitHub Actions are published as the `interpretability-results` workflow artifact, including a generated `research_dashboard.md`. See `docs/AI_SECURITY_RISK_REGISTER.md` for the project risk/evidence mapping to NIST AI RMF / NIST AI 600-1 and OWASP LLM security guidance.

> **Scientific scope:** a positive causal effect on these prompt pairs is evidence for the tested intervention, not proof that a particular head is a universal jailbreak mechanism. Replication across prompts, models, and task types is required before making a broader claim.

## CI security gates

GitHub Actions runs the security regression suite on repository changes. The research workflow separately executes the interpretability experiments and preserves JSON results as downloadable artifacts.

## Roadmap

1. Real Hugging Face/PyTorch model evaluation
2. RAG indirect-injection testing
3. TransformerLens / NNsight activation inspection
4. Attention and residual-stream analysis
5. Sparse Autoencoder feature experiments
6. Activation steering experiments
7. Data-poisoning and inference-attack labs
8. Streamlit AI-security dashboard
9. NIST AI RMF risk register
10. Cloud deployment reference architecture

## Safety

This project is intended for authorized security testing of AI systems. The initial corpus uses benign security-evaluation prompts and does not provide instructions for real-world abuse.
