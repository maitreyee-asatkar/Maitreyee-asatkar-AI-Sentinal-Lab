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
