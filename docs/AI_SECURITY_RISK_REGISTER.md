# AI Security Risk Register

AI Sentinel Lab uses this register to connect technical experiments with security governance. It is a practical project mapping, not a certification claim.

| Risk area | Lab control / evidence | Measurement | Framework alignment |
|---|---|---|---|
| Prompt injection | Clean/adversarial corpus, security regression, causal interpretability experiments | evaluator findings, KL divergence, restoration fraction | OWASP LLM Prompt Injection; NIST AI RMF MAP/MEASURE/MANAGE |
| Sensitive information disclosure | Injection prompts request hidden/system information; evaluator flags disclosure-oriented patterns | regression result and response assessment | OWASP LLM Sensitive Information Disclosure; NIST GAI Profile |
| Model behavior manipulation | Residual probes, attention analysis, NNsight activation inspection | activation delta, cosine similarity, attention changes | NIST AI RMF MEASURE |
| Unsafe intervention / degraded utility | Activation steering is evaluated on adversarial and clean prompts | security restoration and clean utility drift | NIST AI RMF MEASURE/MANAGE |
| ML supply chain | Dependencies are pinned/ranged in dedicated requirements files and exercised in CI | successful reproducible workflow | OWASP LLM Supply Chain; NIST AI RMF GOVERN/MANAGE |
| Regression risk | GitHub Actions runs security and research workflows | test status and retained JSON artifacts | NIST AI RMF GOVERN/MEASURE |

## Evidence model

The project separates three evidence levels:

1. **Descriptive evidence** — activation/attention differences identify candidates.
2. **Causal evidence** — controlled head ablation or activation steering changes model output under a specified intervention.
3. **Generalization evidence** — replication across multiple prompt pairs. Broader model-level claims require additional models and datasets.

## Governance notes

- Experiments use benign synthetic security prompts and are intended for authorized defensive evaluation.
- Raw experimental results are retained as workflow artifacts for reproducibility.
- A positive result is not treated as proof of a universal mechanism.
- Utility is measured alongside security so a mitigation is not considered useful merely because it changes model behavior.

## References

- NIST AI Risk Management Framework 1.0 and NIST AI 600-1 Generative AI Profile.
- OWASP Top 10 for LLM Applications / GenAI Security Project.
