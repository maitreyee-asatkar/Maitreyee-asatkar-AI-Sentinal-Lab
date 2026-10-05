# AI Security Threat Model

## Assets
- Model behavior and integrity
- System/developer instructions
- User and retrieved data
- Secrets and application configuration
- Tool/API access
- Security evaluation results

## Trust boundaries
1. Untrusted user input -> application
2. Retrieved external content -> model context
3. Model -> tools and downstream APIs
4. Training/fine-tuning data -> model artifacts

## Initial threats

| Threat | Stage | Impact | Control |
|---|---|---|---|
| Prompt injection | Inference | Unauthorized behavior | Instruction isolation, input validation |
| Indirect injection | RAG | Context hijacking | Content provenance, delimiters, tool policy |
| Data leakage | Inference | Confidentiality loss | Output controls, secret scanning |
| Data poisoning | Training | Integrity loss | Dataset provenance and validation |
| Model evasion | Inference | Security control bypass | Robust evaluation and adversarial testing |

The MVP maps findings to OWASP LLM Top 10 identifiers. NIST AI RMF mapping is planned.
