# Security Attribution Layer

This module connects attention analysis to security investigation.

## Workflow

1. Run the residual-stream probe.
2. Rank attention heads by behavioral-condition delta.
3. Select a high-signal layer/head.
4. Export token-level attention.
5. Compare clean and adversarial token patterns.
6. Use the selected component as a candidate for causal intervention.

## Interpretation

A head that changes strongly during an attack is a **candidate mechanism**, not a proven cause.

The next step is activation steering/intervention. If modifying a candidate activation reliably changes attack success while preserving normal task performance, the evidence becomes substantially stronger.
