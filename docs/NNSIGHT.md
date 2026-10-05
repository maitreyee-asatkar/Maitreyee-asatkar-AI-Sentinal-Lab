# NNsight Integration

NNsight provides a model-agnostic tracing and intervention interface for PyTorch transformer models.

The project uses it for two purposes:

1. Capture internal layer activations.
2. Apply temporary activation interventions.

The implementation follows the current NNsight tracing pattern: values that need to survive the trace are explicitly saved, and activation edits are applied inside the trace context.

This creates a second interpretability backend alongside TransformerLens.

## Why both?

TransformerLens is useful for circuit-level analysis such as attention patterns and residual-stream inspection. NNsight provides a flexible intervention layer that works directly with wrapped PyTorch/Transformers models.

The two backends should produce comparable experiment artifacts rather than silently mixing incompatible activation definitions.
