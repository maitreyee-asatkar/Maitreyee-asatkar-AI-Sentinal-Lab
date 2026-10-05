from pathlib import Path
import json

def capture_layer_activation(model_name: str, prompt: str, layer: int, output: str = "artifacts/nnsight_activation.json"):
    """Capture a transformer block activation with NNsight."""
    try:
        from nnsight import TransformersModel
    except ImportError as exc:
        raise RuntimeError("Install nnsight before running this probe.") from exc

    model = TransformersModel(model_name, task="text-generation", device_map="auto", dispatch=True)

    with model.trace(prompt):
        hidden = model.transformer.h[layer].output.save()

    tensor = hidden[0] if isinstance(hidden, tuple) else hidden
    result = {
        "model_name": model_name,
        "layer": layer,
        "prompt": prompt,
        "shape": list(tensor.shape),
        "mean": float(tensor.float().mean().item()),
        "std": float(tensor.float().std().item()),
    }

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def steer_last_token(model_name: str, prompt: str, layer: int, direction, coefficient: float = 1.0):
    """Apply a temporary last-token activation intervention with NNsight."""
    try:
        from nnsight import TransformersModel
    except ImportError as exc:
        raise RuntimeError("Install nnsight before running this probe.") from exc

    model = TransformersModel(model_name, task="text-generation", device_map="auto", dispatch=True)

    with model.trace(prompt):
        hidden = model.transformer.h[layer].output
        hidden[:, -1, :] = hidden[:, -1, :] + coefficient * direction
        output = model.output.save()

    return output
