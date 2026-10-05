from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class SAEConfig:
    input_dim: int
    feature_dim: int
    l1_coefficient: float = 1e-3


def encode(activation, encoder_weight, encoder_bias):
    pre = activation @ encoder_weight.T + encoder_bias
    return pre.relu()


def decode(features, decoder_weight, decoder_bias):
    return features @ decoder_weight.T + decoder_bias


def reconstruction_metrics(activation, reconstruction):
    error = activation - reconstruction
    mse = float((error.float() ** 2).mean().item())
    l0 = None
    return {"mse": mse, "mean_absolute_error": float(error.float().abs().mean().item()), "l0": l0}


def save_sae_result(result: dict, output: str = "artifacts/sae_experiment.json"):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
