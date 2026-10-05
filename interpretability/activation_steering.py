from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class SteeringConfig:
    layer: int
    coefficient: float
    direction_source: str = 'clean_minus_adversarial'

def build_direction(clean_activation, adversarial_activation):
    direction = clean_activation - adversarial_activation
    norm = direction.float().norm()
    if norm.item() == 0:
        raise ValueError('Cannot build a steering direction from identical activations.')
    return direction / norm

def apply_direction(activation, direction, coefficient: float):
    return activation + (coefficient * direction)

def save_experiment(result: dict, output: str = 'artifacts/activation_steering.json'):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')

def experiment_schema(model_name, layer, coefficients, baseline, steered):
    return {
        'model_name': model_name,
        'layer': layer,
        'intervention': {
            'type': 'additive_activation_steering',
            'coefficients': coefficients,
            'direction_source': 'clean_minus_adversarial',
        },
        'baseline': baseline,
        'steered_runs': steered,
        'interpretation': 'A change in attack success after intervention is evidence for a candidate causal mechanism, but should be replicated across prompts.',
    }
