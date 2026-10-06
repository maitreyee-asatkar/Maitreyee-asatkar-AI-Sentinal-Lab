"""Activation-steering experiment with security/utility trade-off measurement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformer_lens.model_bridge import TransformerBridge

from interpretability.head_ablation import PROMPT_PAIRS, final_distribution, kl_divergence


def pooled_residual(model, prompt: str, layer: int) -> torch.Tensor:
    _, cache = model.run_with_cache(prompt, names_filter=[f"blocks.{layer}.hook_out"])
    activation = cache[f"blocks.{layer}.hook_out"].float()
    return activation.mean(dim=1, keepdim=True)


def build_direction(model, layer: int) -> torch.Tensor:
    directions = []
    for pair in PROMPT_PAIRS:
        clean = pooled_residual(model, pair["clean"], layer)
        adversarial = pooled_residual(model, pair["adversarial"], layer)
        delta = clean - adversarial
        directions.append(delta / delta.norm().clamp_min(1e-8))
    direction = torch.stack(directions).mean(dim=0)
    return direction / direction.norm().clamp_min(1e-8)


def steered_logits(model, prompt: str, layer: int, direction: torch.Tensor, coefficient: float):
    def steer(residual, hook):
        return residual + coefficient * direction.to(residual.device, residual.dtype)

    return model.run_with_hooks(
        prompt,
        fwd_hooks=[(f"blocks.{layer}.hook_out", steer)],
    )


def evaluate(model, layer: int, direction: torch.Tensor, coefficients):
    rows = []
    for pair in PROMPT_PAIRS:
        clean_p = final_distribution(model(pair["clean"]))
        adversarial_p = final_distribution(model(pair["adversarial"]))
        attack_gap = kl_divergence(adversarial_p, clean_p)

        for coefficient in coefficients:
            steered_adv = final_distribution(
                steered_logits(model, pair["adversarial"], layer, direction, coefficient)
            )
            steered_clean = final_distribution(
                steered_logits(model, pair["clean"], layer, direction, coefficient)
            )
            steered_gap = kl_divergence(steered_adv, clean_p)
            restoration = (
                (attack_gap - steered_gap) / attack_gap if attack_gap else 0.0
            )
            utility_drift = kl_divergence(steered_clean, clean_p)
            rows.append(
                {
                    "prompt_pair_id": pair["id"],
                    "coefficient": coefficient,
                    "baseline_attack_gap_kl": attack_gap,
                    "steered_adversarial_vs_clean_kl": steered_gap,
                    "security_restoration_fraction": restoration,
                    "clean_utility_drift_kl": utility_drift,
                }
            )
    return rows


def summarize(rows, coefficients):
    summary = []
    for coefficient in coefficients:
        subset = [row for row in rows if row["coefficient"] == coefficient]
        mean_restoration = sum(r["security_restoration_fraction"] for r in subset) / len(subset)
        mean_utility_drift = sum(r["clean_utility_drift_kl"] for r in subset) / len(subset)
        summary.append(
            {
                "coefficient": coefficient,
                "mean_security_restoration_fraction": mean_restoration,
                "positive_restoration_pairs": sum(
                    r["security_restoration_fraction"] > 0 for r in subset
                ),
                "pair_count": len(subset),
                "mean_clean_utility_drift_kl": mean_utility_drift,
                "tradeoff_score": mean_restoration / (1.0 + mean_utility_drift),
            }
        )
    return sorted(summary, key=lambda row: row["tradeoff_score"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--coefficients", default="-2,-1,-0.5,0.5,1,2")
    parser.add_argument("--output", default="artifacts/activation_steering.json")
    args = parser.parse_args()

    coefficients = [float(value) for value in args.coefficients.split(",")]
    model = TransformerBridge.boot_transformers(args.model)
    model.enable_compatibility_mode()

    direction = build_direction(model, args.layer)
    rows = evaluate(model, args.layer, direction, coefficients)
    result = {
        "model_name": args.model,
        "layer": args.layer,
        "prompt_pair_count": len(PROMPT_PAIRS),
        "direction_source": "mean normalized clean-minus-adversarial residual direction",
        "aggregate_ranking": summarize(rows, coefficients),
        "pair_results": rows,
        "interpretation": (
            "Security restoration measures movement of adversarial next-token output "
            "toward its paired clean distribution. Utility drift measures how much "
            "the same intervention changes clean output. Results are specific to "
            "this model, prompt set, layer, direction construction, and coefficients."
        ),
    }

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
