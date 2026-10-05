from pathlib import Path
import json

def run_probe(model_name: str, clean_prompt: str, adversarial_prompt: str,
              output: str = "artifacts/activation_probe.json"):
    try:
        from transformer_lens import HookedTransformer
    except ImportError as exc:
        raise RuntimeError("Install requirements-interpretability.txt first.") from exc

    model = HookedTransformer.from_pretrained(model_name)
    _, clean_cache = model.run_with_cache(clean_prompt)
    _, adv_cache = model.run_with_cache(adversarial_prompt)

    layers = []
    for layer in range(model.cfg.n_layers):
        clean = clean_cache[f"blocks.{layer}.hook_resid_post"]
        adv = adv_cache[f"blocks.{layer}.hook_resid_post"]
        delta = (adv.float() - clean.float()).detach()
        layers.append({
            "layer": layer,
            "clean_shape": list(clean.shape),
            "adversarial_shape": list(adv.shape),
            "clean_mean": float(clean.float().mean().item()),
            "adversarial_mean": float(adv.float().mean().item()),
            "delta_mean": float(delta.mean().item()),
            "delta_abs_mean": float(delta.abs().mean().item()),
        })

    result = {
        "model_name": model_name,
        "experiment": "clean_vs_adversarial_residual_stream",
        "clean_prompt": clean_prompt,
        "adversarial_prompt": adversarial_prompt,
        "layers": layers,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
