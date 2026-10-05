from pathlib import Path
import json

def run_attention_probe(
    model_name: str,
    clean_prompt: str,
    adversarial_prompt: str,
    output: str = "artifacts/attention_probe.json",
):
    """Compare attention patterns between clean and adversarial prompts.

    The two prompts may have different token counts, so attention summaries
    are compared at the head level using mean attention mass.
    """
    try:
        from transformer_lens import HookedTransformer
    except ImportError as exc:
        raise RuntimeError("Install requirements-interpretability.txt first.") from exc

    model = HookedTransformer.from_pretrained(model_name)

    _, clean_cache = model.run_with_cache(clean_prompt)
    _, adv_cache = model.run_with_cache(adversarial_prompt)

    heads = []
    for layer in range(model.cfg.n_layers):
        clean = clean_cache[f"blocks.{layer}.attn.hook_pattern"].float()
        adv = adv_cache[f"blocks.{layer}.attn.hook_pattern"].float()

        clean_head_mean = clean.mean(dim=(-1, -2))
        adv_head_mean = adv.mean(dim=(-1, -2))
        delta = (adv_head_mean - clean_head_mean).abs()

        for head in range(model.cfg.n_heads):
            heads.append({
                "layer": layer,
                "head": head,
                "clean_attention_mean": float(clean_head_mean[0, head].item()),
                "adversarial_attention_mean": float(adv_head_mean[0, head].item()),
                "absolute_delta": float(delta[0, head].item()),
            })

    ranked = sorted(heads, key=lambda x: x["absolute_delta"], reverse=True)

    result = {
        "model_name": model_name,
        "experiment": "clean_vs_adversarial_attention",
        "clean_prompt": clean_prompt,
        "adversarial_prompt": adversarial_prompt,
        "top_changed_heads": ranked[:20],
        "all_heads": heads,
    }

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
