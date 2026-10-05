from pathlib import Path
import json

def compare_attention(
    model_name: str,
    clean_prompt: str,
    adversarial_prompt: str,
    layer: int,
    head: int,
    output: str = "artifacts/attention_comparison.json",
):
    try:
        from transformer_lens import HookedTransformer
    except ImportError as exc:
        raise RuntimeError("Install requirements-interpretability.txt first.") from exc

    model = HookedTransformer.from_pretrained(model_name)

    _, clean_cache = model.run_with_cache(clean_prompt)
    _, adv_cache = model.run_with_cache(adversarial_prompt)

    clean_tokens = model.to_str_tokens(clean_prompt)
    adv_tokens = model.to_str_tokens(adversarial_prompt)

    clean_pattern = clean_cache[f"blocks.{layer}.attn.hook_pattern"][0, head].float().detach().cpu()
    adv_pattern = adv_cache[f"blocks.{layer}.attn.hook_pattern"][0, head].float().detach().cpu()

    result = {
        "model_name": model_name,
        "layer": layer,
        "head": head,
        "clean": {
            "tokens": clean_tokens,
            "attention": clean_pattern.tolist(),
        },
        "adversarial": {
            "tokens": adv_tokens,
            "attention": adv_pattern.tolist(),
        },
        "note": "Token-level attention is descriptive evidence, not causal attribution.",
    }

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
