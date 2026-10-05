from pathlib import Path
import json

def summarize_token_attention(
    model_name: str,
    prompt: str,
    layer: int,
    head: int,
    output: str = "artifacts/token_attention.json",
):
    """Export token-to-token attention for one selected head."""
    try:
        from transformer_lens import HookedTransformer
    except ImportError as exc:
        raise RuntimeError("Install requirements-interpretability.txt first.") from exc

    model = HookedTransformer.from_pretrained(model_name)
    tokens = model.to_str_tokens(prompt)
    _, cache = model.run_with_cache(prompt)

    pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0, head].float().detach().cpu()

    result = {
        "model_name": model_name,
        "layer": layer,
        "head": head,
        "tokens": tokens,
        "attention_matrix": pattern.tolist(),
    }

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
