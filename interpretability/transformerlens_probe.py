from pathlib import Path
import json

def _pooled_residual(cache, layer):
    key=f'blocks.{layer}.hook_out'
    if key not in cache: raise KeyError(f'Missing canonical hook: {key}')
    return cache[key].float().mean(dim=1)

def run_residual_probe(model_name, clean_prompt, adversarial_prompt, output='artifacts/residual_probe.json'):
    try: from transformer_lens.model_bridge import TransformerBridge
    except ImportError as exc: raise RuntimeError('Install current TransformerLens first.') from exc
    import torch
    bridge=TransformerBridge.boot_transformers(model_name, device='cpu', dtype=torch.float32)
    bridge.enable_compatibility_mode()
    _, clean_cache=bridge.run_with_cache(clean_prompt)
    _, adv_cache=bridge.run_with_cache(adversarial_prompt)
    records=[]
    for layer in range(bridge.cfg.n_layers):
        clean=_pooled_residual(clean_cache,layer); adv=_pooled_residual(adv_cache,layer); delta=adv-clean
        records.append({'layer':layer,'mean_delta':float(delta.mean()),'mean_abs_delta':float(delta.abs().mean()),'l2_delta':float(delta.norm())})
    result={'model_name':model_name,'experiment':'clean_vs_adversarial_residual','clean_prompt':clean_prompt,'adversarial_prompt':adversarial_prompt,'layers':records}
    p=Path(output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
