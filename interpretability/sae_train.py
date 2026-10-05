from pathlib import Path
import json

def train_sae(activations, feature_multiplier=4, l1_coefficient=1e-3, epochs=50, learning_rate=1e-3, seed=7):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    if activations.ndim != 2: raise ValueError('activations must have shape [samples, d_model]')
    x=activations.float(); d_model=x.shape[-1]; n_features=d_model*feature_multiplier
    ew=torch.nn.Parameter(torch.randn(n_features,d_model)*0.01); eb=torch.nn.Parameter(torch.zeros(n_features))
    dw=torch.nn.Parameter(torch.randn(d_model,n_features)*0.01); db=torch.nn.Parameter(torch.zeros(d_model))
    opt=torch.optim.Adam([ew,eb,dw,db],lr=learning_rate)
    for _ in range(epochs):
        f=F.relu(x@ew.T+eb); r=f@dw.T+db; loss=F.mse_loss(r,x)+l1_coefficient*f.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        f=F.relu(x@ew.T+eb); r=f@dw.T+db
    return {'encoder_weight':ew.detach(),'encoder_bias':eb.detach(),'decoder_weight':dw.detach(),'decoder_bias':db.detach(),'features':f.detach(),'reconstruction':r.detach(),'metrics':{'reconstruction_mse':float(F.mse_loss(r,x).item()),'active_feature_fraction':float((f>0).float().mean().item()),'feature_count':n_features,'input_dimension':d_model}}

def rank_security_features(clean_features, adversarial_features, top_k=20):
    import torch
    cr=(clean_features>0).float().mean(0); ar=(adversarial_features>0).float().mean(0); delta=(ar-cr).abs()
    values,indices=torch.topk(delta,k=min(top_k,delta.numel()))
    return [{'feature':int(i),'clean_activation_rate':float(cr[i]),'adversarial_activation_rate':float(ar[i]),'absolute_rate_delta':float(v)} for v,i in zip(values.tolist(),indices.tolist())]

def save_report(report, output='artifacts/sae_feature_report.json'):
    path=Path(output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(report,indent=2),encoding='utf-8')
