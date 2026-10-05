def test_sae_training_and_ranking():
    import torch
    from interpretability.sae_train import train_sae, rank_security_features
    clean=torch.randn(24,8); adversarial=clean+0.25*torch.randn(24,8)
    result=train_sae(clean,feature_multiplier=2,epochs=3)
    adv=torch.relu(adversarial@result['encoder_weight'].T+result['encoder_bias'])
    ranked=rank_security_features(result['features'],adv,top_k=5)
    assert len(ranked)==5
    assert all('absolute_rate_delta' in x for x in ranked)
