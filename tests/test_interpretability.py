from interpretability.activation_steering import apply_direction, build_direction, experiment_schema

def test_build_direction_is_normalized():
    import torch
    direction = build_direction(torch.tensor([2.0, 0.0]), torch.tensor([0.0, 0.0]))
    assert torch.isclose(direction.norm(), torch.tensor(1.0))

def test_apply_direction_changes_activation():
    import torch
    result = apply_direction(torch.tensor([1.0, 1.0]), torch.tensor([1.0, 0.0]), 0.5)
    assert result.tolist() == [1.5, 1.0]

def test_experiment_schema():
    result = experiment_schema('demo-model', 4, [-1.0, 0.0, 1.0], {'attack_success': 0}, [{'coefficient': 1.0, 'attack_success': 0}])
    assert result['layer'] == 4
    assert result['intervention']['type'] == 'additive_activation_steering'
