import json
import os
import tempfile
from unittest.mock import patch

import torch
from models.sger_refinement import ResidualDepthHead
from gradient_audit import main
from test_residual_audit_batches import SyntheticDTU


def test_last_activation_restores_body_gradient():
    head = ResidualDepthHead(2, hidden_channels=4)
    with torch.no_grad():
        for i in (0, 2):
            head.body[i].weight.fill_(0.02)
            head.body[i].bias.fill_(0.1)
        head.body[4].weight.fill_(-0.02)
        head.body[4].bias.fill_(-0.1)
        head.output.weight.fill_(0.02)
    original = {k: v.clone() for k, v in head.state_dict().items()}
    x = torch.rand(2, 2, 8, 8)
    head(x).sum().backward()
    assert all(p.grad.abs().sum() == 0 for p in head.body.parameters())
    head.zero_grad()
    head.set_final_activation('leaky_relu', 0.01)
    y = head(x)
    y.sum().backward()
    assert all(torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in head.body.parameters())
    assert y.var() > 0 and head.output.weight.grad.abs().sum() > 0
    assert set(original) == set(head.state_dict())
    assert all(torch.equal(original[k], v) for k, v in head.state_dict().items())
    assert isinstance(head.body[1], torch.nn.ReLU) and isinstance(head.body[3], torch.nn.ReLU)
    for slope in (0, -1, float('nan'), float('inf'), 1):
        try:
            head.set_final_activation('leaky_relu', slope)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid slope accepted')


def test_ab_entrypoint():
    with tempfile.TemporaryDirectory() as directory:
        with patch('gradient_audit.find_dataset_def', return_value=SyntheticDTU):
            main(['--trainpath', directory, '--trainlist', 'lists/dtu/train.txt',
                  '--output_json', os.path.join(directory, 'audit.json'),
                  '--audit_epoch', '12', '--device', 'cpu', '--batch_size', '2',
                  '--batch_indices', '0', '--ndepths', '8,8,8', '--activation_ab'])
        with open(os.path.join(directory, 'audit_ab_comparison.json')) as f:
            comparison = json.load(f)
        assert all(v == 0 for v in comparison['full_schedule_raw_invariance'].values())
        for activation in ('relu', 'leaky_relu'):
            with open(os.path.join(directory, 'audit_' + activation + '.json')) as f:
                report = json.load(f)
            assert report['metadata']['actual_residual_final_activation'] == activation
            for case in report['cases']:
                bounds = case['loss_extras']['residual_bound_checks']
                assert bounds['proposal_all_finite'] and bounds['effective_all_finite']
                assert bounds['max_abs_proposal'] <= bounds['proposal_bound'] + 1e-6
                assert bounds['max_abs_effective_residual'] <= bounds['effective_bound'] + 1e-6
                assert 'body.5.input' in case['loss_extras']['residual_diagnostics']['stage3']['activations']
                if case['gradient_summary']['sger_exclusive']:
                    assert case['gradient_summary']['backbone_grad_norm'] == 0


if __name__ == '__main__':
    torch.set_num_threads(1)
    test_last_activation_restores_body_gradient()
    test_ab_entrypoint()
    print('Activation A/B tests: PASS')
