import json
import torch
from models.sger_refinement import SGERBlock
from residual_audit_utils import (activation_stats, capture_residual_outputs,
                                  residual_diagnostics, residual_parameter_gradients)


class TinyBlock(SGERBlock):
    # Exercise the production residual head without unrelated camera geometry.
    def forward(self, features):
        return {'raw_depth_residual': self.max_residual_ratio * torch.tanh(self.residual(features))}


def test_dead_body_and_read_only_hooks():
    torch.manual_seed(2)
    block = TinyBlock(2, hidden_channels=4)
    for layer in block.residual.body:
        if isinstance(layer, torch.nn.Conv2d):
            torch.nn.init.zeros_(layer.weight)
            torch.nn.init.constant_(layer.bias, -1)
    torch.nn.init.constant_(block.residual.output.weight, 0.2)
    torch.nn.init.constant_(block.residual.output.bias, 0.1)
    optimizer = torch.optim.Adam(block.parameters())
    features = torch.randn(2, 26, 8, 8)
    baseline = block(features)['raw_depth_residual']
    baseline.sum().backward()
    gradients = {n: None if p.grad is None else p.grad.clone() for n, p in block.named_parameters()}
    block.zero_grad()
    with capture_residual_outputs(block) as captured:
        proposal = block(features)['raw_depth_residual']
    proposal.sum().backward()
    assert torch.equal(baseline, proposal)
    for name, parameter in block.named_parameters():
        assert ((parameter.grad is None and gradients[name] is None) or
                torch.equal(parameter.grad, gradients[name]))
    stage = dict(raw_depth_residual=proposal, depth_interval=1.0)
    stats = residual_diagnostics(stage, captured[id(proposal)], torch.ones_like(proposal))
    assert stats['activations']['body.1.input']['max'] == -1
    assert stats['activations']['body.1.output']['max'] == 0
    assert stats['activations']['output.input']['spatial_variance_per_sample_channel']['max'] == 0
    assert stats['abs_proposal_minus_bias_only']['max'] == 0
    grads = residual_parameter_gradients(block, optimizer)['']
    assert grads['output.weight']['zero_ratio'] == 1
    assert grads['output.bias']['l2'] > 0
    assert all(grads['body.{}.weight'.format(i)]['zero_ratio'] == 1 for i in (0, 2, 4))
    json.dumps([stats, grads], allow_nan=False)
    try:
        with capture_residual_outputs(block):
            raise RuntimeError('cleanup')
    except RuntimeError:
        pass
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in block.modules())


def test_active_and_nonfinite_statistics():
    stats = activation_stats(torch.tensor([[[[-1., 1.], [3., float('nan')]]]]))
    assert stats['nonfinite_count'] == 1
    assert abs(stats['positive_ratio'] - 2. / 3) < 1e-7
    assert abs(stats['spatial_variance_per_sample_channel']['mean'] - 8. / 3) < 1e-6
    empty = activation_stats(torch.full((1, 1, 2, 2), float('nan')))
    assert empty['positive_ratio'] is None
    json.dumps([stats, empty], allow_nan=False)


if __name__ == '__main__':
    torch.set_num_threads(1)
    test_dead_body_and_read_only_hooks()
    test_active_and_nonfinite_statistics()
    print('Residual audit tests: PASS')
