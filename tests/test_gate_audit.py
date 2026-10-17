import json
import torch
from gate_audit_utils import capture_gate_outputs, distribution, gate_diagnostics
from models.sger_refinement import DualRegionGate
from gradient_audit import audit_loss_inputs
from gradient_audit_utils import build_audit_cases, summarize_gradient_boundary
from models import cas_mvsnet_loss


def test_capture_preserves_forward_and_gradients():
    torch.manual_seed(1)
    model = DualRegionGate(2, hidden_channels=2)
    inputs = (torch.randn(1, 2, 8, 8), torch.rand(1, 3, 8, 8),
              torch.ones(1, 8, 8), torch.ones(1, 8, 8),
              torch.ones(1, 8, 8, dtype=torch.bool))
    baseline = model(*inputs)[0]
    baseline.sum().backward()
    grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    with capture_gate_outputs(model) as captured:
        output = model(*inputs)[0]
    assert torch.equal(output, baseline)
    output.sum().backward()
    assert all(torch.equal(p.grad, g) for p, g in zip(model.parameters(), grads))
    data = captured[id(output)]
    assert not data['logits_a'].requires_grad
    assert torch.allclose(output, data['geometry_weight_a'] * data['sigmoid_a'] +
                          data['geometry_weight_b'] * data['sigmoid_b'])
    assert all(not m._forward_hooks for m in model.modules())
    try:
        with capture_gate_outputs(model):
            raise RuntimeError('cleanup')
    except RuntimeError:
        pass
    assert all(not m._forward_hooks for m in model.modules())


def test_counts_and_empty_nonfinite_distributions():
    raw = torch.full((1, 1, 4), 2.0)
    proposal = torch.tensor([[[-0.2, -0.01, 0.2, 0.0]]])
    stage = dict(depth_raw=raw, raw_depth_residual=proposal, depth_interval=1.0,
                 benefit_gate=torch.tensor([[[1e-13, 0.6, 0.9, 0.0]]]),
                 depth_residual=proposal)
    mask = torch.ones_like(raw, dtype=torch.bool)
    result = gate_diagnostics(stage, {}, torch.ones_like(raw), mask, 0.05)
    assert result['target_positive_count'] == 1
    assert result['predicted_positive_count'] == 2
    assert result['true_positive_count'] == 0
    assert result['gate_below_1e_minus6_count'] == 2
    assert result['positive_gain_below_margin_count'] == 1
    empty = gate_diagnostics(stage, {}, torch.ones_like(raw), ~mask, 0.05)
    assert empty['gate_below_1e_minus6_ratio'] is None
    assert empty['distributions']['proposal_gain']['mean'] is None
    stats = distribution(torch.tensor([float('nan'), 2.0]), torch.tensor([True, True]))
    assert stats['nonfinite_count'] == 1 and stats['p50'] == 2.0
    json.dumps([result, empty, stats], allow_nan=False)


def test_stage3_geometry_excludes_coarse_depth():
    coarse = torch.randn(1, 8, 8, requires_grad=True)
    refined = torch.randn(1, 8, 8, requires_grad=True)
    raw = torch.randn(1, 8, 8, requires_grad=True)
    outputs = {'stage1': {'depth': coarse},
               'stage3': {'depth': refined, 'depth_refined': refined, 'depth_raw': raw}}
    cases = [c for c in build_audit_cases(10, {}) if c.name.startswith('stage3_')]
    assert len(cases) == 4
    for case in cases:
        assert set(audit_loss_inputs(outputs, case)) == {'stage3'}
        assert summarize_gradient_boundary({}, case.name)['sger_exclusive']
    case = next(c for c in cases if c.name == 'stage3_normal_smooth_only')
    gt = {k: torch.ones_like(raw) for k in outputs}
    loss = cas_mvsnet_loss(audit_loss_inputs(outputs, case), gt, gt,
                          dlossw=[0.5, 1, 2], **case.loss_weights)[0]
    loss.backward()
    assert coarse.grad is None
    assert raw.grad is None or raw.grad.abs().sum() == 0
    assert refined.grad.abs().sum() > 0


if __name__ == '__main__':
    test_capture_preserves_forward_and_gradients()
    test_counts_and_empty_nonfinite_distributions()
    test_stage3_geometry_excludes_coarse_depth()
    print('Gate audit tests: PASS')
