"""Executable-step targets, checkpoint migration, and gradient isolation."""
import unittest
import torch

from models.module import feasible_gate_step_loss
from models.sger_refinement import SGERBlock
from models import CascadeMVSNet
from train_sger_gate import freeze_except_gate


def test_feasible_step_targets_and_detached_supervision():
    # Targets: 0.25 (overshoot), 0 (wrong sign), 1 (undershoot),
    # 0.5 (negative proposal), 0 (raw already exact). Geometry capacity=0.5.
    raw = torch.full((1, 1, 5), 10., requires_grad=True)
    proposal = torch.tensor([[[4., 4., 4., -4., 4.]]], requires_grad=True)
    gt = torch.tensor([[[10.5, 9., 13., 9., 10.]]], requires_grad=True)
    z = torch.zeros_like(raw, requires_grad=True)
    stage = dict(depth_raw=raw, raw_depth_residual=proposal,
                 region_a=torch.zeros_like(raw, dtype=torch.bool),
                 region_b_weight=torch.full_like(raw, .5),
                 gate_valid_mask=torch.ones_like(raw, dtype=torch.bool),
                 gate_logits_a=z, gate_logits_b=z, sger_residual_scale=1.)
    loss, mean, ratio = feasible_gate_step_loss(stage, gt, torch.ones_like(raw))
    assert abs(float(mean) - .35) < 1e-6 and float(ratio) == 1.
    loss.backward()
    expected = (torch.tensor([[[.5, .5, .5, .5, .5]]]) - torch.tensor([[[.25, 0., 1., .5, 0.]]])) / 5
    assert torch.allclose(z.grad, expected)
    assert raw.grad is None and proposal.grad is None and gt.grad is None
    stage['sger_residual_scale'] = 0.
    loss, mean, ratio = feasible_gate_step_loss(stage, gt, torch.ones_like(raw))
    assert float(loss) == float(mean) == float(ratio) == 0.
    loss.backward()
    stage['sger_residual_scale'] = .5
    _, mean, _ = feasible_gate_step_loss(stage, gt, torch.ones_like(raw))
    assert abs(float(mean) - .5) < 1e-6


def test_empty_mask_and_saturated_logits():
    raw = torch.ones(1, 2, 2)
    z = torch.full_like(raw, 1000., requires_grad=True)
    stage = dict(depth_raw=raw, raw_depth_residual=raw,
                 region_a=raw.bool(), region_b_weight=raw * 0,
                 gate_valid_mask=raw.bool(), gate_logits_a=z, gate_logits_b=z)
    empty = feasible_gate_step_loss(stage, raw, raw * 0)[0]
    assert float(empty) == 0.
    loss = feasible_gate_step_loss(stage, raw, raw)[0]
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all() and z.grad.min() > 0


def test_zero_adapter_compatibility_and_proposal_gradient_boundary():
    torch.manual_seed(4)
    old = SGERBlock(8, max_residual_ratio=.25).eval()
    with torch.no_grad():
        old.residual.output.weight.normal_(0, .03)
    new = SGERBlock(8, max_residual_ratio=.25, proposal_conditioned=True).eval()
    missing, extra = new.load_state_dict(old.state_dict(), strict=False)
    assert set(missing) == {'gate.proposal_a.weight', 'gate.proposal_b.weight'} and not extra
    depth = torch.ones(2, 8, 8) * 500
    normal = torch.nn.functional.normalize(torch.rand(2, 3, 8, 8), dim=1)
    inputs = (depth, normal, torch.ones_like(depth)*.9, torch.rand(2, 3, 8, 8),
              torch.eye(3).repeat(2, 1, 1), torch.rand(2, 8, 8, 8), 1.)
    a, b = old(*inputs), new(*inputs)
    for key in ('depth_refined', 'benefit_gate', 'raw_depth_residual'):
        assert torch.equal(a[key], b[key]), key
    (b['gate_logits_a'].sum() + b['gate_logits_b'].sum()).backward()
    assert new.residual.output.weight.grad is None
    assert new.gate.proposal_a.weight.grad.abs().sum() > 0
    assert new.gate.proposal_b.weight.grad.abs().sum() > 0
    assert b['depth_residual'].abs().max() <= .25


def test_real_cascade_gate_update_preserves_proposal_backbone_and_bn():
    torch.manual_seed(7)
    model = CascadeMVSNet(ndepths=[8, 8, 8], depth_interals_ratio=[4, 2, 1],
                          use_sger_lite=True, sger_proposal_gate=True)
    with torch.no_grad():
        model.sger_blocks[0].residual.output.bias.fill_(.2)
    freeze_except_gate(model)
    image = torch.rand(1, 3, 3, 32, 32)
    camera = torch.eye(4).view(1, 1, 1, 4, 4).repeat(1, 3, 2, 1, 1)
    projections = {k: camera.clone() for k in ('stage1', 'stage2', 'stage3')}
    depths = torch.linspace(1., 2., 8).view(1, 8)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
    first = model(image, projections, depths)['stage3']
    first['depth_refined'].mean().backward()
    for name, p in model.named_parameters():
        if not name.startswith('sger_blocks.0.gate.'):
            assert p.grad is None, name
        else:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
    optimizer.step()
    with torch.no_grad():
        after = model(image, projections, depths)['stage3']
    for key in ('depth_raw', 'raw_depth_residual', 'photometric_confidence'):
        assert torch.equal(first[key], after[key]), key
    assert not torch.equal(first['benefit_gate'], after['benefit_gate'])
    for name, value in model.state_dict().items():
        if not name.startswith('sger_blocks.0.gate.'):
            assert torch.equal(value, before[name]), name


if __name__ == '__main__':
    torch.set_num_threads(1)
    suite = unittest.TestSuite(unittest.FunctionTestCase(v) for k, v in sorted(globals().items())
                               if k.startswith('test_'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
