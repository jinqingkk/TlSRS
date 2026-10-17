"""Regression: isolated losses must not include coarse-stage depth supervision."""
import torch
from gradient_audit import audit_loss_inputs
from gradient_audit_utils import build_audit_cases, summarize_gradient_boundary
from models import cas_mvsnet_loss


def test_production_loss_isolation():
    for name in ('raw_only', 'refined_only'):
        coarse = torch.ones(1, 4, 4, requires_grad=True)
        raw = torch.ones(1, 4, 4, requires_grad=True)
        residual = torch.full((1, 4, 4), 0.1, requires_grad=True)
        refined = raw.detach() + residual
        outputs = {'stage1': {'depth': coarse},
                   'stage3': {'depth': refined, 'depth_refined': refined,
                              'depth_raw': raw}}
        case = next(c for c in build_audit_cases(10, {}) if c.name == name)
        gt = {k: torch.zeros(1, 4, 4) for k in outputs}
        mask = {k: torch.ones(1, 4, 4) for k in outputs}
        loss = cas_mvsnet_loss(audit_loss_inputs(outputs, case), gt, mask,
                              dlossw=[0.5, 1.0, 2.0], **case.loss_weights)[0]
        loss.backward()
        if name == 'refined_only':
            assert coarse.grad is None
            assert raw.grad is None or raw.grad.abs().sum() == 0
            assert residual.grad.abs().sum() > 0
        else:
            assert coarse.grad.abs().sum() > 0
            assert raw.grad.abs().sum() > 0
            assert residual.grad is None or residual.grad.abs().sum() == 0
        assert 'depth_raw' not in outputs['stage1']


def test_boundary_summary():
    groups = {k: {'l2': v, 'all_finite': True} for k, v in
              [('backbone_feature', 3.0), ('cost_regularization', 4.0),
               ('sger_residual_output', 2.0)]}
    result = summarize_gradient_boundary(groups, 'refined_only')
    assert result['backbone_grad_norm'] == 5.0
    assert result['backbone_to_sger_grad_ratio'] == 2.5
    assert result['backbone_leak_detected'] is True


if __name__ == '__main__':
    test_production_loss_isolation()
    test_boundary_summary()
    print('Production loss isolation tests: PASS')
