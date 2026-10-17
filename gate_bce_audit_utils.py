"""Experiment 15-D diagnostics for the unchanged mixed gate and its logits."""
import torch
from models.module import residual_benefit_target


def bce_forward_tensors(stage, depth_gt, mask, margin_ratio):
    values = {k: stage[k].detach().cpu().clone() for k in (
        'depth_raw', 'photometric_confidence', 'raw_depth_residual',
        'benefit_gate', 'depth_residual', 'depth_refined', 'region_a',
        'region_b_weight', 'gate_valid_mask', 'gate_logits_a', 'gate_logits_b',
        'sger_residual_scale')}
    values['gt_mask'] = (mask > 0.5).detach().cpu().clone()
    values['benefit_target'] = residual_benefit_target(
        stage['depth_raw'], stage['raw_depth_residual'], depth_gt,
        stage['depth_interval'], margin_ratio).cpu()
    return values


def compare_forward(reference, current):
    return {k + '_max_abs_difference': float(
        (current[k].to(torch.float64) - reference[k].to(torch.float64)).abs().max())
        for k in reference}


def gate_bce_gradient_diagnostics(stage, depth_gt, mask, margin_ratio):
    valid = mask > 0.5
    target = residual_benefit_target(stage['depth_raw'], stage['raw_depth_residual'],
                                    depth_gt, stage['depth_interval'], margin_ratio)
    region = stage['region_a'].bool()
    weight = (region.float() + stage['region_b_weight']) * stage['gate_valid_mask'].float()
    positive = valid & target
    result = {
        'valid_count': int(valid.sum()), 'positive_count': int(positive.sum()),
        'negative_count': int((valid & ~target).sum()),
        'structural_zero_count': int((valid & (weight == 0)).sum()),
        'unreachable_positive_count': int((positive & (weight == 0)).sum()),
        'direction_scope': 'individual active logit gradient; positive expects negative, negative expects positive',
        'branches': {},
    }
    for name, active in (('a', region), ('b', (~region) & (weight > 0))):
        z = stage['gate_logits_' + name]
        grad = z.grad
        branch = {'gradient_present': grad is not None, 'all_finite':
                  bool(torch.isfinite(grad).all()) if grad is not None else None}
        for label, selection in (
                ('positive', positive & active),
                ('negative', valid & (~target) & active),
                ('saturated_positive', positive & active & (stage['benefit_gate'] < 1e-6))):
            count = int(selection.sum())
            stats = {'count': count, 'mean_gradient': None, 'correct_sign_count': 0,
                     'wrong_sign_count': 0, 'zero_count': 0}
            if grad is not None and count:
                g = grad[selection]
                correct = g > 0 if label == 'negative' else g < 0
                wrong = g < 0 if label == 'negative' else g > 0
                stats.update(mean_gradient=float(g.double().mean()),
                             correct_sign_count=int(correct.sum()),
                             wrong_sign_count=int(wrong.sum()), zero_count=int((g == 0).sum()))
            branch[label] = stats
        result['branches'][name] = branch
    return result
