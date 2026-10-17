"""Read-only gate instrumentation used exclusively by gradient_audit.py."""
from contextlib import contextmanager

import numpy as np
import torch

from models.sger_refinement import DualRegionGate
from models.module import residual_benefit_target


@contextmanager
def capture_gate_outputs(model):
    """Capture detached branch outputs without changing the production graph."""
    captures, handles = {}, []
    pending = {}

    def branch_hook(identity, key):
        def hook(module, inputs, output):
            pending.setdefault(identity, {})[key] = output.detach().squeeze(1)
        return hook

    def gate_hook(identity):
        def hook(module, inputs, output):
            values = dict(pending.pop(identity, {}))
            gate, region_a, region_b, weight_b = output[:4]
            values.update(region_a=region_a.detach(), region_b=region_b.detach(),
                          geometry_weight_a=region_a.detach().float(),
                          geometry_weight_b=weight_b.detach(),
                          final_gate=gate.detach())
            for branch in ('a', 'b'):
                values['sigmoid_' + branch] = torch.sigmoid(values['logits_' + branch])
            captures[id(gate)] = values
        return hook

    try:
        for module in model.modules():
            if isinstance(module, DualRegionGate):
                identity = id(module)
                handles.append(module.gate_a.register_forward_hook(branch_hook(identity, 'logits_a')))
                handles.append(module.gate_b.register_forward_hook(branch_hook(identity, 'logits_b')))
                handles.append(module.register_forward_hook(gate_hook(identity)))
        yield captures
    finally:
        for handle in handles:
            handle.remove()
        pending.clear()


def distribution(value, mask):
    """Exact CPU percentiles over finite selected pixels; null for empty masks."""
    selected = value.detach().expand_as(mask)[mask].float().cpu().numpy()
    finite = selected[np.isfinite(selected)]
    result = {'count': int(selected.size), 'finite_count': int(finite.size),
              'nonfinite_count': int(selected.size - finite.size)}
    result.update({key: None for key in ('min', 'p01', 'p05', 'p50', 'p95', 'p99', 'max', 'mean')})
    if finite.size:
        values = np.percentile(finite, [0, 1, 5, 50, 95, 99, 100])
        result.update(dict(zip(('min', 'p01', 'p05', 'p50', 'p95', 'p99', 'max'),
                               [float(x) for x in values])))
        result['mean'] = float(finite.astype(np.float64).mean())
    return result


@torch.no_grad()
def gate_diagnostics(stage, captured, depth_gt, mask, margin_ratio):
    valid = mask > 0.5
    gate = stage['benefit_gate'].detach()
    raw = stage['depth_raw'].detach()
    proposal = stage['raw_depth_residual'].detach()
    interval = torch.as_tensor(stage['depth_interval'], device=raw.device, dtype=raw.dtype)
    margin = (interval * margin_ratio).expand_as(raw)
    gain = (raw - depth_gt).abs() - (raw + proposal - depth_gt).abs()
    # Match the production target's floating-point operation ordering exactly.
    target = residual_benefit_target(raw, proposal, depth_gt, interval, margin_ratio)
    predicted = gate >= 0.5
    count = int(valid.sum().item())
    lower = int((valid & (gate < 1e-6)).sum().item())
    upper = int((valid & (gate > 1.0 - 1e-6)).sum().item())
    values = dict(captured)
    values.update(final_gate=gate, raw_residual=proposal,
                  abs_raw_residual=proposal.abs(), proposal_gain=gain,
                  benefit_margin=margin, gain_minus_margin=gain - margin,
                  effective_residual=stage['depth_residual'].detach())
    stats = {key: distribution(value, valid) for key, value in values.items()
             if key not in ('region_a', 'region_b')}
    regions = {}
    for branch in ('a', 'b'):
        region = captured.get('region_' + branch)
        if region is not None:
            region_mask = valid & region.bool()
            regions[branch] = {key: distribution(values[key], region_mask)
                               for key in ('logits_' + branch, 'sigmoid_' + branch,
                                           'geometry_weight_' + branch, 'final_gate')}
    return {
        'mask': 'stage GT mask > 0.5; pixels pooled across batch',
        'valid_pixel_count': count,
        'captured_branch_outputs': bool(captured),
        'distributions': stats,
        'region_distributions': regions,
        'gate_below_1e_minus6_count': lower,
        'gate_below_1e_minus6_ratio': float(lower) / count if count else None,
        'gate_above_1_minus_1e_minus6_count': upper,
        'gate_above_1_minus_1e_minus6_ratio': float(upper) / count if count else None,
        'target_positive_count': int((valid & target).sum().item()),
        'predicted_positive_count': int((valid & predicted).sum().item()),
        'true_positive_count': int((valid & predicted & target).sum().item()),
        'positive_gain_count': int((valid & (gain > 0)).sum().item()),
        'positive_gain_below_margin_count': int((valid & (gain > 0) & ~target).sum().item()),
    }
