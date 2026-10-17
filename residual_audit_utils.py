"""Read-only residual instrumentation; statistics never enter the model graph."""
from contextlib import contextmanager
import time

import numpy as np
import torch
from torch import nn

from gate_audit_utils import distribution
from gradient_audit_utils import gradient_stats
from models.sger_refinement import SGERBlock


@torch.no_grad()
def activation_stats(value):
    # Summarize immediately: production ReLUs are in-place and would overwrite
    # a detached view saved by a Conv hook. No full activation is retained.
    value = value.detach().float()
    finite = torch.isfinite(value)
    result = distribution(value, torch.ones_like(value, dtype=torch.bool))
    count = result['finite_count']
    result['shape'] = list(value.shape)
    result['positive_ratio'] = float((finite & (value > 0)).sum().item()) / count if count else None
    result['zero_ratio'] = float((finite & (value == 0)).sum().item()) / count if count else None
    # Population spatial variance separately for each sample/channel. Exclude
    # nonfinite pixels; empty channels are reported as nonfinite, never as zero.
    # Transfer once and reduce one sample/channel at a time on CPU. Avoid
    # several full-resolution float64 CUDA temporaries alongside the graph.
    flat = value.cpu().numpy().reshape(value.shape[0], value.shape[1], -1)
    variances = []
    for sample in flat:
        for channel in sample:
            selected = channel[np.isfinite(channel)]
            variances.append(float(np.var(selected, dtype=np.float64))
                             if selected.size else float('nan'))
    variance = torch.tensor(variances, dtype=torch.float64)
    result['spatial_variance_per_sample_channel'] = distribution(
        variance, torch.ones_like(variance, dtype=torch.bool))
    return result


@contextmanager
def capture_residual_outputs(model, progress=False):
    captures, handles, pending = {}, [], {}

    def activation_hook(identity, name, use_input=False):
        def hook(module, inputs, output=None):
            started = time.monotonic()
            if progress:
                print("      residual {}: collecting".format(name), flush=True)
            pending.setdefault(identity, {})[name] = activation_stats(
                inputs[0] if use_input else output)
            if progress:
                print("      residual {}: {:.1f}s".format(
                    name, time.monotonic() - started), flush=True)
        return hook

    def block_hook(identity, name):
        @torch.no_grad()
        def hook(module, inputs, output):
            proposal = output['raw_depth_residual']
            values = pending.pop(identity, {})
            bias = module.residual.output.bias.detach()
            captures[id(proposal)] = dict(module_name=name, activations=values,
                                         output_bias=bias.cpu().tolist(),
                                         max_residual_ratio=module.max_residual_ratio)
        return hook

    try:
        for name, block in model.named_modules():
            if not isinstance(block, SGERBlock):
                continue
            identity = id(block)
            for layer_name, layer in block.residual.named_modules():
                if isinstance(layer, (nn.Conv2d, nn.ReLU, nn.LeakyReLU)):
                    handles.append(layer.register_forward_hook(
                        activation_hook(identity, layer_name + '.output')))
                    handles.append(layer.register_forward_pre_hook(
                        activation_hook(identity, layer_name + '.input', True)))
            handles.append(block.register_forward_hook(block_hook(identity, name)))
        yield captures
    finally:
        for handle in handles:
            handle.remove()
        pending.clear()


@torch.no_grad()
def residual_diagnostics(stage, captured, mask):
    if not captured:
        return {'captured': False}
    result = dict(captured)
    result.update(captured=True, activation_scope='all B,C,H,W values; no GT mask')
    proposal = stage['raw_depth_residual'].detach()
    bias = proposal.new_tensor(captured['output_bias']).reshape(1, 1, 1)
    maximum = captured['max_residual_ratio'] * torch.as_tensor(
        stage['depth_interval'], device=proposal.device, dtype=proposal.dtype)
    expected = maximum * torch.tanh(bias)
    valid = mask > 0.5
    result['bias_only_expected_proposal'] = distribution(expected, valid)
    result['abs_proposal_minus_bias_only'] = distribution((proposal - expected).abs(), valid)
    result['proposal_comparison_scope'] = 'stage GT mask > 0.5; pooled batch pixels'
    # Equality alone is not proof of dead ReLU: e.g. zero output weights also
    # produce a bias-only proposal. Read alongside activation/gradient evidence.
    return result


def residual_parameter_gradients(model, optimizer):
    covered = {id(p) for group in optimizer.param_groups for p in group['params']}
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, SGERBlock):
            result[name] = {}
            for key, parameter in module.residual.named_parameters():
                stats = gradient_stats(parameter.grad)
                stats.update(requires_grad=parameter.requires_grad,
                             optimizer_covered=id(parameter) in covered)
                result[name][key] = stats
    return result
