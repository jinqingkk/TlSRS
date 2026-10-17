"""Read-only cross-scene probe of bilateral multi-view matching cues."""
import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import experiment15_momentum_diagnose as diag
import experiment15_residual_feature_separability as base_probe
import experiment15_residual_gradient_decomposition as gradient_audit
import experiment15_signed_probability_cue_separability as signed_probe
import models.cas_mvsnet as cas_module
from experiment15_d_diagnose import rng_snapshot
from models.module import homo_warping


GROUP_NAMES = ('base', 'bilateral_only', 'base_plus_bilateral')
PRIMARY_GROUP = 'base_plus_bilateral'
BILATERAL_CUE_NAMES = (
    'bilateral_mean_advantage',
    'bilateral_median_advantage',
    'bilateral_vote_balance',
    'bilateral_signal_to_dispersion',
)
POOLED_DELTA_GATE = 0.03
SUPPORT_RATIO_GATE = 0.90
SUPPORT_CLASS_GAP_GATE = 0.05


def load_signed_probe(root, probe_root, gradient_root, feature_root):
    """Bind this experiment to the completed negative signed-cue probe."""
    manifest_path = root / 'manifest.json'
    summary_path = root / 'summary.json'
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    if (manifest.get('status') != 'completed'
            or summary.get('status') != 'completed'
            or not summary.get('checks_passed')
            or not summary.get('input_files_unchanged')
            or manifest.get('strict_determinism') is not True
            or Path(manifest.get('source_probe', '')).resolve()
            != probe_root.resolve()
            or Path(manifest.get('source_gradient_audit', '')).resolve()
            != gradient_root.resolve()
            or Path(manifest.get('source_feature_separability', '')).resolve()
            != feature_root.resolve()
            or summary.get('decision', {}).get(
                'signed_probability_cues_established') is not False):
        raise ValueError(
            'Requires the completed negative signed-probability cue probe')
    for section in ('source_sha256', 'input_sha256'):
        for filename, digest in manifest.get(section, {}).items():
            if diag.sha(filename) != digest:
                raise ValueError(
                    'Signed-probe source/input hash mismatch: {}'.format(
                        filename))
    return manifest, summary, (manifest_path, summary_path)


def actual_depth_bin_interval(depth_values, rtol=1e-4, atol=1e-6):
    """Derive and validate the actual local hypothesis spacing."""
    if depth_values.dim() != 4 or depth_values.size(1) < 2:
        raise ValueError('Expected depth hypotheses shaped (B,D,H,W), D>=2')
    if not bool(torch.isfinite(depth_values).all()):
        raise RuntimeError('Nonfinite depth hypotheses')
    increments = depth_values[:, 1:] - depth_values[:, :-1]
    if bool((increments < 0).any()):
        raise ValueError('Depth hypotheses must be monotonically increasing')
    if bool((increments == 0).any()):
        raise ValueError('Depth hypothesis spacing must be positive')
    # CUDA median(dim=...) returns indices and is marked nondeterministic in
    # the legacy PyTorch runtime used by this project.  The hypotheses are
    # read-only, so compute the exact same median deterministically on CPU and
    # move only the resulting per-pixel interval back to the source device.
    delta = increments.detach().cpu().median(dim=1)[0].to(
        device=depth_values.device, dtype=depth_values.dtype)
    if not bool(torch.isfinite(delta).all()) or bool((delta <= 0).any()):
        raise ValueError('Actual depth-bin interval must be finite and positive')
    tolerance = max(float(atol), float(delta.detach().abs().max()) * rtol)
    if not bool(torch.allclose(
            increments, delta.unsqueeze(1), rtol=rtol, atol=tolerance)):
        raise ValueError(
            'Actual depth-bin interval is inconsistent across hypotheses')
    return delta


def bilateral_cues_from_warped(ref_feature, warped_minus, warped_plus,
                               valid_minus, valid_plus, eps=1e-6):
    """Aggregate signed per-view matching advantages into four cues."""
    if len(warped_minus) != len(warped_plus):
        raise ValueError('Bilateral warped-view counts differ')
    if len(valid_minus) != len(warped_minus) or len(valid_plus) != len(
            warped_minus):
        raise ValueError('Warped features and validity-mask counts differ')
    if not warped_minus:
        raise ValueError('At least one source view is required')
    if ref_feature.dim() != 4:
        raise ValueError('Expected reference feature shaped (B,C,H,W)')
    normalized_ref = F.normalize(ref_feature, p=2, dim=1, eps=eps)
    advantages = []
    valid_views = []
    for minus, plus, minus_valid, plus_valid in zip(
            warped_minus, warped_plus, valid_minus, valid_plus):
        if minus.shape != ref_feature.shape or plus.shape != ref_feature.shape:
            raise ValueError('Reference and warped feature shapes differ')
        expected_mask_shape = (
            ref_feature.size(0), ref_feature.size(2), ref_feature.size(3))
        if (tuple(minus_valid.shape) != expected_mask_shape
                or tuple(plus_valid.shape) != expected_mask_shape):
            raise ValueError('Unexpected bilateral validity-mask shape')
        normalized_minus = F.normalize(minus, p=2, dim=1, eps=eps)
        normalized_plus = F.normalize(plus, p=2, dim=1, eps=eps)
        minus_cost = (normalized_ref - normalized_minus).square().mean(1)
        plus_cost = (normalized_ref - normalized_plus).square().mean(1)
        advantage = minus_cost - plus_cost
        valid = (minus_valid.bool() & plus_valid.bool()
                 & torch.isfinite(advantage))
        advantages.append(torch.where(
            valid, advantage, torch.zeros_like(advantage)))
        valid_views.append(valid)

    advantage = torch.stack(advantages, dim=1)
    valid = torch.stack(valid_views, dim=1)
    valid_count = valid.sum(1)
    safe_count = valid_count.clamp_min(1).to(advantage.dtype)
    mean = advantage.sum(1) / safe_count

    infinity = torch.full_like(advantage, float('inf'))
    sorted_values = torch.where(valid, advantage, infinity).sort(dim=1)[0]
    lower_index = ((valid_count - 1).clamp_min(0) // 2).unsqueeze(1)
    upper_index = (valid_count.clamp_min(1) // 2).unsqueeze(1)
    lower = torch.gather(sorted_values, 1, lower_index).squeeze(1)
    upper = torch.gather(sorted_values, 1, upper_index).squeeze(1)
    median = 0.5 * (lower + upper)
    median = torch.where(valid_count > 0, median, torch.zeros_like(median))

    positive_votes = ((advantage > 0) & valid).sum(1)
    negative_votes = ((advantage < 0) & valid).sum(1)
    vote_balance = (
        (positive_votes - negative_votes).to(advantage.dtype) / safe_count)
    centered = advantage - mean.unsqueeze(1)
    variance = (centered.square() * valid.to(advantage.dtype)).sum(1)
    variance = variance / safe_count
    dispersion = variance.clamp_min(0).sqrt()
    signal_to_dispersion = mean / (dispersion + eps)
    support = valid_count >= 2
    cues = torch.stack([
        mean, median, vote_balance, signal_to_dispersion,
    ], dim=1)
    cues = torch.where(support.unsqueeze(1), cues, torch.zeros_like(cues))
    if not bool(torch.isfinite(cues).all()):
        raise RuntimeError('Nonfinite bilateral matching cue')
    return cues, support, valid_count


def _combined_projection(projection):
    result = projection[:, 0].clone()
    result[:, :3, :4] = torch.matmul(
        projection[:, 1, :3, :3], projection[:, 0, :3, :4])
    return result


def bilateral_matching_cues(ref_feature, src_features, proj_matrices,
                            depth_raw, depth_values, eps=1e-6,
                            valid_tolerance=1e-4):
    """Compute inference-only matching asymmetry around Stage3 raw depth."""
    if (ref_feature.dim() != 4 or depth_raw.dim() != 3
            or ref_feature.shape[0] != depth_raw.shape[0]
            or ref_feature.shape[-2:] != depth_raw.shape[-2:]):
        raise ValueError('Reference feature and raw-depth shapes differ')
    if (proj_matrices.dim() != 5 or proj_matrices.size(1) != len(
            src_features) + 1):
        raise ValueError('Feature and projection view counts differ')
    if (depth_values.shape[0] != depth_raw.shape[0]
            or depth_values.shape[-2:] != depth_raw.shape[-2:]):
        raise ValueError('Depth hypothesis and raw-depth shapes differ')
    delta = actual_depth_bin_interval(depth_values)
    d_minus = depth_raw - delta
    d_plus = depth_raw + delta
    candidate_valid = (
        torch.isfinite(d_minus) & torch.isfinite(d_plus)
        & (d_minus > 0) & (d_plus > 0))
    safe_minus = torch.where(
        candidate_valid, d_minus, torch.ones_like(d_minus)).unsqueeze(1)
    safe_plus = torch.where(
        candidate_valid, d_plus, torch.ones_like(d_plus)).unsqueeze(1)

    projections = torch.unbind(proj_matrices, 1)
    ref_projection = _combined_projection(projections[0])
    warped_minus = []
    warped_plus = []
    valid_minus = []
    valid_plus = []
    for src_feature, src_projection in zip(src_features, projections[1:]):
        if src_feature.shape != ref_feature.shape:
            raise ValueError('Reference and source feature shapes differ')
        source_projection = _combined_projection(src_projection)
        ones = torch.ones_like(src_feature[:, :1])
        source_and_validity = torch.cat([src_feature, ones], dim=1)
        minus = homo_warping(
            source_and_validity, source_projection, ref_projection,
            safe_minus).squeeze(2)
        plus = homo_warping(
            source_and_validity, source_projection, ref_projection,
            safe_plus).squeeze(2)
        minus_feature, minus_ones = minus[:, :-1], minus[:, -1]
        plus_feature, plus_ones = plus[:, :-1], plus[:, -1]
        threshold = 1.0 - float(valid_tolerance)
        minus_valid = (
            candidate_valid & torch.isfinite(minus_ones)
            & (minus_ones >= threshold))
        plus_valid = (
            candidate_valid & torch.isfinite(plus_ones)
            & (plus_ones >= threshold))
        warped_minus.append(minus_feature)
        warped_plus.append(plus_feature)
        valid_minus.append(minus_valid)
        valid_plus.append(plus_valid)
    cues, support, valid_count = bilateral_cues_from_warped(
        ref_feature, warped_minus, warped_plus,
        valid_minus, valid_plus, eps=eps)
    return cues, support, valid_count, delta


def capture_bilateral_forward(model, sample, args):
    """Capture fused input and Stage3 geometry without production edits."""
    fused_values = []
    stage_capture = {}
    depth_call_count = [0]
    original_forward = cas_module.DepthNet.forward

    def capture_fused(module, inputs):
        value = inputs[0]
        if value.dim() != 4:
            raise ValueError('Expected fused features shaped (B,C,H,W)')
        fused_values.append(value.detach())

    def capture_depth(module, features, proj_matrices, depth_values,
                      num_depth, cost_regularization,
                      prob_volume_init=None):
        depth_call_count[0] += 1
        stage_capture.clear()
        stage_capture.update({
            'features': [feature.detach() for feature in features],
            'proj_matrices': proj_matrices.detach(),
            'depth_values': depth_values.detach(),
            'num_depth': int(num_depth),
        })
        return original_forward(
            module, features, proj_matrices, depth_values, num_depth,
            cost_regularization, prob_volume_init)

    handle = model.sger_blocks[0].residual.register_forward_pre_hook(
        capture_fused)
    cas_module.DepthNet.forward = capture_depth
    try:
        with torch.no_grad():
            outputs, _, residual_logit = (
                gradient_audit.direction_probe.forward_with_residual_logit(
                    model, sample, args))
    finally:
        cas_module.DepthNet.forward = original_forward
        handle.remove()
    if len(fused_values) != 1:
        raise RuntimeError('Expected exactly one fused SGER residual input')
    if depth_call_count[0] != model.num_stage:
        raise RuntimeError('Expected one DepthNet call per cascade stage')
    if stage_capture.get('num_depth') != model.ndepths[-1]:
        raise RuntimeError('Final DepthNet capture is not Stage3')
    stage = outputs['stage3']
    cues, support, valid_count, delta = bilateral_matching_cues(
        stage_capture['features'][0], stage_capture['features'][1:],
        stage_capture['proj_matrices'], stage['depth_raw'],
        stage_capture['depth_values'])
    if cues.shape[-2:] != fused_values[0].shape[-2:]:
        raise ValueError('Bilateral and fused feature resolutions differ')
    return (outputs, residual_logit.detach(), fused_values[0], cues,
            support, valid_count, delta)


def _input_files_match(expected_hashes):
    return all(
        diag.sha(filename) == digest
        for filename, digest in (expected_hashes or {}).items())


def checked_forward(model, optimizer, sample, args, immutable_hashes=None):
    initial = diag.fingerprints(diag.model_snapshot(model, optimizer))
    rng_before = diag.fingerprints(rng_snapshot())
    runtime = diag.runtime_state(model)
    original_forward = cas_module.DepthNet.forward
    residual_head = model.sger_blocks[0].residual
    hooks_before = tuple(residual_head._forward_pre_hooks.items())
    if not _input_files_match(immutable_hashes):
        raise RuntimeError('Input file fingerprint mismatch before forward')
    try:
        result = capture_bilateral_forward(model, sample, args)
    finally:
        diag.restore_runtime(model, runtime)
    final = diag.fingerprints(diag.model_snapshot(model, optimizer))
    rng_after = diag.fingerprints(rng_snapshot())
    if cas_module.DepthNet.forward is not original_forward:
        raise RuntimeError('DepthNet forward function was not restored')
    hooks_after = tuple(residual_head._forward_pre_hooks.items())
    if (len(hooks_before) != len(hooks_after)
            or any(before_key != after_key or before_hook is not after_hook
                   for (before_key, before_hook), (after_key, after_hook)
                   in zip(hooks_before, hooks_after))):
        raise RuntimeError('Residual-head hook state was not restored')
    if initial != final:
        raise RuntimeError('Bilateral probe mutated model or optimizer')
    if rng_before != rng_after:
        raise RuntimeError('Bilateral probe mutated RNG state')
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError('Bilateral probe populated parameter .grad')
    if any(module.training for module in model.modules()):
        raise RuntimeError('Bilateral probe changed eval mode')
    if not _input_files_match(immutable_hashes):
        raise RuntimeError('Input file fingerprint mismatch after forward')
    return result


def group_channel_names(base_channels):
    base_names = base_probe.channel_names(base_channels)
    return {
        'base': base_names,
        'bilateral_only': list(BILATERAL_CUE_NAMES),
        'base_plus_bilateral': base_names + list(BILATERAL_CUE_NAMES),
    }


def sampled_feature_groups(fused, bilateral, positive, negative, support,
                           maximum):
    if (bilateral.dim() != 4
            or bilateral.size(1) != len(BILATERAL_CUE_NAMES)
            or bilateral.shape[0] != fused.shape[0]
            or bilateral.shape[-2:] != fused.shape[-2:]
            or support.shape != positive.shape):
        raise ValueError('Unexpected bilateral-cue or support shape')
    positive_supported = positive & support
    negative_supported = negative & support
    base_features, labels, sampling, indices = (
        base_probe.balanced_feature_sample(
            fused, positive_supported, negative_supported, maximum))
    bilateral_flat = bilateral.permute(0, 2, 3, 1).reshape(
        -1, bilateral.size(1))
    bilateral_features = bilateral_flat[indices].detach().cpu().double()
    if not bool(torch.isfinite(bilateral_features).all()):
        raise RuntimeError('Nonfinite sampled bilateral features')
    groups = {
        'base': base_features,
        'bilateral_only': bilateral_features,
        'base_plus_bilateral': torch.cat([
            base_features, bilateral_features], dim=1),
    }
    sampling.update({
        'available_positive_before_support': int(positive.sum()),
        'available_negative_before_support': int(negative.sum()),
        'bilateral_supported_only': True,
    })
    return groups, labels, sampling, indices


def feature_maps(fused, bilateral):
    return {
        'base': fused,
        'bilateral_only': bilateral,
        'base_plus_bilateral': torch.cat([fused, bilateral], dim=1),
    }


def empty_support_accumulator(max_views):
    return {
        'all_total': 0, 'all_supported': 0,
        'positive_total': 0, 'positive_supported': 0,
        'negative_total': 0, 'negative_supported': 0,
        'valid_view_count_sum': 0,
        'valid_view_count_histogram': [0] * (int(max_views) + 1),
    }


def update_support_accumulator(row, support, positive, negative, valid_count):
    if support.shape != positive.shape or support.shape != valid_count.shape:
        raise ValueError('Support-statistic shapes differ')
    row['all_total'] += support.numel()
    row['all_supported'] += int(support.sum())
    for label, mask in (('positive', positive), ('negative', negative)):
        row[label + '_total'] += int(mask.sum())
        row[label + '_supported'] += int((mask & support).sum())
    row['valid_view_count_sum'] += int(valid_count.sum())
    histogram = torch.bincount(
        valid_count.detach().reshape(-1).cpu().long(),
        minlength=len(row['valid_view_count_histogram']))
    if histogram.numel() > len(row['valid_view_count_histogram']):
        raise ValueError('Valid source-view count exceeds configured views')
    for index, value in enumerate(histogram.tolist()):
        row['valid_view_count_histogram'][index] += int(value)


def _histogram_quantile(histogram, quantile):
    total = sum(histogram)
    if total <= 0:
        raise ValueError('Cannot summarize an empty view-count histogram')
    target = int(math.ceil(float(quantile) * total))
    target = max(target, 1)
    cumulative = 0
    for index, count in enumerate(histogram):
        cumulative += count
        if cumulative >= target:
            return index
    return len(histogram) - 1


def summarize_support(row):
    def ratio(prefix):
        total = row[prefix + '_total']
        if not total:
            raise ValueError('Support summary has an empty {} set'.format(
                prefix))
        return row[prefix + '_supported'] / total

    histogram = row['valid_view_count_histogram']
    total = row['all_total']
    nonzero = [index for index, count in enumerate(histogram) if count]
    result = dict(row)
    result.update({
        'all_ratio': ratio('all'),
        'positive_ratio': ratio('positive'),
        'negative_ratio': ratio('negative'),
        'positive_negative_ratio_gap': abs(
            ratio('positive') - ratio('negative')),
        'valid_source_views': {
            'mean': row['valid_view_count_sum'] / total,
            'min': min(nonzero),
            'p25': _histogram_quantile(histogram, 0.25),
            'median': _histogram_quantile(histogram, 0.50),
            'p75': _histogram_quantile(histogram, 0.75),
            'p90': _histogram_quantile(histogram, 0.90),
            'max': max(nonzero),
            'histogram': {
                str(index): count for index, count in enumerate(histogram)
            },
        },
    })
    return result


def empty_cue_accumulator():
    return {
        label: {
            'count': 0,
            'sum': [0.0] * len(BILATERAL_CUE_NAMES),
            'square_sum': [0.0] * len(BILATERAL_CUE_NAMES),
            'zero_count': [0] * len(BILATERAL_CUE_NAMES),
            'finite_count': [0] * len(BILATERAL_CUE_NAMES),
        }
        for label in ('all', 'positive', 'negative')
    }


def _update_cue_row(row, values):
    values = values.detach().cpu().double()
    if values.dim() != 2 or values.size(1) != len(BILATERAL_CUE_NAMES):
        raise ValueError('Expected cue matrix shaped (N,4)')
    row['count'] += values.size(0)
    for index in range(values.size(1)):
        column = values[:, index]
        finite = torch.isfinite(column)
        row['finite_count'][index] += int(finite.sum())
        if bool(finite.any()):
            selected = column[finite]
            row['sum'][index] += float(selected.sum())
            row['square_sum'][index] += float(selected.square().sum())
            row['zero_count'][index] += int((selected == 0).sum())


def update_cue_accumulator(row, cues, support, positive, negative):
    flat = cues.permute(0, 2, 3, 1).reshape(-1, cues.size(1))
    masks = {
        'all': support,
        'positive': support & positive,
        'negative': support & negative,
    }
    for label, mask in masks.items():
        _update_cue_row(row[label], flat[mask.reshape(-1)])


def update_sampled_cue_accumulator(row, values, labels):
    _update_cue_row(row['all'], values)
    _update_cue_row(row['positive'], values[labels > 0])
    _update_cue_row(row['negative'], values[labels < 0])


def summarize_cues(row):
    result = {}
    for label, values in row.items():
        if values['count'] <= 0:
            raise ValueError('Cue summary has an empty {} set'.format(label))
        channels = {}
        for index, name in enumerate(BILATERAL_CUE_NAMES):
            finite_count = values['finite_count'][index]
            mean = values['sum'][index] / max(finite_count, 1)
            variance = max(
                values['square_sum'][index] / max(finite_count, 1)
                - mean * mean, 0.0)
            channels[name] = {
                'mean': mean,
                'std': variance ** 0.5,
                'finite_ratio': finite_count / values['count'],
                'zero_ratio': values['zero_count'][index]
                / max(finite_count, 1),
            }
        result[label] = {'count': values['count'], 'channels': channels}
    result['positive_minus_negative_mean'] = {
        name: (result['positive']['channels'][name]['mean']
               - result['negative']['channels'][name]['mean'])
        for name in BILATERAL_CUE_NAMES
    }
    return result


def group_decision(validation, group):
    pooled = validation['pooled'][group]
    scenes = {
        name: values[group]
        for name, values in validation['scenes'].items()
    }
    positive_separation = (
        pooled['score_mean_separation'] > 0.0
        and all(row['score_mean_separation'] > 0.0
                for row in scenes.values()))
    passed = (
        pooled['balanced_accuracy']
        >= base_probe.POOLED_BALANCED_ACCURACY_GATE
        and all(row['balanced_accuracy']
                >= base_probe.PER_SCENE_BALANCED_ACCURACY_GATE
                for row in scenes.values())
        and positive_separation)
    return {
        'group': group,
        'cross_scene_linear_signal_established': passed,
        'pooled_balanced_accuracy_gate':
            base_probe.POOLED_BALANCED_ACCURACY_GATE,
        'per_scene_balanced_accuracy_gate':
            base_probe.PER_SCENE_BALANCED_ACCURACY_GATE,
        'all_scene_score_separations_positive': positive_separation,
    }


def primary_decision(validation, support):
    groups = {
        name: group_decision(validation, name)
        for name in GROUP_NAMES
    }
    pooled_delta = (
        validation['pooled'][PRIMARY_GROUP]['balanced_accuracy']
        - validation['pooled']['base']['balanced_accuracy'])
    scene_deltas = {
        scene: (values[PRIMARY_GROUP]['balanced_accuracy']
                - values['base']['balanced_accuracy'])
        for scene, values in validation['scenes'].items()
    }
    support_rows = [support['train'], support['validation']['pooled']]
    support_rows.extend(support['validation']['scenes'].values())
    support_passed = all(
        row['all_ratio'] >= SUPPORT_RATIO_GATE
        and row['positive_negative_ratio_gap'] <= SUPPORT_CLASS_GAP_GATE
        for row in support_rows)
    relative_passed = (
        pooled_delta >= POOLED_DELTA_GATE
        and all(value >= 0.0 for value in scene_deltas.values()))
    passed = (
        groups[PRIMARY_GROUP]['cross_scene_linear_signal_established']
        and relative_passed and support_passed)
    return {
        'primary_group': PRIMARY_GROUP,
        'bilateral_matching_cues_established': passed,
        'groups': groups,
        'pooled_balanced_accuracy_delta_vs_base': pooled_delta,
        'scene_balanced_accuracy_delta_vs_base': scene_deltas,
        'pooled_delta_gate': POOLED_DELTA_GATE,
        'all_scene_deltas_nonnegative': all(
            value >= 0.0 for value in scene_deltas.values()),
        'relative_improvement_passed': relative_passed,
        'support_gate': SUPPORT_RATIO_GATE,
        'support_class_gap_gate': SUPPORT_CLASS_GAP_GATE,
        'support_passed': support_passed,
        'next_step': (
            'design an isolated Stage3 direction-times-magnitude SGER '
            'residual head and retain raw-versus-refined ablations'
            if passed else
            'do not change production SGER or tune delta/Ridge/threshold; '
            'inspect signs, support, view conflicts and scene shift'),
        'scope': (
            'predeclared conservative engineering gate, not a formal '
            'statistical significance test; pixels within a scene are '
            'correlated'),
    }


def _empty_group_accumulators():
    return {
        name: base_probe.empty_binary_accumulator()
        for name in GROUP_NAMES
    }


def _summarize_group_accumulators(rows):
    return {
        name: base_probe.summarize_binary(rows[name])
        for name in GROUP_NAMES
    }


def preflight(cli, source, probe_root, gradient_root, feature_root,
              signed_root):
    args, plan, trainset, valset = base_probe.build_datasets(source)
    model, optimizer, configured_args, path = gradient_audit.load_snapshot(
        probe_root, 'shared_step0', 'direction_reference', 0,
        source, torch.device('cpu'))
    base_probe.require_configured_snapshot_arguments(
        source['arguments'], configured_args)
    train_item = trainset[plan['train_indices'][0]]
    val_item = valset[plan['val_indices'][0]]
    for label, item in (('train', train_item), ('validation', val_item)):
        image_views = item['imgs'].shape[0]
        projection_views = item['proj_matrices']['stage3'].shape[0]
        if image_views != projection_views or image_views != args.nviews:
            raise ValueError(
                '{} feature/projection view-count mismatch'.format(label))
    channels = model.sger_blocks[0].residual.body[0].in_channels
    names = group_channel_names(channels)
    expected = {'base': 26, 'bilateral_only': 4,
                'base_plus_bilateral': 30}
    actual = {name: len(values) for name, values in names.items()}
    if actual != expected:
        raise ValueError('Unexpected feature group sizes: {}'.format(actual))
    if not cli.output_dir:
        output_parent = Path('outputs/experiment15_d')
    else:
        output_parent = Path(cli.output_dir).parent
    if output_parent.exists() and not output_parent.is_dir():
        raise ValueError('Output parent is not a directory')
    print('Preflight passed:', path,
          'feature groups:', ', '.join(
              '{}={}'.format(key, len(value)) for key, value in names.items()),
          'views:', args.nviews, 'train/validation samples: 50/6',
          'device for full probe:', cli.device,
          'upstream:', gradient_root, feature_root, signed_root, flush=True)
    del model, optimizer


def run(cli):
    probe_root = Path(cli.probe_dir)
    gradient_root = Path(cli.gradient_audit_dir)
    feature_root = Path(cli.feature_probe_dir)
    signed_root = Path(cli.signed_probe_dir)
    source, _ = gradient_audit.load_source(probe_root)
    _, _, gradient_paths = base_probe.load_gradient_audit(
        gradient_root, probe_root)
    _, _, feature_paths = signed_probe.load_feature_probe(
        feature_root, probe_root, gradient_root)
    _, _, signed_paths = load_signed_probe(
        signed_root, probe_root, gradient_root, feature_root)
    if cli.preflight_only:
        preflight(cli, source, probe_root, gradient_root, feature_root,
                  signed_root)
        return

    args, plan, trainset, valset = base_probe.build_datasets(source)
    snapshot = probe_root / 'direction_reference/step_0000.ckpt'
    fingerprint = (
        probe_root / 'direction_reference/fingerprints/initial.json')
    inputs = [
        probe_root / 'manifest.json', probe_root / 'summary.json',
        snapshot, fingerprint,
    ] + list(gradient_paths) + list(feature_paths) + list(signed_paths)
    input_hashes = {str(path): diag.sha(path) for path in inputs}

    if not cli.strict_determinism:
        raise ValueError(
            'The bilateral matching experiment requires --strict_determinism')
    if cli.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if (cli.device == 'cuda'
            and os.environ.get('CUBLAS_WORKSPACE_CONFIG')
            not in (':4096:8', ':16:8')):
        raise ValueError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8')
    torch.use_deterministic_algorithms(True)
    device = torch.device(cli.device)
    diag.seed(source['arguments']['seed'])
    model, optimizer, snapshot_args, _ = gradient_audit.load_snapshot(
        probe_root, 'shared_step0', 'direction_reference', 0,
        source, device)
    argument_compatibility = (
        base_probe.require_configured_snapshot_arguments(
            source['arguments'], snapshot_args))
    base_channels = model.sger_blocks[0].residual.body[0].in_channels
    names = group_channel_names(base_channels)
    source_views = int(args.nviews) - 1

    output = Path(cli.output_dir or (
        'outputs/experiment15_d/bilateral_matching_cue_separability_'
        + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        'status': 'running',
        'command': sys.argv,
        'source_probe': str(probe_root),
        'source_gradient_audit': str(gradient_root),
        'source_feature_separability': str(feature_root),
        'source_signed_probability_probe': str(signed_root),
        'input_sha256': input_hashes,
        'source_sha256': {
            path: diag.sha(path) for path in diag.SOURCES + [
                'experiment15_momentum_diagnose.py',
                'experiment15_residual_direction_probe.py',
                'experiment15_residual_gradient_decomposition.py',
                'experiment15_residual_feature_separability.py',
                'experiment15_signed_probability_cue_separability.py',
                __file__]
        },
        'arguments': source['arguments'],
        'effective_case': source['effective_case'],
        'sample_plan': plan,
        'snapshot': str(snapshot),
        'snapshot_scope': 'shared step0 before direction-BCE intervention',
        'snapshot_argument_compatibility': argument_compatibility,
        'device': cli.device,
        'strict_determinism': True,
        'samples_per_class_per_train_batch':
            base_probe.SAMPLES_PER_CLASS_PER_BATCH,
        'ridge': base_probe.RIDGE,
        'feature_groups': names,
        'primary_group': PRIMARY_GROUP,
        'bilateral_cue_definitions': {
            'delta': (
                'per-pixel median adjacent Stage3 depth-hypothesis '
                'spacing; all adjacent spacings must agree'),
            'advantage': (
                'channel-mean normalized-feature cost at raw-delta minus '
                'cost at raw+delta; positive predicts deeper'),
            BILATERAL_CUE_NAMES[0]: 'valid-source-view mean advantage',
            BILATERAL_CUE_NAMES[1]: 'valid-source-view median advantage',
            BILATERAL_CUE_NAMES[2]:
                '(positive votes - negative votes) / valid view count',
            BILATERAL_CUE_NAMES[3]:
                'mean advantage / (population std + 1e-6)',
            'support': (
                'both candidate depths positive/finite and at least two '
                'source views valid on both all-one-image warps'),
        },
        'decision_gates': {
            'primary_group': PRIMARY_GROUP,
            'pooled_balanced_accuracy':
                base_probe.POOLED_BALANCED_ACCURACY_GATE,
            'each_scene_balanced_accuracy':
                base_probe.PER_SCENE_BALANCED_ACCURACY_GATE,
            'each_scene_score_mean_separation': 'strictly positive',
            'pooled_balanced_accuracy_delta_vs_base': POOLED_DELTA_GATE,
            'each_scene_delta_vs_base': 'nonnegative',
            'train_validation_scene_support_ratio': SUPPORT_RATIO_GATE,
            'positive_negative_support_ratio_gap_max':
                SUPPORT_CLASS_GAP_GATE,
        },
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'boundary': (
            'Bilateral cues use only Stage3 features, cameras, raw depth '
            'and actual depth hypotheses. GT defines external labels only. '
            'No backward, optimizer or scheduler operation is performed.'),
        'interpretation_boundary': (
            'The primary claim concerns a pointwise linear probe on fixed '
            'samples. Failure does not exclude nonlinear or spatial-context '
            'integration of conflicting multi-view votes.'),
    }
    summary = {
        'status': 'running', 'checks_passed': False,
        'train_sampling': [], 'train': {}, 'validation': {},
    }
    diag.dump(output / 'manifest.json', manifest)
    diag.dump(output / 'summary.json', summary)
    original_depth_forward = cas_module.DepthNet.forward
    try:
        train_features = {name: [] for name in GROUP_NAMES}
        train_labels = []
        sampled_head_scores = []
        train_support = empty_support_accumulator(source_views)
        train_cues = empty_cue_accumulator()
        train_loader = DataLoader(
            trainset,
            batch_sampler=[plan['train_indices'][index:index + 2]
                           for index in range(0, 50, 2)],
            num_workers=0)
        for batch_index, sample in enumerate(train_loader):
            sample = diag._move_to_device(sample, device)
            (outputs, residual_logit, fused, bilateral, support,
             valid_count, delta) = checked_forward(
                model, optimizer, sample, args, input_hashes)
            positive, negative = base_probe.direction_labels(
                outputs, sample, args)
            update_support_accumulator(
                train_support, support, positive, negative, valid_count)
            groups, labels, sampling, sampled_indices = (
                sampled_feature_groups(
                    fused, bilateral, positive, negative, support,
                    base_probe.SAMPLES_PER_CLASS_PER_BATCH))
            update_sampled_cue_accumulator(
                train_cues, groups['bilateral_only'], labels)
            sampling['batch'] = batch_index
            sampling['indices'] = plan['train_indices'][
                2 * batch_index:2 * batch_index + 2]
            sampling['delta_min'] = float(delta.min())
            sampling['delta_max'] = float(delta.max())
            summary['train_sampling'].append(sampling)
            for name in GROUP_NAMES:
                train_features[name].append(groups[name])
            train_labels.append(labels)
            sampled_head_scores.append(
                residual_logit.reshape(-1)[sampled_indices]
                .detach().cpu().double())
            print('captured train batch', batch_index + 1, '/25',
                  'sampled', sampling['sampled_total'], flush=True)

        labels = torch.cat(train_labels, dim=0)
        matrices = {
            name: torch.cat(train_features[name], dim=0)
            for name in GROUP_NAMES
        }
        probes = {
            name: base_probe.fit_ridge_probe(
                matrices[name], labels, ridge=base_probe.RIDGE)
            for name in GROUP_NAMES
        }
        sampled_positive = labels > 0
        sampled_negative = labels < 0
        train_rows = {}
        for name in GROUP_NAMES:
            accumulator = base_probe.empty_binary_accumulator()
            base_probe.update_binary_accumulator(
                accumulator,
                base_probe.matrix_probe_scores(
                    matrices[name], probes[name]),
                sampled_positive, sampled_negative)
            train_rows[name] = base_probe.summarize_binary(accumulator)
        head_accumulator = base_probe.empty_binary_accumulator()
        base_probe.update_binary_accumulator(
            head_accumulator, torch.cat(sampled_head_scores, dim=0),
            sampled_positive, sampled_negative)
        summary['train'] = {
            'groups': train_rows,
            'production_residual_logit':
                base_probe.summarize_binary(head_accumulator),
            'support': summarize_support(train_support),
            'bilateral_cues': summarize_cues(train_cues),
            'evaluation_scope': (
                'same deterministic, support-filtered balanced pixels used '
                'to fit all three external probes'),
        }
        summary['probes'] = {
            name: base_probe.serializable_probe(probes[name], names[name])
            for name in GROUP_NAMES
        }
        del (matrices, labels, train_features, train_labels,
             sampled_head_scores)

        pooled = _empty_group_accumulators()
        pooled_head = base_probe.empty_binary_accumulator()
        pooled_support = empty_support_accumulator(source_views)
        pooled_cues = empty_cue_accumulator()
        scenes = {}
        val_loader = DataLoader(
            valset,
            batch_sampler=[plan['val_indices'][index:index + 2]
                           for index in range(0, 6, 2)],
            num_workers=0)
        for batch_index, sample in enumerate(val_loader):
            sample = diag._move_to_device(sample, device)
            (outputs, residual_logit, fused, bilateral, support,
             valid_count, _) = checked_forward(
                model, optimizer, sample, args, input_hashes)
            positive, negative = base_probe.direction_labels(
                outputs, sample, args)
            supported_positive = positive & support
            supported_negative = negative & support
            maps = feature_maps(fused, bilateral)
            scene_names = {meta[0] for meta in plan['val_metas'][
                2 * batch_index:2 * batch_index + 2]}
            if len(scene_names) != 1:
                raise ValueError('Validation batch crosses scenes')
            scene = next(iter(scene_names))
            scene_rows = scenes.setdefault(scene, {
                'groups': _empty_group_accumulators(),
                'production_residual_logit':
                    base_probe.empty_binary_accumulator(),
                'support': empty_support_accumulator(source_views),
                'cues': empty_cue_accumulator(),
            })
            for accumulator in (pooled_support, scene_rows['support']):
                update_support_accumulator(
                    accumulator, support, positive, negative, valid_count)
            for accumulator in (pooled_cues, scene_rows['cues']):
                update_cue_accumulator(
                    accumulator, bilateral, support, positive, negative)
            for name in GROUP_NAMES:
                scores = base_probe.probe_scores(
                    maps[name], probes[name])
                for accumulator in (
                        pooled[name], scene_rows['groups'][name]):
                    base_probe.update_binary_accumulator(
                        accumulator, scores,
                        supported_positive, supported_negative)
            for accumulator in (
                    pooled_head,
                    scene_rows['production_residual_logit']):
                base_probe.update_binary_accumulator(
                    accumulator, residual_logit,
                    supported_positive, supported_negative)
            print('evaluated validation scene', scene, flush=True)

        summary['validation'] = {
            'pooled': dict(
                _summarize_group_accumulators(pooled),
                production_residual_logit=
                    base_probe.summarize_binary(pooled_head),
                support=summarize_support(pooled_support),
                bilateral_cues=summarize_cues(pooled_cues)),
            'scenes': {
                scene: dict(
                    _summarize_group_accumulators(values['groups']),
                    production_residual_logit=base_probe.summarize_binary(
                        values['production_residual_logit']),
                    support=summarize_support(values['support']),
                    bilateral_cues=summarize_cues(values['cues']))
                for scene, values in scenes.items()
            },
        }
        decision_support = {
            'train': summary['train']['support'],
            'validation': {
                'pooled': summary['validation']['pooled']['support'],
                'scenes': {
                    scene: values['support']
                    for scene, values in summary['validation'][
                        'scenes'].items()
                },
            },
        }
        summary['decision'] = primary_decision(
            summary['validation'], decision_support)
        summary['support'] = decision_support
        summary['model_optimizer_bn_mode_unchanged'] = True
        summary['rng_state_restored_per_forward'] = True
        summary['depthnet_forward_restored_per_forward'] = True
        summary['residual_hook_removed_per_forward'] = True
        summary['input_files_unchanged_per_forward'] = True
        summary['checks_passed'] = True
        manifest['status'] = summary['status'] = 'completed'
    except BaseException as error:
        manifest['status'] = summary['status'] = (
            'interrupted' if isinstance(error, KeyboardInterrupt)
            else 'failed')
        summary['error'] = repr(error)
        raise
    finally:
        summary['depthnet_forward_restored_finally'] = (
            cas_module.DepthNet.forward is original_depth_forward)
        summary['input_files_unchanged'] = all(
            diag.sha(path) == digest
            for path, digest in input_hashes.items())
        if (not summary['input_files_unchanged']
                or not summary['depthnet_forward_restored_finally']):
            summary['checks_passed'] = False
            manifest['status'] = summary['status'] = 'failed'
        diag.dump(output / 'manifest.json', manifest)
        diag.dump(output / 'summary.json', summary)
    if not summary['checks_passed']:
        raise RuntimeError(
            'Bilateral matching-cue separability integrity check failed')
    print('Bilateral matching-cue separability probe completed:',
          output / 'summary.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe_dir', required=True)
    parser.add_argument('--gradient_audit_dir', required=True)
    parser.add_argument('--feature_probe_dir', required=True)
    parser.add_argument('--signed_probe_dir', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--output_dir')
    parser.add_argument('--strict_determinism', action='store_true')
    parser.add_argument('--preflight_only', action='store_true')
    cli = parser.parse_args()
    torch.set_num_threads(1)
    run(cli)


if __name__ == '__main__':
    main()
