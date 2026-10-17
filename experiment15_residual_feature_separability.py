"""Read-only cross-scene separability probe for SGER residual direction."""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import experiment15_momentum_diagnose as diag
import experiment15_residual_gradient_decomposition as gradient_audit
from experiment15_d_diagnose import rng_snapshot
from models.module import bounded_residual_target


SAMPLES_PER_CLASS_PER_BATCH = 4096
RIDGE = 1e-3
POOLED_BALANCED_ACCURACY_GATE = 0.60
PER_SCENE_BALANCED_ACCURACY_GATE = 0.55
FIXED_CUE_NAMES = (
    'depth_normalized',
    'normal_pred_x', 'normal_pred_y', 'normal_pred_z',
    'normal_depth_x', 'normal_depth_y', 'normal_depth_z',
    'normal_disagreement', 'photometric_confidence', 'image_edge',
    'depth_edge', 'curvature', 'ref_image_r', 'ref_image_g',
    'ref_image_b',
)
UNCERTAINTY_NAMES = (
    'probability_entropy', 'depth_variance_normalized',
    'top1_top2_margin',
)


def load_gradient_audit(root, probe_root):
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
            != probe_root.resolve()):
        raise ValueError(
            'Requires the completed strict gradient audit for this probe')
    checked = 0
    for snapshot in summary.get('results', {}).values():
        for split in snapshot.values():
            row = split['summary']
            if (row['balanced_direction_both_directions_desired_count'] != 0
                    or row[
                        'balanced_experimental_both_directions_desired_count']
                    != 0):
                raise ValueError(
                    'Balanced counterfactual is not uniformly negative')
            checked += row['batch_count']
    if checked != 18:
        raise ValueError('Expected eighteen audited snapshot/batch rows')
    return manifest, summary, (manifest_path, summary_path)


def channel_names(channels):
    projected = channels - len(FIXED_CUE_NAMES) - len(UNCERTAINTY_NAMES)
    if projected <= 0:
        raise ValueError('Unexpected fused feature channel count: {}'.format(
            channels))
    return (list(FIXED_CUE_NAMES)
            + ['projected_feature_{}'.format(index)
               for index in range(projected)]
            + list(UNCERTAINTY_NAMES))


def require_configured_snapshot_arguments(source_arguments,
                                          configured_arguments):
    """Accept only the loader's explicit, loss-only compatibility rewrite."""
    expected = dict(source_arguments)
    expected['sger_residual_final_activation'] = 'leaky_relu'
    expected['sger_residual_negative_slope'] = 0.01
    expected['gate_bce_mode'] = 'legacy_clamp'
    actual = vars(configured_arguments)
    differing = {
        key: {'expected': expected.get(key), 'actual': actual.get(key)}
        for key in sorted(set(expected) | set(actual))
        if (key not in expected or key not in actual
            or expected[key] != actual[key])
    }
    if differing:
        raise ValueError(
            'Dataset and configured snapshot arguments differ: {}'.format(
                differing))
    return {
        'source_gate_bce_mode': source_arguments.get('gate_bce_mode'),
        'configured_gate_bce_mode': actual.get('gate_bce_mode'),
        'accepted_difference': (
            'load_snapshot configures legacy_clamp for historical loss '
            'compatibility; this read-only forward does not compute a gate '
            'BCE loss'),
    }


def capture_fused_forward(model, sample, args):
    fused_values = []

    def capture(module, inputs):
        value = inputs[0]
        if value.dim() != 4:
            raise ValueError('Expected fused features shaped (B,C,H,W)')
        fused_values.append(value.detach())

    handle = model.sger_blocks[0].residual.register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            outputs, _, residual_logit = (
                gradient_audit.direction_probe.forward_with_residual_logit(
                    model, sample, args))
    finally:
        handle.remove()
    if len(fused_values) != 1:
        raise RuntimeError('Expected exactly one fused SGER residual input')
    return outputs, residual_logit.detach(), fused_values[0]


def direction_labels(outputs, sample, args):
    stage = outputs['stage3']
    gt = sample['depth']['stage3']
    valid = sample['mask']['stage3'] > 0.5
    target = bounded_residual_target(
        stage['depth_raw'], gt, stage['depth_interval'],
        args.residual_target_ratio)
    positive = valid & (target > 0)
    negative = valid & (target < 0)
    if not bool(positive.any()) or not bool(negative.any()):
        raise ValueError('Batch must contain both residual directions')
    return positive, negative


def evenly_spaced_indices(mask, count):
    available = torch.nonzero(mask.reshape(-1), as_tuple=False)[:, 0]
    count = min(int(count), int(available.numel()))
    if count <= 0:
        raise ValueError('Cannot sample an empty class')
    if count == int(available.numel()):
        return available
    positions = torch.linspace(
        0, available.numel() - 1, steps=count,
        device=available.device, dtype=torch.float64).round().long()
    selected = available[positions]
    if int(torch.unique(selected).numel()) != count:
        raise RuntimeError('Deterministic sampling produced duplicates')
    return selected


def balanced_feature_sample(fused, positive, negative, maximum):
    if fused.shape[0] != positive.shape[0] or fused.shape[-2:] != positive.shape[-2:]:
        raise ValueError('Feature and direction-label shapes differ')
    per_class = min(int(positive.sum()), int(negative.sum()), int(maximum))
    positive_indices = evenly_spaced_indices(positive, per_class)
    negative_indices = evenly_spaced_indices(negative, per_class)
    flat = fused.permute(0, 2, 3, 1).reshape(-1, fused.size(1))
    features = torch.cat([
        flat[positive_indices], flat[negative_indices]], dim=0)
    labels = torch.cat([
        torch.ones(per_class, device=fused.device, dtype=fused.dtype),
        -torch.ones(per_class, device=fused.device, dtype=fused.dtype),
    ])
    if not bool(torch.isfinite(features).all()):
        raise RuntimeError('Nonfinite sampled fused features')
    indices = torch.cat([positive_indices, negative_indices])
    return (features.detach().cpu().double(), labels.cpu().double(), {
        'available_positive': int(positive.sum()),
        'available_negative': int(negative.sum()),
        'sampled_per_class': per_class,
        'sampled_total': 2 * per_class,
        'sampling': 'evenly spaced flattened indices within each class',
    }, indices)


def fit_ridge_probe(features, labels, ridge=RIDGE):
    if (features.dim() != 2 or labels.dim() != 1
            or features.size(0) != labels.numel()):
        raise ValueError('Invalid probe training shapes')
    if int((labels > 0).sum()) != int((labels < 0).sum()):
        raise ValueError('Ridge probe requires balanced labels')
    mean = features.mean(0)
    scale = features.std(0, unbiased=False)
    floor = max(float(scale.max()) * 1e-8, 1e-12)
    active = scale > floor
    safe_scale = torch.where(active, scale, torch.ones_like(scale))
    normalized = (features - mean) / safe_scale
    normalized[:, ~active] = 0.0
    design = torch.cat([
        normalized,
        torch.ones(normalized.size(0), 1, dtype=normalized.dtype),
    ], dim=1)
    gram = design.t().matmul(design) / design.size(0)
    penalty = torch.eye(gram.size(0), dtype=gram.dtype) * float(ridge)
    penalty[-1, -1] = 0.0
    rhs = design.t().matmul(labels) / design.size(0)
    coefficient = torch.linalg.solve(gram + penalty, rhs)
    if not bool(torch.isfinite(coefficient).all()):
        raise RuntimeError('Nonfinite ridge-probe solution')
    return {
        'mean': mean,
        'scale': safe_scale,
        'active': active,
        'weight': coefficient[:-1],
        'bias': coefficient[-1],
        'ridge': float(ridge),
        'train_count': int(labels.numel()),
    }


def probe_scores(fused, probe):
    mean = probe['mean'].to(device=fused.device, dtype=fused.dtype)
    scale = probe['scale'].to(device=fused.device, dtype=fused.dtype)
    active = probe['active'].to(device=fused.device)
    weight = probe['weight'].to(device=fused.device, dtype=fused.dtype)
    normalized = (fused.permute(0, 2, 3, 1) - mean) / scale
    normalized[..., ~active] = 0.0
    return normalized.matmul(weight) + float(probe['bias'])


def matrix_probe_scores(features, probe):
    normalized = (features - probe['mean']) / probe['scale']
    normalized[:, ~probe['active']] = 0.0
    return normalized.matmul(probe['weight']) + probe['bias']


def empty_binary_accumulator():
    return {
        'positive_count': 0, 'negative_count': 0,
        'true_positive': 0, 'false_negative': 0,
        'true_negative': 0, 'false_positive': 0,
        'positive_score_sum': 0.0, 'positive_score_sq_sum': 0.0,
        'negative_score_sum': 0.0, 'negative_score_sq_sum': 0.0,
    }


def update_binary_accumulator(row, scores, positive, negative):
    for label, mask in (('positive', positive), ('negative', negative)):
        selected = scores[mask].detach().double()
        row[label + '_count'] += int(selected.numel())
        row[label + '_score_sum'] += float(selected.sum())
        row[label + '_score_sq_sum'] += float(selected.square().sum())
    predicted = scores > 0
    row['true_positive'] += int((predicted & positive).sum())
    row['false_negative'] += int(((~predicted) & positive).sum())
    row['true_negative'] += int(((~predicted) & negative).sum())
    row['false_positive'] += int((predicted & negative).sum())


def summarize_binary(row):
    positive = row['positive_count']
    negative = row['negative_count']
    if not positive or not negative:
        raise ValueError('Binary summary requires both classes')

    def score_stats(label, count):
        mean = row[label + '_score_sum'] / count
        variance = max(
            row[label + '_score_sq_sum'] / count - mean * mean, 0.0)
        return {'mean': mean, 'std': variance ** 0.5}

    tpr = row['true_positive'] / positive
    tnr = row['true_negative'] / negative
    total = positive + negative
    result = dict(row)
    result.update({
        'positive_fraction': positive / total,
        'true_positive_rate': tpr,
        'true_negative_rate': tnr,
        'balanced_accuracy': 0.5 * (tpr + tnr),
        'accuracy': (
            row['true_positive'] + row['true_negative']) / total,
        'positive_scores': score_stats('positive', positive),
        'negative_scores': score_stats('negative', negative),
        'threshold': 0.0,
    })
    result['score_mean_separation'] = (
        result['positive_scores']['mean']
        - result['negative_scores']['mean'])
    return result


def serializable_probe(probe, names):
    weight = probe['weight'].tolist()
    ranked = sorted(
        ({'channel': name, 'standardized_weight': value}
         for name, value in zip(names, weight)),
        key=lambda row: abs(row['standardized_weight']), reverse=True)
    return {
        'ridge': probe['ridge'],
        'train_count': probe['train_count'],
        'bias': float(probe['bias']),
        'active_channel_count': int(probe['active'].sum()),
        'channel_names': names,
        'mean': probe['mean'].tolist(),
        'scale': probe['scale'].tolist(),
        'standardized_weight': weight,
        'ranked_absolute_weights': ranked,
        'fit': (
            'closed-form ridge least-squares classifier on balanced sampled '
            'pixels; positive target is +1 and threshold is zero'),
    }


def direction_signal_decision(validation):
    pooled = validation['pooled']['ridge_probe']
    scenes = {
        name: values['ridge_probe']
        for name, values in validation['scenes'].items()
    }
    positive_separation = (
        pooled['score_mean_separation'] > 0.0
        and all(row['score_mean_separation'] > 0.0
                for row in scenes.values()))
    passed = (
        pooled['balanced_accuracy'] >= POOLED_BALANCED_ACCURACY_GATE
        and all(row['balanced_accuracy']
                >= PER_SCENE_BALANCED_ACCURACY_GATE
                for row in scenes.values())
        and positive_separation)
    return {
        'cross_scene_linear_signal_established': passed,
        'pooled_balanced_accuracy_gate':
            POOLED_BALANCED_ACCURACY_GATE,
        'per_scene_balanced_accuracy_gate':
            PER_SCENE_BALANCED_ACCURACY_GATE,
        'all_scene_score_separations_positive': positive_separation,
        'next_step': (
            'design and isolate a direction-times-magnitude SGER residual '
            'head, retaining raw-versus-refined ablations'
            if passed else
            'do not change to a factorized head yet; audit explicit signed '
            'probability-volume cues against the same cross-scene protocol'),
        'scope': (
            'predeclared conservative engineering gate, not a formal '
            'statistical significance test; pixels within a scene are '
            'correlated'),
    }


def checked_forward(model, optimizer, sample, args):
    initial = diag.fingerprints(diag.model_snapshot(model, optimizer))
    rng_before = diag.fingerprints(rng_snapshot())
    runtime = diag.runtime_state(model)
    try:
        outputs, residual_logit, fused = capture_fused_forward(
            model, sample, args)
    finally:
        diag.restore_runtime(model, runtime)
    final = diag.fingerprints(diag.model_snapshot(model, optimizer))
    rng_after = diag.fingerprints(rng_snapshot())
    if initial != final:
        raise RuntimeError('Feature probe mutated model or optimizer')
    if rng_before != rng_after:
        raise RuntimeError('Feature probe mutated RNG state')
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError('Feature probe populated parameter .grad')
    if any(module.training for module in model.modules()):
        raise RuntimeError('Feature probe changed eval mode')
    return outputs, residual_logit, fused


def build_datasets(source):
    args = argparse.Namespace(**source['arguments'])
    plan = source['sample_plan']
    dataset_type = diag.find_dataset_def(args.dataset)
    trainset = dataset_type(
        args.trainpath, args.trainlist, 'train', args.nviews,
        args.numdepth, args.interval_scale)
    valset = dataset_type(
        args.trainpath, plan['val_list'], 'val', args.nviews,
        args.numdepth, args.interval_scale)
    diag.validate_selection(
        trainset, plan['train_indices'], plan['train_metas'], 50)
    diag.validate_selection(
        valset, plan['val_indices'], plan['val_metas'], 6)
    return args, plan, trainset, valset


def run(cli):
    probe_root = Path(cli.probe_dir)
    gradient_root = Path(cli.gradient_audit_dir)
    source, _ = gradient_audit.load_source(probe_root)
    _, _, gradient_paths = load_gradient_audit(
        gradient_root, probe_root)
    args, plan, trainset, valset = build_datasets(source)
    snapshot = probe_root / 'direction_reference/step_0000.ckpt'
    fingerprint = probe_root / 'direction_reference/fingerprints/initial.json'
    inputs = [
        probe_root / 'manifest.json', probe_root / 'summary.json',
        snapshot, fingerprint,
    ] + list(gradient_paths)
    input_hashes = {str(path): diag.sha(path) for path in inputs}

    if cli.preflight_only:
        model, optimizer, configured_args, path = gradient_audit.load_snapshot(
            probe_root, 'shared_step0', 'direction_reference', 0,
            source, torch.device('cpu'))
        require_configured_snapshot_arguments(
            source['arguments'], configured_args)
        channels = model.sger_blocks[0].residual.body[0].in_channels
        print('Preflight passed:', path, 'fused channels:', channels,
              'train/validation samples: 50/6', flush=True)
        del model, optimizer
        return

    if cli.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if cli.strict_determinism:
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
    argument_compatibility = require_configured_snapshot_arguments(
        source['arguments'], snapshot_args)

    output = Path(cli.output_dir or (
        'outputs/experiment15_d/residual_feature_separability_'
        + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        'status': 'running',
        'command': sys.argv,
        'source_probe': str(probe_root),
        'source_gradient_audit': str(gradient_root),
        'input_sha256': input_hashes,
        'source_sha256': {
            path: diag.sha(path) for path in diag.SOURCES + [
                'experiment15_momentum_diagnose.py',
                'experiment15_residual_direction_probe.py',
                'experiment15_residual_gradient_decomposition.py', __file__]
        },
        'arguments': source['arguments'],
        'effective_case': source['effective_case'],
        'sample_plan': plan,
        'snapshot': str(snapshot),
        'snapshot_scope': 'shared step0 before direction-BCE intervention',
        'snapshot_argument_compatibility': argument_compatibility,
        'device': cli.device,
        'strict_determinism': cli.strict_determinism,
        'samples_per_class_per_train_batch':
            SAMPLES_PER_CLASS_PER_BATCH,
        'ridge': RIDGE,
        'decision_gates': {
            'pooled_balanced_accuracy':
                POOLED_BALANCED_ACCURACY_GATE,
            'each_scene_balanced_accuracy':
                PER_SCENE_BALANCED_ACCURACY_GATE,
            'each_scene_score_mean_separation': 'strictly positive',
        },
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'boundary': (
            'GT defines external direction labels only. Production model, '
            'optimizer, buffers and RNG are read-only; no probe output enters '
            'SGER inference.'),
        'interpretation_boundary': (
            'A cross-scene linear result above chance demonstrates usable '
            'direction information in current fused inputs. A chance result '
            'does not exclude all nonlinear or spatially contextual signals.'),
    }
    summary = {
        'status': 'running', 'checks_passed': False,
        'train_sampling': [], 'train': {}, 'validation': {},
    }
    diag.dump(output / 'manifest.json', manifest)
    diag.dump(output / 'summary.json', summary)
    try:
        train_features = []
        train_labels = []
        train_probe_metrics = empty_binary_accumulator()
        train_head_metrics = empty_binary_accumulator()
        train_loader = DataLoader(
            trainset,
            batch_sampler=[plan['train_indices'][index:index + 2]
                           for index in range(0, 50, 2)],
            num_workers=0)
        # Capture all sampled inputs before fitting; the probe sees no
        # validation labels or validation features during fitting.
        sampled_head_scores = []
        for batch_index, sample in enumerate(train_loader):
            sample = diag._move_to_device(sample, device)
            outputs, residual_logit, fused = checked_forward(
                model, optimizer, sample, args)
            positive, negative = direction_labels(outputs, sample, args)
            features, labels, sampling, sampled_indices = (
                balanced_feature_sample(
                fused, positive, negative,
                SAMPLES_PER_CLASS_PER_BATCH))
            sampling['batch'] = batch_index
            sampling['indices'] = plan['train_indices'][
                2 * batch_index:2 * batch_index + 2]
            summary['train_sampling'].append(sampling)
            train_features.append(features)
            train_labels.append(labels)
            sampled_head_scores.append(
                residual_logit.reshape(-1)[sampled_indices]
                .detach().cpu().double())
            print('captured train batch', batch_index + 1, '/25',
                  'sampled', sampling['sampled_total'], flush=True)

        features = torch.cat(train_features, dim=0)
        labels = torch.cat(train_labels, dim=0)
        head_scores = torch.cat(sampled_head_scores, dim=0)
        probe = fit_ridge_probe(features, labels)
        names = channel_names(features.size(1))
        sampled_positive = labels > 0
        sampled_negative = labels < 0
        update_binary_accumulator(
            train_probe_metrics, matrix_probe_scores(features, probe),
            sampled_positive, sampled_negative)
        update_binary_accumulator(
            train_head_metrics, head_scores,
            sampled_positive, sampled_negative)
        del (features, labels, head_scores, train_features, train_labels,
             sampled_head_scores)
        summary['train'] = {
            'ridge_probe': summarize_binary(train_probe_metrics),
            'production_residual_logit': summarize_binary(
                train_head_metrics),
            'evaluation_scope': (
                'same deterministic balanced sampled pixels used to fit '
                'the external probe'),
        }
        summary['probe'] = serializable_probe(probe, names)

        pooled_probe = empty_binary_accumulator()
        pooled_head = empty_binary_accumulator()
        scenes = {}
        val_loader = DataLoader(
            valset,
            batch_sampler=[plan['val_indices'][index:index + 2]
                           for index in range(0, 6, 2)],
            num_workers=0)
        for batch_index, sample in enumerate(val_loader):
            sample = diag._move_to_device(sample, device)
            outputs, residual_logit, fused = checked_forward(
                model, optimizer, sample, args)
            positive, negative = direction_labels(outputs, sample, args)
            ridge_scores = probe_scores(fused, probe)
            scene_names = {meta[0] for meta in plan['val_metas'][
                2 * batch_index:2 * batch_index + 2]}
            if len(scene_names) != 1:
                raise ValueError('Validation batch crosses scenes')
            scene = next(iter(scene_names))
            scene_rows = scenes.setdefault(scene, {
                'ridge_probe': empty_binary_accumulator(),
                'production_residual_logit': empty_binary_accumulator(),
            })
            for accumulator in (pooled_probe, scene_rows['ridge_probe']):
                update_binary_accumulator(
                    accumulator, ridge_scores, positive, negative)
            for accumulator in (pooled_head,
                                scene_rows['production_residual_logit']):
                update_binary_accumulator(
                    accumulator, residual_logit, positive, negative)
            print('evaluated validation scene', scene, flush=True)

        summary['validation'] = {
            'pooled': {
                'ridge_probe': summarize_binary(pooled_probe),
                'production_residual_logit': summarize_binary(pooled_head),
            },
            'scenes': {
                scene: {
                    name: summarize_binary(row)
                    for name, row in values.items()
                }
                for scene, values in scenes.items()
            },
        }
        summary['decision'] = direction_signal_decision(
            summary['validation'])
        summary['model_state_unchanged'] = True
        summary['rng_state_restored_per_forward'] = True
        summary['checks_passed'] = True
        manifest['status'] = summary['status'] = 'completed'
    except BaseException as error:
        manifest['status'] = summary['status'] = (
            'interrupted' if isinstance(error, KeyboardInterrupt)
            else 'failed')
        summary['error'] = repr(error)
        raise
    finally:
        summary['input_files_unchanged'] = all(
            diag.sha(path) == digest
            for path, digest in input_hashes.items())
        if not summary['input_files_unchanged']:
            summary['checks_passed'] = False
            manifest['status'] = summary['status'] = 'failed'
        diag.dump(output / 'manifest.json', manifest)
        diag.dump(output / 'summary.json', summary)
    if not summary['checks_passed']:
        raise RuntimeError('Feature separability integrity check failed')
    print('Feature separability probe completed:',
          output / 'summary.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe_dir', required=True)
    parser.add_argument('--gradient_audit_dir', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--output_dir')
    parser.add_argument('--strict_determinism', action='store_true')
    parser.add_argument('--preflight_only', action='store_true')
    cli = parser.parse_args()
    torch.set_num_threads(1)
    run(cli)


if __name__ == '__main__':
    main()
