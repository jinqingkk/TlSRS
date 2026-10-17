"""Read-only cross-scene audit of signed probability-volume cues."""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import experiment15_momentum_diagnose as diag
import experiment15_residual_feature_separability as base_probe
import experiment15_residual_gradient_decomposition as gradient_audit
import models.cas_mvsnet as cas_module
from experiment15_d_diagnose import rng_snapshot


GROUP_NAMES = ('base', 'signed_only', 'base_plus_signed')
PRIMARY_GROUP = 'base_plus_signed'
SIGNED_CUE_NAMES = (
    'probability_mode_offset',
    'probability_mass_imbalance',
    'probability_skewness',
    'probability_peak_neighbor_difference',
)


def load_feature_probe(root, probe_root, gradient_root):
    """Require the completed negative result that selected this experiment."""
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
            or summary.get('decision', {}).get(
                'cross_scene_linear_signal_established') is not False):
        raise ValueError(
            'Requires the completed negative fused-feature probe')
    for section in ('source_sha256', 'input_sha256'):
        for filename, digest in manifest.get(section, {}).items():
            if diag.sha(filename) != digest:
                raise ValueError(
                    'Feature-probe source/input hash mismatch: {}'.format(
                        filename))
    return manifest, summary, (manifest_path, summary_path)


def _validate_probability_inputs(probability, depth_values):
    if (probability.dim() != 4
            or probability.shape != depth_values.shape):
        raise ValueError(
            'Expected matching probability/depth tensors shaped (B,D,H,W)')
    if not bool(torch.isfinite(probability).all()
                and torch.isfinite(depth_values).all()):
        raise RuntimeError('Nonfinite probability-volume input')
    if bool((probability < 0).any()):
        raise ValueError('Probability volume contains negative entries')
    if probability.size(1) > 1:
        increments = depth_values[:, 1:] - depth_values[:, :-1]
        if bool((increments < 0).any()):
            raise ValueError('Depth hypotheses must be nondecreasing')
    normalizer = probability.sum(1)
    if not bool(torch.allclose(
            normalizer, torch.ones_like(normalizer),
            rtol=1e-4, atol=1e-5)):
        raise ValueError('Probability volume is not normalized over depth')


def signed_probability_volume_cues(probability, depth_values,
                                   depth_interval):
    """Return four inference-only cues whose signs distinguish depth sides."""
    _validate_probability_inputs(probability, depth_values)
    interval = torch.as_tensor(
        depth_interval, dtype=probability.dtype,
        device=probability.device)
    if interval.numel() != 1 or not bool(torch.isfinite(interval)):
        raise ValueError('Expected one finite stage depth interval')
    if float(interval) <= 0.0:
        raise ValueError('Stage depth interval must be positive')

    depth_mean = (probability * depth_values).sum(1)
    normalized_offset = (
        depth_values - depth_mean.unsqueeze(1)) / interval
    mode_index = probability.argmax(1, keepdim=True)
    mode_depth = torch.gather(depth_values, 1, mode_index).squeeze(1)
    mode_offset = (mode_depth - depth_mean) / interval

    deeper = depth_values > depth_mean.unsqueeze(1)
    shallower = depth_values < depth_mean.unsqueeze(1)
    deeper_mass = (probability * deeper.to(probability.dtype)).sum(1)
    shallower_mass = (probability * shallower.to(probability.dtype)).sum(1)
    mass_imbalance = deeper_mass - shallower_mass

    variance = (probability * normalized_offset.square()).sum(1)
    third_moment = (probability * normalized_offset.pow(3)).sum(1)
    stable_variance = variance > 1e-12
    skewness = torch.where(
        stable_variance,
        third_moment / variance.clamp_min(1e-12).pow(1.5),
        torch.zeros_like(third_moment))

    if probability.size(1) == 1:
        neighbor_difference = torch.zeros_like(depth_mean)
    else:
        maximum_index = probability.size(1) - 1
        shallower_index = (mode_index - 1).clamp(min=0)
        deeper_index = (mode_index + 1).clamp(max=maximum_index)
        peak_depth = mode_depth.unsqueeze(1)
        shallower_depth = torch.gather(
            depth_values, 1, shallower_index)
        deeper_depth = torch.gather(depth_values, 1, deeper_index)
        shallower_probability = torch.gather(
            probability, 1, shallower_index)
        deeper_probability = torch.gather(
            probability, 1, deeper_index)
        shallower_probability = torch.where(
            shallower_depth < peak_depth,
            shallower_probability,
            torch.zeros_like(shallower_probability))
        deeper_probability = torch.where(
            deeper_depth > peak_depth,
            deeper_probability,
            torch.zeros_like(deeper_probability))
        neighbor_difference = (
            deeper_probability - shallower_probability).squeeze(1)

    cues = torch.stack([
        mode_offset, mass_imbalance, skewness, neighbor_difference,
    ], dim=1)
    if not bool(torch.isfinite(cues).all()):
        raise RuntimeError('Nonfinite signed probability-volume cue')
    return cues


def capture_signed_forward(model, sample, args):
    """Capture the production fused input and every stage probability volume."""
    fused_values = []
    probability_values = []
    original_statistics = cas_module.probability_volume_statistics

    def capture_fused(module, inputs):
        value = inputs[0]
        if value.dim() != 4:
            raise ValueError('Expected fused features shaped (B,C,H,W)')
        fused_values.append(value.detach())

    def capture_probability(probability, depth_values):
        probability_values.append((
            probability.detach(), depth_values.detach()))
        return original_statistics(probability, depth_values)

    handle = model.sger_blocks[0].residual.register_forward_pre_hook(
        capture_fused)
    cas_module.probability_volume_statistics = capture_probability
    try:
        with torch.no_grad():
            outputs, _, residual_logit = (
                gradient_audit.direction_probe.forward_with_residual_logit(
                    model, sample, args))
    finally:
        cas_module.probability_volume_statistics = original_statistics
        handle.remove()
    if len(fused_values) != 1:
        raise RuntimeError('Expected exactly one fused SGER residual input')
    if len(probability_values) != model.num_stage:
        raise RuntimeError(
            'Expected one probability volume per cascade stage')
    probability, depth_values = probability_values[-1]
    fused = fused_values[0]
    if (probability.shape[0] != fused.shape[0]
            or probability.shape[-2:] != fused.shape[-2:]):
        raise ValueError('Stage3 probability and fused feature shapes differ')
    signed = signed_probability_volume_cues(
        probability, depth_values,
        outputs['stage3']['depth_interval'])
    return outputs, residual_logit.detach(), fused, signed


def checked_forward(model, optimizer, sample, args):
    initial = diag.fingerprints(diag.model_snapshot(model, optimizer))
    rng_before = diag.fingerprints(rng_snapshot())
    runtime = diag.runtime_state(model)
    original_statistics = cas_module.probability_volume_statistics
    try:
        outputs, residual_logit, fused, signed = capture_signed_forward(
            model, sample, args)
    finally:
        diag.restore_runtime(model, runtime)
    final = diag.fingerprints(diag.model_snapshot(model, optimizer))
    rng_after = diag.fingerprints(rng_snapshot())
    if cas_module.probability_volume_statistics is not original_statistics:
        raise RuntimeError('Probability-statistics function was not restored')
    if initial != final:
        raise RuntimeError('Signed-cue probe mutated model or optimizer')
    if rng_before != rng_after:
        raise RuntimeError('Signed-cue probe mutated RNG state')
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError('Signed-cue probe populated parameter .grad')
    if any(module.training for module in model.modules()):
        raise RuntimeError('Signed-cue probe changed eval mode')
    return outputs, residual_logit, fused, signed


def group_channel_names(base_channels):
    base_names = base_probe.channel_names(base_channels)
    return {
        'base': base_names,
        'signed_only': list(SIGNED_CUE_NAMES),
        'base_plus_signed': base_names + list(SIGNED_CUE_NAMES),
    }


def sampled_feature_groups(fused, signed, positive, negative, maximum):
    if (signed.dim() != 4
            or signed.size(1) != len(SIGNED_CUE_NAMES)
            or signed.shape[0] != fused.shape[0]
            or signed.shape[-2:] != fused.shape[-2:]):
        raise ValueError('Unexpected signed-cue shape')
    base_features, labels, sampling, indices = (
        base_probe.balanced_feature_sample(
            fused, positive, negative, maximum))
    signed_flat = signed.permute(0, 2, 3, 1).reshape(
        -1, signed.size(1))
    signed_features = signed_flat[indices].detach().cpu().double()
    if not bool(torch.isfinite(signed_features).all()):
        raise RuntimeError('Nonfinite sampled signed features')
    groups = {
        'base': base_features,
        'signed_only': signed_features,
        'base_plus_signed': torch.cat([
            base_features, signed_features], dim=1),
    }
    return groups, labels, sampling, indices


def feature_maps(fused, signed):
    return {
        'base': fused,
        'signed_only': signed,
        'base_plus_signed': torch.cat([fused, signed], dim=1),
    }


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


def primary_decision(validation):
    groups = {
        name: group_decision(validation, name)
        for name in GROUP_NAMES
    }
    base_pooled = validation['pooled']['base']['balanced_accuracy']
    primary_pooled = validation['pooled'][PRIMARY_GROUP][
        'balanced_accuracy']
    scene_deltas = {
        scene: (values[PRIMARY_GROUP]['balanced_accuracy']
                - values['base']['balanced_accuracy'])
        for scene, values in validation['scenes'].items()
    }
    passed = groups[PRIMARY_GROUP][
        'cross_scene_linear_signal_established']
    return {
        'primary_group': PRIMARY_GROUP,
        'signed_probability_cues_established': passed,
        'groups': groups,
        'pooled_balanced_accuracy_delta_vs_base':
            primary_pooled - base_pooled,
        'scene_balanced_accuracy_delta_vs_base': scene_deltas,
        'next_step': (
            'design and isolate a direction-times-magnitude SGER residual '
            'head, retaining raw-versus-refined ablations'
            if passed else
            'do not change to a factorized head or tune direction-BCE; '
            'seek stronger inference-only signed multi-view matching cues'),
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


def run(cli):
    probe_root = Path(cli.probe_dir)
    gradient_root = Path(cli.gradient_audit_dir)
    feature_root = Path(cli.feature_probe_dir)
    source, _ = gradient_audit.load_source(probe_root)
    _, _, gradient_paths = base_probe.load_gradient_audit(
        gradient_root, probe_root)
    _, _, feature_paths = load_feature_probe(
        feature_root, probe_root, gradient_root)
    args, plan, trainset, valset = base_probe.build_datasets(source)
    snapshot = probe_root / 'direction_reference/step_0000.ckpt'
    fingerprint = (
        probe_root / 'direction_reference/fingerprints/initial.json')
    inputs = [
        probe_root / 'manifest.json', probe_root / 'summary.json',
        snapshot, fingerprint,
    ] + list(gradient_paths) + list(feature_paths)
    input_hashes = {str(path): diag.sha(path) for path in inputs}

    if cli.preflight_only:
        model, optimizer, configured_args, path = (
            gradient_audit.load_snapshot(
                probe_root, 'shared_step0', 'direction_reference', 0,
                source, torch.device('cpu')))
        base_probe.require_configured_snapshot_arguments(
            source['arguments'], configured_args)
        channels = model.sger_blocks[0].residual.body[0].in_channels
        names = group_channel_names(channels)
        print('Preflight passed:', path,
              'feature groups:',
              ', '.join('{}={}'.format(key, len(value))
                        for key, value in names.items()),
              'train/validation samples: 50/6', flush=True)
        del model, optimizer
        return

    if not cli.strict_determinism:
        raise ValueError(
            'The signed-cue experiment requires --strict_determinism')
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

    output = Path(cli.output_dir or (
        'outputs/experiment15_d/signed_probability_cue_separability_'
        + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        'status': 'running',
        'command': sys.argv,
        'source_probe': str(probe_root),
        'source_gradient_audit': str(gradient_root),
        'source_feature_separability': str(feature_root),
        'input_sha256': input_hashes,
        'source_sha256': {
            path: diag.sha(path) for path in diag.SOURCES + [
                'experiment15_momentum_diagnose.py',
                'experiment15_residual_direction_probe.py',
                'experiment15_residual_gradient_decomposition.py',
                'experiment15_residual_feature_separability.py', __file__]
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
        'signed_cue_definitions': {
            'probability_mode_offset':
                '(argmax depth - probability-mean depth) / stage interval',
            'probability_mass_imbalance':
                'mass deeper than mean minus mass shallower than mean',
            'probability_skewness':
                'standardized third central moment over depth',
            'probability_peak_neighbor_difference':
                'deeper-neighbor probability minus shallower-neighbor '
                'probability around the probability mode',
        },
        'decision_gates': {
            'primary_group': PRIMARY_GROUP,
            'pooled_balanced_accuracy':
                base_probe.POOLED_BALANCED_ACCURACY_GATE,
            'each_scene_balanced_accuracy':
                base_probe.PER_SCENE_BALANCED_ACCURACY_GATE,
            'each_scene_score_mean_separation': 'strictly positive',
        },
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'boundary': (
            'Signed cues use only production Stage3 probability/depth '
            'hypotheses. GT defines external direction labels only. Model, '
            'optimizer, buffers and RNG remain read-only.'),
        'interpretation_boundary': (
            'The primary claim concerns a pointwise linear probe on fixed '
            'samples. Failure does not exclude nonlinear or spatially '
            'contextual signed matching signals.'),
    }
    summary = {
        'status': 'running', 'checks_passed': False,
        'train_sampling': [], 'train': {}, 'validation': {},
    }
    diag.dump(output / 'manifest.json', manifest)
    diag.dump(output / 'summary.json', summary)
    try:
        train_features = {name: [] for name in GROUP_NAMES}
        train_labels = []
        sampled_head_scores = []
        train_loader = DataLoader(
            trainset,
            batch_sampler=[plan['train_indices'][index:index + 2]
                           for index in range(0, 50, 2)],
            num_workers=0)
        for batch_index, sample in enumerate(train_loader):
            sample = diag._move_to_device(sample, device)
            outputs, residual_logit, fused, signed = checked_forward(
                model, optimizer, sample, args)
            positive, negative = base_probe.direction_labels(
                outputs, sample, args)
            groups, labels, sampling, sampled_indices = (
                sampled_feature_groups(
                    fused, signed, positive, negative,
                    base_probe.SAMPLES_PER_CLASS_PER_BATCH))
            sampling['batch'] = batch_index
            sampling['indices'] = plan['train_indices'][
                2 * batch_index:2 * batch_index + 2]
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
            'evaluation_scope': (
                'same deterministic balanced sampled pixels used to fit '
                'each external probe'),
        }
        summary['probes'] = {
            name: base_probe.serializable_probe(probes[name], names[name])
            for name in GROUP_NAMES
        }
        del (matrices, labels, train_features, train_labels,
             sampled_head_scores)

        pooled = _empty_group_accumulators()
        pooled_head = base_probe.empty_binary_accumulator()
        scenes = {}
        val_loader = DataLoader(
            valset,
            batch_sampler=[plan['val_indices'][index:index + 2]
                           for index in range(0, 6, 2)],
            num_workers=0)
        for batch_index, sample in enumerate(val_loader):
            sample = diag._move_to_device(sample, device)
            outputs, residual_logit, fused, signed = checked_forward(
                model, optimizer, sample, args)
            positive, negative = base_probe.direction_labels(
                outputs, sample, args)
            maps = feature_maps(fused, signed)
            scene_names = {meta[0] for meta in plan['val_metas'][
                2 * batch_index:2 * batch_index + 2]}
            if len(scene_names) != 1:
                raise ValueError('Validation batch crosses scenes')
            scene = next(iter(scene_names))
            scene_rows = scenes.setdefault(scene, {
                'groups': _empty_group_accumulators(),
                'production_residual_logit':
                    base_probe.empty_binary_accumulator(),
            })
            for name in GROUP_NAMES:
                scores = base_probe.probe_scores(
                    maps[name], probes[name])
                for accumulator in (
                        pooled[name], scene_rows['groups'][name]):
                    base_probe.update_binary_accumulator(
                        accumulator, scores, positive, negative)
            for accumulator in (
                    pooled_head,
                    scene_rows['production_residual_logit']):
                base_probe.update_binary_accumulator(
                    accumulator, residual_logit, positive, negative)
            print('evaluated validation scene', scene, flush=True)

        summary['validation'] = {
            'pooled': dict(
                _summarize_group_accumulators(pooled),
                production_residual_logit=
                    base_probe.summarize_binary(pooled_head)),
            'scenes': {
                scene: dict(
                    _summarize_group_accumulators(values['groups']),
                    production_residual_logit=base_probe.summarize_binary(
                        values['production_residual_logit']))
                for scene, values in scenes.items()
            },
        }
        summary['decision'] = primary_decision(summary['validation'])
        summary['model_state_unchanged'] = True
        summary['rng_state_restored_per_forward'] = True
        summary['probability_statistics_function_restored_per_forward'] = True
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
        raise RuntimeError('Signed-cue separability integrity check failed')
    print('Signed probability-cue separability probe completed:',
          output / 'summary.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe_dir', required=True)
    parser.add_argument('--gradient_audit_dir', required=True)
    parser.add_argument('--feature_probe_dir', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--output_dir')
    parser.add_argument('--strict_determinism', action='store_true')
    parser.add_argument('--preflight_only', action='store_true')
    cli = parser.parse_args()
    torch.set_num_threads(1)
    run(cli)


if __name__ == '__main__':
    main()
