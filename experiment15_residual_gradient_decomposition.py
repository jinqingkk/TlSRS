"""Read-only residual-loss gradient decomposition for Experiment 15-D."""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import experiment15_momentum_diagnose as diag
import experiment15_residual_direction_probe as direction_probe
from experiment15_d_diagnose import rng_snapshot
from models.module import bounded_residual_target


SNAPSHOTS = (
    ('shared_step0', 'direction_reference', 0),
    ('reference_step25', 'direction_reference', 25),
    ('sign_bce_step25', 'sign_bce', 25),
)
DIRECTION_WEIGHT = direction_probe.DIRECTION_LOSS_WEIGHT
# Independent float32 reductions over hundreds of thousands of pixels can
# differ at roughly 1e-4 relative error when the true gradient is very small.
# Keep both errors in the report and accept only a tight relative or absolute
# match; the CPU production-graph regression currently peaks below 7.3e-8.
RECONSTRUCTION_RTOL = 5e-4
RECONSTRUCTION_ATOL = 1e-7


def load_source(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    summary = json.loads((root / 'summary.json').read_text())
    expected_steps = {name: 25 for name in direction_probe.RUNS}
    if (manifest.get('status') != 'completed'
            or summary.get('status') != 'completed'
            or not summary.get('checks_passed')
            or not summary.get('input_files_unchanged')
            or manifest.get('steps') != 25
            or summary.get('completed_steps') != expected_steps
            or manifest.get('strict_determinism') is not True):
        raise ValueError(
            'Requires a completed, checked, strict 25-step direction probe')
    for section in ('source_sha256', 'input_sha256'):
        for filename, digest in manifest.get(section, {}).items():
            if diag.sha(filename) != digest:
                raise ValueError(
                    'Source/input hash mismatch: {}'.format(filename))
    return manifest, summary


def snapshot_phase(step):
    return 'initial' if step == 0 else 'step{}_update'.format(step)


def load_snapshot(root, label, group, step, source, device):
    path = root / group / ('step_{:04d}.ckpt'.format(step))
    checkpoint = torch.load(str(path), map_location='cpu')
    expected_mode = 'sign_bce' if group == 'sign_bce' else 'reference'
    expected_weight = DIRECTION_WEIGHT if group == 'sign_bce' else 0.0
    if (checkpoint.get('epoch') != 12
            or checkpoint.get('local_step') != step
            or checkpoint.get('source_pressure_step') != 25
            or checkpoint.get('direction_mode') != expected_mode
            or checkpoint.get('direction_loss_weight') != expected_weight
            or checkpoint.get('all_modules_eval') is not True
            or checkpoint.get('arguments') != source['arguments']):
        raise ValueError('Unexpected snapshot metadata: {}'.format(path))
    args = argparse.Namespace(**checkpoint['arguments'])
    model = diag.build_model(args).to(device)
    diag.configure_model(model, 'leaky_relu', args)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.set_sger_residual_scale(source['effective_case']['sger_scale'])
    optimizer, _ = diag.restore_optimizer(model, args, checkpoint)
    trainable = direction_probe.freeze_except_residual(model)
    if trainable != checkpoint.get('trainable_parameters'):
        raise ValueError('Trainable residual tensors changed: {}'.format(path))
    reference = json.loads((
        root / group / 'fingerprints' /
        (snapshot_phase(step) + '.json')).read_text())
    current = diag.fingerprints(diag.model_snapshot(model, optimizer))
    if current != reference:
        raise ValueError('Snapshot fingerprint mismatch: {}'.format(path))
    if any(module.training for module in model.modules()):
        raise RuntimeError('Expected every module in eval mode')
    return model, optimizer, args, path


def select_train_batches(root):
    rows = []
    for step in range(1, 26):
        path = (root / 'direction_reference' / 'train' /
                ('step_{:04d}.json'.format(step)))
        row = json.loads(path.read_text())['before']['direction_counts']
        rows.append({
            'position': step - 1,
            'positive_fraction': row['positive_fraction_nonzero'],
            'positive_count': row['positive_count'],
            'negative_count': row['negative_count'],
            'source_file': str(path),
        })
    negative = min(rows, key=lambda value: value['positive_fraction'])
    balanced = min(
        rows, key=lambda value: abs(value['positive_fraction'] - 0.5))
    positive = max(rows, key=lambda value: value['positive_fraction'])
    selected = {
        'negative_majority': negative,
        'near_balanced': balanced,
        'positive_majority': positive,
    }
    if len({row['position'] for row in selected.values()}) != 3:
        raise ValueError('Training batch selection is not unique')
    return selected


def materialize_batches(source, selected, device):
    args = argparse.Namespace(**source['arguments'])
    plan = source['sample_plan']
    dataset_type = diag.find_dataset_def(args.dataset)
    trainset = dataset_type(
        args.trainpath, args.trainlist, 'train', args.nviews,
        args.numdepth, args.interval_scale)
    positions = [selected[name]['position'] for name in (
        'negative_majority', 'near_balanced', 'positive_majority')]
    train_indices = [
        index for position in positions
        for index in plan['train_indices'][2 * position:2 * position + 2]
    ]
    train_metas = [
        meta for position in positions
        for meta in plan['train_metas'][2 * position:2 * position + 2]
    ]
    diag.validate_selection(trainset, train_indices, train_metas, 6)
    diag.seed(source['arguments']['seed'])
    train_loader = DataLoader(
        trainset,
        batch_sampler=[train_indices[index:index + 2]
                       for index in range(0, 6, 2)],
        num_workers=0)
    train = {}
    for name, sample in zip(
            ('negative_majority', 'near_balanced', 'positive_majority'),
            train_loader):
        train[name] = diag._move_to_device(sample, device)

    valset = dataset_type(
        args.trainpath, plan['val_list'], 'val', args.nviews,
        args.numdepth, args.interval_scale)
    diag.validate_selection(
        valset, plan['val_indices'], plan['val_metas'], 6)
    val_loader = DataLoader(
        valset,
        batch_sampler=[plan['val_indices'][index:index + 2]
                       for index in range(0, 6, 2)],
        num_workers=0)
    validation = {
        'batch{}'.format(index): diag._move_to_device(sample, device)
        for index, sample in enumerate(val_loader)
    }
    return {'train': train, 'validation': validation}, {
        'train_indices': train_indices,
        'train_metas': train_metas,
        'val_indices': plan['val_indices'],
        'val_metas': plan['val_metas'],
    }


def gradient_vector(loss, parameters):
    names = list(parameters)
    values = list(parameters.values())
    gradients = torch.autograd.grad(
        loss, values, retain_graph=True, allow_unused=True)
    missing = [name for name, value in zip(names, gradients)
               if value is None]
    if missing:
        raise RuntimeError(
            'Loss does not cover every residual tensor: {}'.format(missing))
    rows = {}
    vector = []
    for name, value in zip(names, gradients):
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError('Nonfinite gradient: {}'.format(name))
        cpu = value.detach().cpu().double().reshape(-1)
        rows[name] = {
            'l2': float(cpu.norm()),
            'abs_max': float(cpu.abs().max()),
            'mean': float(cpu.mean()),
            'zero': not bool((cpu != 0).any()),
        }
        vector.append(cpu)
    return torch.cat(vector), rows


def vector_report(vector, parameter_rows=None):
    row = {
        'l2': float(vector.norm()),
        'abs_max': float(vector.abs().max()),
        'mean': float(vector.mean()),
        'zero': not bool((vector != 0).any()),
        'finite': bool(torch.isfinite(vector).all()),
        'numel': int(vector.numel()),
    }
    if parameter_rows is not None:
        row['parameters'] = parameter_rows
    return row


def cosine(a, b):
    denominator = float(a.norm() * b.norm())
    if denominator == 0.0:
        return None
    return float(torch.dot(a, b) / denominator)


def reconstruction(reference, candidate):
    difference = reference - candidate
    reference_norm = float(reference.norm())
    relative = float(difference.norm()) / max(
        reference_norm, RECONSTRUCTION_ATOL)
    maximum = float(difference.abs().max())
    passed = (
        relative <= RECONSTRUCTION_RTOL
        or maximum <= RECONSTRUCTION_ATOL)
    return {
        'passed': passed,
        'relative_l2_error': relative,
        'max_abs_error': maximum,
        'rtol': RECONSTRUCTION_RTOL,
        'atol': RECONSTRUCTION_ATOL,
    }


def predicted_logit_changes(update_gradient, logit_gradients):
    positive = -float(torch.dot(
        logit_gradients['positive'], update_gradient))
    negative = -float(torch.dot(
        logit_gradients['negative'], update_gradient))
    return {
        'positive_mean_logit_change_per_unit_sgd': positive,
        'negative_mean_logit_change_per_unit_sgd': negative,
        'direction_separation_change_per_unit_sgd': positive - negative,
        'desired_positive_change': positive > 0.0,
        'desired_negative_change': negative < 0.0,
        'both_directions_desired': positive > 0.0 and negative < 0.0,
        'scope': 'first-order residual-head-only unit SGD step; not Adam',
    }


def decompose_gradients(parameters, production_loss, residual_target_loss,
                        direction_loss, residual_logit, positive, negative,
                        residual_target_weight, direction_weight):
    positive_count = int(positive.sum())
    negative_count = int(negative.sum())
    nonzero_count = positive_count + negative_count
    if not positive_count or not negative_count:
        raise ValueError('Gradient decomposition requires both directions')

    positive_loss = F.binary_cross_entropy_with_logits(
        residual_logit[positive],
        torch.ones_like(residual_logit[positive]), reduction='mean')
    negative_loss = F.binary_cross_entropy_with_logits(
        residual_logit[negative],
        torch.zeros_like(residual_logit[negative]), reduction='mean')
    positive_factor = direction_weight * positive_count / nonzero_count
    negative_factor = direction_weight * negative_count / nonzero_count
    balanced_positive_factor = direction_weight * 0.5
    balanced_negative_factor = direction_weight * 0.5
    balanced_direction_loss = 0.5 * (positive_loss + negative_loss)

    losses = {
        'production_total': production_loss,
        'residual_target_weighted': (
            residual_target_weight * residual_target_loss),
        'direction_weighted': direction_weight * direction_loss,
        'direction_positive_contribution': positive_factor * positive_loss,
        'direction_negative_contribution': negative_factor * negative_loss,
        'direction_balanced_weighted': (
            direction_weight * balanced_direction_loss),
        'direction_balanced_positive_contribution': (
            balanced_positive_factor * positive_loss),
        'direction_balanced_negative_contribution': (
            balanced_negative_factor * negative_loss),
        'experimental_total': (
            production_loss + direction_weight * direction_loss),
        'experimental_balanced_total': (
            production_loss + direction_weight * balanced_direction_loss),
    }
    vectors = {}
    reports = {}
    for name, loss in losses.items():
        vector, parameter_rows = gradient_vector(loss, parameters)
        vectors[name] = vector
        reports[name] = vector_report(vector, parameter_rows)
        reports[name]['loss'] = float(loss.detach())

    vectors['production_other'] = (
        vectors['production_total'] - vectors['residual_target_weighted'])
    reports['production_other'] = vector_report(vectors['production_other'])
    reports['production_other']['definition'] = (
        'production_total gradient minus weighted stage3 residual-target '
        'gradient')

    direction_reconstruction = reconstruction(
        vectors['direction_weighted'],
        vectors['direction_positive_contribution']
        + vectors['direction_negative_contribution'])
    experimental_reconstruction = reconstruction(
        vectors['experimental_total'],
        vectors['production_total'] + vectors['direction_weighted'])
    balanced_direction_reconstruction = reconstruction(
        vectors['direction_balanced_weighted'],
        vectors['direction_balanced_positive_contribution']
        + vectors['direction_balanced_negative_contribution'])
    balanced_experimental_reconstruction = reconstruction(
        vectors['experimental_balanced_total'],
        vectors['production_total']
        + vectors['direction_balanced_weighted'])
    if (not direction_reconstruction['passed']
            or not experimental_reconstruction['passed']
            or not balanced_direction_reconstruction['passed']
            or not balanced_experimental_reconstruction['passed']):
        raise RuntimeError(
            'Gradient reconstruction check failed: direction={} '
            'experimental={} balanced_direction={} '
            'balanced_experimental={}'.format(
                direction_reconstruction, experimental_reconstruction,
                balanced_direction_reconstruction,
                balanced_experimental_reconstruction))

    scalar_direction = (
        positive_factor * positive_loss + negative_factor * negative_loss)
    scalar_expected = direction_weight * direction_loss
    scalar_error = abs(float((scalar_direction - scalar_expected).detach()))
    scalar_tolerance = 1e-6 * max(
        1.0, abs(float(scalar_expected.detach())))
    if scalar_error > scalar_tolerance:
        raise RuntimeError('Direction scalar reconstruction check failed')
    balanced_scalar = (
        balanced_positive_factor * positive_loss
        + balanced_negative_factor * negative_loss)
    balanced_scalar_expected = direction_weight * balanced_direction_loss
    balanced_scalar_error = abs(float(
        (balanced_scalar - balanced_scalar_expected).detach()))
    if balanced_scalar_error > scalar_tolerance:
        raise RuntimeError(
            'Balanced direction scalar reconstruction check failed')

    logit_gradients = {}
    for name, mask in (('positive', positive), ('negative', negative)):
        vector, _ = gradient_vector(residual_logit[mask].mean(), parameters)
        logit_gradients[name] = vector

    changes = {
        name: predicted_logit_changes(vectors[name], logit_gradients)
        for name in ('production_total', 'residual_target_weighted',
                     'production_other', 'direction_weighted',
                     'direction_balanced_weighted', 'experimental_total',
                     'experimental_balanced_total')
    }
    pairs = (
        ('production_total', 'direction_weighted'),
        ('residual_target_weighted', 'direction_weighted'),
        ('production_other', 'direction_weighted'),
        ('direction_positive_contribution',
         'direction_negative_contribution'),
        ('production_total', 'direction_balanced_weighted'),
        ('residual_target_weighted', 'direction_balanced_weighted'),
        ('production_other', 'direction_balanced_weighted'),
        ('direction_balanced_positive_contribution',
         'direction_balanced_negative_contribution'),
    )
    interactions = {
        '{}__{}'.format(left, right): {
            'dot': float(torch.dot(vectors[left], vectors[right])),
            'cosine': cosine(vectors[left], vectors[right]),
            'left_l2': float(vectors[left].norm()),
            'right_l2': float(vectors[right].norm()),
        }
        for left, right in pairs
    }
    return {
        'loss_gradients': reports,
        'interactions': interactions,
        'predicted_logit_changes': changes,
        'checks': {
            'direction_group_gradient_reconstruction':
                direction_reconstruction,
            'experimental_total_gradient_reconstruction':
                experimental_reconstruction,
            'balanced_direction_group_gradient_reconstruction':
                balanced_direction_reconstruction,
            'balanced_experimental_total_gradient_reconstruction':
                balanced_experimental_reconstruction,
            'direction_scalar_reconstruction': {
                'passed': True,
                'abs_error': scalar_error,
                'tolerance': scalar_tolerance,
            },
            'balanced_direction_scalar_reconstruction': {
                'passed': True,
                'abs_error': balanced_scalar_error,
                'tolerance': scalar_tolerance,
            },
            'all_eight_residual_tensors_covered': all(
                len(report.get('parameters', {})) == 8
                for name, report in reports.items()
                if name != 'production_other'),
        },
        'direction_group_losses': {
            'positive_mean': float(positive_loss.detach()),
            'negative_mean': float(negative_loss.detach()),
            'positive_full_mean_factor': positive_factor,
            'negative_full_mean_factor': negative_factor,
            'balanced_positive_factor': balanced_positive_factor,
            'balanced_negative_factor': balanced_negative_factor,
            'balanced_direction_mean': float(
                balanced_direction_loss.detach()),
        },
    }


def protected_forward_fingerprints(outputs, sample):
    values = {}
    diag.flatten(sample, 'input/sample', values)
    stage = outputs['stage3']
    for key in ('depth_raw', 'photometric_confidence', 'normal',
                'benefit_gate', 'gate_logits_a', 'gate_logits_b',
                'region_a', 'region_b_weight', 'gate_valid_mask'):
        diag.flatten(stage[key], 'backbone/fixed/' + key, values)
    return diag.fingerprints(values)


def audit_batch(model, optimizer, sample, args, case):
    initial = diag.fingerprints(diag.model_snapshot(model, optimizer))
    rng_before = diag.fingerprints(rng_snapshot())
    runtime = diag.runtime_state(model)
    parameters = diag.residual_parameters(model)
    try:
        outputs, diagnostics, residual_logit = (
            direction_probe.forward_with_residual_logit(
                model, sample, args))
        stage = outputs['stage3']
        gt = sample['depth']['stage3']
        mask = sample['mask']['stage3']
        losses = diag.cas_mvsnet_loss(
            outputs, sample['depth'], sample['mask'],
            **diag._loss_kwargs(args, sample, case))
        direction_loss, target, positive, negative, counts = (
            direction_probe.direction_loss(
                residual_logit, stage['depth_raw'], gt,
                stage['depth_interval'], mask,
                args.residual_target_ratio))
        valid = mask > 0.5
        residual_target_loss = F.smooth_l1_loss(
            stage['raw_depth_residual'][valid], target[valid],
            reduction='mean')
        recorded_residual_loss = losses[-1][
            'stage3/residual_target_loss']
        if not torch.allclose(
                residual_target_loss.detach(), recorded_residual_loss,
                rtol=1e-6, atol=1e-7):
            raise RuntimeError('Residual-target loss reconstruction failed')
        residual_weight = (
            case.loss_weights['residual_target_loss_weight']
            * case.residual_target_scale)
        report = decompose_gradients(
            parameters, losses[0], residual_target_loss,
            direction_loss, residual_logit, positive, negative,
            residual_weight, DIRECTION_WEIGHT)
        gradient_check = direction_probe.direction_gradient_check(
            direction_loss, residual_logit, positive, negative)
        direction_probe.require_direction_gradient({
            'direction_counts': counts,
            'direction_gradients': gradient_check,
        })
        report.update({
            'production_total_loss': float(losses[0].detach()),
            'residual_target_loss': float(residual_target_loss.detach()),
            'direction_loss': float(direction_loss.detach()),
            'residual_target_weight': residual_weight,
            'direction_weight': DIRECTION_WEIGHT,
            'direction_counts': counts,
            'isolated_logit_gradient_check': gradient_check,
            'residual_diagnostics': diagnostics,
        })
        protected = protected_forward_fingerprints(outputs, sample)
    finally:
        diag.restore_runtime(model, runtime)
    final = diag.fingerprints(diag.model_snapshot(model, optimizer))
    rng_after = diag.fingerprints(rng_snapshot())
    if initial != final:
        raise RuntimeError('Read-only audit mutated model or optimizer')
    if rng_before != rng_after:
        raise RuntimeError('Read-only audit mutated RNG state')
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError('Read-only audit populated parameter .grad')
    if any(module.training for module in model.modules()):
        raise RuntimeError('Read-only audit changed eval mode')
    report['state_unchanged'] = True
    report['rng_unchanged'] = True
    return report, protected


def result_summary(rows):
    values = list(rows.values())
    return {
        'batch_count': len(values),
        'positive_fraction_min': min(
            row['direction_counts']['positive_fraction_nonzero']
            for row in values),
        'positive_fraction_max': max(
            row['direction_counts']['positive_fraction_nonzero']
            for row in values),
        'experimental_both_directions_desired_count': sum(
            row['predicted_logit_changes']['experimental_total'][
                'both_directions_desired']
            for row in values),
        'direction_both_directions_desired_count': sum(
            row['predicted_logit_changes']['direction_weighted'][
                'both_directions_desired']
            for row in values),
        'balanced_direction_both_directions_desired_count': sum(
            row['predicted_logit_changes']['direction_balanced_weighted'][
                'both_directions_desired']
            for row in values),
        'balanced_experimental_both_directions_desired_count': sum(
            row['predicted_logit_changes']['experimental_balanced_total'][
                'both_directions_desired']
            for row in values),
        'production_direction_cosines': [
            row['interactions'][
                'production_total__direction_weighted']['cosine']
            for row in values],
        'residual_target_direction_cosines': [
            row['interactions'][
                'residual_target_weighted__direction_weighted']['cosine']
            for row in values],
        'production_balanced_direction_cosines': [
            row['interactions'][
                'production_total__direction_balanced_weighted']['cosine']
            for row in values],
    }


def run(cli):
    root = Path(cli.probe_dir)
    source, _ = load_source(root)
    selected = select_train_batches(root)
    inputs = [root / 'manifest.json', root / 'summary.json']
    # Selection is an argmin/nearest/argmax over all 25 source rows, so bind
    # every row rather than only the three winners to the audit provenance.
    inputs.extend(
        root / 'direction_reference' / 'train' /
        ('step_{:04d}.json'.format(step))
        for step in range(1, 26))
    for _, group, step in SNAPSHOTS:
        inputs.extend([
            root / group / ('step_{:04d}.ckpt'.format(step)),
            root / group / 'fingerprints' /
            (snapshot_phase(step) + '.json'),
        ])
    input_hashes = {str(path): diag.sha(path) for path in inputs}

    if cli.preflight_only:
        for label, group, step in SNAPSHOTS:
            model, optimizer, _, path = load_snapshot(
                root, label, group, step, source, torch.device('cpu'))
            del model, optimizer
            print('Verified:', label, path, flush=True)
        print('Preflight passed: provenance, three snapshots, and '
              'eight-tensor boundary verified', flush=True)
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
    batches, selection = materialize_batches(source, selected, device)
    output = Path(cli.output_dir or (
        'outputs/experiment15_d/residual_gradient_decomposition_'
        + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    output.mkdir(parents=True, exist_ok=False)
    info = {
        'status': 'running',
        'command': sys.argv,
        'source_probe': str(root),
        'input_sha256': input_hashes,
        'source_sha256': {
            path: diag.sha(path) for path in diag.SOURCES + [
                'experiment15_momentum_diagnose.py',
                'experiment15_residual_direction_probe.py', __file__]
        },
        'arguments': source['arguments'],
        'effective_case': source['effective_case'],
        'snapshots': [
            {'label': label, 'group': group, 'step': step}
            for label, group, step in SNAPSHOTS
        ],
        'selected_source_batches': selected,
        'selected_data': selection,
        'device': cli.device,
        'strict_determinism': cli.strict_determinism,
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'boundary': (
            'Read-only autograd.grad on exactly eight residual-head tensors; '
            'all model/optimizer/BN/RNG state must remain exact'),
        'interpretation_boundary': (
            'Predicted logit changes are first-order unit-SGD derivatives, '
            'not restored-Adam counterfactual updates'),
    }
    summary = {
        'status': 'running',
        'checks_passed': False,
        'results': {},
        'protected_comparisons': {},
    }
    diag.dump(output / 'manifest.json', info)
    diag.dump(output / 'summary.json', summary)
    protected_references = {}
    try:
        for label, group, step in SNAPSHOTS:
            diag.seed(source['arguments']['seed'])
            model, optimizer, args, _ = load_snapshot(
                root, label, group, step, source, device)
            case = diag.audit_cases(args)[0]
            if case._asdict() != source['effective_case']:
                raise ValueError('Effective loss schedule changed')
            snapshot_rows = {}
            for split in ('train', 'validation'):
                split_rows = {}
                for batch_name, sample in batches[split].items():
                    row, protected = audit_batch(
                        model, optimizer, sample, args, case)
                    identity = split + '/' + batch_name
                    if identity not in protected_references:
                        protected_references[identity] = protected
                        comparison = {
                            'exact': True,
                            'different_entries': 0,
                            'first_difference': None,
                            'first_protected_difference': None,
                        }
                    else:
                        comparison = diag.compare_fingerprints(
                            protected_references[identity], protected)
                        if not comparison['exact']:
                            raise RuntimeError(
                                'Fixed input/raw/gate differs: '
                                + label + '/' + identity)
                    summary['protected_comparisons'].setdefault(
                        label, {})[identity] = comparison
                    split_rows[batch_name] = row
                    diag.dump(
                        output / label / split /
                        (batch_name + '.json'), row)
                    print(label, split, batch_name, 'completed', flush=True)
                snapshot_rows[split] = {
                    'batches': split_rows,
                    'summary': result_summary(split_rows),
                }
            summary['results'][label] = snapshot_rows
            diag.dump(output / 'summary.json', summary)
            del model, optimizer
            if cli.device == 'cuda':
                torch.cuda.empty_cache()
        summary['checks_passed'] = True
        info['status'] = summary['status'] = 'completed'
    except BaseException as error:
        info['status'] = summary['status'] = (
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
            info['status'] = summary['status'] = 'failed'
        diag.dump(output / 'manifest.json', info)
        diag.dump(output / 'summary.json', summary)
    if not summary['checks_passed']:
        raise RuntimeError('Gradient decomposition integrity check failed')
    print('Gradient decomposition completed:',
          output / 'summary.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe_dir', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--output_dir')
    parser.add_argument('--strict_determinism', action='store_true')
    parser.add_argument('--preflight_only', action='store_true')
    cli = parser.parse_args()
    torch.set_num_threads(1)
    run(cli)


if __name__ == '__main__':
    main()
