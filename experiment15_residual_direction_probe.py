"""Residual-only paired probe for an auxiliary signed-direction BCE."""
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
import experiment15_gate_selection_audit as selection_audit
import experiment15_residual_sign_audit as sign_audit
from experiment15_d_diagnose import rng_snapshot
from experiment15_d_bce_pressure import pool_metrics
from models.module import bounded_residual_target


RUNS = ('direction_reference', 'direction_repeat', 'sign_bce')
RESIDUAL_PREFIX = 'sger_blocks.0.residual.'
DIRECTION_LOSS_WEIGHT = 0.05


def freeze_except_residual(model):
    names = []
    for name, parameter in model.named_parameters():
        enabled = name.startswith(RESIDUAL_PREFIX)
        parameter.requires_grad_(enabled)
        parameter.grad = None
        if enabled:
            names.append(name)
    if len(names) != 8 or len(model.sger_blocks) != 1:
        raise ValueError('Expected one SGER-Lite residual head with eight parameter tensors')
    model.eval()
    return names


def fixed_state(model, optimizer):
    snapshot = diag.model_snapshot(model, optimizer)
    return diag.fingerprints({key: value for key, value in snapshot.items()
                              if ('/' + RESIDUAL_PREFIX) not in key})


def forward_with_residual_logit(model, sample, args):
    values = []

    def capture(module, inputs, output):
        if output.dim() != 4 or output.size(1) != 1:
            raise ValueError('Expected residual output shape (B,1,H,W)')
        values.append(output[:, 0])

    handle = model.sger_blocks[0].residual.output.register_forward_hook(capture)
    try:
        outputs, diagnostics = diag.proposal_forward(model, sample, args)
    finally:
        handle.remove()
    if len(values) != 1:
        raise RuntimeError('Expected exactly one SGER residual forward')
    return outputs, diagnostics, values[0]


def direction_loss(residual_logit, depth_raw, gt, depth_interval,
                   mask, target_ratio):
    target = bounded_residual_target(
        depth_raw, gt, depth_interval, target_ratio)
    valid = mask > .5
    nonzero = valid & (target != 0)
    count = int(valid.sum())
    nonzero_count = int(nonzero.sum())
    if not count or not nonzero_count:
        raise ValueError('Direction BCE requires valid nonzero residual targets')
    positive = nonzero & (target > 0)
    negative = nonzero & (target < 0)
    truth = positive[nonzero].to(residual_logit.dtype)
    loss = F.binary_cross_entropy_with_logits(
        residual_logit[nonzero], truth, reduction='mean')
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('Nonfinite residual direction BCE')
    info = {
        'valid_count': count,
        'nonzero_count': nonzero_count,
        'positive_count': int(positive.sum()),
        'negative_count': int(negative.sum()),
        'zero_count': int((valid & (target == 0)).sum()),
        'positive_fraction_nonzero': float(positive.sum()) / nonzero_count,
        'normalization': 'pixel mean over valid nonzero bounded targets',
    }
    return loss, target, positive, negative, info


def direction_gradient_check(loss, residual_logit, positive, negative):
    gradient = torch.autograd.grad(
        DIRECTION_LOSS_WEIGHT * loss, residual_logit, retain_graph=True)[0]
    if not bool(torch.isfinite(gradient).all()):
        raise RuntimeError('Nonfinite isolated direction gradient')
    return {
        'positive_correct_gradient_count': int((gradient[positive] < 0).sum()),
        'positive_wrong_gradient_count': int((gradient[positive] > 0).sum()),
        'positive_zero_gradient_count': int((gradient[positive] == 0).sum()),
        'negative_correct_gradient_count': int((gradient[negative] > 0).sum()),
        'negative_wrong_gradient_count': int((gradient[negative] < 0).sum()),
        'negative_zero_gradient_count': int((gradient[negative] == 0).sum()),
        'excluded_zero_gradient_nonzero_count': int(
            (gradient[~(positive | negative)] != 0).sum()),
        'scope': 'isolated direction BCE times fixed weight 0.05; no parameter update',
    }


def require_direction_gradient(metrics):
    counts = metrics['direction_counts']
    gradients = metrics['direction_gradients']
    if not counts['positive_count'] or not counts['negative_count']:
        raise RuntimeError('Starting batch must contain both residual directions')
    expected = {
        'positive_correct_gradient_count': counts['positive_count'],
        'positive_wrong_gradient_count': 0,
        'positive_zero_gradient_count': 0,
        'negative_correct_gradient_count': counts['negative_count'],
        'negative_wrong_gradient_count': 0,
        'negative_zero_gradient_count': 0,
        'excluded_zero_gradient_nonzero_count': 0,
    }
    if any(gradients[key] != value for key, value in expected.items()):
        raise RuntimeError('Direction BCE logit gradient check failed')


def residual_step(model, optimizer, forward):
    """One residual-only update with a read-only same-batch measurement."""
    parameters = {name: value for name, value in model.named_parameters()
                  if value.requires_grad}
    covered = {id(value) for group in optimizer.param_groups
               for value in group['params']}
    if any(id(value) not in covered for value in parameters.values()):
        raise RuntimeError('Residual parameter missing from optimizer')
    frozen = fixed_state(model, optimizer)
    before = {name: value.detach().cpu().double().clone()
              for name, value in parameters.items()}
    optimizer.zero_grad(set_to_none=True)
    runtime = diag.runtime_state(model)
    loss, pre, pre_hashes = forward()
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('Nonfinite total loss')
    loss.backward()
    gradients = {name: value.grad.detach().cpu().double().clone()
                 for name, value in parameters.items()
                 if value.grad is not None}
    if (set(gradients) != set(parameters)
            or not any(bool((value != 0).any())
                       for value in gradients.values())):
        raise RuntimeError('Missing or entirely zero residual gradients')
    gradient_hashes = diag.fingerprints(
        diag.model_snapshot(model, gradients=True))
    optimizer.step()
    update_hashes = diag.fingerprints(diag.model_snapshot(model, optimizer))
    updates = {name: value.detach().cpu().double() - before[name]
               for name, value in parameters.items()}
    if not any(bool((value != 0).any()) for value in updates.values()):
        raise RuntimeError('No actual residual parameter update')
    directions = {
        name: {
            'gradient_l2': float(gradients[name].norm()),
            'update_l2': float(updates[name].norm()),
            'gradient_dot_update': float(
                (gradients[name] * updates[name]).sum()),
        }
        for name in parameters
    }
    with torch.no_grad():
        _, post, post_hashes = forward()
    diag.restore_runtime(model, runtime)
    if update_hashes != diag.fingerprints(
            diag.model_snapshot(model, optimizer)):
        raise RuntimeError('Post measurement mutated model/optimizer/BN')
    if fixed_state(model, optimizer) != frozen:
        raise RuntimeError('Frozen state changed')
    if diag.compare_fingerprints(
            pre_hashes, post_hashes)['first_protected_difference']:
        raise RuntimeError('Raw/gate/feature/normal changed within update')
    return {
        'before': pre,
        'after': post,
        'residual_directions': directions,
        'measurement_state_restored': True,
    }, {
        'before': pre_hashes,
        'gradients': gradient_hashes,
        'update': update_hashes,
        'after': post_hashes,
    }


def signed_groups(stage, target, gt, valid):
    groups, masks = sign_audit.signed_statistics(
        stage['raw_depth_residual'], target, valid)
    raw = stage['depth_raw']
    proposal = stage['raw_depth_residual']
    refined = stage['depth_refined']
    values = {
        'proposal_gain': (raw - gt).abs() - (raw + proposal - gt).abs(),
        'raw_abs_error': (raw - gt).abs(),
        'refined_abs_error': (refined - gt).abs(),
        'raw_to_refined_delta': (raw - gt).abs() - (refined - gt).abs(),
        'gate_mean': stage['benefit_gate'],
    }
    for name, selected in masks.items():
        groups[name]['sums'].update({
            key: float(value.detach()[selected].double().sum())
            for key, value in values.items()
        })
    return groups


def make_forward(model, sample, args, case, use_direction_loss,
                 selection_rows=None):
    def forward():
        outputs, diagnostics, residual_logit = forward_with_residual_logit(
            model, sample, args)
        stage = outputs['stage3']
        gt = sample['depth']['stage3']
        mask = sample['mask']['stage3']
        losses = diag.cas_mvsnet_loss(
            outputs, sample['depth'], sample['mask'],
            **diag._loss_kwargs(args, sample, case))
        auxiliary, target, positive, negative, counts = direction_loss(
            residual_logit, stage['depth_raw'], gt,
            stage['depth_interval'], mask, args.residual_target_ratio)
        total = losses[0]
        if use_direction_loss:
            total = total + DIRECTION_LOSS_WEIGHT * auxiliary
        stats = diag.stage_statistics(
            stage, gt, mask, args.benefit_margin_ratio)
        diag.proposal_statistics(stats, outputs, sample, args, model)
        metrics = diag.aggregate([stats])
        metrics.update(
            loss=float(total.detach()),
            production_total_loss=float(losses[0].detach()),
            residual_target_loss=float(
                losses[-1]['stage3/residual_target_loss'].detach()),
            gate_benefit_loss=float(
                losses[-1]['stage3/gate_benefit_loss'].detach()),
            direction_loss=float(auxiliary.detach()),
            effective_direction_loss_weight=(
                DIRECTION_LOSS_WEIGHT if use_direction_loss else 0.0),
            direction_counts=counts,
            direction_gradients=direction_gradient_check(
                auxiliary, residual_logit, positive, negative)
                if torch.is_grad_enabled() else None,
            signed_groups=signed_groups(
                stage, target, gt, mask > .5),
            residual_diagnostics=diagnostics,
            direction_mode='sign_bce' if use_direction_loss else 'reference',
        )
        values = diag.forward_snapshot(outputs, total)
        diag.flatten(sample, 'input/sample', values)
        for key in ('depth_raw', 'photometric_confidence', 'normal',
                    'benefit_gate', 'gate_logits_a', 'gate_logits_b',
                    'region_a', 'region_b_weight', 'gate_valid_mask'):
            diag.flatten(stage[key], 'backbone/fixed/' + key, values)
        if selection_rows is not None:
            for index in range(len(gt)):
                single = {
                    key: value[index:index + 1]
                    if torch.is_tensor(value) and value.dim() > 0 else value
                    for key, value in stage.items()
                }
                selection_rows.append(selection_audit.selection_statistics(
                    single, gt[index:index + 1], mask[index:index + 1],
                    args.benefit_margin_ratio))
        return total, metrics, diag.fingerprints(values)
    return forward


def evaluate(model, optimizer, dataset, args, case, plan, record, step,
             device):
    before = diag.fingerprints(diag.model_snapshot(model, optimizer))
    runtime = diag.runtime_state(model)
    quality_rows = []
    signed_rows = []
    selection_rows = []
    scene_signed = {}
    scene_selection = {}
    try:
        with torch.no_grad():
            loader = DataLoader(
                dataset,
                batch_sampler=[plan['val_indices'][index:index + 2]
                               for index in range(0, 6, 2)],
                num_workers=0)
            for batch, sample in enumerate(loader):
                sample = diag._move_to_device(sample, device)
                batch_selection = []
                _, metrics, hashes = make_forward(
                    model, sample, args, case, False, batch_selection)()
                quality_rows.append(metrics)
                signed_rows.append(metrics['signed_groups'])
                selection_rows.extend(batch_selection)
                record('val%d_batch%d' % (step, batch), hashes, step == 0)
                for index, row in enumerate(batch_selection):
                    scene = plan['val_metas'][2 * batch + index][0]
                    scene_selection.setdefault(scene, []).append(row)
                # Validation batches are pre-registered as one scene each.
                scenes = {meta[0] for meta in
                          plan['val_metas'][2 * batch:2 * batch + 2]}
                if len(scenes) != 1:
                    raise ValueError('Validation batch crosses scenes')
                scene_signed.setdefault(next(iter(scenes)), []).append(
                    metrics['signed_groups'])
    finally:
        diag.restore_runtime(model, runtime)
    if before != diag.fingerprints(diag.model_snapshot(model, optimizer)):
        raise RuntimeError('Validation changed model/optimizer')
    return {
        'quality': pool_metrics(quality_rows),
        'signed': sign_audit.summarize(signed_rows),
        'signed_scenes': {
            scene: sign_audit.summarize(rows)
            for scene, rows in scene_signed.items()
        },
        'selection': selection_audit.pool(selection_rows),
        'selection_scenes': {
            scene: selection_audit.pool(rows)
            for scene, rows in scene_selection.items()
        },
        'batches': quality_rows,
        'state_restored': True,
        'loss_definition': 'common production loss without direction auxiliary',
    }


def run(cli):
    root = Path(cli.probe_dir)
    source = selection_audit.load_source(root)
    plan = source['sample_plan']
    if len(plan['train_indices']) != 50 or len(plan['train_metas']) != 50:
        raise ValueError('Expected original fifty-sample plan')
    source_checkpoint = root / 'stable/step_0025.ckpt'
    inputs = [
        root / 'manifest.json', root / 'summary.json', source_checkpoint,
        root / 'stable/fingerprints/step25_update.json',
    ]
    input_hashes = {str(path): diag.sha(path) for path in inputs}
    if cli.preflight_only:
        model, optimizer, args = selection_audit.load_model(
            root, 'stable', 25, source, torch.device('cpu'))
        names = freeze_except_residual(model)
        print('Preflight passed; residual tensors:', len(names),
              'source:', source_checkpoint,
              'train samples replayed:', 2 * cli.steps, flush=True)
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
    output = Path(cli.output_dir or (
        'outputs/experiment15_d/residual_direction_%dsteps_' % cli.steps
        + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    output.mkdir(parents=True, exist_ok=False)
    save_steps = [value for value in (0, 1, 2, 5, 10, 25)
                  if value <= cli.steps]
    info = {
        'status': 'running',
        'command': sys.argv,
        'input_sha256': input_hashes,
        'source_probe': str(root),
        'source_checkpoint': str(source_checkpoint),
        'source_sha256': {
            path: diag.sha(path)
            for path in diag.SOURCES + [
                'experiment15_momentum_diagnose.py',
                'experiment15_gate_selection_audit.py',
                'experiment15_residual_sign_audit.py',
                'experiment15_d_bce_pressure.py', __file__]
        },
        'arguments': dict(
            source['arguments'], gate_bce_mode='stable_logprob',
            sger_residual_final_activation='leaky_relu'),
        'effective_case': source['effective_case'],
        'sample_plan': plan,
        'steps': cli.steps,
        'save_steps': save_steps,
        'runs': RUNS,
        'strict_determinism': cli.strict_determinism,
        'device': cli.device,
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'intervention': (
            'Add pixel-mean BCEWithLogits on the sign of nonzero bounded '
            'residual targets; fixed effective weight 0.05'),
        'boundary': (
            'Only eight residual-head tensors trainable; all modules eval; '
            'raw/features/normals/gate/BN frozen'),
        'recovery': (
            'All Adam restored from stable BCE pressure local step25; '
            'no reset or scheduler; original fifty samples replayed'),
    }
    summary = {
        'status': 'running', 'checks_passed': False,
        'completed_steps': {}, 'comparisons': {}, 'validation': {},
    }
    diag.dump(output / 'manifest.json', info)
    diag.dump(output / 'summary.json', summary)
    try:
        for label in RUNS:
            diag.seed(source['arguments']['seed'])
            model, optimizer, args = selection_audit.load_model(
                root, 'stable', 25, source, device)
            trainable = freeze_except_residual(model)
            case = diag.audit_cases(args)[0]
            if case._asdict() != source['effective_case']:
                raise ValueError('Effective schedule changed')
            initial_fixed = fixed_state(model, optimizer)
            directory = output / label
            use_direction = label == 'sign_bce'

            def record(phase, values, exact=False):
                diag.dump(directory / 'fingerprints' / (phase + '.json'),
                          values)
                if label == RUNS[0]:
                    return
                reference = json.loads((
                    output / RUNS[0] / 'fingerprints' /
                    (phase + '.json')).read_text())
                if label == 'sign_bce' and phase == 'step1_before':
                    reference = {key: value for key, value in reference.items()
                                 if key != 'loss/total'}
                    values = {key: value for key, value in values.items()
                              if key != 'loss/total'}
                comparison = diag.compare_fingerprints(reference, values)
                summary['comparisons'].setdefault(label, {})[phase] = comparison
                if (comparison['first_protected_difference']
                        or ((exact or label == 'direction_repeat')
                            and not comparison['exact'])):
                    raise RuntimeError(
                        'Paired invariance failed: ' + label + '/' + phase)

            record('initial', diag.fingerprints(
                diag.model_snapshot(model, optimizer)), True)
            diag.dump(directory / 'trainable_parameters.json', trainable)
            diag.dump(directory / 'optimizer_groups.json', [
                {key: value for key, value in group.items()
                 if key != 'params'}
                for group in optimizer.param_groups
            ])
            dataset = diag.find_dataset_def(args.dataset)
            trainset = dataset(
                args.trainpath, args.trainlist, 'train', args.nviews,
                args.numdepth, args.interval_scale)
            valset = dataset(
                args.trainpath, plan['val_list'], 'val', args.nviews,
                args.numdepth, args.interval_scale)
            diag.validate_selection(
                trainset, plan['train_indices'][:2 * cli.steps],
                plan['train_metas'][:2 * cli.steps], 2 * cli.steps)
            diag.validate_selection(
                valset, plan['val_indices'], plan['val_metas'], 6)

            def validate_and_save(step):
                row = evaluate(
                    model, optimizer, valset, args, case, plan, record,
                    step, device)
                summary['validation'].setdefault(str(step), {})[label] = row
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'arguments': vars(args).copy(),
                    'epoch': 12,
                    'local_step': step,
                    'source_pressure_step': 25,
                    'direction_mode': 'sign_bce' if use_direction else 'reference',
                    'direction_loss_weight': (
                        DIRECTION_LOSS_WEIGHT if use_direction else 0.0),
                    'trainable_parameters': trainable,
                    'all_modules_eval': True,
                }, str(directory / ('step_%04d.ckpt' % step)))
                diag.dump(output / 'summary.json', summary)
                quality = row['quality']
                negative = row['signed']['negative']
                print(label, 'validation', step,
                      'proposal_gain', quality['proposal_gain_mean'],
                      'raw-refined', quality['raw_to_refined_error_delta'],
                      'negative-direction',
                      negative['means']['direction_accuracy'], flush=True)

            validate_and_save(0)
            loader = DataLoader(
                trainset,
                batch_sampler=[plan['train_indices'][index:index + 2]
                               for index in range(0, 2 * cli.steps, 2)],
                num_workers=0)
            for step, sample in enumerate(loader, 1):
                record('step%d_rng' % step,
                       diag.fingerprints(rng_snapshot()), True)
                sample = diag._move_to_device(sample, device)
                forward = make_forward(
                    model, sample, args, case, use_direction)

                def checked_forward():
                    result = forward()
                    if step == 1 and torch.is_grad_enabled():
                        require_direction_gradient(result[1])
                    return result

                row, signatures = residual_step(
                    model, optimizer, checked_forward)
                if (fixed_state(model, optimizer) != initial_fixed
                        or any(parameter.grad is not None
                               for name, parameter in model.named_parameters()
                               if name not in trainable)):
                    raise RuntimeError(
                        'Frozen raw/feature/normal/gate/BN/optimizer changed')
                if any(module.training for module in model.modules()):
                    raise RuntimeError('Expected all modules in eval mode')
                for phase, values in signatures.items():
                    record('step%d_%s' % (step, phase), values,
                           step == 1 and phase == 'before')
                row.update(
                    local_step=step,
                    indices=plan['train_indices'][2 * (step - 1):2 * step],
                    fixed_non_residual_state_unchanged=True,
                )
                diag.dump(directory / 'train' /
                          ('step_%04d.json' % step), row)
                summary['completed_steps'][label] = step
                diag.dump(output / 'summary.json', summary)
                if step in save_steps:
                    validate_and_save(step)
            del model, optimizer
            if cli.device == 'cuda':
                torch.cuda.empty_cache()
        summary['checks_passed'] = True
        info['status'] = summary['status'] = 'completed'
    except BaseException as error:
        info['status'] = summary['status'] = (
            'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed')
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
        raise RuntimeError('Integrity check failed')
    print('Probe completed:', output / 'summary.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe_dir', required=True)
    parser.add_argument('--steps', type=int, choices=(2, 25), default=2)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--output_dir')
    parser.add_argument('--strict_determinism', action='store_true')
    parser.add_argument('--preflight_only', action='store_true')
    cli = parser.parse_args()
    torch.set_num_threads(1)
    run(cli)


if __name__ == '__main__':
    main()
