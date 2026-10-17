"""Independent frozen-backbone 200-step experiment; invoked by the probe CLI."""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import find_dataset_def
from models import cas_mvsnet_loss
from models.module import residual_benefit_target
from gradient_audit import _base_loss_weights, _loss_kwargs, _move_to_device
from gradient_audit_utils import build_audit_cases
from experiment15_d import SOURCE, dump, sha
from experiment15_d_probe import build_model, restore_optimizer, seed, verify_prior_audits
from experiment15_d_diagnose import (flatten, model_snapshot, rng_snapshot, forward_snapshot,
    freeze_backbone_parameters, set_diagnostic_train_mode, frozen_backbone_check)

STEPS = 200
SAVE_STEPS = (0, 25, 50, 100, 200)
RUNS = (('legacy_reference', 'legacy_clamp'), ('legacy_repeat', 'legacy_clamp'),
        ('stable', 'stable_logprob'))


def fingerprints(values):
    """Hash dtype, shape and bytes; reject nonfinite tensors before comparison."""
    result = {}
    for key, value in values.items():
        if torch.is_tensor(value):
            value = value.detach().cpu().contiguous()
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError('Nonfinite tensor: ' + key)
            header = str(value.dtype) + str(tuple(value.shape))
            payload = header.encode() + value.numpy().tobytes()
        else:
            payload = (type(value).__name__ + repr(value)).encode()
        result[key] = hashlib.sha256(payload).hexdigest()
    return result


def compare_fingerprints(reference, current):
    differing = sorted(k for k in set(reference) | set(current)
                       if reference.get(k) != current.get(k))
    protected = [k for k in differing if k.startswith(('backbone/', 'input/', 'rng/'))]
    return {'exact': not differing, 'different_entries': len(differing),
            'first_difference': differing[0] if differing else None,
            'first_protected_difference': protected[0] if protected else None}


def stage_statistics(stage, gt, mask, margin):
    """Sufficient statistics for pooled stage3 valid-pixel metrics."""
    raw, refined = stage['depth_raw'].detach(), stage['depth_refined'].detach()
    proposal, gate = stage['raw_depth_residual'].detach(), stage['benefit_gate'].detach()
    valid = mask > 0.5
    target = residual_benefit_target(raw, proposal, gt, stage['depth_interval'], margin)
    predicted = gate >= 0.5
    raw_error, refined_error = (raw - gt).abs(), (refined - gt).abs()
    improved, worsened = valid & (refined_error < raw_error), valid & (refined_error > raw_error)
    values = {'raw_abs_error': raw_error, 'refined_abs_error': refined_error,
              'raw_to_refined_error_delta': raw_error - refined_error,
              'mean_abs_raw_residual': proposal.abs(),
              'mean_abs_depth_residual': stage['depth_residual'].detach().abs(),
              'benefit_gate_mean': gate,
              'residual_sign_accuracy': ((proposal * (gt - raw)) > 0).float(),
              'proposal_gain_mean': raw_error - (raw + proposal - gt).abs()}
    sums = {k: float(v[valid].double().sum()) for k, v in values.items()}
    sums.update(gate_on_improved_sum=float(gate[improved].double().sum()),
                gate_on_worsened_sum=float(gate[worsened].double().sum()))
    counts = {'valid_pixel_count': int(valid.sum()), 'improved_count': int(improved.sum()),
              'worsened_count': int(worsened.sum()), 'target_positive_count': int((valid & target).sum()),
              'predicted_positive_count': int((valid & predicted).sum()),
              'true_positive_count': int((valid & target & predicted).sum())}
    return {'sums': sums, 'counts': counts}


def aggregate(rows):
    counts = {k: sum(r['counts'][k] for r in rows) for k in rows[0]['counts']}
    sums = {k: sum(r['sums'][k] for r in rows) for k in rows[0]['sums']}
    n = counts['valid_pixel_count']
    def ratio(a, b):
        return a / b if b else None
    result = {k: ratio(v, n) for k, v in sums.items() if not k.endswith('_sum')}
    result.update(counts)
    result.update(gate_precision=ratio(counts['true_positive_count'], counts['predicted_positive_count']),
                  gate_recall=ratio(counts['true_positive_count'], counts['target_positive_count']),
                  benefit_target_positive_ratio=ratio(counts['target_positive_count'], n),
                  refined_improved_pixel_ratio=ratio(counts['improved_count'], n),
                  refined_worsened_pixel_ratio=ratio(counts['worsened_count'], n),
                  gate_on_improved_mean=ratio(sums['gate_on_improved_sum'], counts['improved_count']),
                  gate_on_worsened_mean=ratio(sums['gate_on_worsened_sum'], counts['worsened_count']))
    return result


def validate_selection(dataset, indices, metas, expected):
    if len(indices) != expected or len(metas) != expected:
        raise ValueError('Recorded sample count mismatch: expected %d' % expected)
    for index, meta in zip(indices, metas):
        if json.dumps(dataset.metas[index]) != json.dumps(meta):
            raise ValueError('Dataset metadata differs at index %d' % index)


def run(cli, experiment=None):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the real-data probe')
    if cli.strict_determinism:
        torch.use_deterministic_algorithms(True)
    verify_prior_audits()
    source = (experiment['source'] if experiment else
              json.loads(Path(cli.probe_manifest).read_text()))
    steps = experiment['steps'] if experiment else STEPS
    save_steps = (0, 1, 2) if experiment and steps == 2 else SAVE_STEPS
    runs = experiment['runs'] if experiment else RUNS
    epoch = experiment['epoch'] if experiment else 15
    reference_label, repeat_label = runs[0][0], runs[1][0]
    args = argparse.Namespace(**source['arguments'])
    if sha(args.loadckpt) != source['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from the recorded probe')
    if args.batch_size != 2 or args.sger_residual_final_activation != 'relu':
        raise ValueError('This experiment requires the recorded batch_size=2 and ReLU')
    out = Path(cli.output_dir or ('outputs/experiment15_d/frozen_probe_' +
                                 datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    out.mkdir(parents=True, exist_ok=False)
    info = {'status': 'running', 'command': sys.argv, 'arguments': vars(args).copy(),
            'probe_manifest': cli.probe_manifest, 'probe_manifest_sha256': sha(cli.probe_manifest),
            'checkpoint_sha256': source['checkpoint_sha256'], 'freeze_backbone': True,
            'strict_determinism': cli.strict_determinism, 'steps': steps, 'save_steps': save_steps,
            'runs': runs, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'source_sha256': {p: sha(p) for p in SOURCE + ['experiment15_d_probe.py',
                                'experiment15_d_frozen_probe.py', 'experiment15_d_diagnose.py']},
            'schedule': 'fixed epoch%d scales; restored Adam groups, momentum and learning rates; no scheduler' % epoch}
    summary = {'status': 'running', 'checks_passed': False, 'comparisons': {}, 'validation': {},
               'completed_steps': {}, 'freeze_backbone': True}
    if experiment:
        info['experiment'] = experiment['metadata']
        info['source_sha256'].update(experiment['source_sha256'])
    dump(out / 'manifest.json', info)
    try:
        dataset_type = find_dataset_def(args.dataset)
        trainset = dataset_type(args.trainpath, args.trainlist, 'train', args.nviews, args.numdepth, args.interval_scale)
        valset = dataset_type(args.trainpath, source['val_list'], 'val', args.nviews, args.numdepth, args.interval_scale)
        train_indices, val_indices = source['train_indices'], source['val_indices']
        validate_selection(trainset, train_indices, source['train_metas'], steps * 2)
        validate_selection(valset, val_indices, source['val_metas'], 6)
        info.update({k: source[k] for k in ('train_indices', 'train_metas', 'val_indices', 'val_metas', 'val_list')})
        checkpoint = torch.load(args.loadckpt, map_location='cpu')
        if checkpoint.get('epoch') != epoch:
            raise ValueError('Checkpoint epoch differs from experiment epoch')
        case = build_audit_cases(epoch, _base_loss_weights(args), args.sger_warmup_start_epoch,
                                 args.sger_warmup_end_epoch)[0]
        info['effective_case'] = case._asdict()
        dump(out / 'manifest.json', info)
        reference = {}
        for label, mode in runs:
            seed(args.seed)
            args.gate_bce_mode = mode
            model = build_model(args).cuda()
            if experiment:
                experiment['configure_model'](model, label, args)
            model.load_state_dict(checkpoint['model'], strict=True)
            model.set_sger_residual_scale(case.sger_scale)
            optimizer, mapping = restore_optimizer(model, args, checkpoint)
            freeze_backbone_parameters(model)
            set_diagnostic_train_mode(model, True)
            initial = model_snapshot(model, optimizer)
            covered = {id(p) for g in optimizer.param_groups for p in g['params']}
            if any(p.requires_grad and id(p) not in covered for p in model.parameters()):
                raise RuntimeError('Trainable parameter missing from optimizer')
            directory = out / label
            dump(directory / 'optimizer_mapping.json', mapping)
            dump(directory / 'module_training.json', {n: m.training for n, m in model.named_modules()})
            records = {'train': [], 'validation': {}, 'frozen_backbone_checks': []}
            phases = summary['comparisons'][label] = {}

            def record(phase, values):
                hashes = fingerprints(values)
                dump(directory / 'fingerprints' / (phase + '.json'), hashes)
                if label == reference_label:
                    reference[phase] = hashes
                    return
                result = compare_fingerprints(reference[phase], hashes)
                phases[phase] = result
                dump(out / 'summary.json', summary)
                # Stable BCE may change loss/SGER, but never the protected path.
                if result['first_protected_difference'] or ((label == repeat_label or phase == 'initial') and not result['exact']):
                    raise RuntimeError('Paired invariance failed: %s %s' % (label, phase))

            initial_values = dict(initial)
            initial_values.update(rng_snapshot())
            record('initial', initial_values)
            del initial_values

            def forward(sample, phase):
                values = rng_snapshot()
                flatten(sample, 'input/sample', values)
                sample = _move_to_device(sample, torch.device('cuda'))
                if experiment:
                    outputs, diagnostics = experiment['forward'](model, sample, args)
                else:
                    outputs = model(sample['imgs'], sample['proj_matrices'], sample['depth_values'])
                losses = cas_mvsnet_loss(outputs, sample['depth'], sample['mask'], **_loss_kwargs(args, sample, case))
                stage = outputs['stage3']
                expected = sample['depth']['stage3'].shape
                for key in ('depth_raw', 'depth_refined', 'depth_residual', 'raw_depth_residual', 'benefit_gate'):
                    if stage[key].shape != expected:
                        raise RuntimeError('Stage3 shape mismatch: ' + key)
                bound = args.sger_max_residual_ratio * stage['depth_interval']
                tolerance = 1e-6
                if bool((stage['raw_depth_residual'].detach().abs() > bound + tolerance).any()):
                    raise RuntimeError('Residual proposal exceeds configured bound')
                if bool((stage['depth_residual'].detach().abs() > bound * case.sger_scale + tolerance).any()):
                    raise RuntimeError('Effective residual exceeds configured bound')
                values.update(forward_snapshot(outputs, losses[0]))
                record(phase, values)
                stats = stage_statistics(outputs['stage3'], sample['depth']['stage3'],
                                         sample['mask']['stage3'], args.benefit_margin_ratio)
                if experiment:
                    experiment['statistics'](stats, outputs, sample, args, model)
                    stats['residual_diagnostics'] = diagnostics
                stats['loss'] = float(losses[0].detach())
                stats['loss_extras'] = {k: float(v.detach()) for k, v in losses[-1].items()
                                       if torch.is_tensor(v) and v.numel() == 1}
                return losses[0], stats

            def evaluate(step):
                model.eval()
                rows = []
                with torch.no_grad():
                    loader = DataLoader(valset, batch_sampler=[val_indices[i:i+2] for i in range(0, 6, 2)], num_workers=0)
                    for batch, sample in enumerate(loader):
                        _, stats = forward(sample, 'val%d_batch%d' % (step, batch))
                        rows.append(stats)
                records['validation'][str(step)] = rows
                summary['validation'].setdefault(str(step), {})[label] = aggregate(rows)
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'epoch': epoch, 'probe_step': step, 'gate_bce_mode': mode,
                            'freeze_backbone': True, 'arguments': vars(args).copy()},
                           str(directory / ('step_%04d.ckpt' % step)))
                dump(directory / 'metrics.json', records)
                dump(out / 'summary.json', summary)
                set_diagnostic_train_mode(model, True)
                print(label, 'validation', step, summary['validation'][str(step)][label], flush=True)

            evaluate(0)
            loader = DataLoader(trainset, batch_sampler=[train_indices[i:i+2] for i in range(0, steps * 2, 2)], num_workers=0)
            for step, sample in enumerate(loader, 1):
                set_diagnostic_train_mode(model, True)
                optimizer.zero_grad(set_to_none=True)
                loss, stats = forward(sample, 'step%d_forward' % step)
                loss.backward()
                record('step%d_backward' % step, model_snapshot(model, gradients=True))
                if experiment:
                    stats['residual_parameter_gradients'] = experiment['gradients'](model, optimizer)
                optimizer.step()
                record('step%d_update' % step, model_snapshot(model, optimizer))
                check = frozen_backbone_check(model, optimizer, initial)
                records['frozen_backbone_checks'].append(dict(step=step, **check))
                records['train'].append(dict(step=step, **stats))
                summary['completed_steps'][label] = step
                dump(directory / 'metrics.json', records)
                if not check['passed']:
                    raise RuntimeError('Frozen backbone changed or received gradients')
                if not check['sger_nonzero_gradients'] or not check['sger_parameters_changed']:
                    raise RuntimeError('SGER has no nonzero gradients or no change from initialization')
                del loss, sample
                if step % 10 == 0:
                    print(label, 'step', step, 'delta', aggregate([stats])['raw_to_refined_error_delta'], flush=True)
                if step in save_steps:
                    evaluate(step)
            del model, optimizer, initial
            torch.cuda.empty_cache()
        info['status'] = summary['status'] = 'completed'
        summary['checks_passed'] = True
    except BaseException as error:
        info['status'] = summary['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        info['error'] = summary['error'] = repr(error)
        raise
    finally:
        dump(out / 'manifest.json', info)
        dump(out / 'summary.json', summary)
    print('Frozen probe completed; checks_passed is not a quality verdict:', out / 'summary.json', flush=True)
