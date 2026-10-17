"""Two-step training diagnostic; run explicitly, never updates input checkpoints."""
import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import find_dataset_def
from models import cas_mvsnet_loss
from gradient_audit import _is_sger_lite_parameter, _move_to_device, _loss_kwargs, _base_loss_weights
from gradient_audit_utils import build_audit_cases
from experiment15_d import SOURCE, dump, sha
from experiment15_d_probe import build_model, restore_optimizer, seed


def flatten(value, prefix, result):
    """Own CPU copies: later optimizer/BN updates must not mutate snapshots."""
    if torch.is_tensor(value):
        result[prefix] = value.detach().cpu().clone()
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            flatten(value[key], prefix + '/' + str(key), result)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            flatten(item, prefix + '/' + str(index), result)
    else:
        result[prefix] = value


def compare_snapshots(reference, current):
    rows = {}
    for key in sorted(set(reference) | set(current)):
        if key not in reference or key not in current:
            rows[key] = {'exact': False, 'reason': 'missing key'}
            continue
        a, b = reference[key], current[key]
        if torch.is_tensor(a) and torch.is_tensor(b):
            row = {'exact': False, 'shape': list(a.shape), 'dtype': str(a.dtype)}
            if a.shape != b.shape or a.dtype != b.dtype:
                row['reason'] = 'shape or dtype differs'
            else:
                finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                row.update(exact=bool(torch.equal(a, b)), all_finite=finite,
                           different_fraction=float((a != b).double().mean()) if a.numel() else 0.0)
                if finite:
                    delta = (a.double() - b.double()).abs()
                    row.update(max_abs_diff=float(delta.max()) if delta.numel() else 0.0,
                               mean_abs_diff=float(delta.mean()) if delta.numel() else 0.0)
                else:
                    row.update(exact=False, reason='nonfinite tensor')
            rows[key] = row
        else:
            # None gradients deliberately remain distinct from zero tensors.
            exact = type(a) is type(b) and not torch.is_tensor(a) and a == b
            rows[key] = {'exact': bool(exact), 'reference_type': type(a).__name__,
                         'current_type': type(b).__name__}
    differing = [key for key, row in rows.items() if not row['exact']]
    protected = [key for key in differing if key.startswith(('backbone/', 'input/', 'rng/'))]
    return {'exact': not differing, 'first_difference': differing[0] if differing else None,
            'first_protected_difference': protected[0] if protected else None,
            'different_entries': len(differing), 'entries': rows}


def model_snapshot(model, optimizer=None, gradients=False):
    result = {}
    for name, parameter in model.named_parameters():
        group = 'sger' if _is_sger_lite_parameter(model, name) else 'backbone'
        flatten(parameter.grad if gradients else parameter,
                group + ('/gradients/' if gradients else '/parameters/') + name, result)
        if not gradients:
            result[group + '/requires_grad/' + name] = parameter.requires_grad
        if optimizer is not None:
            flatten(optimizer.state.get(parameter, {}), group + '/optimizer/' + name, result)
    for name, buffer in model.named_buffers():
        group = 'sger' if _is_sger_lite_parameter(model, name) else 'backbone'
        flatten(buffer, group + '/buffers/' + name, result)
    if optimizer is not None:
        flatten([{key: value for key, value in group.items() if key != 'params'}
                 for group in optimizer.param_groups], 'optimizer_groups', result)
    return result


def freeze_backbone_parameters(model):
    """Keep restored optimizer groups, but Adam must skip all backbone tensors."""
    for name, parameter in model.named_parameters():
        if not _is_sger_lite_parameter(model, name):
            parameter.requires_grad_(False)
            parameter.grad = None


def set_diagnostic_train_mode(model, freeze_backbone=False):
    model.train()
    if freeze_backbone:
        # Set each module explicitly: recursive eval() on a mixed container
        # (e.g. normal_head) would also disable its trainable final-stage head.
        for name, module in model.named_modules():
            if name:
                module.training = _is_sger_lite_parameter(model, name + '.')


def frozen_backbone_check(model, optimizer, initial):
    current = model_snapshot(model, optimizer)
    before = {k: v for k, v in initial.items() if k.startswith('backbone/')}
    after = {k: v for k, v in current.items() if k.startswith('backbone/')}
    comparison = compare_snapshots(before, after)
    gradients = [name for name, p in model.named_parameters()
                 if not _is_sger_lite_parameter(model, name) and p.grad is not None]
    changed_sger = [name for name, p in model.named_parameters()
                    if _is_sger_lite_parameter(model, name)
                    and not torch.equal(initial['sger/parameters/' + name], p.detach().cpu())]
    nonzero_sger = [name for name, p in model.named_parameters()
                   if _is_sger_lite_parameter(model, name) and p.grad is not None
                   and bool((p.grad != 0).any())]
    return {'passed': comparison['exact'] and not gradients,
            'backbone_state_exact': comparison['exact'],
            'first_backbone_state_difference': comparison['first_difference'],
            'backbone_gradients_present': gradients,
            'sger_parameters_changed': changed_sger,
            'sger_nonzero_gradients': nonzero_sger}


def rng_snapshot():
    result = {}
    flatten(torch.get_rng_state(), 'rng/torch_cpu', result)
    flatten(torch.cuda.get_rng_state_all(), 'rng/torch_cuda', result)
    flatten(repr(random.getstate()), 'rng/python', result)
    state = np.random.get_state()
    flatten((state[0], torch.from_numpy(state[1].astype(np.int64)), *state[2:]), 'rng/numpy', result)
    return result


def forward_snapshot(outputs, loss):
    result = {}
    for stage in sorted(k for k in outputs if k.startswith('stage') and k[5:].isdigit()):
        values = outputs[stage]
        flatten(values.get('depth_raw', values['depth']), 'backbone/forward/' + stage + '/raw', result)
        flatten(values['photometric_confidence'], 'backbone/forward/' + stage + '/confidence', result)
        for key in ('raw_depth_residual', 'benefit_gate', 'depth_residual', 'depth_refined'):
            if key in values:
                flatten(values[key], 'sger/forward/' + stage + '/' + key, result)
    flatten(loss, 'loss/total', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe_manifest', default='outputs/experiment15_d/short_train/sample_manifest.json')
    parser.add_argument('--output_dir', default=None, help='New directory; default includes timestamp')
    parser.add_argument('--strict_determinism', action='store_true',
                        help='Fail on operations without deterministic implementations; separate diagnostic run')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone parameters and eval-mode buffers; update SGER only')
    cli = parser.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this real-data diagnostic')
    if cli.strict_determinism:
        torch.use_deterministic_algorithms(True)
    source = json.loads(Path(cli.probe_manifest).read_text())
    args = argparse.Namespace(**source['arguments'])
    if sha(args.loadckpt) != source['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from the recorded probe')
    train_indices = source['train_indices'][:4]
    val_indices = source['val_indices']
    if len(train_indices) != 4:
        raise ValueError('Need four recorded training samples for two batches')
    prefix = 'diagnose_frozen_' if cli.freeze_backbone else 'diagnose_'
    out = Path(cli.output_dir or ('outputs/experiment15_d/' + prefix + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    out.mkdir(parents=True, exist_ok=False)
    info = {'status': 'running', 'command': sys.argv, 'arguments': vars(args).copy(),
            'probe_manifest': cli.probe_manifest, 'probe_manifest_sha256': sha(cli.probe_manifest),
            'checkpoint_sha256': source['checkpoint_sha256'], 'train_indices': train_indices,
            'val_indices': val_indices, 'steps_per_run': 2,
            'runs': ['legacy_reference', 'legacy_repeat', 'stable'],
            'strict_determinism': cli.strict_determinism, 'torch': torch.__version__,
            'freeze_backbone': cli.freeze_backbone,
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
            'source_sha256': {p: sha(p) for p in SOURCE + ['experiment15_d_probe.py', 'experiment15_d_diagnose.py']}}
    dump(out / 'manifest.json', info)
    summary = {'status': 'running', 'comparisons': {}, 'freeze_backbone': cli.freeze_backbone,
               'frozen_backbone_checks': {}, 'note':
               'SGER/loss differences in A/B are expected. Protected differences locate evidence, not a cause or quality verdict.'}
    try:
        dataset_type = find_dataset_def(args.dataset)
        trainset = dataset_type(args.trainpath, args.trainlist, 'train', args.nviews, args.numdepth, args.interval_scale)
        valset = dataset_type(args.trainpath, source['val_list'], 'val', args.nviews, args.numdepth, args.interval_scale)
        # Match metadata as well as indices; actual input tensors are compared below.
        for dataset, indices, metas in ((trainset, train_indices, source['train_metas'][:4]),
                                        (valset, val_indices, source['val_metas'])):
            for index, meta in zip(indices, metas):
                if json.dumps(dataset.metas[index]) != json.dumps(meta):
                    raise ValueError('Dataset metadata differs at index %d' % index)
        checkpoint = torch.load(args.loadckpt, map_location='cpu')
        if checkpoint.get('epoch') != 15:
            raise ValueError('This diagnostic reproduces the Epoch15 probe only')
        case = build_audit_cases(15, _base_loss_weights(args), args.sger_warmup_start_epoch,
                                 args.sger_warmup_end_epoch)[0]
        info['effective_case'] = case._asdict()
        reference_dir = out / 'reference_snapshots'
        reference_dir.mkdir()
        for label, mode in (('legacy_reference', 'legacy_clamp'), ('legacy_repeat', 'legacy_clamp'),
                            ('stable', 'stable_logprob')):
            seed(args.seed)
            args.gate_bce_mode = mode
            model = build_model(args)
            model.load_state_dict(checkpoint['model'], strict=True)
            model.cuda()
            model.set_sger_residual_scale(case.sger_scale)
            optimizer, mapping = restore_optimizer(model, args, checkpoint)
            if cli.freeze_backbone:
                freeze_backbone_parameters(model)
            set_diagnostic_train_mode(model, cli.freeze_backbone)
            info[label + '_module_training'] = {name: module.training for name, module in model.named_modules()}
            dump(out / (label + '_optimizer_mapping.json'), mapping)
            summary['comparisons'][label] = {'phases': [], 'first_difference': None,
                                             'first_protected_difference': None}

            def record(phase, values):
                print(label, phase, flush=True)
                path = reference_dir / (phase + '.pt')
                if label == 'legacy_reference':
                    torch.save(values, str(path))
                    return
                result = compare_snapshots(torch.load(str(path), map_location='cpu'), values)
                dump(out / label / (phase + '.json'), result)
                entry = summary['comparisons'][label]
                entry['phases'].append({'phase': phase, **{k: v for k, v in result.items() if k != 'entries'}})
                for key in ('first_difference', 'first_protected_difference'):
                    if entry[key] is None and result[key] is not None:
                        entry[key] = {'phase': phase, 'tensor': result[key]}
                dump(out / 'summary.json', summary)

            values = model_snapshot(model, optimizer)
            frozen_initial = values.copy() if cli.freeze_backbone else None
            values.update(rng_snapshot())
            record('initial', values)
            del values
            # Reproduce the probe's step-0 evaluation and DataLoader RNG consumption.
            model.eval()
            with torch.no_grad():
                loader = DataLoader(valset, batch_sampler=[val_indices[i:i+2] for i in range(0, len(val_indices), 2)], num_workers=0)
                for index, sample in enumerate(loader):
                    values = {}
                    flatten(sample, 'input/validation', values)
                    sample = _move_to_device(sample, torch.device('cuda'))
                    outputs = model(sample['imgs'], sample['proj_matrices'], sample['depth_values'])
                    losses = cas_mvsnet_loss(outputs, sample['depth'], sample['mask'], **_loss_kwargs(args, sample, case))
                    values.update(forward_snapshot(outputs, losses[0]))
                    record('validation0_%d' % index, values)
                    del outputs, losses, sample, values
            set_diagnostic_train_mode(model, cli.freeze_backbone)
            loader = DataLoader(trainset, batch_sampler=[train_indices[:2], train_indices[2:]], num_workers=0)
            for step, sample in enumerate(loader, 1):
                set_diagnostic_train_mode(model, cli.freeze_backbone)
                optimizer.zero_grad(set_to_none=True)
                values = rng_snapshot()
                flatten(sample, 'input/train', values)
                record('step%d_input' % step, values)
                del values
                sample = _move_to_device(sample, torch.device('cuda'))
                outputs = model(sample['imgs'], sample['proj_matrices'], sample['depth_values'])
                losses = cas_mvsnet_loss(outputs, sample['depth'], sample['mask'], **_loss_kwargs(args, sample, case))
                values = forward_snapshot(outputs, losses[0])
                for name, buffer in model.named_buffers():
                    group = 'sger' if _is_sger_lite_parameter(model, name) else 'backbone'
                    flatten(buffer, group + '/buffers/' + name, values)
                record('step%d_forward' % step, values)
                del values
                if not bool(torch.isfinite(losses[0])):
                    raise RuntimeError('Nonfinite loss')
                losses[0].backward()
                record('step%d_backward' % step, model_snapshot(model, gradients=True))
                if not all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None):
                    raise RuntimeError('Nonfinite parameter gradient')
                optimizer.step()
                record('step%d_update' % step, model_snapshot(model, optimizer))
                if cli.freeze_backbone:
                    check = frozen_backbone_check(model, optimizer, frozen_initial)
                    summary['frozen_backbone_checks'].setdefault(label, []).append({'step': step, **check})
                    dump(out / 'summary.json', summary)
                    if not check['passed']:
                        raise RuntimeError('Frozen backbone changed or received gradients; see summary.json')
                del outputs, losses, sample
            del model, optimizer, frozen_initial
            torch.cuda.empty_cache()
        info['status'] = summary['status'] = 'completed'
    except BaseException as error:
        info['status'] = summary['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        info['error'] = repr(error)
        raise
    finally:
        dump(out / 'manifest.json', info)
        dump(out / 'summary.json', summary)
    print('Diagnostic completed (not an audit PASS):', out / 'summary.json', flush=True)
    for label in ('legacy_repeat', 'stable'):
        print(label, 'first protected difference:', summary['comparisons'][label]['first_protected_difference'])


if __name__ == '__main__':
    main()
