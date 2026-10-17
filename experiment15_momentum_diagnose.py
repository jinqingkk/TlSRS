"""One-update, same-batch residual Adam first-moment diagnostic."""
import argparse
import json
import math
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
from models.sger_refinement import ResidualDepthHead
from gradient_audit import _loss_kwargs, _move_to_device
from experiment15_d import dump, sha
from experiment15_d_probe import build_model, restore_optimizer, seed
from experiment15_d_diagnose import (model_snapshot, forward_snapshot, flatten,
    freeze_backbone_parameters, set_diagnostic_train_mode, frozen_backbone_check)
from experiment15_d_frozen_probe import (fingerprints, compare_fingerprints,
    validate_selection, stage_statistics, aggregate)
from experiment15_residual_recovery import (SOURCES, configure_model, audit_cases,
    proposal_forward, proposal_statistics)

RUNS = ('keep', 'keep_repeat', 'zero_residual_exp_avg')


def runtime_state(model):
    return {'buffers': {n: b.detach().clone() for n, b in model.named_buffers()},
            'modes': {n: m.training for n, m in model.named_modules()},
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'python': random.getstate(), 'numpy': np.random.get_state()}


def restore_runtime(model, state):
    with torch.no_grad():
        for name, buffer in model.named_buffers():
            buffer.copy_(state['buffers'][name])
    for name, module in model.named_modules():
        module.training = state['modes'][name]
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])


def residual_parameters(model):
    names = {id(p): n for n, p in model.named_parameters()}
    result = {names[id(p)]: p for m in model.modules()
              if isinstance(m, ResidualDepthHead) for p in m.parameters()}
    if not result:
        raise ValueError('No residual-head parameters')
    return result


def clear_first_moment(model, optimizer):
    """Mutate only existing residual exp_avg; verify all other state is exact."""
    before = fingerprints(model_snapshot(model, optimizer))
    names = residual_parameters(model)
    for name, parameter in names.items():
        state = optimizer.state.get(parameter, {})
        if not all(k in state for k in ('exp_avg', 'exp_avg_sq', 'step')):
            raise ValueError('Missing restored Adam state: ' + name)
        state['exp_avg'].zero_()
    after = fingerprints(model_snapshot(model, optimizer))
    allowed = {'sger/optimizer/' + n + '/exp_avg' for n in names}
    changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
    if not changed.issubset(allowed):
        raise RuntimeError('First-moment reset changed unrelated state')
    return {'parameters': sorted(names), 'changed_entries': sorted(changed),
            'only_residual_exp_avg_changed': True}


def direction_rows(model, before, gradients, moments, optimizer):
    rows = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        delta = parameter.detach().cpu().double() - before[name].double()
        gradient = gradients[name]
        row = {'gradient_present': gradient is not None,
               'update_l2': float(delta.norm()), 'update_abs_max': float(delta.abs().max())}
        if gradient is not None:
            gradient = gradient.double()
            norm = float(gradient.norm())
            dot = float((gradient * delta).sum())
            row.update(gradient_l2=norm, gradient_zero=not bool((gradient != 0).any()),
                       gradient_dot_update=dot,
                       cosine=dot / (norm * row['update_l2']) if norm * row['update_l2'] else None)
        if name in moments:
            old, beta = moments[name]
            state = optimizer.state[parameter]
            row.update(old_exp_avg_l2=float(old.double().norm()),
                       new_exp_avg_sq_l2=float(state['exp_avg_sq'].detach().double().norm()),
                       new_exp_avg_l2=float(state['exp_avg'].detach().double().norm()),
                       adam_step=float(state['step']))
            if gradient is not None:
                row['inherited_moment_l2'] = float((beta * old.double()).norm())
                row['current_gradient_moment_l2'] = float(((1 - beta) * gradient).norm())
        rows[name] = row
    groups = {}
    residual = set(residual_parameters(model))
    for label, names in [('residual_head', residual), ('all_trainable', set(rows))]:
        selected = [rows[n] for n in names if rows[n]['gradient_present']]
        dot = sum(r['gradient_dot_update'] for r in selected)
        gn = math.sqrt(sum(r['gradient_l2'] ** 2 for r in selected))
        dn = math.sqrt(sum(r['update_l2'] ** 2 for r in selected))
        groups[label] = {'gradient_dot_update': dot, 'gradient_l2': gn, 'update_l2': dn,
                         'cosine': dot / (gn * dn) if gn * dn else None}
    return {'parameters': rows, 'groups': groups}


def diagnostic_step(model, optimizer, forward):
    """One real update; counterfactual post-forward uses identical BN/RNG inputs."""
    initial = model_snapshot(model, optimizer)
    covered = {id(p) for group in optimizer.param_groups for p in group['params']}
    if any(p.requires_grad and id(p) not in covered for p in model.parameters()):
        raise RuntimeError('Trainable parameter missing from optimizer')
    before = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    moments = {}
    names = {id(p): n for n, p in model.named_parameters()}
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.requires_grad and 'exp_avg' in optimizer.state.get(p, {}):
                moments[names[id(p)]] = (optimizer.state[p]['exp_avg'].detach().cpu().clone(),
                                         group['betas'][0])
    optimizer.zero_grad(set_to_none=True)
    entry = runtime_state(model)
    loss, pre, pre_hashes = forward()
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('Nonfinite loss')
    loss.backward()
    gradients = {n: p.grad.detach().cpu().clone() if p.grad is not None else None
                 for n, p in model.named_parameters() if p.requires_grad}
    gradient_hashes = fingerprints(model_snapshot(model, gradients=True))
    optimizer.step()
    update_hashes = fingerprints(model_snapshot(model, optimizer))
    canonical = runtime_state(model)
    directions = direction_rows(model, before, gradients, moments, optimizer)
    del loss
    try:
        restore_runtime(model, entry)
        with torch.no_grad():
            _, post, post_hashes = forward()
    finally:
        # Keep exactly one training forward's BN update, not the measurement's.
        restore_runtime(model, canonical)
    if fingerprints(model_snapshot(model, optimizer)) != update_hashes:
        raise RuntimeError('Measurement changed optimizer/model state')
    protected = compare_fingerprints(pre_hashes, post_hashes)
    if protected['first_protected_difference']:
        raise RuntimeError('Same-batch protected output changed')
    check = frozen_backbone_check(model, optimizer, initial)
    if not check['passed']:
        raise RuntimeError('Frozen backbone changed or received gradients')
    if not check['sger_nonzero_gradients'] or not check['sger_parameters_changed']:
        raise RuntimeError('No SGER gradient/update')
    change = {k: post[k] - pre[k] for k in pre
              if isinstance(pre[k], (float, int)) and isinstance(post.get(k), (float, int))}
    return {'before': pre, 'after': post, 'after_minus_before': change,
            'directions': directions, 'frozen_backbone_check': check,
            'measurement_state_restored': True}, {
                'before': pre_hashes, 'gradients': gradient_hashes,
                'update': update_hashes, 'after': post_hashes}


def make_forward(model, sample, args, case):
    def forward():
        outputs, diagnostics = proposal_forward(model, sample, args)
        losses = cas_mvsnet_loss(outputs, sample['depth'], sample['mask'],
                                **_loss_kwargs(args, sample, case))
        stats = stage_statistics(outputs['stage3'], sample['depth']['stage3'],
                                 sample['mask']['stage3'], args.benefit_margin_ratio)
        if stats['counts']['valid_pixel_count'] == 0:
            raise RuntimeError('Empty valid mask')
        proposal_statistics(stats, outputs, sample, args, model)
        metrics = aggregate([stats])
        metrics.update(loss=float(losses[0].detach()), residual_diagnostics=diagnostics)
        metrics['residual_target_loss'] = float(losses[-1]['stage3/residual_target_loss'].detach())
        metrics['gate_benefit_loss'] = float(losses[-1]['stage3/gate_benefit_loss'].detach())
        values = forward_snapshot(outputs, losses[0])
        flatten(sample, 'input/sample', values)
        return losses[0], metrics, fingerprints(values)
    return forward


def preflight(probe_dir):
    root = Path(probe_dir)
    source = json.loads((root / 'manifest.json').read_text())
    summary = json.loads((root / 'summary.json').read_text())
    if (source['status'] != 'completed' or summary['status'] != 'completed'
            or not summary['checks_passed'] or source['steps'] != 2
            or source['experiment']['checkpoint_epoch'] != 12):
        raise ValueError('Requires completed, checked epoch12 two-step recovery probe')
    for filename, expected in source['source_sha256'].items():
        if sha(filename) != expected:
            raise ValueError('Source differs from original probe: ' + filename)
    path = root / 'leaky_relu/step_0001.ckpt'
    checkpoint = torch.load(str(path), map_location='cpu')
    args = checkpoint['arguments']
    if (checkpoint['epoch'] != 12 or checkpoint['probe_step'] != 1
            or checkpoint['gate_bce_mode'] != 'legacy_clamp'
            or not checkpoint['freeze_backbone']
            or args['sger_residual_final_activation'] != 'leaky_relu'
            or args['sger_residual_negative_slope'] != 0.01):
        raise ValueError('Unexpected source snapshot configuration')
    for key, value in source['arguments'].items():
        if key != 'sger_residual_final_activation' and args.get(key) != value:
            raise ValueError('Snapshot arguments differ: ' + key)
    # Bind the snapshot to the original run's parameter/BN/optimizer evidence.
    model_args = argparse.Namespace(**args)
    seed(model_args.seed)
    model = build_model(model_args)
    configure_model(model, 'leaky_relu', model_args)
    model.load_state_dict(checkpoint['model'], strict=True)
    optimizer, _ = restore_optimizer(model, model_args, checkpoint)
    freeze_backbone_parameters(model)
    recorded = json.loads((root/'leaky_relu/fingerprints/step1_update.json').read_text())
    if fingerprints(model_snapshot(model, optimizer)) != recorded:
        raise ValueError('Snapshot differs from original step1 fingerprints')
    return source, checkpoint, path


def run(cli):
    source, checkpoint, path = preflight(cli.probe_dir)
    if cli.preflight_only:
        print('Preflight passed:', path, 'sha256:', sha(path))
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for real DTU diagnostic')
    if cli.strict_determinism:
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
            raise ValueError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8')
        torch.use_deterministic_algorithms(True)
    args = argparse.Namespace(**checkpoint['arguments'])
    case = audit_cases(args)[0]
    if case._asdict() != source['effective_case']:
        raise ValueError('Effective schedule differs from source')
    out = Path(cli.output_dir or ('outputs/experiment15_residual_recovery/momentum_epoch12_' +
                                 datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    out.mkdir(parents=True, exist_ok=False)
    input_hashes = {str(p): sha(p) for p in (path, Path(cli.probe_dir)/'manifest.json',
                                            Path(cli.probe_dir)/'summary.json',
                                            Path(cli.probe_dir)/'leaky_relu/fingerprints/step1_update.json')}
    info = {'status': 'running', 'command': sys.argv, 'source_probe': str(cli.probe_dir),
            'input_sha256': input_hashes, 'source_sha256': {p: sha(p) for p in SOURCES + [__file__]},
            'arguments': vars(args).copy(), 'effective_case': case._asdict(),
            'runs': RUNS, 'updates_per_run': 1, 'strict_determinism': cli.strict_determinism,
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'train_indices': source['train_indices'][2:4],
            'train_metas': source['train_metas'][2:4],
            'note': 'Local paired diagnostic, not exact historical RNG replay. '
                    'Post-update same-batch measurement restores pre-forward BN/RNG; '
                    'canonical state retains one training forward BN update.'}
    result = {'status': 'running', 'checks_passed': False, 'runs': {}, 'comparisons': {}}
    dump(out/'manifest.json', info)
    dump(out/'summary.json', result)
    try:
        seed(args.seed)
        dataset = find_dataset_def(args.dataset)(args.trainpath, args.trainlist, 'train',
                                                args.nviews, args.numdepth, args.interval_scale)
        validate_selection(dataset, info['train_indices'], info['train_metas'], 2)
        sample = next(iter(DataLoader(dataset, batch_sampler=[info['train_indices']], num_workers=0)))
        sample = _move_to_device(sample, torch.device('cuda'))
        reference = None
        for label in RUNS:
            seed(args.seed)
            model = build_model(args).cuda()
            configure_model(model, 'leaky_relu', args)
            model.load_state_dict(checkpoint['model'], strict=True)
            model.set_sger_residual_scale(case.sger_scale)
            optimizer, mapping = restore_optimizer(model, args, checkpoint)
            freeze_backbone_parameters(model)
            set_diagnostic_train_mode(model, True)
            initial = fingerprints(model_snapshot(model, optimizer))
            if reference is not None and initial != reference['initial']:
                raise RuntimeError('Initial state mismatch')
            intervention = clear_first_moment(model, optimizer) if label == RUNS[2] else None
            dump(out/label/'optimizer_mapping.json', mapping)
            dump(out/label/'optimizer_groups.json', [
                {k: v for k, v in g.items() if k != 'params'} for g in optimizer.param_groups])
            dump(out/label/'module_training.json', {n: m.training for n, m in model.named_modules()})
            row, hashes = diagnostic_step(model, optimizer, make_forward(model, sample, args, case))
            row['intervention'] = intervention
            hashes['initial'] = initial
            if reference is None:
                reference = hashes
            else:
                comparisons = {phase: compare_fingerprints(reference[phase], h) for phase, h in hashes.items()}
                result['comparisons'][label] = comparisons
                for phase, comparison in comparisons.items():
                    require_exact = label == 'keep_repeat' or phase in ('initial', 'before', 'gradients')
                    if comparison['first_protected_difference'] or (require_exact and not comparison['exact']):
                        raise RuntimeError('Paired check failed: %s/%s' % (label, phase))
            result['runs'][label] = row
            dump(out/label/'fingerprints.json', hashes)
            dump(out/label/'metrics.json', row)
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'epoch': 12, 'diagnostic_updates': 1, 'arguments': vars(args).copy(),
                        'intervention': intervention, 'source_checkpoint': str(path)}, str(out/label/'after.ckpt'))
            dump(out/'summary.json', result)
            print(label, 'same-batch change:', row['after_minus_before'], flush=True)
            del model, optimizer
            torch.cuda.empty_cache()
        result['checks_passed'] = True
        info['status'] = result['status'] = 'completed'
    except BaseException as error:
        info['status'] = result['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        info['error'] = result['error'] = repr(error)
        raise
    finally:
        unchanged = all(sha(p) == digest for p, digest in input_hashes.items())
        result['input_files_unchanged'] = unchanged
        if not unchanged:
            info['status'] = result['status'] = 'failed'
            result['checks_passed'] = False
            result['error'] = 'Input file hash changed'
        dump(out/'manifest.json', info)
        dump(out/'summary.json', result)
    print('Diagnostic completed:', out/'summary.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe_dir', required=True, help='Completed epoch12 two-step recovery directory')
    parser.add_argument('--output_dir', default=None, help='New directory only')
    parser.add_argument('--strict_determinism', action='store_true')
    parser.add_argument('--preflight_only', action='store_true', help='Read-only source/config checks; no CUDA needed')
    cli = parser.parse_args()
    torch.set_num_threads(1)
    run(cli)


if __name__ == '__main__':
    main()
