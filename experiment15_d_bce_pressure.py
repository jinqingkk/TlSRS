"""Epoch12 fixed-LeakyReLU BCE pressure comparison; no production changes."""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import experiment15_momentum_diagnose as diag
from experiment15_d_diagnose import rng_snapshot

from experiment15_momentum_probe import load_plan
import experiment15_residual_sign_audit as sign_audit
from models.module import bounded_residual_target
from models.module import (residual_benefit_target, balanced_binary_cross_entropy,
                           stable_gate_binary_cross_entropy)

RUNS = ('legacy_reference', 'legacy_repeat', 'stable')


def gate_logit_statistics(stage, valid, saturated, selected):
    """Detached diagnostics on current masks; never feed GT into the model."""
    active = torch.where(stage['region_a'].bool(), stage['gate_logits_a'],
                         stage['gate_logits_b']).detach()
    gate = stage['benefit_gate'].detach()
    result = {}
    for name, mask in (('valid', valid), ('saturated', saturated),
                       ('saturated_positive', selected)):
        count = int(mask.sum())
        row = {'count': count}
        for key, tensor in (('active_logit', active), ('gate', gate)):
            values = tensor[mask].double()
            if not bool(torch.isfinite(values).all()):
                raise RuntimeError('Nonfinite gate/logit diagnostic')
            total = float(values.sum())
            row[key + '_sum'] = total
            row[key + '_mean'] = total / count if count else None
            row[key + '_min'] = float(values.min()) if count else None
            row[key + '_max'] = float(values.max()) if count else None
        result[name] = row
    return result


def pool_gate_logits(rows):
    """Pool sufficient statistics by pixel count, including empty subsets."""
    result = {}
    for name in ('valid', 'saturated', 'saturated_positive'):
        groups = [r['gate_pressure']['logits'][name] for r in rows]
        count = sum(g['count'] for g in groups)
        row = {'count': count}
        for key in ('active_logit', 'gate'):
            total = sum(g[key + '_sum'] for g in groups)
            row[key + '_sum'] = total
            row[key + '_mean'] = total / count if count else None
            row[key + '_min'] = min(g[key + '_min'] for g in groups if g['count']) if count else None
            row[key + '_max'] = max(g[key + '_max'] for g in groups if g['count']) if count else None
        result[name] = row
    return result


def pressure_statistics(stage, gt, mask, args, case):
    valid = mask > .5
    target = residual_benefit_target(stage['depth_raw'], stage['raw_depth_residual'],
                                    gt, stage['depth_interval'], args.benefit_margin_ratio)
    saturated = valid & (stage['benefit_gate'].detach() < 1e-6)
    selected = saturated & target
    counts = {'saturated_count': int(saturated.sum()),
              'saturated_positive_count': int(selected.sum()),
              'logits': gate_logit_statistics(stage, valid, saturated, selected)}
    raw_error = (stage['depth_raw'].detach() - gt).abs()
    refined_error = (stage['depth_refined'].detach() - gt).abs()
    counts['saturated_error_delta_sum'] = float((raw_error-refined_error)[saturated].double().sum())
    if torch.is_grad_enabled():
        if args.gate_bce_mode == 'stable_logprob':
            loss = stable_gate_binary_cross_entropy(
                stage['gate_logits_a'], stage['gate_logits_b'], stage['region_a'],
                stage['region_b_weight'], stage['gate_valid_mask'], target, valid)
        else:
            loss = balanced_binary_cross_entropy(stage['benefit_gate'], target, valid)
        weight = case.loss_weights['gate_benefit_loss_weight'] * case.gate_benefit_scale
        gradients = torch.autograd.grad(weight*loss,
            (stage['gate_logits_a'], stage['gate_logits_b']), retain_graph=True)
        region_a = stage['region_a'].bool()
        active = torch.where(region_a, gradients[0], gradients[1]).detach()
        if not bool(torch.isfinite(active).all()):
            raise RuntimeError('Nonfinite isolated BCE logit gradient')
        counts.update(saturated_positive_correct_gradient_count=int((active[selected] < 0).sum()),
                      saturated_positive_wrong_gradient_count=int((active[selected] > 0).sum()),
                      saturated_positive_zero_gradient_count=int((active[selected] == 0).sum()),
                      gradient_scope='isolated gate BCE with actual epoch12 weight; no parameter update')
    return counts


def require_pressure(metrics, mode):
    pressure = metrics['gate_pressure']
    n = pressure['saturated_positive_count']
    if n == 0:
        raise RuntimeError('Starting batch has no saturated positive benefit targets')
    expected = ('saturated_positive_correct_gradient_count' if mode == 'stable_logprob'
                else 'saturated_positive_zero_gradient_count')
    if pressure[expected] != n:
        raise RuntimeError('Starting BCE gradient does not match pressure hypothesis')


def make_forward(model, sample, args, case):
    def forward():
        outputs, diagnostics = diag.proposal_forward(model, sample, args)
        losses = diag.cas_mvsnet_loss(outputs, sample['depth'], sample['mask'],
                                     **diag._loss_kwargs(args, sample, case))
        stage = outputs['stage3']
        gt, mask = sample['depth']['stage3'], sample['mask']['stage3']
        stats = diag.stage_statistics(stage, gt, mask, args.benefit_margin_ratio)
        diag.proposal_statistics(stats, outputs, sample, args, model)
        metrics = diag.aggregate([stats])
        target = bounded_residual_target(stage['depth_raw'], gt, stage['depth_interval'], args.residual_target_ratio)
        groups, _ = sign_audit.signed_statistics(stage['raw_depth_residual'], target, mask > .5)
        metrics['signed_groups'] = groups
        metrics.update(loss=float(losses[0].detach()), residual_diagnostics=diagnostics,
                       residual_target_loss=float(losses[-1]['stage3/residual_target_loss']),
                       gate_benefit_loss=float(losses[-1]['stage3/gate_benefit_loss']),
                       effective_gate_benefit_weight=(case.loss_weights['gate_benefit_loss_weight'] *
                                                      case.gate_benefit_scale),
                       gate_pressure=pressure_statistics(stage, gt, mask, args, case),
                       gate_bce_mode=args.gate_bce_mode)
        values = diag.forward_snapshot(outputs, losses[0])
        diag.flatten(sample, 'input/sample', values)
        return losses[0], metrics, diag.fingerprints(values)
    return forward


def pool_metrics(rows):
    """Pixel-weight quality means; pool counts before precision/recall ratios."""
    counts = ('valid_pixel_count', 'improved_count', 'worsened_count',
              'target_positive_count', 'predicted_positive_count', 'true_positive_count')
    count = {k: sum(r[k] for r in rows) for k in counts}
    n = count['valid_pixel_count']
    if not n:
        raise ValueError('Empty validation mask')
    result = dict(count)
    excluded = set(counts) | {'gate_precision', 'gate_recall', 'gate_on_improved_mean',
                              'gate_on_worsened_mean', 'residual_diagnostics'}
    for key in rows[0]:
        if key not in excluded and all(isinstance(r[key], (int, float)) for r in rows):
            result[key] = sum(r[key] * r['valid_pixel_count'] for r in rows) / n
    def ratio(a, b):
        return a / b if b else None
    result['gate_precision'] = ratio(count['true_positive_count'], count['predicted_positive_count'])
    result['gate_recall'] = ratio(count['true_positive_count'], count['target_positive_count'])
    for region in ('improved', 'worsened'):
        key = 'gate_on_' + region + '_mean'
        result[key] = ratio(sum((r[key] or 0.0) * r[region + '_count'] for r in rows),
                            count[region + '_count'])
    return result


def evaluate(model, optimizer, dataset, args, case, plan, record, step, device):
    state = diag.runtime_state(model)
    before = diag.fingerprints(diag.model_snapshot(model, optimizer))
    rng_before = diag.fingerprints(rng_snapshot())
    rows, scenes = [], {}
    try:
        model.eval()
        with torch.no_grad():
            loader = DataLoader(dataset, batch_sampler=[plan['val_indices'][i:i+2]
                                for i in range(0, 6, 2)], num_workers=0)
            for batch, sample in enumerate(loader):
                sample = diag._move_to_device(sample, device)
                _, metrics, hashes = make_forward(model, sample, args, case)()
                record('val%d_batch%d' % (step, batch), hashes, step == 0)
                metas = plan['val_metas'][2*batch:2*batch+2]
                if metas[0][0] != metas[1][0]:
                    raise ValueError('Validation batches must each contain one scene')
                rows.append(metrics)
                scenes.setdefault(metas[0][0], []).append(metrics)
    finally:
        diag.restore_runtime(model, state)
    if before != diag.fingerprints(diag.model_snapshot(model, optimizer)):
        raise RuntimeError('Validation changed model/optimizer')
    if rng_before != diag.fingerprints(rng_snapshot()):
        raise RuntimeError('Validation changed RNG')
    return {'pooled': pool_metrics(rows), 'scenes': {s: pool_metrics(r) for s, r in scenes.items()},
            'batches': rows, 'signed': sign_audit.summarize([r['signed_groups'] for r in rows]),
            'signed_scenes': {s: sign_audit.summarize([r['signed_groups'] for r in group])
                              for s, group in scenes.items()},
            'gate_logits': pool_gate_logits(rows),
            'gate_logits_scenes': {s: pool_gate_logits(group) for s, group in scenes.items()},
            'gate_pressure': {k: sum(r['gate_pressure'][k] for r in rows)
                              for k in ('saturated_count', 'saturated_positive_count',
                                        'saturated_error_delta_sum')},
            'state_and_rng_restored': True}


def run(cli):
    checkpoint, path, args, case, plan, input_hashes = load_plan(cli.probe_dir)
    save_steps = tuple(s for s in (0, 1, 2, 5, 10, 25) if s <= cli.steps)
    plan = dict(plan, train_indices=plan['train_indices'][:2*cli.steps],
                train_metas=plan['train_metas'][:2*cli.steps])
    if cli.preflight_only:
        print('Preflight passed:', path, 'updates:', cli.steps,
              'first batch:', plan['train_indices'][:2], 'validation:', plan['val_indices'])
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the real DTU probe')
    if cli.strict_determinism:
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
            raise ValueError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8')
        torch.use_deterministic_algorithms(True)
    out = Path(cli.output_dir or ('outputs/experiment15_d/bce_pressure_%dsteps_epoch12_' % cli.steps +
                                 datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    out.mkdir(parents=True, exist_ok=False)
    info = dict(status='running', command=sys.argv, source_checkpoint=str(path),
                input_sha256=input_hashes, arguments=vars(args).copy(), effective_case=case._asdict(),
                steps=cli.steps, save_steps=save_steps, runs=RUNS, sample_plan=plan,
                reset_policy='Restore all Adam state; no reset, no scheduler',
                loss_policy='Original pixel-mean residual target; only gate BCE mode differs',
                source_sha256={p: diag.sha(p) for p in diag.SOURCES +
                               ['experiment15_momentum_diagnose.py', 'experiment15_momentum_probe.py', 'experiment15_residual_sign_audit.py', __file__]},
                strict_determinism=cli.strict_determinism, torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0), cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                note='Local steps count additional updates after source step1; validation restores BN/RNG. '
                     'Fixed newly seeded paired run, not historical RNG replay. '
                     'Logit subsets use current masks, not fixed initial pixels; '
                     'active logit selects region A or B, while gate also includes geometry weights.')
    summary = dict(status='running', checks_passed=False, comparisons={}, validation={},
                   completed_steps={}, interventions={})
    diag.dump(out/'manifest.json', info)
    diag.dump(out/'summary.json', summary)
    try:
        dataset_type = diag.find_dataset_def(args.dataset)
        trainset = dataset_type(args.trainpath, args.trainlist, 'train', args.nviews, args.numdepth, args.interval_scale)
        valset = dataset_type(args.trainpath, plan['val_list'], 'val', args.nviews, args.numdepth, args.interval_scale)
        diag.validate_selection(trainset, plan['train_indices'], plan['train_metas'], cli.steps*2)
        diag.validate_selection(valset, plan['val_indices'], plan['val_metas'], 6)
        for label in RUNS:
            diag.seed(args.seed)
            model = diag.build_model(args).cuda()
            diag.configure_model(model, 'leaky_relu', args)
            # The historical activation helper resets BCE to legacy_clamp.
            args.gate_bce_mode = 'stable_logprob' if label == 'stable' else 'legacy_clamp'
            model.load_state_dict(checkpoint['model'], strict=True)
            model.set_sger_residual_scale(case.sger_scale)
            optimizer, mapping = diag.restore_optimizer(model, args, checkpoint)
            diag.freeze_backbone_parameters(model)
            diag.set_diagnostic_train_mode(model, True)
            directory = out/label
            def record(phase, hashes, exact=False):
                diag.dump(directory/'fingerprints'/(phase+'.json'), hashes)
                if label == 'legacy_reference':
                    return
                reference = json.loads((out/'legacy_reference/fingerprints'/(phase+'.json')).read_text())
                if label == 'stable' and (phase.startswith('val0_') or phase == 'step1_before'):
                    reference = {k:v for k,v in reference.items() if k != 'loss/total'}
                    hashes = {k:v for k,v in hashes.items() if k != 'loss/total'}
                    exact = True
                comparison = diag.compare_fingerprints(reference, hashes)
                summary['comparisons'].setdefault(label, {})[phase] = comparison
                if comparison['first_protected_difference'] or ((exact or label == 'legacy_repeat') and not comparison['exact']):
                    raise RuntimeError('Paired invariance failed: %s/%s' % (label, phase))
            record('initial', diag.fingerprints(diag.model_snapshot(model, optimizer)), True)
            intervention = dict(gate_bce_mode=args.gate_bce_mode, optimizer_reset=False)
            summary['interventions'][label] = intervention
            initial = diag.model_snapshot(model, optimizer)
            diag.dump(directory/'optimizer_mapping.json', mapping)
            diag.dump(directory/'optimizer_groups.json', [{k: v for k, v in g.items() if k != 'params'}
                                                         for g in optimizer.param_groups])
            diag.dump(directory/'module_training.json', {n: m.training for n, m in model.named_modules()})
            def validate_and_save(step):
                row = evaluate(model, optimizer, valset, args, case, plan, record, step, torch.device('cuda'))
                summary['validation'].setdefault(str(step), {})[label] = row
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=12,
                                local_step=step, source_probe_step=1, arguments=vars(args).copy(),
                                freeze_backbone=True, gate_bce_mode=args.gate_bce_mode, intervention=intervention),
                           str(directory/('step_%04d.ckpt' % step)))
                diag.dump(out/'summary.json', summary)
                print(label, 'validation', step, 'proposal_gain', row['pooled']['proposal_gain_mean'],
                      'raw-refined', row['pooled']['raw_to_refined_error_delta'],
                      'active_logit_mean', row['gate_logits']['valid']['active_logit_mean'],
                      'gate_mean', row['gate_logits']['valid']['gate_mean'], flush=True)
            validate_and_save(0)
            loader = DataLoader(trainset, batch_sampler=[plan['train_indices'][i:i+2]
                                for i in range(0, cli.steps*2, 2)], num_workers=0)
            for step, sample in enumerate(loader, 1):
                diag.set_diagnostic_train_mode(model, True)
                record('step%d_rng' % step, diag.fingerprints(rng_snapshot()), True)
                sample = diag._move_to_device(sample, torch.device('cuda'))
                forward = make_forward(model, sample, args, case)
                def checked_forward():
                    result = forward()
                    if step == 1 and torch.is_grad_enabled():
                        require_pressure(result[1], args.gate_bce_mode)
                    return result
                row, hashes = diag.diagnostic_step(model, optimizer, checked_forward)
                for phase, values in hashes.items():
                    record('step%d_%s' % (step, phase), values, step == 1 and phase == 'before')
                check = diag.frozen_backbone_check(model, optimizer, initial)
                row['frozen_since_initial'] = check
                if not check['passed']:
                    raise RuntimeError('Backbone changed since initialization')
                row.update(local_step=step, indices=plan['train_indices'][2*(step-1):2*step])
                diag.dump(directory/'train'/('step_%04d.json' % step), row)
                summary['completed_steps'][label] = step
                diag.dump(out/'summary.json', summary)
                if step in save_steps:
                    validate_and_save(step)
                print(label, 'step', step, '/', cli.steps, flush=True)
            del model, optimizer
            torch.cuda.empty_cache()
        summary['checks_passed'] = True
        info['status'] = summary['status'] = 'completed'
    except BaseException as error:
        info['status'] = summary['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        summary['error'] = info['error'] = repr(error)
        raise
    finally:
        unchanged = all(diag.sha(p) == h for p, h in input_hashes.items())
        summary['input_files_unchanged'] = unchanged
        if not unchanged:
            info['status'] = summary['status'] = 'failed'
            summary['checks_passed'] = False
            summary['error'] = 'Input file hash changed'
        diag.dump(out/'manifest.json', info)
        diag.dump(out/'summary.json', summary)
    if not summary['checks_passed']:
        raise RuntimeError(summary.get('error', 'Probe checks failed'))
    print('Probe completed:', out/'summary.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe_dir', required=True)
    parser.add_argument('--steps', type=int, choices=(2, 25), default=2)
    parser.add_argument('--output_dir', default=None, help='Must be a new directory')
    parser.add_argument('--strict_determinism', action='store_true')
    parser.add_argument('--preflight_only', action='store_true')
    cli = parser.parse_args()
    torch.set_num_threads(1)
    run(cli)


if __name__ == '__main__':
    main()
