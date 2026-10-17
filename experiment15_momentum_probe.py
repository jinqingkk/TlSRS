"""25-update residual first-moment recovery comparison with fixed validation."""
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

STEPS = 25
SAVE_STEPS = (0, 1, 5, 10, 25)


def load_plan(probe_dir):
    source, checkpoint, path = diag.preflight(probe_dir)
    samples_path = Path(source['probe_manifest'])
    if diag.sha(samples_path) != source['probe_manifest_sha256']:
        raise ValueError('Original sample manifest hash mismatch')
    samples = json.loads(samples_path.read_text())
    for key in ('dataset', 'trainpath', 'trainlist', 'nviews', 'numdepth', 'interval_scale', 'seed'):
        if samples['arguments'][key] != checkpoint['arguments'][key]:
            raise ValueError('Sample configuration mismatch: ' + key)
    for key in ('train_indices', 'train_metas'):
        if samples[key][:4] != source[key]:
            raise ValueError('Two-step training prefix mismatch: ' + key)
    for key in ('val_indices', 'val_metas', 'val_list'):
        if samples[key] != source[key]:
            raise ValueError('Validation selection mismatch: ' + key)
    # Snapshot already consumed the first two examples (one update).
    plan = {key: samples[key][2:2 + STEPS * 2] for key in ('train_indices', 'train_metas')}
    if any(len(plan[key]) != STEPS * 2 for key in plan):
        raise ValueError('Insufficient recorded training samples')
    plan.update({key: samples[key] for key in ('val_indices', 'val_metas', 'val_list')})
    if len(plan['val_indices']) != 6 or len(plan['val_metas']) != 6:
        raise ValueError('Expected six fixed validation samples')
    args = argparse.Namespace(**checkpoint['arguments'])
    case = diag.audit_cases(args)[0]
    if case._asdict() != source['effective_case']:
        raise ValueError('Effective schedule mismatch')
    paths = [path, samples_path, Path(probe_dir)/'manifest.json', Path(probe_dir)/'summary.json',
             Path(probe_dir)/'leaky_relu/fingerprints/step1_update.json']
    return checkpoint, path, args, case, plan, {str(p): diag.sha(p) for p in paths}


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
                _, metrics, hashes = diag.make_forward(model, sample, args, case)()
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
            'batches': rows, 'state_and_rng_restored': True}


def run(cli):
    checkpoint, path, args, case, plan, input_hashes = load_plan(cli.probe_dir)
    if cli.preflight_only:
        print('Preflight passed:', path, 'updates:', STEPS,
              'first batch:', plan['train_indices'][:2], 'validation:', plan['val_indices'])
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the real DTU probe')
    if cli.strict_determinism:
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
            raise ValueError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8')
        torch.use_deterministic_algorithms(True)
    out = Path(cli.output_dir or ('outputs/experiment15_residual_recovery/momentum_25steps_epoch12_' +
                                 datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    out.mkdir(parents=True, exist_ok=False)
    info = dict(status='running', command=sys.argv, source_checkpoint=str(path),
                input_sha256=input_hashes, arguments=vars(args).copy(), effective_case=case._asdict(),
                steps=STEPS, save_steps=SAVE_STEPS, runs=diag.RUNS, sample_plan=plan,
                reset_policy='Residual exp_avg reset once before local step0, only in B; no scheduler',
                source_sha256={p: diag.sha(p) for p in diag.SOURCES +
                               ['experiment15_momentum_diagnose.py', __file__]},
                strict_determinism=cli.strict_determinism, torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0), cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                note='Local steps count additional updates after source step1; validation restores BN/RNG. '
                     'Fixed newly seeded paired run, not historical RNG replay.')
    summary = dict(status='running', checks_passed=False, comparisons={}, validation={},
                   completed_steps={}, interventions={})
    diag.dump(out/'manifest.json', info)
    diag.dump(out/'summary.json', summary)
    try:
        dataset_type = diag.find_dataset_def(args.dataset)
        trainset = dataset_type(args.trainpath, args.trainlist, 'train', args.nviews, args.numdepth, args.interval_scale)
        valset = dataset_type(args.trainpath, plan['val_list'], 'val', args.nviews, args.numdepth, args.interval_scale)
        diag.validate_selection(trainset, plan['train_indices'], plan['train_metas'], STEPS*2)
        diag.validate_selection(valset, plan['val_indices'], plan['val_metas'], 6)
        for label in diag.RUNS:
            diag.seed(args.seed)
            model = diag.build_model(args).cuda()
            diag.configure_model(model, 'leaky_relu', args)
            model.load_state_dict(checkpoint['model'], strict=True)
            model.set_sger_residual_scale(case.sger_scale)
            optimizer, mapping = diag.restore_optimizer(model, args, checkpoint)
            diag.freeze_backbone_parameters(model)
            diag.set_diagnostic_train_mode(model, True)
            directory = out/label
            def record(phase, hashes, exact=False):
                diag.dump(directory/'fingerprints'/(phase+'.json'), hashes)
                if label == 'keep':
                    return
                reference = json.loads((out/'keep/fingerprints'/(phase+'.json')).read_text())
                comparison = diag.compare_fingerprints(reference, hashes)
                summary['comparisons'].setdefault(label, {})[phase] = comparison
                if comparison['first_protected_difference'] or ((exact or label == 'keep_repeat') and not comparison['exact']):
                    raise RuntimeError('Paired invariance failed: %s/%s' % (label, phase))
            record('initial', diag.fingerprints(diag.model_snapshot(model, optimizer)), True)
            intervention = diag.clear_first_moment(model, optimizer) if label == diag.RUNS[2] else None
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
                                freeze_backbone=True, intervention=intervention),
                           str(directory/('step_%04d.ckpt' % step)))
                diag.dump(out/'summary.json', summary)
                print(label, 'validation', step, 'proposal_gain', row['pooled']['proposal_gain_mean'],
                      'raw-refined', row['pooled']['raw_to_refined_error_delta'], flush=True)
            validate_and_save(0)
            loader = DataLoader(trainset, batch_sampler=[plan['train_indices'][i:i+2]
                                for i in range(0, STEPS*2, 2)], num_workers=0)
            for step, sample in enumerate(loader, 1):
                diag.set_diagnostic_train_mode(model, True)
                record('step%d_rng' % step, diag.fingerprints(rng_snapshot()), True)
                sample = diag._move_to_device(sample, torch.device('cuda'))
                row, hashes = diag.diagnostic_step(model, optimizer, diag.make_forward(model, sample, args, case))
                for phase, values in hashes.items():
                    record('step%d_%s' % (step, phase), values, step == 1 and phase in ('before', 'gradients'))
                check = diag.frozen_backbone_check(model, optimizer, initial)
                row['frozen_since_initial'] = check
                if not check['passed']:
                    raise RuntimeError('Backbone changed since initialization')
                row.update(local_step=step, indices=plan['train_indices'][2*(step-1):2*step])
                diag.dump(directory/'train'/('step_%04d.json' % step), row)
                summary['completed_steps'][label] = step
                diag.dump(out/'summary.json', summary)
                if step in SAVE_STEPS:
                    validate_and_save(step)
                print(label, 'step', step, '/', STEPS, flush=True)
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
    parser.add_argument('--output_dir', default=None, help='Must be a new directory')
    parser.add_argument('--strict_determinism', action='store_true')
    parser.add_argument('--preflight_only', action='store_true')
    cli = parser.parse_args()
    torch.set_num_threads(1)
    run(cli)


if __name__ == '__main__':
    main()
