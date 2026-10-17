"""Read-only signed-target audit of fixed train/validation subsets in eval mode."""
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
from models.module import bounded_residual_target

GROUPS = ('keep', 'zero_residual_exp_avg')
STEPS = (0, 10, 25)
TRAIN_BATCHES = (0, 12, 24)


def signed_statistics(proposal, target, valid):
    masks = {'positive': valid & (target > 0), 'negative': valid & (target < 0),
             'zero': valid & (target == 0)}
    values = {'target_mean': target, 'proposal_mean': proposal,
              'proposal_abs_mean': proposal.abs(), 'target_abs_error': (proposal-target).abs(),
              'proposal_positive_ratio': (proposal > 0).float(),
              'proposal_negative_ratio': (proposal < 0).float(),
              'proposal_zero_ratio': (proposal == 0).float(),
              'direction_accuracy': (proposal*target > 0).float()}
    # Raw/refined errors need the unbounded GT, supplied by the caller below.
    rows = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        rows[name] = {'count': count, 'valid_count': int(valid.sum()),
                      'sums': {key: float(value.detach()[mask].double().sum()) for key, value in values.items()}}
    return rows, masks


def summarize(rows):
    result = {}
    for sign in ('positive', 'negative', 'zero'):
        n = sum(r[sign]['count'] for r in rows)
        total = sum(r[sign]['valid_count'] for r in rows)
        keys = rows[0][sign]['sums']
        result[sign] = {'count': n, 'fraction': n/total if total else None,
                        'means': {k: sum(r[sign]['sums'][k] for r in rows)/n if n else None for k in keys}}
    return result


def gradient_report(proposal, target, masks, parameters, weight):
    """Use production smooth L1; separate group means from full-mask contributions."""
    total = sum(int(mask.sum()) for mask in masks.values())
    if total == 0:
        raise ValueError('Empty valid mask')
    names, params = list(parameters), list(parameters.values())
    reports, vectors = {}, {}
    for sign, mask in masks.items():
        count = int(mask.sum())
        if not count:
            reports[sign] = None
            continue
        loss = F.smooth_l1_loss(proposal[mask], target[mask], reduction='mean')
        gradients = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        pixel_gradient = torch.autograd.grad(loss, proposal, retain_graph=True)[0][mask]
        # An infinitesimal residual-head-only -grad step: does mean proposal move down/up?
        mean_gradients = torch.autograd.grad(proposal[mask].mean(), params,
                                             retain_graph=True, allow_unused=True)
        factor = weight * count / total
        rows, vector, derivative = {}, [], 0.0
        for name, parameter, grad, mean_grad in zip(names, params, gradients, mean_gradients):
            if grad is None:
                rows[name] = {'present': False}
                vector.append(torch.zeros(parameter.numel(), dtype=torch.float64))
                continue
            if not bool(torch.isfinite(grad).all()):
                raise RuntimeError('Nonfinite signed gradient: ' + name)
            cpu = grad.detach().cpu().double()
            rows[name] = {'present': True, 'l2': float(cpu.norm()), 'mean': float(cpu.mean()),
                          'zero': not bool((cpu != 0).any()), 'weighted_l2': float(cpu.norm())*factor}
            vector.append(cpu.reshape(-1)*factor)
            if mean_grad is not None:
                derivative -= float((cpu * mean_grad.detach().cpu().double()).sum()) * factor
        vectors[sign] = torch.cat(vector)
        reports[sign] = {'mean_smooth_l1': float(loss.detach()),
                         'actual_loss_contribution': float(loss.detach())*factor,
                         'group_to_full_gradient_factor': factor,
                         'd_loss_d_proposal_mean': float(pixel_gradient.detach().double().mean()),
                         'd_loss_d_proposal_positive_ratio': float((pixel_gradient > 0).double().mean()),
                         'mean_proposal_descent_derivative': derivative, 'parameters': rows}
    a, b = vectors.get('positive'), vectors.get('negative')
    cosine = None
    if a is not None and b is not None and float(a.norm()*b.norm()) > 0:
        cosine = float(torch.dot(a, b)/(a.norm()*b.norm()))
    return {'groups': reports, 'positive_negative_gradient_cosine': cosine,
            'weight': weight, 'scope': 'residual head only; group-normalized loss and weighted full-mask contribution; no update'}


def audit_batch(model, sample, args, weight):
    outputs, activations = diag.proposal_forward(model, sample, args)
    stage = outputs['stage3']
    raw, proposal, refined = stage['depth_raw'], stage['raw_depth_residual'], stage['depth_refined']
    gt, valid = sample['depth']['stage3'], sample['mask']['stage3'] > .5
    target = bounded_residual_target(raw, gt, stage['depth_interval'], args.residual_target_ratio)
    stats = diag.stage_statistics(stage, gt, sample['mask']['stage3'], args.benefit_margin_ratio)
    diag.proposal_statistics(stats, outputs, sample, args, model)
    rows, masks = signed_statistics(proposal, target, valid)
    errors = {'raw_abs_error': (raw-gt).abs(), 'refined_abs_error': (refined-gt).abs(),
              'proposal_gain': (raw-gt).abs()-(raw+proposal-gt).abs(),
              'raw_to_refined_delta': (raw-gt).abs()-(refined-gt).abs(),
              'gate_mean': stage['benefit_gate']}
    for name, mask in masks.items():
        rows[name]['sums'].update({k: float(v.detach()[mask].double().sum()) for k, v in errors.items()})
    gradients = gradient_report(proposal, target, masks, diag.residual_parameters(model), weight)
    values = diag.forward_snapshot(outputs, proposal.sum()*0)
    diag.flatten(sample, 'input/sample', values)
    hashes = diag.fingerprints(values)
    protected = {k: v for k, v in hashes.items() if k.startswith(('backbone/', 'input/'))}
    return {'groups': rows, 'gradient_report': gradients, 'activations': activations,
            'pooled': diag.aggregate([stats])}, protected


def load_source(root):
    manifest = json.loads((root/'manifest.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    if (manifest['status'] != 'completed' or summary['status'] != 'completed'
            or not summary['checks_passed'] or not summary['input_files_unchanged']
            or summary['completed_steps'] != {g: 25 for g in diag.RUNS}):
        raise ValueError('Requires completed checked 25-step momentum probe')
    for filename, digest in manifest['source_sha256'].items():
        if diag.sha(filename) != digest:
            raise ValueError('Source changed: ' + filename)
    return manifest


def load_model(root, group, step, source, device):
    path = root/group/('step_%04d.ckpt' % step)
    checkpoint = torch.load(str(path), map_location='cpu')
    args = argparse.Namespace(**checkpoint['arguments'])
    if (checkpoint['epoch'] != 12 or checkpoint['local_step'] != step
            or not checkpoint['freeze_backbone'] or checkpoint['source_probe_step'] != 1
            or checkpoint['arguments'] != source['arguments']):
        raise ValueError('Unexpected snapshot: ' + str(path))
    model = diag.build_model(args).to(device)
    diag.configure_model(model, 'leaky_relu', args)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.set_sger_residual_scale(source['effective_case']['sger_scale'])
    optimizer, _ = diag.restore_optimizer(model, args, checkpoint)
    diag.freeze_backbone_parameters(model)
    phase = 'initial' if step == 0 else 'step%d_update' % step
    reference = json.loads((root/group/'fingerprints'/(phase+'.json')).read_text())
    current = diag.fingerprints(diag.model_snapshot(model, optimizer))
    differing = {k for k in set(reference)|set(current) if reference.get(k) != current.get(k)}
    allowed = {'sger/optimizer/'+n+'/exp_avg' for n in diag.residual_parameters(model)} if step == 0 and group == GROUPS[1] else set()
    if differing - allowed:
        raise ValueError('Snapshot fingerprint mismatch: ' + str(path))
    if allowed and any(bool((optimizer.state[p]['exp_avg'] != 0).any()) for p in diag.residual_parameters(model).values()):
        raise ValueError('Step0 B moment is not zero')
    model.eval()
    return model, optimizer, args


def run(cli):
    root = Path(cli.probe_dir)
    source = load_source(root)
    plan = source['sample_plan']
    inputs = [root/'manifest.json', root/'summary.json']
    for group in GROUPS:
        for step in STEPS:
            inputs.extend([root/group/('step_%04d.ckpt' % step), root/group/'fingerprints'/
                           (('initial' if step == 0 else 'step%d_update' % step)+'.json')])
    input_hashes = {str(p): diag.sha(p) for p in inputs}
    if cli.preflight_only:
        for group in GROUPS:
            for step in STEPS:
                load_model(root, group, step, source, torch.device('cpu'))
        print('Preflight passed: six snapshots and source hashes verified')
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for DTU audit')
    if cli.strict_determinism:
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
            raise ValueError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8')
        torch.use_deterministic_algorithms(True)
    out = Path(cli.output_dir or ('outputs/experiment15_residual_recovery/sign_audit_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    out.mkdir(parents=True, exist_ok=False)
    selected = {'train': [i for batch in TRAIN_BATCHES for i in plan['train_indices'][2*batch:2*batch+2]],
                'val': plan['val_indices']}
    metas = {'train': [m for batch in TRAIN_BATCHES for m in plan['train_metas'][2*batch:2*batch+2]],
             'val': plan['val_metas']}
    info = dict(status='running', command=sys.argv, input_sha256=input_hashes,
                source_sha256={p: diag.sha(p) for p in diag.SOURCES+['experiment15_momentum_diagnose.py', __file__]},
                selected_indices=selected, selected_metas=metas, train_batch_positions=TRAIN_BATCHES,
                arguments=source['arguments'], effective_case=source['effective_case'],
                mode='eval for train and val; no updates', strict_determinism=cli.strict_determinism,
                torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                grouping='bounded target >0, <0, ==0 exactly; no epsilon; zero is not direction-correct')
    summary = dict(status='running', checks_passed=False, results={})
    diag.dump(out/'manifest.json', info)
    references = {}
    try:
        for group in GROUPS:
            for step in STEPS:
                diag.seed(source['arguments']['seed'])
                model, optimizer, args = load_model(root, group, step, source, torch.device('cuda'))
                initial = diag.fingerprints(diag.model_snapshot(model, optimizer))
                weight = source['effective_case']['loss_weights']['residual_target_loss_weight'] * source['effective_case']['residual_target_scale']
                result = {}
                for split in ('train', 'val'):
                    dataset = diag.find_dataset_def(args.dataset)(args.trainpath, args.trainlist if split == 'train' else plan['val_list'],
                                                                  split, args.nviews, args.numdepth, args.interval_scale)
                    diag.validate_selection(dataset, selected[split], metas[split], 6)
                    loader = DataLoader(dataset, batch_sampler=[selected[split][i:i+2] for i in range(0,6,2)], num_workers=0)
                    rows = []
                    for batch, sample in enumerate(loader):
                        sample = diag._move_to_device(sample, torch.device('cuda'))
                        row, hashes = audit_batch(model, sample, args, weight)
                        identity = split+'/'+str(batch)
                        if identity in references and references[identity] != hashes:
                            raise RuntimeError('Protected raw/input differs across snapshots: '+identity)
                        references[identity] = hashes
                        rows.append(row['groups'])
                        diag.dump(out/group/('step%d_%s_batch%d.json' % (step, split, batch)), row)
                        print(group, step, split, 'batch', batch, 'completed', flush=True)
                    result[split] = summarize(rows)
                if initial != diag.fingerprints(diag.model_snapshot(model, optimizer)) or any(p.grad is not None for p in model.parameters()):
                    raise RuntimeError('Read-only audit mutated model/optimizer or parameter grads')
                summary['results'].setdefault(group, {})[str(step)] = result
                diag.dump(out/'summary.json', summary)
                print(group, step, 'signed audit completed', flush=True)
                del model, optimizer
                torch.cuda.empty_cache()
        info['status'] = summary['status'] = 'completed'
        summary['checks_passed'] = True
    except BaseException as error:
        info['status'] = summary['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        summary['error'] = repr(error)
        raise
    finally:
        summary['input_files_unchanged'] = all(diag.sha(p) == h for p, h in input_hashes.items())
        if not summary['input_files_unchanged']:
            info['status'] = summary['status'] = 'failed'
            summary['checks_passed'] = False
        diag.dump(out/'manifest.json', info)
        diag.dump(out/'summary.json', summary)
    if not summary['checks_passed']:
        raise RuntimeError('Audit integrity check failed')
    print('Read-only sign audit completed:', out/'summary.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe_dir', required=True, help='Completed 25-step momentum probe directory')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--strict_determinism', action='store_true')
    parser.add_argument('--preflight_only', action='store_true')
    cli = parser.parse_args()
    torch.set_num_threads(1)
    run(cli)


if __name__ == '__main__':
    main()
