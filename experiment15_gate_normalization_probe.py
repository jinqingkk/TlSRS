"""Gate-only paired probe: balanced versus pixel-mean stable benefit BCE."""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import experiment15_momentum_diagnose as diag
import experiment15_gate_selection_audit as audit
from experiment15_d_diagnose import rng_snapshot
from experiment15_d_bce_pressure import pool_metrics
from models.module import stable_gate_binary_cross_entropy, residual_benefit_target

RUNS = ('balanced_reference', 'balanced_repeat', 'pixel_mean')
GATE_PREFIX = 'sger_blocks.0.gate.'


def freeze_except_gate(model):
    names = []
    for name, parameter in model.named_parameters():
        enabled = name.startswith(GATE_PREFIX)
        parameter.requires_grad_(enabled)
        parameter.grad = None
        if enabled:
            names.append(name)
    if len(names) != 8 or len(model.sger_blocks) != 1:
        raise ValueError('Expected one SGER-Lite gate with eight parameter tensors')
    model.eval()  # All BN stays fixed; eval does not disable autograd.
    return names


def fixed_state(model, optimizer):
    values = diag.model_snapshot(model, optimizer)
    return diag.fingerprints({k: v for k, v in values.items()
                              if ('/' + GATE_PREFIX) not in k})


def gate_step(model, optimizer, forward):
    """One gate-only update with a read-only same-batch post measurement."""
    parameters = {n:p for n,p in model.named_parameters() if p.requires_grad}
    covered = {id(p) for group in optimizer.param_groups for p in group['params']}
    if any(id(p) not in covered for p in parameters.values()):
        raise RuntimeError('Gate parameter missing from optimizer')
    frozen = fixed_state(model, optimizer)
    before = {n:p.detach().cpu().double().clone() for n,p in parameters.items()}
    optimizer.zero_grad(set_to_none=True)
    state = diag.runtime_state(model)
    loss, pre, pre_hashes = forward()
    loss.backward()
    gradients = {n:p.grad.detach().cpu().double().clone() for n,p in parameters.items()
                 if p.grad is not None}
    if set(gradients) != set(parameters) or not any(bool((g != 0).any()) for g in gradients.values()):
        raise RuntimeError('Missing or entirely zero gate gradients')
    gradient_hashes = diag.fingerprints(diag.model_snapshot(model, gradients=True))
    optimizer.step()
    update_hashes = diag.fingerprints(diag.model_snapshot(model, optimizer))
    updates = {n:p.detach().cpu().double()-before[n] for n,p in parameters.items()}
    if not any(bool((u != 0).any()) for u in updates.values()):
        raise RuntimeError('No actual gate parameter update')
    directions = {n:dict(gradient_l2=float(gradients[n].norm()), update_l2=float(updates[n].norm()),
                         gradient_dot_update=float((gradients[n]*updates[n]).sum())) for n in parameters}
    with torch.no_grad():
        _, post, post_hashes = forward()
    diag.restore_runtime(model, state)
    if update_hashes != diag.fingerprints(diag.model_snapshot(model, optimizer)):
        raise RuntimeError('Post measurement mutated model/optimizer/BN')
    if fixed_state(model, optimizer) != frozen:
        raise RuntimeError('Frozen state changed')
    if diag.compare_fingerprints(pre_hashes, post_hashes)['first_protected_difference']:
        raise RuntimeError('Raw/proposal/geometry changed within update')
    return dict(before=pre, after=post, gate_directions=directions, measurement_state_restored=True), dict(
        before=pre_hashes, gradients=gradient_hashes, update=update_hashes, after=post_hashes)


def benefit_losses(stage, gt, mask, margin):
    valid = mask > .5
    n = int(valid.sum())
    if not n:
        raise ValueError('Empty training mask')
    target = residual_benefit_target(stage['depth_raw'], stage['raw_depth_residual'],
                                     gt, stage['depth_interval'], margin)
    positive, negative = valid & target, valid & ~target
    npos = int(positive.sum())
    inputs = (stage['gate_logits_a'], stage['gate_logits_b'], stage['region_a'],
              stage['region_b_weight'], stage['gate_valid_mask'], target)
    balanced = stable_gate_binary_cross_entropy(*(inputs + (valid,)))
    # Production stable BCE is unweighted on a single-class mask. Recombining
    # its class means gives pixel-mean BCE without a second numerical formula.
    pixel = (stable_gate_binary_cross_entropy(*(inputs + (positive,))) * (npos / n)
             + stable_gate_binary_cross_entropy(*(inputs + (negative,))) * ((n - npos) / n))
    return balanced, pixel, dict(valid_count=n, positive_count=npos, negative_count=n-npos,
        positive_fraction=npos/n, balanced_positive_mass=.5 if 0 < npos < n else npos/n,
        pixel_positive_mass=npos/n)


def make_forward(model, sample, args, case, normalization, selection_rows=None):
    def forward():
        outputs, _ = diag.proposal_forward(model, sample, args)
        stage = outputs['stage3']
        gt, mask = sample['depth']['stage3'], sample['mask']['stage3']
        losses = diag.cas_mvsnet_loss(outputs, sample['depth'], sample['mask'],
                                     **diag._loss_kwargs(args, sample, case))
        balanced, pixel, counts = benefit_losses(stage, gt, mask, args.benefit_margin_ratio)
        recorded = losses[-1]['stage3/gate_benefit_loss']
        if not torch.allclose(balanced.detach(), recorded, rtol=1e-6, atol=1e-7):
            raise RuntimeError('Recomputed balanced BCE differs from production loss')
        weight = case.loss_weights['gate_benefit_loss_weight'] * case.gate_benefit_scale
        chosen = pixel if normalization == 'pixel_mean' else balanced
        total = losses[0] + weight * (chosen - balanced)
        if not bool(torch.isfinite(total)):
            raise RuntimeError('Nonfinite total loss')
        stats = diag.stage_statistics(stage, gt, mask, args.benefit_margin_ratio)
        metrics = diag.aggregate([stats])
        metrics.update(loss=float(total.detach()), production_total_loss=float(losses[0].detach()),
            balanced_gate_bce=float(balanced.detach()), pixel_gate_bce=float(pixel.detach()),
            effective_gate_benefit_weight=weight, normalization=normalization, target_counts=counts)
        values = diag.forward_snapshot(outputs, total)
        diag.flatten(sample, 'input/sample', values)
        # Fixed proposal and geometry must match across groups AND within steps.
        for key in ('depth_raw', 'photometric_confidence', 'raw_depth_residual', 'normal',
                    'region_a', 'region_b_weight', 'gate_valid_mask'):
            diag.flatten(stage[key], 'backbone/fixed/' + key, values)
        if selection_rows is not None:
            for i in range(len(gt)):
                single = {k: v[i:i+1] if torch.is_tensor(v) and v.dim() > 0 else v for k,v in stage.items()}
                selection_rows.append(audit.selection_statistics(single, gt[i:i+1], mask[i:i+1],
                                                                 args.benefit_margin_ratio))
        return total, metrics, diag.fingerprints(values)
    return forward


def evaluate(model, optimizer, dataset, args, case, plan, record, step, device):
    before = diag.fingerprints(diag.model_snapshot(model, optimizer))
    state = diag.runtime_state(model)
    rows, stats = [], []
    try:
        with torch.no_grad():
            loader = DataLoader(dataset, batch_sampler=[plan['val_indices'][i:i+2]
                                for i in range(0, 6, 2)], num_workers=0)
            for batch, sample in enumerate(loader):
                sample = diag._move_to_device(sample, device)
                _, metrics, hashes = make_forward(model, sample, args, case, 'balanced', rows)()
                stats.append(metrics)
                record('val%d_batch%d' % (step, batch), hashes, step == 0)
    finally:
        diag.restore_runtime(model, state)
    if before != diag.fingerprints(diag.model_snapshot(model, optimizer)):
        raise RuntimeError('Validation changed model/optimizer')
    scenes = {}
    for row, meta in zip(rows, plan['val_metas']):
        scenes.setdefault(meta[0], []).append(row)
    return dict(pooled=audit.pool(rows), scenes={s:audit.pool(r) for s,r in scenes.items()},
                quality=pool_metrics(stats), batches=stats, state_restored=True,
                loss_definition='common production balanced BCE')


def run(cli):
    root = Path(cli.probe_dir)
    source = audit.load_source(root)
    plan = source['sample_plan']
    if len(plan['train_indices']) != 50 or len(plan['train_metas']) != 50:
        raise ValueError('Expected original fifty-sample plan')
    source_checkpoint = root/'stable/step_0025.ckpt'
    inputs = [root/'manifest.json', root/'summary.json', source_checkpoint,
              root/'stable/fingerprints/step25_update.json']
    hashes = {str(p): diag.sha(p) for p in inputs}
    if cli.preflight_only:
        model, optimizer, args = audit.load_model(root, 'stable', 25, source, torch.device('cpu'))
        names = freeze_except_gate(model)
        print('Preflight passed; gate tensors:', len(names), 'source:', source_checkpoint,
              'train samples replayed:', 2*cli.steps, flush=True)
        return
    if cli.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if cli.strict_determinism:
        if cli.device == 'cuda' and os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
            raise ValueError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8')
        torch.use_deterministic_algorithms(True)
    device = torch.device(cli.device)
    out = Path(cli.output_dir or ('outputs/experiment15_d/gate_normalization_%dsteps_' % cli.steps
                                  + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    out.mkdir(parents=True, exist_ok=False)
    save_steps = [s for s in (0,1,2,5,10,25) if s <= cli.steps]
    info = dict(status='running', command=sys.argv, input_sha256=hashes, source_probe=str(root),
        source_sha256={p:diag.sha(p) for p in diag.SOURCES + ['experiment15_momentum_diagnose.py',
            'experiment15_gate_selection_audit.py', 'experiment15_residual_sign_audit.py',
            'experiment15_d_bce_pressure.py', __file__]},
        arguments=dict(source['arguments'], gate_bce_mode='stable_logprob',
                       sger_residual_final_activation='leaky_relu'),
        effective_case=source['effective_case'], sample_plan=plan,
        steps=cli.steps, save_steps=save_steps, runs=RUNS, strict_determinism=cli.strict_determinism,
        device=cli.device, torch=torch.__version__, cuda=torch.version.cuda,
        intervention='Only benefit BCE normalization differs; same labels/margin/weight and full production other losses',
        boundary='Only eight gate tensors trainable; eval everywhere; proposal/features/normals/BN frozen',
        recovery='All Adam restored from stable local step25; no scheduler/reset; original fifty samples replayed')
    summary = dict(status='running', checks_passed=False, completed_steps={}, comparisons={}, validation={})
    diag.dump(out/'manifest.json', info)
    diag.dump(out/'summary.json', summary)
    try:
        for label in RUNS:
            diag.seed(source['arguments']['seed'])
            model, optimizer, args = audit.load_model(root, 'stable', 25, source, device)
            trainable = freeze_except_gate(model)
            case = diag.audit_cases(args)[0]
            if case._asdict() != source['effective_case']:
                raise ValueError('Effective schedule changed')
            initial_fixed = fixed_state(model, optimizer)
            directory = out/label
            normalization = 'pixel_mean' if label == 'pixel_mean' else 'balanced'
            def record(phase, values, exact=False):
                diag.dump(directory/'fingerprints'/(phase+'.json'), values)
                if label == RUNS[0]:
                    return
                reference = json.loads((out/RUNS[0]/'fingerprints'/(phase+'.json')).read_text())
                if label == 'pixel_mean' and phase == 'step1_before':
                    reference = {k:v for k,v in reference.items() if k != 'loss/total'}
                    values = {k:v for k,v in values.items() if k != 'loss/total'}
                comparison = diag.compare_fingerprints(reference, values)
                summary['comparisons'].setdefault(label, {})[phase] = comparison
                if comparison['first_protected_difference'] or ((exact or label == 'balanced_repeat') and not comparison['exact']):
                    raise RuntimeError('Paired invariance failed: '+label+'/'+phase)
            record('initial', diag.fingerprints(diag.model_snapshot(model, optimizer)), True)
            diag.dump(directory/'trainable_parameters.json', trainable)
            diag.dump(directory/'optimizer_groups.json', [{k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups])
            dataset = diag.find_dataset_def(args.dataset)
            trainset = dataset(args.trainpath, args.trainlist, 'train', args.nviews, args.numdepth, args.interval_scale)
            valset = dataset(args.trainpath, plan['val_list'], 'val', args.nviews, args.numdepth, args.interval_scale)
            diag.validate_selection(trainset, plan['train_indices'][:2*cli.steps], plan['train_metas'][:2*cli.steps], 2*cli.steps)
            diag.validate_selection(valset, plan['val_indices'], plan['val_metas'], 6)
            def validate_save(step):
                row = evaluate(model, optimizer, valset, args, case, plan, record, step, device)
                summary['validation'].setdefault(str(step), {})[label] = row
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    arguments=vars(args).copy(), epoch=12, local_step=step, source_pressure_step=25,
                    normalization=normalization, trainable_parameters=trainable, all_modules_eval=True),
                    str(directory/('step_%04d.ckpt'%step)))
                diag.dump(out/'summary.json', summary)
                means = row['pooled']['all/all/all/all']['means']
                print(label, 'validation', step, 'actual_gain', means['actual_gain'],
                      'oracle_gain', means['oracle_gain'], flush=True)
            validate_save(0)
            loader = DataLoader(trainset, batch_sampler=[plan['train_indices'][i:i+2]
                                for i in range(0,2*cli.steps,2)], num_workers=0)
            for step, sample in enumerate(loader, 1):
                record('step%d_rng'%step, diag.fingerprints(rng_snapshot()), True)
                sample = diag._move_to_device(sample, device)
                row, signatures = gate_step(model, optimizer,
                    make_forward(model, sample, args, case, normalization))
                if fixed_state(model, optimizer) != initial_fixed or any(
                        p.grad is not None for name,p in model.named_parameters() if name not in trainable):
                    raise RuntimeError('Frozen proposal/feature/normal/BN/optimizer changed')
                if any(module.training for module in model.modules()):
                    raise RuntimeError('Expected all modules in eval mode')
                for phase, values in signatures.items():
                    record('step%d_%s'%(step,phase), values, step==1 and phase=='before')
                row.update(local_step=step, indices=plan['train_indices'][2*(step-1):2*step],
                           fixed_proposal_state_unchanged=True)
                diag.dump(directory/'train'/('step_%04d.json'%step), row)
                summary['completed_steps'][label] = step
                diag.dump(out/'summary.json', summary)
                if step in save_steps:
                    validate_save(step)
            del model, optimizer
            if cli.device == 'cuda':
                torch.cuda.empty_cache()
        summary['checks_passed'] = True
        info['status'] = summary['status'] = 'completed'
    except BaseException as error:
        info['status'] = summary['status'] = 'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
        summary['error'] = repr(error)
        raise
    finally:
        summary['input_files_unchanged'] = all(diag.sha(p)==h for p,h in hashes.items())
        if not summary['input_files_unchanged']:
            summary['checks_passed'] = False
            info['status'] = summary['status'] = 'failed'
        diag.dump(out/'manifest.json', info)
        diag.dump(out/'summary.json', summary)
    if not summary['checks_passed']:
        raise RuntimeError('Integrity check failed')
    print('Probe completed:', out/'summary.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe_dir', required=True)
    parser.add_argument('--steps', type=int, choices=(2,25), default=2)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--output_dir')
    parser.add_argument('--strict_determinism', action='store_true')
    parser.add_argument('--preflight_only', action='store_true')
    cli = parser.parse_args()
    torch.set_num_threads(1)
    run(cli)


if __name__ == '__main__':
    main()
