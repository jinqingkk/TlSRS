"""Residual degeneration audit and activation-only recovery probes (explicit runs)."""
import argparse
import copy
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import find_dataset_def
from models import cas_mvsnet_loss
from models.module import bounded_residual_target
from models.sger_refinement import ResidualDepthHead, SGERBlock
from gradient_audit import _base_loss_weights, _loss_kwargs, _move_to_device, audit_loss_inputs, _is_sger_lite_parameter
from gradient_audit_utils import build_audit_cases, run_audit_suite
from residual_audit_utils import (capture_residual_outputs, residual_diagnostics,
                                  residual_parameter_gradients)
from gate_audit_utils import capture_gate_outputs, gate_diagnostics
from experiment15_d import SOURCE, dump, sha
from experiment15_d_probe import build_model, restore_optimizer, seed, verify_prior_audits
from experiment15_d_diagnose import (freeze_backbone_parameters, set_diagnostic_train_mode,
    model_snapshot, compare_snapshots, flatten, forward_snapshot, rng_snapshot)
from experiment15_d_frozen_probe import (run as run_probe, stage_statistics, aggregate,
                                        fingerprints, compare_fingerprints, validate_selection)

RUNS = (('relu_reference', 'legacy_clamp'), ('relu_repeat', 'legacy_clamp'),
        ('leaky_relu', 'legacy_clamp'))
SOURCES = SOURCE + ['experiment15_d_probe.py', 'experiment15_d_diagnose.py',
                   'experiment15_d_frozen_probe.py', 'experiment15_residual_recovery.py']


def configure_model(model, label, args):
    activation = 'leaky_relu' if label == 'leaky_relu' else 'relu'
    heads = [m for m in model.modules() if isinstance(m, ResidualDepthHead)]
    if not heads:
        raise ValueError('No residual head found')
    for head in heads:
        head.set_final_activation(activation, 0.01)
    args.sger_residual_final_activation = activation
    args.sger_residual_negative_slope = 0.01
    args.gate_bce_mode = 'legacy_clamp'


def compact_stats(value):
    value = value.detach().float()
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError('Nonfinite residual activation')
    flat = value.reshape(value.shape[0], value.shape[1], -1)
    return {'mean': float(value.mean()), 'positive_ratio': float((value > 0).float().mean()),
            'zero_ratio': float((value == 0).float().mean()),
            'spatial_variance_mean': float(flat.var(dim=-1, unbiased=False).mean()),
            'max_abs': float(value.abs().max())}


@contextmanager
def capture_compact(model):
    """Reduce before in-place activation overwrites input; no graph retention."""
    values, handles = {}, []
    def pre(name):
        def hook(module, inputs):
            values[name] = compact_stats(inputs[0])
        return hook
    def post(name):
        def hook(module, inputs, output):
            values[name] = compact_stats(output)
            if name.endswith('/output_before_tanh'):
                bias = module.bias.detach().reshape(1, -1, 1, 1)
                values[name + '/weight_contribution'] = compact_stats(output.detach() - bias)
                values[name + '/bias'] = module.bias.detach().cpu().tolist()
        return hook
    try:
        for name, head in model.named_modules():
            if isinstance(head, ResidualDepthHead):
                handles.append(head.body[5].register_forward_pre_hook(pre(name + '/final_activation_input')))
                handles.append(head.body[5].register_forward_hook(post(name + '/final_activation_output')))
                handles.append(head.output.register_forward_hook(post(name + '/output_before_tanh')))
        yield values
    finally:
        for handle in handles:
            handle.remove()


def proposal_forward(model, sample, args):
    with capture_compact(model) as diagnostics:
        outputs = model(sample['imgs'], sample['proj_matrices'], sample['depth_values'])
    return outputs, diagnostics


def proposal_statistics(stats, outputs, sample, args, model):
    stage = outputs['stage3']
    proposal = stage['raw_depth_residual'].detach()
    valid = sample['mask']['stage3'] > 0.5
    gt = sample['depth']['stage3']
    blocks = [m for m in model.modules() if isinstance(m, SGERBlock)]
    if len(blocks) != 1:
        raise ValueError('This experiment requires a single SGER-lite block')
    block = blocks[0]
    bound = args.sger_max_residual_ratio * stage['depth_interval']
    bias_only = bound * torch.tanh(block.residual.output.bias.detach()).reshape(1, 1, 1)
    target = bounded_residual_target(stage['depth_raw'], gt, stage['depth_interval'], args.residual_target_ratio)
    gate = stage['benefit_gate'].detach()
    metrics = {'proposal_target_abs_error': (proposal - target).abs(),
               'proposal_minus_bias_abs_mean': (proposal - bias_only).abs(),
               'gate_below_1e_minus6_ratio': (gate < 1e-6).float(),
               'gate_above_1_minus_1e_minus6_ratio': (gate > 1 - 1e-6).float(),
               'proposal_near_bound_ratio': (proposal.abs() >= 0.99 * bound).float()}
    for key, value in metrics.items():
        stats['sums'][key] = float(value[valid].double().sum())
    selected = proposal[valid].double()
    stats['proposal_variance_valid_batch'] = float(selected.var(unbiased=False)) if selected.numel() else None
    stats['proposal_spatial_variance_per_sample'] = [
        float(p[m].double().var(unbiased=False)) if bool(m.any()) else None
        for p, m in zip(proposal, valid)]
    for key in ('depth_raw', 'depth_refined', 'raw_depth_residual', 'depth_residual', 'benefit_gate'):
        if stage[key].shape != gt.shape or not bool(torch.isfinite(stage[key]).all()):
            raise RuntimeError('Invalid stage3 tensor: ' + key)
    if bool((proposal.abs() > bound + 1e-6).any()):
        raise RuntimeError('Proposal bound exceeded')
    scale = model.sger_residual_scale
    if bool((stage['depth_residual'].detach().abs() > bound * scale + 1e-6).any()):
        raise RuntimeError('Effective residual bound exceeded')


def load_experiment(cli):
    """Use audited epoch-specific args; borrow only the fixed sample selection."""
    verify_prior_audits()
    report_path = Path('outputs/experiment15_d/audit_relu/gate_bce_epoch%d_batch0000_relu_legacy_clamp.json' % cli.epoch)
    report = json.loads(report_path.read_text())
    args = dict(report['metadata']['arguments'])
    args.update(activation_ab=False, gate_bce_ab=False, gate_bce_mode='legacy_clamp',
                sger_residual_final_activation='relu', sger_residual_negative_slope=0.01,
                loadckpt=cli.checkpoint or args['loadckpt'], audit_epoch=cli.epoch)
    if args['batch_size'] != 2 or not args['use_sger_lite'] or args['sger_share']:
        raise ValueError('Requires batch_size=2 and one unshared SGER-lite block')
    checkpoint_hash = sha(args['loadckpt'])
    runs = json.loads(Path('outputs/experiment15_d/run_manifest.json').read_text())['runs']
    if not any(r['status'] == 'verified' and r['arguments']['audit_epoch'] == cli.epoch
               and r['checkpoint_sha256'] == checkpoint_hash for r in runs):
        raise ValueError('Checkpoint hash is not the audited epoch checkpoint')
    source = copy.deepcopy(json.loads(Path(cli.probe_manifest).read_text()))
    for key in ('dataset', 'trainpath', 'trainlist', 'nviews', 'numdepth', 'interval_scale', 'seed'):
        if source['arguments'][key] != args[key]:
            raise ValueError('Sample manifest configuration differs: ' + key)
    if len(source['train_indices']) != 400 or len(source['train_metas']) != 400:
        raise ValueError('Expected original 400-sample manifest')
    source.update(arguments=args, checkpoint=args['loadckpt'], checkpoint_sha256=checkpoint_hash)
    source['train_indices'] = source['train_indices'][:cli.steps * 2]
    source['train_metas'] = source['train_metas'][:cli.steps * 2]
    return {'source': source, 'steps': cli.steps, 'epoch': cli.epoch, 'runs': RUNS,
            'configure_model': configure_model, 'forward': proposal_forward,
            'statistics': proposal_statistics, 'gradients': residual_parameter_gradients,
            'source_sha256': {p: sha(p) for p in SOURCES},
            'metadata': {'created_protocol_date': '2026-09-20', 'purpose': 'residual final activation only; fixed legacy BCE',
                         'audit_report': str(report_path), 'audit_report_sha256': sha(report_path),
                         'negative_slope': 0.01, 'original_sample_manifest': cli.probe_manifest,
                         'checkpoint_epoch': cli.epoch, 'schedule_epoch': cli.epoch}}


def audit_cases(args):
    original = build_audit_cases(args.audit_epoch, _base_loss_weights(args),
                                args.sger_warmup_start_epoch, args.sger_warmup_end_epoch)
    full = original[0]
    result = [full]
    for name, weight in (('refined_only', 'refined_depth_loss_weight'),
                         ('residual_target_only', 'residual_target_loss_weight')):
        forced = next(c for c in original if c.name == name)
        result.append(forced)
        weights = {key: 0.0 for key in full.loss_weights}
        weights[weight] = full.loss_weights[weight]
        result.append(forced._replace(name=name + '_actual', loss_weights=weights,
                      sger_scale=full.sger_scale, residual_target_scale=full.residual_target_scale,
                      gate_benefit_scale=full.gate_benefit_scale))
    return result


@contextmanager
def capture_case_gradients(model):
    """Keep exact per-case gradient evidence including parameters with no grad."""
    state = {'active': None, 'cases': {}}
    handles = []
    names = {name: ('sger/' if _is_sger_lite_parameter(model, name) else 'backbone/')
             + 'gradients/' + name for name, _ in model.named_parameters()}
    def hook_for(name):
        def hook(gradient):
            state['cases'][state['active']].update(fingerprints({names[name]: gradient}))
        return hook
    state['empty'] = fingerprints({key: None for key in names.values()})
    try:
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                handles.append(parameter.register_hook(hook_for(name)))
        yield state
    finally:
        for handle in handles:
            handle.remove()


def run_audit(cli, experiment):
    out = Path(cli.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    source = experiment['source']
    args = argparse.Namespace(**source['arguments'])
    info = {'status': 'running', 'command': sys.argv, 'arguments': vars(args).copy(),
            'checkpoint_sha256': source['checkpoint_sha256'], 'source_sha256': experiment['source_sha256'],
            'experiment': experiment['metadata'], 'strict_determinism': cli.strict_determinism,
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'freeze_backbone': True, 'optimizer_steps': 0,
            'probe_manifest': cli.probe_manifest, 'probe_manifest_sha256': sha(cli.probe_manifest)}
    summary = {'status': 'running', 'paired_checks_passed': False, 'comparisons': {}, 'findings': {},
               'note': 'Missing/zero residual gradients are diagnostic findings, not a quality PASS.'}
    dump(out/'manifest.json', info)
    try:
        dataset = find_dataset_def(args.dataset)(args.trainpath, args.trainlist, 'train', args.nviews,
                                                args.numdepth, args.interval_scale)
        validate_selection(dataset, source['train_indices'], source['train_metas'], cli.steps * 2)
        selections = [('historical%d' % i, [2*i, 2*i+1]) for i in range(5)]
        selections += [('probe%d' % i, source['train_indices'][2*i:2*i+2]) for i in range(2)]
        info['selections'] = [{'name': n, 'indices': ix, 'metas': [dataset.metas[i] for i in ix]} for n, ix in selections]
        info['cases'] = [c._asdict() for c in audit_cases(args)]
        checkpoint = torch.load(args.loadckpt, map_location='cpu')
        if checkpoint.get('epoch') != cli.epoch:
            raise ValueError('Checkpoint epoch mismatch')
        references = {}
        gradient_references = {}
        initial_reference = None
        for label, _ in RUNS:
            seed(args.seed)
            model = build_model(args).cuda()
            configure_model(model, label, args)
            model.load_state_dict(checkpoint['model'], strict=True)
            optimizer, mapping = restore_optimizer(model, args, checkpoint)
            freeze_backbone_parameters(model)
            set_diagnostic_train_mode(model, True)
            initial = model_snapshot(model, optimizer)
            if initial_reference is None:
                initial_reference = fingerprints(initial)
            elif not compare_fingerprints(initial_reference, fingerprints(initial))['exact']:
                raise RuntimeError('Audit initial weights/optimizer differ')
            dump(out/label/'optimizer_mapping.json', mapping)
            dump(out/label/'module_training.json', {n: m.training for n, m in model.named_modules()})
            loader = DataLoader(dataset, batch_sampler=[ix for _, ix in selections], num_workers=0)
            for (selection, indices), cpu_sample in zip(selections, loader):
                sample = _move_to_device(cpu_sample, torch.device('cuda'))
                def evaluate(case):
                    set_diagnostic_train_mode(model, True)
                    gradient_capture['active'] = case.name
                    gradient_capture['cases'][case.name] = dict(gradient_capture['empty'])
                    with capture_gate_outputs(model) as gates, capture_residual_outputs(model) as residuals:
                        outputs = model(sample['imgs'], sample['proj_matrices'], sample['depth_values'])
                    key = 'stage3'
                    stage = outputs[key]
                    isolated_case = case._replace(name=case.name.replace('_actual', ''))
                    losses = cas_mvsnet_loss(audit_loss_inputs(outputs, isolated_case), sample['depth'],
                                            sample['mask'], **_loss_kwargs(args, sample, case))
                    stats = stage_statistics(stage, sample['depth'][key], sample['mask'][key], args.benefit_margin_ratio)
                    proposal_statistics(stats, outputs, sample, args, model)
                    extras = dict(losses[-1])
                    extras.update(benefit_margin_ratio=args.benefit_margin_ratio,
                                  proposal_statistics=stats, proposal_metrics=aggregate([stats]),
                                  residual_diagnostics={key: residual_diagnostics(stage, residuals.get(id(stage['raw_depth_residual']), {}), sample['mask'][key])},
                                  gate_diagnostics={key: gate_diagnostics(stage, gates.get(id(stage['benefit_gate']), {}), sample['depth'][key], sample['mask'][key], args.benefit_margin_ratio)})
                    values = forward_snapshot(outputs, losses[0])
                    flatten(cpu_sample, 'input/sample', values)
                    values.update(rng_snapshot())
                    hashes = fingerprints(values)
                    identity = selection + '/' + case.name
                    if label == 'relu_reference':
                        references[identity] = hashes
                    else:
                        comparison = compare_fingerprints(references[identity], hashes)
                        summary['comparisons'][label + '/' + identity] = comparison
                        if comparison['first_protected_difference'] or (label == 'relu_repeat' and not comparison['exact']):
                            raise RuntimeError('Audit paired invariance failed: ' + identity)
                    return outputs, losses[0], extras, sample['depth'][key], sample['mask'][key] > 0.5
                with capture_case_gradients(model) as gradient_capture:
                    report = run_audit_suite(model, optimizer, audit_cases(args), evaluate,
                                output_json=str(out/label/(selection + '.json')),
                                metadata={'arguments': vars(args).copy(), 'selection': selection,
                                          'indices': indices, 'freeze_backbone': True, 'optimizer_steps': 0})
                for case_name, hashes in gradient_capture['cases'].items():
                    identity = selection + '/' + case_name
                    dump(out/label/(selection + '_' + case_name + '_gradients.json'), hashes)
                    if label == 'relu_reference':
                        gradient_references[identity] = hashes
                    else:
                        comparison = compare_fingerprints(gradient_references[identity], hashes)
                        summary['comparisons'][label + '/' + identity + '/backward'] = comparison
                        if comparison['first_protected_difference'] or (label == 'relu_repeat' and not comparison['exact']):
                            raise RuntimeError('Audit paired gradients differ: ' + identity)
                for row in report['cases']:
                    if not row['gradient_summary']['all_finite']:
                        raise RuntimeError('Nonfinite audit parameter gradient')
                    if any(g['missing_optimizer_tensors'] for g in row['parameter_gradients'].values()):
                        raise RuntimeError('Audit optimizer coverage missing')
                summary['findings'][label + '/' + selection] = report['fatal_errors']
                comparison = compare_snapshots(initial, model_snapshot(model, optimizer))
                if not comparison['exact']:
                    raise RuntimeError('Audit modified model or optimizer state')
                dump(out/'summary.json', summary)
            del model, optimizer, initial
            torch.cuda.empty_cache()
        info['status'] = summary['status'] = 'completed'
        summary['paired_checks_passed'] = True
    except BaseException as error:
        info['status'] = summary['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        info['error'] = summary['error'] = repr(error)
        raise
    finally:
        dump(out/'manifest.json', info)
        dump(out/'summary.json', summary)
    print('Read-only audit completed; inspect findings:', out/'summary.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('audit', 'probe'), default='audit')
    parser.add_argument('--epoch', type=int, choices=(9, 12), default=12)
    parser.add_argument('--checkpoint', default=None, help='Optional relocated copy; must match audited checkpoint hash')
    parser.add_argument('--steps', type=int, choices=(2, 200), default=2)
    parser.add_argument('--probe_manifest', default='outputs/experiment15_d/short_train/sample_manifest.json')
    parser.add_argument('--strict_determinism', action='store_true')
    parser.add_argument('--output_dir', default=None, help='Must not exist')
    cli = parser.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for real-data experiments')
    if cli.strict_determinism:
        torch.use_deterministic_algorithms(True)
    experiment = load_experiment(cli)
    cli.output_dir = cli.output_dir or ('outputs/experiment15_residual_recovery/%s_epoch%d_%s' %
        (cli.mode + ('_%dsteps' % cli.steps if cli.mode == 'probe' else ''), cli.epoch,
         datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    if cli.mode == 'audit':
        run_audit(cli, experiment)
    else:
        run_probe(cli, experiment)


if __name__ == '__main__':
    main()
