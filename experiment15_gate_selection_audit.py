"""Read-only gate selection audit; GT oracle is an offline binary-choice reference."""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import experiment15_momentum_diagnose as diag
from experiment15_residual_sign_audit import TRAIN_BATCHES

GROUPS = ('legacy_reference', 'stable')
STEPS = (0, 10, 25)


@torch.no_grad()
def selection_statistics(stage, gt, mask, margin_ratio):
    raw, proposal, refined, gate = [stage[k].detach() for k in
        ('depth_raw', 'raw_depth_residual', 'depth_refined', 'benefit_gate')]
    valid = mask > .5
    if not bool(valid.any()):
        raise ValueError('Empty valid mask')
    for value in (raw, proposal, refined, gate, gt):
        if value.shape != gt.shape or not bool(torch.isfinite(value[valid]).all()):
            raise ValueError('Invalid shape or nonfinite valid depth/gate')
    if bool(((gate[valid] < 0) | (gate[valid] > 1)).any()):
        raise ValueError('Gate outside [0, 1]')
    # This probe fixes beta=1. Preserve production precision for depth additions.
    if not torch.equal((raw + gate * proposal)[valid], refined[valid]):
        raise ValueError('Expected beta=1 gated residual depth')
    raw_error = (raw - gt).abs()
    proposal_error = (raw + proposal - gt).abs()
    actual_error = (refined - gt).abs()
    gain = raw_error - proposal_error
    margin = float(stage['depth_interval']) * margin_ratio
    # Oracle chooses either the whole bounded proposal or no change. It is NOT
    # an upper bound over continuously rescaled residuals, nor a deployable gate.
    beneficial = gain > 0
    oracle_error = torch.where(beneficial, proposal_error, raw_error)
    above_margin = gain > margin
    target = gt - raw
    region = stage['region_a'].bool()
    geometry = torch.where(region, torch.ones_like(gate), stage['region_b_weight'])
    logits = torch.where(region, stage['gate_logits_a'], stage['gate_logits_b'])
    values = dict(raw_error=raw_error, actual_error=actual_error,
                  proposal_error=proposal_error, oracle_error=oracle_error,
                  actual_gain=raw_error-actual_error, proposal_gain=gain,
                  oracle_gain=raw_error-oracle_error, gate=gate,
                  effective_residual_abs=(gate*proposal).abs(), proposal_abs=proposal.abs(),
                  negative_proposal=(proposal < 0).float(), geometry_weight=geometry,
                  active_logit=logits,
                  actual_improved=(actual_error < raw_error).float(),
                  actual_worsened=(actual_error > raw_error).float())
    for value in values.values():
        if not bool(torch.isfinite(value[valid]).all()):
            raise ValueError('Nonfinite diagnostic')
    # Reduce each pixel once into disjoint cells, then derive overlapping views.
    # This avoids hundreds of full-resolution GPU mask reductions per image.
    sign_id = torch.where(target > 0, torch.zeros_like(target, dtype=torch.long),
                         torch.where(target < 0, torch.ones_like(target, dtype=torch.long),
                                     torch.full_like(target, 2, dtype=torch.long)))
    benefit_id = torch.zeros_like(sign_id)
    benefit_id[gain == 0] = 1
    benefit_id[beneficial & ~above_margin] = 2
    benefit_id[above_margin] = 3
    gate_id = (gate >= 1e-6).long() + (gate >= .01).long() + (gate >= .5).long()
    cell = (((sign_id * 4 + benefit_id) * 2 + (~region).long()) * 4 + gate_id)[valid].cpu()
    counts = torch.bincount(cell, minlength=96).cpu().numpy().reshape(3, 4, 2, 4)
    sums = {k: torch.bincount(cell, weights=v[valid].double().cpu(), minlength=96)
            .cpu().numpy().reshape(3, 4, 2, 4) for k, v in values.items()}
    whole = slice(None)
    signs = dict(all=whole, positive=slice(0, 1), negative=slice(1, 2), zero=slice(2, 3))
    benefits = dict(all=whole, beneficial=slice(2, 4), harmful=slice(0, 1),
                    neutral=slice(1, 2), above_margin=slice(3, 4), small_benefit=slice(2, 3))
    branches = dict(all=whole, a=slice(0, 1), b=slice(1, 2))
    bins = dict(all=whole, below_1e_6=slice(0, 1), from_1e_6_to_001=slice(1, 2),
                from_001_to_05=slice(2, 3), at_least_05=slice(3, 4))
    rows = {}
    for sn, sm in signs.items():
        for bn, bm in benefits.items():
            for rn, rm in branches.items():
                for gn, gm in bins.items():
                    index = (sm, bm, rm, gm)
                    rows['/'.join((sn, bn, rn, gn))] = dict(count=int(counts[index].sum()),
                        sums={k: float(v[index].sum()) for k, v in sums.items()})
    return rows


def pool(rows):
    result = {}
    total = sum(r['all/all/all/all']['count'] for r in rows)
    for name in rows[0]:
        n = sum(r[name]['count'] for r in rows)
        sums = {k: sum(r[name]['sums'][k] for r in rows) for k in rows[0][name]['sums']}
        result[name] = dict(count=n, fraction=n/total if total else None, sums=sums,
                            means={k: v/n if n else None for k, v in sums.items()})
    return result


def load_source(root):
    source = json.loads((root/'manifest.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    if (source['status'] != 'completed' or summary['status'] != 'completed'
            or not summary['checks_passed'] or not summary['input_files_unchanged']
            or summary['completed_steps'] != {g: 25 for g in ('legacy_reference', 'legacy_repeat', 'stable')}
            or source['effective_case']['sger_scale'] != 1):
        raise ValueError('Requires completed checked beta=1 BCE pressure 25-step probe')
    for key in ('source_sha256', 'input_sha256'):
        for filename, digest in source[key].items():
            if diag.sha(filename) != digest:
                raise ValueError('Source/input hash mismatch: ' + filename)
    return source


def load_model(root, group, step, source, device):
    path = root/group/('step_%04d.ckpt' % step)
    checkpoint = torch.load(str(path), map_location='cpu')
    mode = 'stable_logprob' if group == 'stable' else 'legacy_clamp'
    expected = dict(source['arguments'], gate_bce_mode=mode,
                    sger_residual_final_activation='leaky_relu')
    if (checkpoint['epoch'] != 12 or checkpoint['local_step'] != step
            or checkpoint['source_probe_step'] != 1 or not checkpoint['freeze_backbone']
            or checkpoint['arguments'] != expected or checkpoint['gate_bce_mode'] != mode):
        raise ValueError('Unexpected snapshot configuration: ' + str(path))
    args = argparse.Namespace(**expected)
    model = diag.build_model(args).to(device)
    diag.configure_model(model, 'leaky_relu', args)
    args.gate_bce_mode = mode
    model.load_state_dict(checkpoint['model'], strict=True)
    model.set_sger_residual_scale(1.)
    optimizer, _ = diag.restore_optimizer(model, args, checkpoint)
    diag.freeze_backbone_parameters(model)
    phase = 'initial' if step == 0 else 'step%d_update' % step
    reference = json.loads((root/group/'fingerprints'/(phase+'.json')).read_text())
    if diag.fingerprints(diag.model_snapshot(model, optimizer)) != reference:
        raise ValueError('Snapshot fingerprint mismatch: ' + str(path))
    model.eval()
    return model, optimizer, args


def run(cli):
    root = Path(cli.probe_dir)
    source = load_source(root)
    inputs = [root/'manifest.json', root/'summary.json']
    for g in GROUPS:
        for s in STEPS:
            inputs += [root/g/('step_%04d.ckpt' % s), root/g/'fingerprints'/
                       (('initial' if s == 0 else 'step%d_update' % s)+'.json')]
    hashes = {str(p): diag.sha(p) for p in inputs}
    if cli.preflight_only:
        for g in GROUPS:
            for s in STEPS:
                load_model(root, g, s, source, torch.device('cpu'))
        print('Preflight passed: six snapshots and provenance verified')
        return
    if cli.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu only if intended')
    if cli.strict_determinism:
        if cli.device == 'cuda' and os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
            raise ValueError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8')
        torch.use_deterministic_algorithms(True)
    device = torch.device(cli.device)
    plan = source['sample_plan']
    selected = {'train': [i for b in TRAIN_BATCHES for i in plan['train_indices'][2*b:2*b+2]],
                'val': plan['val_indices']}
    metas = {'train': [i for b in TRAIN_BATCHES for i in plan['train_metas'][2*b:2*b+2]],
             'val': plan['val_metas']}
    out = Path(cli.output_dir or ('outputs/experiment15_d/gate_selection_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    out.mkdir(parents=True, exist_ok=False)
    info = dict(status='running', command=sys.argv, input_sha256=hashes,
                source_sha256={p: diag.sha(p) for p in diag.SOURCES +
                    ['experiment15_momentum_diagnose.py', 'experiment15_residual_sign_audit.py', __file__]},
                source_probe=str(root), selected_indices=selected, selected_metas=metas,
                arguments=source['arguments'], effective_case=source['effective_case'],
                device=cli.device, strict_determinism=cli.strict_determinism,
                torch=torch.__version__, cuda=torch.version.cuda,
                grouping='target sign / proposal benefit / branch / gate bin; overlapping all and margin rows',
                oracle='GT offline choice: full bounded proposal if gain>0 else raw; not continuous-gate upper bound')
    summary = dict(status='running', checks_passed=False, results={})
    diag.dump(out/'manifest.json', info)
    diag.dump(out/'summary.json', summary)
    references = {}
    try:
        for group in GROUPS:
            for step in STEPS:
                diag.seed(source['arguments']['seed'])
                model, optimizer, args = load_model(root, group, step, source, device)
                before = diag.fingerprints(diag.model_snapshot(model, optimizer))
                result = {}
                for split in ('train', 'val'):
                    dataset = diag.find_dataset_def(args.dataset)(args.trainpath,
                        args.trainlist if split == 'train' else plan['val_list'], split,
                        args.nviews, args.numdepth, args.interval_scale)
                    diag.validate_selection(dataset, selected[split], metas[split], 6)
                    loader = DataLoader(dataset, batch_sampler=[selected[split][i:i+2] for i in range(0,6,2)], num_workers=0)
                    rows, scenes = [], {}
                    with torch.no_grad():
                        for batch, sample in enumerate(loader):
                            sample = diag._move_to_device(sample, device)
                            outputs, _ = diag.proposal_forward(model, sample, args)
                            stage = outputs['stage3']
                            values = {'raw': stage['depth_raw'], 'confidence': stage['photometric_confidence']}
                            diag.flatten(sample, 'input', values)
                            signature = diag.fingerprints(values)
                            key = (split, batch)
                            if key in references and references[key] != signature:
                                raise RuntimeError('Protected raw/input changed across snapshots')
                            references[key] = signature
                            for i in range(len(sample['imgs'])):
                                single = {k: v[i:i+1] if torch.is_tensor(v) and v.dim() > 0 else v for k,v in stage.items()}
                                row = selection_statistics(single, sample['depth']['stage3'][i:i+1],
                                    sample['mask']['stage3'][i:i+1], args.benefit_margin_ratio)
                                scene = metas[split][2*batch+i][0]
                                rows.append(row)
                                scenes.setdefault(scene, []).append(row)
                            print(group, step, split, batch, 'completed', flush=True)
                    result[split] = dict(pooled=pool(rows), scenes={s:pool(r) for s,r in scenes.items()})
                if before != diag.fingerprints(diag.model_snapshot(model, optimizer)) or any(p.grad is not None for p in model.parameters()):
                    raise RuntimeError('Read-only audit changed model/optimizer or parameter grads')
                summary['results'].setdefault(group, {})[str(step)] = result
                diag.dump(out/'summary.json', summary)
                del model, optimizer
                if cli.device == 'cuda':
                    torch.cuda.empty_cache()
        summary['checks_passed'] = True
        info['status'] = summary['status'] = 'completed'
    except BaseException as error:
        info['status'] = summary['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        summary['error'] = repr(error)
        raise
    finally:
        summary['input_files_unchanged'] = all(diag.sha(p) == h for p,h in hashes.items())
        if not summary['input_files_unchanged']:
            summary['checks_passed'] = False
            info['status'] = summary['status'] = 'failed'
        diag.dump(out/'manifest.json', info)
        diag.dump(out/'summary.json', summary)
    if not summary['checks_passed']:
        raise RuntimeError('Audit integrity check failed')
    print('Audit completed:', out/'summary.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe_dir', required=True)
    parser.add_argument('--output_dir')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--strict_determinism', action='store_true')
    parser.add_argument('--preflight_only', action='store_true')
    cli = parser.parse_args()
    torch.set_num_threads(1)
    run(cli)


if __name__ == '__main__':
    main()
