"""Read-only Epoch12/15 harm and fixed-amplitude audit, with explicit GT scope."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets import find_dataset_def
from evaluation_record import sha256, source_record, write_json
from experiment15_d_probe import build_model, seed
from gradient_audit import build_parser, _move_to_device
from models.module import sobel_gradient_magnitude

ALPHAS = (0.0, 0.25, 0.5, 1.0)


def statistics(raw, refined, proposal, gate, gt, valid, confidence, edge):
    """Pixel sums; raw bins use [0,.25], (.25,.5], (.5,1], (1,inf)."""
    valid = valid.astype(bool) & np.isfinite(gt) & (gt > 0)
    for value in (raw, refined, proposal, gate, confidence):
        if value.shape != gt.shape or not np.isfinite(value[valid]).all():
            raise ValueError('Invalid diagnostic input')
    if not valid.any():
        raise ValueError('No GT-valid pixels')
    target = gt - raw
    error = np.abs(target)
    delta = refined - raw
    proposal_gain = error - np.abs(raw + proposal - gt)
    gain = error - np.abs(refined - gt)
    values = {'raw_abs_error': error, 'refined_abs_error': np.abs(refined-gt),
              'net_gain': gain, 'harm': np.maximum(-gain, 0),
              'benefit': np.maximum(gain, 0), 'gate': gate,
              'proposal_abs': np.abs(proposal), 'effective_abs': np.abs(delta),
              'proposal_gain': proposal_gain,
              'oracle_proposal_gain': np.maximum(proposal_gain, 0),
              'harmful_proposal_released_harm': np.where(proposal_gain < 0, np.maximum(-gain, 0), 0),
              'beneficial_proposal_unused_gain': np.where(proposal_gain > 0, np.maximum(proposal_gain-gain, 0), 0),
              'direction_correct': (proposal * target > 0).astype(float),
              'improved': (gain > 1e-5).astype(float),
              'worsened': (gain < -1e-5).astype(float),
              'correct_direction_overshoot': ((delta * target > 0) & (gain < -1e-5)).astype(float)}
    for alpha in ALPHAS:
        values['alpha_%g_gain' % alpha] = error - np.abs(raw + alpha * delta - gt)
    masks = {'all': valid, 'high_confidence': valid & (confidence >= .9),
             'positive_target': valid & (target > 1e-5),
             'negative_target': valid & (target < -1e-5),
             'beneficial_proposal': valid & (proposal_gain > 1e-5),
             'harmful_proposal': valid & (proposal_gain < -1e-5),
             'actual_improved': valid & (gain > 1e-5),
             'actual_worsened': valid & (gain < -1e-5)}
    for lower, upper in ((0, .25), (.25, .5), (.5, 1), (1, np.inf)):
        bin_mask = valid & (error <= upper) & ((error > lower) if lower else (error >= 0))
        for high in (False, True):
            for is_edge in (False, True):
                name = 'error_%g_%g/conf_%s/edge_%s' % (lower, upper, high, is_edge)
                masks[name] = bin_mask & ((confidence >= .9) == high) & ((edge >= .05) == is_edge)
    for lower, upper in ((0, .25), (.25, .5), (.5, .75), (.75, 1.01)):
        masks['gate_%g_%g' % (lower, upper)] = valid & (gate >= lower) & (gate < upper)
    return {name: {'count': int(mask.sum()),
                   'sums': {key: float(value[mask].sum(dtype=np.float64))
                            for key, value in values.items()}}
            for name, mask in masks.items()}


def pool(rows):
    result = {}
    for key in rows[0]:
        count = sum(row[key]['count'] for row in rows)
        sums = {name: sum(row[key]['sums'][name] for row in rows)
                for name in rows[0][key]['sums']}
        result[key] = {'count': count, 'sums': sums,
                       'means': {name: value / count if count else None for name, value in sums.items()}}
    positive = result['positive_target']['means']['direction_correct']
    negative = result['negative_target']['means']['direction_correct']
    return {'regions': result, 'balanced_direction_accuracy':
            (positive + negative) / 2 if positive is not None and negative is not None else None}


@torch.no_grad()
def cue_batch_drift(cues, samples):
    """Fixed first image, paired with another real scene; eval mode, no BN changes."""
    first, second = samples
    single = cues(*first)['cue_tensor']
    paired = cues(*[torch.cat((a, b), 0) for a, b in zip(first, second)])['cue_tensor'][:1]
    diff = (paired - single).abs()
    return {'mean_abs_by_channel': diff.mean(dim=(0, 2, 3)).cpu().tolist(),
            'max_abs': float(diff.max()), 'first_image_unchanged': True}


def run(cli):
    out = Path(cli.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    seed(1)
    torch.set_num_threads(1)
    args = build_parser().parse_args(['--trainpath', cli.trainpath, '--trainlist', cli.scan_list,
                                     '--output_json', str(out/'unused.json'), '--audit_epoch', '12'])
    args.nviews = cli.nviews
    dataset = find_dataset_def('dtu_yao')(cli.trainpath, cli.scan_list, 'val', cli.nviews, 192, 1.06)
    views = [int(v) for v in cli.views.split(',')]
    indices = [i for i, meta in enumerate(dataset.metas) if meta[1] == cli.light and meta[2] in views]
    if len(indices) != len(Path(cli.scan_list).read_text().split()) * len(views):
        raise ValueError('Incomplete scene/view selection')
    manifest = {'status': 'running', 'command': sys.argv, 'arguments': vars(cli),
                'source': source_record(), 'script_sha256': sha256(__file__),
                'scan_list_sha256': sha256(cli.scan_list), 'sample_metas': [dataset.metas[i] for i in indices],
                'aggregation': 'pooled GT-valid stage3 pixels; training-format 512x640; light fixed',
                'scope': '2D diagnostic; high confidence is a candidate mask, not Gipuma visibility',
                'missing': ['Gipuma geometric/final mask rates', 'occlusion labels', 'DTU Acc/Comp/Overall'],
                'alpha': ALPHAS, 'checkpoints': {}}
    write_json(out/'manifest.json', manifest)
    all_results = {}
    for checkpoint_path in cli.checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        epoch = checkpoint['epoch']
        name = 'epoch%d' % epoch
        if name in all_results:
            raise ValueError('Duplicate checkpoint epoch')
        model = build_model(args).to(cli.device).eval()
        model.load_state_dict(checkpoint['model'], strict=True)
        model.set_sger_residual_scale(1.0)
        manifest['checkpoints'][name] = {'path': checkpoint_path, 'sha256': sha256(checkpoint_path),
                                         'epoch': epoch, 'sger_residual_scale': 1.0}
        write_json(out/'manifest.json', manifest)
        captured, cue_samples = [], []
        handle = model.sger_blocks[0].cues.register_forward_pre_hook(
            lambda module, inputs: captured.append(tuple(v.detach().clone() for v in inputs)))
        records = []
        loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=0)
        for index, sample in zip(indices, loader):
            sample = _move_to_device(sample, cli.device)
            with torch.no_grad():
                stage = model(sample['imgs'], sample['proj_matrices'], sample['depth_values'])['stage3']
                edge = sobel_gradient_magnitude(sample['imgs'][:, 0], stage['depth_raw'].shape[-2:])
            scan = dataset.metas[index][0]
            if not cue_samples or scan != records[0]['scan'] and len(cue_samples) == 1:
                cue_samples.append(captured[0])
            captured.clear()
            def array(value):
                return value.detach().cpu().numpy()[0]
            row = statistics(*[array(stage[k]) for k in ('depth_raw', 'depth_refined', 'raw_depth_residual', 'benefit_gate')],
                             array(sample['depth']['stage3']), array(sample['mask']['stage3']) > .5,
                             array(stage['photometric_confidence']), array(edge))
            records.append({'scan': scan, 'meta': dataset.metas[index], 'statistics': row})
            print(name, scan, dataset.metas[index][2], 'gain', pool([row])['regions']['all']['means']['net_gain'], flush=True)
        handle.remove()
        result = {'aggregate': pool([r['statistics'] for r in records]),
                  'per_scene': {scan: pool([r['statistics'] for r in records if r['scan'] == scan])
                                for scan in sorted(set(r['scan'] for r in records))},
                  'records': records}
        if len(cue_samples) == 2:
            result['batch_cue_drift'] = cue_batch_drift(model.sger_blocks[0].cues, cue_samples)
        all_results[name] = result
        write_json(out/'summary.json', all_results)
        del model, checkpoint, cue_samples, stage
        if cli.device == 'cuda':
            torch.cuda.empty_cache()
    manifest['status'] = 'complete'
    write_json(out/'manifest.json', manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--trainpath', required=True)
    parser.add_argument('--scan_list', required=True)
    parser.add_argument('--views', default='0,24,48')
    parser.add_argument('--light', type=int, default=3)
    parser.add_argument('--nviews', type=int, default=3)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--output_dir', required=True)
    run(parser.parse_args())
