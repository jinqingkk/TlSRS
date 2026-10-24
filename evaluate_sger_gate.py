"""Paired GT validation of complete SGER checkpoints on held-out scenes."""
import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset

from datasets import find_dataset_def
from evaluation_record import sha256, source_record, write_json
from experiment15_d_probe import build_model, seed
from gradient_audit import build_parser
from train_sger_gate import evaluate


def run(cli):
    scans = [set(Path(p).read_text().split()) for p in
             (cli.trainlist, cli.vallist, cli.testlist)]
    if scans[1] & (scans[0] | scans[2]):
        raise ValueError('Validation scenes overlap training or test scenes')
    out = Path(cli.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    seed(1)
    dataset = find_dataset_def('dtu_yao')(cli.trainpath, cli.vallist, 'val', 3, 192, 1.06)
    views = {int(v) for v in cli.views.split(',')}
    indices = [i for i, meta in enumerate(dataset.metas) if meta[1] == 3 and meta[2] in views]
    if len(indices) != len(scans[1]) * len(views):
        raise ValueError('Missing validation views')
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=0)
    metas = [dataset.metas[i] for i in indices]
    args = build_parser().parse_args(['--trainpath', cli.trainpath, '--trainlist', cli.trainlist,
                                     '--output_json', str(out/'unused.json'), '--audit_epoch', '12'])
    record = {'status': 'running', 'command': sys.argv, 'arguments': vars(cli),
              'source': source_record(), 'script_sha256': sha256(__file__),
              'validation_metas': metas, 'runs': {},
              'scope': '512x640, 3 views, light3, pooled GT-valid pixels; not DTU point-cloud scores'}
    write_json(out/'summary.json', record)
    protected = None
    for path in cli.checkpoints:
        checkpoint = torch.load(path, map_location='cpu')
        args.sger_proposal_gate = checkpoint.get('sger_proposal_gate', False)
        args.sger_depth_normalization = checkpoint.get('sger_depth_normalization', 'batch')
        model = build_model(args).to(cli.device).eval()
        model.load_state_dict(checkpoint['model'], strict=True)
        model.set_sger_residual_scale(1.)
        result = evaluate(model, loader, metas, cli.device)
        if protected is None:
            protected = result['protected_output_sha256']
        elif result['protected_output_sha256'] != protected:
            raise RuntimeError('Expected fixed raw/confidence/proposal across this comparison')
        record['runs'][path] = {'checkpoint_sha256': sha256(path), 'checkpoint_epoch': checkpoint['epoch'],
                                'gate_variant': checkpoint.get('gate_variant', '15-E'), 'result': result}
        write_json(out/'summary.json', record)
        means = result['aggregate']['regions']['all']['means']
        print(path, {k: means[k] for k in ('raw_abs_error', 'refined_abs_error', 'net_gain', 'harm')}, flush=True)
        del model
    record['status'] = 'complete'
    write_json(out/'summary.json', record)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', nargs='+', required=True)
    p.add_argument('--trainpath', default='data/DTU/mvs_training/dtu')
    p.add_argument('--trainlist', default='lists/dtu/train.txt')
    p.add_argument('--vallist', default='lists/dtu/val.txt')
    p.add_argument('--testlist', default='lists/dtu/test.txt')
    p.add_argument('--views', default='0,24,48')
    p.add_argument('--device', default='cuda')
    p.add_argument('--output_dir', required=True)
    run(p.parse_args())
