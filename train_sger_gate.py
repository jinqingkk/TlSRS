"""Experiment 16: train SGER gates with a fixed, learned CasMVSNet/proposal.

Supports a paired four-arm probe and ordinary epoch training/resume. All
checkpoints contain the complete CascadeMVSNet and load directly in test.py.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets import find_dataset_def
from evaluation_record import sha256, source_record, write_json
from experiment15_d_probe import build_model, seed
from experiment15_e_diagnose import statistics, pool
from gradient_audit import build_parser, _move_to_device, _base_loss_weights, _loss_kwargs
from gradient_audit_utils import build_audit_cases
from models import cas_mvsnet_loss
from models.module import sobel_gradient_magnitude


VARIANTS = {'baseline': (False, 'binary'), 'proposal': (True, 'binary'),
            'step': (False, 'feasible_step'), 'proposal_step': (True, 'feasible_step')}


def freeze_except_gate(model):
    """Protect raw depth, proposal features, normal heads, and all BN buffers."""
    model.eval()
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith('sger_blocks.0.gate.'))
    model.sger_blocks[0].gate.train()


def load_initial_weights(model, checkpoint, proposal_conditioned):
    missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
    allowed = {'sger_blocks.0.gate.proposal_a.weight', 'sger_blocks.0.gate.proposal_b.weight'}
    expected = allowed if proposal_conditioned and not checkpoint.get('sger_proposal_gate', False) else set()
    if set(missing) != expected or unexpected:
        raise ValueError('Incompatible pretrained model: missing={}, unexpected={}'.format(missing, unexpected))
    return list(missing)


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@torch.no_grad()
def evaluate(model, loader, metas, device):
    model.eval()
    rows, hashes = [], []
    for meta, sample in zip(metas, loader):
        sample = _move_to_device(sample, device)
        stage = model(sample['imgs'], sample['proj_matrices'], sample['depth_values'])['stage3']
        edge = sobel_gradient_magnitude(sample['imgs'][:, 0], stage['depth_raw'].shape[-2:])
        def array(value):
            return value.detach().cpu().numpy()[0]
        row = statistics(*[array(stage[k]) for k in ('depth_raw', 'depth_refined', 'raw_depth_residual', 'benefit_gate')],
                         array(sample['depth']['stage3']), array(sample['mask']['stage3']) > .5,
                         array(stage['photometric_confidence']), array(edge))
        rows.append({'scan': meta[0], 'meta': meta, 'statistics': row})
        hashes.append({k: tensor_hash(stage[k]) for k in
                       ('depth_raw', 'raw_depth_residual', 'photometric_confidence')})
    return {'aggregate': pool([r['statistics'] for r in rows]),
            'per_scene': {scan: pool([r['statistics'] for r in rows if r['scan'] == scan])
                          for scan in sorted({r['scan'] for r in rows})},
            'protected_output_sha256': hashes}


def run(cli):
    if cli.epochs < 1 or cli.steps < 0 or cli.batch_size < 1 or not 0 < cli.lr < 1:
        raise ValueError('Invalid training settings')
    variants = cli.variants.split(',')
    if len(set(variants)) != len(variants) or any(v not in VARIANTS for v in variants):
        raise ValueError('Use distinct variants from ' + str(list(VARIANTS)))
    if cli.resume and len(variants) != 1:
        raise ValueError('Resume one variant at a time')
    lists = [cli.trainlist, cli.vallist, cli.testlist]
    scans = [set(Path(p).read_text().split()) for p in lists]
    if any(scans[i] & scans[j] for i in range(3) for j in range(i)):
        raise ValueError('Training, validation, and test scenes must be disjoint')
    out = Path(cli.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new output directory, including for resume')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    checkpoint_path = cli.resume or cli.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if checkpoint.get('sger_depth_normalization', 'batch') != 'batch':
        raise ValueError('This controlled experiment requires batch geometry normalization')
    if not cli.resume and checkpoint.get('epoch', -1) < 10:
        raise ValueError('Initialize from a fully trained 15-E checkpoint, epoch >= 10')
    args = build_parser().parse_args(['--trainpath', cli.trainpath, '--trainlist', cli.trainlist,
                                     '--output_json', str(out/'unused.json'), '--audit_epoch', '12'])
    args.gate_bce_mode = 'stable_logprob'
    case = build_audit_cases(12, _base_loss_weights(args), 7, 10)[0]
    dataset_type = find_dataset_def('dtu_yao')
    trainset = dataset_type(cli.trainpath, cli.trainlist, 'train', 3, 192, 1.06)
    valset = dataset_type(cli.trainpath, cli.vallist, 'val', 3, 192, 1.06)
    val_indices = [i for i, meta in enumerate(valset.metas)
                   if cli.validation == 'all' or (meta[1] == 3 and meta[2] == 24)]
    if not val_indices:
        raise ValueError('Empty validation set')
    val_metas = [valset.metas[i] for i in val_indices]
    valloader = DataLoader(Subset(valset, val_indices), batch_size=1, shuffle=False, num_workers=0)
    manifest = {'status': 'running', 'command': sys.argv, 'arguments': vars(cli),
                'source': source_record(), 'script_sha256': sha256(__file__),
                'checkpoint_sha256': sha256(checkpoint_path), 'source_epoch': checkpoint['epoch'],
                'lists_sha256': {p: sha256(p) for p in lists}, 'val_metas': val_metas,
                'optimizer': 'Adam gate parameters only; common fresh state unless explicit resume; constant LR',
                'freeze': 'all parameters except stage3 SGER gate; all backbone/normal/proposal BN fixed',
                'loss_weights': dict(case.loss_weights), 'beta': 1.0,
                'validation_scope': 'GT-valid pixel pooling, 512x640, 3 views; full validation only if requested',
                'runs': {}}
    write_json(out/'manifest.json', manifest)
    raw_reference, initial_reference = None, None
    for variant in variants:
        seed(cli.seed)
        args.sger_proposal_gate, gate_supervision = VARIANTS[variant]
        model = build_model(args).to(cli.device)
        missing = load_initial_weights(model, checkpoint, args.sger_proposal_gate)
        model.set_sger_residual_scale(1.0)
        freeze_except_gate(model)
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        optimizer = torch.optim.Adam([p for n, p in trainable], lr=cli.lr)
        start_epoch = 0
        if cli.resume:
            if checkpoint.get('gate_variant') != variant:
                raise ValueError('Resume variant mismatch')
            previous = checkpoint['training_arguments']
            for key in ('seed', 'lr', 'batch_size', 'steps', 'validation', 'trainpath', 'trainlist', 'vallist', 'testlist'):
                if previous[key] != vars(cli)[key]:
                    raise ValueError('Resume setting mismatch: ' + key)
            if checkpoint['lists_sha256'] != manifest['lists_sha256']:
                raise ValueError('Resume data splits changed')
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
        if start_epoch >= cli.epochs:
            raise ValueError('--epochs is the total target epoch count and must exceed resumed epoch')
        protected = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()
                     if not n.startswith('sger_blocks.0.gate.')}
        before = evaluate(model, valloader, val_metas, cli.device)
        if raw_reference is None:
            raw_reference = before['protected_output_sha256']
            initial_reference = before['aggregate']
        elif raw_reference != before['protected_output_sha256'] or before['aggregate'] != initial_reference:
            raise RuntimeError('Paired variants must have identical initial outputs')
        run_record = {'initialized_keys': missing, 'before': before,
                      'trainable_parameters': [n for n, p in trainable], 'epochs': []}
        manifest['runs'][variant] = run_record
        for epoch in range(start_epoch, cli.epochs):
            seed(cli.seed + epoch)
            order = np.random.RandomState(cli.seed + epoch).permutation(len(trainset)).tolist()
            if cli.steps:
                if cli.steps * cli.batch_size > len(order):
                    raise ValueError('Requested steps exceed one training epoch')
                order = order[:cli.steps * cli.batch_size]
            loader = DataLoader(Subset(trainset, order), batch_size=cli.batch_size,
                                shuffle=False, num_workers=0, drop_last=True)
            initial_gate = {n: p.detach().cpu().clone() for n, p in trainable}
            epoch_record = {'epoch': epoch, 'train_indices': order, 'steps': []}
            for step, sample in enumerate(loader, 1):
                freeze_except_gate(model)
                sample = _move_to_device(sample, cli.device)
                optimizer.zero_grad()
                outputs = model(sample['imgs'], sample['proj_matrices'], sample['depth_values'])
                options = _loss_kwargs(args, sample, case)
                options['gate_supervision'] = gate_supervision
                losses = cas_mvsnet_loss(outputs, sample['depth'], sample['mask'], **options)
                loss = losses[0]
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Nonfinite training loss')
                loss.backward()
                zero = []
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                            raise RuntimeError('Missing/nonfinite gate gradient: ' + name)
                        if not bool((parameter.grad != 0).any()):
                            zero.append(name)
                    elif parameter.grad is not None:
                        raise RuntimeError('Unexpected protected gradient: ' + name)
                if len(zero) == len(trainable):
                    raise RuntimeError('All gate gradients are zero')
                optimizer.step()
                epoch_record['steps'].append({'step': step, 'loss': float(loss), 'zero_gradient': zero,
                                               'metrics': {k: float(v) for k, v in losses[-1].items()
                                                           if torch.is_tensor(v) and v.numel() == 1}})
                if step == 1 or step % 10 == 0:
                    print(variant, 'epoch', epoch, 'step', step, 'loss', float(loss), flush=True)
                del outputs, loss, losses
            for name, value in model.state_dict().items():
                if name in protected and not torch.equal(protected[name], value.detach().cpu()):
                    raise RuntimeError('Protected parameter or buffer changed: ' + name)
            after = evaluate(model, valloader, val_metas, cli.device)
            if after['protected_output_sha256'] != raw_reference:
                raise RuntimeError('Raw depth, confidence, or proposal changed')
            updated = [n for n, p in trainable if not torch.equal(initial_gate[n], p.detach().cpu())]
            if not updated:
                raise RuntimeError('No gate parameter updated')
            epoch_record.update(after=after, updated_parameters=updated, protected_state_exact=True)
            run_record['epochs'].append(epoch_record)
            checkpoint_out = out / '{}_epoch{:03d}.ckpt'.format(variant, epoch)
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                        'source_checkpoint_sha256': manifest['checkpoint_sha256'], 'source_epoch': checkpoint['epoch'],
                        'sger_depth_normalization': 'batch', 'sger_proposal_gate': args.sger_proposal_gate,
                        'gate_supervision': gate_supervision, 'gate_bce_mode': 'stable_logprob',
                        'sger_residual_scale': 1.0, 'gate_variant': variant,
                        'training_arguments': vars(cli), 'lists_sha256': manifest['lists_sha256']}, str(checkpoint_out))
            epoch_record['checkpoint'] = str(checkpoint_out)
            epoch_record['checkpoint_sha256'] = sha256(checkpoint_out)
            write_json(out/'manifest.json', manifest)
            means = after['aggregate']['regions']['all']['means']
            print(variant, 'validation', {k: means[k] for k in ('net_gain', 'harm', 'raw_abs_error', 'refined_abs_error')}, flush=True)
        del model, optimizer
        if cli.device == 'cuda':
            torch.cuda.empty_cache()
    manifest['status'] = 'complete'
    manifest['dtu_metrics'] = 'not evaluated; no point-cloud quality claim'
    write_json(out/'manifest.json', manifest)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='checkpoints/experiment15_e_formal_stable_bce/model_000012.ckpt')
    p.add_argument('--resume', help='resume a checkpoint produced by this trainer into a new output directory')
    p.add_argument('--trainpath', default='data/DTU/mvs_training/dtu')
    p.add_argument('--trainlist', default='lists/dtu/train.txt')
    p.add_argument('--vallist', default='lists/dtu/val.txt')
    p.add_argument('--testlist', default='lists/dtu/test.txt', help='exclusion only; never used for model selection')
    p.add_argument('--variants', default='proposal', help=','.join(VARIANTS))
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--steps', type=int, default=0, help='0: full epochs; otherwise deterministic subset per epoch')
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--validation', choices=('sample', 'all'), default='sample')
    p.add_argument('--device', default='cuda')
    p.add_argument('--output_dir', required=True)
    return p


if __name__ == '__main__':
    cli = parser().parse_args()
    # A failed fresh invocation must not leave a misleading "running" record.
    manifest_path = Path(cli.output_dir) / 'manifest.json'
    existed = manifest_path.exists()
    try:
        run(cli)
    except Exception as error:
        if not existed and manifest_path.exists():
            record = json.loads(manifest_path.read_text())
            record.update(status='failed', error=repr(error))
            write_json(manifest_path, record)
        raise
