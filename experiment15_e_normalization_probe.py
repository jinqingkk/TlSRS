"""Single-variable SGER geometry normalization probe with frozen learned backbone."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets import find_dataset_def
from evaluation_record import sha256, source_record, write_json
from experiment15_d_probe import build_model, restore_optimizer, seed
from experiment15_d_diagnose import freeze_backbone_parameters, set_diagnostic_train_mode
from experiment15_e_diagnose import statistics, pool
from gradient_audit import (build_parser, _move_to_device, _base_loss_weights,
                            _loss_kwargs, _is_sger_lite_parameter)
from gradient_audit_utils import build_audit_cases
from models import cas_mvsnet_loss
from models.module import sobel_gradient_magnitude


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    rows, raw_maps = [], []
    for sample in loader:
        sample = _move_to_device(sample, device)
        stage = model(sample['imgs'], sample['proj_matrices'], sample['depth_values'])['stage3']
        edge = sobel_gradient_magnitude(sample['imgs'][:, 0], stage['depth_raw'].shape[-2:])
        def array(value):
            return value.detach().cpu().numpy()[0]
        rows.append(statistics(*[array(stage[k]) for k in ('depth_raw', 'depth_refined', 'raw_depth_residual', 'benefit_gate')],
                               array(sample['depth']['stage3']), array(sample['mask']['stage3']) > .5,
                               array(stage['photometric_confidence']), array(edge)))
        raw_maps.append(array(stage['depth_raw']).copy())
    return pool(rows), raw_maps


def run(cli):
    if cli.steps < 1:
        raise ValueError('steps must be positive')
    train_scans = set(Path(cli.trainlist).read_text().split())
    val_scans = set(Path(cli.vallist).read_text().split())
    retrospective = set(Path(cli.retrospective_list).read_text().split())
    if train_scans & val_scans or val_scans & retrospective:
        raise ValueError('Validation scenes must be disjoint from training and retrospective test scenes')
    out = Path(cli.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    args = build_parser().parse_args(['--trainpath', cli.trainpath, '--trainlist', cli.trainlist,
                                     '--output_json', str(out/'unused.json'), '--audit_epoch', '12'])
    args.batch_size, args.gate_bce_mode = 2, 'stable_logprob'
    checkpoint = torch.load(cli.checkpoint, map_location='cpu')
    epoch = checkpoint['epoch']
    if epoch < 10 or checkpoint.get('sger_depth_normalization', 'batch') != 'batch':
        raise ValueError('Use a fully warmed-up legacy batch-normalized checkpoint')
    case = build_audit_cases(epoch, _base_loss_weights(args), 7, 10)[0]
    dataset_type = find_dataset_def('dtu_yao')
    trainset = dataset_type(cli.trainpath, cli.trainlist, 'train', 3, 192, 1.06)
    valset = dataset_type(cli.trainpath, cli.vallist, 'val', 3, 192, 1.06)
    train_indices = np.random.RandomState(1).choice(len(trainset), cli.steps*2, replace=False).tolist()
    val_indices = [i for i, meta in enumerate(valset.metas) if meta[1] == 3 and meta[2] == 24]
    manifest = {'status': 'running', 'command': sys.argv, 'arguments': vars(cli),
                'source': source_record(), 'script_sha256': sha256(__file__),
                'checkpoint_sha256': sha256(cli.checkpoint), 'checkpoint_epoch': epoch,
                'lists_sha256': {p: sha256(p) for p in (cli.trainlist, cli.vallist, cli.retrospective_list)},
                'train_indices': train_indices, 'train_metas': [trainset.metas[i] for i in train_indices],
                'val_indices': val_indices, 'val_metas': [valset.metas[i] for i in val_indices],
                'validation_scope': 'pooled stage3 valid pixels; 512x640; light3/view24 per independent scene',
                'optimizer': 'restore identical Adam state and saved learning rates; no scheduler',
                'freeze': 'backbone parameters and BN; final normal head and SGER learn',
                'unique_change': 'SGER cue and gate geometry depth normalization: batch -> sample',
                'loss_weights': dict(case.loss_weights), 'residual_scale': case.sger_scale,
                'residual_target_scale': case.residual_target_scale, 'gate_benefit_scale': case.gate_benefit_scale,
                'seed': 1, 'runs': {}}
    write_json(out/'manifest.json', manifest)
    raw_reference = None
    for mode in ('batch', 'sample'):
        seed(1)
        args.sger_depth_normalization = mode
        model = build_model(args).cuda()
        model.load_state_dict(checkpoint['model'], strict=True)
        model.set_sger_residual_scale(case.sger_scale)
        optimizer, mapping = restore_optimizer(model, args, checkpoint)
        freeze_backbone_parameters(model)
        covered = {id(p) for group in optimizer.param_groups for p in group['params']}
        if any(p.requires_grad and id(p) not in covered for p in model.parameters()):
            raise RuntimeError('Optimizer misses trainable parameters')
        protected = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                     if not _is_sger_lite_parameter(model, k)}
        initial_trainable = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}
        valloader = DataLoader(Subset(valset, val_indices), batch_size=1, shuffle=False, num_workers=0)
        before, raw_maps = evaluate(model, valloader, 'cuda')
        if raw_reference is None:
            raw_reference = raw_maps
        elif not all(np.array_equal(a, b) for a, b in zip(raw_reference, raw_maps)):
            raise RuntimeError('Raw baseline changed before training')
        result = {'before': before, 'steps': [], 'optimizer_groups':
                  [{'name': g.get('name'), 'lr': g['lr']} for g in optimizer.param_groups]}
        trainloader = DataLoader(Subset(trainset, train_indices), batch_size=2, shuffle=False, num_workers=0)
        for step, sample in enumerate(trainloader, 1):
            set_diagnostic_train_mode(model, True)
            sample = _move_to_device(sample, 'cuda')
            optimizer.zero_grad(set_to_none=True)
            outputs = model(sample['imgs'], sample['proj_matrices'], sample['depth_values'])
            losses = cas_mvsnet_loss(outputs, sample['depth'], sample['mask'], **_loss_kwargs(args, sample, case))
            loss = losses[0]
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            absent, zero, nonzero = [], [], []
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    if parameter.grad is not None:
                        raise RuntimeError('Unexpected protected gradient: ' + name)
                elif parameter.grad is None:
                    absent.append(name)
                elif not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError('Nonfinite gradient: ' + name)
                elif bool((parameter.grad != 0).any()):
                    nonzero.append(name)
                else:
                    zero.append(name)
            if absent or not nonzero:
                raise RuntimeError('Missing trainable gradients: ' + str(absent))
            optimizer.step()
            result['steps'].append({'step': step, 'loss': float(loss), 'zero_gradient': zero,
                                    'nonzero_gradient_tensors': len(nonzero)})
            print(mode, step, float(loss), flush=True)
            del outputs, losses, loss
        for name, value in model.state_dict().items():
            if name in protected and not torch.equal(protected[name], value.cpu()):
                raise RuntimeError('Frozen backbone/BN changed: ' + name)
        after, raw_maps = evaluate(model, valloader, 'cuda')
        if not all(np.array_equal(a, b) for a, b in zip(raw_reference, raw_maps)):
            raise RuntimeError('Raw validation changed after training')
        result.update(after=after, protected_state_exact=True, raw_validation_exact=True,
                      updated_parameters=[name for name, parameter in model.named_parameters()
                                          if name in initial_trainable and not torch.equal(initial_trainable[name], parameter.cpu())])
        if not result['updated_parameters']:
            raise RuntimeError('No actual SGER parameter update')
        manifest['runs'][mode] = result
        write_json(out/'manifest.json', manifest)
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                    'probe_steps': cli.steps, 'sger_depth_normalization': mode, 'gate_bce_mode': 'stable_logprob'},
                   str(out/(mode+'.ckpt')))
        del model, optimizer
        torch.cuda.empty_cache()
    def metric(mode, key):
        return manifest['runs'][mode]['after']['regions']['all']['means'][key]
    manifest['screening'] = {'net_gain_change': metric('sample', 'net_gain')-metric('batch', 'net_gain'),
                             'harm_change': metric('sample', 'harm')-metric('batch', 'harm'),
                             'formal_training_authorized_by_evidence': False,
                             'note': 'Short probe only; require repeat/scene stability and matched DTU three-metric evaluation'}
    manifest['status'] = 'complete'
    write_json(out/'manifest.json', manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--trainpath', required=True)
    parser.add_argument('--trainlist', default='lists/dtu/train.txt')
    parser.add_argument('--vallist', default='lists/dtu/val.txt')
    parser.add_argument('--retrospective_list', default='lists/dtu/test.txt')
    parser.add_argument('--steps', type=int, default=25)
    parser.add_argument('--output_dir', required=True)
    run(parser.parse_args())
