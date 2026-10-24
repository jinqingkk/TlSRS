"""Paired 200-step local training probe, gated by the completed 15-D audits."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import find_dataset_def
from models import CascadeMVSNet, cas_mvsnet_loss
from gradient_audit import build_parser, _build_optimizer, _move_to_device, _loss_kwargs, _base_loss_weights
from gradient_audit_utils import build_audit_cases, _forward_metrics, _json_value
from models.module import residual_benefit_target
from experiment15_d import ROOT, SOURCE, sha, dump


def build_model(args):
    return CascadeMVSNet(refine=False, ndepths=[int(x) for x in args.ndepths.split(',')],
        depth_interals_ratio=[float(x) for x in args.depth_inter_r.split(',')],
        share_cr=args.share_cr,cr_base_chs=[int(x) for x in args.cr_base_chs.split(',')],
        grad_method=args.grad_method,use_sger_lite=True,sger_share=args.sger_share,
        sger_feature_channels=args.sger_feature_channels,sger_hidden_channels=args.sger_hidden_channels,
        sger_gate_channels=args.sger_gate_channels,sger_max_residual_ratio=args.sger_max_residual_ratio,
        sger_residual_final_activation='relu',sger_residual_negative_slope=args.sger_residual_negative_slope,
        sger_depth_normalization=getattr(args, "sger_depth_normalization", "batch"),
        sger_proposal_gate=getattr(args, "sger_proposal_gate", False),
        detach_refined_feedback=True,sger_gate_kwargs={
            'threshold_edge':args.region_edge_threshold,'threshold_depth':args.region_depth_threshold,
            'threshold_curv':args.region_curv_threshold,'conf_mid':args.geometry_conf_mid,
            'k_conf':args.geometry_k_conf,'smooth_k':args.region_smooth_k})


def restore_optimizer(model,args,checkpoint):
    optimizer=_build_optimizer(model,args)
    state=checkpoint.get('optimizer')
    if state is None:
        raise ValueError('No saved optimizer; explicit common reset configuration required')
    # Reconstruct the established grouping in named_parameters order and verify
    # every available momentum tensor shape before restoring, not just counts.
    names={id(p):n for n,p in model.named_parameters()}
    assert len(state['param_groups'])==len(optimizer.param_groups)
    mapping=[]
    for saved,current in zip(state['param_groups'],optimizer.param_groups):
        assert saved.get('name')==current.get('name')
        assert len(saved['params'])==len(current['params'])
        group=[]
        for key,p in zip(saved['params'],current['params']):
            for field in ('exp_avg','exp_avg_sq'):
                tensor=state['state'].get(key,{}).get(field)
                if tensor is not None:assert tensor.shape==p.shape,(names[id(p)],field)
            group.append({'name':names[id(p)],'saved_id':key,'shape':list(p.shape),
                          'has_momentum':key in state['state']})
        mapping.append({'name':current['name'],'parameters':group})
    optimizer.load_state_dict(copy.deepcopy(state))
    for saved,current in zip(state['param_groups'],optimizer.param_groups):
        for key,p in zip(saved['params'],current['params']):
            for field,t in state['state'].get(key,{}).items():
                loaded=optimizer.state[p][field]
                if torch.is_tensor(t):assert torch.equal(t.cpu(),loaded.cpu())
                else:assert t==loaded
    return optimizer,mapping


def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def metrics(outputs,sample,extra):
    stage=outputs['stage3'];mask=sample['mask']['stage3']>0.5;gt=sample['depth']['stage3']
    result=_forward_metrics(stage,gt,mask)
    result['loss_extras']={k:float(v.detach()) for k,v in extra.items()
        if torch.is_tensor(v) and v.numel()==1}
    target=residual_benefit_target(stage['depth_raw'],stage['raw_depth_residual'],gt,
                                  stage['depth_interval'],0.05)
    predicted=stage['benefit_gate']>=0.5
    result.update(target_positive_count=int((mask&target).sum()),
                  predicted_positive_count=int((mask&predicted).sum()),
                  true_positive_count=int((mask&target&predicted).sum()))
    raw=stage['depth_raw'].detach();proposal=stage['raw_depth_residual'].detach()
    gain=(raw-gt).abs()-(raw+proposal-gt).abs()
    result['proposal_gain_mean']=float(gain[mask].double().mean())
    result['raw_sha256']=hashlib.sha256(raw.cpu().contiguous().numpy().tobytes()).hexdigest()
    result['confidence_sha256']=hashlib.sha256(stage['photometric_confidence'].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    gate=stage['benefit_gate'].detach()
    result['gate_below_1e_minus6_ratio']=float((gate[mask]<1e-6).double().mean())
    return result


def verify_prior_audits():
    audit=json.loads((ROOT/'audit_summary.json').read_text());assert audit['passed'] and audit['pair_count']==25
    for pair in audit['pairs']:
        assert all(sha(p)==h for p,h in pair['files_sha256'].items())
    manifest=json.loads((ROOT/'run_manifest.json').read_text())
    assert all(any(r['status']=='verified' and r['source_sha256']=={p:sha(p) for p in SOURCE}
        and pair in r.get('pairs',[]) for r in manifest['runs']) for pair in audit['pairs'])


def main():
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Run the independent three-arm frozen-backbone probe')
    parser.add_argument('--strict_determinism', action='store_true')
    parser.add_argument('--probe_manifest', default=str(ROOT/'short_train/sample_manifest.json'))
    args_cli=parser.parse_args()
    if args_cli.freeze_backbone:
        from experiment15_d_frozen_probe import run
        return run(args_cli)
    if args_cli.strict_determinism:
        parser.error('--strict_determinism requires --freeze_backbone')
    out=Path(args_cli.output_dir or str(ROOT/'short_train'))
    verify_prior_audits()
    recorded=json.loads(Path('outputs/experiment15_c/activation_ab_verified_summary_20260913.json').read_text())
    values=dict(next(r['arguments'] for r in recorded['rows'] if r['epoch']==15))
    values.update(activation_ab=False,gate_bce_ab=False,gate_bce_mode='legacy_clamp')
    args=argparse.Namespace(**values)
    dataset_type=find_dataset_def(args.dataset)
    trainset=dataset_type(args.trainpath,args.trainlist,'train',args.nviews,args.numdepth,args.interval_scale)
    valset=dataset_type(args.trainpath,'lists/dtu/val.txt','val',args.nviews,args.numdepth,args.interval_scale)
    train_indices=np.random.RandomState(args.seed).permutation(len(trainset))[:400].tolist()
    train_batches=[train_indices[i:i+2] for i in range(0,400,2)]
    scans=[];val_indices=[]
    for i,meta in enumerate(valset.metas):
        scan,light,ref,_=meta
        if light!=0 or ref not in (0,1):continue
        if scan not in scans:
            if len(scans)==3:continue
            scans.append(scan)
        val_indices.append(i)
    assert len(val_indices)==6 and len(scans)==3
    # Check all selected data before choosing outcomes, saving exact metadata.
    for i in val_indices:valset[i]
    checkpoint=torch.load(args.loadckpt,map_location='cpu')
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'sample_manifest.json').exists(),'Use a new output directory for a repeated probe'
    info={'status':'running','checkpoint':args.loadckpt,'checkpoint_sha256':sha(args.loadckpt),
          'arguments':values,'command':sys.argv,'steps':200,'save_steps':[0,50,100,200],
          'train_indices':train_indices,'train_metas':[trainset.metas[i] for i in train_indices],
          'val_indices':val_indices,'val_metas':[valset.metas[i] for i in val_indices],
          'val_list':'lists/dtu/val.txt','torch':torch.__version__,'cuda':torch.version.cuda,
          'source_sha256':{p:sha(p) for p in SOURCE+['train.py','experiment15_d_probe.py']},
          'schedule':'fixed epoch15 beta/loss scales and restored group learning rates; local probe'}
    dump(out/'sample_manifest.json',info)
    case=build_audit_cases(15,_base_loss_weights(args),args.sger_warmup_start_epoch,args.sger_warmup_end_epoch)[0]
    all_records={}
    for mode in ('legacy_clamp','stable_logprob'):
        seed(args.seed);args.gate_bce_mode=mode
        model=build_model(args);model.load_state_dict(checkpoint['model'],strict=True);model.cuda()
        model.set_sger_residual_scale(case.sger_scale)
        optimizer,mapping=restore_optimizer(model,args,checkpoint)
        info[mode+'_optimizer']={'mapping':mapping,'groups':[
            {k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups]}
        dump(out/'sample_manifest.json',info)
        directory=out/mode;directory.mkdir(exist_ok=True)
        records={'train':[],'validation':{}}
        def evaluate(step):
            model.eval();result=[]
            with torch.no_grad():
                loader=DataLoader(valset,batch_sampler=[val_indices[i:i+2] for i in range(0,6,2)],num_workers=0)
                for sample in loader:
                    sample=_move_to_device(sample,torch.device('cuda'))
                    outputs=model(sample['imgs'],sample['proj_matrices'],sample['depth_values'])
                    losses=cas_mvsnet_loss(outputs,sample['depth'],sample['mask'],**_loss_kwargs(args,sample,case))
                    row=metrics(outputs,sample,losses[-1]);row['loss']=float(losses[0]);result.append(row)
            records['validation'][str(step)]=result
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                        'epoch':15,'probe_step':step,'gate_bce_mode':mode,'arguments':vars(args)},
                       str(directory/('step_%04d.ckpt'%step)))
            dump(directory/'metrics.json',records);model.train()
            print(mode,'validation/snapshot',step,flush=True)
        evaluate(0)
        loader=DataLoader(trainset,batch_sampler=train_batches,num_workers=0)
        for step,sample in enumerate(loader,1):
            sample=_move_to_device(sample,torch.device('cuda'));model.train();optimizer.zero_grad(set_to_none=True)
            outputs=model(sample['imgs'],sample['proj_matrices'],sample['depth_values'])
            losses=cas_mvsnet_loss(outputs,sample['depth'],sample['mask'],**_loss_kwargs(args,sample,case))
            assert torch.isfinite(losses[0]),'nonfinite training loss'
            losses[0].backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            row=metrics(outputs,sample,losses[-1]);row.update(step=step,loss=float(losses[0]))
            optimizer.step();records['train'].append(row)
            if step%10==0:
                dump(directory/'metrics.json',records);print(mode,'step',step,'delta',row['raw_to_refined_error_delta'],flush=True)
            del outputs,losses,sample
            if step in (50,100,200):evaluate(step)
        all_records[mode]=records
        del model,optimizer
        torch.cuda.empty_cache()
    comparisons={}
    for step in ('0','50','100','200'):
        result={}
        for mode in all_records:
            rs=all_records[mode]['validation'][step];n=sum(r['valid_pixel_count'] for r in rs)
            result[mode]={k:sum(r[k]*r['valid_pixel_count'] for r in rs)/n for k in (
                'raw_abs_error','refined_abs_error','raw_to_refined_error_delta','benefit_gate_mean')}
        comparisons[step]=result
    raw_train_equal=all(a['raw_sha256']==b['raw_sha256'] and a['confidence_sha256']==b['confidence_sha256']
        for a,b in zip(all_records['legacy_clamp']['train'],all_records['stable_logprob']['train']))
    raw_val_equal=all(a['raw_sha256']==b['raw_sha256'] and a['confidence_sha256']==b['confidence_sha256']
        for step in ('0','50','100','200') for a,b in zip(all_records['legacy_clamp']['validation'][step],all_records['stable_logprob']['validation'][step]))
    info['status']='completed';dump(out/'sample_manifest.json',info)
    dump(out/'comparison.json',{'validation':comparisons,'raw_training_trajectory_exact':raw_train_equal,
                              'raw_validation_exact':raw_val_equal,'completed_steps_per_variant':200})
    print('Probe completed. Raw train/val exact:',raw_train_equal,raw_val_equal,flush=True)


if __name__=='__main__':main()
