"""Run and verify experiment 15-D BCE audits using the recorded 15-C protocol."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime

from gradient_audit import build_parser

ROOT = Path('outputs/experiment15_d')
SOURCE = ['models/module.py','models/sger_refinement.py','models/cas_mvsnet.py',
          'gradient_audit.py','gradient_audit_utils.py','gate_audit_utils.py',
          'gate_bce_audit_utils.py','residual_audit_utils.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)


def verify_pair(epoch, batch, activation):
    stem=ROOT/('audit_'+activation)/('gate_bce_epoch%d_batch%04d_%s'%(epoch,batch,activation))
    comparison_path=Path(str(stem)+'_bce_comparison.json')
    comp=json.loads(comparison_path.read_text())
    assert len(comp['case_forward_invariance'])==15
    assert all(v==0 for c in comp['case_forward_invariance'].values() for v in c.values())
    reports={}
    files=[comparison_path]
    for mode in ('legacy_clamp','stable_logprob'):
        p=Path(str(stem)+'_'+mode+'.json');files.append(p)
        d=json.loads(p.read_text());reports[mode]=d
        assert len(d['cases'])==15
        assert d['metadata']['actual_gate_bce_mode']==mode
        assert d['metadata']['actual_residual_final_activation']==activation
        assert d['metadata']['checkpoint_epoch']==d['metadata']['audit_epoch']==epoch
        assert not d['metadata']['checkpoint_missing_keys'] and not d['metadata']['checkpoint_unexpected_keys']
        for c in d['cases']:
            assert c['gradient_summary']['all_finite']
            assert all(g['all_finite'] and not g['missing_optimizer_tensors'] for g in c['parameter_gradients'].values())
            if c['gradient_summary']['sger_exclusive']:
                assert c['gradient_summary']['backbone_grad_norm']==0
            b=c['loss_extras']['residual_bound_checks']
            assert b['proposal_all_finite'] and b['effective_all_finite']
            assert b['max_abs_proposal']<=b['proposal_bound']+1e-6
            assert b['max_abs_effective_residual']<=b['effective_bound']+1e-6
            assert not c['loss_extras']['gate_bce_gradients']['unreachable_positive_count']
        gate=next(c for c in d['cases'] if c['name']=='gate_benefit_only')
        assert gate['parameter_gradients']['sger_residual_body']['l2']==0
        assert gate['parameter_gradients']['sger_residual_output']['l2']==0
        if mode=='stable_logprob':
            assert not d['fatal_errors'],d['fatal_errors']
            assert gate['gradient_summary']['benefit_gate_grad_norm']>0
            for branch in gate['loss_extras']['gate_bce_gradients']['branches'].values():
                assert branch['all_finite']
                for label in ('positive','negative'):
                    assert branch[label]['wrong_sign_count']==0
                if activation=='leaky_relu' and epoch in (9,12):
                    stats=branch['saturated_positive']
                    assert stats['correct_sign_count']==stats['count']
        else:
            assert all(e=='gate_benefit_only: missing critical SGER benefit gate gradient' for e in d['fatal_errors'])
    a=dict(reports['legacy_clamp']['metadata']);b=dict(reports['stable_logprob']['metadata'])
    a.pop('actual_gate_bce_mode');b.pop('actual_gate_bce_mode');assert a==b
    row={'epoch':epoch,'batch':batch,'activation':activation,'passed':True,'variants':{},
         'files_sha256':{str(p):sha(p) for p in files}}
    for mode,d in reports.items():
        gate=next(c for c in d['cases'] if c['name']=='gate_benefit_only')
        row['variants'][mode]={'fatal_errors':d['fatal_errors'],'full_schedule':d['cases'][0]['forward_metrics'],
            'gate_loss':gate['loss'],'gate_grad_l2':gate['gradient_summary']['benefit_gate_grad_norm'],
            'logit_directions':gate['loss_extras']['gate_bce_gradients']}
    return row


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('pilots','audit'),default='pilots')
    args=parser.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
    protocol=json.loads(Path('outputs/experiment15_c/activation_ab_verified_summary_20260913.json').read_text())
    manifest_path=ROOT/'run_manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'runs':[]}
    def run(epoch,activation,batches):
        values=dict(next(r['arguments'] for r in protocol['rows'] if r['epoch']==epoch))
        # Historical manifests predate the explicit normalization option.
        values.setdefault('sger_depth_normalization', 'batch')
        values.update(activation_ab=False,gate_bce_ab=True,gate_bce_mode='legacy_clamp',
                      sger_residual_final_activation=activation,batch_indices=','.join(map(str,batches)),
                      output_json=str(ROOT/('audit_'+activation)/('gate_bce_epoch%d.json'%epoch)))
        command=[sys.executable,'-u','gradient_audit.py']
        for k,v in sorted(values.items()):
            if k=='use_dual_region_curvature':
                if not v:command.append('--disable_dual_region_curvature')
                continue
            if v is True:command.append('--'+k)
            elif v is not False and v is not None:command.extend(['--'+k,str(v)])
        parsed=vars(build_parser().parse_args(command[3:]))
        assert parsed==values,(set(parsed.items())^set(values.items()))
        source_hashes={p:sha(p) for p in SOURCE}
        log=ROOT/('audit_'+activation)/('epoch%d_batches%s.log'%(epoch,'-'.join(map(str,batches))))
        record={'started':datetime.now().isoformat(),'command':command,
                'shell_command':'OMP_NUM_THREADS=1 PYTHONPATH=. '+ ' '.join(shlex.quote(p) for p in command),
                'arguments':parsed,'source_sha256':source_hashes,'checkpoint_sha256':sha(values['loadckpt']),
                'log':str(log),'status':'running'}
        manifest['runs'].append(record);dump(manifest_path,manifest)
        log.parent.mkdir(parents=True,exist_ok=True)
        print('Starting',epoch,activation,batches,flush=True)
        env=dict(os.environ,OMP_NUM_THREADS='1',PYTHONPATH='.')
        with log.open('w') as f:code=subprocess.call(command,stdout=f,stderr=subprocess.STDOUT,env=env)
        record.update(exit_code=code,finished=datetime.now().isoformat(),status='verifying')
        dump(manifest_path,manifest)
        assert code in (0,1),'audit crashed; see '+str(log)
        try:verified=[verify_pair(epoch,b,activation) for b in batches]
        except Exception as error:
            record.update(status='verification_failed',error=repr(error));dump(manifest_path,manifest);raise
        record.update(status='verified',pairs=verified);dump(manifest_path,manifest)
        print('Verified',epoch,activation,batches,flush=True)
        return verified
    # Resume only verified pairs produced by exactly the current audit source.
    def existing(e,a,b):
        hashes={p:sha(p) for p in SOURCE}
        for r in manifest['runs']:
            if r['status']=='verified' and r['source_sha256']==hashes:
                for pair in r['pairs']:
                    if (pair['epoch'],pair['activation'],pair['batch'])==(e,a,b):
                        assert all(sha(p)==h for p,h in pair['files_sha256'].items())
                        return verify_pair(e,b,a)
        return None
    pairs=[]
    for e,a in ((9,'leaky_relu'),(15,'relu')):
        found=existing(e,a,0)
        pairs.extend([found] if found else run(e,a,[0]))
    dump(ROOT/'pilot_summary.json',{'passed':True,'pairs':pairs})
    if args.phase=='audit':
        pairs=[]
        for e,a in ((9,'relu'),(12,'relu'),(15,'relu'),(9,'leaky_relu'),(12,'leaky_relu')):
            missing=[]
            for b in range(5):
                found=existing(e,a,b)
                if found:pairs.append(found)
                else:missing.append(b)
            if missing:pairs.extend(run(e,a,missing))
        assert len(pairs)==25
        dump(ROOT/'audit_summary.json',{'passed':True,'pairs':pairs,'pair_count':25,'case_count':750})
    print('Experiment15-D',args.phase,'PASS',flush=True)


if __name__=='__main__':
    main()
