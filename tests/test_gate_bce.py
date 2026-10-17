"""Numerical contract and production entrypoint checks for experiment 15-D."""
import argparse
import json
import math
import os
import tempfile
from unittest.mock import patch

import torch
from models.module import balanced_binary_cross_entropy, stable_gate_binary_cross_entropy


def stable(z, w, y, a=None, mask=None, zb=None, valid=None):
    if a is None:
        a = torch.zeros_like(z, dtype=torch.bool)
    if mask is None:
        mask = torch.ones_like(a)
    if valid is None:
        valid = torch.ones_like(a)
    if zb is None:
        zb = z
    return stable_gate_binary_cross_entropy(z, zb, a, w, valid, y, mask)


def reference_bce(p, y, mask):
    if p.dtype == torch.float32:
        return balanced_binary_cross_entropy(p, y, mask)
    # Legacy helper hard-casts targets to float32 and cannot run float64.
    # Independent float64 expression of its objective in the normal interval.
    q = p[mask].clamp(1e-6, 1-1e-6)
    t = y[mask].to(p.dtype)
    loss = -t * q.log() - (1-t) * torch.log1p(-q)
    fraction = t.mean()
    if fraction == 0 or fraction == 1:
        return loss.mean()
    weights = torch.where(t > 0.5, 1-fraction, fraction)
    return (loss*weights).sum()/(weights.sum()+1e-6)


def numerical_tests(device):
    results = []
    torch.manual_seed(7)
    for dtype in (torch.float32, torch.float64):
        atol, rtol = (1e-6, 1e-4) if dtype == torch.float32 else (1e-10, 1e-7)
        for batch in (1, 2):
            for label in ('mixed', 'positive', 'negative'):
                za = (torch.rand(batch, 3, 7, device=device, dtype=dtype)*8-4).requires_grad_()
                zb = (torch.rand_like(za)*8-4).requires_grad_()
                a = torch.rand_like(za) > 0.5
                w = torch.where(a, torch.zeros_like(za), torch.rand_like(za)*0.5+0.5)
                y = (torch.rand_like(za)>0.5) if label == 'mixed' else torch.full_like(a, label=='positive')
                mask = torch.rand_like(za)>0.2
                p = a.to(dtype)*za.sigmoid() + w*zb.sigmoid()
                old = reference_bce(p,y,mask)
                new = stable(za,w,y,a,mask,zb)
                go = torch.autograd.grad(old,(za,zb),retain_graph=True)
                gn = torch.autograd.grad(new,(za,zb))
                assert torch.allclose(new,old,atol=atol,rtol=rtol),(new,old)
                for x,t in zip(gn,go):
                    assert torch.allclose(x,t,atol=atol,rtol=rtol)
                    assert (x[~mask] == 0).all()
                results.append({'test':'normal_equivalence','dtype':str(dtype),'batch':batch,
                                'labels':label,'loss_abs_diff':float(abs(old-new)),
                                'gradient_max_abs_diff':max(float((x-t).abs().max()) for x,t in zip(gn,go))})
        for region in ('a','b'):
            for weight in (1.,0.5,1e-4) if region=='b' else (1.,):
                for value in (-120.,-80.,-20.,20.,80.,120.):
                    for label in (0.,1.):
                        z=torch.full((1,1,1),value,device=device,dtype=dtype,requires_grad=True)
                        y=torch.full_like(z,label);a=torch.full_like(z,region=='a',dtype=torch.bool)
                        w=torch.full_like(z,0. if region=='a' else weight)
                        loss=stable(z,w,y,a);g,=torch.autograd.grad(loss,z)
                        assert torch.isfinite(loss) and torch.isfinite(g).all() and loss >= 0
                        # Correct predictions may have gradients rounded to zero.
                        assert (g<=0).all() if label else (g>=0).all()
                        if (label and value<0) or (not label and value>0 and weight==1):
                            assert g.abs().sum()>0,(value,label,weight,g)
                        ref=(-math.log(weight)+max(-value,0)+math.log1p(math.exp(-abs(value)))) if label else (
                            math.log1p(math.exp(value)) if weight==1 else
                            -math.log1p(-weight/(1+math.exp(-value))))
                        assert abs(float(loss)-ref) <= atol+rtol*abs(ref),(float(loss),ref)
                        if label and value<0:
                            old=reference_bce(weight*z.sigmoid(),y,torch.ones_like(a))
                            oldg,=torch.autograd.grad(old,z)
                            assert oldg==0
                        results.append({'test':'extreme','dtype':str(dtype),'region':region,
                                        'weight':weight,'logit':value,'label':label,
                                        'loss':float(loss),'gradient':float(g)})
    # A scaled negative target at p~0 must not become negative through cancellation.
    z = torch.full((1,1,10000), -50., device=device, requires_grad=True)
    w = torch.linspace(.001,.97,10000,device=device).view_as(z)
    loss = stable(z,w,torch.zeros_like(z))
    assert loss > 0 and torch.allclose(loss, (w.double()*torch.sigmoid(z.double())).mean().float(), rtol=1e-5, atol=0)
    results.append({'test':'nonnegative_tiny_probability','loss':float(loss)})
    # Empty mask must not inspect masked invalid logits or take log(0).
    z=torch.full((2,2,3),float('nan'),device=device,requires_grad=True)
    loss=stable(z,torch.zeros_like(z),torch.ones_like(z),mask=torch.zeros_like(z,dtype=torch.bool))
    loss.backward();assert float(loss)==0 and (z.grad==0).all()
    z=torch.zeros((1,2,2),device=device,requires_grad=True)
    loss=stable(z,torch.zeros_like(z),torch.zeros_like(z));loss.backward()
    assert float(loss)==0 and (z.grad==0).all()
    for kwargs in ({'w':torch.zeros_like(z)}, {'w':torch.ones_like(z),'a':torch.ones_like(z,dtype=torch.bool)}):
        try:stable(z,kwargs['w'],torch.ones_like(z),a=kwargs.get('a'))
        except ValueError:pass
        else:raise AssertionError('unreachable target or overlapping regions accepted')
    z=torch.randn(2,2,3,dtype=torch.float64,device=device,requires_grad=True)
    w=torch.full_like(z,0.4);y=torch.rand_like(z)>0.5
    assert torch.autograd.gradcheck(lambda x:stable(x,w,y),(z,),eps=1e-6,atol=1e-5,rtol=1e-4)
    results.append({'test':'empty_mask_structural_zero_overlap_and_gradcheck','passed':True})
    return results


def production_test():
    from gradient_audit import main
    from test_residual_audit_batches import SyntheticDTU
    with tempfile.TemporaryDirectory() as directory:
        with patch('gradient_audit.find_dataset_def',return_value=SyntheticDTU):
            main(['--trainpath',directory,'--trainlist','lists/dtu/train.txt',
                  '--output_json',os.path.join(directory,'bce.json'),'--audit_epoch','12',
                  '--device','cpu','--batch_size','2','--ndepths','8,8,8','--gate_bce_ab'])
        comp=json.load(open(os.path.join(directory,'bce_batch0000_relu_bce_comparison.json')))
        assert len(comp['case_forward_invariance'])==15
        assert all(v==0 for row in comp['case_forward_invariance'].values() for v in row.values())
        for mode in ('legacy_clamp','stable_logprob'):
            d=json.load(open(os.path.join(directory,'bce_batch0000_relu_'+mode+'.json')))
            assert d['metadata']['actual_gate_bce_mode']==mode
            for c in d['cases']:
                assert c['gradient_summary']['all_finite']
                if c['gradient_summary']['sger_exclusive']:
                    assert c['gradient_summary']['backbone_grad_norm']==0
                if c['name']=='gate_benefit_only':
                    assert c['parameter_gradients']['sger_residual_body']['l2']==0
                    assert c['parameter_gradients']['sger_residual_output']['l2']==0
                b=c['loss_extras']['residual_bound_checks']
                assert b['proposal_all_finite'] and b['effective_all_finite']
                assert b['max_abs_proposal']<=b['proposal_bound']+1e-6
                assert b['max_abs_effective_residual']<=b['effective_bound']+1e-6


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu')
    parser.add_argument('--output_json');parser.add_argument('--skip_production',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    rows=numerical_tests(args.device)
    if not args.skip_production:production_test()
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)),exist_ok=True)
        with open(args.output_json,'w') as f:
            json.dump({'device':args.device,'torch':torch.__version__,'tests':rows,
                       'production_entrypoint_passed':not args.skip_production},f,indent=2)
    print('Gate BCE tests: PASS')
