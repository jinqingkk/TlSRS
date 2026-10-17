import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

import experiment15_gate_normalization_probe as probe
import test_momentum_diagnose as fixture_module
from test_residual_audit_batches import SyntheticDTU


class ProbeDataset(SyntheticDTU):
    metas = [('scan%d' % (i//2), i) for i in range(60)]

    def __len__(self):
        return 60

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample['depth']['stage3'].fill_(.1)
        sample['depth']['stage3'][:8].fill_(20.)
        return sample


class NormalizationTests(unittest.TestCase):
    def test_pixel_formula_gradient_and_detached_labels(self):
        z = torch.tensor([[[-2., -.5, 1., 2.]]], dtype=torch.float64, requires_grad=True)
        raw = torch.ones_like(z, requires_grad=True)*10
        proposal = torch.ones_like(z, requires_grad=True)
        gt = torch.tensor([[[11., 9., 9., 9.]]], dtype=torch.float64, requires_grad=True)
        stage = dict(depth_raw=raw, raw_depth_residual=proposal, depth_interval=1.,
                     gate_logits_a=z, gate_logits_b=z*2, region_a=torch.ones_like(z).bool(),
                     region_b_weight=torch.zeros_like(z), gate_valid_mask=torch.ones_like(z).bool())
        balanced, pixel, counts = probe.benefit_losses(stage, gt, torch.ones_like(z), .05)
        target = torch.tensor([[[1., 0., 0., 0.]]], dtype=torch.float64)
        expected = F.binary_cross_entropy_with_logits(z, target)
        self.assertTrue(torch.allclose(pixel, expected, atol=1e-12, rtol=1e-12))
        pg = torch.autograd.grad(pixel, z, retain_graph=True)[0]
        eg = torch.autograd.grad(expected, z, retain_graph=True)[0]
        bg = torch.autograd.grad(balanced, z, retain_graph=True)[0]
        self.assertTrue(torch.allclose(pg, eg, atol=1e-12, rtol=1e-12))
        self.assertFalse(torch.allclose(pg, bg))
        self.assertEqual(counts['positive_count'], 1)
        self.assertEqual(counts['balanced_positive_mass'], .5)
        self.assertEqual(counts['pixel_positive_mass'], .25)
        self.assertEqual(torch.autograd.grad(pixel, (raw, proposal, gt), allow_unused=True), (None,None,None))
        for value in (11., 9.):
            single_gt = torch.full_like(gt, value)
            b, p, _ = probe.benefit_losses(stage, single_gt, torch.ones_like(z), .05)
            self.assertTrue(torch.equal(b, p))
        with self.assertRaisesRegex(ValueError, 'Empty'):
            probe.benefit_losses(stage, gt, torch.zeros_like(z), .05)

    def test_scaled_gate_extremes(self):
        z = torch.tensor([[[-100., 100., -100., 100.]]], requires_grad=True)
        raw = torch.ones_like(z)*10
        stage = dict(depth_raw=raw, raw_depth_residual=torch.ones_like(z), depth_interval=1.,
            gate_logits_a=z, gate_logits_b=z, region_a=torch.zeros_like(z).bool(),
            region_b_weight=torch.tensor([[[.5,.5,0.,1.]]]), gate_valid_mask=torch.ones_like(z).bool())
        gt = torch.tensor([[[11.,9.,9.,9.]]])
        _, loss, _ = probe.benefit_losses(stage, gt, torch.ones_like(z), .05)
        grad = torch.autograd.grad(loss,z)[0]
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(grad).all())
        self.assertLess(float(grad[0,0,0]),0)
        self.assertGreater(float(grad[0,0,3]),0)
        gt[0,0,2]=11.
        with self.assertRaisesRegex(ValueError,'unreachable'):
            probe.benefit_losses(stage,gt,torch.ones_like(z),.05)

    def test_two_steps(self):
        self.check_run(2)

    def test_25_steps(self):
        self.check_run(25)

    def check_run(self, steps):
        fixture=fixture_module.MomentumTests()
        fixture.setUp()
        diag=probe.diag
        with torch.no_grad():
            fixture.model.sger_blocks[0].residual.output.bias.fill_(1.)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'source'
            path=root/'stable/step_0025.ckpt'
            path.parent.mkdir(parents=True)
            args=dict(vars(fixture.args),gate_bce_mode='stable_logprob')
            torch.save(dict(model=fixture.model.state_dict(),optimizer=fixture.optimizer.state_dict(),
                arguments=args,epoch=12,local_step=25,source_probe_step=1,freeze_backbone=True,
                gate_bce_mode='stable_logprob'),str(path))
            diag.dump(root/'stable/fingerprints/step25_update.json',diag.fingerprints(diag.model_snapshot(fixture.model,fixture.optimizer)))
            plan=dict(train_indices=list(range(50)),train_metas=ProbeDataset.metas[:50],
                val_indices=list(range(6)),val_metas=ProbeDataset.metas[:6],val_list='x')
            diag.dump(root/'manifest.json',dict(status='completed',source_sha256={},input_sha256={},
                arguments=vars(fixture.args),effective_case=diag.audit_cases(fixture.args)[0]._asdict(),sample_plan=plan))
            diag.dump(root/'summary.json',dict(status='completed',checks_passed=True,input_files_unchanged=True,
                completed_steps={g:25 for g in ('legacy_reference','legacy_repeat','stable')}))
            cli=argparse.Namespace(probe_dir=str(root),output_dir=str(Path(temp)/'out'),
                preflight_only=False,device='cpu',strict_determinism=True,steps=steps)
            with patch.object(diag,'find_dataset_def',return_value=ProbeDataset),patch('builtins.print'):
                probe.run(cli)
            result=json.loads((Path(cli.output_dir)/'summary.json').read_text())
            self.assertTrue(result['checks_passed'])
            self.assertTrue(result['input_files_unchanged'])
            self.assertEqual(result['completed_steps'],{g:steps for g in probe.RUNS})
            self.assertTrue(all(c['exact'] for c in result['comparisons']['balanced_repeat'].values()))
            self.assertTrue(all(c['first_protected_difference'] is None for c in result['comparisons']['pixel_mean'].values()))
            self.assertFalse(result['comparisons']['pixel_mean']['step1_gradients']['exact'])
            for group in probe.RUNS:
                directory=Path(cli.output_dir)/group
                initial=torch.load(str(directory/'step_0000.ckpt'),map_location='cpu')
                final=torch.load(str(directory/('step_%04d.ckpt'%steps)),map_location='cpu')
                for name,value in initial['model'].items():
                    if not name.startswith(probe.GATE_PREFIX):
                        self.assertTrue(torch.equal(value,final['model'][name]),name)
                self.assertEqual(len(initial['trainable_parameters']),8)
                val=result['validation'][str(steps)][group]
                self.assertTrue(val['state_restored'])
                self.assertEqual(set(val['scenes']),{'scan0','scan1','scan2'})
                start=result['validation']['0'][group]['pooled']['all/all/all/all']['means']
                end=val['pooled']['all/all/all/all']['means']
                self.assertEqual(start['raw_error'],end['raw_error'])
                self.assertEqual(start['oracle_gain'],end['oracle_gain'])
                self.assertAlmostEqual(val['quality']['raw_abs_error'], end['raw_error'])
                self.assertAlmostEqual(val['quality']['refined_abs_error'], end['actual_error'])
                self.assertAlmostEqual(val['quality']['raw_to_refined_error_delta'], end['actual_gain'])
                row=json.loads((directory/'train'/('step_%04d.json'%steps)).read_text())
                self.assertTrue(row['fixed_proposal_state_unchanged'])


if __name__=='__main__':
    unittest.main()
