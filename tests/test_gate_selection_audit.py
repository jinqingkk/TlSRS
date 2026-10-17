import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import experiment15_gate_selection_audit as audit
import test_momentum_diagnose as fixture_module
from test_residual_audit_batches import SyntheticDTU


class AuditTests(unittest.TestCase):
    def test_oracle_and_cross_tabs(self):
        raw = torch.tensor([[[10., 10., 10., 10.]]], requires_grad=True)
        proposal = torch.tensor([[[2., 2., 0., 4.]]])
        gate = torch.tensor([[[0., .5, 0., .5]]])
        gt = torch.tensor([[[12., 8., 10., 11.]]])
        stage = dict(depth_raw=raw, raw_depth_residual=proposal, benefit_gate=gate,
                     depth_refined=raw+gate*proposal, depth_interval=1.,
                     region_a=torch.tensor([[[True, False, True, False]]]),
                     region_b_weight=torch.ones_like(raw), gate_logits_a=torch.zeros_like(raw),
                     gate_logits_b=torch.zeros_like(raw))
        rows = audit.selection_statistics(stage, gt, torch.ones_like(raw), .05)
        pooled = audit.pool([rows, rows])
        all_pixels = pooled['all/all/all/all']
        self.assertEqual(all_pixels['count'], 8)
        self.assertEqual(all_pixels['means']['raw_error'], 1.25)
        self.assertEqual(all_pixels['means']['actual_error'], 1.5)
        self.assertEqual(all_pixels['means']['oracle_error'], .75)
        self.assertEqual(pooled['all/beneficial/all/all']['count'], 2)
        self.assertEqual(pooled['negative/harmful/b/at_least_05']['count'], 2)
        self.assertIsNone(pooled['negative/beneficial/all/all']['means']['oracle_gain'])
        self.assertIsNone(raw.grad)
        self.assertEqual(sum(pooled['all/all/all/'+k]['count'] for k in
            ('below_1e_6', 'from_1e_6_to_001', 'from_001_to_05', 'at_least_05')), 8)
        with self.assertRaisesRegex(ValueError, 'Empty'):
            audit.selection_statistics(stage, gt, torch.zeros_like(raw), .05)
        stage['depth_refined'] = raw + proposal
        with self.assertRaisesRegex(ValueError, 'beta=1'):
            audit.selection_statistics(stage, gt, torch.ones_like(raw), .05)

    def test_margin_and_invalid_mask(self):
        raw = torch.tensor([[[1., 1., float('nan')]]])
        proposal = torch.tensor([[[.125, .5, float('nan')]]])
        gate = torch.zeros_like(raw)
        stage = dict(depth_raw=raw, raw_depth_residual=proposal, benefit_gate=gate,
                     depth_refined=raw+gate*proposal, depth_interval=1.,
                     region_a=torch.ones_like(raw).bool(), region_b_weight=gate,
                     gate_logits_a=gate, gate_logits_b=gate)
        gt = torch.tensor([[[2., 2., float('nan')]]])
        result = audit.pool([audit.selection_statistics(stage, gt, torch.tensor([[[1.,1.,0.]]]), .125)])
        self.assertEqual(result['all/beneficial/all/all']['count'], 2)
        self.assertEqual(result['all/above_margin/all/all']['count'], 1)
        self.assertEqual(result['all/small_benefit/all/all']['count'], 1)
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            audit.selection_statistics(stage, gt, torch.ones_like(raw), .125)

    def test_production_read_only_runner(self):
        fixture = fixture_module.MomentumTests()
        fixture.setUp()
        diag = audit.diag
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'source'
            for group in audit.GROUPS:
                mode = 'stable_logprob' if group == 'stable' else 'legacy_clamp'
                args = dict(vars(fixture.args), gate_bce_mode=mode)
                for step in audit.STEPS:
                    path = root/group/('step_%04d.ckpt' % step)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(dict(model=fixture.model.state_dict(), optimizer=fixture.optimizer.state_dict(),
                        arguments=args, epoch=12, local_step=step, source_probe_step=1,
                        freeze_backbone=True, gate_bce_mode=mode), str(path))
                    phase = 'initial' if step == 0 else 'step%d_update' % step
                    diag.dump(root/group/'fingerprints'/(phase+'.json'),
                              diag.fingerprints(diag.model_snapshot(fixture.model, fixture.optimizer)))
            indices = [0,1,2,3,4,5]*9
            plan = dict(train_indices=indices[:50], train_metas=[SyntheticDTU.metas[i] for i in indices[:50]],
                        val_indices=list(range(6)), val_metas=SyntheticDTU.metas, val_list='x')
            diag.dump(root/'manifest.json', dict(status='completed', source_sha256={}, input_sha256={},
                arguments=vars(fixture.args), sample_plan=plan, effective_case=diag.audit_cases(fixture.args)[0]._asdict()))
            diag.dump(root/'summary.json', dict(status='completed', checks_passed=True, input_files_unchanged=True,
                completed_steps={g:25 for g in ('legacy_reference','legacy_repeat','stable')}))
            cli = argparse.Namespace(probe_dir=str(root), output_dir=str(Path(temp)/'out'),
                preflight_only=False, device='cpu', strict_determinism=False)
            with patch.object(diag, 'find_dataset_def', return_value=SyntheticDTU), \
                 patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('Read-only violation')), \
                 patch('builtins.print'):
                audit.run(cli)
            summary = json.loads((Path(cli.output_dir)/'summary.json').read_text())
            self.assertTrue(summary['checks_passed'])
            self.assertTrue(summary['input_files_unchanged'])
            for group in audit.GROUPS:
                for step in audit.STEPS:
                    row = summary['results'][group][str(step)]
                    self.assertEqual(set(row), {'train','val'})
                    self.assertEqual(row['val']['pooled']['all/all/all/all']['count'], 6144)
                    self.assertEqual(set(row['val']['scenes']), {'synthetic'})
            damaged = root/'stable/step_0025.ckpt'
            ckpt = torch.load(str(damaged), map_location='cpu')
            ckpt['model']['sger_blocks.0.residual.output.bias'].add_(1)
            torch.save(ckpt, str(damaged))
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                audit.load_model(root, 'stable', 25, audit.load_source(root), torch.device('cpu'))


if __name__ == '__main__':
    unittest.main()
