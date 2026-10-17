"""Production CPU synthetic 25-step integration, no real DTU training."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import experiment15_sign_balanced_probe as probe
from test_momentum_diagnose import MomentumTests
from test_residual_audit_batches import SyntheticDTU


class ProbeDataset(SyntheticDTU):
    metas = [('scan%d' % (i // 2), i) for i in range(60)]

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        # Raw depth lies in [1,10]; force both signs with unequal counts.
        sample['depth']['stage3'].fill_(.1)
        sample['depth']['stage3'][:8].fill_(20)
        return sample


class BalancedProbeTests(unittest.TestCase):
    def test_balanced_formula_and_gradient_boundary(self):
        proposal = torch.tensor([.1, .2, .3, .4, .5], requires_grad=True)
        target = torch.tensor([.5, .6, .7, -.2, 0.], requires_grad=True)
        valid = torch.ones(5, dtype=torch.bool)
        pixel, balanced, info = probe.target_losses(proposal, target, valid)
        errors = torch.nn.functional.smooth_l1_loss(proposal, target.detach(), reduction='none')
        expected = .4*(errors[:3].mean()+errors[3]) + .2*errors[4]
        self.assertTrue(torch.allclose(balanced, expected))
        self.assertEqual(info['positive_per_pixel_multiplier'], 2/3)
        self.assertEqual(info['negative_per_pixel_multiplier'], 2)
        other = torch.tensor(2., requires_grad=True)
        full = other**2 + .05*pixel
        adjusted = probe.replace_target_term(full, pixel, balanced, .05, True)
        gradients = torch.autograd.grad(adjusted, (proposal, other, target), retain_graph=True, allow_unused=True)
        expected_gradient = torch.autograd.grad(.05*expected, proposal, retain_graph=True)[0]
        self.assertTrue(torch.allclose(gradients[0], expected_gradient))
        self.assertEqual(float(gradients[1]), 4)
        self.assertIsNone(gradients[2])
        self.assertIs(probe.replace_target_term(full, pixel, balanced, .05, False), full)
        for t in (torch.ones(5), -torch.ones(5), torch.zeros(5)):
            a, b, counts = probe.target_losses(proposal, t, valid)
            self.assertIs(a, b)
            self.assertFalse(counts['balancing_active'])
        with self.assertRaises(ValueError):
            probe.target_losses(proposal, target, ~valid)

    def test_pooled_counts_and_empty_precision(self):
        rows = []
        for n, positives, true in [(10, 2, 1), (30, 8, 3)]:
            rows.append(dict(valid_pixel_count=n, improved_count=0, worsened_count=0,
                             target_positive_count=positives, predicted_positive_count=positives,
                             true_positive_count=true, gate_precision=true/positives,
                             gate_recall=true/positives, gate_on_improved_mean=None,
                             gate_on_worsened_mean=None, proposal_gain_mean=float(n)))
        result = probe.pool_metrics(rows)
        self.assertEqual(result['proposal_gain_mean'], 25)
        self.assertEqual(result['gate_precision'], .4)
        for r in rows:
            r['predicted_positive_count'] = 0
            r['true_positive_count'] = 0
        self.assertIsNone(probe.pool_metrics(rows)['gate_precision'])
        self.assertIsNone(probe.pool_metrics(rows)['gate_on_improved_mean'])

    def test_25_step_production_run(self):
        fixture = MomentumTests()
        fixture.setUp()
        diag = probe.diag
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_dir = root/'source'
            ckpt = source_dir/'leaky_relu/step_0001.ckpt'
            ckpt.parent.mkdir(parents=True)
            torch.save(dict(model=fixture.model.state_dict(), optimizer=fixture.optimizer.state_dict(),
                            arguments=vars(fixture.args), epoch=12, probe_step=1,
                            freeze_backbone=True, gate_bce_mode='legacy_clamp'), str(ckpt))
            diag.dump(source_dir/'leaky_relu/fingerprints/step1_update.json',
                      diag.fingerprints(diag.model_snapshot(fixture.model, fixture.optimizer)))
            samples = dict(arguments=vars(fixture.args), train_indices=list(range(52)),
                           train_metas=ProbeDataset.metas[:52], val_indices=list(range(6)),
                           val_metas=ProbeDataset.metas[:6], val_list='x')
            samples_path = root/'samples.json'
            diag.dump(samples_path, samples)
            source = dict(samples, status='completed', steps=2, source_sha256={},
                          train_indices=list(range(4)), train_metas=ProbeDataset.metas[:4],
                          effective_case=diag.audit_cases(fixture.args)[0]._asdict(),
                          experiment={'checkpoint_epoch': 12}, probe_manifest=str(samples_path),
                          probe_manifest_sha256=diag.sha(samples_path))
            diag.dump(source_dir/'manifest.json', source)
            diag.dump(source_dir/'summary.json', dict(status='completed', checks_passed=True))
            source_hash = diag.sha(ckpt)
            cli = argparse.Namespace(probe_dir=str(source_dir), output_dir=str(root/'out'),
                                     preflight_only=False, strict_determinism=False)
            with patch.object(torch.nn.Module, 'cuda', lambda self: self), \
                 patch.object(torch.cuda, 'is_available', return_value=True), \
                 patch.object(torch.cuda, 'get_device_name', return_value='CPU test'), \
                 patch.object(torch.cuda, 'get_rng_state_all', return_value=[]), \
                 patch.object(torch.cuda, 'set_rng_state_all'), \
                 patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False), \
                 patch.object(diag, '_move_to_device', side_effect=lambda sample, device: sample), \
                 patch.object(diag, 'find_dataset_def', return_value=ProbeDataset), \
                 patch.object(diag, 'clear_first_moment', wraps=diag.clear_first_moment) as reset, \
                 patch('builtins.print'):
                probe.run(cli)
            self.assertEqual(reset.call_count, 0)
            result = json.loads((root/'out/summary.json').read_text())
            self.assertTrue(result['checks_passed'])
            self.assertTrue(result['input_files_unchanged'])
            self.assertEqual(result['completed_steps'], {g: 25 for g in probe.RUNS})
            self.assertEqual(set(result['validation']), {'0', '1', '5', '10', '25'})
            self.assertTrue(all(c['exact'] for c in result['comparisons']['pixel_repeat'].values()))
            self.assertTrue(all(c['first_protected_difference'] is None
                                for c in result['comparisons']['sign_balanced'].values()))
            self.assertFalse(result['comparisons']['sign_balanced']['step1_gradients']['exact'])
            for label in probe.RUNS:
                directory = root/'out'/label
                first = json.loads((directory/'train/step_0001.json').read_text())
                last = json.loads((directory/'train/step_0025.json').read_text())
                self.assertEqual(first['indices'], [2, 3])
                self.assertTrue(first['before']['balance']['balancing_active'])
                self.assertEqual(last['indices'], [50, 51])
                self.assertTrue(last['frozen_since_initial']['passed'])
                initial = torch.load(str(directory/'step_0000.ckpt'), map_location='cpu')
                final = torch.load(str(directory/'step_0025.ckpt'), map_location='cpu')
                for name, value in initial['model'].items():
                    if 'num_batches_tracked' in name:
                        self.assertTrue(torch.equal(final['model'][name], value +
                                        (25 if name.startswith('normal_head.2.') else 0)))
                for key, state in initial['optimizer']['state'].items():
                    final_step = final['optimizer']['state'][key]['step']
                    self.assertIn(float(final_step - state['step']), (0, 25))
                val = result['validation']['25'][label]
                self.assertTrue(val['state_and_rng_restored'])
                self.assertEqual(set(val['scenes']), {'scan0', 'scan1', 'scan2'})
                self.assertEqual(set(val['signed']), {'positive', 'negative', 'zero'})
            second = json.loads((root/'out/sign_balanced/train/step_0002.json').read_text())
            self.assertTrue(any(r.get('old_exp_avg_l2', 0) > 0 for n, r in
                                second['directions']['parameters'].items() if '.residual.' in n))
            self.assertEqual(diag.sha(ckpt), source_hash)
            # A changed sample manifest cannot silently change the paired plan.
            samples_path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                probe.load_plan(source_dir)


if __name__ == '__main__':
    unittest.main()
