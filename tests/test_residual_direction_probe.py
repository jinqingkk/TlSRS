"""CPU checks for the residual-only signed-direction probe."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

import experiment15_residual_direction_probe as probe
import test_momentum_diagnose as fixture_module
from test_residual_audit_batches import SyntheticDTU


class ProbeDataset(SyntheticDTU):
    metas = [('scan%d' % (index // 2), index) for index in range(60)]

    def __len__(self):
        return 60

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        # Production raw depth lies in [1,10]. Force both target directions.
        sample['depth']['stage3'].fill_(.1)
        sample['depth']['stage3'][:8].fill_(20.)
        return sample


class DirectionProbeTests(unittest.TestCase):
    def test_direction_formula_gradient_and_detach(self):
        logit = torch.tensor(
            [[[-2., -.5, 1., 2., 0.]]], requires_grad=True)
        raw = torch.full_like(logit, 10., requires_grad=True)
        gt = torch.tensor(
            [[[11., 9., 12., 8., 10.]]], requires_grad=True)
        mask = torch.ones_like(logit)
        loss, target, positive, negative, info = probe.direction_loss(
            logit, raw, gt, 4., mask, .25)
        selected = torch.tensor([True, True, True, True, False])
        truth = torch.tensor([1., 0., 1., 0.])
        expected = F.binary_cross_entropy_with_logits(
            logit.reshape(-1)[selected], truth)
        self.assertTrue(torch.equal(target, torch.tensor(
            [[[1., -1., 1., -1., 0.]]])))
        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(info['positive_count'], 2)
        self.assertEqual(info['negative_count'], 2)
        self.assertEqual(info['zero_count'], 1)
        gradients = torch.autograd.grad(
            loss, (logit, raw, gt), retain_graph=True, allow_unused=True)
        self.assertIsNone(gradients[1])
        self.assertIsNone(gradients[2])
        report = probe.direction_gradient_check(
            loss, logit, positive, negative)
        self.assertEqual(report['positive_correct_gradient_count'], 2)
        self.assertEqual(report['negative_correct_gradient_count'], 2)
        self.assertEqual(report['positive_wrong_gradient_count'], 0)
        self.assertEqual(report['negative_wrong_gradient_count'], 0)
        self.assertEqual(report['excluded_zero_gradient_nonzero_count'], 0)
        probe.require_direction_gradient({
            'direction_counts': info, 'direction_gradients': report})

    def test_extreme_logits_empty_and_wrong_gradient_rejected(self):
        logit = torch.tensor([[[-100., 100.]]], requires_grad=True)
        raw = torch.full_like(logit, 10.)
        gt = torch.tensor([[[11., 9.]]])
        loss, target, positive, negative, info = probe.direction_loss(
            logit, raw, gt, 4., torch.ones_like(logit), .25)
        gradient = torch.autograd.grad(loss, logit, retain_graph=True)[0]
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertLess(float(gradient[positive]), 0.)
        self.assertGreater(float(gradient[negative]), 0.)
        report = probe.direction_gradient_check(
            loss, logit, positive, negative)
        report['negative_correct_gradient_count'] = 0
        with self.assertRaisesRegex(RuntimeError, 'gradient'):
            probe.require_direction_gradient({
                'direction_counts': info, 'direction_gradients': report})
        with self.assertRaisesRegex(ValueError, 'nonzero'):
            probe.direction_loss(
                logit, raw, raw, 4., torch.ones_like(logit), .25)
        with self.assertRaisesRegex(ValueError, 'nonzero'):
            probe.direction_loss(
                logit, raw, gt, 4., torch.zeros_like(logit), .25)

    def test_two_step_production_run(self):
        self.check_production_run(2)

    def test_25_step_production_run(self):
        self.check_production_run(25)

    def check_production_run(self, steps):
        fixture = fixture_module.MomentumTests()
        fixture.setUp()
        diag = probe.diag
        with torch.no_grad():
            fixture.model.sger_blocks[0].residual.output.bias.fill_(.2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'source'
            checkpoint_path = root / 'stable/step_0025.ckpt'
            checkpoint_path.parent.mkdir(parents=True)
            source_arguments = vars(fixture.args).copy()
            checkpoint_arguments = dict(
                source_arguments, gate_bce_mode='stable_logprob',
                sger_residual_final_activation='leaky_relu')
            torch.save({
                'model': fixture.model.state_dict(),
                'optimizer': fixture.optimizer.state_dict(),
                'arguments': checkpoint_arguments,
                'epoch': 12,
                'local_step': 25,
                'source_probe_step': 1,
                'freeze_backbone': True,
                'gate_bce_mode': 'stable_logprob',
            }, str(checkpoint_path))
            diag.dump(
                root / 'stable/fingerprints/step25_update.json',
                diag.fingerprints(
                    diag.model_snapshot(fixture.model, fixture.optimizer)))
            plan = {
                'train_indices': list(range(50)),
                'train_metas': ProbeDataset.metas[:50],
                'val_indices': list(range(6)),
                'val_metas': ProbeDataset.metas[:6],
                'val_list': 'x',
            }
            diag.dump(root / 'manifest.json', {
                'status': 'completed',
                'source_sha256': {},
                'input_sha256': {},
                'arguments': source_arguments,
                'effective_case': diag.audit_cases(fixture.args)[0]._asdict(),
                'sample_plan': plan,
            })
            diag.dump(root / 'summary.json', {
                'status': 'completed',
                'checks_passed': True,
                'input_files_unchanged': True,
                'completed_steps': {
                    group: 25 for group in
                    ('legacy_reference', 'legacy_repeat', 'stable')},
            })
            output = Path(temp) / 'out'
            cli = argparse.Namespace(
                probe_dir=str(root), output_dir=str(output),
                preflight_only=False, device='cpu',
                strict_determinism=True, steps=steps)
            with patch.object(
                    diag, 'find_dataset_def', return_value=ProbeDataset), \
                    patch('builtins.print'):
                probe.run(cli)
            result = json.loads((output / 'summary.json').read_text())
            self.assertTrue(result['checks_passed'])
            self.assertTrue(result['input_files_unchanged'])
            self.assertEqual(
                result['completed_steps'],
                {group: steps for group in probe.RUNS})
            self.assertTrue(all(
                row['exact'] for row in
                result['comparisons']['direction_repeat'].values()))
            self.assertTrue(all(
                row['first_protected_difference'] is None for row in
                result['comparisons']['sign_bce'].values()))
            self.assertFalse(
                result['comparisons']['sign_bce'][
                    'step1_gradients']['exact'])
            expected_steps = {
                str(value) for value in (0, 1, 2, 5, 10, 25)
                if value <= steps
            }
            self.assertEqual(set(result['validation']), expected_steps)
            for group in probe.RUNS:
                directory = output / group
                initial = torch.load(
                    str(directory / 'step_0000.ckpt'), map_location='cpu')
                final = torch.load(
                    str(directory / ('step_%04d.ckpt' % steps)),
                    map_location='cpu')
                changed = {
                    name for name, value in initial['model'].items()
                    if not torch.equal(value, final['model'][name])
                }
                self.assertTrue(changed)
                self.assertTrue(all(
                    name.startswith(probe.RESIDUAL_PREFIX)
                    for name in changed))
                self.assertEqual(len(initial['trainable_parameters']), 8)
                validation = result['validation'][str(steps)][group]
                self.assertTrue(validation['state_restored'])
                self.assertEqual(
                    set(validation['signed_scenes']),
                    {'scan0', 'scan1', 'scan2'})
                self.assertEqual(
                    set(validation['selection_scenes']),
                    {'scan0', 'scan1', 'scan2'})
                start = result['validation']['0'][group]['quality']
                end = validation['quality']
                self.assertEqual(
                    start['raw_abs_error'], end['raw_abs_error'])
                self.assertEqual(
                    start['benefit_gate_mean'], end['benefit_gate_mean'])
                row = json.loads((
                    directory / 'train' /
                    ('step_%04d.json' % steps)).read_text())
                self.assertTrue(row['fixed_non_residual_state_unchanged'])
                self.assertTrue(row['measurement_state_restored'])
            first = json.loads((
                output / 'sign_bce/train/step_0001.json').read_text())
            probe.require_direction_gradient(first['before'])
            self.assertEqual(
                first['before']['effective_direction_loss_weight'],
                probe.DIRECTION_LOSS_WEIGHT)


if __name__ == '__main__':
    unittest.main()
