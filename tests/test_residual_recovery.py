"""CPU-only production model/loss integration; no DTU/checkpoint experiment."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import experiment15_residual_recovery as recovery
import experiment15_d_frozen_probe as probe
from gradient_audit import build_parser, _build_optimizer
from test_residual_audit_batches import SyntheticDTU


class RecoveryDataset(SyntheticDTU):
    metas = [('synthetic', i) for i in range(10)]

    def __len__(self):
        return 10


class RecoveryTests(unittest.TestCase):
    def test_activation_capture_and_gradient_recovery(self):
        from test_activation_ab import test_last_activation_restores_body_gradient
        test_last_activation_restores_body_gradient()
        model = torch.nn.Sequential(recovery.ResidualDepthHead(2, hidden_channels=4))
        with torch.no_grad():
            model[0].body[4].weight.zero_()
            model[0].body[4].bias.fill_(-1)
        with recovery.capture_compact(model) as stats:
            model(torch.ones(2, 2, 8, 8))
        self.assertEqual(stats['0/final_activation_input']['mean'], -1)
        self.assertEqual(stats['0/final_activation_output']['zero_ratio'], 1)
        self.assertFalse(model[0].body[5]._forward_hooks)
        self.assertFalse(model[0].body[5]._forward_pre_hooks)

    def test_actual_cases_use_epoch_schedule(self):
        args = build_parser().parse_args(['--trainpath', '.', '--trainlist', 'x',
                                          '--audit_epoch', '9', '--output_json', 'x'])
        cases = {c.name: c for c in recovery.audit_cases(args)}
        self.assertEqual(cases['residual_target_only'].loss_weights['residual_target_loss_weight'], 1)
        self.assertEqual(cases['residual_target_only_actual'].loss_weights['residual_target_loss_weight'], 0.05)
        self.assertEqual(cases['refined_only_actual'].sger_scale, 0.75)
        self.assertEqual(cases['refined_only'].sger_scale, 1)

    def test_production_audit_and_probe(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = build_parser().parse_args(['--trainpath', '.', '--trainlist', 'x',
                       '--audit_epoch', '12', '--output_json', 'x', '--batch_size', '2', '--ndepths', '8,8,8'])
            torch.manual_seed(1)
            model = recovery.build_model(args)
            # Controlled dead final ReLU, but nonzero output weights.
            with torch.no_grad():
                head = model.sger_blocks[0].residual
                head.body[4].weight.fill_(-0.02)
                head.body[4].bias.fill_(-0.1)
                head.output.weight.fill_(0.02)
            checkpoint = root/'input.ckpt'
            optimizer = _build_optimizer(model, args)
            torch.save({'epoch': 12, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, str(checkpoint))
            args.loadckpt = str(checkpoint)
            source = {'arguments': vars(args), 'checkpoint_sha256': recovery.sha(checkpoint),
                      'train_indices': [0, 1, 2, 3], 'train_metas': RecoveryDataset.metas[:4],
                      'val_indices': list(range(6)), 'val_metas': RecoveryDataset.metas[:6], 'val_list': 'x'}
            manifest = root/'samples.json'
            manifest.write_text(json.dumps(source))
            cli = argparse.Namespace(output_dir=str(root/'audit'), probe_manifest=str(manifest),
                                     epoch=12, steps=2, strict_determinism=False)
            experiment = {'source': source, 'steps': 2, 'epoch': 12, 'runs': recovery.RUNS,
                          'configure_model': recovery.configure_model, 'forward': recovery.proposal_forward,
                          'statistics': recovery.proposal_statistics, 'gradients': recovery.residual_parameter_gradients,
                          'source_sha256': {}, 'metadata': {}}
            with patch.object(torch.nn.Module, 'cuda', lambda self: self), \
                 patch.object(torch.cuda, 'is_available', return_value=True), \
                 patch.object(torch.cuda, 'get_device_name', return_value='CPU test'), \
                 patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False), \
                 patch.object(recovery, 'find_dataset_def', return_value=RecoveryDataset), \
                 patch.object(probe, 'find_dataset_def', return_value=RecoveryDataset), \
                 patch.object(recovery, '_move_to_device', side_effect=lambda sample, device: sample), \
                 patch.object(probe, '_move_to_device', side_effect=lambda sample, device: sample), \
                 patch.object(probe, 'verify_prior_audits'), \
                 patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('Audit must not step')):
                with patch('builtins.print'):
                    recovery.run_audit(cli, experiment)
            summary = json.loads((root/'audit/summary.json').read_text())
            self.assertTrue(summary['paired_checks_passed'])
            self.assertTrue(all(v['exact'] for k, v in summary['comparisons'].items() if k.startswith('relu_repeat/')))
            report = json.loads((root/'audit/leaky_relu/probe0.json').read_text())
            self.assertEqual(len(report['cases']), 5)
            residual = next(c for c in report['cases'] if c['name'] == 'residual_target_only')
            self.assertGreater(residual['residual_parameter_gradients']['sger_blocks.0']['body.4.weight']['l2'], 0)
            self.assertEqual(recovery.sha(checkpoint), source['checkpoint_sha256'])
            cli.output_dir = str(root/'probe')
            with patch.object(torch.nn.Module, 'cuda', lambda self: self), \
                 patch.object(torch.cuda, 'is_available', return_value=True), \
                 patch.object(torch.cuda, 'get_device_name', return_value='CPU test'), \
                 patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False), \
                 patch.object(probe, 'find_dataset_def', return_value=RecoveryDataset), \
                 patch.object(probe, '_move_to_device', side_effect=lambda sample, device: sample), \
                 patch.object(probe, 'verify_prior_audits'), patch('builtins.print'):
                probe.run(cli, experiment)
            summary = json.loads((root/'probe/summary.json').read_text())
            self.assertTrue(summary['checks_passed'])
            self.assertEqual(summary['completed_steps'], {label: 2 for label, _ in recovery.RUNS})
            self.assertEqual(set(summary['validation']), {'0', '1', '2'})
            self.assertTrue(all(r['exact'] for r in summary['comparisons']['relu_repeat'].values()))
            self.assertTrue(all(r['first_protected_difference'] is None for r in summary['comparisons']['leaky_relu'].values()))
            self.assertIn('proposal_minus_bias_abs_mean', summary['validation']['2']['leaky_relu'])
            for label, _ in recovery.RUNS:
                ckpt = torch.load(str(root/'probe'/label/'step_0002.ckpt'), map_location='cpu')
                self.assertEqual(ckpt['epoch'], 12)
                self.assertEqual(ckpt['gate_bce_mode'], 'legacy_clamp')
                expected = 'leaky_relu' if label == 'leaky_relu' else 'relu'
                self.assertEqual(ckpt['arguments']['sger_residual_final_activation'], expected)
            self.assertEqual(recovery.sha(checkpoint), source['checkpoint_sha256'])


if __name__ == '__main__':
    unittest.main()
