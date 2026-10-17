"""CPU tests; never load DTU or a production checkpoint."""
import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import experiment15_d_frozen_probe as probe


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, *args):
        self.metas = list(range(6))

    def __len__(self):
        return 6

    def __getitem__(self, index):
        return {'imgs': torch.ones(2, 2) * (index + 1), 'proj_matrices': {},
                'depth_values': torch.ones(1), 'depth': {'stage3': torch.ones(2, 2)},
                'mask': {'stage3': torch.ones(2, 2)}}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_stage = 3
        self.backbone = torch.nn.Linear(2, 2)
        self.sger_blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])

    def cuda(self):
        return self

    def set_sger_residual_scale(self, scale):
        pass

    def forward(self, imgs, *args):
        raw = self.backbone(imgs)
        residual = self.sger_blocks[0](raw) * 0.01
        return {'stage3': {'depth': raw + residual, 'depth_raw': raw,
                'depth_refined': raw + residual, 'raw_depth_residual': residual,
                'depth_residual': residual, 'benefit_gate': torch.sigmoid(residual),
                'depth_interval': torch.ones_like(raw), 'photometric_confidence': torch.ones_like(raw)}}


class FrozenProbeTests(unittest.TestCase):
    def test_pooled_counts_and_empty_regions(self):
        stage = {'depth_raw': torch.tensor([2., 2.]), 'depth_refined': torch.tensor([1., 3.]),
                 'raw_depth_residual': torch.tensor([-1., 1.]), 'depth_residual': torch.tensor([-1., 1.]),
                 'benefit_gate': torch.tensor([0.8, 0.2]), 'depth_interval': torch.ones(2)}
        row = probe.stage_statistics(stage, torch.ones(2), torch.ones(2), 0.05)
        result = probe.aggregate([row, row])
        self.assertEqual(result['valid_pixel_count'], 4)
        self.assertEqual(result['gate_precision'], 1)
        self.assertEqual(result['gate_recall'], 1)
        self.assertEqual(result['raw_to_refined_error_delta'], 0)
        empty = probe.stage_statistics(stage, torch.ones(2), torch.zeros(2), 0.05)
        self.assertIsNone(probe.aggregate([empty])['gate_precision'])
        self.assertIsNone(probe.aggregate([empty])['raw_abs_error'])

    def test_hash_rejects_nonfinite_and_distinguishes_none(self):
        self.assertNotEqual(probe.fingerprints({'x': None}), probe.fingerprints({'x': torch.zeros(1)}))
        with self.assertRaises(RuntimeError):
            probe.fingerprints({'x': torch.tensor([float('nan')])})
        self.assertIsNotNone(probe.compare_fingerprints({'backbone/x': 'a'}, {})['first_protected_difference'])

    def test_three_arm_runner_and_failure_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.manual_seed(1)
            model = TinyModel()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            model(torch.ones(2, 2))['stage3']['depth'].square().mean().backward()
            optimizer.step()
            ckpt = root / 'input.ckpt'
            torch.save({'epoch': 15, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, str(ckpt))
            source = {'arguments': {'loadckpt': str(ckpt), 'batch_size': 2,
                      'sger_residual_final_activation': 'relu', 'dataset': 'tiny', 'trainpath': '',
                      'trainlist': '', 'nviews': 3, 'numdepth': 2, 'interval_scale': 1,
                      'sger_warmup_start_epoch': 7, 'sger_warmup_end_epoch': 10, 'seed': 1,
                      'sger_max_residual_ratio': 0.25, 'benefit_margin_ratio': 0.05}, 'checkpoint_sha256': probe.sha(ckpt),
                      'train_indices': [0, 1, 2, 3], 'train_metas': [0, 1, 2, 3],
                      'val_indices': list(range(6)), 'val_metas': list(range(6)), 'val_list': ''}
            manifest = root / 'source.json'
            manifest.write_text(json.dumps(source))
            def restore(m, args, checkpoint):
                opt = torch.optim.Adam(m.parameters(), lr=0.01)
                opt.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
                return opt, []
            def loss(outputs, *args, **kwargs):
                value = outputs['stage3']['depth_refined'].square().mean()
                if kwargs.get('gate_bce_mode') == 'stable_logprob':
                    value = value * 1.01
                return value, {'tiny_loss': value.detach()}
            cli = argparse.Namespace(probe_manifest=str(manifest), output_dir=str(root/'run'), strict_determinism=False)
            with patch.multiple(probe, STEPS=2, SAVE_STEPS=(0, 1, 2)), \
                 patch.object(probe.torch.cuda, 'is_available', return_value=True), \
                 patch.object(probe.torch.cuda, 'get_device_name', return_value='CPU test'), \
                 patch.object(probe.torch.cuda, 'is_current_stream_capturing', return_value=False), \
                 patch.object(probe, 'verify_prior_audits'), \
                 patch.object(probe, 'build_model', side_effect=lambda args: TinyModel()), \
                 patch.object(probe, 'restore_optimizer', side_effect=restore), \
                 patch.object(probe, 'find_dataset_def', return_value=TinyDataset), \
                 patch.object(probe, '_move_to_device', side_effect=lambda sample, device: sample), \
                 patch.object(probe, '_base_loss_weights', return_value={}), \
                 patch.object(probe, '_loss_kwargs', side_effect=lambda args, *rest: {'gate_bce_mode': args.gate_bce_mode}), \
                 patch.object(probe, 'cas_mvsnet_loss', side_effect=loss), \
                 patch('builtins.print'):
                probe.run(cli)
                summary = json.loads((root/'run/summary.json').read_text())
                self.assertTrue(summary['checks_passed'])
                self.assertEqual(summary['completed_steps'], {label: 2 for label, _ in probe.RUNS})
                self.assertEqual(set(summary['validation']), {'0', '1', '2'})
                self.assertTrue(all(r['exact'] for r in summary['comparisons']['legacy_repeat'].values()))
                self.assertTrue(any(not r['exact'] for r in summary['comparisons']['stable'].values()))
                self.assertTrue(all(r['first_protected_difference'] is None
                                    for r in summary['comparisons']['stable'].values()))
                self.assertEqual(probe.sha(ckpt), source['checkpoint_sha256'])
                with self.assertRaises(FileExistsError):
                    probe.run(cli)
                cli.output_dir = str(root/'failure')
                with patch.object(probe, 'frozen_backbone_check', side_effect=RuntimeError('injected state failure')):
                    with self.assertRaisesRegex(RuntimeError, 'injected'):
                        probe.run(cli)
                failed = json.loads((root/'failure/summary.json').read_text())
                self.assertEqual(failed['status'], 'failed')
                self.assertFalse(failed['checks_passed'])


if __name__ == '__main__':
    unittest.main()
