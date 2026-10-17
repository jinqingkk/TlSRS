"""CPU synthetic production-model checks for the one-update Adam diagnostic."""
import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import experiment15_momentum_diagnose as diag
from gradient_audit import build_parser, _build_optimizer
from test_residual_audit_batches import SyntheticDTU


class MomentumTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.args = build_parser().parse_args([
            '--trainpath', '.', '--trainlist', 'x', '--audit_epoch', '12',
            '--output_json', 'x', '--batch_size', '2', '--ndepths', '8,8,8'])
        diag.seed(1)
        self.model = diag.build_model(self.args)
        diag.configure_model(self.model, 'leaky_relu', self.args)
        with torch.no_grad():
            head = self.model.sger_blocks[0].residual
            head.body[4].weight.fill_(-0.02)
            head.body[4].bias.fill_(-0.1)
            head.output.weight.fill_(0.02)
        self.optimizer = _build_optimizer(self.model, self.args)
        # Real initialized Adam state with inherited momentum for every tensor.
        for p in self.model.parameters():
            p.grad = torch.full_like(p, 0.001)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        diag.freeze_backbone_parameters(self.model)

    def test_reset_only_first_moment_and_missing_state_rejected(self):
        before = copy.deepcopy(self.optimizer.state_dict())
        result = diag.clear_first_moment(self.model, self.optimizer)
        residual_ids = {id(p) for p in diag.residual_parameters(self.model).values()}
        for group, old_group in zip(self.optimizer.param_groups, before['param_groups']):
            for p, old_id in zip(group['params'], old_group['params']):
                for key, value in before['state'][old_id].items():
                    current = self.optimizer.state[p][key]
                    if key == 'exp_avg' and id(p) in residual_ids:
                        self.assertTrue(torch.equal(current, torch.zeros_like(current)))
                    else:
                        self.assertTrue(torch.equal(current, value))
        self.assertTrue(result['only_residual_exp_avg_changed'])
        p = next(iter(diag.residual_parameters(self.model).values()))
        del self.optimizer.state[p]['exp_avg']
        with self.assertRaises(ValueError):
            diag.clear_first_moment(self.model, self.optimizer)

    def test_full_runner_production_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root/'source'
            ckpt_path = source_dir/'leaky_relu/step_0001.ckpt'
            ckpt_path.parent.mkdir(parents=True)
            torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                        'arguments': vars(self.args), 'epoch': 12, 'probe_step': 1,
                        'freeze_backbone': True, 'gate_bce_mode': 'legacy_clamp'}, str(ckpt_path))
            diag.dump(source_dir/'leaky_relu/fingerprints/step1_update.json',
                      diag.fingerprints(diag.model_snapshot(self.model, self.optimizer)))
            case = diag.audit_cases(self.args)[0]
            manifest = {'status': 'completed', 'steps': 2, 'source_sha256': {},
                        'arguments': vars(self.args), 'effective_case': case._asdict(),
                        'experiment': {'checkpoint_epoch': 12}, 'train_indices': [0, 1, 2, 3],
                        'train_metas': SyntheticDTU.metas[:4]}
            diag.dump(source_dir/'manifest.json', manifest)
            diag.dump(source_dir/'summary.json', {'status': 'completed', 'checks_passed': True})
            original_hash = diag.sha(ckpt_path)
            cli = argparse.Namespace(probe_dir=str(source_dir), output_dir=str(root/'out'),
                                     preflight_only=False, strict_determinism=False)
            forward_states = []
            original_forward = diag.proposal_forward
            def capture_forward(model, sample, args):
                forward_states.append(diag.runtime_state(model))
                return original_forward(model, sample, args)
            with patch.object(torch.nn.Module, 'cuda', lambda self: self), \
                 patch.object(torch.cuda, 'is_available', return_value=True), \
                 patch.object(torch.cuda, 'get_device_name', return_value='CPU test'), \
                 patch.object(torch.cuda, 'get_rng_state_all', return_value=[]), \
                 patch.object(torch.cuda, 'set_rng_state_all'), \
                 patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False), \
                 patch.object(diag, '_move_to_device', side_effect=lambda sample, device: sample), \
                 patch.object(diag, 'find_dataset_def', return_value=SyntheticDTU), \
                 patch.object(diag, 'proposal_forward', side_effect=capture_forward), \
                 patch('builtins.print'):
                diag.run(cli)
            self.assertEqual(len(forward_states), 6)
            for a, b in zip(forward_states[::2], forward_states[1::2]):
                self.assertEqual(a['modes'], b['modes'])
                self.assertTrue(torch.equal(a['torch'], b['torch']))
                self.assertEqual(a['python'], b['python'])
                self.assertEqual(repr(a['numpy']), repr(b['numpy']))
                for name in a['buffers']:
                    self.assertTrue(torch.equal(a['buffers'][name], b['buffers'][name]), name)
            summary = json.loads((root/'out/summary.json').read_text())
            self.assertTrue(summary['checks_passed'])
            self.assertTrue(summary['input_files_unchanged'])
            self.assertEqual(diag.sha(ckpt_path), original_hash)
            self.assertTrue(all(c['exact'] for c in summary['comparisons']['keep_repeat'].values()))
            for phase in ('initial', 'before', 'gradients'):
                self.assertTrue(summary['comparisons']['zero_residual_exp_avg'][phase]['exact'])
            for label in diag.RUNS:
                row = summary['runs'][label]
                self.assertTrue(row['measurement_state_restored'])
                self.assertTrue(row['frozen_backbone_check']['passed'])
                self.assertGreater(row['directions']['groups']['residual_head']['gradient_l2'], 0)
                after = torch.load(str(root/'out'/label/'after.ckpt'), map_location='cpu')
                for name, before in self.model.state_dict().items():
                    if 'num_batches_tracked' in name:
                        expected = before + (1 if name.startswith('normal_head.2.') else 0)
                        self.assertTrue(torch.equal(after['model'][name], expected), name)
                for key in ('loss', 'residual_target_loss', 'proposal_gain_mean'):
                    self.assertAlmostEqual(row['after_minus_before'][key], row['after'][key]-row['before'][key])
            self.assertFalse(summary['comparisons']['zero_residual_exp_avg']['update']['exact'])
            # Historical evidence must detect a replaced/corrupted source snapshot.
            snapshot = torch.load(str(ckpt_path), map_location='cpu')
            snapshot['model']['sger_blocks.0.residual.output.bias'].add_(1)
            torch.save(snapshot, str(ckpt_path))
            with self.assertRaisesRegex(ValueError, 'fingerprints'):
                diag.preflight(source_dir)


if __name__ == '__main__':
    unittest.main()
