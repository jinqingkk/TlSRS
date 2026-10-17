"""CPU-only checks for diagnostic evidence; no DTU data or production audits."""
import unittest

import torch

from experiment15_d_diagnose import (compare_snapshots, model_snapshot,
    freeze_backbone_parameters, set_diagnostic_train_mode, frozen_backbone_check)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_stage = 3
        self.backbone = torch.nn.Linear(2, 1)
        self.sger_blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1)])
        self.register_buffer('running_value', torch.zeros(1))


class DiagnosticTests(unittest.TestCase):
    def test_frozen_backbone_bn_and_restored_momentum(self):
        torch.manual_seed(11)
        model = TinyModel()
        model.backbone = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
        model.normal_head = torch.nn.ModuleList([torch.nn.BatchNorm1d(2) for _ in range(3)])
        model.sger_blocks[0] = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 1))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.1)
        sample = torch.randn(4, 2)
        # Populate Adam momentum before freezing, including the backbone.
        model.sger_blocks[0](model.backbone(sample)).square().sum().backward()
        optimizer.step()
        self.assertTrue(optimizer.state[model.backbone[0].weight])
        freeze_backbone_parameters(model)
        set_diagnostic_train_mode(model, True)
        initial = model_snapshot(model, optimizer)
        raw_before = model.backbone(sample).detach().clone()
        for _ in range(2):
            model.eval()  # Mimic the intervening step-0 validation.
            set_diagnostic_train_mode(model, True)
            self.assertFalse(model.backbone[1].training)
            self.assertFalse(model.normal_head[0].training)
            self.assertTrue(model.normal_head[2].training)
            self.assertTrue(model.sger_blocks[0][0].training)
            optimizer.zero_grad(set_to_none=True)
            raw = model.backbone(sample)
            self.assertFalse(raw.requires_grad)
            self.assertTrue(torch.equal(raw_before, raw))
            (model.sger_blocks[0](raw) - 1).square().mean().backward()
            optimizer.step()
            check = frozen_backbone_check(model, optimizer, initial)
            self.assertTrue(check['passed'])
            self.assertTrue(check['sger_parameters_changed'])
            self.assertTrue(check['sger_nonzero_gradients'])
        with torch.no_grad():
            model.backbone[1].running_mean.add_(1)
        self.assertFalse(frozen_backbone_check(model, optimizer, initial)['passed'])

    def test_snapshot_owns_storage(self):
        model = TinyModel()
        before = model_snapshot(model)
        with torch.no_grad():
            model.backbone.weight.add_(1)
            model.running_value.add_(2)
        comparison = compare_snapshots(before, model_snapshot(model))
        self.assertFalse(comparison['exact'])
        self.assertEqual(comparison['entries']['backbone/buffers/running_value']['max_abs_diff'], 2)

    def test_none_zero_nonfinite_and_missing(self):
        a = {'backbone/gradients/x': None, 'finite': torch.zeros(1)}
        b = {'backbone/gradients/x': torch.zeros(1), 'finite': torch.tensor([float('nan')])}
        result = compare_snapshots(a, b)
        self.assertEqual(result['first_protected_difference'], 'backbone/gradients/x')
        self.assertFalse(result['entries']['finite']['all_finite'])
        self.assertFalse(compare_snapshots({'x': None}, {})['exact'])
        self.assertTrue(compare_snapshots({'x': None}, {'x': None})['exact'])

    def test_two_step_control_and_sger_change(self):
        def run(sger_weight):
            torch.manual_seed(7)
            model = TinyModel()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            snapshots = [model_snapshot(model, optimizer)]
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                raw = model.backbone(torch.ones(1, 2))
                refined = model.sger_blocks[0](raw.detach())
                (raw.square().sum() + sger_weight * refined.square().sum()).backward()
                snapshots.append(model_snapshot(model, gradients=True))
                optimizer.step()
                snapshots.append(model_snapshot(model, optimizer))
            return snapshots
        baseline, repeat, changed = run(1.0), run(1.0), run(-1.0)
        for a, b in zip(baseline, repeat):
            self.assertTrue(compare_snapshots(a, b)['exact'])
        self.assertTrue(compare_snapshots(baseline[0], changed[0])['exact'])
        differences = [compare_snapshots(a, b) for a, b in zip(baseline, changed)]
        self.assertTrue(any(not result['exact'] for result in differences))
        self.assertTrue(all(result['first_protected_difference'] is None for result in differences))
        changed[1]['backbone/gradients/backbone.weight'].add_(0.25)
        result = compare_snapshots(baseline[1], changed[1])
        self.assertEqual(result['first_protected_difference'], 'backbone/gradients/backbone.weight')
        self.assertEqual(result['entries'][result['first_protected_difference']]['max_abs_diff'], 0.25)


if __name__ == '__main__':
    unittest.main()
