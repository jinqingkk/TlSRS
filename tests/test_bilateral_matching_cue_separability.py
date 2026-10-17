"""CPU checks for the bilateral multi-view matching-cue probe."""
import unittest

import torch
from torch.utils.data import DataLoader

import experiment15_bilateral_matching_cue_separability as probe
import test_momentum_diagnose as fixture_module
from test_residual_audit_batches import SyntheticDTU


class BilateralMatchingCueSeparabilityTests(unittest.TestCase):
    def test_synthetic_bilateral_features_have_declared_signs(self):
        ref = torch.tensor([1.0, 0.0]).view(1, 2, 1, 1)
        matching = ref.clone()
        mismatching = torch.tensor([0.0, 1.0]).view(1, 2, 1, 1)
        valid = torch.ones(1, 1, 1, dtype=torch.bool)

        positive, positive_support, positive_count = (
            probe.bilateral_cues_from_warped(
                ref, [mismatching, mismatching], [matching, matching],
                [valid, valid], [valid, valid]))
        negative, negative_support, negative_count = (
            probe.bilateral_cues_from_warped(
                ref, [matching, matching], [mismatching, mismatching],
                [valid, valid], [valid, valid]))

        self.assertTrue(positive_support.all())
        self.assertTrue(negative_support.all())
        self.assertTrue(torch.equal(
            positive_count, torch.full_like(positive_count, 2)))
        self.assertTrue(torch.equal(
            negative_count, torch.full_like(negative_count, 2)))
        self.assertTrue((positive > 0).all())
        self.assertTrue((negative < 0).all())

    def test_bilateral_validity_masks_exclude_padding_evidence(self):
        ref = torch.tensor([1.0, 0.0]).view(1, 2, 1, 1)
        matching = ref.clone()
        padding = torch.zeros_like(ref)
        mismatching = torch.tensor([0.0, 1.0]).view(1, 2, 1, 1)
        valid = torch.ones(1, 1, 1, dtype=torch.bool)
        invalid = torch.zeros_like(valid)

        cues, support, count = probe.bilateral_cues_from_warped(
            ref,
            [padding, matching, matching],
            [matching, mismatching, mismatching],
            [invalid, valid, valid],
            [valid, valid, valid])

        self.assertTrue(support.all())
        self.assertEqual(int(count.item()), 2)
        self.assertTrue((cues < 0).all())

    def test_actual_bin_interval_is_derived_and_invalid_inputs_rejected(self):
        depth = torch.tensor([1.0, 1.5, 2.0, 2.5]).view(1, 4, 1, 1)
        delta = probe.actual_depth_bin_interval(depth)
        self.assertAlmostEqual(float(delta), 0.5)
        with self.assertRaisesRegex(ValueError, 'monotonically'):
            probe.actual_depth_bin_interval(
                torch.tensor([1.0, 2.0, 1.5]).view(1, 3, 1, 1))
        with self.assertRaisesRegex(ValueError, 'positive'):
            probe.actual_depth_bin_interval(
                torch.tensor([1.0, 1.0, 2.0]).view(1, 3, 1, 1))
        with self.assertRaisesRegex(RuntimeError, 'Nonfinite'):
            probe.actual_depth_bin_interval(
                torch.tensor([1.0, float('nan'), 2.0]).view(1, 3, 1, 1))
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            probe.actual_depth_bin_interval(
                torch.tensor([1.0, 1.5, 2.5]).view(1, 3, 1, 1))

    def test_fewer_than_two_valid_sources_is_unsupported(self):
        ref = torch.tensor([1.0, 0.0]).view(1, 2, 1, 1)
        match = ref.clone()
        mismatch = torch.tensor([0.0, 1.0]).view(1, 2, 1, 1)
        valid = torch.ones(1, 1, 1, dtype=torch.bool)
        invalid = torch.zeros_like(valid)

        cues, support, count = probe.bilateral_cues_from_warped(
            ref, [mismatch, mismatch], [match, match],
            [valid, invalid], [valid, invalid])

        self.assertFalse(support.any())
        self.assertEqual(int(count.item()), 1)
        self.assertTrue(torch.equal(cues, torch.zeros_like(cues)))

    def test_three_groups_reuse_exact_supported_pixel_indices(self):
        fused = torch.arange(
            2 * 3 * 2 * 4, dtype=torch.float32).reshape(2, 3, 2, 4)
        bilateral = torch.arange(
            2 * 4 * 2 * 4, dtype=torch.float32).reshape(2, 4, 2, 4)
        positive = torch.zeros(2, 2, 4, dtype=torch.bool)
        positive.reshape(-1)[::2] = True
        negative = ~positive
        support = torch.ones_like(positive)
        support.reshape(-1)[-2:] = False

        groups, labels, sampling, indices = probe.sampled_feature_groups(
            fused, bilateral, positive, negative, support, 3)

        bilateral_flat = bilateral.permute(0, 2, 3, 1).reshape(-1, 4)
        self.assertTrue(torch.equal(
            groups['bilateral_only'], bilateral_flat[indices].double()))
        self.assertTrue(torch.equal(
            groups['base_plus_bilateral'][:, :3], groups['base']))
        self.assertTrue(torch.equal(
            groups['base_plus_bilateral'][:, 3:],
            groups['bilateral_only']))
        self.assertEqual(sampling['sampled_per_class'], 3)
        self.assertEqual(int((labels > 0).sum()), 3)
        self.assertEqual(int((labels < 0).sum()), 3)

    def test_decision_checks_absolute_relative_and_support_gates(self):
        def metric(accuracy, separation=0.2):
            return {
                'balanced_accuracy': accuracy,
                'score_mean_separation': separation,
            }

        validation = {
            'pooled': {
                'base': metric(0.57),
                'bilateral_only': metric(0.60),
                'base_plus_bilateral': metric(0.61),
            },
            'scenes': {
                'scan3': {
                    'base': metric(0.55),
                    'bilateral_only': metric(0.56),
                    'base_plus_bilateral': metric(0.58),
                },
                'scan5': {
                    'base': metric(0.54),
                    'bilateral_only': metric(0.56),
                    'base_plus_bilateral': metric(0.56),
                },
                'scan17': {
                    'base': metric(0.55),
                    'bilateral_only': metric(0.56),
                    'base_plus_bilateral': metric(0.57),
                },
            },
        }

        def support_row(ratio=0.95, gap=0.01):
            return {
                'all_ratio': ratio,
                'positive_ratio': ratio,
                'negative_ratio': ratio - gap,
                'positive_negative_ratio_gap': gap,
            }

        support = {
            'train': support_row(),
            'validation': {
                'pooled': support_row(),
                'scenes': {
                    name: support_row()
                    for name in validation['scenes']
                },
            },
        }
        decision = probe.primary_decision(validation, support)
        self.assertTrue(decision['bilateral_matching_cues_established'])
        self.assertAlmostEqual(
            decision['pooled_balanced_accuracy_delta_vs_base'], 0.04)

        validation['pooled']['base_plus_bilateral'] = metric(0.599)
        self.assertFalse(probe.primary_decision(
            validation, support)['bilateral_matching_cues_established'])
        validation['pooled']['base_plus_bilateral'] = metric(0.61)
        validation['scenes']['scan5']['base_plus_bilateral'] = metric(0.53)
        self.assertFalse(probe.primary_decision(
            validation, support)['bilateral_matching_cues_established'])
        validation['scenes']['scan5']['base_plus_bilateral'] = metric(0.56)
        support['validation']['scenes']['scan5'] = support_row(0.89)
        self.assertFalse(probe.primary_decision(
            validation, support)['bilateral_matching_cues_established'])

    def test_production_capture_is_read_only_and_restores_hook_function(self):
        fixture = fixture_module.MomentumTests()
        fixture.setUp()
        probe.gradient_audit.direction_probe.freeze_except_residual(
            fixture.model)
        fixture.model.eval()
        sample = next(iter(DataLoader(
            SyntheticDTU(), batch_sampler=[[0, 1]], num_workers=0)))
        sample['depth']['stage3'][:, :16].fill_(20.0)
        sample['depth']['stage3'][:, 16:].fill_(0.1)
        before = probe.diag.fingerprints(
            probe.diag.model_snapshot(fixture.model, fixture.optimizer))
        original_forward = probe.cas_module.DepthNet.forward

        result = probe.checked_forward(
            fixture.model, fixture.optimizer, sample, fixture.args)

        (outputs, logits, fused, bilateral, support,
         valid_count, delta) = result
        after = probe.diag.fingerprints(
            probe.diag.model_snapshot(fixture.model, fixture.optimizer))
        positive, negative = probe.base_probe.direction_labels(
            outputs, sample, fixture.args)
        self.assertEqual(before, after)
        self.assertIs(probe.cas_module.DepthNet.forward, original_forward)
        self.assertEqual(fused.size(1), 26)
        self.assertEqual(bilateral.size(1), 4)
        self.assertEqual(logits.shape, positive.shape)
        self.assertEqual(support.shape, positive.shape)
        self.assertEqual(valid_count.shape, positive.shape)
        self.assertEqual(delta.shape, positive.shape)
        self.assertGreater(float(support.float().mean()), 0.8)
        self.assertTrue((valid_count[support] >= 2).all())
        self.assertGreater(int(positive.sum()), 0)
        self.assertGreater(int(negative.sum()), 0)
        self.assertTrue(torch.isfinite(bilateral).all())


if __name__ == '__main__':
    unittest.main()
