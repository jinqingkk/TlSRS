"""CPU checks for the signed probability-volume cue audit."""
import unittest

import torch
from torch.utils.data import DataLoader

import experiment15_signed_probability_cue_separability as probe
import test_momentum_diagnose as fixture_module
from test_residual_audit_batches import SyntheticDTU


class SignedProbabilityCueSeparabilityTests(unittest.TestCase):
    def test_signed_cues_have_declared_direction_and_are_finite(self):
        probability = torch.tensor([
            [0.1, 0.2, 0.6, 0.1],
            [0.1, 0.6, 0.2, 0.1],
        ], dtype=torch.float32).view(2, 4, 1, 1)
        depth = torch.arange(1.0, 5.0).view(1, 4, 1, 1).expand_as(
            probability)

        cues = probe.signed_probability_volume_cues(
            probability, depth, 1.0)

        self.assertEqual(tuple(cues.shape), (2, 4, 1, 1))
        self.assertTrue(torch.isfinite(cues).all())
        self.assertGreater(float(cues[0, 0]), 0.0)
        self.assertGreater(float(cues[0, 1]), 0.0)
        self.assertLess(float(cues[0, 2]), 0.0)
        self.assertLess(float(cues[0, 3]), 0.0)
        self.assertLess(float(cues[1, 0]), 0.0)
        self.assertLess(float(cues[1, 1]), 0.0)
        self.assertGreater(float(cues[1, 2]), 0.0)
        self.assertGreater(float(cues[1, 3]), 0.0)

    def test_signed_cues_handle_one_depth_bin(self):
        probability = torch.ones(1, 1, 2, 3)
        depth = torch.full_like(probability, 2.0)

        cues = probe.signed_probability_volume_cues(
            probability, depth, 0.5)

        self.assertTrue(torch.equal(cues, torch.zeros_like(cues)))

    def test_invalid_probability_or_depth_order_is_rejected(self):
        probability = torch.tensor([0.4, 0.6]).view(1, 2, 1, 1)
        depth = torch.tensor([2.0, 1.0]).view(1, 2, 1, 1)
        with self.assertRaisesRegex(ValueError, 'nondecreasing'):
            probe.signed_probability_volume_cues(
                probability, depth, 1.0)
        with self.assertRaisesRegex(ValueError, 'not normalized'):
            probe.signed_probability_volume_cues(
                probability * 0.5, depth.flip(1), 1.0)

    def test_group_sampling_reuses_exact_pixel_indices(self):
        fused = torch.arange(
            2 * 3 * 2 * 4, dtype=torch.float32).reshape(2, 3, 2, 4)
        signed = torch.arange(
            2 * 4 * 2 * 4, dtype=torch.float32).reshape(2, 4, 2, 4)
        positive = torch.zeros(2, 2, 4, dtype=torch.bool)
        positive.reshape(-1)[::2] = True
        negative = ~positive

        groups, labels, sampling, indices = probe.sampled_feature_groups(
            fused, signed, positive, negative, 3)

        signed_flat = signed.permute(0, 2, 3, 1).reshape(-1, 4)
        self.assertTrue(torch.equal(
            groups['signed_only'], signed_flat[indices].double()))
        self.assertTrue(torch.equal(
            groups['base_plus_signed'][:, :3], groups['base']))
        self.assertTrue(torch.equal(
            groups['base_plus_signed'][:, 3:], groups['signed_only']))
        self.assertEqual(sampling['sampled_per_class'], 3)
        self.assertEqual(int((labels > 0).sum()), 3)
        self.assertEqual(int((labels < 0).sum()), 3)

    def test_production_capture_is_read_only_and_restores_function(self):
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
        original_statistics = probe.cas_module.probability_volume_statistics

        outputs, logits, fused, signed = probe.checked_forward(
            fixture.model, fixture.optimizer, sample, fixture.args)

        after = probe.diag.fingerprints(
            probe.diag.model_snapshot(fixture.model, fixture.optimizer))
        positive, negative = probe.base_probe.direction_labels(
            outputs, sample, fixture.args)
        self.assertEqual(before, after)
        self.assertIs(
            probe.cas_module.probability_volume_statistics,
            original_statistics)
        self.assertEqual(fused.size(1), 26)
        self.assertEqual(signed.size(1), 4)
        self.assertEqual(logits.shape, positive.shape)
        self.assertGreater(int(positive.sum()), 0)
        self.assertGreater(int(negative.sum()), 0)
        self.assertTrue(torch.isfinite(signed).all())

    def test_primary_decision_uses_base_plus_signed_group(self):
        def row(accuracy, separation=0.2):
            return {
                'balanced_accuracy': accuracy,
                'score_mean_separation': separation,
            }

        validation = {
            'pooled': {
                'base': row(0.49),
                'signed_only': row(0.58),
                'base_plus_signed': row(0.61),
            },
            'scenes': {
                'a': {
                    'base': row(0.48),
                    'signed_only': row(0.56),
                    'base_plus_signed': row(0.56),
                },
                'b': {
                    'base': row(0.50),
                    'signed_only': row(0.54),
                    'base_plus_signed': row(0.55),
                },
            },
        }
        decision = probe.primary_decision(validation)
        self.assertTrue(decision['signed_probability_cues_established'])
        self.assertAlmostEqual(
            decision['pooled_balanced_accuracy_delta_vs_base'], 0.12)
        validation['scenes']['b']['base_plus_signed'] = row(0.549)
        self.assertFalse(probe.primary_decision(validation)[
            'signed_probability_cues_established'])


if __name__ == '__main__':
    unittest.main()
