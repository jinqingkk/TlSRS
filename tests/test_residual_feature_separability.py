"""CPU checks for the read-only residual fused-feature probe."""
import argparse
import unittest

import torch
from torch.utils.data import DataLoader

import experiment15_residual_feature_separability as probe
import test_momentum_diagnose as fixture_module
from test_residual_audit_batches import SyntheticDTU


class ResidualFeatureSeparabilityTests(unittest.TestCase):
    def test_only_known_loader_argument_rewrite_is_accepted(self):
        source = {
            'gate_bce_mode': 'stable_logprob',
            'sger_residual_final_activation': 'leaky_relu',
            'sger_residual_negative_slope': 0.01,
            'nviews': 5,
        }
        configured = argparse.Namespace(
            gate_bce_mode='legacy_clamp',
            sger_residual_final_activation='leaky_relu',
            sger_residual_negative_slope=0.01,
            nviews=5,
        )
        row = probe.require_configured_snapshot_arguments(
            source, configured)
        self.assertEqual(row['configured_gate_bce_mode'], 'legacy_clamp')
        configured.nviews = 4
        with self.assertRaisesRegex(ValueError, 'nviews'):
            probe.require_configured_snapshot_arguments(source, configured)

    def test_balanced_sampler_is_deterministic_and_balanced(self):
        fused = torch.arange(2 * 3 * 2 * 4, dtype=torch.float32).reshape(
            2, 3, 2, 4)
        positive = torch.zeros(2, 2, 4, dtype=torch.bool)
        positive.reshape(-1)[::2] = True
        negative = ~positive
        first = probe.balanced_feature_sample(
            fused, positive, negative, 3)
        second = probe.balanced_feature_sample(
            fused, positive, negative, 3)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertTrue(torch.equal(first[3], second[3]))
        self.assertEqual(first[2]['sampled_per_class'], 3)
        self.assertEqual(int((first[1] > 0).sum()), 3)
        self.assertEqual(int((first[1] < 0).sum()), 3)

    def test_closed_form_probe_separates_held_out_points(self):
        positive = torch.tensor([
            [2.0, 0.0], [1.5, 0.2], [2.5, -0.2], [1.8, 0.1],
        ], dtype=torch.float64)
        negative = torch.tensor([
            [-2.0, 0.0], [-1.5, -0.2], [-2.5, 0.2], [-1.8, -0.1],
        ], dtype=torch.float64)
        features = torch.cat([positive, negative])
        labels = torch.cat([
            torch.ones(4, dtype=torch.float64),
            -torch.ones(4, dtype=torch.float64),
        ])
        fitted = probe.fit_ridge_probe(features, labels)
        fused = torch.tensor([
            [[[3.0, -3.0]], [[0.0, 0.0]]],
        ], dtype=torch.float64)
        scores = probe.probe_scores(fused, fitted)
        accumulator = probe.empty_binary_accumulator()
        expected_positive = torch.tensor([[[True, False]]])
        expected_negative = ~expected_positive
        probe.update_binary_accumulator(
            accumulator, scores, expected_positive, expected_negative)
        result = probe.summarize_binary(accumulator)
        self.assertEqual(result['balanced_accuracy'], 1.0)
        self.assertGreater(result['score_mean_separation'], 0.0)

    def test_production_fused_capture_is_read_only_and_has_26_channels(self):
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
        outputs, logits, fused = probe.checked_forward(
            fixture.model, fixture.optimizer, sample, fixture.args)
        after = probe.diag.fingerprints(
            probe.diag.model_snapshot(fixture.model, fixture.optimizer))
        positive, negative = probe.direction_labels(
            outputs, sample, fixture.args)
        self.assertEqual(before, after)
        self.assertEqual(fused.size(1), 26)
        self.assertEqual(logits.shape, positive.shape)
        self.assertGreater(int(positive.sum()), 0)
        self.assertGreater(int(negative.sum()), 0)
        self.assertEqual(len(probe.channel_names(fused.size(1))), 26)

    def test_cross_scene_decision_requires_every_scene(self):
        def row(accuracy, separation=0.2):
            return {
                'balanced_accuracy': accuracy,
                'score_mean_separation': separation,
            }

        validation = {
            'pooled': {'ridge_probe': row(0.7)},
            'scenes': {
                'a': {'ridge_probe': row(0.6)},
                'b': {'ridge_probe': row(0.54)},
            },
        }
        self.assertFalse(probe.direction_signal_decision(validation)[
            'cross_scene_linear_signal_established'])
        validation['scenes']['b']['ridge_probe'] = row(0.55)
        self.assertTrue(probe.direction_signal_decision(validation)[
            'cross_scene_linear_signal_established'])


if __name__ == '__main__':
    unittest.main()
