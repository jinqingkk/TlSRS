"""CPU checks for the read-only residual gradient decomposition."""
import unittest

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import experiment15_residual_gradient_decomposition as audit
import test_momentum_diagnose as fixture_module
from test_residual_audit_batches import SyntheticDTU


class TinyResidual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(2, 3)
        self.output = torch.nn.Linear(3, 1)

    def forward(self, values):
        return self.output(torch.tanh(self.first(values)))[:, 0]


class ResidualGradientDecompositionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = TinyResidual()
        self.parameters = dict(self.model.named_parameters())
        self.inputs = torch.tensor([
            [2.0, 0.0], [1.0, 0.5],
            [-1.0, -0.5], [-2.0, 0.0],
        ])
        self.logits = self.model(self.inputs)
        self.positive = torch.tensor([True, True, False, False])
        self.negative = ~self.positive
        truth = self.positive.to(self.logits.dtype)
        self.direction_loss = F.binary_cross_entropy_with_logits(
            self.logits, truth)
        residual_target = torch.tensor([1.0, 0.5, -0.5, -1.0])
        self.residual_loss = F.smooth_l1_loss(
            torch.tanh(self.logits), residual_target)
        self.production_loss = (
            0.05 * self.residual_loss
            + 0.2 * torch.tanh(self.logits).square().mean())

    def test_exact_decomposition_and_read_only_autograd(self):
        before = {
            name: value.detach().clone()
            for name, value in self.parameters.items()
        }
        report = audit.decompose_gradients(
            self.parameters, self.production_loss, self.residual_loss,
            self.direction_loss, self.logits,
            self.positive, self.negative, 0.05, 0.05)
        checks = report['checks']
        self.assertTrue(
            checks['direction_group_gradient_reconstruction']['passed'])
        self.assertTrue(
            checks['experimental_total_gradient_reconstruction']['passed'])
        self.assertTrue(checks['direction_scalar_reconstruction']['passed'])
        # This tiny fixture has four tensors rather than the production eight.
        self.assertFalse(checks['all_eight_residual_tensors_covered'])
        self.assertEqual(
            set(report['predicted_logit_changes']), {
                'production_total', 'residual_target_weighted',
                'production_other', 'direction_weighted',
                'direction_balanced_weighted', 'experimental_total',
                'experimental_balanced_total'})
        self.assertLessEqual(
            report['checks']['direction_group_gradient_reconstruction'][
                'relative_l2_error'], audit.RECONSTRUCTION_RTOL)
        for name, value in self.parameters.items():
            self.assertTrue(torch.equal(value, before[name]))
            self.assertIsNone(value.grad)

    def test_direction_only_has_correct_class_changes(self):
        report = audit.decompose_gradients(
            self.parameters, self.production_loss, self.residual_loss,
            self.direction_loss, self.logits,
            self.positive, self.negative, 0.05, 0.05)
        changes = report['predicted_logit_changes']['direction_weighted']
        self.assertGreater(
            changes['positive_mean_logit_change_per_unit_sgd'], 0.0)
        self.assertLess(
            changes['negative_mean_logit_change_per_unit_sgd'], 0.0)
        self.assertTrue(changes['both_directions_desired'])
        balanced = report['predicted_logit_changes'][
            'direction_balanced_weighted']
        self.assertGreater(
            balanced['positive_mean_logit_change_per_unit_sgd'], 0.0)
        self.assertLess(
            balanced['negative_mean_logit_change_per_unit_sgd'], 0.0)
        self.assertTrue(balanced['both_directions_desired'])
        interaction = report['interactions'][
            'direction_positive_contribution__direction_negative_contribution']
        self.assertTrue(-1.0 <= interaction['cosine'] <= 1.0)

    def test_requires_both_target_directions(self):
        with self.assertRaisesRegex(ValueError, 'both directions'):
            audit.decompose_gradients(
                self.parameters, self.production_loss, self.residual_loss,
                self.direction_loss, self.logits,
                torch.ones_like(self.positive),
                torch.zeros_like(self.negative), 0.05, 0.05)

    def test_reconstruction_rejects_material_difference(self):
        reference = torch.tensor([1.0, 2.0], dtype=torch.float64)
        good = reference + 1e-8
        bad = reference + 1e-2
        self.assertTrue(audit.reconstruction(reference, good)['passed'])
        self.assertFalse(audit.reconstruction(reference, bad)['passed'])

    def test_production_graph_covers_eight_tensors_without_mutation(self):
        fixture = fixture_module.MomentumTests()
        fixture.setUp()
        audit.direction_probe.freeze_except_residual(fixture.model)
        fixture.model.eval()
        sample = next(iter(DataLoader(
            SyntheticDTU(), batch_sampler=[[0, 1]], num_workers=0)))
        # Force both bounded target directions while retaining valid coarse GT.
        sample['depth']['stage3'][:, :16].fill_(20.0)
        sample['depth']['stage3'][:, 16:].fill_(0.1)
        case = audit.diag.audit_cases(fixture.args)[0]
        report, protected = audit.audit_batch(
            fixture.model, fixture.optimizer, sample, fixture.args, case)
        self.assertTrue(report['state_unchanged'])
        self.assertTrue(report['rng_unchanged'])
        self.assertTrue(
            report['checks']['all_eight_residual_tensors_covered'])
        self.assertGreater(len(protected), 0)
        self.assertGreater(
            report['direction_counts']['positive_count'], 0)
        self.assertGreater(
            report['direction_counts']['negative_count'], 0)
        for check in report['checks'].values():
            if isinstance(check, dict) and 'passed' in check:
                self.assertTrue(check['passed'])


if __name__ == '__main__':
    unittest.main()
