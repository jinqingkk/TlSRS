import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

import experiment15_residual_sign_audit as audit
import test_momentum_diagnose as fixture_module
from test_residual_audit_batches import SyntheticDTU


class SignAuditTests(unittest.TestCase):
    def test_signed_gradient_and_full_loss_decomposition(self):
        bias = torch.nn.Parameter(torch.tensor(.2))
        proposal = bias.expand(1, 1, 4)
        target = torch.tensor([[[-.5, -.2, .5, 0.]]])
        valid = torch.ones_like(target, dtype=torch.bool)
        rows, masks = audit.signed_statistics(proposal, target, valid)
        report = audit.gradient_report(proposal, target, masks, {'bias': bias}, .05)
        self.assertGreater(report['groups']['negative']['d_loss_d_proposal_mean'], 0)
        self.assertLess(report['groups']['negative']['mean_proposal_descent_derivative'], 0)
        actual = .05*torch.nn.functional.smooth_l1_loss(proposal, target)
        self.assertAlmostEqual(sum(r['actual_loss_contribution'] for r in report['groups'].values()), float(actual), places=7)
        self.assertAlmostEqual(report['positive_negative_gradient_cosine'], -1)
        pooled = audit.summarize([rows, rows])
        self.assertEqual(pooled['negative']['count'], 4)
        self.assertEqual(pooled['negative']['fraction'], .5)
        self.assertEqual(pooled['negative']['means']['proposal_positive_ratio'], 1)
        self.assertIsNone(bias.grad)
        target = torch.full_like(target, -.5)
        rows, masks = audit.signed_statistics(proposal, target, valid)
        self.assertIsNone(audit.gradient_report(proposal, target, masks, {'bias': bias}, .05)['groups']['positive'])
        self.assertIsNone(audit.summarize([rows])['zero']['means']['target_mean'])

    def test_production_readonly_eval(self):
        fixture = fixture_module.MomentumTests()
        fixture.setUp()
        model, optimizer, args = fixture.model, fixture.optimizer, fixture.args
        model.eval()
        sample = next(iter(DataLoader(SyntheticDTU(), batch_size=2)))
        before = audit.diag.fingerprints(audit.diag.model_snapshot(model, optimizer))
        with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('Must not update')):
            row, protected = audit.audit_batch(model, sample, args, .05)
        self.assertEqual(before, audit.diag.fingerprints(audit.diag.model_snapshot(model, optimizer)))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertEqual(sum(r['count'] for r in row['groups'].values()), 2048)
        self.assertTrue(protected)
        self.assertTrue(any(r is not None for r in row['gradient_report']['groups'].values()))


if __name__ == '__main__':
    unittest.main()
