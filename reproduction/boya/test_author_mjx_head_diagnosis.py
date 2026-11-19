"""Numerical safeguards for read-only head probes; no optimizer or simulator."""
import copy
import unittest

import torch
from torch.nn import functional as F

import author_training as T
import author_mjx_head_diagnosis as D


class HeadDiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.inference = T.components()[1]

    def test_functional_probe_matches_published_and_never_updates_source_buffers(self):
        torch.manual_seed(194)
        core = self.inference().eval(); x = torch.randn(12, 480)
        original = copy.deepcopy(core.state_dict())
        with torch.no_grad():
            probe, _ = D.forward(core, x)
            self.assertTrue(torch.equal(probe, core(x)[0]))
            D.forward(core, x, batch_stats=True)
            calibrated = D.calibrate_fit_only(core, x.numpy())
        self.assertTrue(all(torch.equal(v, core.state_dict()[k]) for k, v in original.items()))
        self.assertFalse(torch.equal(core.bn1.running_var, calibrated.bn1.running_var))
        self.assertTrue(all(torch.equal(v, dict(calibrated.named_parameters())[k]) for k, v in core.named_parameters()))

    def test_output_intervention_changes_only_last_activation_and_recovers_dead_gradient(self):
        torch.manual_seed(195)
        core = self.inference().eval(); x = torch.randn(12, 480)
        with torch.no_grad():
            rectified, pre = D.forward(core, x)
            linear, linear_pre = D.forward(core, x, activation='linear')
            smooth, smooth_pre = D.forward(core, x, activation='softplus')
        self.assertTrue(torch.equal(pre, linear_pre) and torch.equal(pre, smooth_pre))
        self.assertTrue(torch.equal(rectified, F.relu(linear)))
        self.assertTrue(torch.equal(smooth, F.softplus(linear)))
        z = torch.tensor([-2., -1.], requires_grad=True)
        F.mse_loss(F.relu(z), torch.ones_like(z)).backward()
        self.assertTrue((z.grad == 0).all()); z.grad = None
        F.mse_loss(F.softplus(z), torch.ones_like(z)).backward()
        self.assertTrue((z.grad != 0).all())


if __name__ == '__main__':
    unittest.main()
