import math
import unittest

import torch

from AttentionGuidance.frla import (
    DEFAULT_LAGS,
    FRLAConfig,
    apply_frla,
    fixed_channel_projection,
    local_cosine_descriptor,
)


class FRLATest(unittest.TestCase):
    def make_inputs(self, dtype=torch.float32):
        torch.manual_seed(7)
        latent = torch.randn(2, 4, 32, 32).to(dtype)
        feature = torch.randn(2, 8, 20, 24).to(dtype)
        scheduler_update = torch.randn_like(latent)
        return latent, feature, scheduler_update

    def test_projection_is_reproducible_and_seeded(self):
        first = fixed_channel_projection(8, seed=11, device=torch.device("cpu"))
        second = fixed_channel_projection(8, seed=11, device=torch.device("cpu"))
        other = fixed_channel_projection(8, seed=12, device=torch.device("cpu"))
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other))
        self.assertTrue(torch.allclose(first.norm(dim=1), torch.ones(4)))

    def test_descriptor_has_one_value_per_lag(self):
        value = torch.ones(2, 4, 16, 16)
        descriptor = local_cosine_descriptor(value, DEFAULT_LAGS)
        self.assertEqual(descriptor.shape, (2, len(DEFAULT_LAGS)))
        self.assertTrue(torch.allclose(descriptor, torch.ones_like(descriptor)))

    def test_fixed_operator_reduces_relation_loss_and_respects_cap(self):
        latent, feature, scheduler_update = self.make_inputs()
        feature.requires_grad_(True)
        output = apply_frla(latent, feature, scheduler_update)
        self.assertEqual(output.guided_x0.dtype, latent.dtype)
        self.assertEqual(output.residual.shape, latent.shape)
        self.assertTrue(
            torch.all(output.loss_after <= output.loss_before + 1e-7)
        )
        self.assertTrue(torch.any(output.loss_after < output.loss_before - 1e-8))
        self.assertTrue(torch.all(output.update_ratio <= 0.05 + 1e-6))
        self.assertIsNone(feature.grad)
        self.assertFalse(output.guided_x0.requires_grad)
        self.assertFalse(output.residual.requires_grad)
        self.assertTrue(math.isfinite(output.to_record()["loss_after"][0]))

    def test_zero_scheduler_is_exact_identity(self):
        latent, feature, _scheduler_update = self.make_inputs(torch.bfloat16)
        output = apply_frla(latent, feature, torch.zeros_like(latent))
        self.assertTrue(torch.equal(output.guided_x0, latent))
        self.assertTrue(torch.equal(output.residual, torch.zeros_like(latent)))
        self.assertTrue(torch.equal(output.update_ratio, torch.zeros(2)))

    def test_config_rejects_invalid_lag_and_step(self):
        with self.assertRaises(ValueError):
            FRLAConfig(lags=((16, 0),))
        with self.assertRaises(ValueError):
            FRLAConfig(eta=0.0)

    def test_operator_rejects_nonfinite_inputs(self):
        latent, feature, scheduler_update = self.make_inputs()
        latent[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            apply_frla(latent, feature, scheduler_update)


if __name__ == "__main__":
    unittest.main()
