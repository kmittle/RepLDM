import unittest

import torch

from AttentionGuidance import (
    TrajectoryCorrectionConfig,
    apply_ancestral_correction,
)


class _Config:
    prediction_type = "epsilon"


class _Scheduler:
    config = _Config()
    sigmas = torch.tensor([2.0, 1.0, 0.0])


class AncestralCorrectionTest(unittest.TestCase):
    def setUp(self):
        self.scheduler = _Scheduler()
        self.sample = torch.tensor([[[[1.0, 0.5], [0.0, -0.5]]]])
        self.x0 = torch.tensor([[[[0.2, 0.1], [0.0, -0.1]]]])
        self.euler = self.sample + (self.sample - self.x0) / 2.0 * (1.0 - 2.0)

    def test_zero_mix_is_exact_and_does_not_consume_rng(self):
        generator = torch.Generator().manual_seed(19)
        before = generator.get_state().clone()
        corrected, diagnostics = apply_ancestral_correction(
            scheduler=self.scheduler,
            sample=self.sample,
            pred_original_sample=self.x0,
            euler_prev_sample=self.euler,
            step_index=0,
            config=TrajectoryCorrectionConfig(mix=0.0),
            generator=generator,
        )
        self.assertIs(corrected, self.euler)
        self.assertTrue(torch.equal(corrected, self.euler))
        self.assertTrue(torch.equal(generator.get_state(), before))
        self.assertEqual(diagnostics.applied_correction_norm_ratio, 0.0)

    def test_noise_correction_is_reproducible_and_bounded(self):
        config = TrajectoryCorrectionConfig(
            mix=0.5, noise_mode="sqrt", max_correction_ratio=0.1
        )
        first, first_diag = apply_ancestral_correction(
            scheduler=self.scheduler,
            sample=self.sample,
            pred_original_sample=self.x0,
            euler_prev_sample=self.euler,
            step_index=0,
            config=config,
            generator=torch.Generator().manual_seed(7),
        )
        second, second_diag = apply_ancestral_correction(
            scheduler=self.scheduler,
            sample=self.sample,
            pred_original_sample=self.x0,
            euler_prev_sample=self.euler,
            step_index=0,
            config=config,
            generator=torch.Generator().manual_seed(7),
        )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first_diag.to_record(), second_diag.to_record())
        self.assertLessEqual(first_diag.applied_correction_norm_ratio, 0.1 + 1e-6)
        self.assertTrue(first_diag.capped)

    def test_prediction_type_and_sigma_are_checked(self):
        self.scheduler.config.prediction_type = "v_prediction"
        with self.assertRaises(ValueError):
            apply_ancestral_correction(
                scheduler=self.scheduler,
                sample=self.sample,
                pred_original_sample=self.x0,
                euler_prev_sample=self.euler,
                step_index=0,
                config=TrajectoryCorrectionConfig(mix=0.5, noise_mode="none"),
            )


if __name__ == "__main__":
    unittest.main()
