import unittest

import torch

from AttentionGuidance import (
    AttnGuidance,
    ConstantGuidanceController,
    GuidanceAction,
    GuidanceObservation,
    ScheduleGuidanceController,
)


class AttentionGuidanceTest(unittest.TestCase):
    def make_guidance(self, **kwargs):
        return AttnGuidance(
            dtype=torch.float32,
            device="cpu",
            num_total_steps=4,
            h=8,
            w=8,
            guidance_scale=0.01,
            **kwargs,
        )

    def test_legacy_schedule_is_reentrant(self):
        guidance = self.make_guidance(
            guidance_scale_decay=("linear", 0.0, 1),
            guidance_filter=("linear", 0.25),
        )
        initial = torch.randn(1, 4, 8, 8)

        def rollout():
            latents = initial.clone()
            for t_index in range(3, -1, -1):
                latents = guidance(t_index, latents)
            return latents

        torch.testing.assert_close(rollout(), rollout())

    def test_legacy_decay_follows_sampling_order(self):
        guidance = self.make_guidance(guidance_scale_decay=("linear", 0.0, 1))
        guidance.vanilla_attn_guidance = lambda latents, alpha_t=None: latents + 1
        latents = torch.zeros(1, 4, 2, 2)
        observed = []
        for t_index in range(3, -1, -1):
            updated = guidance(t_index, latents)
            observed.append(float((updated - latents).mean()))
        expected = guidance.guidance_step_scale.tolist()
        self.assertEqual(len(observed), len(expected))
        for actual, target in zip(observed, expected):
            self.assertAlmostEqual(actual, target, places=6)

    def test_equal_band_gains_recover_scalar_guidance(self):
        guidance = self.make_guidance()
        latents = torch.randn(2, 4, 8, 8)
        scalar = guidance(3, latents, scale=0.004)
        spectral = guidance(3, latents, band_scales=(0.004, 0.004, 0.004))
        self.assertTrue(torch.equal(spectral, scalar))

    def test_equal_band_gains_match_legacy_fp16_quantization(self):
        guidance = AttnGuidance(
            dtype=torch.float16,
            device="cpu",
            num_total_steps=4,
            h=8,
            w=8,
            guidance_scale=0.004,
        )
        residual = torch.randn(1, 4, 8, 8, dtype=torch.float16)
        scalar_update = guidance.guidance_step_scale[0] * residual
        band_update = guidance.apply_frequency_band_scales(
            residual, (0.004, 0.004, 0.004)
        )
        self.assertTrue(torch.equal(band_update, scalar_update))

    def test_band_gains_are_differentiable(self):
        guidance = self.make_guidance()
        latents = torch.randn(1, 4, 8, 8)
        scales = torch.tensor([0.002, 0.003, 0.004], requires_grad=True)
        guidance(3, latents, band_scales=scales).square().mean().backward()
        self.assertIsNotNone(scales.grad)
        self.assertTrue(torch.isfinite(scales.grad).all())
        self.assertGreater(float(scales.grad.abs().sum()), 0.0)

    def test_update_ratio_caps_each_batch_item(self):
        update = torch.full((2, 1, 2, 2), 2.0)
        reference = torch.ones_like(update)
        limited = AttnGuidance.limit_update_ratio(update, reference, 0.25)
        ratios = torch.linalg.vector_norm(limited, dim=(1, 2, 3)) / torch.linalg.vector_norm(
            reference, dim=(1, 2, 3)
        )
        torch.testing.assert_close(ratios, torch.full((2,), 0.25))

    def test_controller_contract_and_schedule_order(self):
        observation = GuidanceObservation(
            step_index=1,
            t_index=2,
            timestep=torch.tensor(500),
            alpha_t=torch.tensor(0.5),
            latents=torch.zeros(1, 4, 2, 2),
            denoising_update=torch.ones(1, 4, 2, 2),
        )
        constant = ConstantGuidanceController(band_scales=(0.0, 0.002, 0.004))
        self.assertEqual(tuple(constant(observation).band_scales), (0.0, 0.002, 0.004))
        schedule = ScheduleGuidanceController(scale_schedule=(0.004, 0.003, 0.002, 0.001))
        self.assertEqual(schedule(observation).scale, 0.003)
        with self.assertRaises(ValueError):
            GuidanceAction(scale=0.001, band_scales=(0.001, 0.001, 0.001))


if __name__ == "__main__":
    unittest.main()
