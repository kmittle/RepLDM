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

    def test_explicit_raw_mode_is_exact_default(self):
        guidance = self.make_guidance()
        latents = torch.randn(2, 4, 8, 8)
        default = guidance(3, latents, scale=0.004)
        explicit = guidance(3, latents, scale=0.004, residual_mode="raw")
        self.assertTrue(torch.equal(default, explicit))

    def test_mean_centered_update_preserves_channel_means(self):
        guidance = self.make_guidance()
        latents = torch.randn(2, 4, 8, 8, dtype=torch.float64)
        residual = torch.randn_like(latents)
        guidance.vanilla_attn_guidance = lambda value, alpha_t=None: value + residual
        updated = guidance(3, latents, scale=0.2, residual_mode="mean_centered")
        torch.testing.assert_close(
            updated.mean(dim=(-2, -1)),
            latents.mean(dim=(-2, -1)),
            rtol=0.0,
            atol=1e-12,
        )

    def test_moment_tangent_preserves_channel_mean_and_variance(self):
        guidance = self.make_guidance()
        latents = torch.randn(2, 4, 8, 8, dtype=torch.float64)
        residual = torch.randn_like(latents)
        expected_mean = latents.mean(dim=(-2, -1))
        expected_variance = latents.var(dim=(-2, -1), correction=0)

        for match_raw_energy in (False, True):
            updated = guidance.apply_moment_tangent_update(
                latents,
                residual,
                0.2,
                match_raw_energy=match_raw_energy,
            )
            torch.testing.assert_close(
                updated.mean(dim=(-2, -1)), expected_mean, rtol=0.0, atol=1e-12
            )
            torch.testing.assert_close(
                updated.var(dim=(-2, -1), correction=0),
                expected_variance,
                rtol=1e-12,
                atol=1e-12,
            )

    def test_rescaled_moment_tangent_matches_raw_first_order_energy(self):
        guidance = self.make_guidance()
        latents = torch.randn(2, 4, 8, 8, dtype=torch.float64)
        residual = torch.randn_like(latents)
        scale = 1e-6
        updated = guidance.apply_moment_tangent_update(
            latents, residual, scale, match_raw_energy=True
        )
        observed_speed = torch.linalg.vector_norm(
            updated - latents, dim=(-2, -1)
        ) / scale
        raw_speed = torch.linalg.vector_norm(residual, dim=(-2, -1))
        torch.testing.assert_close(
            observed_speed, raw_speed, rtol=1e-7, atol=1e-7
        )

    def test_moment_tangent_zero_scale_and_degenerate_channels_are_identity(self):
        guidance = self.make_guidance()
        latents = torch.randn(2, 4, 8, 8)
        latents[1, 0] = 3.0
        residual = torch.randn_like(latents)
        updated = guidance.apply_moment_tangent_update(
            latents, residual, torch.tensor([0.0, 0.2])
        )
        self.assertTrue(torch.equal(updated[0], latents[0]))
        self.assertTrue(torch.equal(updated[1, 0], latents[1, 0]))
        self.assertTrue(torch.isfinite(updated).all())

    def test_moment_tangent_has_finite_gradients(self):
        guidance = self.make_guidance()
        latents = torch.randn(1, 4, 8, 8, dtype=torch.float64, requires_grad=True)
        residual = torch.randn_like(latents, requires_grad=True)
        scale = torch.tensor(0.02, dtype=torch.float64, requires_grad=True)
        weights = torch.linspace(
            0.1, 1.0, latents.numel(), dtype=torch.float64
        ).reshape_as(latents)
        updated = guidance.apply_moment_tangent_update(
            latents, residual, scale, match_raw_energy=True
        )
        (updated * weights).sum().backward()
        for value in (latents, residual, scale):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
        self.assertGreater(float(scale.grad.abs()), 0.0)

    def test_moment_tangent_rejects_incompatible_action_constraints(self):
        with self.assertRaises(ValueError):
            GuidanceAction(
                band_scales=(0.001, 0.001, 0.001),
                residual_mode="moment_tangent",
            )
        with self.assertRaises(ValueError):
            GuidanceAction(
                scale=0.001,
                max_update_ratio=0.1,
                residual_mode="moment_tangent",
            )
        with self.assertRaises(ValueError):
            self.make_guidance()(
                3, torch.randn(1, 4, 8, 8), residual_mode="moment_tangent"
            )

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
        tangent_schedule = ScheduleGuidanceController(
            scale_schedule=(0.004, 0.003, 0.002, 0.001),
            residual_mode="moment_tangent",
        )
        self.assertEqual(tangent_schedule(observation).residual_mode, "moment_tangent")
        with self.assertRaises(ValueError):
            GuidanceAction(scale=0.001, band_scales=(0.001, 0.001, 0.001))


if __name__ == "__main__":
    unittest.main()
