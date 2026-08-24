import unittest

import torch

try:
    from diffusers import EulerAncestralDiscreteScheduler, EulerDiscreteScheduler
except ImportError:  # pragma: no cover - exercised in the minimal unit-test env
    EulerAncestralDiscreteScheduler = None
    EulerDiscreteScheduler = None

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

    @unittest.skipIf(
        EulerAncestralDiscreteScheduler is None,
        "diffusers is not installed; native scheduler parity is unavailable",
    )
    def test_mix_one_matches_native_euler_ancestral_scheduler(self):
        """The endpoint must reproduce the native scheduler at every step."""

        euler = EulerDiscreteScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            prediction_type="epsilon",
        )
        ancestral = EulerAncestralDiscreteScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            prediction_type="epsilon",
        )
        num_steps = 5
        euler.set_timesteps(num_steps)
        ancestral.set_timesteps(num_steps)
        self.assertTrue(torch.equal(euler.timesteps, ancestral.timesteps))
        self.assertTrue(torch.allclose(euler.sigmas, ancestral.sigmas, atol=1e-7, rtol=1e-7))

        sample = torch.randn((1, 4, 8, 8), generator=torch.Generator().manual_seed(123))
        model_generator = torch.Generator().manual_seed(456)
        native_generator = torch.Generator().manual_seed(789)
        correction_generator = torch.Generator().manual_seed(789)
        diagnostics = []

        for step_index, (euler_timestep, ancestral_timestep) in enumerate(
            zip(euler.timesteps, ancestral.timesteps)
        ):
            # Both schedulers must receive the same pre-step state. Calling
            # scale_model_input initializes their internal step index.
            euler.scale_model_input(sample, euler_timestep)
            ancestral.scale_model_input(sample, ancestral_timestep)
            model_output = torch.randn(
                sample.shape, generator=model_generator, dtype=sample.dtype
            )
            euler_output = euler.step(
                model_output, euler_timestep, sample, return_dict=True
            )
            native_output = ancestral.step(
                model_output,
                ancestral_timestep,
                sample,
                generator=native_generator,
                return_dict=True,
            )
            corrected, step_diagnostics = apply_ancestral_correction(
                scheduler=euler,
                sample=sample,
                pred_original_sample=euler_output.pred_original_sample,
                euler_prev_sample=euler_output.prev_sample,
                step_index=step_index,
                config=TrajectoryCorrectionConfig(mix=1.0, noise_mode="sqrt"),
                generator=correction_generator,
            )
            self.assertTrue(
                torch.allclose(corrected, native_output.prev_sample, atol=2e-6, rtol=2e-6),
                msg=f"native parity failed at step {step_index}",
            )
            diagnostics.append(step_diagnostics.to_record())
            sample = native_output.prev_sample

        self.assertEqual(len(diagnostics), num_steps)
        self.assertEqual(
            [record["step_index"] for record in diagnostics], list(range(num_steps))
        )
        for previous, current in zip(diagnostics, diagnostics[1:]):
            self.assertGreater(previous["sigma_from"], current["sigma_from"])
            self.assertGreaterEqual(previous["sigma_to"], current["sigma_to"])
        for record in diagnostics:
            self.assertTrue(
                all(
                    torch.isfinite(torch.tensor(record[key]))
                    for key in (
                        "sigma_from",
                        "sigma_to",
                        "sigma_up",
                        "raw_correction_norm_ratio",
                        "applied_correction_norm_ratio",
                    )
                )
            )

    def _assert_native_scheduler_parity(self, dtype):
        """Run a low-precision endpoint comparison against native sampling."""
        euler = EulerDiscreteScheduler(num_train_timesteps=1000, prediction_type="epsilon")
        ancestral = EulerAncestralDiscreteScheduler.from_config(euler.config)
        euler.set_timesteps(5)
        ancestral.set_timesteps(5)
        sample = torch.randn(
            (1, 4, 8, 8),
            generator=torch.Generator().manual_seed(123),
            dtype=dtype,
        )
        model_generator = torch.Generator().manual_seed(456)
        native_generator = torch.Generator().manual_seed(789)
        correction_generator = torch.Generator().manual_seed(789)
        for step_index, (euler_timestep, ancestral_timestep) in enumerate(
            zip(euler.timesteps, ancestral.timesteps)
        ):
            euler.scale_model_input(sample, euler_timestep)
            ancestral.scale_model_input(sample, ancestral_timestep)
            model_output = torch.randn(
                sample.shape, generator=model_generator, dtype=sample.dtype
            )
            euler_output = euler.step(
                model_output, euler_timestep, sample, return_dict=True
            )
            native_output = ancestral.step(
                model_output,
                ancestral_timestep,
                sample,
                generator=native_generator,
                return_dict=True,
            )
            corrected, _ = apply_ancestral_correction(
                scheduler=euler,
                sample=sample,
                pred_original_sample=euler_output.pred_original_sample,
                euler_prev_sample=euler_output.prev_sample,
                step_index=step_index,
                config=TrajectoryCorrectionConfig(mix=1.0, noise_mode="sqrt"),
                generator=correction_generator,
            )
            max_abs = float(
                (corrected.float() - native_output.prev_sample.float()).abs().max()
            )
            self.assertLessEqual(
                max_abs,
                2e-3,
                msg=f"{dtype} native parity failed at step {step_index}: max_abs={max_abs}",
            )
            sample = native_output.prev_sample

    @unittest.skipIf(
        EulerAncestralDiscreteScheduler is None,
        "diffusers is not installed; native scheduler parity is unavailable",
    )
    def test_mix_one_matches_native_scheduler_in_fp16(self):
        """The endpoint must not depend on an fp32-only latent path."""
        self._assert_native_scheduler_parity(torch.float16)

    @unittest.skipIf(
        EulerAncestralDiscreteScheduler is None,
        "diffusers is not installed; native scheduler parity is unavailable",
    )
    def test_mix_one_matches_native_scheduler_in_bfloat16(self):
        """The endpoint preserves native bfloat16 arithmetic as well."""
        self._assert_native_scheduler_parity(torch.bfloat16)

    def test_mix_one_matches_legacy_native_dtype_stub(self):
        """Cover the pre-upcast scheduler path without pinning an old package."""
        sample = self.sample.to(dtype=torch.float16)
        x0 = self.x0.to(dtype=torch.float16)
        sigma_from, sigma_to = self.scheduler.sigmas[:2]
        euler = sample + (sample - x0) / sigma_from * (sigma_to - sigma_from)
        native_generator = torch.Generator().manual_seed(53)
        correction_generator = torch.Generator().manual_seed(53)
        sigma_up = torch.sqrt(
            sigma_to.square()
            * (sigma_from.square() - sigma_to.square())
            / sigma_from.square()
        )
        sigma_down = torch.sqrt(torch.clamp(sigma_to.square() - sigma_up.square(), min=0.0))
        native = sample + (sample - x0) / sigma_from * (sigma_down - sigma_from)
        native = native + torch.randn(
            sample.shape, generator=native_generator, dtype=sample.dtype
        ) * sigma_up
        corrected, _ = apply_ancestral_correction(
            scheduler=self.scheduler,
            sample=sample,
            pred_original_sample=x0,
            euler_prev_sample=euler,
            step_index=0,
            config=TrajectoryCorrectionConfig(mix=1.0, noise_mode="sqrt"),
            generator=correction_generator,
        )
        max_abs = float((corrected.float() - native.float()).abs().max())
        self.assertLessEqual(max_abs, 2e-3, msg=f"legacy max_abs={max_abs}")


if __name__ == "__main__":
    unittest.main()
