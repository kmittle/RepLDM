import importlib.util
from pathlib import Path
import unittest

import torch

from AttentionGuidance.latent_renderer import (
    LatentRendererConfig,
    StructuralUNetFeatureCapture,
    StructuralLatentRenderer,
    _fixed_moment_geodesic,
    _project_fixed_moment_tangent,
    _cast_guided_x0_with_cap,
    build_fixed_coefficient_renderer,
    cap_update_norm,
)


def _load_cfg_batch():
    path = Path(__file__).parents[1] / "InferencePipelines" / "cfg_batch.py"
    spec = importlib.util.spec_from_file_location("repldm_cfg_batch_audit", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_cfg_batch = _load_cfg_batch()
expand_cfg_latents = _cfg_batch.expand_cfg_latents
expand_cfg_time_ids = _cfg_batch.expand_cfg_time_ids
split_cfg_noise_pred = _cfg_batch.split_cfg_noise_pred


class FRLAStructuralAuditTest(unittest.TestCase):
    @staticmethod
    def _deterministic_cast_case(dtype, preserve_moments=True, enforce=True):
        """Return one deliberate overrun and one exact no-op sample."""
        base = torch.tensor(
            [[[[1.0, 2.0], [3.0, 4.0]]], [[[2.0, 4.0], [6.0, 8.0]]]],
            dtype=dtype,
        )
        update = torch.zeros_like(base)
        update[0].fill_(0.02)
        scheduler_update = torch.ones_like(base)
        result = _cast_guided_x0_with_cap(
            base,
            update,
            scheduler_update,
            0.005,
            preserve_moments,
            1e-6,
            enforce,
        )
        return base, update, scheduler_update, result

    @staticmethod
    def _renderer_case(dtype, seed, strict=True, preserve_moments=True):
        torch.manual_seed(seed)
        latent = (torch.randn(2, 4, 8, 8) * 10.0).to(dtype)
        bases = (torch.randn(2, 4, 4, 8, 8) * 1000.0).to(dtype)
        scheduler_update = (torch.randn_like(latent) * 10.0).to(dtype)
        renderer = StructuralLatentRenderer(
            LatentRendererConfig(
                num_bases=4,
                max_update_ratio=0.1,
                preserve_moments=preserve_moments,
                enforce_post_cast_cap=strict,
            )
        )
        with torch.no_grad():
            renderer.policy[-1].bias.fill_(0.9)
        output = renderer(
            latent,
            bases,
            scheduler_update=scheduler_update,
        )
        return latent, scheduler_update, output

    def test_float32_geodesic_residual_stays_inside_trust_cap(self):
        torch.manual_seed(21)
        latent = torch.randn(2, 4, 8, 8) * 10.0
        raw = torch.randn_like(latent) * 100.0
        scheduler_update = torch.randn_like(latent) * 10.0
        tangent = _project_fixed_moment_tangent(latent, raw, 1e-6)
        bounded = cap_update_norm(tangent, scheduler_update, 0.1)
        guided = _fixed_moment_geodesic(latent, bounded, 1e-6)
        ratio = torch.linalg.vector_norm((guided - latent).flatten(1), dim=-1)
        ratio = ratio / torch.linalg.vector_norm(scheduler_update.flatten(1), dim=-1)
        self.assertTrue(torch.all(ratio <= 0.1 + 1e-6))

    def test_bfloat16_cast_overrun_is_explicit_dtype_guard(self):
        torch.manual_seed(21)
        dtype = torch.bfloat16
        latent = (torch.randn(2, 4, 8, 8) * 10.0).to(dtype)
        scheduler_update = (torch.randn_like(latent) * 10.0).to(dtype)
        raw = (torch.randn(2, 4, 8, 8) * 100.0).to(dtype)
        tangent = _project_fixed_moment_tangent(latent, raw, 1e-6)
        bounded = cap_update_norm(tangent, scheduler_update, 0.1)
        guided = _fixed_moment_geodesic(latent, bounded, 1e-6)
        denominator = torch.linalg.vector_norm(
            scheduler_update.float().flatten(1), dim=-1
        )
        float_ratio = torch.linalg.vector_norm(
            (guided - latent.float()).flatten(1), dim=-1
        ) / denominator
        cast_ratio = torch.linalg.vector_norm(
            (guided.to(dtype) - latent).float().flatten(1), dim=-1
        ) / denominator
        self.assertTrue(torch.all(float_ratio <= 0.1 + 1e-6))
        self.assertGreater(float(cast_ratio.max()), 0.1002)

    def test_strict_cap_corrects_one_of_two_low_precision_samples(self):
        for dtype in (torch.float16, torch.bfloat16):
            latent, _update, scheduler_update, result = self._deterministic_cast_case(
                dtype, preserve_moments=True, enforce=True
            )
            _guided_float, guided_x0, observed_overrun, cap_applied, fallback = result
            denominator = torch.linalg.vector_norm(
                scheduler_update.float().flatten(1), dim=-1
            )
            ratio = torch.linalg.vector_norm(
                (guided_x0.float() - latent.float()).flatten(1), dim=-1
            ) / denominator.clamp_min(1e-6)
            self.assertGreater(float(observed_overrun[0]), 0.0)
            self.assertEqual(float(observed_overrun[1]), 0.0)
            self.assertTrue(torch.all(ratio <= 0.005 + 1e-6))
            self.assertTrue(bool(cap_applied[0]))
            self.assertFalse(bool(cap_applied[1]))
            self.assertFalse(bool(fallback[1]))
            if bool(fallback[0]):
                self.assertTrue(torch.equal(guided_x0[0], latent[0]))

    def test_strict_cap_zero_scheduler_and_zero_initialization_are_identity(self):
        torch.manual_seed(29)
        latent = torch.randn(2, 4, 8, 8).bfloat16()
        bases = torch.randn(2, 4, 4, 8, 8).bfloat16()
        zero_scheduler = torch.zeros_like(latent)
        renderer = StructuralLatentRenderer(
            LatentRendererConfig(
                num_bases=4,
                max_update_ratio=0.1,
                enforce_post_cast_cap=True,
            )
        )
        output = renderer(
            latent,
            bases,
            scheduler_update=zero_scheduler,
        )
        self.assertTrue(torch.equal(output.guided_x0, latent))
        self.assertTrue(torch.equal(output.residual, torch.zeros_like(latent)))
        self.assertTrue(torch.equal(output.diagnostics.update_ratio, torch.zeros(2)))
        self.assertTrue(torch.equal(output.diagnostics.postcast_overrun, torch.zeros(2)))

        renderer.policy[-1].bias.data.fill_(0.7)
        output = renderer(latent, bases, scheduler_update=zero_scheduler)
        self.assertTrue(torch.equal(output.guided_x0, latent))
        self.assertTrue(torch.equal(output.residual, torch.zeros_like(latent)))

    def test_strict_fixed_moment_errors_and_builder_flag(self):
        latent, _scheduler_update, output = self._renderer_case(
            torch.float32, 31, strict=True, preserve_moments=True
        )
        self.assertTrue(torch.all(output.diagnostics.mean_error < 1e-5))
        self.assertTrue(torch.all(output.diagnostics.variance_error < 1e-4))
        renderer = build_fixed_coefficient_renderer(
            [0.02, 0.0, 0.0, 0.0],
            max_update_ratio=0.1,
            enforce_post_cast_cap=True,
        )
        self.assertTrue(renderer.config.enforce_post_cast_cap)

    def test_strict_cap_also_applies_without_moment_retraction(self):
        for dtype in (torch.float16, torch.bfloat16):
            latent, _update, scheduler_update, result = self._deterministic_cast_case(
                dtype, preserve_moments=False, enforce=True
            )
            _guided_float, guided_x0, _observed, _applied, _fallback = result
            ratio = torch.linalg.vector_norm(
                (guided_x0.float() - latent.float()).flatten(1), dim=-1
            ) / torch.linalg.vector_norm(
                scheduler_update.float().flatten(1), dim=-1
            ).clamp_min(1e-6)
            self.assertTrue(torch.all(ratio <= 0.005 + 1e-6))

    def test_non_strict_mode_preserves_historical_post_cast_action(self):
        latent, update, _scheduler_update, result = self._deterministic_cast_case(
            torch.bfloat16, preserve_moments=False, enforce=False
        )
        _guided_float, guided_x0, observed_overrun, _cap_applied, _fallback = result
        expected = (latent.float() + update.float()).to(latent.dtype)
        self.assertTrue(torch.equal(guided_x0, expected))
        self.assertGreater(float(observed_overrun[0]), 0.0)
        self.assertEqual(float(observed_overrun[1]), 0.0)

    def test_diagnostics_record_is_additive_and_preserves_boolean_flags(self):
        _latent, _scheduler_update, output = self._renderer_case(
            torch.bfloat16, 2094, strict=True
        )
        record = output.diagnostics.to_record()
        for key in (
            "update_ratio",
            "precast_update_ratio",
            "postcast_update_ratio",
            "precast_overrun",
            "postcast_overrun",
            "observed_postcast_overrun",
        ):
            self.assertIn(key, record)
        self.assertIsInstance(record["postcast_cap_applied"][0], bool)
        self.assertIsInstance(record["postcast_noop_fallback"][0], bool)

    def test_cfg_helpers_and_capture_select_positive_block_for_batch_two(self):
        latents = torch.tensor([[10.0], [20.0]])
        expanded = expand_cfg_latents(latents, enabled=True)
        torch.testing.assert_close(
            expanded, torch.tensor([[10.0], [20.0], [10.0], [20.0]])
        )
        negative, positive = split_cfg_noise_pred(
            torch.tensor([[100.0], [200.0], [300.0], [400.0]])
        )
        torch.testing.assert_close(negative, torch.tensor([[100.0], [200.0]]))
        torch.testing.assert_close(positive, torch.tensor([[300.0], [400.0]]))

        capture = object.__new__(StructuralUNetFeatureCapture)
        capture.batch_size = 2
        capture.do_classifier_free_guidance = True
        rows = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        torch.testing.assert_close(
            capture._select_cfg_rows(rows, "synthetic feature"),
            torch.tensor([[2.0], [3.0]]),
        )

        branch_ids = torch.tensor([[10.0], [20.0]])
        torch.testing.assert_close(
            expand_cfg_time_ids(branch_ids, batch_size=2, enabled=True),
            torch.tensor([[10.0], [10.0], [20.0], [20.0]]),
        )


if __name__ == "__main__":
    unittest.main()
