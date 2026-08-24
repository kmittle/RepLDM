import unittest

import torch

from AttentionGuidance import InjectionOutput, inject_rendered_clean_update


class LatentRendererInjectionTest(unittest.TestCase):
    @staticmethod
    def _inputs(dtype=torch.float32, batch=2):
        prev = torch.zeros(batch, 4, 4, 4, dtype=dtype)
        pred = torch.zeros_like(prev)
        guided = torch.ones_like(prev)
        scheduler_update = torch.ones_like(prev)
        return prev, pred, guided, scheduler_update

    def test_default_call_keeps_historical_expression(self):
        prev, pred, guided, _scheduler_update = self._inputs(torch.float16)
        expected = (prev + guided - pred).to(prev.dtype)
        actual = inject_rendered_clean_update(prev, pred, guided)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.dtype, prev.dtype)

    def test_strict_cap_is_per_sample_for_half_precisions(self):
        for dtype in (torch.float16, torch.bfloat16):
            prev, pred, guided, scheduler_update = self._inputs(dtype, batch=2)
            result = inject_rendered_clean_update(
                prev,
                pred,
                guided,
                scheduler_update=scheduler_update,
                max_update_ratio=0.1,
                enforce_post_cast_cap=True,
                return_diagnostics=True,
            )
            self.assertIsInstance(result, InjectionOutput)
            self.assertEqual(result.sample.dtype, dtype)
            diagnostics = result.diagnostics
            self.assertEqual(diagnostics.postcast_update_ratio.shape, (2,))
            self.assertTrue(torch.all(diagnostics.postcast_update_ratio <= 0.1 + 1e-5))
            self.assertTrue(torch.all(diagnostics.postcast_overrun <= 1e-7))
            self.assertTrue(torch.all(diagnostics.postcast_cap_applied))

    def test_zero_scheduler_update_is_exact_identity(self):
        prev = torch.randn(2, 4, 4, 4, dtype=torch.bfloat16)
        pred = torch.zeros_like(prev)
        guided = torch.ones_like(prev)
        result = inject_rendered_clean_update(
            prev,
            pred,
            guided,
            scheduler_update=torch.zeros_like(prev),
            max_update_ratio=0.1,
            enforce_post_cast_cap=True,
            return_diagnostics=True,
        )
        self.assertTrue(torch.equal(result.sample, prev))
        torch.testing.assert_close(
            result.diagnostics.postcast_update_ratio, torch.zeros(2)
        )

    def test_zero_render_delta_is_exact_identity_in_low_precision(self):
        prev = torch.ones(2, 1, 1, 1, dtype=torch.float16)
        rendered = torch.full_like(prev, 0.1)
        result = inject_rendered_clean_update(
            prev,
            rendered,
            rendered,
            scheduler_update=torch.ones_like(prev),
            max_update_ratio=0.1,
            enforce_post_cast_cap=True,
            return_diagnostics=True,
        )
        self.assertTrue(torch.equal(result.sample, prev))
        torch.testing.assert_close(
            result.diagnostics.postcast_residual_norm, torch.zeros(2)
        )

    def test_nonfinite_candidate_uses_exact_noop_fallback(self):
        prev, pred, guided, scheduler_update = self._inputs(torch.float32)
        guided[0, 0, 0, 0] = float("inf")
        result = inject_rendered_clean_update(
            prev,
            pred,
            guided,
            scheduler_update=scheduler_update,
            max_update_ratio=0.1,
            enforce_post_cast_cap=True,
            return_diagnostics=True,
        )
        self.assertTrue(torch.equal(result.sample[0], prev[0]))
        self.assertTrue(bool(result.diagnostics.postcast_noop_fallback[0]))
        self.assertFalse(bool(result.diagnostics.postcast_noop_fallback[1]))
        self.assertTrue(torch.isfinite(result.sample).all())

    def test_diagnostics_record_encodes_flags_as_json_booleans(self):
        prev, pred, guided, scheduler_update = self._inputs(torch.float16)
        result = inject_rendered_clean_update(
            prev,
            pred,
            guided,
            scheduler_update=scheduler_update,
            max_update_ratio=0.1,
            enforce_post_cast_cap=True,
            return_diagnostics=True,
        )
        record = result.diagnostics.to_record()
        self.assertIsInstance(record["postcast_cap_applied"][0], bool)
        self.assertIsInstance(record["postcast_noop_fallback"][0], bool)
        self.assertIn("observed_postcast_overrun", record)


if __name__ == "__main__":
    unittest.main()
