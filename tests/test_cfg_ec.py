import math
import unittest
from unittest import mock

import torch

from AttentionGuidance.cfg_ec import (
    CFGECConfig,
    CFGECDiagnostics,
    correct_cfg_prediction,
    correct_cfg_prediction_sigma,
)


class CFGECProxyTest(unittest.TestCase):
    """CPU-only contract tests for the unintegrated CFG-OEC proxy."""

    def setUp(self):
        # Each row is an independent two-dimensional prediction.  The first
        # row has orthogonal proxy errors (cos=0); the second has a positive
        # but non-unit alignment.  A threshold of .99 activates both rows.
        self.current_u = torch.tensor(
            [
                [[[1.0, 1.0]]],
                [[[2.0, 1.0]]],
            ],
            dtype=torch.float32,
        )
        self.current_c = torch.tensor(
            [
                [[[0.0, 1.0]]],
                [[[1.0, 0.0]]],
            ],
            dtype=torch.float32,
        )
        self.previous_u = torch.tensor(
            [
                [[[0.0, 0.0]]],
                [[[0.5, -0.5]]],
            ],
            dtype=torch.float32,
        )
        self.previous_c = torch.tensor(
            [
                [[[0.0, 0.0]]],
                [[[0.25, 0.25]]],
            ],
            dtype=torch.float32,
        )
        self.config = CFGECConfig(
            guidance_scale=2.0,
            alignment_threshold=0.99,
            blend=1.0,
        )

    @staticmethod
    def _ordinary_cfg(unconditional, conditional, guidance_scale=2.0):
        return unconditional + (conditional - unconditional) * guidance_scale

    @staticmethod
    def _assert_record_finite(record):
        for key, value in record.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, (int, float)):
                        assert math.isfinite(float(item)), (key, item)
            elif isinstance(value, (int, float)):
                assert math.isfinite(float(value)), (key, value)

    def test_b1_correction_is_finite_and_history_valid(self):
        output, diagnostics = correct_cfg_prediction(
            self.current_u[:1],
            self.current_c[:1],
            self.previous_u[:1],
            self.previous_c[:1],
            current_time=0.0,
            previous_time=1.0,
            config=self.config,
        )
        baseline = self._ordinary_cfg(self.current_u[:1], self.current_c[:1])
        self.assertEqual(output.shape, self.current_u[:1].shape)
        self.assertEqual(output.dtype, self.current_u.dtype)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(diagnostics.history_valid)
        self.assertEqual(diagnostics.applied_rows, (True,))
        self.assertFalse(torch.equal(output, baseline))
        self._assert_record_finite(diagnostics.to_record())

    def test_cfg_config_preserves_legacy_positional_order(self):
        # Before the sigma variant, positional arguments four through six
        # were allow_normalized_time_proxy, time_tolerance, and
        # projection_epsilon.  New guards must remain trailing fields.
        legacy = CFGECConfig(2.0, 0.99, 1.0, True, 0.25, 1e-7)
        self.assertTrue(legacy.allow_normalized_time_proxy)
        self.assertEqual(legacy.time_tolerance, 0.25)
        self.assertEqual(legacy.projection_epsilon, 1e-7)
        self.assertEqual(legacy.max_extrapolation_ratio, 4.0)
        self.assertEqual(legacy.relative_time_tolerance, 1e-6)

    def test_diagnostics_preserve_pre_sigma_positional_order(self):
        legacy = CFGECDiagnostics(
            False,
            (False,),
            (False,),
            (0.0,),
            (0.0,),
            (0.0,),
            0.0,
            (0.0,),
            "no_history",
        )
        self.assertEqual(legacy.negative_alignment_rows, (False,))
        self.assertEqual(legacy.extrapolation_ratio, 1.0)

    def test_b2_rows_are_independent(self):
        batched, batched_diag = correct_cfg_prediction(
            self.current_u,
            self.current_c,
            self.previous_u,
            self.previous_c,
            current_time=0.0,
            previous_time=1.0,
            config=self.config,
        )
        for row in range(2):
            single, single_diag = correct_cfg_prediction(
                self.current_u[row : row + 1],
                self.current_c[row : row + 1],
                self.previous_u[row : row + 1],
                self.previous_c[row : row + 1],
                current_time=0.0,
                previous_time=1.0,
                config=self.config,
            )
            torch.testing.assert_close(batched[row : row + 1], single)
            self.assertEqual(batched_diag.applied_rows[row], single_diag.applied_rows[0])
            self.assertAlmostEqual(
                batched_diag.alignment_cosine[row], single_diag.alignment_cosine[0], places=6
            )

    def test_history_shuffle_changes_only_the_corresponding_rows(self):
        original, _ = correct_cfg_prediction(
            self.current_u,
            self.current_c,
            self.previous_u,
            self.previous_c,
            current_time=0.0,
            previous_time=1.0,
            config=self.config,
        )
        shuffled, shuffled_diag = correct_cfg_prediction(
            self.current_u,
            self.current_c,
            self.previous_u.flip(0),
            self.previous_c.flip(0),
            current_time=0.0,
            previous_time=1.0,
            config=self.config,
        )
        # Distinct previous rows must not be silently broadcast or ignored.
        self.assertFalse(torch.equal(original[0], shuffled[0]))
        self.assertFalse(torch.equal(original[1], shuffled[1]))
        self.assertEqual(shuffled_diag.applied_rows, (True, True))

        # The shuffled batch is exactly equivalent to two independently
        # evaluated rows with the corresponding shuffled history.
        expected0, _ = correct_cfg_prediction(
            self.current_u[:1],
            self.current_c[:1],
            self.previous_u[1:2],
            self.previous_c[1:2],
            current_time=0.0,
            previous_time=1.0,
            config=self.config,
        )
        expected1, _ = correct_cfg_prediction(
            self.current_u[1:2],
            self.current_c[1:2],
            self.previous_u[:1],
            self.previous_c[:1],
            current_time=0.0,
            previous_time=1.0,
            config=self.config,
        )
        torch.testing.assert_close(shuffled[:1], expected0)
        torch.testing.assert_close(shuffled[1:2], expected1)

    def test_first_step_without_history_is_exact_cfg_noop(self):
        before = torch.get_rng_state().clone()
        output, diagnostics = correct_cfg_prediction(
            self.current_u[:1],
            self.current_c[:1],
            None,
            None,
            current_time=None,
            previous_time=None,
            config=self.config,
        )
        after = torch.get_rng_state()
        baseline = self._ordinary_cfg(self.current_u[:1], self.current_c[:1])
        self.assertTrue(torch.equal(output, baseline))
        self.assertTrue(torch.equal(before, after))
        self.assertFalse(diagnostics.history_valid)
        self.assertEqual(diagnostics.reason, "no_history")
        self.assertEqual(diagnostics.applied_rows, (False,))

    def test_proxy_has_no_random_draw_or_extra_call_hook(self):
        # The interface has no scheduler/model argument and must remain a
        # pure tensor transform.  Patching the only common random entry point
        # makes an accidental intervention draw fail loudly.
        with mock.patch.object(
            torch, "randn", side_effect=AssertionError("unexpected RNG draw")
        ):
            output, diagnostics = correct_cfg_prediction(
                self.current_u[:1],
                self.current_c[:1],
                self.previous_u[:1],
                self.previous_c[:1],
                current_time=0.0,
                previous_time=1.0,
                config=self.config,
            )
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(diagnostics.history_valid)

    def test_zero_blend_is_exact_identity_and_does_not_need_history(self):
        config = CFGECConfig(
            guidance_scale=2.0,
            alignment_threshold=0.99,
            blend=0.0,
        )
        before = torch.get_rng_state().clone()
        output, diagnostics = correct_cfg_prediction(
            self.current_u,
            self.current_c,
            self.previous_u,
            self.previous_c,
            current_time=123.0,
            previous_time=-456.0,
            config=config,
        )
        baseline = self._ordinary_cfg(self.current_u, self.current_c)
        self.assertTrue(torch.equal(output, baseline))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertFalse(diagnostics.history_valid)
        self.assertEqual(diagnostics.reason, "zero_blend")
        self.assertEqual(diagnostics.applied_rows, (False, False))
        self._assert_record_finite(diagnostics.to_record())

    def test_nonunit_time_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "non-unit"):
            correct_cfg_prediction(
                self.current_u[:1],
                self.current_c[:1],
                self.previous_u[:1],
                self.previous_c[:1],
                current_time=0.25,
                previous_time=0.75,
                config=self.config,
            )

        opted_in = CFGECConfig(
            guidance_scale=2.0,
            alignment_threshold=0.99,
            blend=1.0,
            allow_normalized_time_proxy=True,
        )
        output, diagnostics = correct_cfg_prediction(
            self.current_u[:1],
            self.current_c[:1],
            self.previous_u[:1],
            self.previous_c[:1],
            current_time=0.25,
            previous_time=0.75,
            config=opted_in,
        )
        self.assertTrue(torch.isfinite(output).all())
        self.assertAlmostEqual(diagnostics.time_delta, -0.5)

    def test_sigma_equal_spacing_degenerates_to_normalized_proxy(self):
        sigma_output, sigma_diag = correct_cfg_prediction_sigma(
            self.current_u,
            self.current_c,
            self.previous_u,
            self.previous_c,
            previous_sigma=2.0,
            current_sigma=1.0,
            next_sigma=0.0,
            config=self.config,
        )
        normalized_output, normalized_diag = correct_cfg_prediction(
            self.current_u,
            self.current_c,
            self.previous_u,
            self.previous_c,
            current_time=0.0,
            previous_time=1.0,
            config=self.config,
        )
        torch.testing.assert_close(sigma_output, normalized_output)
        self.assertEqual(sigma_diag.extrapolation_ratio, 1.0)
        self.assertEqual(normalized_diag.extrapolation_ratio, 1.0)
        self.assertEqual(sigma_diag.applied_rows, normalized_diag.applied_rows)
        self._assert_record_finite(sigma_diag.to_record())

    def test_sigma_nonuniform_horizon_matches_synthetic_linear_trajectory(self):
        # For a linear prediction p(sigma), the registered ratio recovers the
        # exact prediction at next_sigma even when the gaps are unequal.
        previous_sigma, current_sigma, next_sigma = 3.0, 2.0, 0.5
        previous_u = torch.tensor([[[[3.0, 2.5]]]])
        current_u = torch.tensor([[[[2.0, 2.0]]]])
        previous_c = torch.tensor([[[[7.0, 1.0]]]])
        current_c = torch.tensor([[[[5.0, 1.0]]]])
        expected_proxy_u = torch.tensor([[[[0.5, 1.25]]]])
        expected_proxy_c = torch.tensor([[[[2.0, 1.0]]]])
        ratio = (current_sigma - next_sigma) / (previous_sigma - current_sigma)
        self.assertEqual(ratio, 1.5)

        sigma_output, sigma_diag = correct_cfg_prediction_sigma(
            current_u,
            current_c,
            previous_u,
            previous_c,
            previous_sigma=previous_sigma,
            current_sigma=current_sigma,
            next_sigma=next_sigma,
            config=self.config,
        )

        # Feed the same extrapolated proxies through the unit-step API by
        # constructing the equivalent synthetic previous pair.
        synthetic_previous_u = 2.0 * current_u - expected_proxy_u
        synthetic_previous_c = 2.0 * current_c - expected_proxy_c
        reference_output, reference_diag = correct_cfg_prediction(
            current_u,
            current_c,
            synthetic_previous_u,
            synthetic_previous_c,
            current_time=0.0,
            previous_time=1.0,
            config=self.config,
        )
        torch.testing.assert_close(sigma_output, reference_output)
        self.assertAlmostEqual(sigma_diag.extrapolation_ratio, ratio)
        self.assertAlmostEqual(
            sigma_diag.alignment_cosine[0], reference_diag.alignment_cosine[0], places=6
        )
        self.assertEqual(sigma_diag.applied_rows, (True,))
        self.assertTrue(torch.isfinite(sigma_output).all())

    def test_sigma_batch_rows_are_independent(self):
        batched, batched_diag = correct_cfg_prediction_sigma(
            self.current_u,
            self.current_c,
            self.previous_u,
            self.previous_c,
            previous_sigma=3.0,
            current_sigma=2.0,
            next_sigma=0.5,
            config=self.config,
        )
        for row in range(2):
            single, single_diag = correct_cfg_prediction_sigma(
                self.current_u[row : row + 1],
                self.current_c[row : row + 1],
                self.previous_u[row : row + 1],
                self.previous_c[row : row + 1],
                previous_sigma=3.0,
                current_sigma=2.0,
                next_sigma=0.5,
                config=self.config,
            )
            torch.testing.assert_close(batched[row : row + 1], single)
            self.assertEqual(batched_diag.applied_rows[row], single_diag.applied_rows[0])
            self.assertAlmostEqual(
                batched_diag.extrapolation_ratio, single_diag.extrapolation_ratio
            )

    def test_sigma_boundaries_and_nonfinite_values_are_rejected(self):
        kwargs = dict(
            current_unconditional=self.current_u[:1],
            current_conditional=self.current_c[:1],
            previous_unconditional=self.previous_u[:1],
            previous_conditional=self.previous_c[:1],
            config=self.config,
        )
        invalid_triplets = (
            (1.0, 2.0, 0.0),  # previous is not greater than current
            (2.0, 1.0, 1.0),  # zero next gap
            (2.0, 1.0, 1.1),  # next is greater than current
            (-1.0, 0.5, 0.0),  # negative sigma
            (float("nan"), 1.0, 0.0),
            (2.0, float("inf"), 0.0),
            (2.0, 1.0, 0.999999),  # absolute + relative gap tolerance
        )
        for previous_sigma, current_sigma, next_sigma in invalid_triplets:
            with self.subTest(
                sigmas=(previous_sigma, current_sigma, next_sigma)
            ):
                with self.assertRaises(ValueError):
                    correct_cfg_prediction_sigma(
                        **kwargs,
                        previous_sigma=previous_sigma,
                        current_sigma=current_sigma,
                        next_sigma=next_sigma,
                    )

        relative_guard = CFGECConfig(
            2.0,
            0.99,
            1.0,
            relative_time_tolerance=0.1,
            time_tolerance=0.0,
        )
        with self.assertRaises(ValueError):
            correct_cfg_prediction_sigma(
                **{**kwargs, "config": relative_guard},
                previous_sigma=1.0,
                current_sigma=0.95,
                next_sigma=0.9,
            )

        ratio_guard = CFGECConfig(2.0, 0.99, 1.0, max_extrapolation_ratio=1.0)
        with self.assertRaisesRegex(ValueError, "max_extrapolation_ratio"):
            correct_cfg_prediction_sigma(
                **{**kwargs, "config": ratio_guard},
                previous_sigma=3.0,
                current_sigma=2.0,
                next_sigma=0.5,
            )

        with self.assertRaises(ValueError):
            CFGECConfig(2.0, 0.99, 1.0, max_extrapolation_ratio=0.0)
        with self.assertRaises(ValueError):
            CFGECConfig(2.0, 0.99, 1.0, max_extrapolation_ratio=0.5)
        with self.assertRaises(ValueError):
            CFGECConfig(2.0, 0.99, 1.0, relative_time_tolerance=-1.0)

    def test_sigma_no_history_and_zero_blend_skip_sigma_validation(self):
        baseline = self._ordinary_cfg(self.current_u[:1], self.current_c[:1])
        no_history_output, no_history_diag = correct_cfg_prediction_sigma(
            self.current_u[:1],
            self.current_c[:1],
            None,
            None,
            previous_sigma=None,
            current_sigma=None,
            next_sigma=None,
            config=self.config,
        )
        torch.testing.assert_close(no_history_output, baseline)
        self.assertFalse(no_history_diag.history_valid)
        self.assertEqual(no_history_diag.extrapolation_ratio, 0.0)

        zero_config = CFGECConfig(2.0, 0.99, 0.0)
        zero_output, zero_diag = correct_cfg_prediction_sigma(
            self.current_u[:1],
            self.current_c[:1],
            self.previous_u[:1],
            self.previous_c[:1],
            previous_sigma=float("nan"),
            current_sigma=2.0,
            next_sigma=1.0,
            config=zero_config,
        )
        torch.testing.assert_close(zero_output, baseline)
        self.assertFalse(zero_diag.history_valid)
        self.assertEqual(zero_diag.extrapolation_ratio, 0.0)

    def test_negative_proxy_alignment_is_skipped_and_reported(self):
        # A and B point in opposite directions.  Applying the paper's raw
        # negative cosine mix would extrapolate; the registered smoke policy
        # keeps this row at ordinary CFG and records the reason explicitly.
        current_u = torch.tensor([[[[0.0, 0.0]]]])
        current_c = torch.tensor([[[[1.0, 0.0]]]])
        previous_u = torch.tensor([[[[1.0, 0.0]]]])
        previous_c = torch.tensor([[[[0.0, 0.0]]]])
        output, diagnostics = correct_cfg_prediction(
            current_u,
            current_c,
            previous_u,
            previous_c,
            current_time=0.0,
            previous_time=1.0,
            config=CFGECConfig(2.0, 0.99, 1.0),
        )
        torch.testing.assert_close(output, self._ordinary_cfg(current_u, current_c))
        self.assertEqual(diagnostics.applied_rows, (False,))
        self.assertEqual(diagnostics.negative_alignment_rows, (True,))
        self.assertEqual(diagnostics.reason, "negative_alignment")
        self._assert_record_finite(diagnostics.to_record())

    def test_time_order_and_partial_history_are_rejected(self):
        args = dict(
            current_unconditional=self.current_u[:1],
            current_conditional=self.current_c[:1],
            current_time=0.0,
            previous_time=1.0,
            config=self.config,
        )
        with self.assertRaises(ValueError):
            correct_cfg_prediction(
                args["current_unconditional"],
                args["current_conditional"],
                self.previous_u[:1],
                None,
                current_time=args["current_time"],
                previous_time=args["previous_time"],
                config=args["config"],
            )
        with self.assertRaises(ValueError):
            correct_cfg_prediction(
                args["current_unconditional"],
                args["current_conditional"],
                self.previous_u[:1],
                self.previous_c[:1],
                current_time=1.0,
                previous_time=0.0,
                config=args["config"],
            )
        with self.assertRaises(ValueError):
            correct_cfg_prediction(
                args["current_unconditional"],
                args["current_conditional"],
                self.previous_u[:1],
                self.previous_c[:1],
                current_time=0.0,
                previous_time=0.0,
                config=args["config"],
            )

    def test_shape_dtype_device_and_finite_checks(self):
        common = dict(
            current_time=0.0,
            previous_time=1.0,
            config=self.config,
        )
        with self.assertRaises(ValueError):
            correct_cfg_prediction(
                self.current_u,
                self.current_c[:, :, :, :1],
                self.previous_u,
                self.previous_c,
                **common,
            )
        with self.assertRaises(ValueError):
            correct_cfg_prediction(
                self.current_u,
                self.current_c,
                self.previous_u[:1],
                self.previous_c[:1],
                **common,
            )
        with self.assertRaises(ValueError):
            correct_cfg_prediction(
                self.current_u,
                self.current_c,
                self.previous_u.double(),
                self.previous_c.double(),
                **common,
            )
        with self.assertRaises(ValueError):
            correct_cfg_prediction(
                self.current_u,
                self.current_c,
                self.previous_u,
                self.previous_c,
                current_time=float("nan"),
                previous_time=1.0,
                config=self.config,
            )
        bad_current = self.current_u.clone()
        bad_current[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            correct_cfg_prediction(
                bad_current,
                self.current_c,
                self.previous_u,
                self.previous_c,
                **common,
            )
        with self.assertRaises(TypeError):
            correct_cfg_prediction(
                self.current_u.to(dtype=torch.int64),
                self.current_c.to(dtype=torch.int64),
                self.previous_u.to(dtype=torch.int64),
                self.previous_c.to(dtype=torch.int64),
                **common,
            )

        # Meta is available in CPU-only PyTorch and lets us exercise the
        # cross-device rejection without requiring a CUDA worker.
        meta_previous = torch.empty_like(self.previous_u, device="meta")
        meta_conditional = torch.empty_like(self.previous_c, device="meta")
        with self.assertRaises(ValueError):
            correct_cfg_prediction(
                self.current_u,
                self.current_c,
                meta_previous,
                meta_conditional,
                **common,
            )

    def test_diagnostics_are_rowwise_and_finite(self):
        output, diagnostics = correct_cfg_prediction(
            self.current_u,
            self.current_c,
            self.previous_u,
            self.previous_c,
            current_time=0.0,
            previous_time=1.0,
            config=self.config,
        )
        record = diagnostics.to_record()
        self.assertEqual(len(record["alignment_cosine"]), 2)
        self.assertEqual(len(record["correction_norm_ratio"]), 2)
        self.assertEqual(len(record["effective_blend"]), 2)
        self.assertEqual(record["extrapolation_ratio"], 1.0)
        self._assert_record_finite(record)
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
