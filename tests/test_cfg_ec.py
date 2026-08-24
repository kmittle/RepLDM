import math
import unittest
from unittest import mock

import torch

from AttentionGuidance.cfg_ec import CFGECConfig, correct_cfg_prediction


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
        with mock.patch.object(torch, "randn", side_effect=AssertionError("unexpected RNG draw")):
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
        self._assert_record_finite(record)
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
