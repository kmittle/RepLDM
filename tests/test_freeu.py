import unittest
from unittest import mock

import torch
from diffusers.models.unets import unet_2d_blocks

from AttentionGuidance import (
    DIFFUSERS_FREEU_IMPLEMENTATION,
    PAPER_FREEU_IMPLEMENTATION,
    PAPER_FREEU_SDXL_PARAMETERS,
    PAPER_FREEU_PORT_DIFFUSERS_VERSION,
    PAPER_FREEU_SOURCE_COMMIT,
    FreeUParameters,
    FreeUSchedule,
    apply_paper_freeu,
    installed_freeu_implementation,
    match_channel_moments,
)


class _FakeUNet:
    def __init__(self):
        self.calls = []
        self.disabled = 0

    def enable_freeu(self, s1, s2, b1, b2):
        self.calls.append((s1, s2, b1, b2))

    def disable_freeu(self):
        self.disabled += 1


class FreeUTest(unittest.TestCase):
    def test_constant_schedule_is_reentrant_and_interpolates(self):
        schedule = FreeUSchedule.constant((0.6, 0.4, 1.1, 1.2))
        model = _FakeUNet()
        first = schedule.apply(model, 0, 3)
        last = schedule.apply(model, 2, 3)
        self.assertEqual(first.as_tuple(), last.as_tuple())
        self.assertEqual(len(model.calls), 2)
        schedule.disable(model)
        self.assertEqual(model.disabled, 1)

        dynamic = FreeUSchedule(
            (
                (0.0, (1.0, 1.0, 1.0, 1.0)),
                (1.0, (0.5, 0.25, 1.5, 2.0)),
            )
        )
        middle = dynamic.at(0.5)
        self.assertEqual(middle.as_tuple(), (0.75, 0.625, 1.25, 1.5))

    def test_schedule_requires_endpoints_and_validates_ranges(self):
        with self.assertRaises(ValueError):
            FreeUSchedule(((0.5, (1.0, 1.0, 1.0, 1.0)),))
        with self.assertRaises(ValueError):
            FreeUParameters.from_sequence((1.0, 1.0, 1.0))
        with self.assertRaises(ValueError):
            FreeUParameters(3.0, 1.0, 1.0, 1.0)

    def test_step_bounds_are_checked(self):
        schedule = FreeUSchedule.constant((1.0, 1.0, 1.0, 1.0))
        with self.assertRaises(ValueError):
            schedule.apply(_FakeUNet(), 3, 3)
        with self.assertRaises(ValueError):
            schedule.apply(_FakeUNet(), 0, 0)

    def test_moment_projection_preserves_reference_channel_statistics(self):
        reference = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4)
        candidate = reference * 3.0 + 11.0
        projected = match_channel_moments(candidate, reference)
        for tensor in (projected, reference):
            self.assertTrue(torch.isfinite(tensor).all())
        self.assertTrue(
            torch.allclose(
                projected.mean(dim=(-2, -1)),
                reference.mean(dim=(-2, -1)),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                projected.std(dim=(-2, -1), unbiased=False),
                reference.std(dim=(-2, -1), unbiased=False),
                atol=1e-6,
            )
        )
        with self.assertRaises(ValueError):
            match_channel_moments(candidate[:, :, :-1], reference)

    @staticmethod
    def _pinned_paper_reference(resolution_idx, hidden, skip, **parameters):
        del resolution_idx
        if hidden.shape[1] == 1280:
            suffix = "1"
        elif hidden.shape[1] == 640:
            suffix = "2"
        else:
            return hidden, skip
        hidden_mean = hidden.mean(1).unsqueeze(1)
        batch = hidden_mean.shape[0]
        hidden_max = hidden_mean.view(batch, -1).max(dim=-1, keepdim=True).values
        hidden_min = hidden_mean.view(batch, -1).min(dim=-1, keepdim=True).values
        hidden_mean = (hidden_mean - hidden_min[:, :, None, None]) / (
            hidden_max - hidden_min
        )[:, :, None, None]
        half = hidden.shape[1] // 2
        scaled = hidden.clone()
        scaled[:, :half] = scaled[:, :half] * (
            (parameters[f"b{suffix}"] - 1) * hidden_mean + 1
        )
        skip_frequency = torch.fft.fftshift(
            torch.fft.fftn(skip, dim=(-2, -1)), dim=(-2, -1)
        )
        _, _, height, width = skip_frequency.shape
        mask = torch.ones(skip_frequency.shape, device=skip.device)
        center_row, center_column = height // 2, width // 2
        mask[
            ...,
            center_row - 1 : center_row + 1,
            center_column - 1 : center_column + 1,
        ] = parameters[f"s{suffix}"]
        filtered = torch.fft.ifftn(
            torch.fft.ifftshift(skip_frequency * mask, dim=(-2, -1)),
            dim=(-2, -1),
        ).real.to(skip.dtype)
        return scaled, filtered

    def test_paper_freeu_matches_pinned_adaptive_formula(self):
        self.assertEqual(
            PAPER_FREEU_SOURCE_COMMIT,
            "3676d3652a44101f9cca030c33f82756dab249d7",
        )
        self.assertEqual(PAPER_FREEU_SDXL_PARAMETERS, (0.9, 0.2, 1.3, 1.4))
        torch.manual_seed(79)
        parameters = dict(
            zip(("s1", "s2", "b1", "b2"), PAPER_FREEU_SDXL_PARAMETERS)
        )
        for resolution_idx, channels in ((1, 1280), (2, 640), (0, 320)):
            hidden = torch.randn(1, channels, 4, 4)
            skip = torch.randn(1, 5, 4, 4)
            expected_hidden, expected_skip = self._pinned_paper_reference(
                resolution_idx, hidden, skip, **parameters
            )
            actual_hidden, actual_skip = apply_paper_freeu(
                resolution_idx, hidden, skip, **parameters
            )
            torch.testing.assert_close(actual_hidden, expected_hidden, rtol=0, atol=0)
            torch.testing.assert_close(actual_skip, expected_skip, rtol=0, atol=0)

    def test_paper_freeu_zero_range_is_finite_and_leaves_backbone(self):
        hidden = torch.ones(1, 1280, 4, 4)
        skip = torch.randn(1, 4, 4, 4)
        parameters = dict(
            zip(("s1", "s2", "b1", "b2"), PAPER_FREEU_SDXL_PARAMETERS)
        )
        actual_hidden, actual_skip = apply_paper_freeu(
            0, hidden, skip, **parameters
        )
        self.assertTrue(torch.equal(actual_hidden, hidden))
        self.assertTrue(torch.isfinite(actual_skip).all())

    def test_paper_freeu_preserves_nonzero_subnormal_range_formula(self):
        parameters = dict(zip(("s1", "s2", "b1", "b2"), PAPER_FREEU_SDXL_PARAMETERS))
        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                value = torch.finfo(dtype).tiny / 2
                hidden = torch.zeros(1, 1280, 2, 2, dtype=dtype)
                hidden[:, :, 0, 0] = value
                skip = torch.randn(1, 4, 2, 2, dtype=dtype)
                spatial_mean = hidden.mean(dim=1, keepdim=True)
                spatial_min = spatial_mean.flatten(1).amin(dim=1).reshape(-1, 1, 1, 1)
                spatial_range = (
                    spatial_mean.flatten(1).amax(dim=1).reshape(-1, 1, 1, 1)
                    - spatial_min
                )
                normalized = (spatial_mean - spatial_min) / spatial_range
                expected_hidden = hidden.clone()
                expected_hidden[:, :640] *= 1.0 + (parameters["b1"] - 1.0) * normalized
                with mock.patch(
                    "diffusers.utils.torch_utils.fourier_filter",
                    side_effect=lambda tensor, **_: tensor,
                ):
                    actual_hidden, actual_skip = apply_paper_freeu(
                        2, hidden, skip, **parameters
                    )
                torch.testing.assert_close(actual_hidden, expected_hidden, rtol=0, atol=0)
                self.assertIs(actual_skip, skip)

    def test_paper_freeu_context_restores_diffusers_operator(self):
        original = unet_2d_blocks.apply_freeu
        parameters = dict(
            zip(("s1", "s2", "b1", "b2"), PAPER_FREEU_SDXL_PARAMETERS)
        )
        hidden = torch.randn(1, 1280, 4, 4)
        skip = torch.randn(1, 4, 4, 4)
        with installed_freeu_implementation(PAPER_FREEU_IMPLEMENTATION) as runtime:
            self.assertIsNot(unet_2d_blocks.apply_freeu, original)
            unet_2d_blocks.apply_freeu(0, hidden, skip, **parameters)
            self.assertEqual(runtime["operator_calls_total"], 1)
            self.assertEqual(runtime["resolution_idx_call_counts"], {"0": 1})
            self.assertEqual(runtime["hidden_channel_call_counts"], {"1280": 1})
            self.assertEqual(
                runtime["resolution_channel_call_counts"], {"0:1280": 1}
            )
            self.assertEqual(runtime["operator_effect_call_counts"], {"b1_s1": 1})
        self.assertIs(unet_2d_blocks.apply_freeu, original)

        with installed_freeu_implementation(DIFFUSERS_FREEU_IMPLEMENTATION) as runtime:
            self.assertIsNot(unet_2d_blocks.apply_freeu, original)
            unet_2d_blocks.apply_freeu(1, hidden, skip, **parameters)
            self.assertEqual(runtime["operator_calls_total"], 1)
            self.assertEqual(runtime["resolution_idx_call_counts"], {"1": 1})
            self.assertEqual(runtime["operator_effect_call_counts"], {"b2_s2": 1})
        self.assertIs(unet_2d_blocks.apply_freeu, original)

        with mock.patch("diffusers.__version__", "0.33.1"):
            with self.assertRaisesRegex(RuntimeError, "requires diffusers 0.32.1"):
                with installed_freeu_implementation(PAPER_FREEU_IMPLEMENTATION):
                    pass
        self.assertEqual(PAPER_FREEU_PORT_DIFFUSERS_VERSION, "0.32.1")

        with self.assertRaisesRegex(RuntimeError, "parity probe"):
            with installed_freeu_implementation(PAPER_FREEU_IMPLEMENTATION):
                raise RuntimeError("parity probe")
        self.assertIs(unet_2d_blocks.apply_freeu, original)

    def test_paper_freeu_full_sdxl_call_topology_is_channel_dispatched(self):
        parameters = dict(
            zip(("s1", "s2", "b1", "b2"), PAPER_FREEU_SDXL_PARAMETERS)
        )
        calls = (
            [(0, 1280)] * 3
            + [(1, 1280), (1, 640), (1, 640)]
            + [(2, 640), (2, 320), (2, 320)]
        )
        with mock.patch(
            "diffusers.utils.torch_utils.fourier_filter",
            side_effect=lambda tensor, **_: tensor,
        ), installed_freeu_implementation(PAPER_FREEU_IMPLEMENTATION) as runtime:
            for resolution_idx, channels in calls:
                hidden = torch.randn(1, channels, 2, 2)
                skip = torch.randn(1, 4, 2, 2)
                unet_2d_blocks.apply_freeu(
                    resolution_idx, hidden, skip, **parameters
                )
        self.assertEqual(runtime["operator_calls_total"], 9)
        self.assertEqual(
            runtime["resolution_idx_call_counts"], {"0": 3, "1": 3, "2": 3}
        )
        self.assertEqual(
            runtime["hidden_channel_call_counts"],
            {"1280": 4, "640": 3, "320": 2},
        )
        self.assertEqual(
            runtime["resolution_channel_call_counts"],
            {
                "0:1280": 3,
                "1:1280": 1,
                "1:640": 2,
                "2:640": 1,
                "2:320": 2,
            },
        )
        self.assertEqual(
            runtime["operator_effect_call_counts"],
            {"b1_s1": 4, "b2_s2": 3, "no_op": 2},
        )


if __name__ == "__main__":
    unittest.main()
