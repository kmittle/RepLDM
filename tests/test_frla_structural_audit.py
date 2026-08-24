import importlib.util
from pathlib import Path
import unittest

import torch

from AttentionGuidance.latent_renderer import (
    StructuralUNetFeatureCapture,
    _fixed_moment_geodesic,
    _project_fixed_moment_tangent,
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
