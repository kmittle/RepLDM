import math
import unittest

import torch

from AttentionGuidance.local_relational_basis import (
    LOCAL_RELATIONAL_OFFSET_ORBITS,
    LOCAL_RELATIONAL_ORBIT_NAMES,
    LocalRelationalBasisProvider,
)


class LocalRelationalBasisProviderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(101)

    def test_shape_order_dtype_and_diagnostics(self):
        grid = 5
        provider = LocalRelationalBasisProvider(
            grid_size=grid,
            feature_norm_epsilon=1e-6,
            affinity_floor=1e-6,
        )
        yy, xx = torch.meshgrid(
            torch.arange(grid), torch.arange(grid), indexing="ij"
        )
        pattern = yy.square() + 10 * xx.square()
        x0 = pattern[None, None].expand(2, 4, -1, -1).half()
        feature = torch.ones(2, 7, grid, grid, dtype=torch.float16)

        bases, diagnostics = provider(x0, feature)

        self.assertEqual(bases.shape, (2, 3, 4, grid, grid))
        self.assertEqual(bases.dtype, torch.float16)
        self.assertEqual(diagnostics.orbit_names, LOCAL_RELATIONAL_ORBIT_NAMES)
        torch.testing.assert_close(
            bases[0, :, 0, 2, 2].float(),
            torch.tensor([5.5, 11.0, 22.0]),
            rtol=0,
            atol=0,
        )
        self.assertEqual(
            diagnostics.undirected_edge_counts,
            (2 * grid * (grid - 1), 2 * (grid - 1) ** 2, 2 * grid * (grid - 2)),
        )
        self.assertTrue(torch.all(diagnostics.edge_weight_min > 0))
        self.assertTrue(torch.all(diagnostics.row_affinity_sum_min > 0))
        torch.testing.assert_close(
            diagnostics.row_probability_sum_min,
            torch.ones_like(diagnostics.row_probability_sum_min),
        )
        torch.testing.assert_close(
            diagnostics.row_probability_sum_max,
            torch.ones_like(diagnostics.row_probability_sum_max),
        )
        record = diagnostics.to_record()
        self.assertEqual(record["orbit_names"], list(LOCAL_RELATIONAL_ORBIT_NAMES))
        self.assertNotIn("temperature", record)
        self.assertNotIn("quality", record)

    def test_constant_clean_latent_has_exact_zero_bases(self):
        provider = LocalRelationalBasisProvider(grid_size=6)
        x0 = torch.full((2, 4, 12, 18), 1.75)
        feature = torch.randn(2, 9, 8, 10)

        bases, diagnostics = provider(x0, feature)

        self.assertTrue(torch.equal(bases, torch.zeros_like(bases)))
        self.assertTrue(
            torch.equal(diagnostics.basis_rms, torch.zeros_like(diagnostics.basis_rms))
        )

    def test_matches_symmetric_row_normalized_dense_graph(self):
        grid = 4
        feature_norm_epsilon = 1e-6
        affinity_floor = 1e-6
        provider = LocalRelationalBasisProvider(
            grid_size=grid,
            feature_norm_epsilon=feature_norm_epsilon,
            affinity_floor=affinity_floor,
        )
        x0 = torch.randn(1, 4, grid, grid)
        feature = torch.randn(1, 3, grid, grid)
        actual, _ = provider(x0, feature)
        feature_unit = feature / torch.linalg.vector_norm(
            feature, dim=1, keepdim=True
        )
        feature_tokens = feature_unit.flatten(2).transpose(1, 2)
        x0_tokens = x0.flatten(2).transpose(1, 2)
        references = []

        for _name, offsets in LOCAL_RELATIONAL_OFFSET_ORBITS:
            adjacency = torch.zeros(1, grid * grid, grid * grid)
            for dy, dx in offsets:
                for y in range(grid):
                    for x in range(grid):
                        target_y = y + dy
                        target_x = x + dx
                        if not (0 <= target_y < grid and 0 <= target_x < grid):
                            continue
                        source = y * grid + x
                        target = target_y * grid + target_x
                        cosine = (
                            feature_tokens[:, source] * feature_tokens[:, target]
                        ).sum(dim=-1).clamp(-1.0, 1.0)
                        weight = affinity_floor + 0.5 * (1.0 + cosine)
                        adjacency[:, source, target] = weight
                        adjacency[:, target, source] = weight
            self.assertTrue(torch.equal(adjacency, adjacency.transpose(-1, -2)))
            transition = adjacency / adjacency.sum(dim=-1, keepdim=True)
            torch.testing.assert_close(
                transition.sum(dim=-1),
                torch.ones(1, grid * grid),
                rtol=0,
                atol=2e-7,
            )
            residual = torch.bmm(transition, x0_tokens) - x0_tokens
            references.append(
                residual.transpose(1, 2).reshape(1, 4, grid, grid)
            )

        torch.testing.assert_close(
            actual, torch.stack(references, dim=1), rtol=2e-5, atol=2e-6
        )

    def test_affinity_matches_registered_cosine_constants(self):
        affinity_floor = 1e-6
        provider = LocalRelationalBasisProvider(
            grid_size=4, affinity_floor=affinity_floor
        )
        x0_tokens = torch.zeros(1, 16, 4)
        feature_tokens = torch.zeros(1, 16, 2)
        feature_tokens[..., 0] = 1.0
        feature_tokens[0, 1] = torch.tensor([-1.0, 0.0])
        feature_tokens[0, 2] = torch.tensor([0.0, 1.0])
        feature_tokens[0, 3] = torch.tensor([0.0, 1.0])

        _residual, weights, _degree, row_probability, _edges = provider._orbit_basis(
            x0_tokens, feature_tokens, ((0, 1),)
        )

        torch.testing.assert_close(
            weights[0, :3],
            torch.tensor(
                [affinity_floor, 0.5 + affinity_floor, 1.0 + affinity_floor]
            ),
            rtol=0,
            atol=1e-7,
        )
        torch.testing.assert_close(
            row_probability, torch.ones_like(row_probability), rtol=0, atol=1e-6
        )

    def test_all_d4_rotations_and_reflections_are_equivariant(self):
        provider = LocalRelationalBasisProvider(
            grid_size=6,
            feature_norm_epsilon=1e-7,
            affinity_floor=1e-7,
        )
        x0 = torch.randn(1, 4, 12, 18)
        feature = torch.randn(1, 5, 18, 24)
        expected, _ = provider(x0, feature)

        for turns in range(4):
            with self.subTest(turns=turns, reflected=False):
                transformed_x0 = torch.rot90(x0, turns, dims=(-2, -1))
                transformed_feature = torch.rot90(feature, turns, dims=(-2, -1))
                actual, _ = provider(transformed_x0, transformed_feature)
                torch.testing.assert_close(
                    actual,
                    torch.rot90(expected, turns, dims=(-2, -1)),
                    rtol=2e-5,
                    atol=2e-5,
                )
            with self.subTest(turns=turns, reflected=True):
                transformed_x0 = torch.flip(
                    torch.rot90(x0, turns, dims=(-2, -1)), dims=(-1,)
                )
                transformed_feature = torch.flip(
                    torch.rot90(feature, turns, dims=(-2, -1)), dims=(-1,)
                )
                actual, _ = provider(transformed_x0, transformed_feature)
                transformed_expected = torch.flip(
                    torch.rot90(expected, turns, dims=(-2, -1)), dims=(-1,)
                )
                torch.testing.assert_close(
                    actual, transformed_expected, rtol=2e-5, atol=2e-5
                )

    def test_boundaries_do_not_wrap(self):
        provider = LocalRelationalBasisProvider(grid_size=5)
        x0 = torch.zeros(1, 4, 5, 5)
        x0[:, :, 0, 0] = 1
        feature = torch.ones(1, 3, 5, 5)

        bases, _ = provider(x0, feature)

        self.assertEqual(float(bases[0, 0, 0, 0, -1]), 0.0)
        self.assertEqual(float(bases[0, 1, 0, -1, -1]), 0.0)
        self.assertEqual(float(bases[0, 2, 0, 0, 3]), 0.0)
        self.assertGreater(float(bases[0, 0, 0, 0, 1]), 0.0)
        self.assertGreater(float(bases[0, 1, 0, 1, 1]), 0.0)
        self.assertGreater(float(bases[0, 2, 0, 0, 2]), 0.0)

    def test_feature_and_x0_are_detached(self):
        provider = LocalRelationalBasisProvider(grid_size=4)
        x0 = torch.randn(2, 4, 8, 8, requires_grad=True)
        feature = torch.randn(2, 6, 8, 8, requires_grad=True)

        bases, diagnostics = provider(x0, feature)

        self.assertFalse(bases.requires_grad)
        self.assertIsNone(x0.grad)
        self.assertIsNone(feature.grad)
        self.assertFalse(diagnostics.basis_rms.requires_grad)

    def test_float16_inputs_are_finite_and_return_float16(self):
        provider = LocalRelationalBasisProvider(
            grid_size=4,
            feature_norm_epsilon=1e-6,
            affinity_floor=1e-6,
        )
        x0 = (torch.randn(1, 4, 8, 8) * 100).half()
        feature = (torch.randn(1, 32, 8, 8) * 100).half()

        bases, diagnostics = provider(x0, feature)

        self.assertEqual(bases.dtype, torch.float16)
        self.assertTrue(torch.isfinite(bases).all())
        for value in (
            diagnostics.edge_weight_min,
            diagnostics.edge_weight_max,
            diagnostics.row_affinity_sum_min,
            diagnostics.row_affinity_sum_max,
            diagnostics.row_probability_sum_min,
            diagnostics.row_probability_sum_max,
            diagnostics.basis_rms,
        ):
            self.assertTrue(torch.isfinite(value).all())

    def test_float16_d4_equivariance(self):
        provider = LocalRelationalBasisProvider(grid_size=6)
        x0 = torch.randn(1, 4, 12, 18).half()
        feature = torch.randn(1, 5, 18, 24).half()
        expected, _ = provider(x0, feature)

        for turns in range(4):
            for reflected in (False, True):
                with self.subTest(turns=turns, reflected=reflected):
                    transformed_x0 = torch.rot90(x0, turns, dims=(-2, -1))
                    transformed_feature = torch.rot90(
                        feature, turns, dims=(-2, -1)
                    )
                    transformed_expected = torch.rot90(
                        expected, turns, dims=(-2, -1)
                    )
                    if reflected:
                        transformed_x0 = torch.flip(transformed_x0, dims=(-1,))
                        transformed_feature = torch.flip(
                            transformed_feature, dims=(-1,)
                        )
                        transformed_expected = torch.flip(
                            transformed_expected, dims=(-1,)
                        )
                    actual, _ = provider(transformed_x0, transformed_feature)
                    torch.testing.assert_close(
                        actual,
                        transformed_expected,
                        rtol=2e-3,
                        atol=2e-3,
                    )

    def test_float16_cast_overflow_fails_closed(self):
        provider = LocalRelationalBasisProvider(grid_size=4)
        x0 = torch.zeros(1, 4, 4, 4, dtype=torch.float16)
        limit = torch.finfo(torch.float16).max
        x0[:, :, 0, 0] = limit
        x0[:, :, 0, 1] = -limit
        feature = torch.ones(1, 3, 4, 4, dtype=torch.float16)

        with self.assertRaisesRegex(RuntimeError, "after conversion"):
            provider(x0, feature)

    def test_each_orbit_is_non_degenerate(self):
        provider = LocalRelationalBasisProvider(grid_size=7)
        x0 = torch.randn(3, 4, 14, 14)
        feature = torch.randn(3, 8, 14, 14)

        bases, diagnostics = provider(x0, feature)

        per_orbit_norm = bases.float().flatten(2).norm(dim=-1)
        self.assertTrue(torch.all(per_orbit_norm > 0))
        self.assertTrue(torch.all(diagnostics.basis_rms > 0))

    def test_invalid_parameters_fail_closed(self):
        invalid_cases = (
            ({"grid_size": 3}, "grid_size"),
            ({"grid_size": 4.0}, "grid_size"),
            ({"grid_size": True}, "grid_size"),
            ({"feature_norm_epsilon": 0.0}, "feature_norm_epsilon"),
            ({"feature_norm_epsilon": 1.0}, "feature_norm_epsilon"),
            ({"feature_norm_epsilon": math.nan}, "feature_norm_epsilon"),
            ({"feature_norm_epsilon": 1e-50}, "feature_norm_epsilon"),
            ({"affinity_floor": 0.0}, "affinity_floor"),
            ({"affinity_floor": 1.0}, "affinity_floor"),
            ({"affinity_floor": math.nan}, "affinity_floor"),
            ({"affinity_floor": 1e-50}, "affinity_floor"),
        )
        for kwargs, message in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    LocalRelationalBasisProvider(**kwargs)

    def test_invalid_inputs_fail_closed(self):
        provider = LocalRelationalBasisProvider(grid_size=4)
        x0 = torch.randn(2, 4, 8, 8)
        feature = torch.randn(2, 6, 8, 8)
        cases = (
            (x0[:, 0], feature, "pred_original_sample"),
            (x0[:, :3], feature, "four channels"),
            (x0.to(torch.int64), feature, "floating-point"),
            (x0, feature[:, 0], "feature"),
            (x0, feature[:1], "batch"),
            (x0, feature.to(torch.int64), "floating-point"),
            (x0[:, :, :3], feature, "grid_size"),
            (x0, feature[:, :, :3], "grid_size"),
        )
        for invalid_x0, invalid_feature, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    provider(invalid_x0, invalid_feature)

        nonfinite_x0 = x0.clone()
        nonfinite_x0[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            provider(nonfinite_x0, feature)
        nonfinite_feature = feature.clone()
        nonfinite_feature[0, 0, 0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            provider(x0, nonfinite_feature)

        with self.assertRaisesRegex(RuntimeError, "registered threshold"):
            provider(x0, torch.zeros_like(feature))


if __name__ == "__main__":
    unittest.main()
