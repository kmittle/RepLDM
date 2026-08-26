import hashlib
import math
import unittest

import torch
from torch.nn import functional as F

from AttentionGuidance.local_relational_basis import (
    LOCAL_RELATIONAL_AFFINITY_SOURCES,
    LOCAL_RELATIONAL_OFFSET_ORBITS,
    LOCAL_RELATIONAL_ORBIT_NAMES,
    RANDOM_EDGE_COUNTER_SCHEMA,
    LocalRelationalBasisProvider,
    canonical_random_edge_nodes,
    random_edge_counter_bytes,
    random_edge_counter_set_sha256,
    random_edge_uniform,
)


REGISTERED_OFFSET_ORBITS = (
    ("axis-r1", ((0, 1), (1, 0))),
    ("diagonal-r1", ((1, 1), (1, -1))),
    ("axis-r2", ((0, 2), (2, 0))),
)
REGISTERED_RANDOM_EDGE_KEY = {
    "experiment_id": "ao-search-v1",
    "split_role": "search",
    "prompt_row_id": "search-0001",
    "seed": 123456789,
    "step_index": 0,
}


def d4_node_maps(grid_size):
    """Map old row-major node ids to positions after each tensor D4 action."""
    node_ids = torch.arange(grid_size * grid_size).reshape(grid_size, grid_size)
    positions = torch.arange(grid_size * grid_size)
    mappings = []
    for turns in range(4):
        rotated = torch.rot90(node_ids, turns, dims=(-2, -1))
        for reflected in (False, True):
            transformed = torch.flip(rotated, dims=(-1,)) if reflected else rotated
            mapping = torch.empty_like(positions)
            mapping[transformed.flatten()] = positions
            mappings.append(tuple(mapping.tolist()))
    return tuple(mappings)


class LocalRelationalBasisProviderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(101)

    def _dense_reference(self, x0, control, affinity_floor):
        grid = x0.shape[-1]
        x0_tokens = x0.float().flatten(2).transpose(1, 2)
        if control is None:
            control_tokens = None
        else:
            control = control.float()
            control = control / torch.linalg.vector_norm(
                control, dim=1, keepdim=True
            )
            control_tokens = control.flatten(2).transpose(1, 2)
        references = []
        for _name, offsets in REGISTERED_OFFSET_ORBITS:
            adjacency = torch.zeros(
                x0.shape[0], grid * grid, grid * grid, dtype=torch.float32
            )
            for dy, dx in offsets:
                for y in range(grid):
                    for x in range(grid):
                        target_y = y + dy
                        target_x = x + dx
                        if not (0 <= target_y < grid and 0 <= target_x < grid):
                            continue
                        source = y * grid + x
                        target = target_y * grid + target_x
                        if control_tokens is None:
                            weight = torch.ones(x0.shape[0])
                        else:
                            cosine = (
                                control_tokens[:, source]
                                * control_tokens[:, target]
                            ).sum(dim=-1).clamp(-1.0, 1.0)
                            weight = affinity_floor + 0.5 * (1.0 + cosine)
                        adjacency[:, source, target] = weight
                        adjacency[:, target, source] = weight
            self.assertTrue(torch.equal(adjacency, adjacency.transpose(-1, -2)))
            transition = adjacency / adjacency.sum(dim=-1, keepdim=True)
            torch.testing.assert_close(
                transition.sum(dim=-1),
                torch.ones(x0.shape[0], grid * grid),
                rtol=0,
                atol=2e-7,
            )
            residual = torch.bmm(transition, x0_tokens) - x0_tokens
            references.append(
                residual.transpose(1, 2).reshape(x0.shape[0], 4, grid, grid)
            )
        return torch.stack(references, dim=1)

    def _random_dense_reference(self, x0, affinity_floor, key):
        grid = x0.shape[-1]
        x0_tokens = x0.float().flatten(2).transpose(1, 2)
        references = []
        for orbit_name, offsets in REGISTERED_OFFSET_ORBITS:
            adjacency = torch.zeros(
                x0.shape[0], grid * grid, grid * grid, dtype=torch.float32
            )
            for dy, dx in offsets:
                for y in range(grid):
                    for x in range(grid):
                        target_y = y + dy
                        target_x = x + dx
                        if not (0 <= target_y < grid and 0 <= target_x < grid):
                            continue
                        source = y * grid + x
                        target = target_y * grid + target_x
                        low, high = sorted((source, target))
                        low, high = canonical_random_edge_nodes(grid, low, high)
                        weight = affinity_floor + random_edge_uniform(
                            **key,
                            orbit_name=orbit_name,
                            edge_low=low,
                            edge_high=high,
                        )
                        adjacency[:, source, target] = weight
                        adjacency[:, target, source] = weight
            self.assertTrue(torch.equal(adjacency, adjacency.transpose(-1, -2)))
            transition = adjacency / adjacency.sum(dim=-1, keepdim=True)
            torch.testing.assert_close(
                transition.sum(dim=-1),
                torch.ones(x0.shape[0], grid * grid),
                rtol=0,
                atol=2e-7,
            )
            residual = torch.bmm(transition, x0_tokens) - x0_tokens
            references.append(
                residual.transpose(1, 2).reshape(x0.shape[0], 4, grid, grid)
            )
        return torch.stack(references, dim=1)

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
        self.assertEqual(diagnostics.affinity_source, "feature")
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
        self.assertEqual(record["affinity_source"], "feature")
        self.assertEqual(record["predicted_clean_norm_epsilon"], 1e-6)
        self.assertNotIn("temperature", record)
        self.assertNotIn("quality", record)
        self.assertNotIn("random_edge_counter_schema", record)
        self.assertNotIn("random_edge_actual_edge_counts", record)
        self.assertNotIn("random_edge_unique_canonical_key_counts", record)
        self.assertNotIn("random_edge_actual_edges_unique", record)

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
        torch.testing.assert_close(
            actual,
            self._dense_reference(x0, feature, affinity_floor),
            rtol=2e-5,
            atol=2e-6,
        )

    def test_uniform_local_matches_dense_graph_and_has_exact_unit_affinity(self):
        grid = 5
        provider = LocalRelationalBasisProvider(grid_size=grid)
        x0 = torch.randn(2, 4, grid, grid)

        actual, diagnostics = provider.uniform_local(x0)

        torch.testing.assert_close(
            actual,
            self._dense_reference(x0, None, provider.affinity_floor),
            rtol=2e-5,
            atol=2e-6,
        )
        self.assertEqual(diagnostics.affinity_source, "uniform_local")
        self.assertTrue(
            torch.equal(
                diagnostics.edge_weight_min,
                torch.ones_like(diagnostics.edge_weight_min),
            )
        )
        self.assertTrue(
            torch.equal(
                diagnostics.edge_weight_max,
                torch.ones_like(diagnostics.edge_weight_max),
            )
        )
        self.assertTrue(
            torch.equal(
                diagnostics.row_probability_sum_min,
                torch.ones_like(diagnostics.row_probability_sum_min),
            )
        )
        self.assertTrue(
            torch.equal(
                diagnostics.row_probability_sum_max,
                torch.ones_like(diagnostics.row_probability_sum_max),
            )
        )

    def test_predicted_clean_matches_independent_dense_graph(self):
        grid = 5
        affinity_floor = 2e-5
        provider = LocalRelationalBasisProvider(
            grid_size=grid,
            predicted_clean_norm_epsilon=1e-7,
            affinity_floor=affinity_floor,
        )
        x0 = torch.randn(2, 4, 10, 15) + 0.25

        actual, diagnostics = provider.predicted_clean(x0)
        pooled = F.adaptive_avg_pool2d(x0.float(), (grid, grid))
        coarse_reference = self._dense_reference(
            pooled, pooled, affinity_floor
        )
        reference = F.interpolate(
            coarse_reference.flatten(0, 1),
            size=x0.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).reshape(x0.shape[0], 3, 4, *x0.shape[-2:])

        torch.testing.assert_close(
            actual,
            reference,
            rtol=2e-5,
            atol=2e-6,
        )
        self.assertEqual(diagnostics.affinity_source, "predicted_clean")
        self.assertEqual(
            diagnostics.to_record()["affinity_source"], "predicted_clean"
        )

    def test_random_edge_counter_bytes_and_uint24_mapping_are_exact(self):
        self.assertEqual(canonical_random_edge_nodes(16, 17, 18), (17, 18))
        expected = (
            b'{"edge_high":18,"edge_low":17,"experiment_id":"ao-search-v1",'
            b'"orbit_name":"axis-r1","prompt_row_id":"search-0001",'
            b'"schema":"ao-random-edge-counter-v1","seed":123456789,'
            b'"split_role":"search","step_index":0}'
        )
        counter = random_edge_counter_bytes(
            **REGISTERED_RANDOM_EDGE_KEY,
            orbit_name="axis-r1",
            edge_low=17,
            edge_high=18,
        )

        self.assertEqual(RANDOM_EDGE_COUNTER_SCHEMA, "ao-random-edge-counter-v1")
        self.assertEqual(counter, expected)
        self.assertEqual(
            hashlib.sha256(counter).hexdigest(),
            "9f66bae057b4506b502956e16abec22a4bd24956e56ee959c90438ba190bec5f",
        )
        self.assertEqual(
            random_edge_uniform(
                **REGISTERED_RANDOM_EDGE_KEY,
                orbit_name="axis-r1",
                edge_low=17,
                edge_high=18,
            ),
            10446522 / 2**24,
        )

    def test_random_edge_nodes_use_exhaustive_d4_canonical_representatives(self):
        for grid in (4, 5):
            mappings = d4_node_maps(grid)
            self.assertEqual(len(set(mappings)), 8)
            for low in range(grid * grid):
                for high in range(low + 1, grid * grid):
                    images = {
                        tuple(sorted((mapping[low], mapping[high])))
                        for mapping in mappings
                    }
                    expected = min(images)
                    with self.subTest(grid=grid, edge=(low, high)):
                        self.assertEqual(
                            canonical_random_edge_nodes(grid, low, high),
                            expected,
                        )
                        for image_low, image_high in images:
                            self.assertEqual(
                                canonical_random_edge_nodes(
                                    grid, image_low, image_high
                                ),
                                expected,
                            )

    def test_random_edge_canonical_reuse_keeps_actual_edge_uniqueness(self):
        provider = LocalRelationalBasisProvider(grid_size=4)
        x0_tokens = torch.zeros(1, 16, 4)
        key = (
            REGISTERED_RANDOM_EDGE_KEY["experiment_id"],
            REGISTERED_RANDOM_EDGE_KEY["split_role"],
            REGISTERED_RANDOM_EDGE_KEY["prompt_row_id"],
            REGISTERED_RANDOM_EDGE_KEY["seed"],
            REGISTERED_RANDOM_EDGE_KEY["step_index"],
        )
        seen_actual_edges = set()
        canonical_uniforms = {}

        first = provider._random_edge_weights(
            x0_tokens,
            torch.tensor([0]),
            torch.tensor([1]),
            key=key,
            orbit_name="axis-r1",
            seen_actual_edges=seen_actual_edges,
            canonical_uniforms=canonical_uniforms,
        )
        symmetric = provider._random_edge_weights(
            x0_tokens,
            torch.tensor([2]),
            torch.tensor([3]),
            key=key,
            orbit_name="axis-r1",
            seen_actual_edges=seen_actual_edges,
            canonical_uniforms=canonical_uniforms,
        )

        self.assertTrue(torch.equal(first, symmetric))
        self.assertEqual(len(seen_actual_edges), 2)
        self.assertEqual(len(canonical_uniforms), 1)
        with self.assertRaisesRegex(RuntimeError, "duplicate actual edge"):
            provider._random_edge_weights(
                x0_tokens,
                torch.tensor([1]),
                torch.tensor([0]),
                key=key,
                orbit_name="axis-r1",
                seen_actual_edges=seen_actual_edges,
                canonical_uniforms=canonical_uniforms,
            )

    def test_random_edge_matches_symmetric_row_normalized_dense_graph(self):
        grid = 4
        affinity_floor = 1e-6
        provider = LocalRelationalBasisProvider(
            grid_size=grid,
            affinity_floor=affinity_floor,
        )
        sample = torch.randn(1, 4, grid, grid)
        x0 = sample.expand(2, -1, -1, -1).clone()

        actual, diagnostics = provider.random_edge(
            x0, **REGISTERED_RANDOM_EDGE_KEY
        )
        torch.manual_seed(999999)
        repeated, repeated_diagnostics = provider.random_edge(
            x0, **REGISTERED_RANDOM_EDGE_KEY
        )

        torch.testing.assert_close(
            actual,
            self._random_dense_reference(
                x0,
                affinity_floor,
                REGISTERED_RANDOM_EDGE_KEY,
            ),
            rtol=2e-5,
            atol=2e-6,
        )
        self.assertTrue(torch.equal(actual, repeated))
        self.assertTrue(torch.equal(actual[0], actual[1]))
        self.assertEqual(diagnostics.affinity_source, "random_edge")
        self.assertEqual(repeated_diagnostics.affinity_source, "random_edge")
        self.assertEqual(diagnostics.random_edge_counter_schema, RANDOM_EDGE_COUNTER_SCHEMA)
        self.assertEqual(diagnostics.random_edge_actual_edge_counts, (24, 18, 16))
        self.assertEqual(
            diagnostics.random_edge_unique_canonical_key_counts,
            (4, 4, 2),
        )
        self.assertEqual(
            diagnostics.random_edge_counter_set_sha256[0],
            random_edge_counter_set_sha256(
                **REGISTERED_RANDOM_EDGE_KEY,
                orbit_name="axis-r1",
                grid_size=grid,
            ),
        )
        self.assertTrue(
            all(
                len(value) == 64
                for value in diagnostics.random_edge_counter_set_sha256
            )
        )
        self.assertIs(diagnostics.random_edge_actual_edges_unique, True)
        random_record = diagnostics.to_record()
        self.assertEqual(
            random_record["random_edge_counter_schema"],
            RANDOM_EDGE_COUNTER_SCHEMA,
        )
        self.assertEqual(random_record["random_edge_actual_edge_counts"], [24, 18, 16])
        self.assertEqual(
            random_record["random_edge_unique_canonical_key_counts"],
            [4, 4, 2],
        )
        self.assertEqual(
            random_record["random_edge_counter_set_sha256"],
            list(diagnostics.random_edge_counter_set_sha256),
        )
        self.assertIs(random_record["random_edge_actual_edges_unique"], True)
        self.assertTrue(torch.all(diagnostics.edge_weight_min >= affinity_floor))
        self.assertTrue(torch.all(diagnostics.edge_weight_max < 1 + affinity_floor))
        torch.testing.assert_close(
            diagnostics.row_probability_sum_min,
            torch.ones_like(diagnostics.row_probability_sum_min),
        )
        torch.testing.assert_close(
            diagnostics.row_probability_sum_max,
            torch.ones_like(diagnostics.row_probability_sum_max),
        )

    def test_random_edge_counter_changes_with_every_registered_key_dimension(self):
        base = {
            **REGISTERED_RANDOM_EDGE_KEY,
            "orbit_name": "axis-r1",
            "edge_low": 17,
            "edge_high": 18,
        }
        baseline = random_edge_uniform(**base)
        changes = {
            "prompt_row_id": "search-0002",
            "seed": 123456790,
            "step_index": 1,
            "orbit_name": "diagonal-r1",
            "experiment_id": "ao-replay-v1",
            "split_role": "replay",
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                changed = dict(base)
                changed[field] = value
                self.assertNotEqual(random_edge_uniform(**changed), baseline)

    def test_repeated_truncated_values_are_valid_for_distinct_counters(self):
        collision_key = {
            **REGISTERED_RANDOM_EDGE_KEY,
            "prompt_row_id": "search-00804",
        }
        first = {
            **collision_key,
            "orbit_name": "diagonal-r1",
            "edge_low": 5,
            "edge_high": 20,
        }
        second = {
            **collision_key,
            "orbit_name": "diagonal-r1",
            "edge_low": 54,
            "edge_high": 71,
        }
        self.assertEqual(canonical_random_edge_nodes(16, 5, 20), (5, 20))
        self.assertEqual(canonical_random_edge_nodes(16, 54, 71), (54, 71))
        self.assertNotEqual(
            random_edge_counter_bytes(**first),
            random_edge_counter_bytes(**second),
        )
        self.assertEqual(random_edge_uniform(**first), random_edge_uniform(**second))

        provider = LocalRelationalBasisProvider(grid_size=16)
        x0 = torch.randn(1, 4, 16, 16)
        bases, diagnostics = provider.random_edge(x0, **collision_key)
        self.assertTrue(torch.isfinite(bases).all())
        self.assertEqual(diagnostics.affinity_source, "random_edge")
        self.assertEqual(
            diagnostics.random_edge_unique_canonical_key_counts,
            (64, 64, 56),
        )

    def test_random_edge_counter_invalid_inputs_fail_closed(self):
        valid = {
            **REGISTERED_RANDOM_EDGE_KEY,
            "orbit_name": "axis-r1",
            "edge_low": 17,
            "edge_high": 18,
        }
        allowed_punctuation = dict(valid)
        allowed_punctuation.update(
            {
                "experiment_id": "AO.v1:test-run",
                "split_role": "search_role",
                "prompt_row_id": "prompt_0001",
            }
        )
        self.assertIsInstance(
            random_edge_counter_bytes(**allowed_punctuation), bytes
        )
        invalid = (
            ("experiment_id", "", "experiment_id"),
            ("experiment_id", "ao search", "experiment_id"),
            ("experiment_id", "ao/search", "experiment_id"),
            ("split_role", "search!", "split_role"),
            ("prompt_row_id", None, "prompt_row_id"),
            ("prompt_row_id", "search@0001", "prompt_row_id"),
            ("orbit_name", "axis-r3", "orbit_name"),
            ("seed", True, "seed"),
            ("seed", -1, "seed"),
            ("seed", 1.0, "seed"),
            ("step_index", False, "step_index"),
            ("step_index", -1, "step_index"),
            ("edge_low", -1, "edge_low"),
            ("edge_low", True, "edge_low"),
            ("edge_high", 17, "edge_low < edge_high"),
            ("edge_low", 19, "edge_low < edge_high"),
        )
        for field, value, message in invalid:
            with self.subTest(field=field, value=value):
                counter = dict(valid)
                counter[field] = value
                with self.assertRaisesRegex(ValueError, message):
                    random_edge_counter_bytes(**counter)

        provider = LocalRelationalBasisProvider(grid_size=4)
        x0 = torch.randn(1, 4, 4, 4)
        invalid_key = dict(REGISTERED_RANDOM_EDGE_KEY)
        invalid_key["prompt_row_id"] = "bad prompt"
        with self.assertRaisesRegex(ValueError, "prompt_row_id"):
            provider.random_edge(x0, **invalid_key)

    def test_affinity_sources_are_explicit_and_complete(self):
        self.assertEqual(
            LOCAL_RELATIONAL_AFFINITY_SOURCES,
            ("feature", "uniform_local", "predicted_clean", "random_edge"),
        )

    def test_offset_orbits_match_registered_protocol(self):
        self.assertEqual(LOCAL_RELATIONAL_OFFSET_ORBITS, REGISTERED_OFFSET_ORBITS)
        self.assertEqual(
            LOCAL_RELATIONAL_ORBIT_NAMES,
            tuple(name for name, _offsets in REGISTERED_OFFSET_ORBITS),
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

        (
            _residual,
            weights,
            _degree,
            row_probability,
            _edges,
            canonical_keys,
        ) = provider._orbit_basis(x0_tokens, feature_tokens, ((0, 1),))

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
        self.assertIsNone(canonical_keys)

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

    def test_uniform_and_predicted_clean_are_d4_equivariant(self):
        provider = LocalRelationalBasisProvider(grid_size=6)
        x0 = torch.randn(1, 4, 12, 18) + 0.5

        for affinity_source in ("uniform_local", "predicted_clean"):
            build = getattr(provider, affinity_source)
            expected, diagnostics = build(x0)
            self.assertEqual(diagnostics.affinity_source, affinity_source)
            for turns in range(4):
                for reflected in (False, True):
                    with self.subTest(
                        affinity_source=affinity_source,
                        turns=turns,
                        reflected=reflected,
                    ):
                        transformed_x0 = torch.rot90(
                            x0, turns, dims=(-2, -1)
                        )
                        transformed_expected = torch.rot90(
                            expected, turns, dims=(-2, -1)
                        )
                        if reflected:
                            transformed_x0 = torch.flip(
                                transformed_x0, dims=(-1,)
                            )
                            transformed_expected = torch.flip(
                                transformed_expected, dims=(-1,)
                            )
                        actual, _ = build(transformed_x0)
                        torch.testing.assert_close(
                            actual,
                            transformed_expected,
                            rtol=2e-5,
                            atol=2e-5,
                        )

    def test_random_edge_is_fully_d4_equivariant(self):
        provider = LocalRelationalBasisProvider(grid_size=6)
        x0 = torch.randn(1, 4, 12, 18)
        expected, expected_diagnostics = provider.random_edge(
            x0, **REGISTERED_RANDOM_EDGE_KEY
        )

        for turns in range(4):
            for reflected in (False, True):
                with self.subTest(turns=turns, reflected=reflected):
                    transformed_x0 = torch.rot90(x0, turns, dims=(-2, -1))
                    transformed_expected = torch.rot90(
                        expected, turns, dims=(-2, -1)
                    )
                    if reflected:
                        transformed_x0 = torch.flip(transformed_x0, dims=(-1,))
                        transformed_expected = torch.flip(
                            transformed_expected, dims=(-1,)
                        )
                    actual, diagnostics = provider.random_edge(
                        transformed_x0, **REGISTERED_RANDOM_EDGE_KEY
                    )
                    torch.testing.assert_close(
                        actual,
                        transformed_expected,
                        rtol=2e-5,
                        atol=2e-5,
                    )
                    self.assertEqual(
                        diagnostics.random_edge_actual_edge_counts,
                        expected_diagnostics.random_edge_actual_edge_counts,
                    )
                    self.assertEqual(
                        diagnostics.random_edge_unique_canonical_key_counts,
                        expected_diagnostics.random_edge_unique_canonical_key_counts,
                    )

    def test_boundaries_do_not_wrap(self):
        provider = LocalRelationalBasisProvider(grid_size=5)
        x0 = torch.ones(1, 4, 5, 5)
        x0[:, :, 0, 0] = 2
        feature = torch.ones(1, 3, 5, 5)
        controls = (
            ("feature", lambda: provider(x0, feature)),
            ("uniform_local", lambda: provider.uniform_local(x0)),
            ("predicted_clean", lambda: provider.predicted_clean(x0)),
            (
                "random_edge",
                lambda: provider.random_edge(x0, **REGISTERED_RANDOM_EDGE_KEY),
            ),
        )

        for affinity_source, build in controls:
            with self.subTest(affinity_source=affinity_source):
                bases, diagnostics = build()
                self.assertEqual(diagnostics.affinity_source, affinity_source)
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

        outputs = (
            provider(x0, feature),
            provider.uniform_local(x0),
            provider.predicted_clean(x0),
            provider.random_edge(x0, **REGISTERED_RANDOM_EDGE_KEY),
        )

        for bases, diagnostics in outputs:
            self.assertFalse(bases.requires_grad)
            self.assertFalse(diagnostics.basis_rms.requires_grad)
            for value in (
                diagnostics.edge_weight_min,
                diagnostics.edge_weight_max,
                diagnostics.row_affinity_sum_min,
                diagnostics.row_affinity_sum_max,
            ):
                self.assertFalse(value.requires_grad)
        self.assertIsNone(x0.grad)
        self.assertIsNone(feature.grad)

    def test_float16_inputs_are_finite_and_return_float16(self):
        provider = LocalRelationalBasisProvider(
            grid_size=4,
            feature_norm_epsilon=1e-6,
            affinity_floor=1e-6,
        )
        x0 = (torch.randn(1, 4, 8, 8) * 100).half()
        feature = (torch.randn(1, 32, 8, 8) * 100).half()

        outputs = (
            provider(x0, feature),
            provider.uniform_local(x0),
            provider.predicted_clean(x0),
            provider.random_edge(x0, **REGISTERED_RANDOM_EDGE_KEY),
        )

        self.assertEqual(
            tuple(diagnostics.affinity_source for _bases, diagnostics in outputs),
            LOCAL_RELATIONAL_AFFINITY_SOURCES,
        )
        for bases, diagnostics in outputs:
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
        x0 = torch.ones(1, 4, 4, 4, dtype=torch.float16)
        limit = torch.finfo(torch.float16).max
        x0[:, :, 0, 0] = limit
        x0[:, :, 0, 1] = -limit
        feature = torch.ones(1, 3, 4, 4, dtype=torch.float16)

        controls = (
            lambda: provider(x0, feature),
            lambda: provider.uniform_local(x0),
            lambda: provider.predicted_clean(x0),
            lambda: provider.random_edge(x0, **REGISTERED_RANDOM_EDGE_KEY),
        )
        for build in controls:
            with self.subTest(build=build):
                with self.assertRaisesRegex(RuntimeError, "after conversion"):
                    build()

    def test_each_orbit_is_non_degenerate(self):
        provider = LocalRelationalBasisProvider(grid_size=7)
        x0 = torch.randn(3, 4, 14, 14)
        feature = torch.randn(3, 8, 14, 14)

        bases, diagnostics = provider(x0, feature)

        per_orbit_norm = bases.float().flatten(2).norm(dim=-1)
        self.assertTrue(torch.all(per_orbit_norm > 0))
        self.assertTrue(torch.all(diagnostics.basis_rms > 0))

    def test_uniform_and_predicted_clean_constant_and_non_degenerate_cases(self):
        provider = LocalRelationalBasisProvider(grid_size=6)
        constant = torch.full((2, 4, 12, 18), 1.25)
        random_x0 = torch.randn(2, 4, 12, 18) + 0.25

        for affinity_source in ("uniform_local", "predicted_clean"):
            build = getattr(provider, affinity_source)
            with self.subTest(affinity_source=affinity_source, case="constant"):
                constant_bases, _ = build(constant)
                self.assertTrue(
                    torch.equal(
                        constant_bases, torch.zeros_like(constant_bases)
                    )
                )
            with self.subTest(
                affinity_source=affinity_source, case="non_degenerate"
            ):
                random_bases, diagnostics = build(random_x0)
                self.assertTrue(
                    torch.all(random_bases.float().flatten(2).norm(dim=-1) > 0)
                )
                self.assertTrue(torch.all(diagnostics.basis_rms > 0))

    def test_predicted_clean_zero_and_threshold_norms_fail_closed(self):
        norm_epsilon = 1e-4
        provider = LocalRelationalBasisProvider(
            grid_size=4,
            predicted_clean_norm_epsilon=norm_epsilon,
        )
        zero = torch.zeros(1, 4, 4, 4)
        at_threshold = zero.clone()
        at_threshold[:, 0] = norm_epsilon
        one_zero_token = torch.ones_like(zero)
        one_zero_token[:, :, 1, 2] = 0.0
        one_threshold_token = torch.ones_like(zero)
        one_threshold_token[:, :, 1, 2] = 0.0
        one_threshold_token[:, 0, 1, 2] = norm_epsilon

        for case, x0 in (
            ("all_zero", zero),
            ("all_at_threshold", at_threshold),
            ("one_zero_token", one_zero_token),
            ("one_threshold_token", one_threshold_token),
        ):
            with self.subTest(case=case):
                with self.assertRaisesRegex(
                    RuntimeError, "predicted clean.*registered threshold"
                ):
                    provider.predicted_clean(x0)
        uniform, diagnostics = provider.uniform_local(zero)
        self.assertTrue(torch.equal(uniform, torch.zeros_like(uniform)))
        self.assertEqual(diagnostics.affinity_source, "uniform_local")

    def test_predicted_clean_norm_epsilon_is_independent(self):
        provider = LocalRelationalBasisProvider(
            grid_size=4,
            feature_norm_epsilon=0.5,
            predicted_clean_norm_epsilon=1e-6,
        )
        x0 = torch.full((1, 4, 4, 4), 1e-3)

        bases, diagnostics = provider.predicted_clean(x0)

        self.assertTrue(torch.equal(bases, torch.zeros_like(bases)))
        self.assertEqual(diagnostics.feature_norm_epsilon, 0.5)
        self.assertEqual(diagnostics.predicted_clean_norm_epsilon, 1e-6)

    def test_invalid_parameters_fail_closed(self):
        invalid_cases = (
            ({"grid_size": 3}, "grid_size"),
            ({"grid_size": 4.0}, "grid_size"),
            ({"grid_size": True}, "grid_size"),
            ({"feature_norm_epsilon": 0.0}, "feature_norm_epsilon"),
            ({"feature_norm_epsilon": 1.0}, "feature_norm_epsilon"),
            ({"feature_norm_epsilon": math.nan}, "feature_norm_epsilon"),
            ({"feature_norm_epsilon": 1e-50}, "feature_norm_epsilon"),
            (
                {"predicted_clean_norm_epsilon": 0.0},
                "predicted_clean_norm_epsilon",
            ),
            (
                {"predicted_clean_norm_epsilon": -1e-6},
                "predicted_clean_norm_epsilon",
            ),
            (
                {"predicted_clean_norm_epsilon": 1.0},
                "predicted_clean_norm_epsilon",
            ),
            (
                {"predicted_clean_norm_epsilon": float("inf")},
                "predicted_clean_norm_epsilon",
            ),
            (
                {"predicted_clean_norm_epsilon": math.nan},
                "predicted_clean_norm_epsilon",
            ),
            (
                {"predicted_clean_norm_epsilon": True},
                "predicted_clean_norm_epsilon",
            ),
            (
                {"predicted_clean_norm_epsilon": 1e-50},
                "predicted_clean_norm_epsilon",
            ),
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
