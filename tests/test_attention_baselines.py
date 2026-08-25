import unittest

import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from entmax import entmax15

from AttentionGuidance.attention_baselines import (
    GAG_EQ13_IMPLEMENTATION,
    GAG_PAPER_EQUATIONS,
    GAG_PAPER_ID,
    PLADIS_OPERATOR_PORT_IMPLEMENTATION,
    PLADIS_PINNED_PROBABILITY_DTYPE,
    PLADIS_PINNED_SDXL_GROUP_COUNTS,
    PLADIS_PINNED_SDXL_LAYERS,
    PLADIS_PINNED_SDXL_PROCESSOR_COUNT,
    PLADIS_PINNED_SDXL_PROCESSOR_NAMES_SHA256,
    PLADIS_SOURCE_COMMIT,
    SparseDenseAttentionProcessor,
    _gag_parallel_component,
    _pladis_guided_output,
    attention_processor_names_sha256,
    installed_attention_baseline,
)


class FakeUNet:
    def __init__(self):
        self.original_cross = AttnProcessor2_0()
        self.original_self = AttnProcessor2_0()
        self.attn_processors = {
            "block.attn1.processor": self.original_self,
            "block.attn2.processor": self.original_cross,
        }

    def set_attn_processor(self, processors):
        self.attn_processors = dict(processors)


class LayeredFakeUNet:
    def __init__(self):
        self.attn_processors = {
            "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor": AttnProcessor2_0(),
            "mid_block.attentions.0.transformer_blocks.0.attn2.processor": AttnProcessor2_0(),
            "up_blocks.0.attentions.0.transformer_blocks.0.attn2.processor": AttnProcessor2_0(),
            "up_blocks.0.attentions.0.transformer_blocks.0.attn1.processor": AttnProcessor2_0(),
        }

    def set_attn_processor(self, processors):
        self.attn_processors = dict(processors)


class AttentionBaselineTest(unittest.TestCase):
    def test_gag_reimplementation_provenance_constants_are_pinned(self):
        self.assertEqual(
            GAG_EQ13_IMPLEMENTATION,
            "gag_eq13_reimplementation_2603.02531v2",
        )
        self.assertEqual(GAG_PAPER_ID, "2603.02531v2")
        self.assertEqual(GAG_PAPER_EQUATIONS, (12, 13))

    def test_processors_have_finite_cross_and_self_outputs(self):
        for kind, scale in (("pladis", 2.0), ("gag", 10.0)):
            processor = SparseDenseAttentionProcessor(
                kind=kind, attention_scale=scale
            )
            attention = Attention(
                query_dim=8,
                cross_attention_dim=6,
                heads=2,
                dim_head=4,
                residual_connection=False,
            )
            attention.set_processor(processor)
            cross = attention(torch.randn(2, 5, 8), encoder_hidden_states=torch.randn(2, 7, 6))
            self.assertEqual(cross.shape, (2, 5, 8))
            self.assertTrue(torch.isfinite(cross).all())

            self_attention = Attention(
                query_dim=8, heads=2, dim_head=4, residual_connection=True
            )
            self_attention.set_processor(processor)
            spatial = self_attention(torch.randn(2, 8, 3, 4))
            self.assertEqual(spatial.shape, (2, 8, 3, 4))
            self.assertTrue(torch.isfinite(spatial).all())
            self.assertEqual(processor.call_count, 2)

    def test_cross_attention_bool_and_additive_masks_use_key_length(self):
        torch.manual_seed(13)
        hidden_states = torch.randn(1, 2, 8)
        encoder_hidden_states = torch.randn(1, 3, 6)
        changed_masked_token = encoder_hidden_states.clone()
        changed_masked_token[:, 2] = 1_000.0
        for mask in (
            torch.tensor([[[True, True, False]]]),
            torch.tensor([[[0.0, 0.0, -10_000.0]]]),
        ):
            attention = Attention(
                query_dim=8,
                cross_attention_dim=6,
                heads=2,
                dim_head=4,
                residual_connection=False,
            )
            attention.set_processor(
                SparseDenseAttentionProcessor(kind="pladis", attention_scale=2.0)
            )
            original = attention(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=mask,
            )
            changed = attention(
                hidden_states,
                encoder_hidden_states=changed_masked_token,
                attention_mask=mask,
            )
            torch.testing.assert_close(original, changed, rtol=0, atol=0)

        attention.set_processor(
            SparseDenseAttentionProcessor(
                kind="pladis",
                attention_scale=2.0,
                attention_mask_policy="none",
            )
        )
        with self.assertRaisesRegex(ValueError, "attention_mask=None"):
            attention(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=torch.tensor([[[True, True, False]]]),
            )

    def test_baseline_context_restores_processors(self):
        unet = FakeUNet()
        originals = dict(unet.attn_processors)
        with installed_attention_baseline(
            unet, kind="pladis", attention_scale=2.0
        ):
            self.assertIsInstance(
                unet.attn_processors["block.attn2.processor"],
                SparseDenseAttentionProcessor,
            )
            self.assertIs(unet.attn_processors["block.attn1.processor"], originals["block.attn1.processor"])
        self.assertIs(unet.attn_processors["block.attn2.processor"], originals["block.attn2.processor"])
        self.assertIs(unet.attn_processors["block.attn1.processor"], originals["block.attn1.processor"])

    def test_pladis_pinned_layer_set_excludes_mid_and_restores_all(self):
        self.assertEqual(
            PLADIS_SOURCE_COMMIT,
            "248b9d15701c08094c47dc90b4ae24afbf5cf7a9",
        )
        self.assertEqual(
            PLADIS_OPERATOR_PORT_IMPLEMENTATION,
            "pladis_operator_port_248b9d1",
        )
        self.assertEqual(PLADIS_PINNED_SDXL_LAYERS, ("up", "down"))
        self.assertEqual(PLADIS_PINNED_PROBABILITY_DTYPE, "query")
        self.assertEqual(PLADIS_PINNED_SDXL_GROUP_COUNTS, {"up": 36, "down": 24})
        self.assertEqual(PLADIS_PINNED_SDXL_PROCESSOR_COUNT, 60)
        self.assertEqual(
            PLADIS_PINNED_SDXL_PROCESSOR_NAMES_SHA256,
            "2d66ed06dfc07e6d0b0d7ce1a8d39bd262c5e5e86b5d1947e1e412f5b6fe8c8f",
        )
        unet = LayeredFakeUNet()
        originals = dict(unet.attn_processors)
        with installed_attention_baseline(
            unet,
            kind="pladis",
            attention_scale=2.0,
            applied_layers=PLADIS_PINNED_SDXL_LAYERS,
            probability_dtype=PLADIS_PINNED_PROBABILITY_DTYPE,
        ):
            down = next(name for name in originals if name.startswith("down_blocks."))
            mid = next(name for name in originals if name.startswith("mid_block."))
            up = next(
                name
                for name in originals
                if name.startswith("up_blocks.") and name.endswith("attn2.processor")
            )
            self.assertIsInstance(
                unet.attn_processors[down], SparseDenseAttentionProcessor
            )
            self.assertIsInstance(
                unet.attn_processors[up], SparseDenseAttentionProcessor
            )
            self.assertIs(unet.attn_processors[mid], originals[mid])
        for name, processor in originals.items():
            self.assertIs(unet.attn_processors[name], processor)

    def test_pinned_topology_is_observed_and_missing_group_fails_closed(self):
        unet = LayeredFakeUNet()
        selected_names = [
            name
            for name in unet.attn_processors
            if name.endswith("attn2.processor")
            and (name.startswith("up_blocks.") or name.startswith("down_blocks."))
        ]
        expected_digest = attention_processor_names_sha256(selected_names)
        with installed_attention_baseline(
            unet,
            kind="pladis",
            attention_scale=2.0,
            applied_layers=["up", "down"],
            expected_group_counts={"up": 1, "down": 1},
            expected_processor_names_sha256=expected_digest,
        ) as topology:
            self.assertEqual(topology["group_counts"], {"up": 1, "down": 1})
            self.assertEqual(topology["processor_count"], 2)
            self.assertEqual(topology["processor_names_sha256"], expected_digest)
            for name in selected_names:
                unet.attn_processors[name].call_count = 50
        self.assertEqual(topology["processors_called"], 2)
        self.assertEqual(topology["processor_calls_total"], 100)
        self.assertEqual(topology["processor_call_count_min"], 50)
        self.assertEqual(topology["processor_call_count_max"], 50)

        down_only = LayeredFakeUNet()
        down_only.attn_processors = {
            name: processor
            for name, processor in down_only.attn_processors.items()
            if name.startswith("down_blocks.")
        }
        with self.assertRaisesRegex(ValueError, "requested layer groups"):
            with installed_attention_baseline(
                down_only,
                kind="pladis",
                attention_scale=2.0,
                applied_layers=["up", "down"],
            ):
                pass

        with self.assertRaisesRegex(ValueError, "group counts differ"):
            with installed_attention_baseline(
                unet,
                kind="pladis",
                attention_scale=2.0,
                applied_layers=["up", "down"],
                expected_group_counts={"up": 2, "down": 1},
            ):
                pass

    def test_pladis_query_dtype_weights_match_pinned_upstream_equation(self):
        torch.manual_seed(71)
        query = torch.randn(2, 3, 5, 4, dtype=torch.float64)
        key = torch.randn(2, 3, 7, 4, dtype=torch.float64)
        processor = SparseDenseAttentionProcessor(
            kind="pladis",
            attention_scale=2.0,
            probability_dtype="query",
        )
        sparse, dense = processor._weights(query, key, None)

        logits = query @ key.transpose(-1, -2) * (query.shape[-1] ** -0.5)
        expected_sparse = entmax15(logits, dim=-1)
        expected_dense = torch.softmax(logits, dim=-1)
        torch.testing.assert_close(sparse, expected_sparse, rtol=0, atol=0)
        torch.testing.assert_close(dense, expected_dense, rtol=0, atol=0)
        value = torch.randn(2, 3, 7, 4, dtype=torch.float64)
        actual_output = _pladis_guided_output(sparse, dense, value, 2.0)
        expected_output = torch.matmul(
            2.0 * expected_sparse - expected_dense, value
        )
        torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)

    def test_gag_fp16_small_positive_projection_matches_equation_13(self):
        sparse_output = torch.tensor([[[[1e-4, 0.0]]]], dtype=torch.float16)
        dense_output = torch.zeros_like(sparse_output)
        denominator = sparse_output.float().square().sum(dim=-1, keepdim=True)
        self.assertGreater(float(denominator.item()), 0.0)
        self.assertLess(float(denominator.item()), torch.finfo(torch.float16).eps)
        parallel = _gag_parallel_component(sparse_output, dense_output)
        torch.testing.assert_close(parallel, sparse_output.float(), rtol=0, atol=0)

        zero = torch.zeros_like(sparse_output)
        self.assertTrue(
            torch.equal(_gag_parallel_component(zero, zero), zero.float())
        )

    def test_layer_and_probability_modes_fail_closed(self):
        unet = LayeredFakeUNet()
        with self.assertRaisesRegex(ValueError, "layer groups"):
            with installed_attention_baseline(
                unet,
                kind="pladis",
                attention_scale=2.0,
                applied_layers=["encoder"],
            ):
                pass
        with self.assertRaisesRegex(ValueError, "probability_dtype"):
            SparseDenseAttentionProcessor(
                kind="pladis",
                attention_scale=2.0,
                probability_dtype="automatic",
            )


if __name__ == "__main__":
    unittest.main()
