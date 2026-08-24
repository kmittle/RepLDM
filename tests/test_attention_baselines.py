import unittest

import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

from AttentionGuidance.attention_baselines import (
    SparseDenseAttentionProcessor,
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


class AttentionBaselineTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

