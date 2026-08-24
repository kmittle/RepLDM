import unittest

import torch
from torch import nn
from diffusers import EulerDiscreteScheduler

from AttentionGuidance.semantic_transport import (
    QKCapture,
    affinity_from_qk,
    affinity_from_tokens,
    deterministic_permutation,
    fixed_moment_transport,
    inject_predicted_clean_update,
    infer_token_grid,
)


class TinySelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 2
        self.to_q = nn.Linear(8, 8, bias=False)
        self.to_k = nn.Linear(8, 8, bias=False)
        self.norm_q = None
        self.norm_k = None


class SemanticTransportTest(unittest.TestCase):
    def test_token_grid_preserves_aspect_and_count(self):
        self.assertEqual(infer_token_grid(1024, 128, 128), (32, 32))
        self.assertEqual(infer_token_grid(2048, 128, 256), (32, 64))

    def test_reciprocal_graph_is_mutual_and_row_stochastic(self):
        torch.manual_seed(3)
        tokens = torch.randn(2, 4, 8, dtype=torch.float64)
        graph, entropy = affinity_from_tokens(
            tokens, mode="reciprocal_latent", topk=2
        )
        support = graph > 0
        self.assertTrue(torch.equal(support, support.transpose(-1, -2)))
        torch.testing.assert_close(
            graph.sum(dim=-1), torch.ones_like(graph.sum(dim=-1)), rtol=0, atol=1e-6
        )
        self.assertTrue(torch.isfinite(entropy).all())

    def test_permutation_is_bijective_and_reproducible(self):
        first = deterministic_permutation(32, 1729, torch.device("cpu"))
        second = deterministic_permutation(32, 1729, torch.device("cpu"))
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(torch.unique(first).numel(), 32)

    def test_fixed_moment_transport_preserves_channel_moments(self):
        torch.manual_seed(4)
        x0 = torch.randn(2, 4, 16, 16, dtype=torch.float64)
        tokens = x0.flatten(2).transpose(1, 2)
        graph, _ = affinity_from_tokens(tokens, mode="reciprocal_latent", topk=4)
        confidence = torch.tensor([0.2, 0.35], dtype=torch.float64)
        moved = fixed_moment_transport(
            x0,
            graph,
            angle=0.2,
            confidence=confidence,
            grid_height=16,
            grid_width=16,
        )
        torch.testing.assert_close(
            moved.mean(dim=(-2, -1)), x0.mean(dim=(-2, -1)), rtol=1e-10, atol=1e-10
        )
        torch.testing.assert_close(
            moved.var(dim=(-2, -1), correction=0),
            x0.var(dim=(-2, -1), correction=0),
            rtol=1e-8,
            atol=1e-10,
        )

    def test_zero_angle_is_exact_identity(self):
        x0 = torch.randn(1, 4, 8, 8)
        tokens = x0.flatten(2).transpose(1, 2)
        graph, _ = affinity_from_tokens(tokens, mode="clean_tfsa", topk=4)
        moved = fixed_moment_transport(
            x0,
            graph,
            angle=0.0,
            confidence=torch.ones(1),
            grid_height=8,
            grid_width=8,
        )
        self.assertTrue(torch.equal(moved, x0))

    def test_capture_selects_positive_cfg_rows(self):
        layer = TinySelfAttention()
        capture = QKCapture(layer)
        inputs = torch.randn(2, 5, 8)
        with capture.forward():
            expected_q = layer.to_q(inputs)
            expected_k = layer.to_k(inputs)
        query, key = capture.get_conditional(
            do_classifier_free_guidance=True, batch_size=1
        )
        torch.testing.assert_close(query, expected_q[1:])
        torch.testing.assert_close(key, expected_k[1:])

    def test_injection_uses_scheduler_returned_predicted_clean_sample(self):
        scheduler = EulerDiscreteScheduler(
            num_train_timesteps=1000, prediction_type="epsilon"
        )
        scheduler.set_timesteps(4, device="cpu")
        sample = torch.randn(1, 4, 8, 8, dtype=torch.float16)
        model_output = torch.randn_like(sample)
        step_output = scheduler.step(
            model_output, scheduler.timesteps[0], sample, return_dict=True
        )
        guided_x0 = step_output.pred_original_sample + 0.125
        injected = inject_predicted_clean_update(
            step_output.prev_sample, step_output.pred_original_sample, guided_x0
        )
        expected = (step_output.prev_sample + 0.125).to(step_output.prev_sample.dtype)
        self.assertTrue(torch.equal(injected, expected))

    def test_qk_affinity_uses_mean_heads(self):
        layer = TinySelfAttention()
        query = torch.randn(1, 9, 8)
        key = torch.randn(1, 9, 8)
        graph, entropy = affinity_from_qk(
            query,
            key,
            attention_module=layer,
            mode="reciprocal_semantic",
            topk=3,
        )
        self.assertEqual(graph.shape, (1, 9, 9))
        self.assertTrue(torch.isfinite(entropy).all())


if __name__ == "__main__":
    unittest.main()
