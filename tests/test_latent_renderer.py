import unittest

import torch

from AttentionGuidance import (
    LatentRendererConfig,
    StructuralLatentRenderer,
    build_feature_difference_basis,
    build_graph_transport_basis,
    build_laplacian_basis,
    build_spectral_bases,
    inject_rendered_clean_update,
)


class LatentRendererTest(unittest.TestCase):
    def make_inputs(self, batch=2, channels=4, height=8, width=8):
        torch.manual_seed(17)
        latent = torch.randn(batch, channels, height, width)
        bases = torch.randn(batch, 4, channels, height, width)
        scheduler_update = torch.randn_like(latent)
        return latent, bases, scheduler_update

    def test_spectral_bases_form_a_partition(self):
        latent, _bases, _update = self.make_inputs()
        bands = build_spectral_bases(latent)
        self.assertEqual(bands.shape, (2, 3, 4, 8, 8))
        torch.testing.assert_close(
            bands.sum(dim=1), latent, rtol=1e-5, atol=1e-5
        )

    def test_graph_transport_basis_requires_row_stochastic_graph(self):
        latent, _bases, _update = self.make_inputs()
        identity = torch.eye(16).expand(2, -1, -1)
        basis = build_graph_transport_basis(latent, identity, 4, 4)
        self.assertEqual(basis.shape, (2, 1, 4, 8, 8))
        self.assertTrue(torch.equal(basis, torch.zeros_like(basis)))
        with self.assertRaises(ValueError):
            build_graph_transport_basis(latent, identity * 2, 4, 4)

    def test_laplacian_and_feature_bases_have_expected_shapes(self):
        latent, _bases, _update = self.make_inputs()
        laplacian = build_laplacian_basis(latent)
        self.assertEqual(laplacian.shape, (2, 1, 4, 8, 8))
        backbone = torch.randn(2, 6, 4, 4)
        skip = torch.randn(2, 6, 8, 8)
        feature = build_feature_difference_basis(backbone, skip, (8, 8))
        self.assertEqual(feature.shape, (2, 1, 6, 8, 8))

    def test_scheduler_injection_preserves_scheduler_step(self):
        prev = torch.randn(1, 4, 8, 8)
        predicted = torch.randn_like(prev)
        guided = predicted + 0.125
        injected = inject_rendered_clean_update(prev, predicted, guided)
        torch.testing.assert_close(injected, prev + 0.125)
        half_injected = inject_rendered_clean_update(
            prev.half(), predicted.half(), guided.half()
        )
        self.assertEqual(half_injected.dtype, torch.float16)

    def test_zero_initialised_renderer_is_exact_identity(self):
        latent, bases, scheduler_update = self.make_inputs()
        renderer = StructuralLatentRenderer(
            LatentRendererConfig(num_bases=4, max_update_ratio=0.1)
        )
        output = renderer(
            latent,
            bases,
            timestep=torch.tensor([0.2, 0.8]),
            scheduler_update=scheduler_update,
        )
        self.assertTrue(torch.equal(output.guided_x0, latent))
        self.assertTrue(torch.equal(output.residual, torch.zeros_like(latent)))
        self.assertTrue(torch.equal(output.coefficients, torch.zeros(2, 4)))

    def test_renderer_preserves_moments_and_trust_region(self):
        latent, bases, scheduler_update = self.make_inputs()
        renderer = StructuralLatentRenderer(
            LatentRendererConfig(num_bases=4, max_update_ratio=0.1)
        )
        renderer.policy[-1].bias.data.fill_(0.7)
        output = renderer(
            latent,
            bases,
            timestep=torch.tensor([0.2, 0.8]),
            scheduler_update=scheduler_update,
        )
        self.assertTrue(torch.all(output.diagnostics.update_ratio <= 0.10001))
        self.assertTrue(torch.all(output.diagnostics.update_ratio >= 0))
        self.assertTrue(torch.all(output.diagnostics.mean_error < 1e-5))
        self.assertTrue(torch.all(output.diagnostics.variance_error < 1e-4))

    def test_renderer_gradients_are_finite(self):
        latent, bases, scheduler_update = self.make_inputs()
        latent.requires_grad_()
        bases.requires_grad_()
        renderer = StructuralLatentRenderer(
            LatentRendererConfig(
                num_bases=4, prompt_dim=3, state_dim=2, max_update_ratio=0.1
            )
        )
        renderer.policy[-1].bias.data.fill_(0.2)
        prompt = torch.randn(2, 3)
        state = torch.randn(2, 2)
        output = renderer(
            latent,
            bases,
            timestep=torch.tensor([0.2, 0.8]),
            prompt_embedding=prompt,
            state_features=state,
            scheduler_update=scheduler_update,
        )
        loss = output.guided_x0.square().mean() + output.coefficients.square().mean()
        loss.backward()
        self.assertIsNotNone(latent.grad)
        self.assertIsNotNone(bases.grad)
        self.assertTrue(torch.isfinite(latent.grad).all())
        self.assertTrue(torch.isfinite(bases.grad).all())
        self.assertTrue(
            all(parameter.grad is not None for parameter in renderer.parameters())
        )
        self.assertTrue(
            all(torch.isfinite(parameter.grad).all() for parameter in renderer.parameters())
        )

    def test_context_dimensions_are_checked(self):
        latent, bases, scheduler_update = self.make_inputs()
        renderer = StructuralLatentRenderer(
            LatentRendererConfig(num_bases=4, prompt_dim=3, max_update_ratio=0.1)
        )
        with self.assertRaises(ValueError):
            renderer(latent, bases, scheduler_update=scheduler_update)
        with self.assertRaises(ValueError):
            renderer(
                latent,
                bases,
                prompt_embedding=torch.randn(2, 2),
                scheduler_update=scheduler_update,
            )

    def test_renderer_is_spatially_equivariant_for_flip(self):
        latent, bases, scheduler_update = self.make_inputs()
        renderer = StructuralLatentRenderer(
            LatentRendererConfig(num_bases=4, max_update_ratio=0.1)
        )
        renderer.policy[-1].bias.data.copy_(torch.tensor([0.2, -0.1, 0.05, 0.3]))
        output = renderer(
            latent,
            bases,
            timestep=torch.tensor([0.2, 0.8]),
            scheduler_update=scheduler_update,
        )
        flipped = renderer(
            torch.flip(latent, dims=(-1,)),
            torch.flip(bases, dims=(-1,)),
            timestep=torch.tensor([0.2, 0.8]),
            scheduler_update=torch.flip(scheduler_update, dims=(-1,)),
        )
        torch.testing.assert_close(
            flipped.guided_x0,
            torch.flip(output.guided_x0, dims=(-1,)),
            rtol=1e-4,
            atol=1e-5,
        )

    def test_unconstrained_renderer_can_be_used_for_diagnostics(self):
        latent, bases, _scheduler_update = self.make_inputs()
        renderer = StructuralLatentRenderer(
            LatentRendererConfig(
                num_bases=4, max_update_ratio=None, preserve_moments=False
            )
        )
        renderer.policy[-1].bias.data.fill_(0.1)
        output = renderer(latent, bases)
        self.assertTrue(torch.isfinite(output.guided_x0).all())
        self.assertIsNone(output.diagnostics.update_ratio)


if __name__ == "__main__":
    unittest.main()
