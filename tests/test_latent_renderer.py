import unittest
from unittest import mock

import torch
from torch import nn

from AttentionGuidance import (
    LatentRendererConfig,
    LazyLatentStructureBasisProvider,
    RendererObservation,
    StructuralLatentRenderer,
    StructuralUNetBasisProvider,
    build_fixed_coefficient_renderer,
    build_feature_difference_basis,
    build_graph_transport_basis,
    build_laplacian_basis,
    build_spectral_basis,
    build_spectral_bases,
    euler_clean_update_gain,
    euler_model_output_from_clean_sample,
    inject_euler_clean_update,
    inject_rendered_clean_update,
    prepare_euler_clean_endpoint,
    normalize_latent_structure_bases,
)


class _FakeAttention(nn.Module):
    heads = 2

    def __init__(self, channels=8):
        super().__init__()
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(channels, channels, bias=False)

    def forward(self, tokens):
        return self.to_q(tokens), self.to_k(tokens)


class _FakeUpBlock(nn.Module):
    def forward(self, hidden_states, res_hidden_states_tuple=None):
        return hidden_states + res_hidden_states_tuple[-1]


class _FakeUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_blocks = nn.ModuleList([_FakeUpBlock()])
        self.attn = _FakeAttention()

    def forward(self, hidden_states, skip):
        output = self.up_blocks[0](
            hidden_states=hidden_states,
            res_hidden_states_tuple=(skip,),
        )
        self.attn(output.flatten(2).transpose(1, 2))
        return output


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

    def test_single_spectral_basis_is_constructed_lazily(self):
        latent, _bases, _update = self.make_inputs(batch=1)
        with mock.patch(
            "AttentionGuidance.latent_renderer.build_spectral_bases",
            side_effect=AssertionError("full spectral decomposition was called"),
        ):
            basis = build_spectral_basis(latent, "spectral_low")
        self.assertEqual(tuple(basis.shape), (1, 4, 8, 8))
        self.assertTrue(torch.isfinite(basis).all())

    def test_lazy_provider_empty_request_is_six_zero_slots(self):
        latent, _bases, update = self.make_inputs(batch=1)
        provider = LazyLatentStructureBasisProvider(
            None,
            batch_size=1,
            do_classifier_free_guidance=False,
            requested_bases=[],
            required_hook_names=[],
            scheduler_mapping="euler_clean_endpoint",
            basis_normalization="match_rms",
            provider_provenance_id="fixture-provider-v1",
        )
        observation = RendererObservation(
            latents_before_step=latent,
            pred_original_sample=latent,
            scheduler_update=update,
            step_index=0,
            timestep=torch.tensor([1.0]),
            normalized_timestep=torch.tensor([0.0]),
        )
        condition = provider(observation)
        self.assertEqual(tuple(condition.bases.shape), (1, 6, 4, 8, 8))
        self.assertTrue(torch.equal(condition.bases, torch.zeros_like(condition.bases)))
        self.assertEqual(provider.last_diagnostics["requested_bases"], [])
        self.assertEqual(provider.last_diagnostics["constructed_bases"], [])
        self.assertEqual(provider.last_diagnostics["registered_hook_names"], [])
        self.assertEqual(provider.last_diagnostics["basis_rms"], [[0.0] * 6])

    def test_lazy_provider_constructs_only_requested_band_and_no_hooks(self):
        latent, _bases, update = self.make_inputs(batch=1)
        provider = LazyLatentStructureBasisProvider(
            None,
            batch_size=1,
            do_classifier_free_guidance=False,
            requested_bases=["spectral_low"],
            required_hook_names=[],
        )
        observation = RendererObservation(
            latents_before_step=latent,
            pred_original_sample=latent,
            scheduler_update=update,
            step_index=0,
            timestep=torch.tensor([1.0]),
            normalized_timestep=torch.tensor([0.0]),
        )
        with mock.patch(
            "AttentionGuidance.latent_renderer.build_spectral_bases",
            side_effect=AssertionError("full spectral decomposition was called"),
        ):
            condition = provider(observation)
        diagnostics = provider.last_diagnostics
        self.assertEqual(diagnostics["requested_bases"], ["spectral_low"])
        self.assertEqual(diagnostics["constructed_bases"], ["spectral_low"])
        self.assertEqual(diagnostics["registered_hook_names"], [])
        self.assertEqual(tuple(condition.bases.shape), (1, 6, 4, 8, 8))
        self.assertEqual(diagnostics["basis_rms"][0][0], 0.0)
        self.assertGreater(diagnostics["basis_rms"][0][1], 0.0)
        self.assertEqual(diagnostics["basis_rms"][0][2:], [0.0] * 4)

    def test_structural_provider_records_explicit_native_contract(self):
        fake_unet = _FakeUNet()
        fake_unet.attn = _FakeAttention(channels=4)
        provider = StructuralUNetBasisProvider(
            fake_unet,
            batch_size=1,
            do_classifier_free_guidance=False,
            semantic_layer=None,
            prompt_dim=0,
            state_dim=0,
            scheduler_mapping="euler_clean_endpoint",
            basis_normalization="match_rms",
            provider_provenance_id="historical-provider-v1",
        )
        latent, _bases, update = self.make_inputs(batch=1)
        observation = RendererObservation(
            latents_before_step=latent,
            pred_original_sample=latent,
            scheduler_update=update,
            step_index=0,
            timestep=torch.tensor([1.0]),
            normalized_timestep=torch.tensor([0.0]),
        )
        with provider.capture_forward():
            fake_unet(latent, latent)
            provider(observation)
        self.assertEqual(
            provider.last_diagnostics["scheduler_mapping"],
            "euler_clean_endpoint",
        )
        self.assertEqual(provider.last_diagnostics["basis_normalization"], "match_rms")
        self.assertEqual(
            provider.last_diagnostics["provider_provenance_id"],
            "historical-provider-v1",
        )

    def test_structural_provider_rejects_requested_subset(self):
        with self.assertRaisesRegex(ValueError, "all six canonical"):
            StructuralUNetBasisProvider(
                _FakeUNet(),
                batch_size=1,
                do_classifier_free_guidance=False,
                semantic_layer=None,
                requested_bases=["spectral_low"],
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

    def test_euler_clean_update_matches_recomputed_derivative(self):
        torch.manual_seed(29)
        sample = torch.randn(2, 4, 8, 8)
        epsilon = torch.randn_like(sample)
        clean_delta = torch.randn_like(sample) * 0.05
        sigma_from = torch.tensor([2.0, 4.0])
        sigma_to = torch.tensor([1.0, 3.0])
        predicted = sample - sigma_from[:, None, None, None] * epsilon
        guided = predicted + clean_delta
        base_prev = sample + epsilon * (
            sigma_to - sigma_from
        )[:, None, None, None]
        guided_epsilon = (
            sample - guided
        ) / sigma_from[:, None, None, None]
        expected = sample + guided_epsilon * (
            sigma_to - sigma_from
        )[:, None, None, None]
        actual = inject_euler_clean_update(
            base_prev,
            predicted,
            guided,
            sigma_from=sigma_from,
            sigma_to=sigma_to,
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            actual - base_prev,
            clean_delta * torch.tensor([0.5, 0.25])[:, None, None, None],
        )

    def test_euler_clean_update_zero_delta_is_bitwise_identity(self):
        prev = torch.randn(1, 4, 8, 8, dtype=torch.float16)
        predicted = torch.randn_like(prev)
        actual = inject_euler_clean_update(
            prev,
            predicted,
            predicted,
            sigma_from=3.0,
            sigma_to=2.0,
        )
        self.assertTrue(torch.equal(actual, prev))

    def test_euler_clean_update_rejects_invalid_sigmas_and_batch(self):
        prev = torch.randn(2, 4, 8, 8)
        with self.assertRaisesRegex(ValueError, "positive"):
            euler_clean_update_gain(0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "sigma_to"):
            euler_clean_update_gain(1.0, 2.0)
        with self.assertRaisesRegex(ValueError, "one value per batch"):
            inject_euler_clean_update(
                prev,
                prev,
                prev,
                sigma_from=torch.tensor([3.0, 2.0, 1.0]),
                sigma_to=torch.tensor([2.0, 1.0, 0.0]),
            )

    def test_frozen_euler_schedule_exposes_legacy_amplification(self):
        gain = euler_clean_update_gain(1.093915224, 1.023925781)
        self.assertAlmostEqual(float(gain), 0.063980683, places=8)
        self.assertGreater(1.0 / float(gain), 15.6)

    def test_euler_endpoint_round_trip_for_all_prediction_types(self):
        torch.manual_seed(37)
        sample = torch.randn(2, 4, 8, 8)
        model_output = torch.randn_like(sample)
        for prediction_type in ("epsilon", "sample", "v_prediction"):
            endpoint = prepare_euler_clean_endpoint(
                sample,
                model_output,
                sigma_from=2.5,
                sigma_to=1.75,
                prediction_type=prediction_type,
            )
            reconstructed = euler_model_output_from_clean_sample(
                sample,
                endpoint.pred_original_sample,
                sigma_from=2.5,
                prediction_type=prediction_type,
                output_dtype=model_output.dtype,
            )
            torch.testing.assert_close(reconstructed, model_output)
            torch.testing.assert_close(
                endpoint.nominal_update,
                endpoint.clean_update_gain.reshape(1, 1, 1, 1)
                * (endpoint.pred_original_sample - sample),
            )

    def test_euler_endpoint_rejects_unsupported_prediction_and_vector_sigma(self):
        sample = torch.randn(2, 4, 8, 8)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            prepare_euler_clean_endpoint(
                sample,
                sample,
                sigma_from=1.0,
                sigma_to=0.5,
                prediction_type="flow_prediction",
            )
        with self.assertRaisesRegex(ValueError, "scalar sigmas"):
            prepare_euler_clean_endpoint(
                sample,
                sample,
                sigma_from=torch.tensor([2.0, 1.0]),
                sigma_to=0.5,
                prediction_type="epsilon",
            )

    def test_euler_endpoint_matches_diffusers_native_step(self):
        from diffusers import EulerDiscreteScheduler

        torch.manual_seed(41)
        sample = torch.randn(1, 4, 8, 8)
        model_output = torch.randn_like(sample)
        clean_delta = torch.randn_like(sample) * 0.03
        for prediction_type in ("epsilon", "sample", "v_prediction"):
            schedulers = [
                EulerDiscreteScheduler(
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    prediction_type=prediction_type,
                    timestep_spacing="leading",
                    steps_offset=1,
                )
                for _ in range(2)
            ]
            for scheduler in schedulers:
                scheduler.set_timesteps(4)
                scheduler.scale_model_input(sample, scheduler.timesteps[0])
            base_scheduler, guided_scheduler = schedulers
            step_index = base_scheduler.step_index
            sigma_from = base_scheduler.sigmas[step_index]
            sigma_to = base_scheduler.sigmas[step_index + 1]
            endpoint = prepare_euler_clean_endpoint(
                sample,
                model_output,
                sigma_from=sigma_from,
                sigma_to=sigma_to,
                prediction_type=prediction_type,
            )
            base = base_scheduler.step(
                model_output,
                base_scheduler.timesteps[0],
                sample,
            )
            guided_model_output = euler_model_output_from_clean_sample(
                sample,
                endpoint.pred_original_sample + clean_delta,
                sigma_from=sigma_from,
                prediction_type=prediction_type,
                output_dtype=model_output.dtype,
            )
            guided = guided_scheduler.step(
                guided_model_output,
                guided_scheduler.timesteps[0],
                sample,
            )
            expected = (
                base.prev_sample
                + endpoint.clean_update_gain.reshape(1, 1, 1, 1) * clean_delta
            )
            torch.testing.assert_close(guided.prev_sample, expected)

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

    def test_fixed_coefficient_renderer_emits_constant_action(self):
        latent, bases, scheduler_update = self.make_inputs()
        coefficients = [0.1, -0.05, 0.02, 0.0]
        renderer = build_fixed_coefficient_renderer(
            coefficients, max_update_ratio=0.1
        )
        output = renderer(
            latent,
            bases,
            timestep=torch.tensor([0.2, 0.8]),
            scheduler_update=scheduler_update,
        )
        torch.testing.assert_close(
            output.coefficients,
            torch.tensor(coefficients).expand(2, -1),
            rtol=1e-5,
            atol=1e-6,
        )
        with self.assertRaises(ValueError):
            build_fixed_coefficient_renderer([1.0, 0.0])

    def test_match_rms_basis_normalization_is_resolution_stable(self):
        for size in (8, 16):
            latent, bases, _scheduler_update = self.make_inputs(
                batch=2, channels=4, height=size, width=size
            )
            renderer = StructuralLatentRenderer(
                LatentRendererConfig(
                    num_bases=4,
                    basis_normalization="match_rms",
                )
            )
            prepared = renderer._prepare_bases(latent, bases)
            latent_rms = latent.float().square().mean(dim=(1, 2, 3)).sqrt()
            basis_rms = prepared.square().mean(dim=(2, 3, 4)).sqrt()
            torch.testing.assert_close(
                basis_rms,
                latent_rms[:, None].expand_as(basis_rms),
                rtol=1e-5,
                atol=1e-6,
            )

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

    def test_euler_gain_caps_the_applied_scheduler_update(self):
        latent, bases, scheduler_update = self.make_inputs()
        renderer = StructuralLatentRenderer(
            LatentRendererConfig(num_bases=4, max_update_ratio=0.1)
        )
        with torch.no_grad():
            renderer.policy[-1].bias.fill_(8.0)
        gain = torch.tensor([0.1, 0.25])
        output = renderer(
            latent,
            bases * 1000,
            timestep=torch.tensor([0.2, 0.8]),
            scheduler_update=scheduler_update,
            clean_update_gain=gain,
        )
        torch.testing.assert_close(output.diagnostics.clean_update_gain, gain)
        self.assertTrue(
            torch.all(output.diagnostics.applied_update_ratio <= 0.10001)
        )
        self.assertTrue(
            torch.all(
                output.diagnostics.applied_update_norm
                <= output.diagnostics.bounded_update_norm + 1e-6
            )
        )

    def test_renderer_rejects_invalid_clean_update_gain(self):
        latent, bases, scheduler_update = self.make_inputs()
        renderer = StructuralLatentRenderer(
            LatentRendererConfig(num_bases=4, max_update_ratio=0.1)
        )
        with self.assertRaisesRegex(ValueError, "0 < gain <= 1"):
            renderer(
                latent,
                bases,
                scheduler_update=scheduler_update,
                clean_update_gain=torch.tensor([0.0, 0.5]),
            )

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

    def test_spatial_head_is_zero_initialised_but_trainable(self):
        latent, bases, scheduler_update = self.make_inputs()
        renderer = StructuralLatentRenderer(
            LatentRendererConfig(
                num_bases=4,
                spatial_hidden_dim=8,
                max_update_ratio=0.1,
            )
        )
        identity = renderer(latent, bases, scheduler_update=scheduler_update)
        self.assertTrue(torch.equal(identity.guided_x0, latent))
        renderer.spatial_head["output"].weight.data.normal_(std=0.01)
        renderer.spatial_head["output"].bias.data.fill_(0.1)
        output = renderer(latent, bases, scheduler_update=scheduler_update)
        self.assertTrue(torch.isfinite(output.guided_x0).all())
        self.assertTrue(torch.all(output.diagnostics.update_ratio <= 0.10001))
        self.assertTrue(torch.all(output.diagnostics.mean_error < 1e-5))
        loss = output.guided_x0.square().mean()
        loss.backward()
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in renderer.spatial_head.parameters()
            )
        )

    def test_spatial_head_preserves_flip_equivariance(self):
        latent, bases, scheduler_update = self.make_inputs()
        renderer = StructuralLatentRenderer(
            LatentRendererConfig(num_bases=4, spatial_hidden_dim=8, max_update_ratio=0.1)
        )
        renderer.spatial_head["output"].weight.data.normal_(std=0.01)
        output = renderer(latent, bases, scheduler_update=scheduler_update)
        flipped = renderer(
            torch.flip(latent, dims=(-1,)),
            torch.flip(bases, dims=(-1,)),
            scheduler_update=torch.flip(scheduler_update, dims=(-1,)),
        )
        torch.testing.assert_close(
            flipped.guided_x0,
            torch.flip(output.guided_x0, dims=(-1,)),
            rtol=1e-4,
            atol=1e-5,
        )

    def test_spatial_head_preserves_quarter_turn_equivariance(self):
        latent, bases, scheduler_update = self.make_inputs()
        renderer = StructuralLatentRenderer(
            LatentRendererConfig(num_bases=4, spatial_hidden_dim=8, max_update_ratio=0.1)
        )
        renderer.spatial_head["output"].weight.data.normal_(std=0.01)
        output = renderer(latent, bases, scheduler_update=scheduler_update)
        rotated = renderer(
            torch.rot90(latent, 1, dims=(-2, -1)),
            torch.rot90(bases, 1, dims=(-2, -1)),
            scheduler_update=torch.rot90(scheduler_update, 1, dims=(-2, -1)),
        )
        torch.testing.assert_close(
            rotated.guided_x0,
            torch.rot90(output.guided_x0, 1, dims=(-2, -1)),
            rtol=1e-4,
            atol=1e-5,
        )

    def test_nonzero_renderer_is_d4_equivariant_given_equivariant_bases(self):
        torch.manual_seed(43)
        latent = torch.randn(1, 4, 8, 8)
        bases = torch.randn(1, 4, 4, 8, 8)
        scheduler_update = torch.randn_like(latent)
        renderer = StructuralLatentRenderer(
            LatentRendererConfig(
                num_bases=4,
                spatial_hidden_dim=8,
                max_update_ratio=0.1,
            )
        )
        with torch.no_grad():
            for parameter in renderer.parameters():
                parameter.normal_(mean=0.0, std=0.05)
        base = renderer(
            latent,
            bases,
            timestep=torch.tensor([0.37]),
            scheduler_update=scheduler_update,
            clean_update_gain=torch.tensor([0.2]),
        )

        transforms = []
        for turns in range(4):
            transforms.append(
                lambda value, turns=turns: torch.rot90(
                    value, turns, dims=(-2, -1)
                )
            )
            transforms.append(
                lambda value, turns=turns: torch.flip(
                    torch.rot90(value, turns, dims=(-2, -1)), dims=(-1,)
                )
            )
        for transform in transforms:
            transformed = renderer(
                transform(latent),
                transform(bases),
                timestep=torch.tensor([0.37]),
                scheduler_update=transform(scheduler_update),
                clean_update_gain=torch.tensor([0.2]),
            )
            torch.testing.assert_close(
                transformed.coefficients,
                base.coefficients,
                rtol=1e-5,
                atol=1e-5,
            )
            torch.testing.assert_close(
                transformed.guided_x0,
                transform(base.guided_x0),
                rtol=2e-5,
                atol=2e-5,
            )

    def test_diagnostics_have_json_safe_records(self):
        latent, bases, scheduler_update = self.make_inputs()
        renderer = StructuralLatentRenderer(
            LatentRendererConfig(num_bases=4, max_update_ratio=0.1)
        )
        output = renderer(latent, bases, scheduler_update=scheduler_update)
        record = output.diagnostics.to_record()
        self.assertEqual(len(record["update_ratio"]), 2)
        self.assertEqual(len(record["mean_error"]), 2)
        self.assertIsInstance(record["variance_error"][0], float)

    def test_structural_provider_captures_cfg_features_and_qk(self):
        torch.manual_seed(23)
        unet = _FakeUNet()
        provider = StructuralUNetBasisProvider(
            unet,
            batch_size=1,
            do_classifier_free_guidance=True,
            semantic_layer="attn",
            feature_block="up_blocks.0",
            prompt_dim=4,
            state_dim=16,
        )
        hidden = torch.randn(2, 8, 4, 4)
        skip = torch.randn_like(hidden)
        with provider.capture_forward():
            unet(hidden, skip)
        x0 = torch.randn(1, 4, 8, 8)
        observation = RendererObservation(
            latents_before_step=torch.randn_like(x0),
            pred_original_sample=x0,
            scheduler_update=torch.randn_like(x0),
            step_index=2,
            timestep=torch.tensor([500.0]),
            normalized_timestep=torch.tensor([0.5]),
            pooled_prompt_embeds=torch.randn(1, 16),
        )
        condition = provider(observation)
        self.assertEqual(condition.bases.shape, (1, 6, 4, 8, 8))
        self.assertEqual(condition.prompt_embedding.shape, (1, 4))
        self.assertEqual(condition.state_features.shape, (1, 16))
        self.assertTrue(torch.isfinite(condition.bases).all())
        self.assertTrue(torch.isfinite(condition.state_features).all())
        self.assertEqual(provider.last_diagnostics["semantic_token_grid"], [4, 4])
        self.assertEqual(provider.capture.backbone.shape[0], 2)

    def test_structural_provider_requires_a_forward_capture(self):
        provider = StructuralUNetBasisProvider(
            _FakeUNet(),
            batch_size=1,
            do_classifier_free_guidance=False,
            semantic_layer="attn",
            feature_block="up_blocks.0",
            prompt_dim=0,
            state_dim=0,
        )
        x0 = torch.randn(1, 4, 8, 8)
        observation = RendererObservation(
            latents_before_step=torch.randn_like(x0),
            pred_original_sample=x0,
            scheduler_update=torch.randn_like(x0),
            step_index=0,
            timestep=torch.tensor([999.0]),
            normalized_timestep=torch.tensor([0.0]),
        )
        with self.assertRaises(RuntimeError):
            provider(observation)

    def _lazy_observation(self, *, batch=1):
        x0 = torch.randn(batch, 4, 8, 8)
        return RendererObservation(
            latents_before_step=torch.randn_like(x0),
            pred_original_sample=x0,
            scheduler_update=torch.randn_like(x0),
            step_index=0,
            timestep=torch.tensor([999.0]).expand(batch),
            normalized_timestep=torch.tensor([0.0]).expand(batch),
        )

    def test_lazy_provider_empty_request_is_zero_identity_without_unet(self):
        self.assertEqual(normalize_latent_structure_bases([]), ())
        provider = LazyLatentStructureBasisProvider(
            None,
            batch_size=1,
            do_classifier_free_guidance=False,
            requested_bases=[],
        )
        condition = provider(self._lazy_observation())
        self.assertEqual(condition.bases.shape, (1, 6, 4, 8, 8))
        self.assertTrue(torch.equal(condition.bases, torch.zeros_like(condition.bases)))
        self.assertEqual(provider.last_diagnostics["constructed_bases"], [])
        self.assertEqual(provider.last_diagnostics["registered_hook_names"], [])

    def test_lazy_spectral_laplacian_provider_installs_no_hooks(self):
        provider = LazyLatentStructureBasisProvider(
            None,
            batch_size=1,
            do_classifier_free_guidance=False,
            requested_bases=["spectral_low", "laplacian"],
        )
        condition = provider(self._lazy_observation())
        self.assertEqual(condition.bases.shape[1], 6)
        self.assertGreater(float(condition.bases[:, 1].abs().sum()), 0.0)
        self.assertGreater(float(condition.bases[:, 5].abs().sum()), 0.0)
        self.assertTrue(torch.equal(condition.bases[:, 0], torch.zeros_like(condition.bases[:, 0])))
        self.assertEqual(provider.required_hook_names, ())
        self.assertEqual(provider.last_diagnostics["registered_hook_names"], [])

    def test_lazy_semantic_and_freeu_requests_use_only_matching_hooks(self):
        semantic = LazyLatentStructureBasisProvider(
            _FakeUNet(),
            batch_size=1,
            do_classifier_free_guidance=False,
            requested_bases=["semantic"],
            semantic_layer="attn",
            feature_block="up_blocks.0",
            prompt_dim=0,
            state_dim=0,
        )
        self.assertEqual(semantic.required_hook_names, ("attn_qk",))
        self.assertIsNone(semantic._feature_capture)
        self.assertEqual(len(semantic._semantic_module.to_q._forward_hooks), 0)
        with semantic.capture_forward():
            self.assertEqual(len(semantic._semantic_module.to_q._forward_hooks), 1)
        self.assertEqual(len(semantic._semantic_module.to_q._forward_hooks), 0)

        freeu = LazyLatentStructureBasisProvider(
            _FakeUNet(),
            batch_size=1,
            do_classifier_free_guidance=False,
            requested_bases=["freeu"],
            semantic_layer="missing",
            feature_block="up_blocks.0",
            prompt_dim=0,
            state_dim=0,
        )
        self.assertEqual(freeu.required_hook_names, ("up_blocks.0_backbone_skip",))
        self.assertIsNone(freeu._semantic_capture)
        self.assertEqual(len(freeu.capture.feature_module._forward_pre_hooks), 0)
        with freeu.capture_forward():
            self.assertEqual(len(freeu.capture.feature_module._forward_pre_hooks), 1)
        self.assertEqual(len(freeu.capture.feature_module._forward_pre_hooks), 0)

    def test_lazy_provider_rejects_mismatched_required_hooks(self):
        with self.assertRaises(ValueError):
            LazyLatentStructureBasisProvider(
                None,
                batch_size=1,
                do_classifier_free_guidance=False,
                requested_bases=["spectral_low"],
                required_hook_names=["unexpected"],
            )


if __name__ == "__main__":
    unittest.main()
