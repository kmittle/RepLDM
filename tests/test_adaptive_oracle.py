from contextlib import contextmanager
from dataclasses import FrozenInstanceError
import hashlib
import json
import sys
from types import MethodType, SimpleNamespace
import unittest
from unittest import mock
import warnings

import torch
from torch import nn

from AttentionGuidance.adaptive_oracle import (
    ADAPTIVE_ORACLE_BASIS_PROVIDER_ID,
    AdaptiveOracleBasisProvider,
    AdaptiveOracleFeatureCapture,
    AdaptiveOracleRandomContext,
    FixedRatioMomentGeodesicRenderer,
)
from AttentionGuidance.latent_renderer import RendererCondition, RendererObservation
from AttentionGuidance.latent_renderer import (
    euler_model_output_from_clean_sample,
    prepare_euler_clean_endpoint,
)
from AttentionGuidance.local_relational_basis import LocalRelationalBasisProvider


def tearDownModule():
    # Importing the pinned diffusers scheduler creates this process-global
    # torch JIT workspace; close it explicitly so warning-as-error gates stay clean.
    instantiator = sys.modules.get("torch.distributed.nn.jit.instantiator")
    temporary = getattr(instantiator, "_TEMP_DIR", None)
    if temporary is not None:
        temporary.cleanup()


class _KeywordUpBlock(nn.Module):
    def forward(self, hidden_states=None):
        return hidden_states + 1


class _FakeUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_blocks = nn.ModuleList([_KeywordUpBlock()])


class _PipelineUpBlock(nn.Module):
    def forward(self, hidden_states=None):
        return hidden_states


class _PipelineUNet(nn.Module):
    """Small deterministic denoiser that exposes the production hook shape."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(in_channels=4, sample_size=128)
        self.up_blocks = nn.ModuleList([_PipelineUpBlock()])
        generator = torch.Generator().manual_seed(2026082605)
        self.register_buffer(
            "feature_pattern",
            torch.randn(1, 1280, 32, 32, generator=generator),
            persistent=False,
        )
        self.register_buffer(
            "epsilon_pattern",
            torch.randn(1, 4, 16, 16, generator=generator),
            persistent=False,
        )
        self.forward_calls = 0

    @property
    def dtype(self):
        return self.feature_pattern.dtype

    @property
    def device(self):
        return self.feature_pattern.device

    def forward(
        self,
        sample,
        _timestep,
        *,
        encoder_hidden_states=None,
        cross_attention_kwargs=None,
        added_cond_kwargs=None,
        return_dict=False,
    ):
        del cross_attention_kwargs, added_cond_kwargs
        if return_dict:
            raise AssertionError("production smoke must request tuple U-Net output")
        self.forward_calls += 1
        feature = self.feature_pattern.expand(sample.shape[0], -1, -1, -1)
        self.up_blocks[0](hidden_states=feature)
        pattern = torch.nn.functional.interpolate(
            self.epsilon_pattern,
            size=sample.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        conditioning = encoder_hidden_states.float().mean(dim=(1, 2)).reshape(
            sample.shape[0], 1, 1, 1
        )
        return (0.03 * sample + 0.05 * pattern + 0.001 * conditioning,)


class _PipelineVAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            block_out_channels=(1, 1, 1, 1, 1, 1, 1),
            force_upcast=False,
            scaling_factor=1.0,
        )
        self.register_buffer("dtype_anchor", torch.zeros(()), persistent=False)
        self.decode_calls = 0

    @property
    def dtype(self):
        return self.dtype_anchor.dtype

    @property
    def device(self):
        return self.dtype_anchor.device

    def decode(self, latent, return_dict=False):
        if return_dict:
            raise AssertionError("production smoke must request tuple VAE output")
        self.decode_calls += 1
        return (latent[:, :3].float(),)


class _PipelineImageProcessor:
    @staticmethod
    def postprocess(image, output_type):
        if output_type != "pil":
            raise AssertionError("production smoke freezes the final decode path")
        return [image.detach().clone()]


class _PipelineProgress:
    def __init__(self, total):
        self.total = int(total)
        self.updates = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None and self.updates != self.total:
            raise AssertionError("production progress count differs from scheduler steps")
        return False

    def update(self):
        self.updates += 1


class AdaptiveOracleProductionPipelineTest(unittest.TestCase):
    @staticmethod
    def make_scheduler():
        from diffusers import EulerDiscreteScheduler

        return EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            prediction_type="epsilon",
            timestep_spacing="leading",
            steps_offset=1,
        )

    def make_pipeline(self):
        from InferencePipelines.RepLDM.pipeline_repldm_sdxl import (
            RepLDMSDXLPipeline,
        )

        unet = _PipelineUNet()
        vae = _PipelineVAE()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pipe = RepLDMSDXLPipeline(
                vae=vae,
                text_encoder=None,
                text_encoder_2=None,
                tokenizer=None,
                tokenizer_2=None,
                unet=unet,
                scheduler=self.make_scheduler(),
                add_watermarker=False,
            )

        pipe.check_inputs = MethodType(lambda self, *args, **kwargs: None, pipe)

        def encode_prompt(self, **kwargs):
            device = kwargs["device"]
            positive = torch.full((1, 2, 4), 0.25, device=device)
            negative = torch.full((1, 2, 4), -0.25, device=device)
            positive_pooled = torch.full((1, 4), 0.5, device=device)
            negative_pooled = torch.full((1, 4), -0.5, device=device)
            return positive, negative, positive_pooled, negative_pooled

        def get_add_time_ids(self, *args, dtype, **kwargs):
            return torch.zeros(1, 6, dtype=dtype)

        pipe.encode_prompt = MethodType(encode_prompt, pipe)
        pipe._get_add_time_ids = MethodType(get_add_time_ids, pipe)
        pipe.image_processor = _PipelineImageProcessor()
        pipe.maybe_free_model_hooks = MethodType(lambda self: None, pipe)
        pipe.progress_bar = MethodType(
            lambda self, iterable=None, total=None: _PipelineProgress(total),
            pipe,
        )
        return pipe, unet, vae

    @staticmethod
    def tensor_sha256(value):
        payload = (
            value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
        )
        return hashlib.sha256(payload).hexdigest()

    def test_production_pipeline_isolates_p0_feature_and_random_trajectories(self):
        pipe, unet, vae = self.make_pipeline()
        raw_latents = torch.randn(
            1, 4, 16, 16, generator=torch.Generator().manual_seed(2026082606)
        )
        raw_latents_before = raw_latents.clone()
        schedulers = []

        def run_action(name, *, renderer=None, provider=None):
            scheduler = self.make_scheduler()
            schedulers.append(scheduler)
            pipe.scheduler = scheduler
            calls_before = unet.forward_calls
            decode_before = vae.decode_calls
            with mock.patch("builtins.print"):
                output = pipe(
                    prompt="registered production path smoke",
                    height=1024,
                    width=1024,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    guidance_rescale=0.0,
                    latents=raw_latents,
                    output_type="pil",
                    record_latent_audit=True,
                    attn_guidance_scale=0.0,
                    latent_renderer=renderer,
                    latent_renderer_basis_provider=provider,
                    latent_renderer_scheduler_mapping=(
                        "euler_clean_endpoint" if renderer is not None else "legacy_unit"
                    ),
                )

            self.assertEqual(unet.forward_calls - calls_before, 50)
            self.assertEqual(vae.decode_calls - decode_before, 1)
            self.assertEqual(pipe._last_unet_calls_total, 50)
            self.assertEqual(pipe._last_unet_calls_per_step, [1] * 50)
            self.assertEqual(pipe._last_scheduler_calls_total, 50)
            self.assertEqual(pipe._last_scheduler_calls_per_step, [1] * 50)
            self.assertEqual(pipe._last_final_decode_calls, 1)
            self.assertEqual(pipe._last_intermediate_decode_calls, 0)
            self.assertEqual(scheduler.step_index, 50)
            self.assertEqual(len(pipe._last_latents_before_step_sha256), 50)
            self.assertEqual(len(pipe._last_latents_after_step_sha256), 50)
            self.assertEqual(
                pipe._last_prepared_initial_latent_sha256,
                pipe._last_latents_before_step_sha256[0],
            )
            self.assertEqual(
                pipe._last_latents_before_step_sha256[1:],
                pipe._last_latents_after_step_sha256[:-1],
            )
            self.assertEqual(len(pipe._last_scheduler_schedule_record["timesteps"]), 50)
            self.assertEqual(len(pipe._last_scheduler_schedule_record["sigmas"]), 51)
            self.assertEqual(len(unet.up_blocks[0]._forward_pre_hooks), 0)

            diagnostics = json.loads(
                json.dumps(pipe._last_latent_renderer_step_diagnostics)
            )
            return {
                "name": name,
                "diagnostics": diagnostics,
                "initial_sha256": pipe._last_prepared_initial_latent_sha256,
                "final_latent_sha256": pipe._last_latents_after_step_sha256[-1],
                "output": output[0].clone(),
                "output_sha256": self.tensor_sha256(output[0]),
            }

        p0_first = run_action("p0-first")
        feature_positive = run_action(
            "feature-positive",
            renderer=FixedRatioMomentGeodesicRenderer(sign=1),
            provider=AdaptiveOracleBasisProvider(
                unet,
                batch_size=1,
                affinity_source="feature",
                orbit_name="axis-r1",
            ),
        )
        random_negative = run_action(
            "random-negative",
            renderer=FixedRatioMomentGeodesicRenderer(sign=-1),
            provider=AdaptiveOracleBasisProvider(
                batch_size=1,
                affinity_source="random_edge",
                orbit_name="diagonal-r1",
                random_context=AdaptiveOracleRandomContext(
                    experiment_id="adaptive-oracle-production-test",
                    split_role="engineering",
                    prompt_row_id="smoke-0001",
                    seed=2026082606,
                ),
            ),
        )
        p0_repeat = run_action("p0-repeat")

        records = (p0_first, feature_positive, random_negative, p0_repeat)
        self.assertEqual(len({id(scheduler) for scheduler in schedulers}), 4)
        self.assertEqual(len({record["initial_sha256"] for record in records}), 1)
        self.assertTrue(torch.equal(raw_latents, raw_latents_before))
        self.assertEqual(p0_first["diagnostics"], [])
        self.assertEqual(p0_repeat["diagnostics"], [])
        self.assertEqual(
            p0_first["final_latent_sha256"], p0_repeat["final_latent_sha256"]
        )
        self.assertEqual(p0_first["output_sha256"], p0_repeat["output_sha256"])
        self.assertTrue(torch.equal(p0_first["output"], p0_repeat["output"]))
        self.assertEqual(
            len(
                {
                    p0_first["final_latent_sha256"],
                    feature_positive["final_latent_sha256"],
                    random_negative["final_latent_sha256"],
                }
            ),
            3,
        )

        for record in (feature_positive, random_negative):
            self.assertEqual(len(record["diagnostics"]), 50)
            for step_index, diagnostic in enumerate(record["diagnostics"]):
                self.assertEqual(diagnostic["step_index"], step_index)
                mapped = diagnostic["scheduler_mapped_intervention"]
                self.assertLessEqual(mapped["target_ratio_error"][0], 5e-4)
                self.assertFalse(mapped["cap_hit"][0])

        for diagnostic in feature_positive["diagnostics"]:
            capture = diagnostic["provider_diagnostics"]["capture_record"]
            self.assertEqual(capture["hook_calls"], 1)
            self.assertEqual(capture["consume_calls"], 1)
            self.assertTrue(capture["capture_complete"])
            self.assertTrue(capture["detached"])
            self.assertEqual(capture["conditional_rows"], "second_half")

        for diagnostic in random_negative["diagnostics"]:
            provider_record = diagnostic["provider_diagnostics"]
            self.assertNotIn("capture_record", provider_record)
            self.assertEqual(
                provider_record["random_counter_context"]["step_index"],
                diagnostic["step_index"],
            )


class AdaptiveOracleFeatureCaptureTest(unittest.TestCase):
    def make_capture(self):
        unet = _FakeUNet()
        capture = AdaptiveOracleFeatureCapture(
            unet,
            batch_size=1,
            expected_channels=8,
            expected_size=(4, 4),
        )
        return unet, capture

    def test_captures_exact_detached_conditional_half_once(self):
        unet, capture = self.make_capture()
        hidden = torch.randn(2, 8, 4, 4, requires_grad=True)
        expected = hidden[1:].detach().clone()
        with capture.capture_forward():
            unet.up_blocks[0](hidden_states=hidden)
        hidden.data.zero_()
        observed = capture.conditional_feature()
        self.assertTrue(torch.equal(observed, expected))
        self.assertFalse(observed.requires_grad)
        self.assertEqual(capture.to_record()["hook_calls"], 1)
        self.assertEqual(capture.to_record()["consume_calls"], 1)
        with self.assertRaisesRegex(RuntimeError, "only once"):
            capture.conditional_feature()

    def test_requires_one_forward_and_registered_shape(self):
        unet, capture = self.make_capture()
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            with capture.capture_forward():
                pass
        with self.assertRaisesRegex(RuntimeError, "registered CFG shape"):
            with capture.capture_forward():
                unet.up_blocks[0](hidden_states=torch.randn(1, 8, 4, 4))

    def test_rejects_positional_or_multiple_calls_and_removes_hook(self):
        unet, capture = self.make_capture()
        hidden = torch.randn(2, 8, 4, 4)
        with self.assertRaisesRegex(RuntimeError, "keyword hidden_states"):
            with capture.capture_forward():
                unet.up_blocks[0](hidden)
        self.assertEqual(len(unet.up_blocks[0]._forward_pre_hooks), 0)
        with self.assertRaisesRegex(RuntimeError, "more than once"):
            with capture.capture_forward():
                unet.up_blocks[0](hidden_states=hidden)
                unet.up_blocks[0](hidden_states=hidden)
        self.assertEqual(len(unet.up_blocks[0]._forward_pre_hooks), 0)

    def test_capture_cannot_be_nested_or_consumed_early(self):
        unet, capture = self.make_capture()
        with self.assertRaisesRegex(RuntimeError, "no complete"):
            capture.conditional_feature()
        with capture.capture_forward():
            with self.assertRaisesRegex(RuntimeError, "nested"):
                with capture.capture_forward():
                    pass
            unet.up_blocks[0](hidden_states=torch.randn(2, 8, 4, 4))


class AdaptiveOracleBasisProviderTest(unittest.TestCase):
    RANDOM_CONTEXT = AdaptiveOracleRandomContext(
        experiment_id="ao-search-v1",
        split_role="search",
        prompt_row_id="search-0001",
        seed=123456789,
    )

    @staticmethod
    def make_observation(step_index=0, *, batch=1, requires_grad=False):
        generator = torch.Generator().manual_seed(2026082601)
        x0 = torch.randn(batch, 4, 16, 16, generator=generator)
        x0.requires_grad_(requires_grad)
        return RendererObservation(
            latents_before_step=torch.randn(
                batch, 4, 16, 16, generator=generator
            ),
            pred_original_sample=x0,
            scheduler_update=torch.randn(
                batch, 4, 16, 16, generator=generator
            ),
            step_index=step_index,
            timestep=torch.tensor([999.0]),
            normalized_timestep=torch.tensor([1.0]),
        )

    def test_feature_mode_uses_one_hook_and_selects_exact_orbit(self):
        unet = _FakeUNet()
        provider = AdaptiveOracleBasisProvider(
            unet,
            batch_size=1,
            affinity_source="feature",
            orbit_name="diagonal-r1",
        )
        observation = self.make_observation(requires_grad=True)
        generator = torch.Generator().manual_seed(2026082602)
        hidden = torch.randn(
            2, 1280, 32, 32, generator=generator, requires_grad=True
        )

        self.assertEqual(len(unet.up_blocks[0]._forward_pre_hooks), 0)
        with provider.capture_forward():
            self.assertEqual(len(unet.up_blocks[0]._forward_pre_hooks), 1)
            unet.up_blocks[0](hidden_states=hidden)
        self.assertEqual(len(unet.up_blocks[0]._forward_pre_hooks), 0)

        condition = provider(observation)
        expected, _diagnostics = LocalRelationalBasisProvider()(
            observation.pred_original_sample,
            hidden[1:].detach(),
        )
        self.assertIsInstance(condition, RendererCondition)
        torch.testing.assert_close(
            condition.bases, expected[:, 1:2], rtol=0, atol=0
        )
        self.assertEqual(condition.bases.shape, (1, 1, 4, 16, 16))
        self.assertIsNone(condition.prompt_embedding)
        self.assertIsNone(condition.state_features)
        self.assertFalse(condition.bases.requires_grad)
        self.assertIsNone(condition.bases.grad_fn)

        record = provider.last_diagnostics
        self.assertEqual(
            record["implementation"], ADAPTIVE_ORACLE_BASIS_PROVIDER_ID
        )
        self.assertEqual(record["affinity_source"], "feature")
        self.assertEqual(record["selected_orbit"], "diagonal-r1")
        self.assertEqual(record["selected_orbit_index"], 1)
        self.assertEqual(record["capture_record"]["hook_calls"], 1)
        self.assertEqual(record["capture_record"]["consume_calls"], 1)
        self.assertTrue(record["capture_record"]["capture_complete"])
        self.assertNotIn("random_counter_context", record)
        with self.assertRaisesRegex(RuntimeError, "only once"):
            provider(observation)

    def test_non_feature_modes_install_no_hook_or_feature_capture(self):
        observation = self.make_observation()
        unet = _FakeUNet()
        modes = ("uniform_local", "predicted_clean", "random_edge")
        for affinity_source in modes:
            with self.subTest(affinity_source=affinity_source):
                kwargs = {}
                if affinity_source == "random_edge":
                    kwargs["random_context"] = self.RANDOM_CONTEXT
                with mock.patch(
                    "AttentionGuidance.adaptive_oracle.AdaptiveOracleFeatureCapture",
                    side_effect=AssertionError("feature capture was constructed"),
                ):
                    provider = AdaptiveOracleBasisProvider(
                        unet,
                        batch_size=1,
                        affinity_source=affinity_source,
                        orbit_name="axis-r2",
                        **kwargs,
                    )
                self.assertIsNone(provider.capture)
                with provider.capture_forward():
                    self.assertEqual(len(unet.up_blocks[0]._forward_pre_hooks), 0)
                condition = provider(observation)
                self.assertEqual(condition.bases.shape, (1, 1, 4, 16, 16))
                self.assertNotIn("capture_record", provider.last_diagnostics)

    def test_control_modes_select_the_exact_public_basis(self):
        observation = self.make_observation(step_index=7)
        local = LocalRelationalBasisProvider()
        controls = (
            (
                "uniform_local",
                {},
                lambda: local.uniform_local(observation.pred_original_sample),
            ),
            (
                "predicted_clean",
                {},
                lambda: local.predicted_clean(observation.pred_original_sample),
            ),
            (
                "random_edge",
                {"random_context": self.RANDOM_CONTEXT},
                lambda: local.random_edge(
                    observation.pred_original_sample,
                    experiment_id=self.RANDOM_CONTEXT.experiment_id,
                    split_role=self.RANDOM_CONTEXT.split_role,
                    prompt_row_id=self.RANDOM_CONTEXT.prompt_row_id,
                    seed=self.RANDOM_CONTEXT.seed,
                    step_index=observation.step_index,
                ),
            ),
        )
        for affinity_source, kwargs, build_expected in controls:
            with self.subTest(affinity_source=affinity_source):
                provider = AdaptiveOracleBasisProvider(
                    batch_size=1,
                    affinity_source=affinity_source,
                    orbit_name="axis-r1",
                    **kwargs,
                )
                condition = provider(observation)
                expected, _diagnostics = build_expected()
                torch.testing.assert_close(
                    condition.bases, expected[:, 0:1], rtol=0, atol=0
                )

    def test_random_context_is_immutable_and_step_keyed(self):
        context = AdaptiveOracleRandomContext(
            experiment_id="ao-search-v1",
            split_role="search",
            prompt_row_id="search-0008",
            seed=987654321,
        )
        provider = AdaptiveOracleBasisProvider(
            batch_size=1,
            affinity_source="random_edge",
            orbit_name="diagonal-r1",
            random_context=context,
        )
        step_zero = provider(self.make_observation(step_index=0)).bases
        zero_record = json.loads(json.dumps(provider.last_diagnostics))
        step_one = provider(self.make_observation(step_index=1)).bases
        one_record = json.loads(json.dumps(provider.last_diagnostics))
        repeated = provider(self.make_observation(step_index=0)).bases
        repeated_record = json.loads(json.dumps(provider.last_diagnostics))

        self.assertFalse(torch.equal(step_zero, step_one))
        self.assertTrue(torch.equal(step_zero, repeated))
        self.assertNotEqual(
            zero_record["selected_basis_sha256"],
            one_record["selected_basis_sha256"],
        )
        self.assertEqual(zero_record["random_counter_context"]["step_index"], 0)
        self.assertEqual(one_record["random_counter_context"]["step_index"], 1)
        self.assertEqual(
            zero_record["random_counter_context"]["orbit_name"], "diagonal-r1"
        )
        self.assertEqual(len(zero_record["random_counter_set_sha256"]), 64)
        self.assertNotEqual(
            zero_record["random_counter_set_sha256"],
            one_record["random_counter_set_sha256"],
        )
        self.assertEqual(
            zero_record["random_counter_set_sha256"],
            repeated_record["random_counter_set_sha256"],
        )
        with self.assertRaises(FrozenInstanceError):
            context.seed = 1
        with self.assertRaises(AttributeError):
            provider.random_context = self.RANDOM_CONTEXT

    def test_invalid_adapter_and_counter_contexts_fail_closed(self):
        invalid_contexts = (
            {"experiment_id": "bad id"},
            {"split_role": ""},
            {"prompt_row_id": "row/1"},
            {"seed": True},
            {"seed": -1},
            {"seed": 1.5},
        )
        baseline = {
            "experiment_id": "ao-search-v1",
            "split_role": "search",
            "prompt_row_id": "search-0001",
            "seed": 1,
        }
        for change in invalid_contexts:
            with self.subTest(change=change):
                values = {**baseline, **change}
                with self.assertRaises(ValueError):
                    AdaptiveOracleRandomContext(**values)

        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            AdaptiveOracleBasisProvider(
                batch_size=2,
                affinity_source="uniform_local",
                orbit_name="axis-r1",
            )
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            AdaptiveOracleBasisProvider(
                batch_size=1.0,
                affinity_source="uniform_local",
                orbit_name="axis-r1",
            )
        with self.assertRaisesRegex(ValueError, "affinity_source"):
            AdaptiveOracleBasisProvider(
                batch_size=1,
                affinity_source="unknown",
                orbit_name="axis-r1",
            )
        with self.assertRaisesRegex(ValueError, "orbit_name"):
            AdaptiveOracleBasisProvider(
                batch_size=1,
                affinity_source="uniform_local",
                orbit_name="axis-r3",
            )
        with self.assertRaisesRegex(ValueError, "requires a U-Net"):
            AdaptiveOracleBasisProvider(
                batch_size=1,
                affinity_source="feature",
                orbit_name="axis-r1",
            )
        with self.assertRaisesRegex(ValueError, "requires an immutable"):
            AdaptiveOracleBasisProvider(
                batch_size=1,
                affinity_source="random_edge",
                orbit_name="axis-r1",
            )
        with self.assertRaisesRegex(ValueError, "does not accept"):
            AdaptiveOracleBasisProvider(
                batch_size=1,
                affinity_source="uniform_local",
                orbit_name="axis-r1",
                random_context=self.RANDOM_CONTEXT,
            )

        provider = AdaptiveOracleBasisProvider(
            batch_size=1,
            affinity_source="random_edge",
            orbit_name="axis-r1",
            random_context=self.RANDOM_CONTEXT,
        )
        with self.assertRaisesRegex(ValueError, "step_index"):
            provider(self.make_observation(step_index=True))
        with self.assertRaisesRegex(ValueError, "step_index"):
            provider(self.make_observation(step_index=-1))
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            provider(self.make_observation(batch=2))

    def test_diagnostics_hash_detach_and_json_safety(self):
        provider = AdaptiveOracleBasisProvider(
            batch_size=1,
            affinity_source="predicted_clean",
            orbit_name="axis-r2",
        )
        condition = provider(self.make_observation(requires_grad=True))
        record = provider.last_diagnostics
        raw = bytes(
            condition.bases.contiguous()
            .cpu()
            .view(torch.uint8)
            .reshape(-1)
            .tolist()
        )

        self.assertFalse(condition.bases.requires_grad)
        self.assertIsNone(condition.bases.grad_fn)
        self.assertEqual(record["selected_basis_shape"], [1, 1, 4, 16, 16])
        self.assertEqual(record["selected_basis_dtype"], "float32")
        self.assertEqual(
            record["selected_basis_sha256"], hashlib.sha256(raw).hexdigest()
        )
        self.assertEqual(
            record["local_diagnostics"]["affinity_source"], "predicted_clean"
        )
        self.assertNotIn("capture_record", record)
        self.assertNotIn("random_counter_context", record)
        json.dumps(record, allow_nan=False, sort_keys=True)


class FixedRatioMomentGeodesicRendererTest(unittest.TestCase):
    @staticmethod
    def make_inputs(dtype=torch.float32):
        generator = torch.Generator().manual_seed(20260826)
        latent = torch.randn(2, 4, 16, 16, generator=generator).to(dtype)
        basis = torch.randn(2, 1, 4, 16, 16, generator=generator).to(dtype)
        scheduler_update = (
            0.04 * torch.randn(2, 4, 16, 16, generator=generator)
        ).to(dtype)
        gain = torch.tensor([0.15, 0.6], dtype=torch.float32)
        return latent, basis, scheduler_update, gain

    def render(self, sign, dtype=torch.float32):
        latent, basis, scheduler_update, gain = self.make_inputs(dtype)
        renderer = FixedRatioMomentGeodesicRenderer(sign=sign)
        output = renderer(
            latent,
            basis,
            scheduler_update=scheduler_update,
            clean_update_gain=gain,
        )
        return renderer, output, latent, basis, scheduler_update, gain

    def test_hits_registered_scheduler_ratio_and_preserves_moments(self):
        renderer, output, latent, _basis, _scheduler_update, _gain = self.render(1)
        diagnostics = output.diagnostics
        torch.testing.assert_close(
            diagnostics.applied_update_ratio,
            torch.full((2,), 0.02),
            rtol=1e-5,
            atol=1e-6,
        )
        self.assertTrue(torch.all(diagnostics.mean_error <= 1e-6))
        self.assertTrue(torch.all(diagnostics.variance_error <= 1e-5))
        self.assertTrue(torch.all(diagnostics.covariance_drift <= 2e-5))
        self.assertFalse(torch.any(diagnostics.cap_hit))
        self.assertEqual(renderer.parameter_count, 0)
        self.assertEqual(sum(parameter.numel() for parameter in renderer.parameters()), 0)
        self.assertEqual(output.guided_x0.dtype, latent.dtype)
        json.dumps(diagnostics.to_record(), allow_nan=False)

    def test_antithetic_endpoints_have_equal_norm_and_angle(self):
        _, positive, latent, basis, scheduler_update, gain = self.render(1)
        negative = FixedRatioMomentGeodesicRenderer(sign=-1)(
            latent,
            basis,
            scheduler_update=scheduler_update,
            clean_update_gain=gain,
        )
        torch.testing.assert_close(
            positive.diagnostics.bounded_update_norm,
            negative.diagnostics.bounded_update_norm,
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            positive.diagnostics.applied_update_ratio,
            negative.diagnostics.applied_update_ratio,
            rtol=1e-6,
            atol=1e-7,
        )
        torch.testing.assert_close(
            positive.diagnostics.angle,
            negative.diagnostics.angle,
            rtol=0,
            atol=0,
        )
        self.assertFalse(torch.equal(positive.guided_x0, negative.guided_x0))

    def test_preserves_full_channel_gram_for_correlated_latents(self):
        generator = torch.Generator().manual_seed(20260828)
        independent = torch.randn(1, 4, 16, 16, generator=generator)
        mixing = torch.tensor(
            [
                [1.0, 0.6, 0.2, 0.0],
                [0.4, 1.0, 0.3, 0.1],
                [0.2, 0.5, 1.0, 0.4],
                [0.1, 0.2, 0.5, 1.0],
            ]
        )
        latent = torch.einsum("cd,bdhw->bchw", mixing, independent)
        basis = torch.randn(1, 1, 4, 16, 16, generator=generator)
        scheduler_update = 0.04 * torch.randn(
            1, 4, 16, 16, generator=generator
        )
        output = FixedRatioMomentGeodesicRenderer(sign=1)(
            latent,
            basis,
            scheduler_update=scheduler_update,
            clean_update_gain=torch.tensor([0.2]),
        )

        def gram(value):
            centered = value.float() - value.float().mean(
                dim=(-2, -1), keepdim=True
            )
            flat = centered.flatten(2)
            return torch.bmm(flat, flat.transpose(1, 2))

        torch.testing.assert_close(
            gram(output.guided_x0), gram(latent), rtol=2e-5, atol=2e-5
        )
        self.assertLessEqual(float(output.diagnostics.covariance_drift[0]), 2e-5)

    def test_is_equivariant_under_d4_transform(self):
        latent, basis, scheduler_update, gain = self.make_inputs()
        renderer = FixedRatioMomentGeodesicRenderer(sign=1)
        base = renderer(
            latent,
            basis,
            scheduler_update=scheduler_update,
            clean_update_gain=gain,
        )
        transform = lambda value: torch.rot90(value.transpose(-1, -2), 1, (-2, -1))
        transformed = renderer(
            transform(latent),
            transform(basis),
            scheduler_update=transform(scheduler_update),
            clean_update_gain=gain,
        )
        torch.testing.assert_close(
            transformed.guided_x0,
            transform(base.guided_x0),
            rtol=2e-5,
            atol=2e-5,
        )

    def test_float16_path_remains_within_registered_tolerances(self):
        _, output, latent, _basis, _scheduler_update, _gain = self.render(
            1, torch.float16
        )
        self.assertEqual(output.guided_x0.dtype, torch.float16)
        self.assertTrue(torch.all(output.diagnostics.target_ratio_error <= 5e-4))
        self.assertTrue(torch.all(output.diagnostics.mean_error <= 1e-4))
        self.assertTrue(torch.all(output.diagnostics.variance_error <= 1e-3))

    def test_real_euler_scheduler_round_trip_for_all_fifty_steps(self):
        from diffusers import EulerDiscreteScheduler

        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                generator = torch.Generator().manual_seed(20260827)
                scheduler = EulerDiscreteScheduler(
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    prediction_type="epsilon",
                    timestep_spacing="leading",
                    steps_offset=1,
                )
                scheduler.set_timesteps(50)
                renderer = FixedRatioMomentGeodesicRenderer(sign=1)
                pred_original_errors = []
                prev_sample_errors = []
                covariance_drifts = []
                calibrated_steps = 0
                for timestep in scheduler.timesteps:
                    clean = torch.randn(
                        1, 4, 64, 64, generator=generator
                    )
                    epsilon = torch.randn(
                        1, 4, 64, 64, generator=generator
                    )
                    step_index = scheduler.step_index
                    if step_index is None:
                        step_index = 0
                    sigma_from = scheduler.sigmas[step_index]
                    sigma_to = scheduler.sigmas[step_index + 1]
                    sample = (clean + sigma_from * epsilon).to(dtype)
                    model_output = epsilon.to(dtype)
                    scheduler.scale_model_input(sample, timestep)
                    step_index = scheduler.step_index
                    endpoint = prepare_euler_clean_endpoint(
                        sample,
                        model_output,
                        sigma_from=scheduler.sigmas[step_index],
                        sigma_to=scheduler.sigmas[step_index + 1],
                        prediction_type="epsilon",
                    )
                    basis = LocalRelationalBasisProvider().uniform_local(
                        endpoint.pred_original_sample
                    )[0][:, :1]
                    mapped = renderer.forward_euler_mapped(
                        endpoint.pred_original_sample,
                        basis,
                        sample=sample,
                        native_model_output=model_output,
                        sigma_from=endpoint.sigma_from,
                        sigma_to=endpoint.sigma_to,
                        prediction_type="epsilon",
                        scheduler_update=endpoint.nominal_update,
                        clean_update_gain=endpoint.clean_update_gain,
                    )
                    rendered = mapped.rendered
                    calibrated_steps += int(mapped.solver_evaluations > 1)
                    step_output = scheduler.step(
                        mapped.model_output, timestep, sample, return_dict=True
                    )
                    self.assertTrue(
                        torch.equal(
                            step_output.prev_sample,
                            mapped.predicted_prev_sample,
                        )
                    )
                    self.assertLessEqual(
                        abs(float(mapped.mapped_intervention.ratio[0]) - 0.02),
                        renderer.ratio_tolerance,
                    )
                    self.assertIn(mapped.solver_evaluations, {1, 14})
                    covariance_drifts.append(
                        rendered.diagnostics.covariance_drift.detach()
                    )
                    expected_prev = sample.float() + endpoint.clean_update_gain.reshape(
                        1, 1, 1, 1
                    ) * (rendered.guided_x0.float() - sample.float())
                    for errors, actual, expected in (
                        (
                            pred_original_errors,
                            step_output.pred_original_sample,
                            rendered.guided_x0,
                        ),
                        (prev_sample_errors, step_output.prev_sample, expected_prev),
                    ):
                        difference = torch.linalg.vector_norm(
                            (actual.float() - expected.float()).flatten(1), dim=1
                        )
                        denominator = torch.linalg.vector_norm(
                            expected.float().flatten(1), dim=1
                        )
                        errors.append(
                            difference / (denominator + 1e-12)
                        )
                self.assertLessEqual(
                    float(torch.cat(pred_original_errors).amax()),
                    renderer.scheduler_pred_original_relative_l2_tolerance,
                )
                self.assertLessEqual(
                    float(torch.cat(prev_sample_errors).amax()),
                    renderer.scheduler_prev_sample_relative_l2_tolerance,
                )
                self.assertLessEqual(
                    float(torch.cat(covariance_drifts).amax()),
                    renderer.covariance_tolerance,
                )
                self.assertEqual(scheduler.step_index, 50)
                if dtype == torch.float16:
                    self.assertGreater(calibrated_steps, 0)

    def test_real_euler_mapped_inputs_are_reconstructed_and_fail_closed(self):
        from diffusers import EulerDiscreteScheduler

        generator = torch.Generator().manual_seed(20260829)
        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            prediction_type="epsilon",
            timestep_spacing="leading",
            steps_offset=1,
        )
        scheduler.set_timesteps(50)
        timestep = scheduler.timesteps[0]
        clean = torch.randn(1, 4, 64, 64, generator=generator)
        epsilon = torch.randn(1, 4, 64, 64, generator=generator)
        sample = (clean + scheduler.sigmas[0] * epsilon).to(torch.float16)
        model_output = epsilon.to(torch.float16)
        scheduler.scale_model_input(sample, timestep)
        step_index = scheduler.step_index
        endpoint = prepare_euler_clean_endpoint(
            sample,
            model_output,
            sigma_from=scheduler.sigmas[step_index],
            sigma_to=scheduler.sigmas[step_index + 1],
            prediction_type="epsilon",
        )
        quantized_latent = endpoint.pred_original_sample.to(torch.float16)
        quantized_update = endpoint.nominal_update.to(torch.float16)
        quantized_gain = endpoint.clean_update_gain.to(torch.float16)
        basis = LocalRelationalBasisProvider().uniform_local(
            endpoint.pred_original_sample
        )[0][:, :1]
        renderer = FixedRatioMomentGeodesicRenderer(sign=1)

        mapped = renderer.forward_euler_mapped(
            quantized_latent,
            basis,
            sample=sample,
            native_model_output=model_output,
            sigma_from=endpoint.sigma_from,
            sigma_to=endpoint.sigma_to,
            prediction_type="epsilon",
            scheduler_update=quantized_update,
            clean_update_gain=quantized_gain,
        )
        self.assertLessEqual(
            abs(float(mapped.mapped_intervention.ratio[0]) - 0.02),
            renderer.ratio_tolerance,
        )

        inconsistent_cases = (
            (
                "latent",
                {"latent": endpoint.pred_original_sample * 1.03},
                "latent differs from the reconstructed native endpoint",
            ),
            (
                "scheduler_update",
                {"scheduler_update": endpoint.nominal_update * 2.0},
                "scheduler_update differs from the reconstructed native endpoint",
            ),
            (
                "clean_update_gain",
                {"clean_update_gain": endpoint.clean_update_gain * 1.1},
                "clean_update_gain differs from the reconstructed native endpoint",
            ),
        )
        common = {
            "latent": endpoint.pred_original_sample,
            "scheduler_update": endpoint.nominal_update,
            "clean_update_gain": endpoint.clean_update_gain,
        }
        for name, replacement, message in inconsistent_cases:
            inputs = {**common, **replacement}
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                renderer.forward_euler_mapped(
                    inputs["latent"],
                    basis,
                    sample=sample,
                    native_model_output=model_output,
                    sigma_from=endpoint.sigma_from,
                    sigma_to=endpoint.sigma_to,
                    prediction_type="epsilon",
                    scheduler_update=inputs["scheduler_update"],
                    clean_update_gain=inputs["clean_update_gain"],
                )

    def test_zero_or_collinear_tangent_fails_closed(self):
        latent, _basis, scheduler_update, gain = self.make_inputs()
        renderer = FixedRatioMomentGeodesicRenderer(sign=1)
        with self.assertRaisesRegex(RuntimeError, "tangent norm"):
            renderer(
                latent,
                latent[:, None],
                scheduler_update=scheduler_update,
                clean_update_gain=gain,
            )
        with self.assertRaisesRegex(RuntimeError, "channel energy"):
            renderer(
                torch.ones_like(latent),
                torch.randn_like(latent[:, None]),
                scheduler_update=scheduler_update,
                clean_update_gain=gain,
            )
        repeated_channel = latent[:, :1].expand(-1, 4, -1, -1).clone()
        with self.assertRaisesRegex(RuntimeError, "channel Gram"):
            renderer(
                repeated_channel,
                _basis,
                scheduler_update=scheduler_update,
                clean_update_gain=gain,
            )

    def test_invalid_scheduler_contract_and_conditioning_fail_closed(self):
        latent, basis, scheduler_update, gain = self.make_inputs()
        renderer = FixedRatioMomentGeodesicRenderer(sign=1)
        with self.assertRaisesRegex(ValueError, "clean_update_gain"):
            renderer(
                latent,
                basis,
                scheduler_update=scheduler_update,
                clean_update_gain=torch.tensor([0.0, 0.2]),
            )
        with self.assertRaisesRegex(RuntimeError, "scheduler update norm"):
            renderer(
                latent,
                basis,
                scheduler_update=torch.zeros_like(scheduler_update),
                clean_update_gain=gain,
            )
        with self.assertRaisesRegex(ValueError, "prompt_embedding"):
            renderer(
                latent,
                basis,
                prompt_embedding=torch.ones(2, 1),
                scheduler_update=scheduler_update,
                clean_update_gain=gain,
            )

    def test_unreachable_or_guard_violating_geometry_fails_closed(self):
        latent, basis, scheduler_update, gain = self.make_inputs()
        huge_update = scheduler_update * 1000
        renderer = FixedRatioMomentGeodesicRenderer(sign=1)
        with self.assertRaisesRegex(RuntimeError, "geodesic reach|covariance drift"):
            renderer(
                latent,
                basis,
                scheduler_update=huge_update,
                clean_update_gain=gain,
            )

    def test_constructor_and_shape_validation(self):
        renderer = FixedRatioMomentGeodesicRenderer(sign=1)
        self.assertIs(renderer.requires_strict_scheduler_round_trip, True)
        self.assertIs(renderer.requires_strict_scheduler_mapped_ratio, True)
        self.assertEqual(
            renderer.scheduler_pred_original_relative_l2_tolerance, 0.01
        )
        self.assertEqual(
            renderer.scheduler_prev_sample_relative_l2_tolerance, 1e-3
        )
        with self.assertRaises(ValueError):
            FixedRatioMomentGeodesicRenderer(sign=0)
        with self.assertRaises(ValueError):
            FixedRatioMomentGeodesicRenderer(
                sign=1, target_update_ratio=0.05, hard_update_cap=0.05
            )
        latent, basis, scheduler_update, gain = self.make_inputs()
        with self.assertRaisesRegex(ValueError, "shape"):
            FixedRatioMomentGeodesicRenderer(sign=1)(
                latent,
                basis[:, :, :, :-1],
                scheduler_update=scheduler_update,
                clean_update_gain=gain,
            )


if __name__ == "__main__":
    unittest.main()
