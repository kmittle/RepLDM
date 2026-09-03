from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
from unittest import mock
import warnings

import pytest
import torch
from torch import nn

from AttentionGuidance.latent_renderer import (
    RendererCondition,
    RendererObservation,
    predict_euler_no_churn_prev_sample,
    prepare_euler_clean_endpoint,
)
from latent_renderer_training.authorization import TrainingAuthorization
from latent_renderer_training.collector import (
    EulerNativeRolloutCollector,
    EulerRolloutStep,
    EulerTransitionResult,
)
from latent_renderer_training.renderer import (
    CALIBRATION_STATE_COUNT,
    EulerMappedOutput,
    EulerNativeFrameV1,
    FrameCalibration,
)
from latent_renderer_training.run_contract import (
    RUN_CONTRACT_SCHEMA,
    TrainingRunContract,
)


def teardown_module() -> None:
    instantiator = sys.modules.get("torch.distributed.nn.jit.instantiator")
    temporary = getattr(instantiator, "_TEMP_DIR", None)
    if temporary is not None:
        temporary.cleanup()


def _renderer(*, channels: int) -> EulerNativeFrameV1:
    calibration = FrameCalibration(
        (True,) * 6,
        CALIBRATION_STATE_COUNT,
        (1.0,) * 6,
        (1.0,) * 6,
        manifest_sha256="a" * 64,
        source_sha256="b" * 64,
        state_provenance_sha256="c" * 64,
    )
    return EulerNativeFrameV1(calibration=calibration, latent_channels=channels)


class _CollectorUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0

    def forward(self, state: torch.Tensor, index: int) -> torch.Tensor:
        self.forward_calls += 1
        return 0.05 * state + 0.01 * (index + 1)


def _collector(
    unet: _CollectorUNet,
    *,
    forwards_per_observation: int = 1,
    transition_dtype: torch.dtype | None = None,
    verify_physical: bool = True,
) -> EulerNativeRolloutCollector:
    renderer = _renderer(channels=2)

    def observe(state: torch.Tensor, index: int) -> EulerRolloutStep:
        native_output = None
        for _ in range(forwards_per_observation):
            native_output = unet(state, index)
        if native_output is None:
            native_output = 0.05 * state + 0.01 * (index + 1)
        sigma_from = float(2 - index)
        sigma_to = float(1 - index)
        endpoint = prepare_euler_clean_endpoint(
            state,
            native_output,
            sigma_from=sigma_from,
            sigma_to=sigma_to,
            prediction_type="epsilon",
        )
        observation = RendererObservation(
            latents_before_step=state.detach(),
            pred_original_sample=endpoint.pred_original_sample.detach(),
            scheduler_update=endpoint.nominal_update.detach(),
            step_index=index,
            timestep=torch.tensor([float(index)]),
            normalized_timestep=torch.tensor([float(index)]),
        )
        condition = RendererCondition(
            bases=torch.arange(1 * 6 * 2 * 4 * 4, dtype=state.dtype).reshape(
                1, 6, 2, 4, 4
            ),
            prompt_embedding=torch.zeros(1, 32, dtype=state.dtype),
            state_features=torch.zeros(1, 16, dtype=state.dtype),
        )
        return EulerRolloutStep(
            observation=observation,
            condition=condition,
            native_model_output=native_output.detach(),
            native_prev_sample=(state + endpoint.nominal_update).detach(),
            sigma_from=sigma_from,
            sigma_to=sigma_to,
        )

    def transition(
        state: torch.Tensor,
        _index: int,
        model_output: torch.Tensor,
        step: EulerRolloutStep,
    ) -> EulerTransitionResult:
        latent = predict_euler_no_churn_prev_sample(
            state,
            model_output,
            sigma_from=step.sigma_from,
            sigma_to=step.sigma_to,
            prediction_type=step.prediction_type,
        )
        if transition_dtype is not None:
            latent = latent.to(transition_dtype)
        return EulerTransitionResult(state=latent, latent=latent)

    return EulerNativeRolloutCollector(
        renderer,
        decision_indices=(0,),
        registered_decision_indices=(0,),
        total_steps=1,
        observe_fn=observe,
        transition_fn=transition,
        physical_unet=unet if verify_physical else None,
    )


def test_collector_counts_verified_physical_unet_forwards() -> None:
    unet = _CollectorUNet()
    collector = _collector(unet)
    collector.collect(
        torch.zeros(1, 2, 4, 4),
        noise_by_decision={0: torch.zeros(1, 6)},
    )

    assert unet.forward_calls == 1
    assert collector.last_stats.observe_calls == 1
    assert collector.last_stats.verified_unet_forwards == 1
    assert len(unet._forward_hooks) == 0

    informal = _collector(_CollectorUNet(), verify_physical=False)
    informal.collect(
        torch.zeros(1, 2, 4, 4),
        noise_by_decision={0: torch.zeros(1, 6)},
    )
    assert informal.last_stats.observe_calls == 1
    assert informal.last_stats.verified_unet_forwards is None


@pytest.mark.parametrize("forwards", (0, 2))
def test_collector_rejects_unverified_physical_unet_cardinality(forwards: int) -> None:
    unet = _CollectorUNet()
    collector = _collector(unet, forwards_per_observation=forwards)

    with pytest.raises(RuntimeError, match=rf"exactly one physical U-Net forward.*{forwards}"):
        collector.collect(
            torch.zeros(1, 2, 4, 4),
            noise_by_decision={0: torch.zeros(1, 6)},
        )
    assert unet.forward_calls == forwards
    assert len(unet._forward_hooks) == 0


def test_collector_rejects_scheduler_transition_dtype_drift() -> None:
    unet = _CollectorUNet()
    collector = _collector(unet, transition_dtype=torch.float64)

    with pytest.raises(ValueError, match="transition.*wrong dtype"):
        collector.collect(
            torch.zeros(1, 2, 4, 4, dtype=torch.float32),
            noise_by_decision={0: torch.zeros(1, 6)},
        )


def _raw_authorization() -> TrainingAuthorization:
    return TrainingAuthorization(
        receipt_path=Path("/nonexistent/receipt.json"),
        selected_view_manifest_path=Path("/nonexistent/selected-view.json"),
        candidate_parent_manifest_path=Path("/nonexistent/catalog.json"),
        selected_config_path=Path("/nonexistent/selected-config.json"),
        selected_payload_path=Path("/nonexistent/selected.jsonl"),
        selected_view_manifest_sha256="a" * 64,
        candidate_parent_manifest_sha256="b" * 64,
        selected_config_sha256="c" * 64,
        selected_payload_sha256="d" * 64,
        selected_view_release_id="selected-view-" + "e" * 20,
        candidate_parent_release_id="catalog-" + "f" * 20,
        selected_view_id="formal",
        selected_rows=1,
        code_commit="0" * 40,
        repository_root=Path("/nonexistent/repository"),
    )


def test_collector_rejects_raw_training_authorization() -> None:
    unet = _CollectorUNet()
    with pytest.raises(TypeError, match="AuthorizationBinding"):
        EulerNativeRolloutCollector(
            _renderer(channels=2),
            decision_indices=(0,),
            registered_decision_indices=(0,),
            total_steps=1,
            observe_fn=lambda *_args: None,
            transition_fn=lambda *_args: None,
            physical_unet=unet,
            run_contract={"schema": RUN_CONTRACT_SCHEMA},
            authorization=_raw_authorization(),
        )


@pytest.mark.parametrize(
    "run_contract",
    (
        {"schema": RUN_CONTRACT_SCHEMA},
        TrainingRunContract.from_mapping(
            {"schema": RUN_CONTRACT_SCHEMA}, require_complete=False
        ),
    ),
)
def test_formal_run_contract_cannot_disable_authorization(run_contract: object) -> None:
    with pytest.raises(RuntimeError, match="formal training run contract.*binding"):
        EulerNativeRolloutCollector(
            _renderer(channels=2),
            decision_indices=(0,),
            registered_decision_indices=(0,),
            total_steps=1,
            observe_fn=lambda *_args: None,
            transition_fn=lambda *_args: None,
            physical_unet=_CollectorUNet(),
            run_contract=run_contract,
            require_authorization=False,
        )


class _PipelineUpBlock(nn.Module):
    def forward(self, hidden_states: torch.Tensor | None = None) -> torch.Tensor:
        if hidden_states is None:
            raise AssertionError("capture block requires keyword hidden_states")
        return hidden_states


class _BatchedPipelineUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(in_channels=4, sample_size=128)
        self.up_blocks = nn.ModuleList([_PipelineUpBlock()])
        self.register_buffer("anchor", torch.zeros(()), persistent=False)
        self.forward_calls = 0
        self.calls: list[dict[str, torch.Tensor]] = []
        self.last_feature: torch.Tensor | None = None

    @property
    def dtype(self) -> torch.dtype:
        return self.anchor.dtype

    @property
    def device(self) -> torch.device:
        return self.anchor.device

    def forward(
        self,
        sample: torch.Tensor,
        _timestep: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor,
        cross_attention_kwargs: object = None,
        added_cond_kwargs: dict[str, torch.Tensor],
        return_dict: bool = False,
    ) -> tuple[torch.Tensor]:
        del cross_attention_kwargs
        if return_dict:
            raise AssertionError("production integration must request tuple output")
        self.forward_calls += 1
        prompt_ids = encoder_hidden_states[:, 0, 0].float()
        pooled_ids = added_cond_kwargs["text_embeds"][:, 0].float()
        time_ids = added_cond_kwargs["time_ids"][:, 0].float()
        feature_ids = prompt_ids + pooled_ids / 1000.0 + time_ids / 1_000_000.0
        feature = feature_ids.reshape(-1, 1, 1, 1).expand(
            -1, 1, sample.shape[-2], sample.shape[-1]
        )
        self.last_feature = self.up_blocks[0](hidden_states=feature)
        self.calls.append(
            {
                "sample": sample.detach().clone(),
                "prompt_ids": prompt_ids.detach().clone(),
                "pooled_ids": pooled_ids.detach().clone(),
                "time_ids": time_ids.detach().clone(),
                "feature": self.last_feature.detach().clone(),
            }
        )
        prediction = 0.02 * sample + prompt_ids.reshape(-1, 1, 1, 1) / 10_000.0
        return (prediction,)


class _PipelineVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            block_out_channels=(1, 1, 1, 1, 1, 1, 1),
            force_upcast=False,
            scaling_factor=1.0,
        )
        self.register_buffer("anchor", torch.zeros(()), persistent=False)

    @property
    def dtype(self) -> torch.dtype:
        return self.anchor.dtype

    @property
    def device(self) -> torch.device:
        return self.anchor.device

    def decode(
        self, latent: torch.Tensor, return_dict: bool = False
    ) -> tuple[torch.Tensor]:
        if return_dict:
            raise AssertionError("production integration must request tuple decode")
        return (latent[:, :3].float(),)


class _ImageProcessor:
    @staticmethod
    def postprocess(image: torch.Tensor, output_type: str) -> list[torch.Tensor]:
        assert output_type == "pil"
        return [image.detach().clone()]


class _Progress:
    def __init__(self, total: int) -> None:
        self.total = int(total)
        self.updates = 0

    def __enter__(self) -> "_Progress":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if exc_type is None:
            assert self.updates == self.total
        return False

    def update(self) -> None:
        self.updates += 1


class _BatchedCfgBasisProvider:
    def __init__(self, unet: _BatchedPipelineUNet, *, batch_size: int) -> None:
        self.unet = unet
        self.batch_size = batch_size
        self.captured: list[torch.Tensor] = []
        self.observed_pooled: list[torch.Tensor] = []
        self._current: torch.Tensor | None = None
        self.last_diagnostics: dict[str, object] | None = None

    @contextmanager
    def capture_forward(self):
        before = self.unet.forward_calls
        yield
        calls = self.unet.forward_calls - before
        if calls != 1 or self.unet.last_feature is None:
            raise RuntimeError("provider capture requires one physical U-Net forward")
        feature = self.unet.last_feature.detach().clone()
        if feature.shape[0] != 2 * self.batch_size:
            raise RuntimeError("provider capture received an invalid CFG batch")
        self._current = feature[self.batch_size :]

    def __call__(self, observation: RendererObservation) -> RendererCondition:
        if self._current is None:
            raise RuntimeError("provider consumed before feature capture")
        captured = self._current
        self._current = None
        self.captured.append(captured.clone())
        pooled = observation.pooled_prompt_embeds
        if pooled is None:
            raise RuntimeError("production observation omitted positive pooled rows")
        self.observed_pooled.append(pooled.detach().clone())

        height, width = observation.latents_before_step.shape[-2:]
        y = torch.linspace(-1.0, 1.0, height, dtype=observation.latents_before_step.dtype)
        x = torch.linspace(-1.0, 1.0, width, dtype=observation.latents_before_step.dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        patterns = []
        for slot in range(6):
            pattern = torch.sin((slot + 1) * grid_x) + torch.cos(
                (slot + 2) * grid_y
            )
            patterns.append(
                pattern.reshape(1, 1, height, width).expand(
                    self.batch_size, 4, -1, -1
                )
                + captured[:, :1] / 10_000.0
            )
        bases = torch.stack(patterns, dim=1)
        self.last_diagnostics = {
            "conditional_feature_ids": captured[:, 0, 0, 0].cpu().tolist(),
            "conditional_rows": "second_half",
        }
        return RendererCondition(
            bases=bases,
            prompt_embedding=pooled,
            state_features=torch.zeros(
                self.batch_size,
                16,
                device=pooled.device,
                dtype=pooled.dtype,
            ),
        )


def _production_pipeline():
    from diffusers import EulerDiscreteScheduler
    from InferencePipelines.RepLDM.pipeline_repldm_sdxl import RepLDMSDXLPipeline

    unet = _BatchedPipelineUNet()
    scheduler = EulerDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
        timestep_spacing="leading",
        steps_offset=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pipe = RepLDMSDXLPipeline(
            vae=_PipelineVAE(),
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
            unet=unet,
            scheduler=scheduler,
            add_watermarker=False,
        )

    pipe.check_inputs = MethodType(lambda self, *args, **kwargs: None, pipe)

    def encode_prompt(self, **kwargs):
        del self
        device = kwargs["device"]
        positive = torch.stack(
            [torch.full((2, 4), value, device=device) for value in (101.0, 202.0)]
        )
        negative = torch.stack(
            [torch.full((2, 4), value, device=device) for value in (-11.0, -22.0)]
        )
        positive_pooled = torch.stack(
            [torch.full((32,), value, device=device) for value in (1001.0, 2002.0)]
        )
        negative_pooled = torch.stack(
            [torch.full((32,), value, device=device) for value in (-1001.0, -2002.0)]
        )
        return positive, negative, positive_pooled, negative_pooled

    def get_add_time_ids(self, original_size, *_args, dtype, **_kwargs):
        del self
        return torch.full((1, 6), float(original_size[0]), dtype=dtype)

    pipe.encode_prompt = MethodType(encode_prompt, pipe)
    pipe._get_add_time_ids = MethodType(get_add_time_ids, pipe)
    pipe.image_processor = _ImageProcessor()
    pipe.maybe_free_model_hooks = MethodType(lambda self: None, pipe)
    pipe.progress_bar = MethodType(
        lambda self, iterable=None, total=None: _Progress(total), pipe
    )
    return pipe, unet


def test_production_euler_native_frame_uses_one_physical_unet_for_batched_cfg() -> None:
    pipe, unet = _production_pipeline()
    renderer = _renderer(channels=4)
    provider = _BatchedCfgBasisProvider(unet, batch_size=2)
    latents = torch.stack(
        (torch.full((4, 16, 16), 0.1), torch.full((4, 16, 16), 0.2))
    )
    physical_calls = 0

    def count_physical_forward(
        _module: nn.Module, _args: tuple[object, ...], _output: object
    ) -> None:
        nonlocal physical_calls
        physical_calls += 1

    handle = unet.register_forward_hook(count_physical_forward)
    try:
        with mock.patch("builtins.print"):
            pipe(
                prompt=["first", "second"],
                height=1024,
                width=1024,
                num_inference_steps=2,
                guidance_scale=7.5,
                guidance_rescale=0.0,
                latents=latents,
                output_type="pil",
                record_latent_audit=True,
                attn_guidance_scale=0.0,
                original_size=(31, 32),
                target_size=(33, 34),
                negative_original_size=(41, 42),
                negative_target_size=(43, 44),
                latent_renderer=renderer,
                latent_renderer_basis_provider=provider,
                latent_renderer_scheduler_mapping="euler_native_frame_v1",
            )
    finally:
        handle.remove()

    assert physical_calls == 2
    assert unet.forward_calls == 2
    assert pipe._last_unet_calls_total == 2
    assert pipe._last_unet_calls_per_step == [1, 1]
    assert pipe._last_scheduler_calls_per_step == [1, 1]
    assert pipe._last_latent_renderer_scheduler_mapping == "euler_native_frame_v1"
    assert pipe._last_latent_renderer_frame_contract_hash == renderer.frame_contract_hash
    assert len(pipe._last_latent_renderer_step_diagnostics) == 2
    assert len(provider.captured) == 2
    assert len(unet._forward_hooks) == 0

    expected_prompt = torch.tensor([-11.0, -22.0, 101.0, 202.0])
    expected_pooled = torch.tensor([-1001.0, -2002.0, 1001.0, 2002.0])
    expected_time = torch.tensor([41.0, 41.0, 31.0, 31.0])
    expected_positive_pooled = torch.tensor([1001.0, 2002.0])
    for call, captured, pooled in zip(
        unet.calls, provider.captured, provider.observed_pooled
    ):
        assert call["sample"].shape[0] == 4
        torch.testing.assert_close(call["sample"][:2], call["sample"][2:])
        torch.testing.assert_close(call["prompt_ids"], expected_prompt)
        torch.testing.assert_close(call["pooled_ids"], expected_pooled)
        torch.testing.assert_close(call["time_ids"], expected_time)
        torch.testing.assert_close(captured, call["feature"][2:])
        torch.testing.assert_close(pooled[:, 0], expected_positive_pooled)


class _OffsetEulerRenderer(EulerNativeFrameV1):
    """Inject a controlled native prediction error for round-trip tests."""

    def __init__(self, *, offset: float) -> None:
        base = _renderer(channels=4)
        super().__init__(calibration=base.calibration, latent_channels=4)
        self._offset = float(offset)

    def forward_euler_mapped(
        self, *args: object, **kwargs: object
    ) -> EulerMappedOutput:
        result = super().forward_euler_mapped(*args, **kwargs)
        predicted = result.predicted_prev_sample.float() * (1.0 + self._offset)
        return replace(result, predicted_prev_sample=predicted)


@pytest.mark.parametrize(
    ("offset", "should_fail"),
    ((0.0009, False), (0.0011, True)),
)
def test_production_euler_native_round_trip_enforces_registered_tolerance(
    offset: float, should_fail: bool
) -> None:
    pipe, unet = _production_pipeline()
    renderer = _OffsetEulerRenderer(offset=offset)
    provider = _BatchedCfgBasisProvider(unet, batch_size=2)
    latents = torch.stack(
        (torch.full((4, 16, 16), 0.1), torch.full((4, 16, 16), 0.2))
    )

    def run() -> object:
        return pipe(
            prompt=["first", "second"],
            height=1024,
            width=1024,
            num_inference_steps=1,
            guidance_scale=7.5,
            guidance_rescale=0.0,
            latents=latents,
            output_type="pil",
            record_latent_audit=True,
            attn_guidance_scale=0.0,
            original_size=(31, 32),
            target_size=(33, 34),
            negative_original_size=(41, 42),
            negative_target_size=(43, 44),
            latent_renderer=renderer,
            latent_renderer_basis_provider=provider,
            latent_renderer_scheduler_mapping="euler_native_frame_v1",
        )

    with mock.patch("builtins.print"):
        if should_fail:
            with pytest.raises(RuntimeError, match="scheduler output differs"):
                run()
        else:
            assert run() is not None
