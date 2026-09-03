from dataclasses import replace
import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from latent_renderer_training.ledger import QueryLedger
from latent_renderer_training.operations import (
    LedgeredOperationExecutor,
    OperationContext,
    operation_output_sha256,
)
from latent_renderer_training.rewards import (
    IMAGE_REWARD_PREPROCESS_SHA256,
    ImageRewardTensorAdapter,
)
from latent_renderer_training.renderer import contract_hash, tensor_sha256
from latent_renderer_training.teachers import TargetStepConfig, construct_reward_targets
from latent_renderer_training.sdxl_adapter import (
    SdxlEulerTrainingAdapter,
    SdxlPromptConditioning,
)
from tests.test_euler_native_integration_contract import (
    _BatchedCfgBasisProvider,
    _production_pipeline,
    _renderer,
)
from tests.test_latent_renderer_training import (
    _complete_run_contract,
    _training_authorization,
)


def _checkpoint_hash() -> str:
    return "9" * 64


def test_sdxl_adapter_uses_one_cfg_forward_and_branch_safe_scheduler_state() -> None:
    pipe, unet = _production_pipeline()
    provider = _BatchedCfgBasisProvider(unet, batch_size=2)
    adapter = SdxlEulerTrainingAdapter(
        pipe,
        provider,
        total_steps=2,
        decision_indices=(0,),
        guidance_scale=7.5,
        vae_scaling_factor=1.0,
    )
    state = adapter.initial_state(
        prompts=("first", "second"),
        prompt_ids=("prompt-1", "prompt-2"),
        generation_seeds=(11, 22),
        checkpoint_hash=_checkpoint_hash(),
        split="train",
        height=128,
        width=128,
    )

    step = adapter.observe(state, 0)
    assert unet.forward_calls == 1
    assert step.metadata["physical_unet_forwards"] == 1
    assert step.metadata["schedule_sha256"] == adapter.schedule_hash
    torch.testing.assert_close(provider.captured[0], unet.calls[0]["feature"][2:])

    plus = adapter.clone_state(state, branch="plus")
    minus = adapter.clone_state(state, branch="minus")
    assert plus.latents.data_ptr() != minus.latents.data_ptr()
    assert plus.conditioning.prompt_embeds.data_ptr() != minus.conditioning.prompt_embeds.data_ptr()
    with torch.no_grad():
        plus.latents.add_(1.0)
    assert not torch.equal(plus.latents, minus.latents)

    plus_step = adapter.observe(plus, 0)
    minus_step = adapter.observe(minus, 0)
    plus_result = adapter.transition(
        plus,
        0,
        plus_step.native_model_output,
        plus_step,
        action=torch.zeros(2, 6),
    )
    minus_result = adapter.transition(
        minus,
        0,
        minus_step.native_model_output,
        minus_step,
        action=torch.zeros(2, 6),
    )
    assert plus_result.state.step_index == 1
    assert minus_result.state.step_index == 1
    assert plus_result.state.branch == "plus"
    assert minus_result.state.branch == "minus"
    torch.testing.assert_close(
        plus_result.latent, plus_step.native_prev_sample, rtol=2e-3, atol=3e-5
    )
    torch.testing.assert_close(
        minus_result.latent, minus_step.native_prev_sample, rtol=2e-3, atol=3e-5
    )


def test_sdxl_adapter_rejects_stale_schedule_dtype_and_action_history() -> None:
    pipe, unet = _production_pipeline()
    provider = _BatchedCfgBasisProvider(unet, batch_size=2)
    adapter = SdxlEulerTrainingAdapter(
        pipe,
        provider,
        total_steps=2,
        decision_indices=(0,),
        guidance_scale=7.5,
        vae_scaling_factor=1.0,
    )
    state = adapter.initial_state(
        prompts=("first", "second"),
        prompt_ids=("prompt-1", "prompt-2"),
        generation_seeds=(11, 22),
        checkpoint_hash=_checkpoint_hash(),
        split="train",
        height=128,
        width=128,
    )
    step = adapter.observe(state, 0)
    calls = unet.forward_calls
    with pytest.raises(ValueError, match="sigma_from"):
        adapter.transition(
            state,
            0,
            step.native_model_output,
            replace(step, sigma_from=step.sigma_from + 1),
            action=torch.zeros(2, 6),
        )
    with pytest.raises(ValueError, match="device, dtype"):
        adapter.transition(
            state,
            0,
            step.native_model_output.double(),
            step,
            action=torch.zeros(2, 6),
        )
    stale = replace(state, step_index=1)
    with pytest.raises(ValueError, match="action history"):
        adapter.observe(stale, 1)
    assert unet.forward_calls == calls


def test_sdxl_state_binds_f0_metric_action_without_changing_identity() -> None:
    pipe, unet = _production_pipeline()
    provider = _BatchedCfgBasisProvider(unet, batch_size=2)
    adapter = SdxlEulerTrainingAdapter(
        pipe,
        provider,
        total_steps=2,
        decision_indices=(0,),
        guidance_scale=7.5,
        vae_scaling_factor=1.0,
    )
    unbound = adapter.initial_state(
        prompts=("first", "second"),
        prompt_ids=("prompt-1", "prompt-2"),
        generation_seeds=(11, 22),
        checkpoint_hash=_checkpoint_hash(),
        split="train",
        height=128,
        width=128,
    )
    conditioning = SdxlPromptConditioning(
        prompt_ids=("prompt-1",),
        prompts=("first",),
        generation_seeds=(11,),
        prompt_embeds=torch.cat(
            (
                unbound.conditioning.prompt_embeds[0:1],
                unbound.conditioning.prompt_embeds[2:3],
            )
        ),
        pooled_prompt_embeds=unbound.conditioning.pooled_prompt_embeds[0:1],
        add_text_embeds=torch.cat(
            (
                unbound.conditioning.add_text_embeds[0:1],
                unbound.conditioning.add_text_embeds[2:3],
            )
        ),
        add_time_ids=torch.cat(
            (
                unbound.conditioning.add_time_ids[0:1],
                unbound.conditioning.add_time_ids[2:3],
            )
        ),
        do_classifier_free_guidance=True,
    )
    state = adapter.state_from_tensors(
        unbound.latents[0:1],
        conditioning,
        checkpoint_hash=_checkpoint_hash(),
        split="train",
    )
    metadata = {
        "f0_metric_branch": "anchor",
        "reuse_scope": "shared_prompt_seed",
        "target_record_sha256": None,
        "pixel_summary_schema": "repldm.renderer_f0_pixel_summary.v1",
    }

    context = state.operation_context(action_metadata=metadata)

    assert context.action == metadata
    assert context.split == state.split
    assert context.prompt == state.conditioning.prompt_ids[0]
    assert context.seed == state.conditioning.generation_seeds[0]
    assert context.branch == state.branch
    with pytest.raises(ValueError, match="cannot combine"):
        state.operation_context(
            action=torch.zeros(1, 6), action_metadata=metadata
        )
    with pytest.raises(TypeError, match="must be a mapping"):
        state.operation_context(action_metadata=["not", "a", "mapping"])


def test_sdxl_adapter_removes_forward_hook_after_unet_exception() -> None:
    pipe, unet = _production_pipeline()
    provider = _BatchedCfgBasisProvider(unet, batch_size=2)
    adapter = SdxlEulerTrainingAdapter(
        pipe,
        provider,
        total_steps=2,
        decision_indices=(0,),
        guidance_scale=7.5,
        vae_scaling_factor=1.0,
    )
    state = adapter.initial_state(
        prompts=("first", "second"),
        prompt_ids=("prompt-1", "prompt-2"),
        generation_seeds=(11, 22),
        checkpoint_hash=_checkpoint_hash(),
        split="train",
        height=128,
        width=128,
    )

    def fail_forward(*_args, **_kwargs):
        raise RuntimeError("planned U-Net failure")

    unet.forward = fail_forward
    with pytest.raises(RuntimeError, match="planned U-Net failure"):
        adapter.observe(state, 0)
    assert len(unet._forward_hooks) == 0


def test_force_upcast_vae_receives_float32_without_breaking_latent_gradient() -> None:
    pipe, unet = _production_pipeline()
    pipe.vae.config.force_upcast = True
    observed = []
    original_decode = pipe.vae.decode

    def record_decode(latent, return_dict=False):
        observed.append(latent.dtype)
        return original_decode(latent, return_dict=return_dict)

    pipe.vae.decode = record_decode
    provider = _BatchedCfgBasisProvider(unet, batch_size=2)
    adapter = SdxlEulerTrainingAdapter(
        pipe,
        provider,
        total_steps=2,
        decision_indices=(0,),
        guidance_scale=7.5,
        vae_scaling_factor=1.0,
    )
    base = adapter.initial_state(
        prompts=("first", "second"),
        prompt_ids=("prompt-1", "prompt-2"),
        generation_seeds=(11, 22),
        checkpoint_hash=_checkpoint_hash(),
        split="train",
        height=128,
        width=128,
    )
    latent = base.latents.half().detach().requires_grad_(True)
    image = adapter.decode(replace(base, latents=latent))
    image.sum().backward()
    assert observed == [torch.float32]
    assert latent.grad is not None and torch.count_nonzero(latent.grad) > 0


def test_sdxl_adapter_collector_completes_registered_fifty_step_rollout() -> None:
    pipe, unet = _production_pipeline()
    provider = _BatchedCfgBasisProvider(unet, batch_size=2)
    adapter = SdxlEulerTrainingAdapter(
        pipe,
        provider,
        total_steps=50,
        decision_indices=(8, 24, 40),
        guidance_scale=7.5,
        vae_scaling_factor=1.0,
    )
    initial = adapter.initial_state(
        prompts=("first", "second"),
        prompt_ids=("prompt-1", "prompt-2"),
        generation_seeds=(11, 22),
        checkpoint_hash=_checkpoint_hash(),
        split="train",
        height=128,
        width=128,
    )
    renderer = _renderer(channels=4)
    collector = adapter.make_collector(renderer)
    collection = collector.collect(
        initial,
        noise_by_decision={
            index: torch.zeros(2, 6) for index in adapter.decision_indices
        },
    )
    assert collector.last_stats.observe_calls == 132
    assert collector.last_stats.verified_unet_forwards == 132
    assert collector.last_stats.transition_calls == 134
    assert collector.last_stats.decision_transitions == 9
    assert collector.last_stats.auxiliary_transitions == 125
    assert unet.forward_calls == 132
    for terminal in collection.terminal_states.values():
        assert terminal.step_index == 50
        assert len(terminal.action_history) == 3
    plus = collection.terminal_states["plus"].latents
    assert torch.equal(plus, collection.terminal_states["minus"].latents)
    assert torch.equal(plus, collection.terminal_states["anchor"].latents)
    for rows in collection.all_transitions.values():
        for row in rows:
            assert torch.equal(row.rendered_transition, row.native_transition)


def test_sdxl_adapter_opd_collector_shares_prefix_and_adds_one_teacher_branch() -> None:
    pipe, unet = _production_pipeline()
    provider = _BatchedCfgBasisProvider(unet, batch_size=2)
    adapter = SdxlEulerTrainingAdapter(
        pipe,
        provider,
        total_steps=50,
        decision_indices=(8, 24, 40),
        guidance_scale=7.5,
        vae_scaling_factor=1.0,
    )
    initial = adapter.initial_state(
        prompts=("one", "two"),
        prompt_ids=("prompt-1", "prompt-2"),
        generation_seeds=(11, 22),
        checkpoint_hash=_checkpoint_hash(),
        split="train",
        height=128,
        width=128,
    )
    renderer = _renderer(channels=4)
    teacher = copy.deepcopy(renderer)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher.eval()
    collector = adapter.make_collector(renderer)
    collection, teacher_result = collector.collect_with_teacher(
        initial,
        noise_by_decision={
            index: torch.zeros(2, 6) for index in adapter.decision_indices
        },
        teacher_renderer=teacher,
        teacher_checkpoint_hash="a" * 64,
    )
    # 9 shared-prefix observations + 4 branches x 41 suffix observations.
    assert collector.last_stats.observe_calls == 173
    assert collector.last_stats.verified_unet_forwards == 173
    assert collector.last_stats.transition_calls == 176
    assert collector.last_stats.decision_transitions == 12
    assert collector.last_stats.auxiliary_transitions == 164
    assert unet.forward_calls == 173
    assert teacher_result.stats.observe_calls == 42
    assert teacher_result.stats.transition_calls == 42
    assert teacher_result.stats.decision_transitions == 3
    assert teacher_result.stats.auxiliary_transitions == 39
    assert teacher_result.stats.verified_unet_forwards == 41
    assert len(teacher_result.branch.transitions) == 3
    assert teacher_result.terminal_state.step_index == 50
    for row in teacher_result.branch.all_transitions:
        for value in (
            row.action,
            row.native_transition,
            row.rendered_transition,
            row.pre_squash,
            row.behavior_mean,
            row.reference_mean,
        ):
            if value is not None:
                assert value.requires_grad is False
    assert all(parameter.requires_grad is False for parameter in teacher.parameters())


def test_formal_sdxl_collector_rechecks_authorization_before_unet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipe, unet = _production_pipeline()
    provider = _BatchedCfgBasisProvider(unet, batch_size=2)
    probe = SdxlEulerTrainingAdapter(
        pipe,
        provider,
        total_steps=50,
        decision_indices=(8, 24, 40),
        guidance_scale=7.5,
        vae_scaling_factor=1.0,
    )
    renderer = _renderer(channels=4)
    authorization = _training_authorization(tmp_path, monkeypatch)
    contract = _complete_run_contract(
        authorization,
        renderer_frame_contract_hash=renderer.frame_contract_hash,
        calibration_hash=renderer.calibration_hash,
        action_contract_hash=contract_hash(renderer.contract),
        scheduler_config_sha256=probe.scheduler_config_hash,
        scheduler_schedule_sha256=probe.schedule_hash,
        query_budget={"unet_forward": 132, "scheduler_step": 134},
    )
    binding = authorization.bind_run_contract(contract)
    ledger = QueryLedger(
        contract["paths"]["ledger_path"],
        contract["query_budget"],
        run_contract=contract,
        authorization_binding=binding,
    )
    executor = LedgeredOperationExecutor(ledger, authorization_binding=binding)
    adapter = SdxlEulerTrainingAdapter(
        pipe,
        provider,
        total_steps=50,
        decision_indices=(8, 24, 40),
        guidance_scale=7.5,
        vae_scaling_factor=1.0,
        operation_executor=executor,
    )
    collector = adapter.make_collector(renderer)
    selected_row = json.loads(
        authorization.selected_payload_path.read_text(encoding="utf-8").splitlines()[0]
    )
    payload = Path(selected_row["image_path"])
    payload.write_bytes(b"changed-after-collector-construction")

    with pytest.raises(RuntimeError, match="selected raw image .* changed"):
        collector.collect(
            None,
            noise_by_decision={index: torch.zeros(2, 6) for index in (8, 24, 40)},
        )
    assert unet.forward_calls == 0
    assert not Path(contract["paths"]["ledger_path"]).exists()


class _TensorReward(nn.Module):
    def score_tensor(self, images: torch.Tensor, prompts: tuple[str, ...]) -> torch.Tensor:
        assert len(prompts) == images.shape[0]
        return images.flatten(1).mean(dim=1)


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, prompts, **kwargs):
        self.calls.append((tuple(prompts), dict(kwargs)))
        batch = len(prompts)
        return {
            "input_ids": torch.ones(batch, 35, dtype=torch.long),
            "attention_mask": torch.ones(batch, 35, dtype=torch.long),
        }


class _FakeImageReward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.last_image = None

    def score_gard(self, input_ids, attention_mask, image):
        assert input_ids.shape == attention_mask.shape == (image.shape[0], 35)
        self.last_image = image
        return image.flatten(1).mean(dim=1, keepdim=True) * self.scale


def test_image_reward_tensor_adapter_preserves_image_gradient_and_freezes_model() -> None:
    model = _FakeImageReward()
    tokenizer = _FakeTokenizer()
    reward = ImageRewardTensorAdapter(model, tokenizer=tokenizer)
    image = torch.linspace(0.0, 1.0, 2 * 3 * 320 * 480).reshape(
        2, 3, 320, 480
    ).requires_grad_(True)

    scores = reward.score_tensor(image, ("first", "second"))
    scores.sum().backward()

    assert scores.shape == (2,)
    assert model.last_image.shape == (2, 3, 224, 224)
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert torch.count_nonzero(image.grad) > 0
    assert model.scale.grad is None
    assert model.scale.requires_grad is False
    assert reward.preprocess_sha256 == IMAGE_REWARD_PREPROCESS_SHA256
    assert tokenizer.calls == [
        (
            ("first", "second"),
            {
                "padding": "max_length",
                "truncation": True,
                "max_length": 35,
                "return_tensors": "pt",
            },
        )
    ]


def test_sdxl_adapter_keeps_vae_reward_gradient_to_latent() -> None:
    pipe, unet = _production_pipeline()
    provider = _BatchedCfgBasisProvider(unet, batch_size=2)
    reward_model = _TensorReward()
    adapter = SdxlEulerTrainingAdapter(
        pipe,
        provider,
        total_steps=2,
        decision_indices=(0,),
        guidance_scale=7.5,
        vae_scaling_factor=1.0,
        reward_model=reward_model,
    )
    base = adapter.initial_state(
        prompts=("first", "second"),
        prompt_ids=("prompt-1", "prompt-2"),
        generation_seeds=(11, 22),
        checkpoint_hash=_checkpoint_hash(),
        split="train",
        height=128,
        width=128,
    )
    latent = base.latents.detach().clone().requires_grad_(True)
    state = adapter.state_from_tensors(
        latent,
        base.conditioning,
        checkpoint_hash=_checkpoint_hash(),
    )

    image = adapter.decode(state)
    reward = adapter.reward(state, image)
    gradient = adapter.reward_gradient(state, reward, latent)

    assert image.requires_grad
    assert reward.requires_grad
    assert gradient.shape == latent.shape
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0
    assert all(not parameter.requires_grad for parameter in reward_model.parameters())
    assert all(not parameter.requires_grad for parameter in adapter.vae.parameters())


def test_formal_reward_gradient_binds_forward_and_image_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipe, unet = _production_pipeline()
    provider = _BatchedCfgBasisProvider(unet, batch_size=2)
    probe = SdxlEulerTrainingAdapter(
        pipe,
        provider,
        total_steps=50,
        decision_indices=(8, 24, 40),
        guidance_scale=7.5,
        vae_scaling_factor=1.0,
    )
    renderer = _renderer(channels=4)
    authorization = _training_authorization(tmp_path, monkeypatch)
    budget = {"vae_decode": 1, "reward_forward": 1, "reward_backward": 1}
    contract = _complete_run_contract(
        authorization,
        renderer_frame_contract_hash=renderer.frame_contract_hash,
        calibration_hash=renderer.calibration_hash,
        action_contract_hash=contract_hash(renderer.contract),
        scheduler_config_sha256=probe.scheduler_config_hash,
        scheduler_schedule_sha256=probe.schedule_hash,
        query_budget=budget,
    )
    binding = authorization.bind_run_contract(contract)
    ledger = QueryLedger(
        contract["paths"]["ledger_path"],
        budget,
        run_contract=contract,
        authorization_binding=binding,
    )
    executor = LedgeredOperationExecutor(ledger, authorization_binding=binding)
    adapter = SdxlEulerTrainingAdapter(
        pipe,
        provider,
        total_steps=50,
        decision_indices=(8, 24, 40),
        guidance_scale=7.5,
        vae_scaling_factor=1.0,
        operation_executor=executor,
        reward_model=_TensorReward(),
    )
    unbound = probe.initial_state(
        prompts=("formal prompt", "unused prompt"),
        prompt_ids=("formal-prompt-1", "unused-prompt-2"),
        generation_seeds=(2026090101, 2026090102),
        checkpoint_hash=_checkpoint_hash(),
        split="test",
        height=128,
        width=128,
    )
    conditioning = SdxlPromptConditioning(
        prompt_ids=("formal-prompt-1",),
        prompts=("formal prompt",),
        generation_seeds=(2026090101,),
        prompt_embeds=torch.cat(
            (unbound.conditioning.prompt_embeds[0:1], unbound.conditioning.prompt_embeds[2:3])
        ),
        pooled_prompt_embeds=unbound.conditioning.pooled_prompt_embeds[0:1],
        add_text_embeds=torch.cat(
            (unbound.conditioning.add_text_embeds[0:1], unbound.conditioning.add_text_embeds[2:3])
        ),
        add_time_ids=torch.cat(
            (unbound.conditioning.add_time_ids[0:1], unbound.conditioning.add_time_ids[2:3])
        ),
        do_classifier_free_guidance=True,
    )
    latent = unbound.latents[0:1].detach().clone()
    state = adapter.state_from_tensors(
        latent,
        conditioning,
        checkpoint_hash=str(contract["initial_renderer_state_sha256"]),
        split="train",
        branch="f0-gradient",
        prefix=8,
    )
    receipt_context = {}

    def clean_from_u(value):
        return latent + value[:, :1, None, None]

    def reward_from_clean(clean):
        clean_state = replace(state, latents=clean)
        image = adapter.decode(clean_state)
        receipt_context.update(state=clean_state, image_hash=tensor_sha256(image))
        return adapter.reward(clean_state, image)

    pair = construct_reward_targets(
        torch.zeros(1, 2),
        clean_from_u,
        reward_from_clean,
        reward_gradient=lambda reward, inputs: adapter.reward_gradient(
            receipt_context["state"],
            reward,
            inputs,
            image_hash=receipt_context["image_hash"],
        ),
        candidate_validator=lambda value: torch.ones(
            value.shape[0], dtype=torch.bool, device=value.device
        ),
        config=TargetStepConfig(target_steps=2),
    )

    assert pair.valid is not None and bool(pair.valid[0])
    assert torch.count_nonzero(pair.gradient_u) > 0
    rows = [
        json.loads(line)
        for line in ledger.path.read_text(encoding="utf-8").splitlines()
    ]
    receipts = [row for row in rows if row["type"] == "receipt"]
    forward = next(
        row for row in receipts if row["kind"] == "reward_forward"
    )
    backward = next(
        row for row in receipts if row["kind"] == "reward_backward"
    )
    assert backward["result"]["cached_parent"] == forward["result"]["output_hash"]
    assert backward["result"]["image_hash"] == receipt_context["image_hash"]
    assert backward["result"]["parent_reservation_id"] == forward["reservation_id"]


def test_formal_reward_gradient_rejects_unledgered_parent_before_backward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor, contract = _formal_executor(
        tmp_path,
        monkeypatch,
        budget={"reward_forward": 1, "reward_backward": 1},
    )

    class MinimalAdapter:
        def __init__(self):
            self.operation_executor = executor
            self._reward_receipts = {}

        def _preflight(self):
            self.operation_executor._preflight()

    # Exercise the method without constructing another full SDXL runtime; the
    # lineage check precedes autograd and the budget reservation.
    reward = torch.ones(1, requires_grad=True)
    inputs = torch.ones(1, 1, requires_grad=True)
    state = type("State", (), {})()
    with pytest.raises(RuntimeError, match="not descended"):
        SdxlEulerTrainingAdapter.reward_gradient(
            MinimalAdapter(),
            state,
            reward,
            inputs,
            image_hash="a" * 64,
        )
    assert executor.ledger.summary()["reserved"] == {
        "reward_forward": 0,
        "reward_backward": 0,
    }


def _formal_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    budget: dict[str, int],
) -> tuple[LedgeredOperationExecutor, dict[str, object]]:
    authorization = _training_authorization(tmp_path, monkeypatch)
    contract = _complete_run_contract(authorization, query_budget=budget)
    binding = authorization.bind_run_contract(contract)
    ledger = QueryLedger(
        tmp_path / "operations.jsonl",
        budget,
        run_contract=contract,
        authorization_binding=binding,
    )
    return (
        LedgeredOperationExecutor(ledger, authorization_binding=binding),
        contract,
    )


def _context(contract: dict[str, object]) -> OperationContext:
    return OperationContext(
        split="train",
        prompt="source:row-0",
        seed=0,
        step=8,
        prefix=0,
        branch="plus",
        action=[[0.1, 0.0]],
        checkpoint_hash=str(contract["initial_renderer_state_sha256"]),
    )


def test_ledgered_operation_records_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor, contract = _formal_executor(
        tmp_path, monkeypatch, budget={"unet_forward": 2}
    )
    value = executor.execute(
        "unet_forward", _context(contract), lambda: torch.tensor([1.0])
    )
    torch.testing.assert_close(value, torch.tensor([1.0]))

    with pytest.raises(RuntimeError, match="planned failure"):
        executor.execute(
            "unet_forward",
            _context(contract),
            lambda: (_ for _ in ()).throw(RuntimeError("planned failure")),
        )

    summary = executor.ledger.summary()
    assert summary["reserved"] == {"unet_forward": 2}
    assert summary["completed_receipts"] == 2
    assert summary["unfinished_reservations"] == 0
    rows = [
        __import__("json").loads(line)
        for line in executor.ledger.path.read_text(encoding="utf-8").splitlines()
    ]
    receipts = [row for row in rows if row["type"] == "receipt"]
    assert [row["success"] for row in receipts] == [True, False]
    assert receipts[0]["result"]["output_hash"] == operation_output_sha256(
        torch.tensor([1.0])
    )
    assert receipts[1]["result"]["failure"]["message"] == "planned failure"


def test_ledgered_operation_rejects_unbudgeted_kind_before_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor, contract = _formal_executor(
        tmp_path, monkeypatch, budget={"reward_forward": 1}
    )
    called: list[bool] = []
    with pytest.raises(ValueError, match="absent from the run budget"):
        executor.execute(
            "scheduler_step",
            _context(contract),
            lambda: called.append(True),
        )
    assert called == []
    assert not executor.ledger.path.exists()
