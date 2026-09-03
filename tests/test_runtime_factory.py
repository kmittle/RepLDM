from __future__ import annotations

from dataclasses import dataclass, replace
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from latent_renderer_training.runtime_factory import (
    BASIS_RUNTIME_SCHEMA,
    BoundAsset,
    REWARD_RUNTIME_SCHEMA,
    ModelStage,
    RendererBundle,
    RewardStage,
    RuntimeFactoryDependencies,
    RuntimeInfrastructure,
    RuntimeLoadSpec,
    build_training_runtime,
    _cleanup_default_reward,
    _load_default_renderers,
    _require_independent_frozen_teacher,
    _require_teacher_checkpoint_provenance,
    _stage_default_reward,
)
from latent_renderer_training.artifacts import module_state_sha256
from latent_renderer_training.contracts import contract_hash
from latent_renderer_training.renderer import (
    CALIBRATION_STATE_COUNT,
    EulerNativeFrameV1,
    FrameCalibration,
)


def _write(path: Path, value: bytes) -> Path:
    path.write_bytes(value)
    return path


def _json(path: Path, value: object) -> Path:
    return _write(
        path,
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n",
    )


@dataclass(frozen=True)
class _Snapshot:
    label: str
    path: str
    sha256: str
    size: int
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int

    @property
    def identity(self):
        return self.device, self.inode, self.size, self.mtime_ns, self.ctime_ns

    @classmethod
    def capture(cls, label: str, path: Path):
        status = path.stat()
        return cls(
            label,
            str(path),
            hashlib.sha256(path.read_bytes()).hexdigest(),
            status.st_size,
            status.st_dev,
            status.st_ino,
            status.st_mtime_ns,
            status.st_ctime_ns,
        )


class _Artifacts:
    def __init__(self, descriptors, payload, events):
        self.descriptors = tuple(descriptors)
        self.payload_files = tuple(payload)
        self.events = events

    def validate_current(self):
        self.events.append("artifacts.validate")


class _Binding:
    def __init__(self, artifacts, events):
        self.run_artifacts = artifacts
        self.events = events
        self.contract_hash = "c" * 64
        self.run_contract = object()
        self.authorization = SimpleNamespace(repository_root=Path("/unused"))
        self.contract = {
            "nfe": 50,
            "prediction_type": "epsilon",
            "do_classifier_free_guidance": True,
            "guidance_scale": 7.5,
            "guidance_rescale": 0.0,
            "decision_indices": [8, 24, 40],
            "basis_provider_contract_hash": "b" * 64,
            "selected_rows": 1,
            "runtime": {
                "device": "cpu",
                "batch_size": 1,
                "model_dtype": "float16",
                "vae_dtype": "float32",
                "reward_dtype": "float32",
                "vae_scaling_factor": 0.13025,
                "height": 1024,
                "width": 1024,
            },
        }

    def validate_current(self):
        self.events.append("binding.validate")

    def validate_initial_renderer(self, renderer):
        self.events.append(("renderer.validate", id(renderer)))

    def validate_component(self, renderer):
        self.events.append(("component.validate", id(renderer)))
        if renderer.frame_contract_hash != self.contract.get(
            "renderer_frame_contract_hash"
        ):
            raise ValueError("renderer frame mismatch")
        if renderer.calibration_hash != self.contract.get("calibration_hash"):
            raise ValueError("renderer calibration mismatch")
        if contract_hash(renderer.contract) != self.contract.get("action_contract_hash"):
            raise ValueError("renderer action contract mismatch")


class EulerDiscreteScheduler:
    def __init__(self):
        self.config = SimpleNamespace(prediction_type="epsilon")
        self.timesteps = torch.empty(0)
        self.sigmas = torch.empty(0)

    def set_timesteps(self, steps, *, device):
        self.timesteps = torch.arange(steps, device=device)
        self.sigmas = torch.arange(steps + 1, device=device)


class _VAE(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dtype=dtype), requires_grad=False)
        self.config = SimpleNamespace(force_upcast=True, scaling_factor=0.13025)


class _Pipeline:
    def __init__(self, *, vae_dtype=torch.float32):
        self.unet = nn.Linear(1, 1).half()
        self.vae = _VAE(vae_dtype)
        self.scheduler = EulerDiscreteScheduler()
        for parameter in self.unet.parameters():
            parameter.requires_grad_(False)


class _Reward(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1), requires_grad=False)

    def score_gard(self, *args):
        return self.weight


class _Executor:
    def __init__(self, ledger, binding):
        self.ledger = ledger
        self.authorization_binding = binding


class _Store:
    def __init__(self, binding):
        self.authorization_binding = binding


def _fixture(tmp_path: Path, *, selected_rows: int = 1):
    events = []
    model = _write(tmp_path / "model.bin", b"model")
    reward = _write(tmp_path / "reward.bin", b"reward")

    def row(path):
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }

    selected = _write(
        tmp_path / "selected.jsonl",
        b"".join(
            json.dumps({"id": f"row-{index}", "prompt": f"test {index}"}).encode("ascii")
            + b"\n"
            for index in range(selected_rows)
        ),
    )
    descriptors = [
        _Snapshot.capture("data_manifest", selected),
        _Snapshot.capture(
            "model_assets_manifest",
            _json(
                tmp_path / "model-assets.json",
                {"schema": "repldm.file_manifest.v1", "files": [row(model)]},
            ),
        ),
        _Snapshot.capture(
            "reward_assets_manifest",
            _json(
                tmp_path / "reward-assets.json",
                {"schema": "repldm.file_manifest.v1", "files": [row(reward)]},
            ),
        ),
        _Snapshot.capture(
            "reward_config",
            _json(
                tmp_path / "reward.json",
                {
                    "schema": REWARD_RUNTIME_SCHEMA,
                    "implementation": "ImageReward-v1.0",
                    "dtype": "float32",
                    "preprocess_sha256": "a" * 64,
                },
            ),
        ),
        _Snapshot.capture(
            "basis_provider_config",
            _json(
                tmp_path / "basis.json",
                {
                    "schema": BASIS_RUNTIME_SCHEMA,
                    "basis_provider_contract_hash": "b" * 64,
                },
            ),
        ),
        _Snapshot.capture("calibration", _json(tmp_path / "calibration.json", {"test": 1})),
        _Snapshot.capture("renderer_frame_contract", _json(tmp_path / "frame.json", {"test": 1})),
        _Snapshot.capture("initial_renderer_state", _json(tmp_path / "initial.json", {"test": 1})),
    ]
    payload = [
        _Snapshot.capture("model_assets_manifest[0]", model),
        _Snapshot.capture("reward_assets_manifest[0]", reward),
    ]
    artifacts = _Artifacts(descriptors, payload, events)
    binding = _Binding(artifacts, events)
    binding.contract["selected_rows"] = selected_rows
    return binding, events


def _dependencies(binding, events, tmp_path, *, vae_dtype=torch.float32):
    ledger = object()
    executor = _Executor(ledger, binding)
    store = _Store(binding)
    behavior = nn.Linear(1, 1)
    reference = nn.Linear(1, 1)
    reference.load_state_dict(behavior.state_dict())
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    def validate_binding(value):
        events.append("binding.require")
        if value is not binding:
            raise TypeError("wrong binding")
        return value

    def stage_model(value, assets, parent):
        assert value is binding
        assert [asset.path.name for asset in assets] == ["model.bin"]
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert "HF_TOKEN" not in os.environ
        with pytest.raises(RuntimeError, match="network access is disabled"):
            socket.create_connection(("example.invalid", 443))
        events.append("model.stage")
        return ModelStage(SimpleNamespace(), {"stage": "model"}, "d" * 64)

    def load_pipeline(stage, device, model_dtype, required_vae_dtype):
        assert stage.record == {"stage": "model"}
        assert device == torch.device("cpu")
        assert model_dtype is torch.float16
        assert required_vae_dtype is torch.float32
        events.append("pipeline.load")
        return _Pipeline(vae_dtype=vae_dtype)

    def cleanup_model(stage):
        events.append("model.cleanup")

    def stage_reward(config, assets, parent):
        assert config["implementation"] == "ImageReward-v1.0"
        assert [asset.path.name for asset in assets] == ["reward.bin"]
        events.append("reward.stage")
        return RewardStage(tmp_path, tmp_path / "c", tmp_path / "m", tmp_path, tmp_path)

    def load_reward(stage, device, dtype):
        events.append("reward.load")
        return _Reward()

    def cleanup_reward(stage):
        events.append("reward.cleanup")

    def load_renderers(value, descriptors, payload, device):
        assert value is binding
        events.append("renderers.load")
        return RendererBundle(behavior, reference, SimpleNamespace(sha256="e" * 64))

    def make_infrastructure(value):
        assert value is binding
        events.append("infrastructure.make")
        return RuntimeInfrastructure(ledger, executor, store)

    def make_basis(unet, config, batch_size, cfg):
        assert unet is not None and batch_size == 1 and cfg is True
        events.append("basis.make")
        return object()

    def make_adapter(pipeline, basis, reward, supplied_executor, contract):
        assert supplied_executor is executor
        events.append("adapter.make")
        return SimpleNamespace(operation_executor=supplied_executor)

    return RuntimeFactoryDependencies(
        validate_binding=validate_binding,
        stage_model=stage_model,
        load_pipeline=load_pipeline,
        cleanup_model=cleanup_model,
        stage_reward=stage_reward,
        load_reward=load_reward,
        cleanup_reward=cleanup_reward,
        load_renderers=load_renderers,
        make_infrastructure=make_infrastructure,
        make_basis_provider=make_basis,
        wrap_reward=lambda model: model,
        make_adapter=make_adapter,
        reward_preprocess_sha256="a" * 64,
    )


def test_factory_builds_one_identity_bound_runtime_and_restores_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    binding, events = _fixture(tmp_path)
    dependencies = _dependencies(binding, events, tmp_path)
    monkeypatch.setenv("HF_TOKEN", "not-visible-to-loader")
    monkeypatch.setenv("HTTP_PROXY", "http://not-visible.example")

    runtime = build_training_runtime(binding, dependencies=dependencies)

    assert runtime.binding is binding
    assert runtime.adapter.operation_executor is runtime.operation_executor
    assert runtime.operation_executor.ledger is runtime.ledger
    assert runtime.rollout_store.authorization_binding is binding
    assert runtime.renderer is not runtime.reference_renderer
    assert runtime.teacher_renderer is None
    assert runtime.teacher_checkpoint_provenance is None
    assert "opsd_teacher_checkpoint_sha256" not in runtime.provenance
    assert runtime.selected_records[0]["id"] == "row-0"
    with pytest.raises(TypeError):
        runtime.selected_records[0]["id"] = "changed"
    assert os.environ["HF_TOKEN"] == "not-visible-to-loader"
    assert os.environ["HTTP_PROXY"] == "http://not-visible.example"
    assert events.count("infrastructure.make") == 1
    assert events.count("adapter.make") == 1

    runtime.close()
    runtime.close()
    assert events[-2:] == ["reward.cleanup", "model.cleanup"]


@pytest.mark.parametrize(
    "mismatched",
    [
        RuntimeLoadSpec(device="cpu", batch_size=2),
        RuntimeLoadSpec(device="cpu", batch_size=1, height=768),
        RuntimeLoadSpec(device="cpu", batch_size=1, width=768),
        RuntimeLoadSpec(device="cuda:0", batch_size=1),
    ],
)
def test_explicit_spec_must_equal_the_contract_before_any_loader_runs(
    tmp_path: Path, mismatched: RuntimeLoadSpec
):
    binding, events = _fixture(tmp_path)
    dependencies = _dependencies(binding, events, tmp_path)

    with pytest.raises(ValueError, match="differs from the run contract"):
        build_training_runtime(binding, spec=mismatched, dependencies=dependencies)

    assert "infrastructure.make" not in events
    assert "model.stage" not in events


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_dtype", "float32", "pinned SDXL runtime requires float16"),
        ("vae_dtype", "float16", "force-upcast VAE must remain float32"),
        ("reward_dtype", "float16", "frozen ImageReward runtime requires float32"),
        ("height", 1025, "positive multiples of eight"),
        ("width", 0, "positive multiples of eight"),
    ],
)
def test_invalid_contract_runtime_field_fails_before_loaders(
    tmp_path: Path, field: str, value: object, message: str
):
    binding, events = _fixture(tmp_path)
    binding.contract["runtime"][field] = value
    dependencies = _dependencies(binding, events, tmp_path)

    with pytest.raises(ValueError, match=message):
        build_training_runtime(binding, dependencies=dependencies)
    assert "infrastructure.make" not in events
    assert "model.stage" not in events


def test_failure_after_reward_stage_cleans_both_private_stages(tmp_path: Path):
    binding, events = _fixture(tmp_path)
    dependencies = _dependencies(binding, events, tmp_path)

    def fail_reward(*args):
        events.append("reward.load.failed")
        raise RuntimeError("load failed")

    dependencies = replace(dependencies, load_reward=fail_reward)
    with pytest.raises(RuntimeError, match="load failed"):
        build_training_runtime(binding, dependencies=dependencies)
    assert events[-2:] == ["reward.cleanup", "model.cleanup"]


def test_cleanup_failure_does_not_skip_the_other_private_stage(tmp_path: Path):
    binding, events = _fixture(tmp_path)
    dependencies = _dependencies(binding, events, tmp_path)

    def fail_reward(*args):
        raise ValueError("primary failure")

    def fail_reward_cleanup(stage):
        events.append("reward.cleanup.failed")
        raise OSError("cleanup failure")

    dependencies = replace(
        dependencies,
        load_reward=fail_reward,
        cleanup_reward=fail_reward_cleanup,
    )
    with pytest.raises(RuntimeError, match="cleanup also failed") as caught:
        build_training_runtime(binding, dependencies=dependencies)
    assert isinstance(caught.value.original_error, ValueError)
    assert len(caught.value.cleanup_errors) == 1
    assert events[-2:] == ["reward.cleanup.failed", "model.cleanup"]


def test_context_cleanup_does_not_mask_body_exception(tmp_path: Path):
    binding, events = _fixture(tmp_path)
    runtime = build_training_runtime(
        binding, dependencies=_dependencies(binding, events, tmp_path)
    )

    def fail_cleanup(stage):
        events.append("reward.cleanup.failed")
        raise OSError("cleanup failure")

    runtime._cleanup_reward = fail_cleanup
    with pytest.raises(ValueError, match="body failure") as caught:
        with runtime:
            raise ValueError("body failure")

    cleanup_error = getattr(caught.value, "_runtime_cleanup_error", None)
    assert isinstance(cleanup_error, RuntimeError)
    assert len(cleanup_error.cleanup_errors) == 1
    assert events[-2:] == ["reward.cleanup.failed", "model.cleanup"]


def test_fp16_vae_is_rejected_and_stages_are_cleaned(tmp_path: Path):
    binding, events = _fixture(tmp_path)
    dependencies = _dependencies(binding, events, tmp_path, vae_dtype=torch.float16)

    with pytest.raises(ValueError, match="VAE was not kept permanently in float32"):
        build_training_runtime(binding, dependencies=dependencies)
    assert events[-2:] == ["reward.cleanup", "model.cleanup"]


def test_replaced_executor_is_rejected_before_model_loading(tmp_path: Path):
    binding, events = _fixture(tmp_path)
    dependencies = _dependencies(binding, events, tmp_path)
    wrong_ledger = object()

    def bad_infrastructure(value):
        return RuntimeInfrastructure(
            wrong_ledger,
            _Executor(object(), binding),
            _Store(binding),
        )

    dependencies = replace(dependencies, make_infrastructure=bad_infrastructure)
    with pytest.raises(RuntimeError, match="does not own the runtime ledger"):
        build_training_runtime(binding, dependencies=dependencies)
    assert "model.stage" not in events


def test_renderer_and_reference_must_have_equal_nonaliased_c0_state(tmp_path: Path):
    binding, events = _fixture(tmp_path)
    dependencies = _dependencies(binding, events, tmp_path)
    behavior = nn.Linear(1, 1)
    different = nn.Linear(1, 1)
    with torch.no_grad():
        different.weight.add_(10.0)
    for parameter in different.parameters():
        parameter.requires_grad_(False)

    dependencies = replace(
        dependencies,
        load_renderers=lambda *args: RendererBundle(
            behavior, different, SimpleNamespace(sha256="e" * 64)
        ),
    )
    with pytest.raises(ValueError, match="states differ at C0"):
        build_training_runtime(binding, dependencies=dependencies)
    assert events[-2:] == ["reward.cleanup", "model.cleanup"]


def test_renderer_and_reference_cannot_share_parameter_storage(tmp_path: Path):
    binding, events = _fixture(tmp_path)
    dependencies = _dependencies(binding, events, tmp_path)
    behavior = nn.Linear(1, 1)
    reference = nn.Linear(1, 1)
    reference.weight = nn.Parameter(behavior.weight.data)
    reference.bias.data.copy_(behavior.bias.data)
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    dependencies = replace(
        dependencies,
        load_renderers=lambda *args: RendererBundle(
            behavior, reference, SimpleNamespace(sha256="e" * 64)
        ),
    )
    with pytest.raises(RuntimeError, match="storage aliases"):
        build_training_runtime(binding, dependencies=dependencies)
    assert events[-2:] == ["reward.cleanup", "model.cleanup"]


def test_selected_records_preserve_the_formal_96_row_shape(tmp_path: Path):
    binding, events = _fixture(tmp_path, selected_rows=96)
    dependencies = _dependencies(binding, events, tmp_path)

    with build_training_runtime(binding, dependencies=dependencies) as runtime:
        assert len(runtime.selected_records) == 96
        assert [row["id"] for row in runtime.selected_records] == [
            f"row-{index}" for index in range(96)
        ]
        with pytest.raises(TypeError):
            runtime.selected_records[-1]["prompt"] = "mutated"


def test_binding_validator_cannot_replace_the_capability(tmp_path: Path):
    binding, events = _fixture(tmp_path)
    dependencies = _dependencies(binding, events, tmp_path)
    dependencies = replace(dependencies, validate_binding=lambda value: object())

    with pytest.raises(RuntimeError, match="replaced the authorization capability"):
        build_training_runtime(binding, dependencies=dependencies)
    assert events == []


def test_default_reward_stage_copies_every_role_into_a_private_tree(tmp_path: Path):
    source_root = tmp_path / "source-input"
    tokenizer_root = tmp_path / "tokenizer-input"
    source_root.mkdir()
    tokenizer_root.mkdir()
    checkpoint = _write(tmp_path / "ImageReward.pt", b"checkpoint")
    med_config = _write(tmp_path / "med_config.json", b"{}")
    source = _write(source_root / "module.py", b"VALUE = 1\n")
    tokenizer = _write(tokenizer_root / "vocab.txt", b"token\n")

    def bound(label, path):
        snapshot = _Snapshot.capture(label, path)
        return BoundAsset(
            snapshot.label,
            Path(snapshot.path),
            snapshot.sha256,
            snapshot.size,
            snapshot.identity,
        )

    assets = tuple(
        bound(label, path)
        for label, path in (
            ("reward_assets_manifest[0]", checkpoint),
            ("reward_assets_manifest[1]", med_config),
            ("reward_assets_manifest[2]", source),
            ("reward_assets_manifest[3]", tokenizer),
        )
    )
    config = {
        "schema": REWARD_RUNTIME_SCHEMA,
        "implementation": "ImageReward-v1.0",
        "dtype": "float32",
        "preprocess_sha256": "a" * 64,
        "checkpoint": str(checkpoint),
        "med_config": str(med_config),
        "source_root": str(source_root),
        "source_files": ["module.py"],
        "tokenizer_root": str(tokenizer_root),
        "tokenizer_files": ["vocab.txt"],
    }

    stage = _stage_default_reward(config, assets, tmp_path)
    try:
        assert stage.root.name.startswith("repldm-reward-")
        assert stage.checkpoint.read_bytes() == b"checkpoint"
        assert (stage.source_root / "module.py").read_bytes() == b"VALUE = 1\n"
        assert (stage.tokenizer_root / "vocab.txt").read_bytes() == b"token\n"
        assert stat.S_IMODE(stage.checkpoint.stat().st_mode) == 0o400
    finally:
        _cleanup_default_reward(stage)
    assert not stage.root.exists()


class _RendererBinding:
    def __init__(self, contract, initial_state_hash):
        self.contract = contract
        self.initial_state_hash = initial_state_hash

    def validate_component(self, renderer):
        if renderer.frame_contract_hash != self.contract["renderer_frame_contract_hash"]:
            raise ValueError("renderer frame mismatch")
        if renderer.calibration_hash != self.contract["calibration_hash"]:
            raise ValueError("renderer calibration mismatch")
        if contract_hash(renderer.contract) != self.contract["action_contract_hash"]:
            raise ValueError("renderer action contract mismatch")

    def validate_initial_renderer(self, renderer):
        if module_state_sha256(renderer) != self.initial_state_hash:
            raise ValueError("initial renderer mismatch")


def _bound(path: Path, label: str) -> BoundAsset:
    snapshot = _Snapshot.capture(label, path)
    return BoundAsset(
        snapshot.label,
        Path(snapshot.path),
        snapshot.sha256,
        snapshot.size,
        snapshot.identity,
    )


def _renderer_asset_fixture(
    tmp_path: Path, *, teacher_hash_override: str | None = None
):
    calibration = FrameCalibration(
        (True,) * 6,
        CALIBRATION_STATE_COUNT,
        (1.0,) * 6,
        (1.0,) * 6,
        manifest_sha256="1" * 64,
        source_sha256="2" * 64,
        state_provenance_sha256="3" * 64,
    )
    config = {
        "latent_channels": 2,
        "hidden_dim": 8,
        "depth": 1,
        "prompt_dim": 2,
        "state_dim": 2,
        "timestep_dim": 2,
        "coefficient_bound": 1.0,
        "max_update_ratio": 0.05,
        "preserve_moments": True,
        "theta_max": 0.05,
        "epsilon": 1e-12,
        "pre_squash_sigma": 0.25,
    }
    initial = EulerNativeFrameV1(calibration=calibration, **config)
    teacher_source = copy.deepcopy(initial)
    with torch.no_grad():
        next(teacher_source.parameters()).add_(0.125)
    initial_hash = module_state_sha256(initial)
    teacher_hash = module_state_sha256(teacher_source)
    registered_teacher_hash = teacher_hash_override or teacher_hash

    calibration_path = _json(tmp_path / "real-calibration.json", calibration.to_dict())
    frame_path = _json(
        tmp_path / "real-frame.json",
        {
            "schema": "repldm.euler_native_frame_runtime.v1",
            "renderer_frame_contract_hash": initial.frame_contract_hash,
            "action_contract_hash": contract_hash(initial.contract),
            "renderer_config": config,
        },
    )
    initial_checkpoint = tmp_path / "initial-renderer.pt"
    torch.save(initial.state_dict(), initial_checkpoint)
    teacher_checkpoint = tmp_path / "opsd-teacher.pt"
    torch.save(
        {
            "schema": "repldm.latent_renderer_checkpoint.v1",
            "model": teacher_source.state_dict(),
        },
        teacher_checkpoint,
    )

    def file_binding(path: Path):
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }

    initial_state_path = _json(
        tmp_path / "initial-renderer.json",
        {
            "schema": "repldm.renderer_initial_state.v1",
            "renderer_state_sha256": initial_hash,
            "checkpoint": file_binding(initial_checkpoint),
        },
    )
    teacher_state_path = _json(
        tmp_path / "opsd-teacher.json",
        {
            "schema": "repldm.renderer_opsd_teacher_state.v1",
            "status": "frozen",
            "role": "T_OPSD",
            "f0_run_contract_sha256": "4" * 64,
            "renderer_state_sha256": registered_teacher_hash,
            "checkpoint": file_binding(teacher_checkpoint),
        },
    )
    contract = {
        "renderer_frame_contract_hash": initial.frame_contract_hash,
        "calibration_hash": initial.calibration_hash,
        "action_contract_hash": contract_hash(initial.contract),
        "opsd_teacher_renderer_sha256": registered_teacher_hash,
    }
    binding = _RendererBinding(contract, initial_hash)
    descriptors = {
        "calibration": _bound(calibration_path, "calibration"),
        "renderer_frame_contract": _bound(
            frame_path, "renderer_frame_contract"
        ),
        "initial_renderer_state": _bound(
            initial_state_path, "initial_renderer_state"
        ),
        "opsd_teacher_state": _bound(teacher_state_path, "opsd_teacher_state"),
    }
    payload = (
        _bound(initial_checkpoint, "initial renderer checkpoint"),
        _bound(teacher_checkpoint, "OPSD teacher checkpoint"),
    )
    return binding, descriptors, payload, teacher_hash


def test_default_renderer_loader_safely_loads_frozen_independent_opsd_teacher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    binding, descriptors, payload, teacher_hash = _renderer_asset_fixture(tmp_path)
    original_load = torch.load
    weights_only_calls = []

    def observed_load(source, map_location=None, *, weights_only=False):
        weights_only_calls.append(weights_only)
        return original_load(
            source, map_location=map_location, weights_only=weights_only
        )

    monkeypatch.setattr(torch, "load", observed_load)
    bundle = _load_default_renderers(
        binding, descriptors, payload, torch.device("cpu")
    )

    assert weights_only_calls == [True, True]
    assert bundle.teacher_renderer is not None
    assert bundle.teacher_checkpoint_provenance is not None
    assert module_state_sha256(bundle.teacher_renderer) == teacher_hash
    assert bundle.teacher_renderer.training is False
    assert all(
        not parameter.requires_grad
        for parameter in bundle.teacher_renderer.parameters()
    )
    checked_hash = _require_independent_frozen_teacher(
        bundle.teacher_renderer,
        bundle.renderer,
        bundle.reference_renderer,
        binding=binding,
        contract=binding.contract,
        device=torch.device("cpu"),
    )
    assert checked_hash == teacher_hash
    assert (
        _require_teacher_checkpoint_provenance(
            bundle.teacher_checkpoint_provenance,
            expected_state_hash=teacher_hash,
        )
        is bundle.teacher_checkpoint_provenance
    )


def test_default_renderer_loader_rejects_opsd_teacher_state_hash_mismatch(
    tmp_path: Path,
):
    binding, descriptors, payload, _teacher_hash = _renderer_asset_fixture(
        tmp_path, teacher_hash_override="f" * 64
    )

    with pytest.raises(ValueError, match="teacher renderer state differs"):
        _load_default_renderers(binding, descriptors, payload, torch.device("cpu"))


def _enable_gated_runtime(
    binding: _Binding,
    tmp_path: Path,
    bundle: RendererBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import latent_renderer_training.gates as gates

    descriptor_paths = {
        label: _json(tmp_path / f"{label}.json", {})
        for label in (
            "reward_statistics",
            "f0_gate",
            "opsd_teacher_state",
            "cohort_manifest",
        )
    }
    binding.run_artifacts.descriptors += tuple(
        _Snapshot.capture(label, path) for label, path in descriptor_paths.items()
    )
    teacher_provenance = bundle.teacher_checkpoint_provenance
    teacher_checkpoint = Path(teacher_provenance.path)
    binding.run_artifacts.payload_files += (
        _Snapshot.capture("OPSD teacher checkpoint", teacher_checkpoint),
    )
    f0_hash = "6" * 64
    teacher_hash = module_state_sha256(bundle.teacher_renderer)
    binding.contract.update(
        {
            "method": "dpo",
            "renderer_frame_contract_hash": bundle.renderer.frame_contract_hash,
            "calibration_hash": bundle.renderer.calibration_hash,
            "action_contract_hash": contract_hash(bundle.renderer.contract),
            "opsd_teacher_renderer_sha256": teacher_hash,
            "f0_gate_sha256": "7" * 64,
            "reward_statistics_sha256": "8" * 64,
            "cohort_id": "cohort-test",
            "cohort_manifest_sha256": "9" * 64,
        }
    )
    checkpoint_binding = {
        "path": str(teacher_checkpoint),
        "sha256": teacher_provenance.sha256,
        "bytes": teacher_provenance.bytes,
    }
    monkeypatch.setattr(
        gates,
        "validate_reward_statistics",
        lambda value, *, contract: {"status": "validated"},
    )
    monkeypatch.setattr(
        gates,
        "validate_f0_gate",
        lambda value, *, contract: (
            {
                "f0_run_contract_sha256": f0_hash,
                "reward_statistics_sha256": contract["reward_statistics_sha256"],
            },
            (),
        ),
    )
    monkeypatch.setattr(
        gates,
        "validate_opsd_teacher_state",
        lambda value, *, contract: (
            {
                "f0_run_contract_sha256": f0_hash,
                "renderer_state_sha256": teacher_hash,
            },
            checkpoint_binding,
        ),
    )
    monkeypatch.setattr(
        gates,
        "validate_training_cohort",
        lambda value, *, contract: {"cohort_id": contract["cohort_id"]},
    )


def test_gated_factory_exposes_verified_teacher_and_checkpoint_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    renderer_binding, descriptors, payload, teacher_hash = _renderer_asset_fixture(
        tmp_path
    )
    bundle = _load_default_renderers(
        renderer_binding, descriptors, payload, torch.device("cpu")
    )
    binding, events = _fixture(tmp_path)
    _enable_gated_runtime(binding, tmp_path, bundle, monkeypatch)
    dependencies = replace(
        _dependencies(binding, events, tmp_path),
        load_renderers=lambda *args: bundle,
    )

    with build_training_runtime(binding, dependencies=dependencies) as runtime:
        assert runtime.teacher_renderer is bundle.teacher_renderer
        assert runtime.teacher_checkpoint_provenance is (
            bundle.teacher_checkpoint_provenance
        )
        assert module_state_sha256(runtime.teacher_renderer) == teacher_hash
        assert runtime.provenance["opsd_teacher_renderer_sha256"] == teacher_hash
        assert runtime.provenance["opsd_teacher_checkpoint_sha256"] == (
            bundle.teacher_checkpoint_provenance.sha256
        )
        assert runtime.provenance["opsd_teacher_checkpoint_bytes"] == (
            bundle.teacher_checkpoint_provenance.bytes
        )


def test_gated_factory_rejects_teacher_object_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    renderer_binding, descriptors, payload, _teacher_hash = _renderer_asset_fixture(
        tmp_path
    )
    bundle = _load_default_renderers(
        renderer_binding, descriptors, payload, torch.device("cpu")
    )
    binding, events = _fixture(tmp_path)
    _enable_gated_runtime(binding, tmp_path, bundle, monkeypatch)
    aliased = replace(bundle, teacher_renderer=bundle.renderer)
    dependencies = replace(
        _dependencies(binding, events, tmp_path),
        load_renderers=lambda *args: aliased,
    )

    with pytest.raises(RuntimeError, match="distinct renderer object"):
        build_training_runtime(binding, dependencies=dependencies)


def test_teacher_validation_rejects_student_parameter_storage_alias(tmp_path: Path):
    binding, descriptors, payload, _teacher_hash = _renderer_asset_fixture(tmp_path)
    bundle = _load_default_renderers(
        binding, descriptors, payload, torch.device("cpu")
    )
    teacher_parameter = next(bundle.teacher_renderer.parameters())
    student_parameter = next(bundle.renderer.parameters())
    teacher_parameter.data = student_parameter.data

    with pytest.raises(RuntimeError, match="storage aliases"):
        _require_independent_frozen_teacher(
            bundle.teacher_renderer,
            bundle.renderer,
            bundle.reference_renderer,
            binding=binding,
            contract=binding.contract,
            device=torch.device("cpu"),
        )
