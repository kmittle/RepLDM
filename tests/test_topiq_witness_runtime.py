from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import latent_renderer_training.gates as gates_module
import latent_renderer_training.runtime_factory as runtime_factory_module
from latent_renderer_training.ledger import QueryLedger
from latent_renderer_training.operations import (
    LedgeredOperationExecutor,
    OperationContext,
)
from latent_renderer_training.renderer import tensor_sha256
from latent_renderer_training.runtime_factory import (
    BoundAsset,
    WITNESS_RUNTIME_SCHEMA,
    WitnessStage,
    _cleanup_default_witness,
    _load_default_witness,
    _stage_default_witness,
    build_training_runtime,
)
from latent_renderer_training.run_contract import TrainingRunContract
from latent_renderer_training.witnesses import (
    TOPIQ_NR_MODEL_ID,
    TOPIQ_NR_PREPROCESS_SHA256,
    TOPIQ_NR_ROLE,
    TopiqNrTensorWitness,
)
from tests.test_latent_renderer_training import (
    _complete_run_contract,
    _training_authorization,
)
from tests.test_runtime_factory import (
    _Snapshot,
    _dependencies,
    _fixture,
    _json,
    _write,
)


class _Metric(nn.Module):
    def __init__(self, *, result: str = "valid") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.result = result
        self.grad_enabled: list[bool] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.grad_enabled.append(torch.is_grad_enabled())
        if self.result == "wrong_shape":
            return torch.zeros(images.shape[0], 2, device=images.device)
        if self.result == "nan":
            return torch.full((images.shape[0], 1), float("nan"), device=images.device)
        return images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1) * self.weight


class _FactoryWitness(nn.Module):
    model_id = TOPIQ_NR_MODEL_ID
    role = TOPIQ_NR_ROLE
    preprocess_sha256 = TOPIQ_NR_PREPROCESS_SHA256

    def __init__(self, executor: object) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1), requires_grad=False)
        self.operation_executor = executor
        self.eval()


def _bound(label: str, path: Path) -> BoundAsset:
    status = path.stat()
    return BoundAsset(
        label=label,
        path=path,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        size=status.st_size,
        identity=(
            status.st_dev,
            status.st_ino,
            status.st_size,
            status.st_mtime_ns,
            status.st_ctime_ns,
        ),
    )


def _stage_parts(tmp_path: Path):
    checkpoint = _write(tmp_path / "topiq.pth", b"topiq")
    backbone = _write(tmp_path / "resnet.safetensors", b"resnet")
    pyiqa_root = tmp_path / "pyiqa-src"
    timm_root = tmp_path / "timm-src"
    pyiqa_root.mkdir()
    timm_root.mkdir()
    pyiqa = _write(pyiqa_root / "__init__.py", b"# pyiqa\n")
    timm = _write(timm_root / "__init__.py", b"# timm\n")
    config = {
        "schema": WITNESS_RUNTIME_SCHEMA,
        "implementation": TOPIQ_NR_MODEL_ID,
        "role": TOPIQ_NR_ROLE,
        "dtype": "float32",
        "preprocess_sha256": TOPIQ_NR_PREPROCESS_SHA256,
        "checkpoint": str(checkpoint),
        "backbone": str(backbone),
        "package_versions": {
            "pyiqa": "0.1.15.post2",
            "timm": "1.0.12",
            "torch": "2.5.1",
            "torchvision": "0.20.1",
            "safetensors": "0.4.5",
        },
        "source_packages": {
            "pyiqa": {"root": str(pyiqa_root), "files": ["__init__.py"]},
            "timm": {"root": str(timm_root), "files": ["__init__.py"]},
        },
    }
    assets = tuple(
        _bound(f"witness_assets_manifest[{index}]", path)
        for index, path in enumerate((checkpoint, backbone, pyiqa, timm))
    )
    return config, assets


def test_tensor_witness_is_native_batch_safe_frozen_and_no_grad() -> None:
    metric = _Metric()
    witness = TopiqNrTensorWitness(metric)
    images = torch.rand(3, 3, 19, 23, requires_grad=True)

    scores = witness.score_tensor(images)

    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()
    assert not scores.requires_grad
    assert metric.grad_enabled == [False]
    assert not metric.training
    assert not metric.weight.requires_grad


@pytest.mark.parametrize(
    "images",
    [
        torch.zeros(3, 8, 8),
        torch.zeros(1, 1, 8, 8),
        torch.zeros(1, 3, 0, 8),
        torch.zeros(1, 3, 8, 8, dtype=torch.int64),
        torch.full((1, 3, 8, 8), float("nan")),
        torch.full((1, 3, 8, 8), -0.01),
        torch.full((1, 3, 8, 8), 1.01),
    ],
)
def test_tensor_witness_rejects_invalid_input(images: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        TopiqNrTensorWitness(_Metric()).score_tensor(images)


@pytest.mark.parametrize("result", ["wrong_shape", "nan"])
def test_tensor_witness_rejects_invalid_metric_output(result: str) -> None:
    with pytest.raises((ValueError, RuntimeError)):
        TopiqNrTensorWitness(_Metric(result=result))(torch.rand(2, 3, 8, 8))


def test_witness_stage_binds_every_asset_and_cleans_independently(
    tmp_path: Path,
) -> None:
    config, assets = _stage_parts(tmp_path)
    parent = tmp_path / "stages"
    parent.mkdir()

    stage = _stage_default_witness(config, assets, parent)

    assert stage.checkpoint.read_bytes() == b"topiq"
    assert stage.backbone.read_bytes() == b"resnet"
    assert (stage.source_root / "pyiqa" / "__init__.py").is_file()
    assert (stage.source_root / "timm" / "__init__.py").is_file()
    assert dict(stage.package_versions) == config["package_versions"]
    _cleanup_default_witness(stage)
    assert not stage.root.exists()


def test_witness_stage_fails_closed_for_missing_or_unassigned_assets(
    tmp_path: Path,
) -> None:
    config, assets = _stage_parts(tmp_path)
    with pytest.raises(ValueError, match="checkpoint or ResNet-50"):
        _stage_default_witness({**config, "checkpoint": str(tmp_path / "missing")}, assets, None)
    with pytest.raises(ValueError, match="absent or duplicate source"):
        _stage_default_witness(config, assets[:-1], None)
    extra = _write(tmp_path / "unassigned.bin", b"extra")
    with pytest.raises(ValueError, match="unassigned"):
        _stage_default_witness(config, (*assets, _bound("extra", extra)), None)
    with pytest.raises(ValueError, match="preprocessing hash"):
        _stage_default_witness(
            {**config, "preprocess_sha256": "0" * 64}, assets, None
        )


def test_witness_stage_detects_source_bytes_changed_after_verification(
    tmp_path: Path,
) -> None:
    config, assets = _stage_parts(tmp_path)
    assets[-1].path.write_bytes(b"# changed timm source\n")

    with pytest.raises(RuntimeError, match="identity changed"):
        _stage_default_witness(config, assets, None)


def test_witness_loader_rejects_package_version_mismatch_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, assets = _stage_parts(tmp_path)
    stage = _stage_default_witness(config, assets, None)
    monkeypatch.setattr(
        runtime_factory_module.importlib_metadata,
        "version",
        lambda distribution: "unexpected-version",
    )
    try:
        with pytest.raises(ValueError, match="version differs"):
            _load_default_witness(stage, torch.device("cpu"), torch.float32)
    finally:
        _cleanup_default_witness(stage)


def test_witness_loader_blocks_network_and_forces_bound_weights_only_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "safe-topiq.pth"
    torch.save({"weight": torch.tensor(2.0)}, checkpoint)
    backbone = _write(tmp_path / "safe-resnet.safetensors", b"bound-backbone")
    pyiqa_root = tmp_path / "safe-pyiqa"
    timm_root = tmp_path / "safe-timm"
    pyiqa_root.mkdir()
    timm_root.mkdir()
    pyiqa = _write(
        pyiqa_root / "__init__.py",
        b"import socket\n"
        b"import torch\n"
        b"from torch import nn\n"
        b"import timm\n"
        b"class Metric(nn.Module):\n"
        b"    def __init__(self):\n"
        b"        super().__init__()\n"
        b"        self.weight = nn.Parameter(torch.tensor(0.0))\n"
        b"    def forward(self, images):\n"
        b"        return images.mean((1, 2, 3)).unsqueeze(1) * self.weight\n"
        b"def create_metric(name, device=None, pretrained_model_path=None):\n"
        b"    try:\n"
        b"        socket.create_connection(('example.invalid', 443))\n"
        b"    except RuntimeError as error:\n"
        b"        if 'network access is disabled' not in str(error):\n"
        b"            raise\n"
        b"    else:\n"
        b"        raise RuntimeError('network was not blocked')\n"
        b"    timm.create_model('resnet50', pretrained=True)\n"
        b"    metric = Metric()\n"
        b"    state = torch.load(pretrained_model_path, weights_only=False)\n"
        b"    metric.load_state_dict(state)\n"
        b"    return metric.to(device)\n",
    )
    timm = _write(
        timm_root / "__init__.py",
        b"from types import SimpleNamespace\n"
        b"from torch import nn\n"
        b"class Backbone(nn.Module):\n"
        b"    def __init__(self):\n"
        b"        super().__init__()\n"
        b"        self.weight = nn.Parameter(__import__('torch').ones(1))\n"
        b"def create_model(name, *args, **kwargs):\n"
        b"    return Backbone()\n"
        b"class Models:\n"
        b"    @staticmethod\n"
        b"    def load_checkpoint(model, path, strict=False):\n"
        b"        return SimpleNamespace(missing_keys=[], unexpected_keys=['fc.bias', 'fc.weight'])\n"
        b"models = Models()\n",
    )
    versions = {
        "pyiqa": "pinned-pyiqa",
        "timm": "pinned-timm",
        "torch": "pinned-torch",
        "torchvision": "pinned-torchvision",
        "safetensors": "pinned-safetensors",
    }
    config = {
        "schema": WITNESS_RUNTIME_SCHEMA,
        "implementation": TOPIQ_NR_MODEL_ID,
        "role": TOPIQ_NR_ROLE,
        "dtype": "float32",
        "preprocess_sha256": TOPIQ_NR_PREPROCESS_SHA256,
        "checkpoint": str(checkpoint),
        "backbone": str(backbone),
        "package_versions": versions,
        "source_packages": {
            "pyiqa": {"root": str(pyiqa_root), "files": ["__init__.py"]},
            "timm": {"root": str(timm_root), "files": ["__init__.py"]},
        },
    }
    assets = tuple(
        _bound(f"witness_assets_manifest[{index}]", path)
        for index, path in enumerate((checkpoint, backbone, pyiqa, timm))
    )
    stage = _stage_default_witness(config, assets, None)
    monkeypatch.setattr(
        runtime_factory_module.importlib_metadata,
        "version",
        lambda distribution: versions[distribution],
    )
    try:
        metric = _load_default_witness(stage, torch.device("cpu"), torch.float32)
        output = metric(torch.ones(1, 3, 4, 4))
        torch.testing.assert_close(output, torch.tensor([[2.0]]))
        assert not metric.weight.requires_grad
    finally:
        _cleanup_default_witness(stage)


def _add_factory_witness(binding: object, tmp_path: Path) -> None:
    checkpoint = _write(tmp_path / "factory-topiq.pth", b"topiq")
    backbone = _write(tmp_path / "factory-resnet.safetensors", b"resnet")
    source_root = tmp_path / "factory-source"
    pyiqa_root = source_root / "pyiqa"
    timm_root = source_root / "timm"
    pyiqa_root.mkdir(parents=True)
    timm_root.mkdir(parents=True)
    pyiqa = _write(pyiqa_root / "__init__.py", b"# pyiqa\n")
    timm = _write(timm_root / "__init__.py", b"# timm\n")

    def row(path: Path) -> dict[str, object]:
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }

    manifest = _json(
        tmp_path / "factory-witness-assets.json",
        {
            "schema": "repldm.file_manifest.v1",
            "files": [row(value) for value in (checkpoint, backbone, pyiqa, timm)],
        },
    )
    config = _json(
        tmp_path / "factory-witness.json",
        {
            "schema": WITNESS_RUNTIME_SCHEMA,
            "implementation": TOPIQ_NR_MODEL_ID,
            "role": TOPIQ_NR_ROLE,
            "dtype": "float32",
            "preprocess_sha256": TOPIQ_NR_PREPROCESS_SHA256,
            "checkpoint": str(checkpoint),
            "backbone": str(backbone),
            "package_versions": {
                "pyiqa": "test",
                "timm": "test",
                "torch": "test",
                "torchvision": "test",
                "safetensors": "test",
            },
            "source_packages": {
                "pyiqa": {"root": str(pyiqa_root), "files": ["__init__.py"]},
                "timm": {"root": str(timm_root), "files": ["__init__.py"]},
            },
        },
    )
    artifacts = binding.run_artifacts
    artifacts.descriptors = (
        *artifacts.descriptors,
        _Snapshot.capture("witness_config", config),
        _Snapshot.capture("witness_assets_manifest", manifest),
    )
    artifacts.payload_files = (
        *artifacts.payload_files,
        *(
            _Snapshot.capture(f"witness_assets_manifest[{index}]", value)
            for index, value in enumerate((checkpoint, backbone, pyiqa, timm))
        ),
    )
    binding.contract.update(
        {
            "method": "f0",
            "witness_preprocess_sha256": TOPIQ_NR_PREPROCESS_SHA256,
        }
    )


def test_factory_loads_and_cleans_witness_as_an_independent_stage(
    tmp_path: Path,
) -> None:
    binding, events = _fixture(tmp_path)
    _add_factory_witness(binding, tmp_path)
    dependencies = _dependencies(binding, events, tmp_path)

    def stage_witness(config, assets, parent):
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        with pytest.raises(RuntimeError, match="network access is disabled"):
            socket.create_connection(("example.invalid", 443))
        assert config["implementation"] == TOPIQ_NR_MODEL_ID
        assert len(assets) == 4
        events.append("witness.stage")
        return WitnessStage(
            tmp_path / "witness-stage",
            tmp_path / "topiq",
            tmp_path / "backbone",
            tmp_path / "source",
            SimpleNamespace(items=lambda: ()),
        )

    def load_witness(stage, device, dtype):
        assert device == torch.device("cpu") and dtype is torch.float32
        events.append("witness.load")
        return _Metric()

    def cleanup_witness(stage):
        events.append("witness.cleanup")

    dependencies = replace(
        dependencies,
        stage_witness=stage_witness,
        load_witness=load_witness,
        cleanup_witness=cleanup_witness,
        wrap_witness=lambda model, executor: _FactoryWitness(executor),
        witness_preprocess_sha256=TOPIQ_NR_PREPROCESS_SHA256,
    )

    runtime = build_training_runtime(binding, dependencies=dependencies)

    assert runtime.witness_model is not None
    assert events.index("witness.load") < events.index("model.stage")
    assert runtime.provenance["witness_model"] == TOPIQ_NR_MODEL_ID
    runtime.close()
    assert events[-3:] == ["reward.cleanup", "model.cleanup", "witness.cleanup"]


def test_factory_failure_after_witness_load_cleans_witness_and_model_stage(
    tmp_path: Path,
) -> None:
    binding, events = _fixture(tmp_path)
    _add_factory_witness(binding, tmp_path)
    dependencies = _dependencies(binding, events, tmp_path)

    def stage_witness(config, assets, parent):
        events.append("witness.stage")
        return WitnessStage(
            tmp_path / "witness-stage",
            tmp_path / "topiq",
            tmp_path / "backbone",
            tmp_path / "source",
            {},
        )

    def fail_pipeline(*args):
        events.append("pipeline.load.failed")
        raise RuntimeError("planned pipeline failure")

    dependencies = replace(
        dependencies,
        stage_witness=stage_witness,
        load_witness=lambda *args: _Metric(),
        cleanup_witness=lambda stage: events.append("witness.cleanup"),
        wrap_witness=lambda model, executor: _FactoryWitness(executor),
        witness_preprocess_sha256=TOPIQ_NR_PREPROCESS_SHA256,
        load_pipeline=fail_pipeline,
    )

    with pytest.raises(RuntimeError, match="planned pipeline failure"):
        build_training_runtime(binding, dependencies=dependencies)

    assert events[-2:] == ["model.cleanup", "witness.cleanup"]


def test_f0_contract_requires_independent_witness_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        gates_module, "validate_f0_power_evidence", lambda *args, **kwargs: None,
        raising=False,
    )
    authorization = _training_authorization(tmp_path, monkeypatch)
    contract = _complete_run_contract(authorization, method="f0")
    missing = dict(contract)
    missing.pop("witness_preprocess_sha256")

    with pytest.raises(ValueError, match="witness_preprocess_sha256"):
        TrainingRunContract.from_mapping(missing)


def _formal_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    budget: dict[str, int],
    *,
    ledger_name: str = "witness-ledger.jsonl",
):
    monkeypatch.setattr(
        gates_module, "validate_f0_power_evidence", lambda *args, **kwargs: None,
        raising=False,
    )
    authorization = _training_authorization(tmp_path, monkeypatch)
    contract = _complete_run_contract(
        authorization, method="f0", query_budget=budget
    )
    binding = authorization.bind_run_contract(contract)
    ledger = QueryLedger(
        tmp_path / ledger_name,
        budget,
        run_contract=binding.run_contract,
        authorization_binding=binding,
    )
    return (
        LedgeredOperationExecutor(ledger, authorization_binding=binding),
        ledger,
        contract,
        binding,
    )


def _decode_context(contract: dict[str, object]) -> OperationContext:
    return OperationContext(
        split="validation",
        prompt="source:row-0",
        seed=7,
        step=50,
        prefix=0,
        branch="anchor",
        action={"setting": "conference_settings"},
        checkpoint_hash=str(contract["initial_renderer_state_sha256"]),
    )


def _witness_context(
    parent_context: OperationContext,
    images: torch.Tensor,
    parent_output_hash: str,
) -> OperationContext:
    return replace(
        parent_context,
        image_hash=tensor_sha256(images),
        cached_parent=parent_output_hash,
    )


def test_formal_witness_uses_shared_reward_budget_with_distinct_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget = {"vae_decode": 1, "reward_forward": 2}
    executor, ledger, contract, _binding = _formal_executor(
        tmp_path, monkeypatch, budget
    )
    images = torch.rand(1, 3, 8, 8)
    decoded, decode_receipt = executor.execute_with_receipt(
        "vae_decode", _decode_context(contract), lambda: images
    )
    assert decoded is images
    context = _witness_context(decode_receipt.context, images, decode_receipt.output_hash)
    executor.execute(
        "reward_forward",
        context,
        lambda: torch.tensor([0.1]),
        scalar_or_gradient="scalar",
    )
    scores, witness_receipt = TopiqNrTensorWitness(
        _Metric(), operation_executor=executor
    ).score_with_receipt(images, context, parent_receipt=decode_receipt)

    assert scores.shape == (1,)
    assert witness_receipt.kind == "reward_forward"
    assert witness_receipt.model == TOPIQ_NR_MODEL_ID
    records = [
        json.loads(line)
        for line in ledger.path.read_text(encoding="utf-8").splitlines()
    ]
    reservations = [row for row in records if row["type"] == "reservation"]
    receipts = [row for row in records if row["type"] == "receipt"]
    assert [row["kind"] for row in reservations] == [
        "vae_decode",
        "reward_forward",
        "reward_forward",
    ]
    assert [(row["metadata"]["model"], row["metadata"]["role"]) for row in reservations] == [
        ("repldm-runtime", "vae_decode"),
        ("ImageReward-v1.0", "training_reward"),
        (TOPIQ_NR_MODEL_ID, TOPIQ_NR_ROLE),
    ]
    assert receipts[2]["result"]["model"] == TOPIQ_NR_MODEL_ID
    assert receipts[2]["result"]["role"] == TOPIQ_NR_ROLE
    assert (
        receipts[2]["result"]["model_config_sha256"]
        == contract["witness_config_sha256"]
    )
    assert (
        receipts[2]["result"]["model_asset_manifest_sha256"]
        == contract["witness_asset_manifest_sha256"]
    )
    assert (
        receipts[2]["result"]["reward_preprocess_hash"]
        == TOPIQ_NR_PREPROCESS_SHA256
    )
    assert receipts[2]["result"]["parent_reservation_id"] == decode_receipt.reservation_id
    assert ledger.summary()["reserved"] == budget


def test_formal_witness_rejects_parent_from_another_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget = {"vae_decode": 1, "reward_forward": 1}
    executor, ledger, contract, binding = _formal_executor(
        tmp_path, monkeypatch, budget, ledger_name="primary.jsonl"
    )
    foreign_ledger = QueryLedger(
        tmp_path / "foreign.jsonl",
        budget,
        run_contract=binding.run_contract,
        authorization_binding=binding,
    )
    foreign = LedgeredOperationExecutor(
        foreign_ledger, authorization_binding=binding
    )
    images = torch.rand(1, 3, 8, 8)
    _decoded, parent = foreign.execute_with_receipt(
        "vae_decode", _decode_context(contract), lambda: images
    )
    context = _witness_context(parent.context, images, parent.output_hash)

    with pytest.raises(TypeError, match="not issued by this operation executor"):
        TopiqNrTensorWitness(
            _Metric(), operation_executor=executor
        ).score_with_receipt(images, context, parent_receipt=parent)

    assert not ledger.path.exists()


def test_formal_witness_rejects_decode_receipt_for_a_different_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget = {"vae_decode": 1, "reward_forward": 1}
    executor, ledger, contract, _binding = _formal_executor(
        tmp_path, monkeypatch, budget
    )
    first = torch.rand(1, 3, 8, 8)
    _decoded, parent = executor.execute_with_receipt(
        "vae_decode", _decode_context(contract), lambda: first
    )
    second = first.clone()
    second[0, 0, 0, 0] = 1.0 - second[0, 0, 0, 0]
    context = _witness_context(parent.context, second, parent.output_hash)

    with pytest.raises(ValueError, match="different decoded tensor"):
        TopiqNrTensorWitness(
            _Metric(), operation_executor=executor
        ).score_with_receipt(second, context, parent_receipt=parent)

    assert ledger.summary()["reserved"] == {"vae_decode": 1, "reward_forward": 0}


def test_formal_witness_rejects_non_decode_parent_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget = {"reward_forward": 2}
    executor, ledger, contract, _binding = _formal_executor(
        tmp_path, monkeypatch, budget
    )
    images = torch.rand(1, 3, 8, 8)
    _value, parent = executor.execute_with_receipt(
        "reward_forward", _decode_context(contract), lambda: images
    )
    context = _witness_context(parent.context, images, parent.output_hash)

    with pytest.raises(ValueError, match="not a VAE decode receipt"):
        TopiqNrTensorWitness(
            _Metric(), operation_executor=executor
        ).score_with_receipt(images, context, parent_receipt=parent)

    assert ledger.summary()["reserved"] == {"reward_forward": 1}
