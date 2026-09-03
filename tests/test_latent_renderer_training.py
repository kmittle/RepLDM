import json
import hashlib
import random
import copy
from dataclasses import replace
from pathlib import Path
import subprocess

import pytest
import torch

from latent_renderer_training.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
import latent_renderer_training.authorization as authorization_module
import latent_renderer_training.launcher as launcher_module
from latent_renderer_training.authorization import (
    AuthorizationBinding,
    TrainingAuthorization,
    require_authorization_binding,
    write_authorization_receipt,
)
from latent_renderer_training.cli import main as training_probe
from latent_renderer_training.contracts import ActionSpaceContract, contract_hash
from latent_renderer_training.distributions import SquashedGaussian, transformed_gaussian_kl
from latent_renderer_training.f0_metrics import (
    build_f0_screen_registration,
    write_f0_phase_evidence,
    write_f0_screen_registration_evidence,
)
from latent_renderer_training.ledger import QueryLedger
from latent_renderer_training.operations import LedgeredOperationExecutor
from latent_renderer_training.collector import EulerNativeRolloutCollector
from latent_renderer_training.launcher import (
    TRAINING_CONFIG_SCHEMA,
    _load_config,
    launch_training,
)
from latent_renderer_training.runtime import METHOD_RESULT_SCHEMA
from latent_renderer_training.objectives import (
    dpo_loss,
    normalized_transition_loss,
    normalized_transition_losses,
    opd_loss,
    per_decision_rl_loss,
    search_distill_loss,
)
from latent_renderer_training.rollout import (
    Transition,
    collect_antithetic_rollout,
    replay_shared_prefix,
)
from latent_renderer_training.trainer import RendererTrainer, run_updates
from latent_renderer_training.run_contract import RUN_CONTRACT_SCHEMA, TrainingRunContract
from latent_renderer_training.artifacts import module_state_sha256
from latent_renderer_training.renderer import (
    CALIBRATION_STATE_COUNT,
    EulerNativeFrameV1,
    FrameCalibration,
)
from latent_renderer_training.teachers import TargetStepConfig, construct_reward_targets
from latent_renderer_training.witnesses import (
    TOPIQ_NR_PREPROCESS_SHA256 as _TOPIQ_NR_PREPROCESS_SHA256,
)
from tests.test_f0_metrics import (
    _passing_rows as _passing_metric_rows,
    _sealed_metric_ledger,
)
from tests.test_selected_view_release import COMMIT, SelectedReleaseFixture


class _BoundLinear(torch.nn.Linear):
    """Tiny renderer-shaped module for formal trainer tests."""

    def __init__(self) -> None:
        super().__init__(1, 1)
        self.frame_contract_hash = "f" * 64
        self.calibration_hash = "c" * 64
        self.contract = ActionSpaceContract(1, (True,))


def _passing_f0_rows(phase: str, **kwargs) -> list[dict[str, object]]:
    return copy.deepcopy(_passing_metric_rows(phase, **kwargs))


_F0_LEDGER_CACHE: dict[tuple[str, ...], tuple[bytes, bytes]] = {}


def _ledger_metadata(kind="reward_forward", amount=1):
    digest = "0" * 64
    return {
        "code_hash": digest,
        "data_hash": digest,
        "checkpoint_hash": digest,
        "method_allocation": {"kind": kind, "amount": amount},
        "split": "train",
        "prompt": "prompt-0",
        "seed": 0,
        "step": 0,
        "prefix": 0,
        "branch": "anchor",
        "action": [0.0, 0.0],
    }


def _ledger_result():
    return {
        "image_hash": "1" * 64,
        "reward_preprocess_hash": "2" * 64,
        "scalar_or_gradient": "scalar",
        "cached_parent": None,
        "value": 0.5,
    }


def _training_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    validator=None,
) -> TrainingAuthorization:
    """Create a complete two-stage release for trainer contract tests."""
    monkeypatch.setattr(
        authorization_module,
        "_validate_selected_view_release",
        validator if validator is not None else (lambda *args: None),
    )
    # These tests exercise the training capability after a selected release
    # has already been accepted.  The production runtime registry is tested
    # separately; keep this CPU fixture independent of heavyweight gate models.
    monkeypatch.setattr(
        authorization_module, "_validate_selected_view_runtime", lambda *args: None
    )
    monkeypatch.setattr(authorization_module, "_git_state", lambda *args: (COMMIT, True))
    monkeypatch.setattr(
        authorization_module, "_git_upstream", lambda *args, **kwargs: None
    )
    fixture = SelectedReleaseFixture(tmp_path)
    return write_authorization_receipt(
        tmp_path / "training_authorization.json",
        selected_view_manifest=fixture.release / "manifest.json",
        code_commit=COMMIT,
        repository_root=Path(__file__).resolve().parents[1],
    )


def _forged_authorization(
    authorized: TrainingAuthorization,
    authorization_type=TrainingAuthorization,
) -> TrainingAuthorization:
    return authorization_type(
        receipt_path=authorized.receipt_path,
        selected_view_manifest_path=authorized.selected_view_manifest_path,
        candidate_parent_manifest_path=authorized.candidate_parent_manifest_path,
        selected_config_path=authorized.selected_config_path,
        selected_payload_path=authorized.selected_payload_path,
        selected_view_manifest_sha256=authorized.selected_view_manifest_sha256,
        candidate_parent_manifest_sha256=authorized.candidate_parent_manifest_sha256,
        selected_config_sha256=authorized.selected_config_sha256,
        selected_payload_sha256=authorized.selected_payload_sha256,
        selected_view_release_id=authorized.selected_view_release_id,
        candidate_parent_release_id=authorized.candidate_parent_release_id,
        selected_view_id=authorized.selected_view_id,
        selected_rows=authorized.selected_rows,
        code_commit=authorized.code_commit,
        repository_root=authorized.repository_root,
    )


def _complete_run_contract(
    authorization: TrainingAuthorization, *, query_budget=None, **overrides
) -> dict[str, object]:
    """Build a minimal complete contract for capability-bound CPU tests."""
    method = str(overrides.get("method", "rl"))
    frame_hash = str(overrides.get("renderer_frame_contract_hash", "f" * 64))
    calibration_hash = str(overrides.get("calibration_hash", "c" * 64))
    action_hash = str(overrides.get("action_contract_hash", "b" * 64))
    provider_hash = str(overrides.get("basis_provider_contract_hash", "8" * 64))
    initial_state_hash = str(overrides.get("initial_renderer_state_sha256", "9" * 64))
    artifact_key = hashlib.sha256(
        json.dumps(
            {
                "frame": frame_hash,
                "calibration": calibration_hash,
                "action": action_hash,
                "provider": provider_hash,
                "initial_state": initial_state_hash,
            },
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()[:16]
    artifact_root = authorization.receipt_path.parent / f"run-artifacts-{artifact_key}"
    artifact_root.mkdir(exist_ok=True)

    def write_once(path: Path, payload: bytes) -> Path:
        if path.exists():
            if path.read_bytes() != payload:
                raise RuntimeError(f"test artifact collision at {path}")
        else:
            path.write_bytes(payload)
        return path

    def canonical_file(path: Path, value: object) -> Path:
        return write_once(
            path,
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
            + b"\n",
        )

    def descriptor(path: Path) -> dict[str, str]:
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def payload_entry(path: Path) -> dict[str, object]:
        payload_bytes = path.read_bytes()
        return {
            "path": str(path),
            "sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "bytes": len(payload_bytes),
        }

    prompt_manifest = write_once(
        artifact_root / "prompts.jsonl",
        b'{"id":"source:row-0","prompt":"unit test prompt"}\n',
    )
    reward_config = canonical_file(
        artifact_root / "reward.json",
        {"schema": "repldm.reward_config.v1", "model": "unit-test-reward"},
    )
    reward_asset = write_once(artifact_root / "reward.bin", b"reward-weights")
    reward_assets = canonical_file(
        artifact_root / "reward-assets.json",
        {"schema": "repldm.file_manifest.v1", "files": [payload_entry(reward_asset)]},
    )
    model_asset = write_once(artifact_root / "model.bin", b"model-weights")
    model_assets = canonical_file(
        artifact_root / "model-assets.json",
        {"schema": "repldm.file_manifest.v1", "files": [payload_entry(model_asset)]},
    )
    provider = canonical_file(
        artifact_root / "basis-provider.json",
        {"basis_provider_contract_hash": provider_hash},
    )
    calibration = canonical_file(
        artifact_root / "calibration.json",
        {"calibration_hash": calibration_hash},
    )
    frame = canonical_file(
        artifact_root / "renderer-frame.json",
        {
            "renderer_frame_contract_hash": frame_hash,
            "action_contract_hash": action_hash,
        },
    )
    initial_checkpoint = write_once(
        artifact_root / "initial-renderer.pt", b"unit-test-renderer-checkpoint"
    )
    initial_state = canonical_file(
        artifact_root / "initial-renderer.json",
        {
            "schema": "repldm.renderer_initial_state.v1",
            "renderer_state_sha256": initial_state_hash,
            "checkpoint": payload_entry(initial_checkpoint),
        },
    )
    artifacts = {
        "data_manifest": descriptor(authorization.selected_manifest_path),
        "prompt_manifest": descriptor(prompt_manifest),
        "reward_config": descriptor(reward_config),
        "reward_assets_manifest": descriptor(reward_assets),
        "model_assets_manifest": descriptor(model_assets),
        "basis_provider_config": descriptor(provider),
        "calibration": descriptor(calibration),
        "renderer_frame_contract": descriptor(frame),
        "initial_renderer_state": descriptor(initial_state),
    }
    witness_config = None
    witness_assets = None
    witness_preprocess_hash = None
    if method == "f0":
        from latent_renderer_training.runtime_factory import WITNESS_RUNTIME_SCHEMA
        from latent_renderer_training.witnesses import (
            TOPIQ_NR_MODEL_ID,
            TOPIQ_NR_PREPROCESS_SHA256,
            TOPIQ_NR_ROLE,
        )

        topiq_checkpoint = write_once(
            artifact_root / "topiq_nr.pth", b"unit-test-topiq-checkpoint"
        )
        topiq_backbone = write_once(
            artifact_root / "resnet50.safetensors", b"unit-test-resnet-backbone"
        )
        pyiqa_root = artifact_root / "pyiqa-source"
        timm_root = artifact_root / "timm-source"
        pyiqa_root.mkdir(exist_ok=True)
        timm_root.mkdir(exist_ok=True)
        pyiqa_source = write_once(
            pyiqa_root / "__init__.py", b"# pinned unit-test pyiqa source\n"
        )
        timm_source = write_once(
            timm_root / "__init__.py", b"# pinned unit-test timm source\n"
        )
        witness_assets = canonical_file(
            artifact_root / "witness-assets.json",
            {
                "schema": "repldm.file_manifest.v1",
                "files": [
                    payload_entry(topiq_checkpoint),
                    payload_entry(topiq_backbone),
                    payload_entry(pyiqa_source),
                    payload_entry(timm_source),
                ],
            },
        )
        witness_config = canonical_file(
            artifact_root / "witness.json",
            {
                "schema": WITNESS_RUNTIME_SCHEMA,
                "implementation": TOPIQ_NR_MODEL_ID,
                "role": TOPIQ_NR_ROLE,
                "dtype": "float32",
                "preprocess_sha256": TOPIQ_NR_PREPROCESS_SHA256,
                "checkpoint": str(topiq_checkpoint),
                "backbone": str(topiq_backbone),
                "package_versions": {
                    "pyiqa": "unit-test",
                    "timm": "unit-test",
                    "torch": "unit-test",
                    "torchvision": "unit-test",
                    "safetensors": "unit-test",
                },
                "source_packages": {
                    "pyiqa": {"root": str(pyiqa_root), "files": ["__init__.py"]},
                    "timm": {"root": str(timm_root), "files": ["__init__.py"]},
                },
            },
        )
        artifacts.update(
            {
                "witness_config": descriptor(witness_config),
                "witness_assets_manifest": descriptor(witness_assets),
            }
        )
        witness_preprocess_hash = TOPIQ_NR_PREPROCESS_SHA256
    payload: dict[str, object] = {
        "schema": RUN_CONTRACT_SCHEMA,
        "run_id": "unit-test-run",
        "method": method,
        "code_commit": authorization.code_commit,
        "catalog_release_id": authorization.catalog_release_id,
        "catalog_manifest_sha256": authorization.catalog_manifest_sha256,
        "selected_view_release_id": authorization.selected_view_release_id,
        "selected_view_manifest_sha256": authorization.selected_view_manifest_sha256,
        "selected_view_config_sha256": authorization.selected_config_sha256,
        "selected_view_id": authorization.selected_view_id,
        "selected_payload_manifest_sha256": authorization.selected_manifest_sha256,
        "selected_rows": authorization.selected_rows,
        "selected_splits": {"train": 64, "validation": 32},
        "data_manifest_sha256": authorization.selected_manifest_sha256,
        "prompt_manifest_sha256": descriptor(prompt_manifest)["sha256"],
        "renderer_frame_contract_hash": frame_hash,
        "renderer_frame_contract_artifact_sha256": descriptor(frame)["sha256"],
        "calibration_hash": calibration_hash,
        "calibration_artifact_sha256": descriptor(calibration)["sha256"],
        "action_contract_hash": action_hash,
        "basis_provider_contract_hash": provider_hash,
        "basis_provider_config_sha256": descriptor(provider)["sha256"],
        "reward_config_sha256": descriptor(reward_config)["sha256"],
        "reward_preprocess_sha256": "a" * 64,
        "reward_asset_manifest_sha256": descriptor(reward_assets)["sha256"],
        "model_asset_manifest_sha256": descriptor(model_assets)["sha256"],
        "initial_renderer_state_sha256": initial_state_hash,
        "initial_renderer_state_manifest_sha256": descriptor(initial_state)["sha256"],
        "artifacts": artifacts,
        "query_budget": {"reward_forward": 1, "optimizer_step": 64}
        if query_budget is None
        else query_budget,
        "nfe": 50,
        "scheduler_config_sha256": "4" * 64,
        "scheduler_schedule_sha256": "5" * 64,
        "prediction_type": "epsilon",
        "do_classifier_free_guidance": True,
        "guidance_scale": 7.5,
        "guidance_rescale": 0.0,
        "decision_indices": [8, 24, 40],
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 1e-4,
            "betas": [0.9, 0.999],
            "weight_decay": 0.01,
            "gradient_norm_cap": 1.0,
            "ema_decay": 0.995,
        },
        "method_hyperparameters": {
            "method": str(overrides.get("method", "rl")),
            "rounds": 2,
            "updates_per_round": 32,
        },
        "runtime": {
            "device": "cuda:4",
            "batch_size": 1,
            "model_dtype": "float16",
            "vae_dtype": "float32",
            "reward_dtype": "float32",
            "vae_scaling_factor": 0.13025,
            "height": 1024,
            "width": 1024,
        },
        "paths": {
            "run_dir": str((artifact_root / "run").resolve()),
            "output_dir": str((artifact_root / "run" / "outputs").resolve()),
            "checkpoint_dir": str((artifact_root / "run" / "checkpoints").resolve()),
            "ledger_path": str((artifact_root / "run" / "ledger.jsonl").resolve()),
            "rollout_dir": str((artifact_root / "run" / "rollouts").resolve()),
        },
        "seed": 0,
    }
    if method == "f0":
        assert witness_config is not None
        assert witness_assets is not None
        assert witness_preprocess_hash is not None
        payload.update(
            {
                "witness_config_sha256": descriptor(witness_config)["sha256"],
                "witness_preprocess_sha256": witness_preprocess_hash,
                "witness_asset_manifest_sha256": descriptor(witness_assets)["sha256"],
            }
        )
    payload.update(overrides)
    if "method" in overrides and "method_hyperparameters" not in overrides:
        payload["method_hyperparameters"]["method"] = str(overrides["method"])

    if payload["method"] in {"opd", "search_distill", "dpo", "rl"}:
        reward_preprocess_hash = str(payload["reward_preprocess_sha256"])
        anchors = [
            {
                "prompt_id": f"train-{prompt_index:02d}",
                "generation_seed": generation_seed,
                "image_sha256": hashlib.sha256(
                    f"anchor:{prompt_index}:{generation_seed}".encode("ascii")
                ).hexdigest(),
                "reward": float(prompt_index) / 64.0,
            }
            for prompt_index in range(64)
            for generation_seed in (2026090101, 2026090102)
        ]
        reward_statistics = canonical_file(
            artifact_root / "reward-statistics.json",
            {
                "schema": "repldm.renderer_reward_statistics.v1",
                "status": "frozen",
                "anchor_count": 128,
                "estimator": "median_iqr_over_1.349_floor_1e-6",
                "location": 0.5,
                "scale": 0.25,
                "initial_renderer_state_sha256": payload[
                    "initial_renderer_state_sha256"
                ],
                "reward_config_sha256": payload["reward_config_sha256"],
                "reward_preprocess_sha256": reward_preprocess_hash,
                "selected_view_release_id": payload["selected_view_release_id"],
                "anchors": anchors,
            },
        )
        teacher_checkpoint = write_once(
            artifact_root / "opsd-teacher.pt", b"unit-test-opsd-teacher"
        )
        teacher_renderer_hash = "d" * 64
        f0_contract_hash = "e" * 64
        teacher_state = canonical_file(
            artifact_root / "opsd-teacher.json",
            {
                "schema": "repldm.renderer_opsd_teacher_state.v1",
                "status": "frozen",
                "role": "T_OPSD",
                "f0_run_contract_sha256": f0_contract_hash,
                "renderer_state_sha256": teacher_renderer_hash,
                "checkpoint": payload_entry(teacher_checkpoint),
            },
        )
        payload.update(
            {
                "cohort_id": "unit-test-primary-cohort",
                "opsd_teacher_state_manifest_sha256": descriptor(teacher_state)[
                    "sha256"
                ],
                "opsd_teacher_renderer_sha256": teacher_renderer_hash,
                "reward_statistics_sha256": descriptor(reward_statistics)["sha256"],
            }
        )
        registration_binding = write_f0_screen_registration_evidence(
            artifact_root / "f0-screen-registration.json",
            build_f0_screen_registration(
                f0_run_contract_sha256=f0_contract_hash
            ),
        )
        crossfit_renderer_hashes = {
            str(fold): hashlib.sha256(
                f"unit-test-crossfit:{artifact_key}:{fold}".encode("ascii")
            ).hexdigest()
            for fold in range(4)
        }
        evidence_row_kwargs = {
            "initial_renderer_state_sha256": payload[
                "initial_renderer_state_sha256"
            ],
            "opsd_teacher_renderer_sha256": teacher_renderer_hash,
            "crossfit_renderer_state_sha256": crossfit_renderer_hashes,
        }
        train_rows = _passing_f0_rows("train", **evidence_row_kwargs)
        validation_rows = _passing_f0_rows("validation", **evidence_row_kwargs)
        train_metrics, train_summary = write_f0_phase_evidence(
            artifact_root / "f0-train-metrics.jsonl",
            train_rows,
            phase="train",
            reward_scale=0.25,
            reward_statistics_sha256=str(payload["reward_statistics_sha256"]),
            f0_run_contract_sha256=f0_contract_hash,
            screen_registration_sha256=str(registration_binding["sha256"]),
        )
        validation_metrics, validation_summary = write_f0_phase_evidence(
            artifact_root / "f0-validation-metrics.jsonl",
            validation_rows,
            phase="validation",
            reward_scale=0.25,
            reward_statistics_sha256=str(payload["reward_statistics_sha256"]),
            f0_run_contract_sha256=f0_contract_hash,
            screen_registration_sha256=str(registration_binding["sha256"]),
        )
        f0_evidence_shared = {
            "code_commit": payload["code_commit"],
            "selected_payload_manifest_sha256": payload[
                "selected_payload_manifest_sha256"
            ],
            "renderer_frame_contract_hash": payload[
                "renderer_frame_contract_hash"
            ],
            "calibration_hash": payload["calibration_hash"],
            "initial_renderer_state_sha256": payload[
                "initial_renderer_state_sha256"
            ],
            # The shared F0 ledger builder is also used while constructing
            # non-F0 contracts for rejection tests.  Keep the synthetic
            # validation realization bound to the same teacher hash that the
            # generated evidence rows and gate already carry.
            "opsd_teacher_renderer_sha256": teacher_renderer_hash,
            "crossfit_renderer_state_sha256": crossfit_renderer_hashes,
            "reward_config_sha256": payload["reward_config_sha256"],
            "reward_preprocess_sha256": reward_preprocess_hash,
            "reward_asset_manifest_sha256": payload[
                "reward_asset_manifest_sha256"
            ],
            "witness_config_sha256": hashlib.sha256(
                b"unit-test-topiq-config"
            ).hexdigest(),
            "witness_preprocess_sha256": _TOPIQ_NR_PREPROCESS_SHA256,
            "witness_asset_manifest_sha256": hashlib.sha256(
                b"unit-test-topiq-assets"
            ).hexdigest(),
        }
        ledger_cache_key = tuple(str(value) for value in f0_evidence_shared.values())
        if ledger_cache_key not in _F0_LEDGER_CACHE:
            _F0_LEDGER_CACHE[ledger_cache_key] = _sealed_metric_ledger(
                [*train_rows, *validation_rows], f0_evidence_shared
            )
        ledger_payload, seal_payload = _F0_LEDGER_CACHE[ledger_cache_key]
        f0_ledger = write_once(artifact_root / "f0-ledger.jsonl", ledger_payload)
        f0_seal = write_once(
            artifact_root / "f0-ledger.jsonl.seal", seal_payload
        )
        f0_gate = canonical_file(
            artifact_root / "f0-gate.json",
            {
                "schema": "repldm.renderer_f0_gate.v5",
                "status": "passed",
                "code_commit": payload["code_commit"],
                "f0_run_contract_sha256": f0_contract_hash,
                "selected_view_release_id": payload["selected_view_release_id"],
                "selected_view_manifest_sha256": payload[
                    "selected_view_manifest_sha256"
                ],
                "selected_payload_sha256": payload[
                    "selected_payload_manifest_sha256"
                ],
                "renderer_frame_contract_hash": payload[
                    "renderer_frame_contract_hash"
                ],
                "calibration_hash": payload["calibration_hash"],
                "action_contract_hash": payload["action_contract_hash"],
                "initial_renderer_state_sha256": payload[
                    "initial_renderer_state_sha256"
                ],
                "reward_config_sha256": payload["reward_config_sha256"],
                "reward_preprocess_sha256": reward_preprocess_hash,
                "reward_asset_manifest_sha256": payload[
                    "reward_asset_manifest_sha256"
                ],
                "witness_config_sha256": f0_evidence_shared[
                    "witness_config_sha256"
                ],
                "witness_preprocess_sha256": f0_evidence_shared[
                    "witness_preprocess_sha256"
                ],
                "witness_asset_manifest_sha256": f0_evidence_shared[
                    "witness_asset_manifest_sha256"
                ],
                "reward_statistics_sha256": payload["reward_statistics_sha256"],
                "opsd_teacher_state_manifest_sha256": payload[
                    "opsd_teacher_state_manifest_sha256"
                ],
                "opsd_teacher_renderer_sha256": teacher_renderer_hash,
                "crossfit_renderer_state_sha256": crossfit_renderer_hashes,
                "train_gate": {
                    "passed": train_summary["passed"],
                    "prompt_count": train_summary["prompt_count"],
                    "state_count": train_summary["state_count"],
                    "safety_violations": train_summary["safety_violations"],
                    "metrics": train_metrics,
                },
                "validation_gate": {
                    "passed": validation_summary["passed"],
                    "prompt_count": validation_summary["prompt_count"],
                    "state_count": validation_summary["state_count"],
                    "safety_violations": validation_summary["safety_violations"],
                    "metrics": validation_metrics,
                },
                "screen_registration": registration_binding,
                "ledger": payload_entry(f0_ledger),
                "ledger_seal": payload_entry(f0_seal),
            },
        )
        payload["f0_gate_sha256"] = descriptor(f0_gate)["sha256"]
        shared_fields = (
            "code_commit", "catalog_release_id", "catalog_manifest_sha256",
            "selected_view_release_id", "selected_view_manifest_sha256",
            "selected_view_config_sha256", "selected_view_id",
            "selected_payload_manifest_sha256", "data_manifest_sha256",
            "prompt_manifest_sha256", "renderer_frame_contract_hash",
            "calibration_hash", "action_contract_hash",
            "basis_provider_contract_hash", "reward_config_sha256",
            "reward_preprocess_sha256", "model_asset_manifest_sha256",
            "initial_renderer_state_sha256", "f0_gate_sha256",
            "opsd_teacher_state_manifest_sha256", "opsd_teacher_renderer_sha256",
            "reward_statistics_sha256", "nfe", "scheduler_config_sha256",
            "scheduler_schedule_sha256", "prediction_type",
            "do_classifier_free_guidance", "guidance_scale", "guidance_rescale",
            "decision_indices",
        )
        runtime_fields = (
            "batch_size", "model_dtype", "vae_dtype", "reward_dtype",
            "vae_scaling_factor", "height", "width",
        )
        cohort = canonical_file(
            artifact_root / "cohort.json",
            {
                "schema": "repldm.renderer_training_cohort.v1",
                "status": "registered",
                "cohort_id": payload["cohort_id"],
                "methods": ["opd", "search_distill", "dpo", "rl"],
                "optimization_seeds": [202609011, 202609012, 202609013],
                "shared": {field: payload[field] for field in shared_fields},
                "runtime_invariants": {
                    field: payload["runtime"][field] for field in runtime_fields
                },
            },
        )
        payload["cohort_manifest_sha256"] = descriptor(cohort)["sha256"]
        artifacts.update(
            {
                "f0_gate": descriptor(f0_gate),
                "opsd_teacher_state": descriptor(teacher_state),
                "reward_statistics": descriptor(reward_statistics),
                "cohort_manifest": descriptor(cohort),
            }
        )
    payload.update(overrides)
    if "method" in overrides and "method_hyperparameters" not in overrides:
        payload["method_hyperparameters"]["method"] = str(overrides["method"])
    return payload


def _operation_executor(contract, binding) -> LedgeredOperationExecutor:
    ledger = QueryLedger(
        contract["paths"]["ledger_path"],
        contract["query_budget"],
        run_contract=contract,
        authorization_binding=binding,
    )
    return LedgeredOperationExecutor(ledger, authorization_binding=binding)


def _optimizer_context(trainer: RendererTrainer):
    return trainer.optimizer_context(
        split="train",
        batch_id=f"optimizer-batch-{trainer.step}",
        action={"prompt_ids": ["source:row-0"]},
    )


def test_action_contract_has_fixed_measure_and_rejects_inactive_values():
    contract = ActionSpaceContract(4, (True, False, True, False))
    assert contract.rank == 2
    assert contract_hash(contract) == contract_hash(contract.to_dict())
    contract.validate_action(torch.tensor([[0.1, 0.0, -0.2, 0.0]]))
    with pytest.raises(ValueError, match="inactive"):
        contract.validate_action(torch.tensor([[0.1, 0.01, -0.2, 0.0]]))
    with pytest.raises(ValueError, match="inactive"):
        contract.validate_action(torch.tensor([[0.1, 1e-9, -0.2, 0.0]]))


def test_squashed_gaussian_round_trip_and_inactive_dirac():
    contract = ActionSpaceContract(3, (True, False, True), pre_squash_sigma=0.25)
    mean = torch.tensor([[0.2, 9.0, -0.1]], requires_grad=True)
    distribution = SquashedGaussian(mean, contract)
    action, pre_squash = distribution.rsample(torch.zeros_like(mean))
    assert action[0, 1].item() == 0.0
    selected = pre_squash[..., torch.tensor(contract.active_mask, dtype=torch.bool)]
    jacobian = torch.log(torch.tensor(contract.coefficient_bound)) + torch.log1p(-torch.tanh(selected).square())
    torch.testing.assert_close(distribution.log_prob(action), distribution.log_prob_u(pre_squash) - jacobian.sum(dim=-1))
    distribution.log_prob(action).sum().backward()
    assert torch.isfinite(mean.grad).all()
    with pytest.raises(ValueError, match="inactive pre_squash"):
        distribution.log_prob_u(torch.tensor([[0.0, 99.0, 0.0]]))


def test_squashed_gaussian_retains_pre_squash_calculation_precision():
    contract = ActionSpaceContract(2, (True, True))
    mean = torch.zeros(1, 2, dtype=torch.float16)
    action, pre_squash = SquashedGaussian(mean, contract).rsample(
        torch.tensor([[0.1234, -0.5678]], dtype=torch.float16)
    )
    assert action.dtype == torch.float16
    assert pre_squash.dtype == torch.float32
    expected = mean.float() + contract.pre_squash_sigma * torch.tensor(
        [[0.1234, -0.5678]], dtype=torch.float16
    ).float()
    torch.testing.assert_close(pre_squash, expected)


def test_transformed_gaussian_kl_ignores_inactive_slots():
    contract = ActionSpaceContract(3, (True, False, True))
    first = torch.tensor([[0.1, 100.0, 0.3]])
    second = torch.tensor([[0.0, -100.0, 0.1]])
    expected = 0.5 * ((torch.tensor([0.1, 0.2]) / 0.25) ** 2).sum()
    torch.testing.assert_close(transformed_gaussian_kl(first, second, contract).squeeze(), expected)
    torch.testing.assert_close(
        transformed_gaussian_kl(first, second, contract, reduction="mean").squeeze(),
        expected / 2,
    )
    with pytest.raises(ValueError, match="finite"):
        transformed_gaussian_kl(torch.tensor([[float("nan"), 0.0, 0.0]]), second, contract)


def test_native_transition_loss_detaches_teacher():
    predicted = torch.zeros(2, 4, requires_grad=True)
    target = torch.ones(2, 4, requires_grad=True)
    nominal = torch.ones(2, 4)
    loss = normalized_transition_loss(predicted, target, nominal)
    loss.backward()
    assert predicted.grad is not None
    assert target.grad is not None


def test_opd_target_is_detached_and_dpo_has_gradient():
    contract = ActionSpaceContract(2, (True, True))
    predicted = torch.zeros(2, 4, requires_grad=True)
    target = torch.ones(2, 4, requires_grad=True)
    nominal = torch.ones(2, 4)
    mean = torch.zeros(2, 2, requires_grad=True)
    reference = torch.zeros(2, 2)
    loss = opd_loss(predicted, target, nominal, mean, reference, contract)
    loss.backward()
    assert target.grad is None
    chosen_mean = torch.zeros(2, 2, requires_grad=True)
    rejected_mean = torch.ones(2, 2)
    actions = torch.tensor([[0.1, -0.1], [0.2, -0.2]])
    dpo = dpo_loss(actions, -actions, chosen_mean, rejected_mean, torch.zeros_like(chosen_mean), torch.zeros_like(rejected_mean), contract)
    dpo.backward()
    assert torch.isfinite(chosen_mean.grad).all()


def test_search_distill_weights_are_applied_per_sample():
    contract = ActionSpaceContract(2, (True, True))
    predicted = torch.zeros(2, 2, requires_grad=True)
    target = torch.tensor([[1.0, 0.0], [10.0, 0.0]])
    nominal = torch.ones(2, 2)
    mean = torch.zeros(2, 2, requires_grad=True)
    reference = torch.zeros(2, 2)
    per_sample = normalized_transition_losses(predicted, target, nominal)
    loss = search_distill_loss(
        predicted, target, nominal, mean, reference, contract,
        chosen_weight=torch.tensor([1.0, 0.0]),
    )
    expected = per_sample[0] + 0.10 * (mean.square().mean())
    torch.testing.assert_close(loss, expected)


def test_dpo_uses_one_trajectory_margin_and_freezes_reference():
    contract = ActionSpaceContract(2, (True, True))
    chosen_mean = torch.zeros(1, 2, 2, requires_grad=True)
    rejected_mean = torch.zeros(1, 2, 2, requires_grad=True)
    reference_chosen = torch.zeros(1, 2, 2, requires_grad=True)
    reference_rejected = torch.zeros(1, 2, 2, requires_grad=True)
    chosen = torch.full((1, 2, 2), 0.1)
    rejected = torch.full((1, 2, 2), -0.1)
    loss = dpo_loss(
        chosen, rejected, chosen_mean, rejected_mean,
        reference_chosen, reference_rejected, contract,
    )
    loss.backward()
    assert reference_chosen.grad is None
    assert reference_rejected.grad is None
    assert torch.isfinite(chosen_mean.grad).all()
    assert torch.isfinite(rejected_mean.grad).all()
    tied = dpo_loss(
        chosen, rejected, chosen_mean.detach(), rejected_mean.detach(),
        reference_chosen.detach(), reference_rejected.detach(), contract,
        tie_mask=torch.tensor([True]),
    )
    assert tied.item() == 0.0


def test_dpo_flattens_all_decision_axes_into_one_trajectory_margin():
    contract = ActionSpaceContract(2, (True, True))
    chosen_mean = torch.zeros(1, 2, 3, 2, requires_grad=True)
    rejected_mean = torch.zeros(1, 2, 3, 2, requires_grad=True)
    reference_chosen = torch.zeros_like(chosen_mean)
    reference_rejected = torch.zeros_like(rejected_mean)
    chosen = torch.full_like(chosen_mean, 0.1)
    rejected = torch.full_like(rejected_mean, -0.1)
    loss = dpo_loss(
        chosen,
        rejected,
        chosen_mean,
        rejected_mean,
        reference_chosen,
        reference_rejected,
        contract,
    )
    assert loss.ndim == 0
    loss.backward()
    assert torch.isfinite(chosen_mean.grad).all()
    assert torch.isfinite(rejected_mean.grad).all()


def test_dpo_treats_sampled_actions_as_detached_data():
    contract = ActionSpaceContract(2, (True, True))
    chosen = torch.full((1, 2), 0.1, requires_grad=True)
    rejected = torch.full((1, 2), -0.1, requires_grad=True)
    chosen_mean = torch.zeros(1, 2, requires_grad=True)
    rejected_mean = torch.zeros(1, 2, requires_grad=True)
    loss = dpo_loss(
        chosen,
        rejected,
        chosen_mean,
        rejected_mean,
        torch.zeros_like(chosen_mean),
        torch.zeros_like(rejected_mean),
        contract,
    )
    loss.backward()
    assert chosen.grad is None or torch.equal(chosen.grad, torch.zeros_like(chosen))
    assert rejected.grad is None or torch.equal(rejected.grad, torch.zeros_like(rejected))


def test_transformed_density_can_require_and_use_exact_pre_squash_samples():
    contract = ActionSpaceContract(2, (True, True), pre_squash_sigma=0.25)
    mean = torch.tensor([[0.15, -0.2]], requires_grad=True)
    distribution = SquashedGaussian(mean, contract)
    # A large coordinate is deliberately close to the open action boundary;
    # reconstructing it from fp32 ``action`` loses measurable information.
    pre_squash = torch.tensor([[7.25, -0.75]])
    action = contract.coefficient_bound * torch.tanh(pre_squash)
    exact = distribution.log_prob_from_pre_squash(pre_squash)
    reconstructed = distribution.log_prob(action)
    assert torch.isfinite(exact).all()
    assert torch.isfinite(reconstructed).all()
    assert not torch.equal(exact, reconstructed)
    with pytest.raises(ValueError, match="pre_squash"):
        dpo_loss(
            action,
            -action,
            mean,
            mean.detach(),
            torch.zeros_like(mean),
            torch.zeros_like(mean),
            contract,
            require_pre_squash=True,
        )
    loss = dpo_loss(
        action,
        -action,
        mean,
        mean.detach(),
        torch.zeros_like(mean),
        torch.zeros_like(mean),
        contract,
        chosen_pre_squash=pre_squash,
        rejected_pre_squash=-pre_squash,
        require_pre_squash=True,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(mean.grad).all()


def test_dpo_detaches_recorded_pre_squash_samples():
    contract = ActionSpaceContract(2, (True, True))
    mean = torch.zeros(1, 2, requires_grad=True)
    chosen_u = torch.tensor([[0.2, -0.1]], requires_grad=True)
    rejected_u = torch.tensor([[-0.2, 0.1]], requires_grad=True)
    chosen = torch.tanh(chosen_u.detach())
    rejected = torch.tanh(rejected_u.detach())
    loss = dpo_loss(
        chosen,
        rejected,
        mean,
        mean,
        torch.zeros_like(mean),
        torch.zeros_like(mean),
        contract,
        chosen_pre_squash=chosen_u,
        rejected_pre_squash=rejected_u,
        require_pre_squash=True,
    )
    loss.backward()
    assert chosen_u.grad is None
    assert rejected_u.grad is None
    assert mean.grad is not None and torch.isfinite(mean.grad).all()


def test_rl_uses_one_ratio_per_decision():
    contract = ActionSpaceContract(2, (True, True))
    actions = torch.zeros(3, 2)
    behavior = torch.zeros(3, 2)
    policy = torch.tensor([[0.0, 0.0], [0.3, 0.0], [-0.3, 0.0]], requires_grad=True)
    reference = torch.zeros(3, 2)
    advantages = torch.tensor([1.0, -1.0, 0.0])
    loss, ratio, kl = per_decision_rl_loss(actions, behavior, policy, reference, advantages, contract)
    assert ratio.shape == (3,)
    assert kl.shape == (3,)
    loss.backward()
    assert torch.isfinite(policy.grad).all()


def test_rl_masks_nonfinite_density_rows():
    contract = ActionSpaceContract(2, (True, True))
    actions = torch.zeros(2, 2)
    behavior = torch.zeros(2, 2)
    policy = torch.tensor([[0.0, 0.0], [1e20, 0.0]], requires_grad=True)
    reference = torch.zeros(2, 2)
    advantages = torch.ones(2)
    loss, ratio, kl = per_decision_rl_loss(
        actions, behavior, policy, reference, advantages, contract
    )
    assert torch.isfinite(loss)
    assert ratio[1].item() == 0.0
    assert kl[1].item() == 0.0


def test_transition_and_dpo_fail_closed_on_invalid_or_overflow_rows():
    nominal = torch.ones(2, 2)
    losses = normalized_transition_losses(
        torch.tensor([[0.0, 0.0], [1e20, 0.0]]),
        torch.zeros(2, 2), nominal,
    )
    assert losses[0].item() == 0.0
    assert losses[1].item() == 0.0
    contract = ActionSpaceContract(2, (True, True))
    chosen_mean = torch.tensor([[1e20, 0.0]], requires_grad=True)
    rejected_mean = torch.tensor([[-1e20, 0.0]])
    loss = dpo_loss(
        torch.zeros(1, 2), torch.zeros(1, 2), chosen_mean, rejected_mean,
        torch.zeros(1, 2), torch.zeros(1, 2), contract,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(chosen_mean.grad).all()


def test_transition_objectives_drop_near_zero_rows_before_anchor_penalty():
    """A rejected native transition must not receive a reference loss."""
    contract = ActionSpaceContract(2, (True, True))
    predicted = torch.zeros(2, 2, requires_grad=True)
    target = torch.ones(2, 2)
    nominal = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    mean = torch.tensor([[0.0, 0.0], [1.0, 1.0]], requires_grad=True)
    reference = torch.zeros(2, 2)

    opd = opd_loss(predicted, target, nominal, mean, reference, contract)
    distilled = search_distill_loss(
        predicted, target, nominal, mean, reference, contract
    )
    # Only the first row is valid: its normalized transition loss is exactly
    # one and its anchor penalty is zero.  The invalid second row must not
    # change either objective despite its non-zero mean.
    torch.testing.assert_close(opd, torch.tensor(1.0))
    torch.testing.assert_close(distilled, torch.tensor(1.0))
    (opd + distilled).backward()
    assert torch.equal(mean.grad[1], torch.zeros_like(mean.grad[1]))


def test_transition_valid_mask_is_shape_and_dtype_checked():
    nominal = torch.ones(2, 2)
    predicted = torch.zeros(2, 2)
    target = torch.ones(2, 2)
    with pytest.raises(ValueError, match="valid_mask"):
        normalized_transition_losses(
            predicted, target, nominal, valid_mask=torch.ones(2)
        )
    with pytest.raises(ValueError, match="valid_mask"):
        normalized_transition_losses(
            predicted, target, nominal, valid_mask=torch.ones(1, dtype=torch.bool)
        )


def test_rl_invalid_action_is_zero_weight():
    contract = ActionSpaceContract(2, (True, True))
    actions = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    policy = torch.zeros(2, 2, requires_grad=True)
    loss, ratio, kl = per_decision_rl_loss(
        actions, torch.zeros(2, 2), policy, torch.zeros(2, 2),
        torch.ones(2), contract,
    )
    assert torch.isfinite(loss)
    assert ratio[1].item() == 0.0
    assert kl[1].item() == 0.0


def test_rl_masks_nonfinite_reference_and_advantage_rows():
    contract = ActionSpaceContract(2, (True, True))
    actions = torch.zeros(2, 2)
    policy = torch.zeros(2, 2, requires_grad=True)
    reference = torch.tensor([[0.0, 0.0], [float("nan"), 0.0]])
    advantages = torch.tensor([1.0, 3.0])
    loss, ratio, kl = per_decision_rl_loss(
        actions,
        torch.zeros_like(actions),
        policy,
        reference,
        advantages,
        contract,
        kl_weight=0.0,
    )
    assert torch.isfinite(loss)
    assert ratio.tolist() == [1.0, 0.0]
    assert kl.tolist() == [0.0, 0.0]
    torch.testing.assert_close(loss, torch.tensor(-1.0))
    loss.backward()
    assert torch.isfinite(policy.grad).all()


def test_rl_all_invalid_policy_returns_finite_zero_loss():
    contract = ActionSpaceContract(2, (True, True))
    policy = torch.full((2, 2), float("nan"), requires_grad=True)
    loss, ratio, kl = per_decision_rl_loss(
        torch.zeros(2, 2),
        torch.zeros(2, 2),
        policy,
        torch.zeros(2, 2),
        torch.ones(2),
        contract,
    )
    assert loss.item() == 0.0
    assert torch.isfinite(loss)
    loss.backward()
    assert policy.grad is not None and torch.isfinite(policy.grad).all()


def test_shared_prefix_is_replayed_once():
    calls = []

    def step(state, action, index):
        calls.append((state, action, index))
        next_state = state + 1
        if action is None:
            return next_state
        transition = Transition(state, action, next_state, torch.tensor([1.0]), torch.tensor([2.0]), step_index=index)
        return next_state, transition

    trajectories = replay_shared_prefix(0, 2, step, {"plus": [torch.tensor([1.0])], "minus": [torch.tensor([-1.0])]})
    assert len([call for call in calls if call[1] is None]) == 2
    assert [item.step_index for item in trajectories["plus"].transitions] == [2]
    assert [item.step_index for item in trajectories["minus"].transitions] == [2]


def test_shared_prefix_preserves_tuple_valued_state_in_prefix_and_branches():
    calls = []

    def step(state, action, index):
        calls.append((state, action, index))
        return (state[0] + 1, state[1] + 1)

    replay_shared_prefix(
        (0, 10),
        1,
        step,
        {"left": [None], "right": [None]},
    )

    assert calls == [
        ((0, 10), None, 0),
        ((1, 11), None, 1),
        ((1, 11), None, 1),
    ]


def test_shared_prefix_branches_do_not_alias_mutable_state():
    starts = []

    def step(state, action, index):
        if action is None:
            state["value"] += 1
            return state
        starts.append(state["value"])
        state["value"] += int(action.item())
        transition = Transition(state["value"], action, state["value"], torch.ones(1), torch.ones(1), step_index=index)
        return state, transition

    replay_shared_prefix(
        {"value": 0}, 1, step,
        {"plus": [torch.tensor([1])], "minus": [torch.tensor([0])]}
    )
    assert starts == [1, 1]


def test_shared_prefix_requires_clone_hook_for_uncloneable_state():
    class Uncloneable:
        def __deepcopy__(self, memo):
            raise RuntimeError("no copy")

    with pytest.raises(TypeError, match="state_clone_fn"):
        replay_shared_prefix(Uncloneable(), 0, lambda state, action, index: state, {"a": []})


def test_generic_antithetic_collector_preserves_pre_squash_coordinates():
    def step(state, action, index):
        next_state = state + 1
        if action is None:
            return next_state
        transition = Transition(
            state,
            action,
            next_state,
            torch.ones(1),
            torch.ones(1),
            step_index=index,
        )
        return next_state, transition

    def sample(mean, noise):
        pre_squash = mean + noise
        return torch.tanh(pre_squash), pre_squash, -pre_squash.square().sum(dim=-1)

    collection = collect_antithetic_rollout(
        0.0,
        decision_indices=(1,),
        total_steps=2,
        step_fn=step,
        mean_fn=lambda _state, _index: torch.zeros(1, 2),
        action_from_mean=sample,
        noise_by_decision={1: torch.tensor([[0.75, -0.5]])},
        registered_decision_indices=(1,),
        require_pre_squash=True,
    )
    proposal = collection.proposals[0]
    torch.testing.assert_close(proposal.plus_pre_squash, torch.tensor([[0.75, -0.5]]))
    torch.testing.assert_close(proposal.minus_pre_squash, torch.tensor([[-0.75, 0.5]]))
    torch.testing.assert_close(
        collection.branches["plus"].transitions[0].pre_squash,
        proposal.plus_pre_squash,
    )


def test_query_ledger_conservatively_charges_unfinished_reservation(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    ledger = QueryLedger(path, {"reward_forward": 2}, run_contract="contract-a")
    reservation = ledger.reserve(
        "reward_forward", 1, metadata=_ledger_metadata()
    )
    assert ledger.remaining("reward_forward") == 1
    assert ledger.summary()["unfinished_reservations"] == 1
    with pytest.raises(ValueError, match="does not match"):
        ledger.receipt(
            type(reservation)(reservation.reservation_id, "reward_forward", 99, reservation.metadata),
            result=_ledger_result(), success=False,
        )
    ledger.receipt(reservation, result=_ledger_result(), success=True)
    assert ledger.summary()["unfinished_reservations"] == 0
    with pytest.raises(RuntimeError, match="budget exceeded"):
        ledger.reserve("reward_forward", 2)
    assert len(path.read_text().splitlines()) == 2
    assert all(json.loads(line)["run_contract"] == "contract-a" for line in path.read_text().splitlines())


def test_query_ledger_rejects_tampering(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    ledger = QueryLedger(path, {"reward_forward": 2}, run_contract="contract-a")
    ledger.reserve("reward_forward", 1, metadata=_ledger_metadata())
    row = json.loads(path.read_text())
    row["amount"] = 2
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(RuntimeError, match="hash"):
        ledger.remaining("reward_forward")


def test_query_ledger_requires_provenance_and_detects_deleted_pair(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    ledger = QueryLedger(path, {"reward_forward": 4}, run_contract="contract-a")
    with pytest.raises(ValueError, match="provenance"):
        ledger.reserve("reward_forward", 1, metadata={"prompt": "p"})
    first = ledger.reserve("reward_forward", 1, metadata=_ledger_metadata())
    ledger.receipt(first, result=_ledger_result(), success=True)
    second = ledger.reserve(
        "reward_forward", 1,
        metadata={**_ledger_metadata(), "prompt": "prompt-1", "step": 1},
    )
    ledger.receipt(second, result=_ledger_result(), success=True)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[2:]) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="seal|sequence|chain"):
        ledger.summary()


def test_query_ledger_binds_budget_and_strictness_on_reopen(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    ledger = QueryLedger(path, {"reward_forward": 1}, run_contract="contract-a")
    ledger.reserve("reward_forward", 1, metadata=_ledger_metadata())
    with pytest.raises(RuntimeError, match="contract|budget"):
        QueryLedger(path, {"reward_forward": 2}, run_contract="contract-a").summary()
    with pytest.raises(RuntimeError, match="contract|strict"):
        QueryLedger(path, {"reward_forward": 1}, run_contract="contract-a", strict_provenance=False).summary()


def test_query_ledger_rejects_deleted_primary_file_with_surviving_seal(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    ledger = QueryLedger(path, {"reward_forward": 2}, run_contract="contract-a")
    ledger.reserve("reward_forward", 1, metadata=_ledger_metadata())
    path.unlink()
    with pytest.raises(RuntimeError, match="seal|ledger"):
        ledger.summary()


def test_query_ledger_recovers_after_append_before_seal_crash(tmp_path: Path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    ledger = QueryLedger(path, {"reward_forward": 2}, run_contract="contract-a")
    original_replace = Path.replace
    failed = {"value": False}

    def fail_seal_replace(source, destination):
        if destination == ledger.seal_path and not failed["value"]:
            failed["value"] = True
            raise OSError("simulated crash before seal publication")
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", fail_seal_replace)
    with pytest.raises(OSError, match="simulated"):
        ledger.reserve("reward_forward", 1, metadata=_ledger_metadata())
    monkeypatch.setattr(Path, "replace", original_replace)
    summary = ledger.summary()
    assert summary["reserved"]["reward_forward"] == 1
    assert summary["remaining"]["reward_forward"] == 1


def test_query_ledger_computes_and_checks_result_hash(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    ledger = QueryLedger(path, {"reward_forward": 1}, run_contract="contract-a")
    reservation = ledger.reserve("reward_forward", 1, metadata=_ledger_metadata())
    result = _ledger_result()
    ledger.receipt(reservation, result=result, success=True)
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[1])
    assert len(row["result"]["result_hash"]) == 64
    assert row["result"]["result_hash"] == hashlib.sha256(
        json.dumps(
            {key: value for key, value in row["result"].items() if key != "result_hash"},
            sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ).encode()
    ).hexdigest()


def test_query_ledger_public_verified_snapshots_are_defensive(tmp_path: Path):
    ledger = QueryLedger(
        tmp_path / "public-ledger.jsonl",
        {"reward_forward": 1},
        run_contract="contract-a",
    )
    reservation = ledger.reserve(
        "reward_forward", 1, metadata=_ledger_metadata()
    )
    ledger.receipt(reservation, result=_ledger_result(), success=True)

    records = ledger.verified_records()
    pairs = ledger.successful_receipt_pairs("reward_forward")
    records[0]["metadata"]["prompt"] = "mutated-copy"
    pairs[0][1]["result"]["scalar_or_gradient"] = "mutated-copy"

    fresh = ledger.verified_records()
    fresh_pairs = ledger.successful_receipt_pairs("reward_forward")
    assert fresh[0]["metadata"]["prompt"] == "prompt-0"
    assert fresh_pairs[0][1]["result"]["scalar_or_gradient"] == "scalar"


def test_bound_ledger_requires_matching_provenance_and_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    authorization = _training_authorization(tmp_path, monkeypatch)
    contract = _complete_run_contract(
        authorization, query_budget={"reward_forward": 1}
    )
    binding = authorization.bind_run_contract(contract)
    bound_meta = {
        **_ledger_metadata(),
        "run_contract_hash": binding.contract_hash,
        "renderer_frame_contract_hash": contract["renderer_frame_contract_hash"],
        "calibration_hash": contract["calibration_hash"],
        "data_manifest_sha256": contract["data_manifest_sha256"],
        "reward_config_sha256": contract["reward_config_sha256"],
    }
    bound_result = {
        **_ledger_result(),
        "run_contract_hash": binding.contract_hash,
        "renderer_frame_contract_hash": contract["renderer_frame_contract_hash"],
        "calibration_hash": contract["calibration_hash"],
        "data_manifest_sha256": contract["data_manifest_sha256"],
        "reward_config_sha256": contract["reward_config_sha256"],
    }
    ledger = QueryLedger(
        tmp_path / "bound.jsonl",
        {"reward_forward": 1},
        run_contract=contract,
        authorization_binding=binding,
    )
    reservation = ledger.reserve("reward_forward", 1, metadata=bound_meta)
    ledger.receipt(reservation, result=bound_result, success=True)
    assert ledger.summary()["completed_receipts"] == 1
    with pytest.raises(ValueError, match="budget"):
        QueryLedger(
            tmp_path / "other.jsonl",
            {"reward_forward": 2},
            run_contract=contract,
            authorization_binding=binding,
        )
    with pytest.raises(TypeError, match="AuthorizationBinding"):
        QueryLedger(
            tmp_path / "forged.jsonl",
            {"reward_forward": 1},
            run_contract=contract,
            authorization_binding=copy.copy(binding),
        )


def test_training_probe_is_read_only_for_missing_ledger(tmp_path: Path, capsys):
    ledger_path = tmp_path / "nested" / "probe.jsonl"
    assert training_probe(["--ledger", str(ledger_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["ledger"] == {
        "path": str(ledger_path),
        "exists": False,
        "read_only": True,
    }
    assert not ledger_path.exists()
    assert not ledger_path.parent.exists()


def test_training_launcher_binds_the_only_config_payload_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    authorization = _training_authorization(tmp_path, monkeypatch)
    contract = _complete_run_contract(authorization)
    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {"schema": TRAINING_CONFIG_SCHEMA, "run_contract": contract},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    calls = []

    def execute(binding):
        calls.append(binding)
        return {
            "schema": METHOD_RESULT_SCHEMA,
            "run_id": binding.contract["run_id"],
            "method": binding.contract["method"],
            "run_contract_sha256": binding.contract_hash,
            "status": "training_complete",
            "benchmark_status": "pending",
        }

    monkeypatch.setattr(launcher_module, "dispatch_training", execute)

    result = launch_training(
        receipt_path=authorization.receipt_path,
        config_path=config_path,
        repository_root=authorization.repository_root,
    )
    assert result.executed is True
    assert result.validation_only is False
    assert result.value["status"] == "training_complete"
    assert result.config_sha256 == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert len(calls) == 1
    assert calls[0].contract_hash == result.run_contract_sha256


def test_training_launcher_validation_only_is_explicit_and_never_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    authorization = _training_authorization(tmp_path, monkeypatch)
    contract = _complete_run_contract(authorization)
    config_path = tmp_path / "validate.json"
    config_path.write_text(
        json.dumps({"schema": TRAINING_CONFIG_SCHEMA, "run_contract": contract}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        launcher_module,
        "dispatch_training",
        lambda _binding: (_ for _ in ()).throw(AssertionError("must not dispatch")),
    )

    result = launch_training(
        receipt_path=authorization.receipt_path,
        config_path=config_path,
        repository_root=authorization.repository_root,
        validation_only=True,
    )
    assert result.executed is False
    assert result.validation_only is True
    assert result.value is None


def test_training_config_rejects_unbound_top_level_settings_and_bad_yaml(
    tmp_path: Path,
):
    extra = tmp_path / "extra.json"
    extra.write_text(
        json.dumps(
            {
                "schema": TRAINING_CONFIG_SCHEMA,
                "run_contract": {},
                "learning_rate": 1e-4,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only schema and run_contract"):
        _load_config(extra)
    malformed = tmp_path / "bad.yaml"
    malformed.write_text("schema: [unterminated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        _load_config(malformed)


def test_run_contract_rejects_registered_trajectory_optimizer_and_path_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    authorization = _training_authorization(tmp_path, monkeypatch)
    baseline = _complete_run_contract(authorization)
    cases = []

    changed = copy.deepcopy(baseline)
    changed["nfe"] = 49
    cases.append((changed, "50-step"))
    changed = copy.deepcopy(baseline)
    changed["decision_indices"] = [8, 23, 40]
    cases.append((changed, "decision_indices"))
    changed = copy.deepcopy(baseline)
    changed["guidance_scale"] = 7.0
    cases.append((changed, "CFG settings"))
    changed = copy.deepcopy(baseline)
    changed["optimizer"]["learning_rate"] = 2e-4
    cases.append((changed, "optimizer settings"))
    changed = copy.deepcopy(baseline)
    changed["method_hyperparameters"]["method"] = "dpo"
    cases.append((changed, "method_hyperparameters"))
    changed = copy.deepcopy(baseline)
    changed["paths"]["ledger_path"] = str((tmp_path / "outside.jsonl").resolve())
    cases.append((changed, "inside paths.run_dir"))

    for payload, message in cases:
        with pytest.raises(ValueError, match=message):
            TrainingRunContract.from_mapping(payload)


def test_run_updates_rejects_before_obtaining_a_side_effecting_iterator():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = RendererTrainer(model, optimizer, contract={"schema": "test"})
    observed = []

    class SideEffectingIterable:
        def __iter__(self):
            observed.append("iterated")
            yield lambda: model(torch.ones(1, 1)).square().sum()

    with pytest.raises(RuntimeError, match="TrainingAuthorization"):
        run_updates(trainer, SideEffectingIterable())
    assert observed == []


def test_checkpoint_round_trip_rejects_contract_drift(tmp_path: Path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    contract = {"mask_hash": "abc", "rank": 2}
    path = tmp_path / "checkpoint.pt"
    info = save_checkpoint(path, model=model, optimizer=optimizer, step=3, contract=contract)
    assert info["sha256"]
    restored = torch.nn.Linear(2, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    payload = load_checkpoint(path, model=restored, optimizer=restored_optimizer, expected_contract=contract)
    assert payload["step"] == 3
    with pytest.raises(ValueError, match="does not match"):
        load_checkpoint(path, model=restored, expected_contract={"mask_hash": "different"})


def test_formal_checkpoint_binds_authorization_and_run_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    authorization = _training_authorization(tmp_path, monkeypatch)
    contract = _complete_run_contract(authorization)
    binding = authorization.bind_run_contract(contract)
    path = tmp_path / "formal.pt"
    model = torch.nn.Linear(2, 2)
    save_checkpoint(
        path,
        model=model,
        step=0,
        contract=contract,
        authorization_binding=binding,
        require_authorization=True,
    )
    payload = load_checkpoint(
        path,
        model=torch.nn.Linear(2, 2),
        expected_contract=contract,
        authorization_binding=binding,
        require_authorization=True,
        restore_rng=False,
    )
    assert payload["extra"]["authorization_binding"] == binding.provenance()
    with pytest.raises(ValueError, match="contract"):
        load_checkpoint(
            path,
            model=torch.nn.Linear(2, 2),
            expected_contract={**contract, "seed": 1},
            authorization_binding=binding,
            require_authorization=True,
            restore_rng=False,
        )
    with pytest.raises(TypeError, match="AuthorizationBinding"):
        save_checkpoint(
            tmp_path / "forged.pt",
            model=model,
            step=0,
            contract=contract,
            authorization_binding=copy.copy(binding),
            require_authorization=True,
        )
    assert not (tmp_path / "forged.pt").exists()


def test_checkpoint_rejects_fractional_step(tmp_path: Path):
    model = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError, match="non-negative integer"):
        save_checkpoint(
            tmp_path / "fractional.pt", model=model, step=1.5,
            contract={"schema": "test"},
        )


def test_checkpoint_rejects_boolean_step_on_load(tmp_path: Path):
    model = torch.nn.Linear(1, 1)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model=model, step=0, contract={"schema": "test"})
    payload = torch.load(path, weights_only=False)
    payload["step"] = True
    torch.save(payload, path)
    with pytest.raises(ValueError, match="step"):
        load_checkpoint(path, model=torch.nn.Linear(1, 1), restore_rng=False)


def test_checkpoint_requires_complete_rng_state():
    state = capture_rng_state()
    incomplete = dict(state)
    incomplete.pop("python")
    with pytest.raises(ValueError, match="RNG"):
        restore_rng_state(incomplete)
    incomplete = dict(state)
    incomplete.pop("numpy")
    with pytest.raises(ValueError, match="RNG"):
        restore_rng_state(incomplete)


def test_checkpoint_rejects_shape_mismatch_without_mutating_model(tmp_path: Path):
    source = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    path = tmp_path / "corrupt.pt"
    save_checkpoint(path, model=source, step=0, contract={"schema": "test"})
    payload = torch.load(path, weights_only=False)
    key = "0.weight"
    payload["model"][key] = payload["model"][key][:-1]
    torch.save(payload, path)
    target = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    before = {name: value.detach().clone() for name, value in target.state_dict().items()}
    with pytest.raises(ValueError, match="shape|state"):
        load_checkpoint(path, model=target, restore_rng=False)
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_checkpoint_resume_defaults_to_trainer_contract_and_requires_optimizer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model = _BoundLinear()
    initial_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    authorization = _training_authorization(tmp_path, monkeypatch)
    contract = _complete_run_contract(
        authorization,
        renderer_frame_contract_hash=model.frame_contract_hash,
        calibration_hash=model.calibration_hash,
        action_contract_hash=contract_hash(model.contract),
        initial_renderer_state_sha256=module_state_sha256(model),
    )
    binding = authorization.bind_run_contract(contract)
    trainer = RendererTrainer(
        model,
        optimizer,
        contract=contract,
        authorization_binding=binding,
        operation_executor=_operation_executor(contract, binding),
    )
    path = tmp_path / "model.pt"
    trainer.save(str(path))
    other = _BoundLinear()
    other_optimizer = torch.optim.AdamW(
        other.parameters(), lr=1e-4, weight_decay=0.01
    )
    other_contract = _complete_run_contract(
        authorization,
        run_id="different-run",
        renderer_frame_contract_hash=other.frame_contract_hash,
        calibration_hash=other.calibration_hash,
        action_contract_hash=contract_hash(other.contract),
        initial_renderer_state_sha256=module_state_sha256(other),
    )
    other_binding = authorization.bind_run_contract(other_contract)
    other_trainer = RendererTrainer(
        other,
        other_optimizer,
        contract=other_contract,
        authorization_binding=other_binding,
        operation_executor=_operation_executor(other_contract, other_binding),
    )
    with pytest.raises(ValueError, match="does not match"):
        other_trainer.load(str(path))
    with pytest.raises(ValueError, match="trainer contract"):
        other_trainer.load(str(path), expected_contract=contract)
    with pytest.raises(RuntimeError, match="authorization binding"):
        load_checkpoint(
            path,
            model=other,
            optimizer=other_optimizer,
            trainer=other_trainer,
            expected_contract=contract,
            restore_rng=False,
        )
    payload = torch.load(path, weights_only=False)
    payload["optimizer"] = None
    torch.save(payload, path)
    restored = _BoundLinear()
    restored.load_state_dict(initial_state)
    restored_optimizer = torch.optim.AdamW(
        restored.parameters(), lr=1e-4, weight_decay=0.01
    )
    with pytest.raises(ValueError, match="optimizer state"):
        load_checkpoint(
            path,
            model=restored,
            optimizer=restored_optimizer,
            expected_contract=contract,
            authorization_binding=binding,
            restore_rng=False,
        )


def test_trainer_updates_and_saves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model = _BoundLinear()
    initial_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    authorization = _training_authorization(tmp_path, monkeypatch)
    contract = _complete_run_contract(
        authorization,
        renderer_frame_contract_hash=model.frame_contract_hash,
        calibration_hash=model.calibration_hash,
        action_contract_hash=contract_hash(model.contract),
        initial_renderer_state_sha256=module_state_sha256(model),
    )
    binding = authorization.bind_run_contract(contract)
    trainer = RendererTrainer(
        model,
        optimizer,
        contract=contract,
        authorization_binding=binding,
        operation_executor=_operation_executor(contract, binding),
    )
    record = trainer.update(
        lambda: (model(torch.ones(2, 1)) - 1).square().mean(),
        operation_context=_optimizer_context(trainer),
    )
    assert record.step == 1
    assert torch.isfinite(torch.tensor(record.loss))
    summary = trainer.operation_executor.ledger.summary()
    assert summary["reserved"]["optimizer_step"] == 1
    assert summary["completed_receipts"] == 1
    info = trainer.save(str(tmp_path / "model.pt"))
    assert info["step"] == 1
    restored = _BoundLinear()
    restored.load_state_dict(initial_state)
    restored_optimizer = torch.optim.AdamW(
        restored.parameters(), lr=1e-4, weight_decay=0.01
    )
    restored_trainer = RendererTrainer(
        restored,
        restored_optimizer,
        contract=contract,
        authorization_binding=binding,
        operation_executor=_operation_executor(contract, binding),
    )
    restored_trainer.load(str(tmp_path / "model.pt"), expected_contract=contract)
    assert restored_trainer.step == 1
    assert set(restored_trainer.ema_state) == set(restored.state_dict())


def test_failed_optimizer_step_is_charged_and_fully_rolled_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model = _BoundLinear()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    authorization = _training_authorization(tmp_path, monkeypatch)
    contract = _complete_run_contract(
        authorization,
        renderer_frame_contract_hash=model.frame_contract_hash,
        calibration_hash=model.calibration_hash,
        action_contract_hash=contract_hash(model.contract),
        initial_renderer_state_sha256=module_state_sha256(model),
    )
    binding = authorization.bind_run_contract(contract)
    trainer = RendererTrainer(
        model,
        optimizer,
        contract=contract,
        authorization_binding=binding,
        operation_executor=_operation_executor(contract, binding),
    )
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    def fail_after_mutation():
        with torch.no_grad():
            next(model.parameters()).add_(10.0)
        raise RuntimeError("planned optimizer failure")

    optimizer.step = fail_after_mutation
    with pytest.raises(RuntimeError, match="planned optimizer failure"):
        trainer.update(
            lambda: model(torch.ones(2, 1)).square().mean(),
            operation_context=_optimizer_context(trainer),
        )

    assert trainer.step == 0
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
    summary = trainer.operation_executor.ledger.summary()
    assert summary["reserved"]["optimizer_step"] == 1
    assert summary["completed_receipts"] == 1
    rows = [
        json.loads(line)
        for line in Path(contract["paths"]["ledger_path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    receipt = [row for row in rows if row["type"] == "receipt"][0]
    assert receipt["success"] is False
    assert receipt["result"]["failure"]["message"] == "planned optimizer failure"


def test_trainer_update_requires_catalog_authorization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    trainer = RendererTrainer(model, optimizer, contract={"schema": "test"})
    with pytest.raises(RuntimeError, match="TrainingAuthorization"):
        trainer.update(lambda: model(torch.ones(1, 1)).square().sum())

    authorized = _training_authorization(tmp_path, monkeypatch)
    with pytest.raises(TypeError, match="AuthorizationBinding"):
        RendererTrainer(
            model, optimizer, contract={"schema": "test"}, authorization=authorized
        )
    model = _BoundLinear()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    contract = _complete_run_contract(
        authorized,
        renderer_frame_contract_hash=model.frame_contract_hash,
        calibration_hash=model.calibration_hash,
        action_contract_hash=contract_hash(model.contract),
        initial_renderer_state_sha256=module_state_sha256(model),
    )
    binding = authorized.bind_run_contract(contract)
    trainer = RendererTrainer(
        model,
        optimizer,
        contract=contract,
        authorization_binding=binding,
        operation_executor=_operation_executor(contract, binding),
    )
    selected_row = json.loads(
        authorized.selected_payload_path.read_text(encoding="utf-8").splitlines()[0]
    )
    image_path = Path(selected_row["image_path"])
    image_path.write_bytes(b"changed-payload")
    with pytest.raises(RuntimeError, match="selected raw image .* changed"):
        trainer.update(lambda: model(torch.ones(1, 1)).square().sum())


def test_training_authorization_requires_explicit_repository_root(tmp_path: Path):
    receipt = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="repository_root is required"):
        TrainingAuthorization.load(receipt, repository_root=None)


@pytest.mark.parametrize("index_flag", ("--assume-unchanged", "--skip-worktree"))
def test_git_state_rejects_index_flags_that_can_hide_tracked_edits(
    tmp_path: Path, index_flag: str
):
    """Formal checkout checks must not trust stat-suppression index bits."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "repldm-tests@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "RepLDM tests"],
        cwd=tmp_path,
        check=True,
    )
    tracked = tmp_path / "tracked.py"
    tracked.write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()

    subprocess.run(
        ["git", "update-index", index_flag, "tracked.py"],
        cwd=tmp_path,
        check=True,
    )
    tracked.write_text("modified after authorization\n", encoding="utf-8")

    observed_commit, clean = authorization_module._git_state(tmp_path)
    assert observed_commit == commit
    assert clean is False


@pytest.mark.parametrize("environment_name", ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"))
def test_git_state_ignores_environment_overrides_for_the_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment_name: str,
):
    """Git state must describe the supplied checkout, regardless of Git env."""
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "repldm-tests@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "RepLDM tests"],
        cwd=repository,
        check=True,
    )
    tracked = repository / "tracked.py"
    tracked.write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repository, check=True)
    pristine = tmp_path / "pristine"
    subprocess.run(["git", "clone", "-q", str(repository), str(pristine)], check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    tracked.write_text("dirty worktree\n", encoding="utf-8")

    override = {
        "GIT_DIR": str(pristine / ".git"),
        "GIT_WORK_TREE": str(pristine),
        "GIT_INDEX_FILE": str(pristine / ".git" / "index"),
    }[environment_name]
    monkeypatch.setenv(environment_name, override)

    observed_commit, clean = authorization_module._git_state(repository)
    assert observed_commit == commit
    assert clean is False


def test_git_state_does_not_resolve_git_through_caller_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A PATH-first fake Git executable cannot forge the checkout state."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "repldm-tests@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "RepLDM tests"],
        cwd=tmp_path,
        check=True,
    )
    tracked = tmp_path / "tracked.py"
    tracked.write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()

    # Keep the injected executable outside the checkout so it does not itself
    # become an untracked worktree entry.
    fake_bin = tmp_path.parent / f"{tmp_path.name}-fake-bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' '0000000000000000000000000000000000000000'\n",
        encoding="ascii",
    )
    fake_git.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin))

    observed_commit, clean = authorization_module._git_state(tmp_path)
    assert observed_commit == commit
    assert clean is True


def test_git_environment_is_minimal_and_ignores_process_injection(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("GIT_CONFIG_PARAMETERS", "'core.worktree=/tmp/shadow'")
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", "/tmp/objects")
    monkeypatch.setenv("PATH", "/tmp/bin")
    monkeypatch.setenv("PYTHONPATH", "/tmp/shadow")

    environment = authorization_module._git_environment()

    assert environment == authorization_module._GIT_ENVIRONMENT
    assert "GIT_CONFIG_PARAMETERS" not in environment
    assert "GIT_OBJECT_DIRECTORY" not in environment
    assert "PATH" not in environment
    assert "PYTHONPATH" not in environment


def test_git_state_binds_worktree_and_disables_configured_helpers(tmp_path: Path):
    """Local core.worktree/fsmonitor settings cannot redirect or execute probes."""
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "repldm-tests@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "RepLDM tests"],
        cwd=repository,
        check=True,
    )
    tracked = repository / "tracked.py"
    tracked.write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repository, check=True)
    pristine = tmp_path / "pristine"
    subprocess.run(["git", "clone", "-q", str(repository), str(pristine)], check=True)

    marker = tmp_path / "fsmonitor-ran"
    helper = tmp_path / "fsmonitor.sh"
    helper.write_text(
        "#!/bin/sh\n"
        f"touch {marker}\n"
        "exit 0\n",
        encoding="ascii",
    )
    helper.chmod(0o755)
    subprocess.run(
        ["git", "config", "core.worktree", str(pristine)],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "core.fsmonitor", str(helper)],
        cwd=repository,
        check=True,
    )
    tracked.write_text("modified in supplied worktree\n", encoding="utf-8")

    observed_commit, clean = authorization_module._git_state(repository)

    assert len(observed_commit) == 40
    assert clean is False
    assert not marker.exists()


def test_git_upstream_rejects_a_local_branch_as_pushed_provenance(tmp_path: Path):
    """A same-commit local branch is not evidence that code was pushed."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "repldm-tests@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "RepLDM tests"],
        cwd=tmp_path,
        check=True,
    )
    tracked = tmp_path / "tracked.py"
    tracked.write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    subprocess.run(["git", "branch", "local-upstream"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "branch", "--set-upstream-to=local-upstream"],
        cwd=tmp_path,
        check=True,
    )

    with pytest.raises(ValueError, match="remote-tracking ref"):
        authorization_module._git_upstream(tmp_path, expected_commit=commit)


def test_training_authorization_rejects_v1_receipt(tmp_path: Path):
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps({"schema": "repldm.training_authorization.v1"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="v2 child release required"):
        TrainingAuthorization.load(
            receipt, repository_root=Path(__file__).resolve().parents[1]
        )


def test_training_authorization_runs_canonical_child_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = []
    authorized = _training_authorization(
        tmp_path,
        monkeypatch,
        validator=lambda release, repository: calls.append((release, repository)),
    )
    assert calls == [
        (authorized.selected_view_manifest_path.parent, authorized.repository_root)
    ]


def test_training_authorization_runs_fixed_runtime_revalidator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A valid receipt cannot load without the independent runtime pass."""
    authorized = _training_authorization(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(
        authorization_module,
        "_validate_selected_view_runtime",
        lambda release, repository: calls.append((release, repository)),
    )

    loaded = TrainingAuthorization.load(
        authorized.receipt_path, repository_root=authorized.repository_root
    )

    assert loaded.selected_view_release_id == authorized.selected_view_release_id
    assert calls == [
        (authorized.selected_view_manifest_path.parent, authorized.repository_root)
    ]


def test_training_authorization_rejects_runtime_revalidation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    authorized = _training_authorization(tmp_path, monkeypatch)

    def fail(*_args):
        raise ValueError("fixed runtime did not reproduce gate evidence")

    monkeypatch.setattr(authorization_module, "_validate_selected_view_runtime", fail)
    with pytest.raises(ValueError, match="runtime did not reproduce"):
        TrainingAuthorization.load(
            authorized.receipt_path, repository_root=authorized.repository_root
        )


def test_authorization_writer_does_not_follow_predictable_temp_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    trap = tmp_path / "unrelated.txt"
    trap.write_text("unchanged\n", encoding="utf-8")
    predictable = tmp_path / "training_authorization.json.tmp"
    predictable.symlink_to(trap)
    _training_authorization(tmp_path, monkeypatch)
    assert trap.read_text(encoding="utf-8") == "unchanged\n"
    assert predictable.is_symlink()


def test_training_authorization_rejects_missing_parent_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        authorization_module, "_validate_selected_view_release", lambda *args: None
    )
    monkeypatch.setattr(authorization_module, "_git_state", lambda *args: (COMMIT, True))
    monkeypatch.setattr(
        authorization_module, "_git_upstream", lambda *args, **kwargs: None
    )
    fixture = SelectedReleaseFixture(tmp_path)
    (fixture.parent / "benchmark_holdouts.jsonl").unlink()
    with pytest.raises(ValueError, match="candidate parent artifact"):
        write_authorization_receipt(
            tmp_path / "training_authorization.json",
            selected_view_manifest=fixture.release / "manifest.json",
            code_commit=COMMIT,
            repository_root=Path(__file__).resolve().parents[1],
        )


def test_training_authorization_rejects_parent_self_declaring_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        authorization_module, "_validate_selected_view_release", lambda *args: None
    )
    monkeypatch.setattr(authorization_module, "_git_state", lambda *args: (COMMIT, True))
    monkeypatch.setattr(
        authorization_module, "_git_upstream", lambda *args, **kwargs: None
    )
    fixture = SelectedReleaseFixture(tmp_path)
    parent_path = fixture.parent / "manifest.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent["complete"] = True
    parent["training_ready"] = True
    parent_path.write_text(json.dumps(parent, sort_keys=True) + "\n", encoding="utf-8")
    fixture.rebind_parent()
    with pytest.raises(ValueError, match="candidate parent must remain non-training-ready"):
        write_authorization_receipt(
            tmp_path / "training_authorization.json",
            selected_view_manifest=fixture.release / "manifest.json",
            code_commit=COMMIT,
            repository_root=Path(__file__).resolve().parents[1],
        )


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("child", "selected-view manifest changed"),
        ("parent", "candidate parent manifest changed"),
        ("config", "selected-view config changed"),
        ("payload", "selected payload changed"),
        ("model", "external asset .*model.* changed"),
        ("calibration", "external asset .*calibration.* changed"),
        ("index", "external asset .*protected_index.* changed"),
        ("license", "external asset .*license_evidence.* changed"),
    ],
)
def test_training_authorization_rechecks_every_child_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    message: str,
):
    authorized = _training_authorization(tmp_path, monkeypatch)
    paths = {
        "child": authorized.selected_view_manifest_path,
        "parent": authorized.candidate_parent_manifest_path,
        "config": authorized.selected_config_path,
        "payload": authorized.selected_payload_path,
    }
    config = json.loads(authorized.selected_config_path.read_text(encoding="utf-8"))
    if target == "model":
        paths[target] = Path(config["classifier"]["model"]["files"][0]["path"])
    elif target == "calibration":
        paths[target] = Path(config["semantic_text"]["calibration"]["path"])
    elif target == "index":
        paths[target] = Path(config["protected_index"]["semantic_text"]["path"])
    elif target == "license":
        paths[target] = Path(
            config["sources"]["four_k_lsdb"]["license_evidence"][0]["path"]
        )
    paths[target].write_bytes(paths[target].read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match=message):
        authorized.validate_current()


def test_training_authorization_rechecks_receipt_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    authorized = _training_authorization(tmp_path, monkeypatch)
    authorized.receipt_path.write_bytes(authorized.receipt_path.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="receipt changed"):
        authorized.validate_current()


@pytest.mark.parametrize("failure", ("dirty", "unpushed"))
def test_training_authorization_rejects_before_running_child_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
):
    authorized = _training_authorization(tmp_path, monkeypatch)
    called = []
    monkeypatch.setattr(
        authorization_module,
        "_validate_selected_view_release",
        lambda *args, **kwargs: called.append(True),
    )
    if failure == "dirty":
        monkeypatch.setattr(
            authorization_module,
            "_git_state",
            lambda *args: (authorized.code_commit, False),
        )
    else:
        monkeypatch.setattr(
            authorization_module,
            "_git_state",
            lambda *args: (authorized.code_commit, True),
        )
        monkeypatch.setattr(
            authorization_module,
            "_git_upstream",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                ValueError("not pushed")
            ),
        )
    with pytest.raises(ValueError):
        TrainingAuthorization.load(
            authorized.receipt_path,
            repository_root=authorized.repository_root,
        )
    assert called == []


def test_authorization_binding_rejects_copies_and_requires_component_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    authorization = _training_authorization(tmp_path, monkeypatch)
    contract = _complete_run_contract(authorization)
    binding = authorization.bind_run_contract(contract)
    assert binding.is_validated()
    assert require_authorization_binding(binding) is binding
    for forged in (
        copy.copy(binding),
        replace(binding),
        AuthorizationBinding(authorization, TrainingRunContract.from_mapping(contract)),
    ):
        with pytest.raises(TypeError, match="AuthorizationBinding"):
            require_authorization_binding(forged)
    with pytest.raises(TypeError, match="frame contract hash"):
        binding.validate_component(torch.nn.Linear(1, 1))


def test_authorization_binding_detects_contract_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    authorization = _training_authorization(tmp_path, monkeypatch)
    binding = authorization.bind_run_contract(_complete_run_contract(authorization))
    # Frozen dataclasses protect ordinary callers; this models an integration
    # layer that accidentally mutates a private field and must be caught before
    # any renderer or reward work.
    changed = binding.contract
    changed["seed"] = 99
    object.__setattr__(binding.run_contract, "_canonical_json", json.dumps(changed))
    with pytest.raises(RuntimeError, match="contract"):
        binding.validate_current()


def test_formal_collector_rejects_untyped_callbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    authorization = _training_authorization(tmp_path, monkeypatch)
    calibration = FrameCalibration(
        (True,) * 6,
        CALIBRATION_STATE_COUNT,
        (1.0,) * 6,
        (1.0,) * 6,
        manifest_sha256="1" * 64,
        source_sha256="2" * 64,
        state_provenance_sha256="3" * 64,
    )
    renderer = EulerNativeFrameV1(calibration=calibration, latent_channels=2)
    contract = _complete_run_contract(
        authorization,
        renderer_frame_contract_hash=renderer.frame_contract_hash,
        calibration_hash=renderer.calibration_hash,
        action_contract_hash=contract_hash(renderer.contract),
    )
    binding = authorization.bind_run_contract(contract)
    calls = []
    with pytest.raises(RuntimeError, match="requires SdxlEulerTrainingAdapter"):
        EulerNativeRolloutCollector(
            renderer,
            decision_indices=(0,),
            registered_decision_indices=(0,),
            total_steps=1,
            observe_fn=lambda *_args: calls.append("observe"),
            transition_fn=lambda *_args: calls.append("transition"),
            physical_unet=torch.nn.Identity(),
            run_contract=contract,
            authorization_binding=binding,
        )
    assert calls == []


def test_training_authorization_rejects_manually_constructed_object(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    authorized = _training_authorization(tmp_path, monkeypatch)
    forged = _forged_authorization(authorized)
    assert forged.is_validated() is False
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    with pytest.raises(TypeError, match="AuthorizationBinding"):
        RendererTrainer(
            model, optimizer, contract={"schema": "test"}, authorization=forged
        )


def test_trainer_rejects_authorization_replacement_after_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    authorized = _training_authorization(tmp_path, monkeypatch)
    model = _BoundLinear()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    contract = _complete_run_contract(
        authorized,
        renderer_frame_contract_hash=model.frame_contract_hash,
        calibration_hash=model.calibration_hash,
        action_contract_hash=contract_hash(model.contract),
        initial_renderer_state_sha256=module_state_sha256(model),
    )
    binding = authorized.bind_run_contract(contract)
    trainer = RendererTrainer(
        model,
        optimizer,
        contract=contract,
        authorization_binding=binding,
        operation_executor=_operation_executor(contract, binding),
    )
    trainer.authorization_binding = object()
    with pytest.raises(RuntimeError, match="cleared or replaced"):
        trainer.update(lambda: model(torch.ones(1, 1)).square().sum())


def test_trainer_rejects_training_authorization_subclass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    authorized = _training_authorization(tmp_path, monkeypatch)

    class ForgedAuthorization(TrainingAuthorization):
        def validate_current(self):
            raise AssertionError("forged validation should never run")

    forged = _forged_authorization(authorized, ForgedAuthorization)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    with pytest.raises(TypeError, match="AuthorizationBinding"):
        RendererTrainer(
            model, optimizer, contract={"schema": "test"}, authorization=forged
        )


def test_training_authorization_rechecks_unrelated_catalog_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    authorized = _training_authorization(tmp_path, monkeypatch)
    # The second artifact is unrelated to the selected payload but remains
    # part of the content-addressed catalog contract.
    views = authorized.candidate_parent_manifest_path.parent / "training_views.jsonl"
    views.write_text(views.read_text(encoding="utf-8") + " \n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="candidate parent artifact.*changed"):
        authorized.validate_current()


def test_target_config_rejects_nonfinite_constants():
    with pytest.raises(ValueError):
        TargetStepConfig(eta_target=float("nan"))

    with pytest.raises(ValueError, match="integer"):
        TargetStepConfig(target_steps=2.5)


def _test_reward_gradient(reward: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    gradient = torch.autograd.grad(reward.sum(), inputs, allow_unused=True)[0]
    return torch.zeros_like(inputs) if gradient is None else gradient


def test_reward_targets_mark_zero_gradient_rows_invalid():
    anchor = torch.zeros(2, 3)
    pair = construct_reward_targets(
        anchor,
        lambda value: value,
        lambda clean: clean.square().sum(dim=-1),
        reward_gradient=_test_reward_gradient,
        config=TargetStepConfig(target_steps=1),
    )
    assert pair.valid is not None
    assert not bool(pair.valid[0])


def test_reward_targets_are_detached_and_trust_region_bounded():
    anchor = torch.zeros(2, 3, requires_grad=True)
    reward_calls = []
    gradient_calls = []

    def gradient(reward, inputs):
        gradient_calls.append((reward, inputs))
        return _test_reward_gradient(reward, inputs)

    pair = construct_reward_targets(
        anchor,
        lambda value: value.square(),
        lambda clean: (reward_calls.append(clean) or clean[..., 0].square()),
        reward_gradient=gradient,
        config=TargetStepConfig(eta_target=2.0, target_steps=2, trust_radius_u=0.25),
    )
    assert not pair.plus_u.requires_grad
    assert not pair.minus_u.requires_grad
    assert torch.all(torch.linalg.vector_norm(pair.plus_u - pair.anchor_u, dim=-1) <= 0.250001)
    assert torch.all(torch.linalg.vector_norm(pair.minus_u - pair.anchor_u, dim=-1) <= 0.250001)
    assert len(reward_calls) == 1
    assert len(gradient_calls) == 1


def test_reward_target_gradient_respects_inactive_action_slots():
    contract = ActionSpaceContract(3, (True, False, True))
    anchor = torch.tensor([[0.2, 0.0, -0.3]])
    pair = construct_reward_targets(
        anchor,
        lambda value: value,
        lambda clean: clean.square().sum(dim=-1),
        reward_gradient=_test_reward_gradient,
        contract=contract,
        candidate_validator=lambda value: torch.ones(
            value.shape[0], dtype=torch.bool, device=value.device
        ),
        config=TargetStepConfig(target_steps=1),
    )
    assert pair.gradient_u[0, 1].item() == 0.0
    assert pair.plus_u[0, 1].item() == 0.0
    assert pair.minus_u[0, 1].item() == 0.0


def test_reward_targets_fail_closed_to_anchor_after_partial_walk():
    anchor = torch.tensor([[0.25, -0.25]])
    calls = 0

    def validator(value):
        nonlocal calls
        calls += 1
        # Accept the first candidate for each sign, then reject every second
        # substep and all of its backtracking candidates.
        return torch.tensor(
            [calls in {1, 9}], dtype=torch.bool, device=value.device
        )

    pair = construct_reward_targets(
        anchor,
        lambda value: value,
        lambda clean: clean.sum(dim=-1),
        reward_gradient=_test_reward_gradient,
        candidate_validator=validator,
        config=TargetStepConfig(target_steps=2),
    )

    assert pair.valid is not None and not bool(pair.valid[0])
    assert torch.equal(pair.plus_u, anchor)
    assert torch.equal(pair.minus_u, anchor)
    assert torch.count_nonzero(pair.gradient_u) == 0


def test_reward_targets_reject_numeric_validator_and_gradient_coercion():
    anchor = torch.zeros(1, 2)
    with pytest.raises(TypeError, match="bool flags"):
        construct_reward_targets(
            anchor,
            lambda value: value,
            lambda clean: clean.sum(dim=-1),
            reward_gradient=_test_reward_gradient,
            candidate_validator=lambda value: value.new_tensor([float("nan")]),
            config=TargetStepConfig(target_steps=1),
        )
    with pytest.raises(ValueError, match="floating point"):
        construct_reward_targets(
            anchor,
            lambda value: value,
            lambda clean: clean.sum(dim=-1),
            reward_gradient=lambda reward, inputs: torch.ones_like(
                inputs, dtype=torch.int64
            ),
            config=TargetStepConfig(target_steps=1),
        )


def test_reward_targets_require_geometry_validator_with_action_contract():
    with pytest.raises(ValueError, match="candidate validator"):
        construct_reward_targets(
            torch.zeros(1, 3),
            lambda value: value,
            lambda clean: clean.sum(dim=-1),
            reward_gradient=_test_reward_gradient,
            contract=ActionSpaceContract(3, (True, False, True)),
            config=TargetStepConfig(target_steps=1),
        )
