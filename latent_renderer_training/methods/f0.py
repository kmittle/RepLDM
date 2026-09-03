"""Formal F0 feasibility runner for the scheduler-native latent renderer.

F0 is deliberately separate from the online OPD/DPO/RL arms.  It builds the
shared OPSD-style teacher once, using four cross-fitted renderers for evidence
and one final all-prompt renderer for deployment.  Reward graphs are consumed
only while constructing immutable targets; fitting reads detached targets.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

import torch
from torch import Tensor

from ..artifacts import module_state_sha256
from ..authorization import require_authorization_binding
from ..collector import CachedEulerDecision, CachedSuffixResult, F0AnchorTrace
from ..checkpoint import load_checkpoint
from ..f0_metrics import (
    F0_METRIC_ROW_SCHEMA,
    F0_PIXEL_SUMMARY_SCHEMA,
    build_f0_metric_action,
    build_f0_screen_registration,
    compute_f0_pixel_summary,
    validate_f0_screen_registration_evidence,
    write_f0_phase_evidence,
    write_f0_screen_registration_evidence,
)
from ..f0_targets import (
    F0FoldSplit,
    F0TargetRow,
    f0_crossfit_splits,
    f0_objective,
    f0_target_stable_id,
    validate_f0_rows,
)
from ..f0_teacher import TerminalAnchorRewardReceipt, construct_f0_target
from ..gates import (
    F0_GATE_SCHEMA,
    OPSD_TEACHER_STATE_SCHEMA,
    REWARD_STATISTICS_SCHEMA,
    validate_f0_run_outputs,
    validate_reward_statistics,
)
from ..operations import (
    OperationReceipt,
    operation_output_sha256,
    tensor_operation_output_sha256,
)
from ..renderer import EulerFrameOutput, EulerNativeFrameV1, tensor_sha256
from ..storage import (
    AtomicF0AnchorStore,
    AtomicF0TargetStore,
    CheckpointProvenance,
    F0AnchorScoreRecovery,
)
from ..trainer import RendererTrainer, UpdateRecord
from ..witnesses import (
    TOPIQ_NR_MODEL_ID,
    TOPIQ_NR_PREPROCESS_SHA256,
    TOPIQ_NR_ROLE,
)
from .runner import (
    METHOD_RESULT_SCHEMA,
    PromptRecord,
    PromptSeedBlock,
    _atomic_json,
    _blocks,
    _copy_ema,
    _file_descriptor,
    _make_optimizer,
    _prompt_records,
    _result_sha256,
    _save_renderer_checkpoint,
)


F0_TRAINING_SEEDS = (2026090101, 2026090102)
F0_VALIDATION_SEEDS = (2026090191, 2026090192)
F0_DECISION_INDICES = (8, 24, 40)
F0_FOLD_OPTIMIZATION_SEEDS = (
    2026090100,
    2026090101,
    2026090102,
    2026090103,
)
F0_FINAL_OPTIMIZATION_SEED = 2026090104
F0_FIT_ORDER_SCHEMA = "sha256_epoch_order_v1"
F0_FIT_BATCH_SIZE = 8
F0_FIT_STEPS = 200

# Complete train plus confirmation allocation.  ImageReward and TOPIQ-NR are
# separate physical reward forwards even when they score the same decoded image.
F0_QUERY_BUDGET = {
    "unet_forward": 52_800,
    "scheduler_step": 54_528,
    "vae_decode": 2_496,
    "reward_forward": 4_416,
    "reward_backward": 576,
    "optimizer_step": 1_000,
}

F0_TRAIN_QUERY_USAGE = {
    "unet_forward": 35_200,
    "scheduler_step": 36_352,
    "vae_decode": 1_664,
    "reward_forward": 2_944,
    "reward_backward": 384,
    "optimizer_step": 800,
}

F0_SCREEN_REGISTRATION_FILENAME = "f0-screen-registration.jsonl"
F0_REWARD_STATISTICS_FILENAME = "reward-statistics.json"
F0_TRAIN_EVIDENCE_FILENAME = "f0-train-evidence.jsonl"
F0_VALIDATION_EVIDENCE_FILENAME = "f0-validation-evidence.jsonl"
F0_TEACHER_STATE_FILENAME = "opsd-teacher-state.json"
F0_GATE_FILENAME = "f0-gate.json"
F0_RESULT_FILENAME = "f0-result.json"
F0_FAILURE_FILENAME = "f0-failure.json"
F0_FAILURE_SCHEMA = "repldm.renderer_f0_failure.v1"

F0_FIXED_HYPERPARAMETERS = {
    "method": "f0",
    "training_generation_seeds": list(F0_TRAINING_SEEDS),
    "validation_generation_seeds": list(F0_VALIDATION_SEEDS),
    "decision_indices": list(F0_DECISION_INDICES),
    "eta_target": 0.25,
    "target_steps": 2,
    "trust_radius_u": 0.50,
    "backtracking": [1.0, 0.5, 0.25, 0.125],
    "epsilon_grad": 1e-12,
    "branch_coefficient": 1.0,
    "anchor_weight": 0.10,
    "fit_steps_per_fold": F0_FIT_STEPS,
    "fit_batch_size": F0_FIT_BATCH_SIZE,
    "fit_order": F0_FIT_ORDER_SCHEMA,
    "fold_optimizer_seeds": list(F0_FOLD_OPTIMIZATION_SEEDS),
    "final_optimizer_seed": F0_FINAL_OPTIMIZATION_SEED,
    "deployment_weights": "final_ema",
}


@dataclass(frozen=True)
class F0FitResult:
    """One independent 200-update fit and its frozen EMA checkpoint."""

    role: str
    optimizer_seed: int
    held_out_fold: int | None
    renderer: torch.nn.Module
    checkpoint: CheckpointProvenance
    updates: tuple[Mapping[str, Any], ...]
    raw_renderer_state_sha256: str
    ema_renderer_state_sha256: str


@dataclass(frozen=True)
class F0ScoredImage:
    """One decoded image with both independent scores and durable receipts."""

    image_sha256: str
    reward: Tensor
    topiq_nr: Tensor
    pixel: Mapping[str, float]
    decode_receipt: OperationReceipt
    reward_receipt: OperationReceipt
    witness_receipt: OperationReceipt

    def __post_init__(self) -> None:
        for name in ("reward", "topiq_nr"):
            value = getattr(self, name)
            if (
                not isinstance(value, Tensor)
                or value.numel() != 1
                or value.requires_grad
                or not value.is_floating_point()
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"F0 {name} must be one detached finite tensor")
            object.__setattr__(self, name, value.detach().clone())
        if not isinstance(self.image_sha256, str) or len(self.image_sha256) != 64:
            raise ValueError("F0 image SHA-256 is invalid")
        if not isinstance(self.pixel, Mapping) or set(self.pixel) != {
            "clipped_fraction",
            "mean_saturation",
            "contrast",
        }:
            raise ValueError("F0 pixel summary is incomplete")
        for name, value in self.pixel.items():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"F0 pixel metric {name} must be finite")
        for receipt in (
            self.decode_receipt,
            self.reward_receipt,
            self.witness_receipt,
        ):
            if type(receipt) is not OperationReceipt:
                raise TypeError("F0 scores require native operation receipts")
        object.__setattr__(
            self,
            "pixel",
            {name: float(value) for name, value in self.pixel.items()},
        )


@dataclass(frozen=True)
class F0AnchorBlock:
    """Strict-C0 trace and terminal measurements for one prompt-seed block."""

    block: PromptSeedBlock
    trace: F0AnchorTrace
    score: F0ScoredImage


@dataclass(frozen=True)
class F0BranchAudit:
    """Compact safety and scoring record for one completed native suffix."""

    action: Tensor
    rendered_transition: Tensor
    frame_valid: bool
    finite: bool
    scheduler_valid: bool
    moment_valid: bool
    hard_cap_valid: bool
    mapped_update_ratio: float
    at_98pct_cap: bool
    score: F0ScoredImage


@dataclass(frozen=True)
class F0TargetEvaluation:
    """Detached target plus compact audits of its two terminal branches."""

    target: F0TargetRow
    plus: F0BranchAudit
    minus: F0BranchAudit


_T = TypeVar("_T")


def _stable_order_key(row: F0TargetRow, *, optimizer_seed: int, epoch: int) -> str:
    return hashlib.sha256(
        (
            f"{F0_FIT_ORDER_SCHEMA}\0{optimizer_seed}\0{epoch}\0"
            f"{row.stable_id}"
        ).encode("utf-8")
    ).hexdigest()


def f0_fit_batches(
    values: Iterable[F0TargetRow],
    *,
    optimizer_seed: int,
    updates: int = F0_FIT_STEPS,
    batch_size: int = F0_FIT_BATCH_SIZE,
) -> tuple[tuple[F0TargetRow, ...], ...]:
    """Build the frozen epoch-keyed fit order without versioned RNG behavior."""
    if (
        type(optimizer_seed) is not int
        or optimizer_seed
        not in {*F0_FOLD_OPTIMIZATION_SEEDS, F0_FINAL_OPTIMIZATION_SEED}
    ):
        raise ValueError("F0 fit requires one registered optimizer seed")
    if type(updates) is not int or updates != F0_FIT_STEPS:
        raise ValueError("F0 fit must contain exactly 200 optimizer updates")
    if type(batch_size) is not int or batch_size != F0_FIT_BATCH_SIZE:
        raise ValueError("F0 fit batch size must remain eight logical states")
    rows = tuple(values)
    if len(rows) not in {288, 384}:
        raise ValueError("F0 fits require exactly 288 cross-fit or 384 final rows")
    if any(not isinstance(row, F0TargetRow) or row.split != "train" for row in rows):
        raise ValueError("F0 fitting accepts only typed training targets")
    stable_ids = tuple(row.stable_id for row in rows)
    if len(set(stable_ids)) != len(stable_ids):
        raise ValueError("F0 fit rows contain duplicate stable IDs")
    if len(rows) % batch_size:
        raise RuntimeError("the frozen F0 matrix must divide into complete batches")

    batches: list[tuple[F0TargetRow, ...]] = []
    epoch = 0
    while len(batches) < updates:
        ordered = sorted(
            rows,
            key=lambda row: _stable_order_key(
                row, optimizer_seed=optimizer_seed, epoch=epoch
            ),
        )
        for offset in range(0, len(ordered), batch_size):
            batches.append(tuple(ordered[offset : offset + batch_size]))
            if len(batches) == updates:
                break
        epoch += 1
    return tuple(batches)


def _require_f0_contract(binding: Any) -> Mapping[str, Any]:
    binding = require_authorization_binding(binding)
    binding.validate_current()
    contract = binding.contract
    if contract.get("method") != "f0":
        raise ValueError("F0 runner requires an authorized f0 contract")
    if contract.get("query_budget") != F0_QUERY_BUDGET:
        raise ValueError("F0 query budget differs from the frozen allocation")
    settings = contract.get("method_hyperparameters")
    if not isinstance(settings, Mapping):
        raise ValueError("F0 method hyperparameters are missing")
    fixed = {key: settings.get(key) for key in F0_FIXED_HYPERPARAMETERS}
    if fixed != F0_FIXED_HYPERPARAMETERS:
        raise ValueError("F0 hyperparameters differ from the frozen protocol")
    # The immutable screen-registration artifact is the only source of truth
    # for sample counts, thresholds, bootstrap behavior, and benchmark duties.
    # Caller-supplied power claims were retired because they were not evidence.
    if set(settings) != set(F0_FIXED_HYPERPARAMETERS):
        raise ValueError("F0 hyperparameters differ from the frozen protocol")
    if contract.get("selected_rows") != 96:
        raise ValueError("F0 currently requires the frozen 64+32 selected view")
    if tuple(contract.get("decision_indices", ())) != F0_DECISION_INDICES:
        raise ValueError("F0 decision schedule differs from the protocol")
    return contract


def _move_value(value: _T, device: torch.device | str) -> _T:
    """Move a detached nested replay snapshot without retaining GPU graphs."""
    if isinstance(value, Tensor):
        return value.detach().to(device=device)  # type: ignore[return-value]
    if is_dataclass(value):
        return replace(  # type: ignore[return-value]
            value,
            **{
                field.name: _move_value(getattr(value, field.name), device)
                for field in fields(value)
                if field.init
            },
        )
    if isinstance(value, Mapping):
        return type(value)(
            (key, _move_value(item, device)) for key, item in value.items()
        )  # type: ignore[return-value,call-arg]
    if isinstance(value, tuple):
        return tuple(_move_value(item, device) for item in value)  # type: ignore[return-value]
    if isinstance(value, list):
        return [_move_value(item, device) for item in value]  # type: ignore[return-value]
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii") + b"\n"
    except (TypeError, ValueError) as exc:
        raise ValueError("F0 artifact must contain finite JSON values") from exc


def _write_json_once(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Publish canonical JSON once and accept only byte-identical resume state."""
    payload = _canonical_json(value)
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace F0 artifact: {path}")
        return _file_descriptor(path)
    try:
        _atomic_json(path, value)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise
    return _file_descriptor(path)


def _publish_screen_registration(
    binding: Any, contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Freeze the F0 decision rule before a ledgered image can be produced."""
    run_dir = Path(contract["paths"]["run_dir"])
    path = run_dir / F0_SCREEN_REGISTRATION_FILENAME
    if not os.path.lexists(path):
        ledger_path = Path(contract["paths"]["ledger_path"])
        evidence_paths = (
            run_dir / "f0_targets",
            Path(contract["paths"]["output_dir"]),
            Path(contract["paths"]["checkpoint_dir"]),
        )
        if os.path.lexists(ledger_path) and (
            ledger_path.is_symlink()
            or not ledger_path.is_file()
            or ledger_path.stat().st_size > 0
        ):
            raise RuntimeError(
                "F0 ledger exists without a prior screen registration"
            )
        for candidate in evidence_paths:
            if os.path.lexists(candidate) and (
                candidate.is_symlink()
                or not candidate.is_dir()
                or any(candidate.iterdir())
            ):
                raise RuntimeError(
                    "F0 output exists without a prior screen registration"
                )
    artifact = build_f0_screen_registration(
        f0_run_contract_sha256=binding.contract_hash
    )
    result = write_f0_screen_registration_evidence(path, artifact)
    validated, normalized = validate_f0_screen_registration_evidence(
        result, f0_run_contract_sha256=binding.contract_hash
    )
    if validated != artifact or normalized != result:
        raise RuntimeError("published F0 screen registration did not round-trip")
    return result


def _score_value(value: Tensor, *, label: str) -> float:
    if (
        not isinstance(value, Tensor)
        or value.numel() != 1
        or not value.is_floating_point()
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"{label} must contain one finite score")
    return float(value.detach().double().cpu().reshape(()))


def _score_terminal(
    runtime: Any,
    state: Any,
    *,
    metric_branch: str,
    target_record_sha256: str | None,
) -> F0ScoredImage:
    """Decode once, then score the exact tensor with ImageReward and TOPIQ-NR."""
    runtime.binding.validate_current()
    witness = runtime.witness_model
    if witness is None or getattr(witness, "operation_executor", None) is not (
        runtime.operation_executor
    ):
        raise RuntimeError("F0 requires the runtime's independent TOPIQ-NR witness")
    action_metadata = build_f0_metric_action(
        branch=metric_branch,
        target_record_sha256=target_record_sha256,
        renderer_action_history=state.action_history,
    )
    with torch.no_grad():
        image, decode_receipt = runtime.adapter.decode_with_receipt(
            state, action_metadata=action_metadata
        )
        reward, reward_receipt = runtime.adapter.reward_with_receipt(
            state,
            image,
            parent_receipt=decode_receipt,
            action_metadata=action_metadata,
        )
        image_sha256 = tensor_sha256(image)
        witness_context = state.operation_context(
            action_metadata=action_metadata,
            image_hash=image_sha256,
            cached_parent=operation_output_sha256(image),
        )
        topiq_nr, witness_receipt = witness.score_with_receipt(
            image,
            witness_context,
            parent_receipt=decode_receipt,
        )
        pixel = compute_f0_pixel_summary(image)
    _score_value(reward, label="F0 ImageReward")
    _score_value(topiq_nr, label="F0 TOPIQ-NR")
    return F0ScoredImage(
        image_sha256=image_sha256,
        reward=reward.detach().cpu(),
        topiq_nr=topiq_nr.detach().cpu(),
        pixel=pixel,
        decode_receipt=decode_receipt,
        reward_receipt=reward_receipt,
        witness_receipt=witness_receipt,
    )


def _receipt_reference(receipt: OperationReceipt) -> dict[str, str]:
    return {
        "reservation_id": receipt.reservation_id,
        "output_sha256": receipt.output_hash,
    }


def _score_provenance(
    score: F0ScoredImage, *, reuse_scope: str
) -> dict[str, Any]:
    checkpoint_hashes = {
        score.decode_receipt.context.checkpoint_hash,
        score.reward_receipt.context.checkpoint_hash,
        score.witness_receipt.context.checkpoint_hash,
    }
    if len(checkpoint_hashes) != 1:
        raise RuntimeError("F0 score receipts do not share one renderer checkpoint")
    return {
        "reuse_scope": reuse_scope,
        "renderer_state_sha256": checkpoint_hashes.pop(),
        "image_sha256": score.image_sha256,
        "image_reward": _receipt_reference(score.reward_receipt),
        "topiq_nr": _receipt_reference(score.witness_receipt),
    }


def _restore_anchor_score(
    runtime: Any,
    trace: F0AnchorTrace,
    recovery: F0AnchorScoreRecovery,
) -> F0ScoredImage:
    """Re-seal a stored anchor score and bind every value to its durable receipt."""
    if not isinstance(trace, F0AnchorTrace) or not isinstance(
        recovery, F0AnchorScoreRecovery
    ):
        raise TypeError("F0 anchor recovery requires a trace and score recovery")
    state = trace.terminal_state
    action_metadata = build_f0_metric_action(
        branch="anchor",
        target_record_sha256=None,
        renderer_action_history=state.action_history,
    )
    executor = runtime.operation_executor
    decode_receipt = executor.restore_receipt(
        recovery.vae_decode_reservation_id
    )
    reward_receipt = executor.restore_receipt(
        recovery.image_reward_reservation_id
    )
    witness_receipt = executor.restore_receipt(
        recovery.topiq_nr_reservation_id
    )
    image_output = tensor_operation_output_sha256(recovery.image_sha256)
    decode_context = state.operation_context(action_metadata=action_metadata)
    executor.validate_receipt(
        decode_receipt,
        kind="vae_decode",
        output_hash=image_output,
        context=decode_context,
        scalar_or_gradient="tensor",
    )
    score_context = state.operation_context(
        action_metadata=action_metadata,
        image_hash=recovery.image_sha256,
        cached_parent=image_output,
    )
    reward = recovery.image_reward.reshape(1)
    topiq_nr = recovery.topiq_nr.reshape(1)
    executor.validate_receipt(
        reward_receipt,
        kind="reward_forward",
        output_hash=operation_output_sha256(reward),
        context=score_context,
        scalar_or_gradient="scalar",
        parent_reservation_id=decode_receipt.reservation_id,
    )
    executor.validate_receipt(
        witness_receipt,
        kind="reward_forward",
        output_hash=operation_output_sha256(topiq_nr),
        context=score_context,
        scalar_or_gradient="scalar",
        model=TOPIQ_NR_MODEL_ID,
        role=TOPIQ_NR_ROLE,
        preprocess_hash=TOPIQ_NR_PREPROCESS_SHA256,
        parent_reservation_id=decode_receipt.reservation_id,
    )
    return F0ScoredImage(
        image_sha256=recovery.image_sha256,
        reward=reward,
        topiq_nr=topiq_nr,
        pixel=recovery.pixel_summary,
        decode_receipt=decode_receipt,
        reward_receipt=reward_receipt,
        witness_receipt=witness_receipt,
    )


def _percentile_type7(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("invalid F0 quantile request")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("F0 reward statistics contain non-finite values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _reward_statistics_payload(
    anchors: Sequence[F0AnchorBlock], contract: Mapping[str, Any]
) -> dict[str, Any]:
    ordered = sorted(
        anchors,
        key=lambda item: (
            item.block.record.record_id,
            item.block.generation_seed,
        ),
    )
    if len(ordered) != 128:
        raise ValueError("F0 reward statistics require 128 strict-C0 anchors")
    rewards = [_score_value(item.score.reward, label="F0 anchor reward") for item in ordered]
    location = _percentile_type7(rewards, 0.5)
    scale = max(
        (_percentile_type7(rewards, 0.75) - _percentile_type7(rewards, 0.25))
        / 1.349,
        1e-6,
    )
    payload = {
        "schema": REWARD_STATISTICS_SCHEMA,
        "status": "frozen",
        "anchor_count": 128,
        "estimator": "median_iqr_over_1.349_floor_1e-6",
        "location": location,
        "scale": scale,
        "initial_renderer_state_sha256": contract[
            "initial_renderer_state_sha256"
        ],
        "reward_config_sha256": contract["reward_config_sha256"],
        "reward_preprocess_sha256": contract["reward_preprocess_sha256"],
        "selected_view_release_id": contract["selected_view_release_id"],
        "anchors": [
            {
                "prompt_id": item.block.record.record_id,
                "generation_seed": item.block.generation_seed,
                "image_sha256": item.score.image_sha256,
                "reward": _score_value(item.score.reward, label="F0 anchor reward"),
            }
            for item in ordered
        ],
    }
    return validate_reward_statistics(payload, contract=contract)


def _renderer_device(renderer: torch.nn.Module) -> torch.device:
    try:
        return next(renderer.parameters()).device
    except StopIteration as exc:
        raise ValueError("F0 renderer has no parameters") from exc


def _values_equal(left: Any, right: Any) -> bool:
    """Compare detached replay structures without dtype or tolerance coercion."""
    if isinstance(left, Tensor) or isinstance(right, Tensor):
        return isinstance(left, Tensor) and isinstance(right, Tensor) and torch.equal(
            left.detach().cpu(), right.detach().cpu()
        )
    if is_dataclass(left) or is_dataclass(right):
        if type(left) is not type(right) or not is_dataclass(left):
            return False
        return all(
            _values_equal(getattr(left, field.name), getattr(right, field.name))
            for field in fields(left)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_values_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_values_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def _collect_anchor_blocks(
    runtime: Any,
    records: Sequence[PromptRecord],
    *,
    split: str,
) -> tuple[F0AnchorBlock, ...]:
    """Collect each strict-C0 prefix once and retain its three cached decisions."""
    if split not in {"train", "validation"}:
        raise ValueError("F0 anchor collection requires train or validation")
    renderer = runtime.renderer
    reference = runtime.reference_renderer
    if not isinstance(renderer, EulerNativeFrameV1) or not isinstance(
        reference, EulerNativeFrameV1
    ):
        raise TypeError("F0 runtime requires EulerNativeFrameV1 renderers")
    checkpoint = runtime.initial_checkpoint_provenance
    if not isinstance(checkpoint, CheckpointProvenance):
        raise TypeError("F0 runtime omitted initial checkpoint provenance")
    checkpoint.validate_current()
    state_hash = module_state_sha256(renderer)
    if (
        checkpoint.renderer_state_sha256 != state_hash
        or module_state_sha256(reference) != state_hash
    ):
        raise RuntimeError("F0 anchors must use the shared strict-C0 checkpoint")
    runtime.binding.validate_initial_renderer(renderer)
    runtime.binding.validate_initial_renderer(reference)
    contract = runtime.binding.contract
    blocks = _blocks(records, split=split)
    expected = 128 if split == "train" else 64
    if len(blocks) != expected:
        raise RuntimeError(f"F0 {split} anchor block count differs from the protocol")

    store = AtomicF0AnchorStore(runtime.binding)
    result: list[F0AnchorBlock] = []
    for block in blocks:
        runtime.binding.validate_current(component=renderer)
        recovered = store.load_or_none(block)
        if recovered is not None:
            score = _restore_anchor_score(
                runtime, recovered.trace, recovered.score_recovery
            )
            result.append(
                F0AnchorBlock(
                    block=recovered.block,
                    trace=recovered.trace,
                    score=score,
                )
            )
            continue
        initial = runtime.adapter.initial_state(
            prompts=block.record.prompt,
            prompt_ids=block.record.record_id,
            generation_seeds=block.generation_seed,
            checkpoint_hash=state_hash,
            split=split,
            height=int(contract["runtime"]["height"]),
            width=int(contract["runtime"]["width"]),
        )
        collector = runtime.adapter.make_collector(
            renderer,
            reference_renderer=reference,
            preserve_graph=False,
        )
        trace = collector.collect_f0_anchor_trace(initial, branch="anchor")
        if collector.last_stats != trace.stats:
            raise RuntimeError("F0 collector statistics changed after anchor collection")
        score = _score_terminal(
            runtime,
            trace.terminal_state,
            metric_branch="anchor",
            target_record_sha256=None,
        )
        anchor = F0AnchorBlock(
            block=block,
            trace=_move_value(trace, "cpu"),
            score=score,
        )
        store.save(anchor)
        result.append(anchor)
    return tuple(result)


def _moment_valid(clean: Tensor, guided: Tensor) -> bool:
    clean = clean.float()
    guided = guided.float()
    clean_mean = clean.mean(dim=(-2, -1))
    guided_mean = guided.mean(dim=(-2, -1))
    clean_centered = clean - clean_mean[..., None, None]
    guided_centered = guided - guided_mean[..., None, None]
    clean_energy = clean_centered.square().sum(dim=(-2, -1))
    guided_energy = guided_centered.square().sum(dim=(-2, -1))
    return bool(
        torch.isclose(guided_mean, clean_mean, rtol=2e-4, atol=2e-5).all()
        and torch.isclose(
            guided_energy, clean_energy, rtol=2e-4, atol=2e-5
        ).all()
    )


def _hard_cap_valid(
    renderer: EulerNativeFrameV1, output: EulerFrameOutput
) -> bool:
    diagnostics = output.diagnostics
    values = (
        diagnostics.angle,
        diagnostics.angle_cap_multiplier,
        diagnostics.scheduler_cap_multiplier,
        diagnostics.mapped_update_ratio,
    )
    if any(not torch.isfinite(value).all() for value in values):
        return False
    return bool(
        (diagnostics.angle_cap_multiplier > 0).all()
        and (diagnostics.angle_cap_multiplier <= 1.0 + 1e-5).all()
        and (diagnostics.scheduler_cap_multiplier > 0).all()
        and (diagnostics.scheduler_cap_multiplier <= 1.0 + 1e-5).all()
        and (diagnostics.angle <= renderer.theta_max + 1e-5).all()
        and (
            diagnostics.mapped_update_ratio
            <= renderer.max_update_ratio + 1e-5
        ).all()
    )


def _branch_audit(
    renderer: EulerNativeFrameV1,
    state: Any,
    frame: Any,
    suffix: CachedSuffixResult,
    score: F0ScoredImage,
) -> F0BranchAudit:
    action = suffix.applied_action
    output = renderer.apply_coefficients(frame, action)
    tensors = (
        action,
        suffix.selected_transition.rendered_transition,
        state.latents,
        output.guided_x0,
        output.residual,
        output.coefficients,
    )
    finite = all(torch.isfinite(value).all() for value in tensors)
    expected_transition = frame.native_update.to(output.residual.dtype) + (
        frame.kappa.to(device=output.residual.device, dtype=output.residual.dtype)
        .reshape(-1, 1, 1, 1)
        * output.residual
    )
    scheduler_valid = torch.allclose(
        suffix.selected_transition.rendered_transition.detach().float(),
        expected_transition.detach().float(),
        rtol=2e-3,
        atol=3e-5,
    )
    ratio = float(
        output.diagnostics.mapped_update_ratio.detach().double().cpu().reshape(())
    )
    return F0BranchAudit(
        action=action.detach().cpu(),
        rendered_transition=suffix.selected_transition.rendered_transition.detach().cpu(),
        frame_valid=bool(suffix.frame_valid.detach().cpu().reshape(())),
        finite=bool(finite),
        scheduler_valid=bool(scheduler_valid),
        moment_valid=_moment_valid(frame.clean_latent, output.guided_x0),
        hard_cap_valid=_hard_cap_valid(renderer, output),
        mapped_update_ratio=ratio,
        at_98pct_cap=bool(ratio >= 0.98 * renderer.max_update_ratio),
        score=score,
    )


def _construct_target_evaluations(
    runtime: Any,
    anchors: Sequence[F0AnchorBlock],
    *,
    split: str,
    reward_statistics: Mapping[str, Any],
) -> tuple[F0TargetEvaluation, ...]:
    """Construct every bounded target and score both common-prefix suffixes."""
    renderer = runtime.renderer
    reference = runtime.reference_renderer
    device = _renderer_device(renderer)
    target_store = AtomicF0TargetStore(runtime.binding)
    result: list[F0TargetEvaluation] = []
    for anchor in anchors:
        trace = _move_value(anchor.trace, device)
        collector = runtime.adapter.make_collector(
            renderer,
            reference_renderer=reference,
            preserve_graph=False,
        )
        terminal_receipt = TerminalAnchorRewardReceipt(
            reward=anchor.score.reward,
            decode_receipt=anchor.score.decode_receipt,
            reward_receipt=anchor.score.reward_receipt,
        )
        for decision_index in F0_DECISION_INDICES:
            cached = trace.cached_decisions[decision_index]
            stable_id = f0_target_stable_id(
                prompt_id=anchor.block.record.record_id,
                generation_seed=anchor.block.generation_seed,
                decision_index=decision_index,
                split=split,
            )
            cpu_target = target_store.load_or_none(stable_id)
            if cpu_target is None:
                target = construct_f0_target(
                    cached,
                    renderer,
                    reference,
                    runtime.adapter,
                    prompt_id=anchor.block.record.record_id,
                    source=anchor.block.record.source,
                    stratum=anchor.block.record.stratum,
                    fold=anchor.block.record.fold,
                    terminal_anchor_reward=terminal_receipt,
                    reward_location=float(reward_statistics["location"]),
                    reward_scale=float(reward_statistics["scale"]),
                )
                # The gradient query is already charged at this point. Publish its
                # detached target before either suffix so an interrupted run does
                # not lose paid supervision together with an in-memory object.
                cpu_target = target.to("cpu")
                target_store.save(cpu_target)
            expected_metadata = (
                anchor.block.record.record_id,
                anchor.block.generation_seed,
                decision_index,
                split,
                anchor.block.record.source,
                anchor.block.record.stratum,
                anchor.block.record.fold,
            )
            actual_metadata = (
                cpu_target.prompt_id,
                cpu_target.generation_seed,
                cpu_target.decision_index,
                cpu_target.split,
                cpu_target.source,
                cpu_target.stratum,
                cpu_target.fold,
            )
            if actual_metadata != expected_metadata or not _values_equal(
                cpu_target.state, cached.frame
            ):
                raise RuntimeError("restored F0 target differs from its C0 cached state")
            target = cpu_target.to(device)
            plus_action = renderer.deterministic_action_from_mean(target.plus_u)
            minus_action = renderer.deterministic_action_from_mean(target.minus_u)
            plus_suffix = collector.replay_f0_suffix(
                cached,
                plus_action,
                branch="plus",
            )
            plus_score = _score_terminal(
                runtime,
                plus_suffix.terminal_state,
                metric_branch="plus",
                target_record_sha256=cpu_target.record_sha256,
            )
            minus_suffix = collector.replay_f0_suffix(
                cached,
                minus_action,
                branch="minus",
            )
            minus_score = _score_terminal(
                runtime,
                minus_suffix.terminal_state,
                metric_branch="minus",
                target_record_sha256=cpu_target.record_sha256,
            )
            plus = _branch_audit(
                renderer,
                plus_suffix.terminal_state,
                target.state,
                plus_suffix,
                plus_score,
            )
            minus = _branch_audit(
                renderer,
                minus_suffix.terminal_state,
                target.state,
                minus_suffix,
                minus_score,
            )
            result.append(F0TargetEvaluation(cpu_target, plus, minus))
        del trace
    rows = validate_f0_rows((item.target for item in result), split=split)
    if len(rows) != len(result):
        raise RuntimeError("F0 target validation changed the state count")
    return tuple(result)


def _fit_crossfolds(
    runtime: Any, targets: Sequence[F0TargetRow]
) -> dict[int, F0FitResult]:
    splits: tuple[F0FoldSplit, ...] = f0_crossfit_splits(targets)
    if len(splits) != len(F0_FOLD_OPTIMIZATION_SEEDS):
        raise RuntimeError("F0 cross-fit count differs from the optimizer seeds")
    result: dict[int, F0FitResult] = {}
    for split, optimizer_seed in zip(splits, F0_FOLD_OPTIMIZATION_SEEDS):
        result[split.held_out_fold] = _train_fit(
            runtime,
            split.fit_rows,
            optimizer_seed=optimizer_seed,
            role=f"crossfit-fold-{split.held_out_fold}",
            held_out_fold=split.held_out_fold,
        )
    if set(result) != set(range(4)):
        raise RuntimeError("F0 did not produce all four independent cross-fits")
    return result


def _with_policy_checkpoint(
    cached: CachedEulerDecision, checkpoint_hash: str
) -> CachedEulerDecision:
    state = replace(cached.state, checkpoint_hash=checkpoint_hash)
    return replace(cached, state=state, checkpoint_hash=checkpoint_hash)


def _realize_targets(
    runtime: Any,
    anchors: Sequence[F0AnchorBlock],
    evaluations: Sequence[F0TargetEvaluation],
    *,
    renderers: Mapping[int | None, F0FitResult],
    phase: str,
) -> dict[str, F0BranchAudit]:
    """Score one held-out or final-teacher realization for every target state."""
    if phase not in {"train", "validation"}:
        raise ValueError("F0 realization phase must be train or validation")
    anchors_by_key = {
        (item.block.record.record_id, item.block.generation_seed): item
        for item in anchors
    }
    result: dict[str, F0BranchAudit] = {}
    branch = "out_of_fold_student" if phase == "train" else "final_teacher"
    for evaluation in evaluations:
        target = evaluation.target
        fit_key: int | None = target.fold if phase == "train" else None
        fit = renderers.get(fit_key)
        if fit is None:
            raise RuntimeError("F0 realization has no registered renderer")
        renderer = fit.renderer
        device = _renderer_device(renderer)
        row = target.to(device)
        anchor = anchors_by_key.get((target.prompt_id, target.generation_seed))
        if anchor is None:
            raise RuntimeError("F0 realization cannot find its strict-C0 anchor")
        cached = _move_value(
            anchor.trace.cached_decisions[target.decision_index], device
        )
        if cached.checkpoint_hash != runtime.initial_checkpoint_provenance.renderer_state_sha256:
            raise RuntimeError("F0 realization source is not the strict-C0 anchor")
        fit.checkpoint.validate_current()
        if (
            fit.held_out_fold != fit_key
            or fit.checkpoint.renderer_state_sha256 != fit.ema_renderer_state_sha256
            or module_state_sha256(renderer) != fit.ema_renderer_state_sha256
        ):
            raise RuntimeError("F0 realization renderer differs from its frozen fit")
        cached = _with_policy_checkpoint(cached, fit.ema_renderer_state_sha256)
        with torch.no_grad():
            mean = renderer.action_parameters(row.state)
            action = renderer.deterministic_action_from_mean(mean)
        collector = runtime.adapter.make_collector(
            renderer,
            reference_renderer=runtime.reference_renderer,
            preserve_graph=False,
        )
        suffix = collector.replay_f0_suffix(cached, action, branch=branch)
        score = _score_terminal(
            runtime,
            suffix.terminal_state,
            metric_branch=branch,
            target_record_sha256=target.record_sha256,
        )
        result[target.stable_id] = _branch_audit(
            renderer,
            suffix.terminal_state,
            row.state,
            suffix,
            score,
        )
    if len(result) != len(evaluations):
        raise RuntimeError("F0 realization records are missing or duplicated")
    return result


def _anchor_noop_violation(trace: F0AnchorTrace) -> bool:
    decisions = [
        transition
        for transition in trace.transitions
        if transition.step_index in F0_DECISION_INDICES
    ]
    return bool(
        len(decisions) != len(F0_DECISION_INDICES)
        or any(
            torch.count_nonzero(transition.action).item() != 0
            or not torch.equal(
                transition.native_transition, transition.rendered_transition
            )
            for transition in decisions
        )
    )


def _target_finite(target: F0TargetRow) -> bool:
    values = (
        target.anchor_u,
        target.plus_u,
        target.minus_u,
        target.reward_gradient,
        target.plus_transition,
        target.minus_transition,
        target.reference_u,
        target.state.sample,
        target.state.clean_latent,
        target.state.native_update,
    )
    return all(torch.isfinite(value).all() for value in values)


def _metric_rows(
    anchors: Sequence[F0AnchorBlock],
    evaluations: Sequence[F0TargetEvaluation],
    realized: Mapping[str, F0BranchAudit],
    *,
    phase: str,
) -> tuple[dict[str, Any], ...]:
    """Join immutable targets, independent branches, and their ledger receipts."""
    if phase not in {"train", "validation"}:
        raise ValueError("F0 metric rows require train or validation")
    anchors_by_key = {
        (item.block.record.record_id, item.block.generation_seed): item
        for item in anchors
    }
    realization_field = (
        "out_of_fold_student" if phase == "train" else "final_teacher"
    )
    realization_pixel_prefix = (
        "out_of_fold" if phase == "train" else "final_teacher"
    )
    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        target = evaluation.target
        anchor = anchors_by_key.get((target.prompt_id, target.generation_seed))
        realization = realized.get(target.stable_id)
        if anchor is None or realization is None:
            raise RuntimeError("F0 metric join is missing an anchor or realization")
        plus_matches = torch.allclose(
            evaluation.plus.rendered_transition.float(),
            target.plus_transition.float(),
            rtol=2e-3,
            atol=3e-5,
        )
        minus_matches = torch.allclose(
            evaluation.minus.rendered_transition.float(),
            target.minus_transition.float(),
            rtol=2e-3,
            atol=3e-5,
        )
        violations = {
            "no_op_parity": _anchor_noop_violation(anchor.trace),
            "scheduler_parity": not bool(
                plus_matches
                and minus_matches
                and evaluation.plus.scheduler_valid
                and evaluation.minus.scheduler_valid
                and realization.scheduler_valid
            ),
            "finite_value": not bool(
                _target_finite(target)
                and evaluation.plus.finite
                and evaluation.minus.finite
                and realization.finite
            ),
            "moment": not bool(
                evaluation.plus.moment_valid
                and evaluation.minus.moment_valid
                and realization.moment_valid
            ),
            "hard_cap": not bool(
                evaluation.plus.hard_cap_valid
                and evaluation.minus.hard_cap_valid
                and realization.hard_cap_valid
            ),
        }
        anchor_score = anchor.score
        plus_score = evaluation.plus.score
        minus_score = evaluation.minus.score
        realized_score = realization.score
        row = {
            "schema": F0_METRIC_ROW_SCHEMA,
            "phase": phase,
            "prompt_id": target.prompt_id,
            "generation_seed": target.generation_seed,
            "decision_index": target.decision_index,
            "provenance": {
                "target_record_sha256": target.record_sha256,
                "pixel_summary_schema": F0_PIXEL_SUMMARY_SCHEMA,
                "held_out_fold": target.fold,
                "branches": {
                    "anchor": _score_provenance(
                        anchor_score, reuse_scope="shared_prompt_seed"
                    ),
                    "plus": _score_provenance(
                        plus_score, reuse_scope="forbidden"
                    ),
                    "minus": _score_provenance(
                        minus_score, reuse_scope="forbidden"
                    ),
                    "realized": _score_provenance(
                        realized_score, reuse_scope="forbidden"
                    ),
                },
            },
            "valid_nonzero_gradient": bool(
                target.valid
                and evaluation.plus.frame_valid
                and evaluation.minus.frame_valid
                and realization.frame_valid
            ),
            "violations": violations,
            "reward_select": {
                "anchor": _score_value(anchor_score.reward, label="anchor reward"),
                "plus": _score_value(plus_score.reward, label="plus reward"),
                "minus": _score_value(minus_score.reward, label="minus reward"),
                realization_field: _score_value(
                    realized_score.reward, label="realized reward"
                ),
            },
            "topiq_nr": {
                "anchor": _score_value(anchor_score.topiq_nr, label="anchor TOPIQ"),
                "plus": _score_value(plus_score.topiq_nr, label="plus TOPIQ"),
                "minus": _score_value(minus_score.topiq_nr, label="minus TOPIQ"),
                realization_field: _score_value(
                    realized_score.topiq_nr, label="realized TOPIQ"
                ),
            },
            "pixel": {
                **{
                    f"anchor_{name}": value
                    for name, value in anchor_score.pixel.items()
                },
                **{
                    f"plus_{name}": value
                    for name, value in plus_score.pixel.items()
                },
                **{
                    f"minus_{name}": value
                    for name, value in minus_score.pixel.items()
                },
                **{
                    f"{realization_pixel_prefix}_{name}": value
                    for name, value in realized_score.pixel.items()
                },
            },
            "target_cap": {
                "plus_at_98pct": evaluation.plus.at_98pct_cap,
                "minus_at_98pct": evaluation.minus.at_98pct_cap,
            },
        }
        rows.append(row)
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row["prompt_id"],
                row["generation_seed"],
                row["decision_index"],
            ),
        )
    )


def _phase_gate(
    summary: Mapping[str, Any], metrics: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "passed": bool(summary["passed"]),
        "prompt_count": int(summary["prompt_count"]),
        "state_count": int(summary["state_count"]),
        "safety_violations": int(summary["safety_violations"]),
        "metrics": dict(metrics),
    }


def _require_ledger_usage(runtime: Any, expected: Mapping[str, int]) -> dict[str, Any]:
    summary = runtime.ledger.summary()
    if summary.get("reserved") != dict(expected):
        raise RuntimeError(
            "F0 ledger usage differs from the completed registered phase"
        )
    if summary.get("unfinished_reservations") != 0:
        raise RuntimeError("completed F0 phase has unfinished ledger reservations")
    return summary


def _teacher_state(
    binding: Any, fit: F0FitResult
) -> dict[str, Any]:
    fit.checkpoint.validate_current()
    return {
        "schema": OPSD_TEACHER_STATE_SCHEMA,
        "status": "frozen",
        "role": "T_OPSD",
        "f0_run_contract_sha256": binding.contract_hash,
        "renderer_state_sha256": fit.ema_renderer_state_sha256,
        "checkpoint": {
            "path": fit.checkpoint.path,
            "sha256": fit.checkpoint.sha256,
            "bytes": fit.checkpoint.bytes,
        },
    }


def _failure_result(
    binding: Any,
    *,
    stage: str,
    reason: str,
    registration: Mapping[str, Any],
    runtime: Any,
    train_gate: Mapping[str, Any] | None = None,
    validation_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": F0_FAILURE_SCHEMA,
        "run_id": binding.contract["run_id"],
        "method": "f0",
        "run_contract_sha256": binding.contract_hash,
        "status": "screen_failed",
        "stage": stage,
        "reason": reason,
        "screen_registration": dict(registration),
        "train_gate": None if train_gate is None else dict(train_gate),
        "validation_gate": (
            None if validation_gate is None else dict(validation_gate)
        ),
        "ledger": runtime.ledger.summary(),
        "runtime": dict(runtime.provenance),
        "benchmark_status": "not_applicable",
    }
    output = Path(binding.contract["paths"]["output_dir"])
    descriptor = _write_json_once(output / F0_FAILURE_FILENAME, payload)
    result = dict(payload)
    result["failure_artifact"] = descriptor
    return result


def _failure_file_descriptor(path: Path) -> dict[str, Any] | None:
    """Describe an existing ordinary failure-side artifact without masking an error."""
    try:
        if os.path.lexists(path):
            if path.is_symlink() or not path.is_file():
                return {"path": str(path), "invalid": True}
            return _file_descriptor(path)
    except BaseException as exc:
        return {
            "path": str(path),
            "unreadable": f"{type(exc).__module__}.{type(exc).__qualname__}: {exc}",
        }
    return None


def _json_safe_mapping(value: Any) -> dict[str, Any] | None:
    """Copy a runtime mapping only when it is safe to embed in JSON evidence."""
    if not isinstance(value, Mapping):
        return None
    try:
        normalized = json.loads(
            json.dumps(
                dict(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError):
        return None
    return normalized if isinstance(normalized, dict) else None


def _capture_runtime_failure_metadata(runtime: Any) -> dict[str, Any]:
    """Take a best-effort immutable snapshot before runtime cleanup runs."""
    captured: dict[str, Any] = {}
    try:
        ledger = getattr(runtime, "ledger", None)
        summary = ledger.summary() if ledger is not None else None
    except BaseException:
        summary = None
    normalized_summary = _json_safe_mapping(summary)
    if normalized_summary is not None:
        captured["ledger_summary"] = normalized_summary
    try:
        provenance = getattr(runtime, "provenance", None)
    except BaseException:
        provenance = None
    normalized_provenance = _json_safe_mapping(provenance)
    if normalized_provenance is not None:
        captured["runtime"] = normalized_provenance
    return captured


def _attach_runtime_failure_metadata(error: BaseException, runtime: Any) -> None:
    """Attach a serializable snapshot without allowing diagnostics to mask errors."""
    try:
        metadata = _capture_runtime_failure_metadata(runtime)
    except BaseException:
        # Diagnostics are strictly optional and must not alter the failure
        # semantics if a runtime exposes a hostile or partially torn-down
        # property.
        return
    if not metadata:
        return
    try:
        setattr(error, "_f0_runtime_failure_metadata", metadata)
    except BaseException:
        # Some third-party exceptions disallow custom attributes.  The failure
        # itself remains authoritative even when optional diagnostics are lost.
        pass


@contextmanager
def _f0_runtime_context(runtime_builder: Callable[..., Any], binding: Any):
    """Expose runtime diagnostics while preserving the body exception.

    The snapshot is taken inside the runtime context, before ``__exit__`` can
    release model resources or raise a cleanup error.
    """
    active_runtime: Any = None
    try:
        with runtime_builder(binding) as runtime:
            active_runtime = runtime
            try:
                yield runtime
            except BaseException as error:
                _attach_runtime_failure_metadata(error, runtime)
                raise
    except BaseException as error:
        # A context manager is allowed to raise from cleanup.  Capture the
        # latest runtime snapshot as well; the original body exception remains
        # available through Python's exception context chain.
        if active_runtime is not None:
            _attach_runtime_failure_metadata(error, active_runtime)
        raise


def _unexpected_failure_result(
    binding: Any,
    contract: Mapping[str, Any],
    *,
    registration: Mapping[str, Any] | None,
    stage: str,
    error: BaseException,
    runtime_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish a contract-bound artifact for failures outside a phase gate.

    The artifact is intentionally written before the original exception is
    re-raised.  It records only immutable provenance and file descriptors; it
    never claims that a partially completed phase passed.
    """
    output_dir = Path(contract["paths"]["output_dir"])
    ledger_path = Path(contract["paths"]["ledger_path"])
    seal_path = ledger_path.with_name(ledger_path.name + ".seal")
    payload: dict[str, Any] = {
        "schema": F0_FAILURE_SCHEMA,
        "run_id": contract["run_id"],
        "method": "f0",
        "status": "screen_failed",
        "stage": stage,
        "reason": "uncaught F0 execution failure",
        "error": {
            "type": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": str(error),
        },
        "code_commit": contract["code_commit"],
        "run_contract_sha256": binding.contract_hash,
        "screen_registration": (
            None if registration is None else dict(registration)
        ),
        "ledger": _failure_file_descriptor(ledger_path),
        "ledger_seal": _failure_file_descriptor(seal_path),
        "benchmark_status": "not_applicable",
    }
    if isinstance(runtime_metadata, Mapping):
        # Keep the file descriptor fields stable while adding optional live
        # runtime diagnostics captured before model cleanup.
        summary = _json_safe_mapping(runtime_metadata.get("ledger_summary"))
        provenance = _json_safe_mapping(runtime_metadata.get("runtime"))
        if summary is not None:
            payload["ledger_summary"] = summary
        if provenance is not None:
            payload["runtime"] = provenance
    path = output_dir / F0_FAILURE_FILENAME
    expected_bytes = _canonical_json(payload)
    expected_sha256 = hashlib.sha256(expected_bytes).hexdigest()
    try:
        descriptor = _write_json_once(path, payload)
    except FileExistsError:
        # A previous attempt may already have published a valid failure.  Do
        # not replace it with a second, potentially different exception.  Only
        # byte-identical evidence is safely reusable.
        descriptor = _failure_file_descriptor(path)
        if (
            descriptor is None
            or descriptor.get("invalid")
            or descriptor.get("sha256") != expected_sha256
            or descriptor.get("bytes") != len(expected_bytes)
        ):
            raise
    result = dict(payload)
    result["failure_artifact"] = descriptor
    return result


def _read_fit_update(
    path: Path,
    *,
    binding: Any,
    role: str,
    held_out_fold: int | None,
    optimizer_seed: int,
    update_index: int,
    stable_ids: Sequence[str],
    progress_path: Path,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("F0 fit update record is missing or not an ordinary file")
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("F0 fit update record is unreadable") from exc
    if not isinstance(value, Mapping) or payload != _canonical_json(value):
        raise ValueError("F0 fit update record is not canonical JSON")
    required = {
        "schema",
        "run_contract_sha256",
        "fit_role",
        "held_out_fold",
        "optimizer_seed",
        "update_index",
        "targets",
        "optimizer",
        "progress_checkpoint",
    }
    optimizer = value.get("optimizer")
    progress_value = value.get("progress_checkpoint")
    if not isinstance(progress_value, Mapping):
        raise ValueError("F0 fit update checkpoint provenance is missing")
    try:
        progress_checkpoint = CheckpointProvenance(**dict(progress_value))
    except (TypeError, ValueError) as exc:
        raise ValueError("F0 fit update checkpoint provenance is invalid") from exc
    if progress_checkpoint.path != str(progress_path):
        raise ValueError("F0 fit update names the wrong progress checkpoint")
    progress_checkpoint.validate_current()
    if (
        set(value) != required
        or value.get("schema") != "repldm.renderer_f0_update.v2"
        or value.get("run_contract_sha256") != binding.contract_hash
        or value.get("fit_role") != role
        or value.get("held_out_fold") != held_out_fold
        or value.get("optimizer_seed") != optimizer_seed
        or value.get("update_index") != update_index
        or value.get("targets") != list(stable_ids)
        or not isinstance(optimizer, Mapping)
        or set(optimizer) != {"step", "loss", "gradient_norm"}
        or optimizer.get("step") != update_index + 1
        or value.get("progress_checkpoint") != asdict(progress_checkpoint)
    ):
        raise ValueError("F0 fit update record differs from its registered step")
    for field in ("loss", "gradient_norm"):
        metric = optimizer.get(field)
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            raise ValueError("F0 fit update metric is not finite")
        if not math.isfinite(float(metric)):
            raise ValueError("F0 fit update metric is not finite")
    return dict(value)


def _fit_ledger_step_count(
    runtime: Any, *, role: str, optimizer_seed: int
) -> int:
    """Require one successful, contiguous optimizer receipt per persisted step."""
    records = runtime.ledger.verified_records()
    reservations = {
        row["reservation_id"]: row
        for row in records
        if row.get("type") == "reservation"
        and row.get("kind") == "optimizer_step"
        and isinstance(row.get("metadata"), Mapping)
        and isinstance(row["metadata"].get("action"), Mapping)
        and row["metadata"]["action"].get("fit_role") == role
    }
    receipts = {
        row["reservation_id"]: row
        for row in records
        if row.get("type") == "receipt" and row.get("reservation_id") in reservations
    }
    if any(
        type(row.get("amount")) is not int or row.get("amount") != 1
        for row in (*reservations.values(), *receipts.values())
    ):
        raise RuntimeError("F0 optimizer reservations and receipts must have amount one")
    if set(receipts) != set(reservations) or any(
        row.get("success") is not True for row in receipts.values()
    ):
        raise RuntimeError("F0 fit has a failed or interrupted optimizer reservation")
    steps: set[int] = set()
    for reservation_id, reservation in reservations.items():
        metadata = reservation["metadata"]
        action = metadata["action"]
        step = metadata.get("step")
        receipt = receipts[reservation_id]
        result = receipt.get("result")
        if (
            type(step) is not int
            or step < 0
            or metadata.get("prompt") != f"{role}-update-{step:03d}"
            or action.get("optimizer_seed") != optimizer_seed
            or not isinstance(action.get("target_stable_ids"), list)
            or len(action["target_stable_ids"]) != F0_FIT_BATCH_SIZE
            or not isinstance(result, Mapping)
            or result.get("scalar_or_gradient") != "optimizer_state"
            or result.get("failure") is not None
        ):
            raise RuntimeError("F0 optimizer receipt differs from its fit step")
        steps.add(step)
    if steps != set(range(len(steps))):
        raise RuntimeError("F0 optimizer receipts are not one contiguous fit prefix")
    return len(steps)


def _load_or_save_f0_checkpoint(
    runtime: Any,
    renderer: torch.nn.Module,
    *,
    filename: str,
    step: int,
    role: str,
) -> CheckpointProvenance:
    """Publish a final EMA once or authenticate the identical prior commit."""
    path = Path(runtime.binding.contract["paths"]["checkpoint_dir"]) / filename
    expected_hash = module_state_sha256(renderer)
    if not os.path.lexists(path):
        return _save_renderer_checkpoint(
            runtime, renderer, filename=filename, step=step, role=role
        )
    if path.is_symlink() or not path.is_file():
        raise ValueError("F0 renderer checkpoint is not an ordinary file")
    probe = copy.deepcopy(renderer)
    payload = load_checkpoint(
        path,
        model=probe,
        expected_contract=runtime.binding.contract,
        restore_rng=False,
        authorization_binding=runtime.binding,
        require_authorization=True,
    )
    extra = payload.get("extra")
    if (
        payload.get("step") != step
        or not isinstance(extra, Mapping)
        or extra.get("role") != role
        or extra.get("renderer_state_sha256") != expected_hash
        or module_state_sha256(probe) != expected_hash
    ):
        raise ValueError("existing F0 renderer checkpoint differs from the frozen fit")
    return CheckpointProvenance.capture(
        path, renderer_state_sha256=expected_hash
    )


def _train_fit(
    runtime: Any,
    values: Sequence[F0TargetRow],
    *,
    optimizer_seed: int,
    role: str,
    held_out_fold: int | None,
) -> F0FitResult:
    """Fit one independent C0 copy and freeze its final EMA for realization."""
    binding = runtime.binding
    binding.validate_current(component=runtime.renderer)
    if not isinstance(role, str) or not role:
        raise ValueError("F0 fit role must be a non-empty string")
    if held_out_fold is not None and (
        type(held_out_fold) is not int or held_out_fold not in range(4)
    ):
        raise ValueError("F0 held-out fold must be 0..3 or None")

    renderer = copy.deepcopy(runtime.renderer)
    for parameter in renderer.parameters():
        parameter.requires_grad_(True)
    renderer.train()
    initial_hash = module_state_sha256(renderer)
    if initial_hash != runtime.initial_checkpoint_provenance.renderer_state_sha256:
        raise RuntimeError("F0 fit did not start from an independent C0 copy")

    optimizer_config = binding.contract["optimizer"]
    trainer = RendererTrainer(
        renderer,
        _make_optimizer(renderer, optimizer_config),
        contract=binding.contract,
        authorization_binding=binding,
        operation_executor=runtime.operation_executor,
        grad_norm_cap=float(optimizer_config["gradient_norm_cap"]),
        ema_decay=float(optimizer_config["ema_decay"]),
    )
    batches = f0_fit_batches(values, optimizer_seed=optimizer_seed)
    output_root = Path(binding.contract["paths"]["output_dir"])
    progress_root = (
        Path(binding.contract["paths"]["checkpoint_dir"])
        / "f0-fit-progress"
        / role
    )
    progress_paths = tuple(
        progress_root / f"step-{step:03d}.pt"
        for step in range(1, F0_FIT_STEPS + 1)
    )
    present = [os.path.lexists(path) for path in progress_paths]
    if any(present[index] and not all(present[:index]) for index in range(len(present))):
        raise RuntimeError("F0 fit progress checkpoints are not contiguous")
    completed = sum(present)
    for update_index in range(completed, F0_FIT_STEPS):
        record_path = (
            output_root / "f0-updates" / role / f"update-{update_index:03d}.json"
        )
        if os.path.lexists(record_path):
            raise RuntimeError("F0 fit update record exists without its checkpoint")
    if _fit_ledger_step_count(
        runtime, role=role, optimizer_seed=optimizer_seed
    ) != completed:
        raise RuntimeError("F0 fit checkpoint and optimizer ledger progress differ")
    if completed:
        payload = trainer.load(str(progress_paths[completed - 1]))
        expected_ids = [row.stable_id for row in batches[completed - 1]]
        progress = payload.get("extra", {}).get("f0_fit_progress")
        expected_progress = {
            "schema": "repldm.renderer_f0_fit_progress.v1",
            "fit_role": role,
            "held_out_fold": held_out_fold,
            "optimizer_seed": optimizer_seed,
            "update_index": completed - 1,
            "targets": expected_ids,
        }
        if (
            not isinstance(progress, Mapping)
            or set(progress) != {*expected_progress, "optimizer"}
            or any(progress.get(key) != value for key, value in expected_progress.items())
            or not isinstance(progress.get("optimizer"), Mapping)
            or set(progress["optimizer"]) != {"step", "loss", "gradient_norm"}
            or progress["optimizer"].get("step") != completed
        ):
            raise ValueError("F0 fit progress checkpoint metadata is mismatched")
        if trainer.step != completed:
            raise RuntimeError("F0 fit progress restored the wrong trainer step")
        latest_record_path = (
            output_root / "f0-updates" / role / f"update-{completed - 1:03d}.json"
        )
        if not os.path.lexists(latest_record_path):
            progress_checkpoint = CheckpointProvenance.capture(
                progress_paths[completed - 1],
                renderer_state_sha256=module_state_sha256(renderer),
            )
            recovered_record = {
                "schema": "repldm.renderer_f0_update.v2",
                "run_contract_sha256": binding.contract_hash,
                "fit_role": role,
                "held_out_fold": held_out_fold,
                "optimizer_seed": optimizer_seed,
                "update_index": completed - 1,
                "targets": expected_ids,
                "optimizer": dict(progress["optimizer"]),
                "progress_checkpoint": asdict(progress_checkpoint),
            }
            _write_json_once(latest_record_path, recovered_record)

    device = next(renderer.parameters()).device
    update_rows: list[Mapping[str, Any]] = []
    for update_index in range(completed):
        batch = batches[update_index]
        stable_ids = [row.stable_id for row in batch]
        record_path = (
            output_root / "f0-updates" / role / f"update-{update_index:03d}.json"
        )
        _read_fit_update(
            record_path,
            binding=binding,
            role=role,
            held_out_fold=held_out_fold,
            optimizer_seed=optimizer_seed,
            update_index=update_index,
            stable_ids=stable_ids,
            progress_path=progress_paths[update_index],
        )
        update_rows.append(_file_descriptor(record_path))

    for update_index, batch in enumerate(batches[completed:], start=completed):
        device_batch = tuple(row.to(device) for row in batch)
        stable_ids = [row.stable_id for row in batch]

        def loss_fn(batch_values: tuple[F0TargetRow, ...] = device_batch):
            return f0_objective(
                renderer,
                runtime.reference_renderer,
                batch_values,
                branch_coefficient=1.0,
                anchor_weight=0.10,
            )

        context = trainer.optimizer_context(
            split="train",
            batch_id=f"{role}-update-{update_index:03d}",
            action={
                "fit_role": role,
                "optimizer_seed": optimizer_seed,
                "target_stable_ids": stable_ids,
            },
            prefix=8,
        )
        update: UpdateRecord = trainer.update(loss_fn, operation_context=context)
        record = {
            "schema": "repldm.renderer_f0_update.v2",
            "run_contract_sha256": binding.contract_hash,
            "fit_role": role,
            "held_out_fold": held_out_fold,
            "optimizer_seed": optimizer_seed,
            "update_index": update_index,
            "targets": stable_ids,
            "optimizer": asdict(update),
        }
        progress_path = progress_paths[update_index]
        trainer.save(
            str(progress_path),
            extra={
                "f0_fit_progress": {
                    "schema": "repldm.renderer_f0_fit_progress.v1",
                    "fit_role": role,
                    "held_out_fold": held_out_fold,
                    "optimizer_seed": optimizer_seed,
                    "update_index": update_index,
                    "targets": stable_ids,
                    "optimizer": asdict(update),
                }
            },
        )
        progress_checkpoint = CheckpointProvenance.capture(
            progress_path,
            renderer_state_sha256=module_state_sha256(renderer),
        )
        record["progress_checkpoint"] = asdict(progress_checkpoint)
        path = output_root / "f0-updates" / role / f"update-{update_index:03d}.json"
        _write_json_once(path, record)
        binding.validate_current(component=renderer)
        update_rows.append(_file_descriptor(path))

    if trainer.step != F0_FIT_STEPS:
        raise RuntimeError("F0 fit did not execute exactly 200 optimizer updates")
    raw_hash = module_state_sha256(renderer)
    ema_renderer = copy.deepcopy(renderer)
    _copy_ema(trainer, ema_renderer)
    ema_hash = module_state_sha256(ema_renderer)
    checkpoint = _load_or_save_f0_checkpoint(
        runtime,
        ema_renderer,
        filename=f"{role}-ema-step-{F0_FIT_STEPS:03d}.pt",
        step=F0_FIT_STEPS,
        role=role,
    )
    return F0FitResult(
        role=role,
        optimizer_seed=optimizer_seed,
        held_out_fold=held_out_fold,
        renderer=ema_renderer,
        checkpoint=checkpoint,
        updates=tuple(update_rows),
        raw_renderer_state_sha256=raw_hash,
        ema_renderer_state_sha256=ema_hash,
    )


def _run_f0_impl(
    binding: Any,
    *,
    runtime_builder: Callable[..., Any] | None = None,
) -> Mapping[str, Any]:
    """Execute the frozen OPSD-style teacher screen and publish auditable outputs."""
    contract = _require_f0_contract(binding)
    registration = _publish_screen_registration(binding, contract)
    if runtime_builder is None:
        from ..runtime_factory import build_training_runtime

        runtime_builder = build_training_runtime

    with _f0_runtime_context(runtime_builder, binding) as runtime:
        if runtime.binding is not binding:
            raise RuntimeError("F0 runtime replaced the authorization binding")
        if runtime.operation_executor is not runtime.adapter.operation_executor:
            raise RuntimeError("F0 adapter and runtime do not share one operation executor")
        if runtime.reward_statistics is not None:
            raise RuntimeError("F0 must construct, not consume, reward statistics")
        if any(
            value is not None
            for value in (
                runtime.f0_gate,
                runtime.opsd_teacher_state,
                runtime.training_cohort,
                runtime.teacher_renderer,
            )
        ):
            raise RuntimeError("F0 runtime cannot consume downstream training artifacts")

        records = _prompt_records(runtime.selected_records)
        train_anchors = _collect_anchor_blocks(runtime, records, split="train")
        reward_statistics = _reward_statistics_payload(train_anchors, contract)
        output_dir = Path(contract["paths"]["output_dir"])
        reward_statistics_binding = _write_json_once(
            output_dir / F0_REWARD_STATISTICS_FILENAME,
            reward_statistics,
        )
        if reward_statistics_binding["sha256"] != hashlib.sha256(
            _canonical_json(reward_statistics)
        ).hexdigest():
            raise RuntimeError("F0 reward statistics binding is inconsistent")

        train_evaluations = _construct_target_evaluations(
            runtime,
            train_anchors,
            split="train",
            reward_statistics=reward_statistics,
        )
        train_targets = validate_f0_rows(
            (item.target for item in train_evaluations), split="train"
        )
        crossfits = _fit_crossfolds(runtime, train_targets)
        train_realized = _realize_targets(
            runtime,
            train_anchors,
            train_evaluations,
            renderers=crossfits,
            phase="train",
        )
        train_rows = _metric_rows(
            train_anchors,
            train_evaluations,
            train_realized,
            phase="train",
        )
        train_metrics, train_summary = write_f0_phase_evidence(
            output_dir / F0_TRAIN_EVIDENCE_FILENAME,
            train_rows,
            phase="train",
            reward_scale=float(reward_statistics["scale"]),
            reward_statistics_sha256=str(reward_statistics_binding["sha256"]),
            f0_run_contract_sha256=binding.contract_hash,
            screen_registration_sha256=str(registration["sha256"]),
        )
        train_gate = _phase_gate(train_summary, train_metrics)
        _require_ledger_usage(runtime, F0_TRAIN_QUERY_USAGE)
        if train_summary.get("passed") is not True:
            return _failure_result(
                binding,
                stage="train_gate",
                reason="the frozen training engineering screen did not pass",
                registration=registration,
                runtime=runtime,
                train_gate=train_gate,
            )

        final_fit = _train_fit(
            runtime,
            train_targets,
            optimizer_seed=F0_FINAL_OPTIMIZATION_SEED,
            role="T_OPSD",
            held_out_fold=None,
        )
        validation_anchors = _collect_anchor_blocks(
            runtime, records, split="validation"
        )
        validation_evaluations = _construct_target_evaluations(
            runtime,
            validation_anchors,
            split="validation",
            reward_statistics=reward_statistics,
        )
        validation_realized = _realize_targets(
            runtime,
            validation_anchors,
            validation_evaluations,
            renderers={None: final_fit},
            phase="validation",
        )
        validation_rows = _metric_rows(
            validation_anchors,
            validation_evaluations,
            validation_realized,
            phase="validation",
        )
        validation_metrics, validation_summary = write_f0_phase_evidence(
            output_dir / F0_VALIDATION_EVIDENCE_FILENAME,
            validation_rows,
            phase="validation",
            reward_scale=float(reward_statistics["scale"]),
            reward_statistics_sha256=str(reward_statistics_binding["sha256"]),
            f0_run_contract_sha256=binding.contract_hash,
            screen_registration_sha256=str(registration["sha256"]),
        )
        validation_gate = _phase_gate(validation_summary, validation_metrics)
        ledger_summary = _require_ledger_usage(runtime, F0_QUERY_BUDGET)
        if validation_summary.get("passed") is not True:
            return _failure_result(
                binding,
                stage="validation_gate",
                reason="the frozen validation confirmation did not pass",
                registration=registration,
                runtime=runtime,
                train_gate=train_gate,
                validation_gate=validation_gate,
            )

        teacher_state = _teacher_state(binding, final_fit)
        teacher_state_binding = _write_json_once(
            output_dir / F0_TEACHER_STATE_FILENAME,
            teacher_state,
        )
        ledger_binding = _file_descriptor(Path(runtime.ledger.path))
        ledger_seal_binding = _file_descriptor(Path(runtime.ledger.seal_path))
        gate = {
            "schema": F0_GATE_SCHEMA,
            "status": "passed",
            "code_commit": contract["code_commit"],
            "f0_run_contract_sha256": binding.contract_hash,
            "selected_view_release_id": contract["selected_view_release_id"],
            "selected_view_manifest_sha256": contract[
                "selected_view_manifest_sha256"
            ],
            "selected_payload_sha256": contract[
                "selected_payload_manifest_sha256"
            ],
            "renderer_frame_contract_hash": contract[
                "renderer_frame_contract_hash"
            ],
            "calibration_hash": contract["calibration_hash"],
            "action_contract_hash": contract["action_contract_hash"],
            "initial_renderer_state_sha256": contract[
                "initial_renderer_state_sha256"
            ],
            "reward_config_sha256": contract["reward_config_sha256"],
            "reward_preprocess_sha256": contract["reward_preprocess_sha256"],
            "reward_asset_manifest_sha256": contract[
                "reward_asset_manifest_sha256"
            ],
            "witness_config_sha256": contract["witness_config_sha256"],
            "witness_preprocess_sha256": contract[
                "witness_preprocess_sha256"
            ],
            "witness_asset_manifest_sha256": contract[
                "witness_asset_manifest_sha256"
            ],
            "reward_statistics_sha256": reward_statistics_binding["sha256"],
            "opsd_teacher_state_manifest_sha256": teacher_state_binding["sha256"],
            "opsd_teacher_renderer_sha256": final_fit.ema_renderer_state_sha256,
            "crossfit_renderer_state_sha256": {
                str(fold): fit.ema_renderer_state_sha256
                for fold, fit in sorted(crossfits.items())
            },
            "train_gate": train_gate,
            "validation_gate": validation_gate,
            "screen_registration": dict(registration),
            "ledger": ledger_binding,
            "ledger_seal": ledger_seal_binding,
        }
        normalized_ledger, normalized_seal = validate_f0_run_outputs(
            (*train_rows, *validation_rows),
            ledger=ledger_binding,
            ledger_seal=ledger_seal_binding,
            gate=gate,
            contract=contract,
            f0_run_contract_sha256=binding.contract_hash,
        )
        if normalized_ledger != ledger_binding or normalized_seal != ledger_seal_binding:
            raise RuntimeError("F0 ledger bindings changed during reconciliation")
        gate_binding = _write_json_once(output_dir / F0_GATE_FILENAME, gate)
        result = {
            "schema": METHOD_RESULT_SCHEMA,
            "run_id": contract["run_id"],
            "method": "f0",
            "run_contract_sha256": binding.contract_hash,
            "status": "training_complete",
            "screen_status": "passed",
            "screen_registration": dict(registration),
            "reward_statistics": dict(reward_statistics_binding),
            "crossfits": [
                {
                    "held_out_fold": fold,
                    "optimizer_seed": fit.optimizer_seed,
                    "ema_renderer_state_sha256": fit.ema_renderer_state_sha256,
                    "checkpoint": asdict(fit.checkpoint),
                    "updates": list(fit.updates),
                }
                for fold, fit in sorted(crossfits.items())
            ],
            "teacher_state": dict(teacher_state_binding),
            "f0_gate": dict(gate_binding),
            "train_gate": train_gate,
            "validation_gate": validation_gate,
            "ledger": ledger_summary,
            "ledger_artifacts": {
                "records": ledger_binding,
                "seal": ledger_seal_binding,
            },
            "runtime": dict(runtime.provenance),
            "benchmark_status": "pending",
        }
        result_path = output_dir / F0_RESULT_FILENAME
        result["result_path"] = str(result_path)
        result["result_sha256"] = _result_sha256(result)
        _write_json_once(result_path, result)
        binding.validate_current(component=final_fit.renderer)
        return result


def run_f0(
    binding: Any,
    *,
    runtime_builder: Callable[..., Any] | None = None,
) -> Mapping[str, Any]:
    """Run F0 and persist a failure record for every unexpected exception."""
    # Contract validation itself is a prerequisite for writing a trustworthy
    # artifact.  If it fails, no run is authorized and no failure record is
    # claimed.  Once the contract is valid, all later failures are recorded.
    contract = _require_f0_contract(binding)
    registration: Mapping[str, Any] | None = None
    try:
        registration = _publish_screen_registration(binding, contract)
        return _run_f0_impl(binding, runtime_builder=runtime_builder)
    except BaseException as exc:
        stage = "registration" if registration is None else "execution"
        runtime_metadata = getattr(exc, "_f0_runtime_failure_metadata", None)
        try:
            _unexpected_failure_result(
                binding,
                contract,
                registration=registration,
                stage=stage,
                error=exc,
                runtime_metadata=runtime_metadata,
            )
        except BaseException as artifact_error:
            # Failure evidence is useful, but it must never hide the execution
            # error that caused the run to abort.  Attach the publication error
            # for debuggers and re-raise the original traceback below.
            try:
                setattr(exc, "_f0_failure_artifact_error", artifact_error)
            except BaseException:
                pass
        raise


__all__ = [
    "F0_DECISION_INDICES",
    "F0_FINAL_OPTIMIZATION_SEED",
    "F0_FIT_BATCH_SIZE",
    "F0_FIT_ORDER_SCHEMA",
    "F0_FIT_STEPS",
    "F0_FIXED_HYPERPARAMETERS",
    "F0_FOLD_OPTIMIZATION_SEEDS",
    "F0_QUERY_BUDGET",
    "F0_TRAINING_SEEDS",
    "F0_VALIDATION_SEEDS",
    "F0FitResult",
    "f0_fit_batches",
    "run_f0",
]
