"""Formal two-round runner shared by search-distill, DPO, and RL."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import copy
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence
import uuid

import torch
from torch import Tensor

from ..artifacts import module_state_sha256
from ..authorization import require_authorization_binding
from ..checkpoint import save_checkpoint
from ..f0_teacher import (
    _decode_target_with_receipt,
    _reward_target_with_receipt,
    _validate_decode_image,
    _validate_reward_scores,
)
from ..gates import (
    validate_f0_gate,
    validate_opsd_teacher_state,
    validate_reward_statistics,
    validate_training_cohort,
)
from ..operations import OperationContext, operation_output_sha256
from ..preferences import PreferenceLabelProvenance
from ..renderer import tensor_sha256
from ..storage import CheckpointProvenance
from ..trainer import RendererTrainer
from .common import (
    OpdTeacherLabel,
    ScoredRollout,
    _policy_transition,
    dpo_objective,
    opd_anchor_state_sha256,
    opd_objective,
    rl_objective,
    search_distill_objective,
)


PAIR_METHODS = frozenset({"search_distill", "dpo", "rl"})
ONLINE_METHODS = frozenset((*PAIR_METHODS, "opd"))
METHOD_RESULT_SCHEMA = "repldm.renderer_training_result.v1"
PROPOSAL_RNG_SCHEMA = "repldm.antithetic_proposal.v1"
BLOCK_ORDER_SCHEMA = "repldm.prompt_block_order.v1"
TRAIN_GENERATION_SEEDS = (2026090101, 2026090102)
VALIDATION_GENERATION_SEEDS = (2026090191, 2026090192)
OPTIMIZATION_SEEDS = (202609011, 202609012, 202609013)
DECISION_INDICES = (8, 24, 40)
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")

PAIR_QUERY_BUDGET = {
    "unet_forward": 42_240,
    "scheduler_step": 42_880,
    "vae_decode": 960,
    "reward_forward": 960,
    "label": 320,
    "optimizer_step": 64,
}

# OPD has one additional teacher rollout branch per prompt-seed block and
# records one detached teacher transition label at each of the three decisions.
# The scheduler count is intentionally separate from the U-Net count: every
# decision applies an extra teacher transition, while the first decision's
# observation is shared with the student prefix.
OPD_QUERY_BUDGET = {
    "unet_forward": 55_360,
    "scheduler_step": 56_320,
    "vae_decode": 1_280,
    "reward_forward": 1_280,
    "label": 960,
    "optimizer_step": 64,
}

COMMON_HYPERPARAMETERS = {
    "rounds": 2,
    "updates_per_round": 32,
    "blocks_per_update": 4,
    "training_generation_seeds": list(TRAIN_GENERATION_SEEDS),
    "validation_generation_seeds": list(VALIDATION_GENERATION_SEEDS),
    "decision_indices": list(DECISION_INDICES),
    "proposal_sigma": 0.25,
    "proposal_rng_schema": PROPOSAL_RNG_SCHEMA,
    "block_order_schema": BLOCK_ORDER_SCHEMA,
    "block_order_seed": 20260901,
    "tie_epsilon": 1e-6,
    "anchor_weight": 0.10,
    "dpo_beta": 0.10,
    "rl_clip_low": 0.8,
    "rl_clip_high": 1.2,
    "rl_kl_weight": 0.01,
    "behavior_update": "ema_after_each_update",
    "round_two_start": "round_one_ema_with_reset_adamw_state",
}


@dataclass(frozen=True)
class PromptRecord:
    record_id: str
    prompt: str
    split: str
    source: str
    stratum: str
    fold: int | None


@dataclass(frozen=True)
class PromptSeedBlock:
    record: PromptRecord
    generation_seed: int

    @property
    def stable_id(self) -> str:
        digest = hashlib.sha256(
            f"{self.record.record_id}\0{self.generation_seed}".encode("utf-8")
        ).hexdigest()[:16]
        readable = _SAFE_ID_RE.sub("-", self.record.record_id).strip("-._")[:40]
        return f"{readable or 'record'}-{self.generation_seed}-{digest}"


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("method result must contain finite JSON values") from exc


def _atomic_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    prepublish: Callable[[], None] | None = None,
) -> None:
    payload = _canonical(dict(value)) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace existing run artifact: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if prepublish is not None:
            prepublish()
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to replace existing run artifact: {path}"
            ) from exc
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _file_descriptor(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"run artifact is missing or not an ordinary file: {path}")
    payload = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _result_sha256(value: Mapping[str, Any]) -> str:
    """Hash a result envelope without the self-referential digest field.

    The digest is calculated over the exact canonical JSON payload that is
    written when ``result_sha256`` is omitted.  This keeps the result file
    self-describing while avoiding an impossible hash fixed point.
    """
    core = dict(value)
    core.pop("result_sha256", None)
    return hashlib.sha256(_canonical(core) + b"\n").hexdigest()


def _require_online_contract(binding: Any, method: str) -> Mapping[str, Any]:
    binding = require_authorization_binding(binding)
    binding.validate_current()
    contract = binding.contract
    if method not in ONLINE_METHODS or contract.get("method") != method:
        raise ValueError("online runner method differs from the authorized contract")
    expected_hyperparameters = {"method": method, **COMMON_HYPERPARAMETERS}
    if contract.get("method_hyperparameters") != expected_hyperparameters:
        raise ValueError("method hyperparameters differ from the frozen online protocol")
    expected_budget = OPD_QUERY_BUDGET if method == "opd" else PAIR_QUERY_BUDGET
    if contract.get("query_budget") != expected_budget:
        label = "OPD" if method == "opd" else "pair"
        raise ValueError(f"query budget differs from the frozen {label} allocation")
    if contract.get("seed") not in OPTIMIZATION_SEEDS:
        raise ValueError("optimization seed is not one of the three registered runs")
    if contract.get("selected_rows") != 96:
        raise ValueError("pair training requires exactly 64 train and 32 validation rows")
    if tuple(contract.get("decision_indices", ())) != DECISION_INDICES:
        raise ValueError("decision schedule differs from the pair protocol")
    return contract


def _require_pair_contract(binding: Any, method: str) -> Mapping[str, Any]:
    """Validate one of the sampled-pair arms (legacy public helper)."""
    if method not in PAIR_METHODS:
        raise ValueError("pair runner method differs from the authorized contract")
    return _require_online_contract(binding, method)


def _require_opd_contract(binding: Any) -> Mapping[str, Any]:
    """Validate the distinct external-teacher OPD allocation."""
    return _require_online_contract(binding, "opd")


def _require_runtime_gates(runtime: Any, contract: Mapping[str, Any]) -> None:
    """Consume and cross-check every artifact that authorizes a main arm."""
    statistics = validate_reward_statistics(runtime.reward_statistics, contract=contract)
    f0_gate, _evidence = validate_f0_gate(runtime.f0_gate, contract=contract)
    teacher_state, _checkpoint = validate_opsd_teacher_state(
        runtime.opsd_teacher_state, contract=contract
    )
    cohort = validate_training_cohort(runtime.training_cohort, contract=contract)
    if f0_gate["f0_run_contract_sha256"] != teacher_state["f0_run_contract_sha256"]:
        raise RuntimeError("F0 gate and frozen teacher do not come from the same F0 run")
    if f0_gate["reward_statistics_sha256"] != contract["reward_statistics_sha256"]:
        raise RuntimeError("F0 gate does not bind the shared reward statistics")
    if statistics["initial_renderer_state_sha256"] != contract[
        "initial_renderer_state_sha256"
    ]:
        raise RuntimeError("reward statistics do not come from the shared C0 renderer")
    if cohort["cohort_id"] != contract["cohort_id"]:
        raise RuntimeError("runtime cohort differs from the authorized cohort")


def _module_state_tensors(module: Any) -> tuple[Tensor, ...]:
    """Return state tensors without allowing teacher/storage aliasing."""
    try:
        state = module.state_dict()
    except (AttributeError, TypeError) as exc:
        raise TypeError("OPD teacher must expose a state_dict") from exc
    if not isinstance(state, Mapping):
        raise TypeError("OPD teacher state_dict must be a mapping")
    values = tuple(value for value in state.values() if isinstance(value, Tensor))
    if not values or len(values) != len(state):
        raise TypeError("OPD teacher state_dict must contain only tensors")
    return values


def _require_opd_teacher(runtime: Any, contract: Mapping[str, Any]) -> Any:
    """Authenticate the immutable external T_OPSD before any OPD query."""
    teacher = getattr(runtime, "teacher_renderer", None)
    renderer = getattr(runtime, "renderer", None)
    reference = getattr(runtime, "reference_renderer", None)
    if teacher is None:
        raise RuntimeError("OPD requires the frozen external T_OPSD teacher")
    if renderer is None or reference is None:
        raise RuntimeError("OPD runtime omitted the student/reference renderer")
    if teacher is renderer or teacher is reference:
        raise RuntimeError("OPD teacher must be a distinct renderer object")
    if type(teacher) is not type(renderer) or type(teacher) is not type(reference):
        raise TypeError("OPD teacher, student, and reference types must match")
    if any(module.training for module in teacher.modules()):
        raise ValueError("OPD teacher must remain in eval mode")
    teacher_parameters = tuple(teacher.parameters())
    if not teacher_parameters or any(parameter.requires_grad for parameter in teacher_parameters):
        raise ValueError("OPD teacher must be completely frozen")
    for name in ("frame_contract_hash", "calibration_hash"):
        expected = getattr(renderer, name, None)
        if getattr(teacher, name, None) != expected or getattr(reference, name, None) != expected:
            raise ValueError(f"OPD {name} differs between teacher and student")
    teacher_contract = getattr(teacher, "contract", None)
    renderer_contract = getattr(renderer, "contract", None)
    reference_contract = getattr(reference, "contract", None)
    if teacher_contract is None or renderer_contract is None or reference_contract is None:
        raise TypeError("OPD renderers must expose an action contract")
    def contract_payload(value: Any) -> Any:
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return to_dict()
        if isinstance(value, Mapping):
            return dict(value)
        raise TypeError("OPD renderers expose an unsupported action contract")

    expected_action = contract_payload(renderer_contract)
    if contract_payload(teacher_contract) != expected_action or contract_payload(reference_contract) != expected_action:
        raise ValueError("OPD teacher/action contracts differ")
    expected_hash = contract.get("opsd_teacher_renderer_sha256")
    teacher_hash = module_state_sha256(teacher)
    if teacher_hash != expected_hash:
        raise ValueError("OPD teacher state differs from the authorized T_OPSD hash")
    checkpoint = getattr(runtime, "teacher_checkpoint_provenance", None)
    if not isinstance(checkpoint, CheckpointProvenance):
        raise RuntimeError("OPD runtime omitted T_OPSD checkpoint provenance")
    if checkpoint.renderer_state_sha256 != teacher_hash:
        raise ValueError("OPD teacher checkpoint provenance has the wrong state hash")
    checkpoint.validate_current()
    # Validate the capability for the teacher as well as the initial student.
    runtime.binding.validate_component(teacher)
    renderer_tensors = _module_state_tensors(renderer)
    reference_tensors = _module_state_tensors(reference)
    teacher_tensors = _module_state_tensors(teacher)
    if len(teacher_tensors) != len(renderer_tensors) or len(reference_tensors) != len(renderer_tensors):
        raise ValueError("OPD renderer state schemas differ")
    if any(
        teacher_tensor.device != renderer_tensors[index].device
        or teacher_tensor.dtype != renderer_tensors[index].dtype
        for index, teacher_tensor in enumerate(teacher_tensors)
        if index < len(renderer_tensors)
    ):
        raise ValueError("OPD teacher tensors do not match the student device/dtype")
    if hasattr(renderer, "active_mask") and getattr(teacher, "active_mask", None) != renderer.active_mask:
        raise ValueError("OPD teacher active mask differs from the student")
    for teacher_tensor in teacher_tensors:
        for other_tensor in (*renderer_tensors, *reference_tensors):
            if teacher_tensor is other_tensor:
                raise RuntimeError("OPD teacher state aliases another renderer")
            if (
                teacher_tensor.numel()
                and other_tensor.numel()
                and teacher_tensor.untyped_storage().data_ptr()
                == other_tensor.untyped_storage().data_ptr()
            ):
                raise RuntimeError("OPD teacher storage aliases another renderer")
    return teacher


def _prompt_records(rows: Iterable[Mapping[str, Any]]) -> tuple[PromptRecord, ...]:
    records: list[PromptRecord] = []
    seen: set[str] = set()
    cells: dict[tuple[str, str, str], int] = {}
    folds: dict[tuple[str, str], set[int]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("selected data rows must be mappings")
        record_id = row.get("id")
        prompt = row.get("model_prompt")
        split = row.get("selected_split")
        source = row.get("source")
        stratum = row.get("stratum")
        fold = row.get("fold")
        if not isinstance(record_id, str) or not record_id or record_id in seen:
            raise ValueError("selected record IDs must be unique non-empty strings")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"selected row {record_id!r} has no model_prompt")
        if split not in {"train", "validation"}:
            raise ValueError(f"selected row {record_id!r} has an invalid split")
        if source not in {"four_k_lsdb", "pixverve_95k"}:
            raise ValueError(f"selected row {record_id!r} has an invalid source")
        if stratum not in {
            "nature", "urban", "people", "food", "artwork", "cgi",
            "animals", "architecture",
        }:
            raise ValueError(f"selected row {record_id!r} has an invalid stratum")
        if split == "train":
            if type(fold) is not int or fold not in range(4):
                raise ValueError(f"training row {record_id!r} has an invalid fold")
            key = (source, stratum)
            folds.setdefault(key, set()).add(fold)
        elif fold is not None:
            raise ValueError(f"validation row {record_id!r} must not have a fold")
        seen.add(record_id)
        cells[(source, stratum, split)] = cells.get((source, stratum, split), 0) + 1
        records.append(PromptRecord(record_id, prompt, split, source, stratum, fold))
    if len(records) != 96:
        raise ValueError(f"selected view must contain exactly 96 rows, got {len(records)}")
    for source in ("four_k_lsdb", "pixverve_95k"):
        for stratum in (
            "nature", "urban", "people", "food", "artwork", "cgi",
            "animals", "architecture",
        ):
            if cells.get((source, stratum, "train")) != 4:
                raise ValueError("selected view violates a train source-stratum quota")
            if cells.get((source, stratum, "validation")) != 2:
                raise ValueError("selected view violates a validation source-stratum quota")
            if folds.get((source, stratum)) != {0, 1, 2, 3}:
                raise ValueError("selected view does not assign one train row per fold")
    return tuple(records)


def _block_order_key(block: PromptSeedBlock) -> str:
    return hashlib.sha256(
        f"20260901\0{block.record.record_id}\0{block.generation_seed}".encode("utf-8")
    ).hexdigest()


def _blocks(
    records: Sequence[PromptRecord], *, split: str
) -> tuple[PromptSeedBlock, ...]:
    seeds = TRAIN_GENERATION_SEEDS if split == "train" else VALIDATION_GENERATION_SEEDS
    values = [
        PromptSeedBlock(record, seed)
        for record in records
        if record.split == split
        for seed in seeds
    ]
    expected = 128 if split == "train" else 64
    if len(values) != expected:
        raise ValueError(f"{split} block count must be {expected}")
    return tuple(sorted(values, key=_block_order_key))


def _proposal_noise(
    block: PromptSeedBlock,
    step_index: int,
    *,
    optimization_seed: int,
    active_mask: Sequence[bool],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if step_index not in DECISION_INDICES:
        raise ValueError("proposal noise requested outside a registered decision")
    if optimization_seed not in OPTIMIZATION_SEEDS:
        raise ValueError("proposal noise requires a registered optimization seed")
    mask = tuple(active_mask)
    if len(mask) != 6 or any(type(value) is not bool for value in mask):
        raise ValueError("proposal noise requires the frozen six-slot active mask")
    key = (
        f"{PROPOSAL_RNG_SCHEMA}\0{block.record.record_id}\0"
        f"{block.generation_seed}\0{optimization_seed}\0{step_index}"
    )
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    active = torch.randn(sum(mask), generator=generator, dtype=torch.float32)
    noise = torch.zeros(1, 6, dtype=torch.float32)
    noise[0, torch.tensor(mask, dtype=torch.bool)] = active
    return noise.to(device=device, dtype=dtype)


def _batches(values: Sequence[PromptSeedBlock]) -> tuple[tuple[PromptSeedBlock, ...], ...]:
    if len(values) != 128:
        raise ValueError("training permutation must contain 128 blocks")
    result = tuple(tuple(values[index : index + 4]) for index in range(0, 128, 4))
    if len(result) != 32 or any(len(batch) != 4 for batch in result):
        raise RuntimeError("failed to construct the frozen 32 x 4 update schedule")
    return result


def _freeze(module: torch.nn.Module) -> torch.nn.Module:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()
    return module


def _copy_ema(trainer: RendererTrainer, destination: torch.nn.Module) -> None:
    destination.load_state_dict(
        {
            name: value.detach().clone().to(
                device=destination.state_dict()[name].device,
                dtype=destination.state_dict()[name].dtype,
            )
            for name, value in trainer.ema_state.items()
        },
        strict=True,
    )
    _freeze(destination)


def _make_optimizer(
    renderer: torch.nn.Module, config: Mapping[str, Any]
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        renderer.parameters(),
        lr=float(config["learning_rate"]),
        betas=tuple(float(value) for value in config["betas"]),
        weight_decay=float(config["weight_decay"]),
    )


def _score_terminal(
    runtime: Any,
    state: Any,
) -> tuple[Tensor, Tensor]:
    """Decode and score one terminal state with an authenticated parent chain."""
    adapter = runtime.adapter
    if getattr(adapter, "operation_executor", None) is None:
        image = adapter.decode(state)
        _validate_decode_image(image, state, label="terminal image")
        reward = adapter.reward(state, image)
    else:
        image, decode_receipt = _decode_target_with_receipt(
            adapter,
            state,
            action_metadata=None,
        )
        reward, _reward_receipt = _reward_target_with_receipt(
            adapter,
            state,
            image,
            parent_receipt=decode_receipt,
            action_metadata=None,
        )
    _validate_decode_image(image, state, label="terminal image")
    _validate_reward_scores(reward, state, label="terminal reward")
    return image, reward


def _save_renderer_checkpoint(
    runtime: Any,
    renderer: torch.nn.Module,
    *,
    filename: str,
    step: int,
    role: str,
) -> CheckpointProvenance:
    contract = runtime.binding.contract
    path = Path(contract["paths"]["checkpoint_dir"]) / filename
    save_checkpoint(
        path,
        model=renderer,
        step=step,
        contract=contract,
        extra={"role": role, "renderer_state_sha256": module_state_sha256(renderer)},
        authorization_binding=runtime.binding,
        require_authorization=True,
    )
    return CheckpointProvenance.capture(
        path, renderer_state_sha256=module_state_sha256(renderer)
    )


def _score_collection(
    runtime: Any,
    collection: Any,
    *,
    rollout_id: str,
    ledger_label: bool = True,
) -> tuple[ScoredRollout, PreferenceLabelProvenance]:
    rewards: dict[str, Tensor] = {}
    image_hashes: dict[str, str] = {}
    for branch in ("plus", "minus", "anchor"):
        state = collection.terminal_states[branch]
        with torch.no_grad():
            image, reward = _score_terminal(runtime, state)
        image_hashes[branch] = tensor_sha256(image)
        rewards[branch] = reward.detach()
        collection.branches[branch].terminal_reward = reward.detach()
    if not isinstance(runtime.reward_statistics, Mapping):
        raise RuntimeError("pair training requires frozen reward statistics")
    contract = runtime.binding.contract
    if ledger_label:
        label_payload = runtime.adapter.preference_label(
            collection.terminal_states["plus"],
            rewards["plus"],
            rewards["minus"],
            rollout_id=rollout_id,
            reward_statistics=runtime.reward_statistics,
            reward_statistics_sha256=contract["reward_statistics_sha256"],
            plus_image_sha256=image_hashes["plus"],
            minus_image_sha256=image_hashes["minus"],
            reward_config_sha256=contract["reward_config_sha256"],
            reward_preprocess_sha256=contract["reward_preprocess_sha256"],
        )
        label = PreferenceLabelProvenance.from_mapping(label_payload)
    else:
        # OPD's budget reserves one logical label per decision for the
        # external teacher.  The sampled pair label is only a convenience for
        # the shared ScoredRollout container and must not consume that budget.
        label = PreferenceLabelProvenance.from_rewards(
            rollout_id=rollout_id,
            prompt_id=collection.terminal_states["plus"].conditioning.prompt_ids[0],
            generation_seed=collection.terminal_states["plus"].conditioning.generation_seeds[0],
            split=collection.terminal_states["plus"].split,
            plus_reward=float(rewards["plus"].detach().double().cpu()),
            minus_reward=float(rewards["minus"].detach().double().cpu()),
            reward_location=runtime.reward_statistics["location"],
            reward_scale=runtime.reward_statistics["scale"],
            plus_image_sha256=image_hashes["plus"],
            minus_image_sha256=image_hashes["minus"],
            reward_statistics_sha256=contract["reward_statistics_sha256"],
            reward_config_sha256=contract["reward_config_sha256"],
            reward_preprocess_sha256=contract["reward_preprocess_sha256"],
        )
    state = collection.terminal_states["plus"]
    expected = {
        "rollout_id": rollout_id,
        "prompt_id": state.conditioning.prompt_ids[0],
        "generation_seed": state.conditioning.generation_seeds[0],
        "split": state.split,
        "plus_image_sha256": image_hashes["plus"],
        "minus_image_sha256": image_hashes["minus"],
        "reward_statistics_sha256": contract["reward_statistics_sha256"],
        "reward_config_sha256": contract["reward_config_sha256"],
        "reward_preprocess_sha256": contract["reward_preprocess_sha256"],
    }
    for field, value in expected.items():
        if getattr(label, field) != value:
            raise RuntimeError(f"ledgered preference label has mismatched {field}")
    if (
        label.reward_location != float(runtime.reward_statistics["location"])
        or label.reward_scale != float(runtime.reward_statistics["scale"])
    ):
        raise RuntimeError("ledgered preference label changed the frozen reward statistics")
    return ScoredRollout(collection, rewards, label), label


def _collect_block(
    runtime: Any,
    behavior_renderer: Any,
    block: PromptSeedBlock,
    *,
    behavior_checkpoint: CheckpointProvenance,
    reference_checkpoint: CheckpointProvenance,
    storage_id: str,
    round_index: int,
    update_index: int | None,
) -> tuple[ScoredRollout, Mapping[str, Any], Mapping[str, Any]]:
    state_hash = module_state_sha256(behavior_renderer)
    if behavior_checkpoint.renderer_state_sha256 != state_hash:
        raise RuntimeError("behavior checkpoint does not match the collection policy")
    runtime.binding.validate_current(component=behavior_renderer)
    state = runtime.adapter.initial_state(
        prompts=block.record.prompt,
        prompt_ids=block.record.record_id,
        generation_seeds=block.generation_seed,
        checkpoint_hash=state_hash,
        split=block.record.split,
        height=int(runtime.binding.contract["runtime"]["height"]),
        width=int(runtime.binding.contract["runtime"]["width"]),
    )
    parameter = next(behavior_renderer.parameters())
    noise = {
        step: _proposal_noise(
            block,
            step,
            optimization_seed=int(runtime.binding.contract["seed"]),
            active_mask=behavior_renderer.active_mask,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        for step in DECISION_INDICES
    }
    collector = runtime.adapter.make_collector(
        behavior_renderer,
        reference_renderer=runtime.reference_renderer,
        preserve_graph=False,
    )
    collection = collector.collect(state, noise_by_decision=noise)
    scored, label = _score_collection(runtime, collection, rollout_id=storage_id)
    metadata = {
        "schema": "repldm.scored_pair_rollout.v1",
        "split": block.record.split,
        "prompt_id": block.record.record_id,
        "generation_seed": block.generation_seed,
        "source": block.record.source,
        "stratum": block.record.stratum,
        "fold": block.record.fold,
        "round": round_index,
        "update": update_index,
        "label": label.to_dict(),
        "label_receipt_output_sha256": label.sha256,
        "terminal_rewards": {
            branch: float(value.detach().double().cpu())
            for branch, value in scored.rewards.items()
        },
        "collector_stats": asdict(collector.last_stats),
        "proposal_rng_schema": PROPOSAL_RNG_SCHEMA,
    }
    published_manifest = runtime.rollout_store.save(
        storage_id,
        collection,
        behavior_checkpoint=behavior_checkpoint,
        reference_checkpoint=reference_checkpoint,
        metadata=metadata,
    )
    manifest_path = Path(runtime.rollout_store.root) / f"{storage_id}.json"
    if json.loads(manifest_path.read_text(encoding="utf-8")) != published_manifest:
        raise RuntimeError("published rollout manifest differs from the stored manifest")
    return scored, metadata, _file_descriptor(manifest_path)


def _opd_label_context(
    block: PromptSeedBlock,
    decision_index: int,
    target: Tensor,
    *,
    teacher_checkpoint_hash: str,
    student_state_hash: str,
) -> OperationContext:
    """Build provenance for one detached external-teacher transition label."""
    return OperationContext(
        split=block.record.split,
        prompt=block.record.record_id,
        seed=block.generation_seed,
        step=decision_index,
        prefix=8,
        branch="teacher",
        action={
            "kind": "opd_teacher_transition",
            "decision_index": decision_index,
            "target_sha256": tensor_sha256(target),
            "raw_target_sha256": tensor_sha256(target),
            "student_state_sha256": student_state_hash,
            "prompt_id": block.record.record_id,
            "prompt": block.record.prompt,
            "split": block.record.split,
            "seed": block.generation_seed,
        },
        checkpoint_hash=teacher_checkpoint_hash,
    )


def _record_opd_teacher_labels(
    runtime: Any,
    teacher_renderer: Any,
    block: PromptSeedBlock,
    scored: ScoredRollout,
    *,
    teacher_checkpoint_hash: str,
) -> tuple[OpdTeacherLabel, ...]:
    """Evaluate T_OPSD on exactly the detached student-anchor frames.

    The detached tensors are ledgered as logical labels and returned as typed
    records.  The optimizer later consumes those exact bytes, making paid
    teacher supervision independently auditable and preventing accidental
    reuse of a stale student's states.
    """
    anchor = scored.collection.branches.get("anchor")
    if anchor is None or len(anchor.transitions) != len(DECISION_INDICES):
        raise RuntimeError("OPD requires the three current student anchor states")
    executor = getattr(runtime, "operation_executor", None)
    records: list[OpdTeacherLabel] = []
    checkpoint = getattr(runtime, "teacher_checkpoint_provenance", None)
    teacher_checkpoint_file_hash = getattr(checkpoint, "sha256", None)
    if teacher_checkpoint_file_hash is not None and (
        not isinstance(teacher_checkpoint_file_hash, str)
        or len(teacher_checkpoint_file_hash) != 64
        or teacher_checkpoint_file_hash != teacher_checkpoint_file_hash.lower()
        or any(c not in "0123456789abcdef" for c in teacher_checkpoint_file_hash)
    ):
        raise RuntimeError("OPD teacher checkpoint provenance has an invalid file hash")
    for transition, decision_index in zip(anchor.transitions, DECISION_INDICES):
        if transition.step_index != decision_index:
            raise RuntimeError("OPD anchor decision schedule is stale")
        with torch.no_grad():
            target, _mean, valid = _policy_transition(teacher_renderer, transition)
        target = target.detach().clone()
        if target.ndim != 4 or not torch.isfinite(target).all():
            raise RuntimeError("OPD teacher produced a non-finite transition target")
        student_state_hash = opd_anchor_state_sha256(transition)
        context = _opd_label_context(
            block,
            decision_index,
            target,
            teacher_checkpoint_hash=teacher_checkpoint_hash,
            student_state_hash=student_state_hash,
        )
        reservation_id = None
        if executor is not None and callable(getattr(executor, "execute_with_receipt", None)):
            value, receipt = executor.execute_with_receipt(
                "label",
                context,
                lambda target=target: target,
                scalar_or_gradient="teacher_transition",
                output_hasher=operation_output_sha256,
            )
            if not isinstance(value, Tensor) or not torch.equal(value.detach(), target):
                raise RuntimeError("OPD label executor changed the teacher target")
            output_hash = receipt.output_hash
            reservation_id = receipt.reservation_id
            operation_context = asdict(receipt.context)
            if (
                receipt.kind != "label"
                or receipt.context != context
                or receipt.output_hash != operation_output_sha256(target)
            ):
                raise RuntimeError("OPD label receipt does not bind its operation context/output")
        else:
            # This path is useful for isolated CPU schedule tests.  A formal
            # runtime always supplies the ledgered executor above.
            output_hash = operation_output_sha256(target)
            operation_context = asdict(context)
        records.append(
            OpdTeacherLabel.from_target(
                decision_index,
                target,
                receipt_output_sha256=output_hash,
                frame_valid=bool(valid.detach().all().cpu()),
                teacher_checkpoint_sha256=teacher_checkpoint_hash,
                teacher_state_sha256=teacher_checkpoint_hash,
                teacher_checkpoint_file_sha256=teacher_checkpoint_file_hash,
                reservation_id=reservation_id,
                operation_context=operation_context,
                student_state_sha256=student_state_hash,
            )
        )
    return tuple(records)


def _score_teacher_rollout(runtime: Any, teacher_rollout: Any) -> Mapping[str, Any]:
    """Score the independent teacher branch separately from OPD labels."""
    terminal_state = getattr(teacher_rollout, "terminal_state", None)
    if terminal_state is None:
        raise RuntimeError("OPD collector omitted the independent teacher terminal state")
    with torch.no_grad():
        image, reward = _score_terminal(runtime, terminal_state)
    if not isinstance(reward, Tensor) or reward.numel() != 1 or not torch.isfinite(reward).all():
        raise RuntimeError("OPD teacher terminal reward is invalid")
    if not isinstance(image, Tensor) or not torch.isfinite(image).all():
        raise RuntimeError("OPD teacher terminal image is invalid")
    branch = getattr(teacher_rollout, "branch", None)
    if branch is not None and hasattr(branch, "terminal_reward"):
        branch.terminal_reward = reward.detach()
    stats = getattr(teacher_rollout, "stats", None)
    return {
        "image_sha256": tensor_sha256(image),
        "reward": float(reward.detach().double().cpu()),
        "collector_stats": (
            asdict(stats)
            if stats is not None and hasattr(stats, "__dataclass_fields__")
            else (dict(stats) if isinstance(stats, Mapping) else None)
        ),
    }


def _collect_opd_block(
    runtime: Any,
    behavior_renderer: Any,
    block: PromptSeedBlock,
    *,
    behavior_checkpoint: CheckpointProvenance,
    reference_checkpoint: CheckpointProvenance,
    storage_id: str,
    round_index: int,
    update_index: int | None,
) -> tuple[ScoredRollout, Mapping[str, Any], Mapping[str, Any]]:
    """Collect one fresh student pair plus one external-teacher branch."""
    contract = runtime.binding.contract
    teacher = _require_opd_teacher(runtime, contract)
    state_hash = module_state_sha256(behavior_renderer)
    teacher_hash = module_state_sha256(teacher)
    if behavior_checkpoint.renderer_state_sha256 != state_hash:
        raise RuntimeError("behavior checkpoint does not match the OPD collection policy")
    runtime.binding.validate_current(component=behavior_renderer)
    runtime.binding.validate_component(teacher)
    state = runtime.adapter.initial_state(
        prompts=block.record.prompt,
        prompt_ids=block.record.record_id,
        generation_seeds=block.generation_seed,
        checkpoint_hash=state_hash,
        split=block.record.split,
        height=int(contract["runtime"]["height"]),
        width=int(contract["runtime"]["width"]),
    )
    parameter = next(behavior_renderer.parameters())
    noise = {
        step: _proposal_noise(
            block,
            step,
            optimization_seed=int(contract["seed"]),
            active_mask=behavior_renderer.active_mask,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        for step in DECISION_INDICES
    }
    collector = runtime.adapter.make_collector(
        behavior_renderer,
        reference_renderer=runtime.reference_renderer,
        preserve_graph=False,
    )
    collect_with_teacher = getattr(collector, "collect_with_teacher", None)
    if not callable(collect_with_teacher):
        raise RuntimeError("OPD collector does not implement the shared-prefix teacher branch")
    result = collect_with_teacher(
        state,
        noise_by_decision=noise,
        teacher_renderer=teacher,
        teacher_checkpoint_hash=teacher_hash,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("OPD collector returned an invalid teacher-branch result")
    collection, teacher_rollout = result
    scored, convenience_label = _score_collection(
        runtime, collection, rollout_id=storage_id, ledger_label=False
    )
    teacher_labels = _record_opd_teacher_labels(
        runtime,
        teacher,
        block,
        scored,
        teacher_checkpoint_hash=teacher_hash,
    )
    # Keep the charged target object on the scored rollout and persist a second
    # detached copy in the collection payload.  The optimizer therefore uses
    # exactly the bytes that the label operation was charged for.
    for label in teacher_labels:
        label.validate(strict=True)
    scored = replace(scored, opd_teacher_labels=teacher_labels)
    collection.opd_teacher_targets = tuple(
        label.target.detach().clone() for label in teacher_labels
    )
    collection.opd_teacher_label_provenance = tuple(
        label.to_dict() for label in teacher_labels
    )
    teacher_score = _score_teacher_rollout(runtime, teacher_rollout)
    teacher_checkpoint = getattr(runtime, "teacher_checkpoint_provenance", None)
    teacher_checkpoint_file_hash = getattr(teacher_checkpoint, "sha256", None)
    if not isinstance(teacher_checkpoint_file_hash, str):
        raise RuntimeError("OPD runtime omitted teacher checkpoint file provenance")
    metadata = {
        "schema": "repldm.scored_opd_rollout.v1",
        "split": block.record.split,
        "prompt_id": block.record.record_id,
        "generation_seed": block.generation_seed,
        "source": block.record.source,
        "stratum": block.record.stratum,
        "fold": block.record.fold,
        "round": round_index,
        "update": update_index,
        "student_pair_label": convenience_label.to_dict(),
        "student_pair_label_receipt": None,
        "teacher_labels": [label.to_dict() for label in teacher_labels],
        "teacher_label_receipt_output_sha256": [
            label.receipt_output_sha256 for label in teacher_labels
        ],
        "terminal_rewards": {
            **{
                branch: float(value.detach().double().cpu())
                for branch, value in scored.rewards.items()
            },
            "teacher": teacher_score["reward"],
        },
        "terminal_image_sha256": {
            "teacher": teacher_score["image_sha256"],
        },
        "teacher_checkpoint_sha256": teacher_hash,
        "teacher_state_sha256": teacher_hash,
        "teacher_checkpoint_file_sha256": teacher_checkpoint_file_hash,
        "teacher_target_sha256": [label.raw_target_sha256 for label in teacher_labels],
        "teacher_collector_stats": teacher_score["collector_stats"],
        "student_collector_stats": asdict(collector.last_stats),
        "proposal_rng_schema": PROPOSAL_RNG_SCHEMA,
    }
    published_manifest = runtime.rollout_store.save(
        storage_id,
        collection,
        behavior_checkpoint=behavior_checkpoint,
        reference_checkpoint=reference_checkpoint,
        metadata=metadata,
    )
    manifest_path = Path(runtime.rollout_store.root) / f"{storage_id}.json"
    if json.loads(manifest_path.read_text(encoding="utf-8")) != published_manifest:
        raise RuntimeError("published OPD rollout manifest differs from the stored manifest")
    return scored, metadata, _file_descriptor(manifest_path)


def _objective(method: str, runtime: Any, values: Sequence[ScoredRollout]):
    if method == "opd":
        teacher = _require_opd_teacher(runtime, runtime.binding.contract)
        teacher_state_hash = module_state_sha256(teacher)
        teacher_checkpoint = getattr(runtime, "teacher_checkpoint_provenance", None)
        teacher_checkpoint_file_hash = getattr(teacher_checkpoint, "sha256", None)
        if not isinstance(teacher_checkpoint_file_hash, str):
            raise RuntimeError("OPD runtime omitted teacher checkpoint file provenance")
        return opd_objective(
            runtime.renderer,
            teacher,
            runtime.reference_renderer,
            values,
            require_stored_labels=True,
            teacher_checkpoint_hash=teacher_state_hash,
            teacher_checkpoint_file_hash=teacher_checkpoint_file_hash,
        )
    if method == "search_distill":
        return search_distill_objective(
            runtime.renderer, runtime.reference_renderer, values
        )
    if method == "dpo":
        return dpo_objective(runtime.renderer, runtime.reference_renderer, values)
    if method == "rl":
        return rl_objective(runtime.renderer, runtime.reference_renderer, values)
    raise ValueError(f"unsupported pair method {method!r}")


def _reward_summary(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty rollout set")
    differences = [
        float(row["terminal_rewards"]["plus"])
        - float(row["terminal_rewards"]["minus"])
        for row in rows
    ]
    non_ties = [value for value in differences if abs(value) > 1e-6]
    return {
        "blocks": len(rows),
        "plus_wins": sum(value > 1e-6 for value in differences),
        "minus_wins": sum(value < -1e-6 for value in differences),
        "ties": len(differences) - len(non_ties),
        "mean_plus_minus_reward": sum(differences) / len(differences),
    }


def _run_scheduled_training(
    binding: Any,
    method: str,
    *,
    runtime_builder: Callable[..., Any] | None = None,
) -> Mapping[str, Any]:
    """Execute the frozen two-round schedule for one registered online arm."""
    contract = _require_online_contract(binding, method)
    is_opd = method == "opd"
    expected_budget = OPD_QUERY_BUDGET if is_opd else PAIR_QUERY_BUDGET
    if runtime_builder is None:
        from ..runtime_factory import build_training_runtime

        runtime_builder = build_training_runtime
    with runtime_builder(binding) as runtime:
        if runtime.binding is not binding:
            raise RuntimeError("runtime replaced the authorization binding")
        if runtime.operation_executor is not runtime.adapter.operation_executor:
            raise RuntimeError("adapter and runtime do not share one operation executor")
        _require_runtime_gates(runtime, contract)
        if is_opd:
            # Keep the authenticated object alive for the complete run and
            # revalidate it at every block through _collect_opd_block.
            _require_opd_teacher(runtime, contract)
        records = _prompt_records(runtime.selected_records)
        training_blocks = _blocks(records, split="train")
        validation_blocks = _blocks(records, split="validation")
        optimizer_config = contract["optimizer"]
        optimizer = _make_optimizer(runtime.renderer, optimizer_config)
        trainer = RendererTrainer(
            runtime.renderer,
            optimizer,
            contract=contract,
            authorization_binding=binding,
            operation_executor=runtime.operation_executor,
            grad_norm_cap=float(optimizer_config["gradient_norm_cap"]),
            ema_decay=float(optimizer_config["ema_decay"]),
        )
        behavior = _freeze(copy.deepcopy(runtime.renderer))
        reference_checkpoint = runtime.initial_checkpoint_provenance
        if not isinstance(reference_checkpoint, CheckpointProvenance):
            raise TypeError("runtime omitted initial checkpoint provenance")
        reference_checkpoint.validate_current()
        if module_state_sha256(behavior) != reference_checkpoint.renderer_state_sha256:
            raise RuntimeError("initial behavior differs from the frozen reference checkpoint")
        behavior_checkpoint = reference_checkpoint
        round_results = []
        round_two_transition = None
        all_rollout_manifests: list[Mapping[str, Any]] = []
        all_label_hashes: list[str] = []
        for round_index in (1, 2):
            if round_index == 2:
                round_two_transition = trainer.start_round_two_from_ema(
                    _make_optimizer(runtime.renderer, optimizer_config)
                )
                if round_two_transition["new_optimizer_state_entries"] != 0:
                    raise RuntimeError("round-two AdamW state was not reset")
            round_rows = []
            update_artifacts = []
            for update_index, batch in enumerate(_batches(training_blocks)):
                scored_batch = []
                batch_ids = []
                batch_rows = []
                batch_rollout_manifests = []
                for block_index, block in enumerate(batch):
                    storage_id = (
                        f"train-r{round_index:02d}-u{update_index:02d}-"
                        f"b{block_index:02d}-{block.stable_id}"
                    )
                    collector_fn = _collect_opd_block if is_opd else _collect_block
                    scored, metadata, rollout_manifest = collector_fn(
                        runtime,
                        behavior,
                        block,
                        behavior_checkpoint=behavior_checkpoint,
                        reference_checkpoint=reference_checkpoint,
                        storage_id=storage_id,
                        round_index=round_index,
                        update_index=update_index,
                    )
                    scored_batch.append(scored)
                    batch_rows.append(metadata)
                    batch_rollout_manifests.append(rollout_manifest)
                    all_rollout_manifests.append(rollout_manifest)
                    if is_opd:
                        labels = metadata["teacher_label_receipt_output_sha256"]
                        if len(labels) != len(DECISION_INDICES):
                            raise RuntimeError("OPD block did not produce three teacher labels")
                        all_label_hashes.extend(labels)
                    else:
                        all_label_hashes.append(metadata["label_receipt_output_sha256"])
                    batch_ids.append(block.stable_id)
                objective_diagnostics: dict[str, Any] = {}

                def loss_fn() -> Tensor:
                    value = _objective(method, runtime, scored_batch)
                    if isinstance(value, tuple):
                        loss, ratios, kl = value
                        objective_diagnostics.update(
                            {
                                "ratio_min": float(ratios.min().double().cpu()),
                                "ratio_max": float(ratios.max().double().cpu()),
                                "mean_reference_kl": float(kl.mean().double().cpu()),
                            }
                        )
                        return loss
                    return value

                context = trainer.optimizer_context(
                    split="train",
                    batch_id=f"round-{round_index:02d}-update-{update_index:02d}",
                    action={"prompt_seed_blocks": batch_ids},
                    prefix=8,
                )
                update = trainer.update(loss_fn, operation_context=context)
                _copy_ema(trainer, behavior)
                behavior_checkpoint = _save_renderer_checkpoint(
                    runtime,
                    behavior,
                    filename=f"behavior-step-{trainer.step:03d}.pt",
                    step=trainer.step,
                    role="ema_behavior",
                )
                update_record = {
                    "schema": "repldm.renderer_update_record.v1",
                    "run_contract_sha256": binding.contract_hash,
                    "method": method,
                    "round": round_index,
                    "update": update_index,
                    "prompt_seed_blocks": batch_ids,
                    "rollouts": batch_rollout_manifests,
                    "labels": (
                        [
                            label
                            for row in batch_rows
                            for label in row["teacher_label_receipt_output_sha256"]
                        ]
                        if is_opd
                        else [
                            row["label_receipt_output_sha256"] for row in batch_rows
                        ]
                    ),
                    "optimizer": asdict(update),
                    "objective_diagnostics": objective_diagnostics,
                    "behavior_checkpoint": asdict(behavior_checkpoint),
                }
                update_path = (
                    Path(contract["paths"]["output_dir"])
                    / "updates"
                    / f"round-{round_index:02d}-update-{update_index:02d}.json"
                )
                _atomic_json(
                    update_path,
                    update_record,
                    prepublish=lambda: binding.validate_current(
                        component=runtime.renderer
                    ),
                )
                update_artifacts.append(_file_descriptor(update_path))
                round_rows.extend(batch_rows)
            round_checkpoint = _save_renderer_checkpoint(
                runtime,
                behavior,
                filename=f"ema-round-{round_index:02d}.pt",
                step=trainer.step,
                role=f"round_{round_index}_ema",
            )
            trainer_checkpoint_path = (
                Path(contract["paths"]["checkpoint_dir"])
                / f"trainer-round-{round_index:02d}.pt"
            )
            trainer.save(
                str(trainer_checkpoint_path),
                extra={"round": round_index, "ema_checkpoint": asdict(round_checkpoint)},
            )
            round_results.append(
                {
                    "round": round_index,
                    "optimizer_steps": 32,
                    "ema_checkpoint": asdict(round_checkpoint),
                    "trainer_checkpoint": str(trainer_checkpoint_path),
                    "training_preferences": _reward_summary(round_rows),
                    "update_records": update_artifacts,
                    "round_transition": (
                        round_two_transition if round_index == 2 else None
                    ),
                }
            )

        validation_rows = []
        validation_rollout_manifests = []
        validation_step_before = trainer.step
        for block_index, block in enumerate(validation_blocks):
            storage_id = f"validation-b{block_index:03d}-{block.stable_id}"
            collector_fn = _collect_opd_block if is_opd else _collect_block
            _scored, metadata, rollout_manifest = collector_fn(
                runtime,
                behavior,
                block,
                behavior_checkpoint=behavior_checkpoint,
                reference_checkpoint=reference_checkpoint,
                storage_id=storage_id,
                round_index=2,
                update_index=None,
            )
            validation_rows.append(metadata)
            validation_rollout_manifests.append(rollout_manifest)
            all_rollout_manifests.append(rollout_manifest)
            if is_opd:
                labels = metadata["teacher_label_receipt_output_sha256"]
                if len(labels) != len(DECISION_INDICES):
                    raise RuntimeError("OPD validation block did not produce three teacher labels")
                all_label_hashes.extend(labels)
            else:
                all_label_hashes.append(metadata["label_receipt_output_sha256"])

        validation_proof = {
            "schema": "repldm.renderer_validation_no_update.v1",
            "run_contract_sha256": binding.contract_hash,
            "split": "validation",
            "rollout_count": len(validation_rollout_manifests),
            "rollouts": validation_rollout_manifests,
            "optimizer_step_before": validation_step_before,
            "optimizer_step_after": trainer.step,
            "weights_unchanged": validation_step_before == trainer.step,
        }
        validation_path = (
            Path(contract["paths"]["output_dir"]) / "validation-no-update.json"
        )
        _atomic_json(
            validation_path,
            validation_proof,
            prepublish=lambda: binding.validate_current(component=runtime.renderer),
        )
        validation_artifact = _file_descriptor(validation_path)

        if trainer.step != 64:
            raise RuntimeError("online runner did not execute exactly 64 optimizer updates")
        ledger_summary = runtime.ledger.summary()
        if ledger_summary["reserved"] != expected_budget:
            raise RuntimeError("completed online run did not consume its exact allocation")
        if ledger_summary["unfinished_reservations"] != 0:
            raise RuntimeError("completed pair run has unfinished ledger reservations")
        expected_labels = 960 if is_opd else 320
        if len(all_rollout_manifests) != 320 or len(all_label_hashes) != expected_labels:
            raise RuntimeError("online run did not persist the exact rollout/label counts")
        label_receipts = runtime.ledger.successful_output_hashes("label")
        if tuple(all_label_hashes) != label_receipts:
            raise RuntimeError("rollout preference labels do not match the ledger receipts")
        rollout_index = {
            "schema": "repldm.renderer_rollout_index.v1",
            "run_contract_sha256": binding.contract_hash,
            "rollout_count": len(all_rollout_manifests),
            "label_count": len(all_label_hashes),
            "rollouts": all_rollout_manifests,
            "label_receipt_output_sha256": all_label_hashes,
        }
        rollout_index_path = (
            Path(contract["paths"]["output_dir"]) / "rollout-index.json"
        )
        _atomic_json(
            rollout_index_path,
            rollout_index,
            prepublish=lambda: binding.validate_current(component=runtime.renderer),
        )
        ledger_artifacts = {
            "records": _file_descriptor(Path(runtime.ledger.path)),
            "seal": _file_descriptor(Path(runtime.ledger.seal_path)),
        }
        result = {
            "schema": METHOD_RESULT_SCHEMA,
            "run_id": contract["run_id"],
            "method": method,
            "run_contract_sha256": binding.contract_hash,
            "status": "training_complete",
            "optimization_seed": contract["seed"],
            "rounds": round_results,
            "validation_preferences": _reward_summary(validation_rows),
            "ledger": ledger_summary,
            "ledger_artifacts": ledger_artifacts,
            "rollout_index": _file_descriptor(rollout_index_path),
            "validation_no_update": validation_artifact,
            "runtime": dict(runtime.provenance),
            "benchmark_status": "pending",
        }
        result_path = Path(contract["paths"]["output_dir"]) / "training-result.json"
        result["result_path"] = str(result_path)
        result["result_sha256"] = _result_sha256(result)
        _atomic_json(
            result_path,
            result,
            prepublish=lambda: binding.validate_current(component=runtime.renderer),
        )
        binding.validate_current(component=runtime.renderer)
        return result


def run_pair_training(
    binding: Any,
    method: str,
    *,
    runtime_builder: Callable[..., Any] | None = None,
) -> Mapping[str, Any]:
    """Execute one of the sampled-pair arms.

    ``opd`` is intentionally routed through its own source-registered wrapper;
    accepting it here would make a caller able to silently change the teacher
    allocation while still calling a pair API.
    """
    if method not in PAIR_METHODS:
        raise ValueError("run_pair_training only accepts sampled-pair methods")
    return _run_scheduled_training(
        binding,
        method,
        runtime_builder=runtime_builder,
    )


def run_opd_training(
    binding: Any,
    *,
    runtime_builder: Callable[..., Any] | None = None,
) -> Mapping[str, Any]:
    """Execute the external-teacher OPD schedule with its distinct budget."""
    return _run_scheduled_training(
        binding,
        "opd",
        runtime_builder=runtime_builder,
    )


__all__ = [
    "BLOCK_ORDER_SCHEMA",
    "COMMON_HYPERPARAMETERS",
    "DECISION_INDICES",
    "OPTIMIZATION_SEEDS",
    "ONLINE_METHODS",
    "OPD_QUERY_BUDGET",
    "PAIR_METHODS",
    "PAIR_QUERY_BUDGET",
    "PROPOSAL_RNG_SCHEMA",
    "PromptRecord",
    "PromptSeedBlock",
    "TRAIN_GENERATION_SEEDS",
    "VALIDATION_GENERATION_SEEDS",
    "run_opd_training",
    "run_pair_training",
]
