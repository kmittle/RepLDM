"""Atomic, contract-bound storage for detached renderer rollouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping
import uuid

import torch

from .authorization import require_authorization_binding
from .operations import operation_output_sha256
from .renderer import tensor_sha256
from .rollout import RolloutCollection


ROLLOUT_ENVELOPE_SCHEMA = "repldm.renderer_rollout.v1"
ROLLOUT_MANIFEST_SCHEMA = "repldm.renderer_rollout_manifest.v1"
F0_TARGET_ENVELOPE_SCHEMA = "repldm.renderer_f0_target_envelope.v1"
F0_TARGET_MANIFEST_SCHEMA = "repldm.renderer_f0_target_manifest.v1"
F0_ANCHOR_ENVELOPE_SCHEMA = "repldm.renderer_f0_anchor_envelope.v1"
F0_ANCHOR_MANIFEST_SCHEMA = "repldm.renderer_f0_anchor_manifest.v1"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_BLOCK_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_F0_PIXEL_FIELDS = frozenset(
    {"clipped_fraction", "mean_saturation", "contrast"}
)
# The native scheduler may return reduced-precision bytes while the analytic
# Euler observation is accumulated in float32.  Keep this tolerance explicit
# and shared with the collector's round-trip contract.
_F0_TRACE_ROUND_TRIP_RTOL = 2e-3
_F0_TRACE_ROUND_TRIP_ATOL = 3e-5


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
        raise ValueError("rollout provenance must contain canonical JSON values") from exc


def _hash_file(path: Path) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"rollout artifact must be an ordinary file: {path}")
    digest = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    after = path.stat()

    def identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if identity(before) != identity(after):
        raise RuntimeError(f"rollout artifact changed while hashing: {path}")
    return digest.hexdigest(), int(after.st_size)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _prepare_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"rollout directory cannot contain a symlink: {current}")
    if not path.is_dir():
        raise ValueError("rollout path must be a directory")


def _prepare_f0_directory(
    path: Path, *, create: bool, label: str = "F0 target"
) -> None:
    if not isinstance(label, str) or not label:
        raise ValueError("F0 directory label must be a non-empty string")
    if not path.is_absolute():
        raise ValueError(f"{label} directory must be absolute")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path.anchor, flags)
    current = Path(path.anchor)
    try:
        for part in path.parts[1:]:
            current /= part
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise ValueError(f"{label} directory is unavailable") from None
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                try:
                    child = os.open(part, flags, dir_fd=descriptor)
                except OSError as exc:
                    raise ValueError(
                        f"{label} directory cannot contain a symlink: {current}"
                    ) from exc
            except OSError as exc:
                raise ValueError(
                    f"{label} directory cannot contain a symlink: {current}"
                ) from exc
            status = os.fstat(child)
            if not stat.S_ISDIR(status.st_mode):
                os.close(child)
                raise ValueError(
                    f"{label} directory component is not a directory: {current}"
                )
            os.close(descriptor)
            descriptor = child
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _read_f0_ordinary_file(path: Path, *, label: str) -> tuple[bytes, str, int]:
    """Read and hash one exact regular-file inode without following a final link."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label} is missing or not an ordinary file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise ValueError(f"{label} must be a non-empty ordinary file")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(after):
            raise RuntimeError(f"{label} changed while it was being read")
        return b"".join(chunks), digest.hexdigest(), int(after.st_size)
    finally:
        os.close(descriptor)


def _publish_f0_exclusive(source: Path, destination: Path, *, label: str) -> None:
    """Hard-link a complete temporary inode without an overwrite race."""
    try:
        os.link(source, destination, follow_symlinks=False)
    except FileExistsError as exc:
        raise FileExistsError(f"{label} already exists: {destination.name}") from exc
    except OSError as exc:
        raise RuntimeError(f"cannot publish {label}: {destination.name}") from exc
    try:
        _fsync_directory(destination.parent)
    except BaseException:
        destination.unlink(missing_ok=True)
        try:
            _fsync_directory(destination.parent)
        except OSError:
            pass
        raise


def _f0_manifest_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(dict(value))).hexdigest()


def _f0_anchor_dict(
    value: Any, required: set[str], *, label: str
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != required:
        raise ValueError(f"{label} has an unsupported shape")
    return value


def _f0_anchor_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer of at least {minimum}")
    return value


def _f0_anchor_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _f0_anchor_hash(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 hash")
    return value


def _f0_anchor_tensor(
    value: Any,
    *,
    label: str,
    floating: bool | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if type(value) is not torch.Tensor:
        raise ValueError(f"{label} must be a plain tensor")
    if value.layout != torch.strided or value.device.type == "meta" or value.is_complex():
        raise ValueError(f"{label} must be a materialized strided real tensor")
    if value.requires_grad:
        raise ValueError(f"{label} must be detached")
    if floating is True and not value.is_floating_point():
        raise ValueError(f"{label} must use floating point")
    if floating is False and value.is_floating_point():
        raise ValueError(f"{label} must not use floating point")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{label} must use {dtype}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{label} contains non-finite values")
    return value.detach().to(device="cpu").contiguous().clone()


def _f0_anchor_optional_tensor(
    value: Any, *, label: str, floating: bool = True
) -> torch.Tensor | None:
    if value is None:
        return None
    return _f0_anchor_tensor(value, label=label, floating=floating)


def _f0_anchor_json(value: Any, *, label: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain finite JSON values") from exc
    return json.loads(encoded)


@dataclass(frozen=True)
class F0AnchorScoreRecovery:
    """Minimal values needed to reconstruct one ledger-backed anchor score."""

    image_sha256: str
    image_reward: torch.Tensor
    topiq_nr: torch.Tensor
    pixel_summary: Mapping[str, float]
    vae_decode_reservation_id: str
    image_reward_reservation_id: str
    topiq_nr_reservation_id: str

    def __post_init__(self) -> None:
        _f0_anchor_hash(self.image_sha256, label="F0 anchor image_sha256")
        for name in ("image_reward", "topiq_nr"):
            value = _f0_anchor_tensor(
                getattr(self, name),
                label=f"F0 anchor {name}",
                floating=True,
                dtype=torch.float32,
            )
            if value.numel() != 1:
                raise ValueError(f"F0 anchor {name} must contain one value")
            object.__setattr__(self, name, value.reshape(()))
        if not isinstance(self.pixel_summary, Mapping) or set(
            self.pixel_summary
        ) != _F0_PIXEL_FIELDS:
            raise ValueError("F0 anchor pixel summary is incomplete")
        pixel: dict[str, float] = {}
        for name in sorted(_F0_PIXEL_FIELDS):
            value = self.pixel_summary[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"F0 anchor pixel metric {name} must be numeric")
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError(f"F0 anchor pixel metric {name} must be finite")
            pixel[name] = normalized
        object.__setattr__(self, "pixel_summary", pixel)
        reservation_ids = []
        for name in (
            "vae_decode_reservation_id",
            "image_reward_reservation_id",
            "topiq_nr_reservation_id",
        ):
            reservation_ids.append(
                _f0_anchor_string(
                    getattr(self, name), label=f"F0 anchor {name}"
                )
            )
        if len(set(reservation_ids)) != 3:
            raise ValueError("F0 anchor score reservation IDs must be distinct")

    @classmethod
    def from_scored_image(cls, score: Any) -> "F0AnchorScoreRecovery":
        """Discard in-process receipt capabilities while retaining ledger keys."""
        from .methods.f0 import F0ScoredImage
        from .operations import OperationReceipt

        if type(score) is not F0ScoredImage:
            raise TypeError("score must be an F0ScoredImage")
        receipts = (
            score.decode_receipt,
            score.reward_receipt,
            score.witness_receipt,
        )
        if any(type(receipt) is not OperationReceipt for receipt in receipts):
            raise TypeError("F0 anchor score requires native operation receipts")
        if (
            score.decode_receipt.kind != "vae_decode"
            or score.reward_receipt.kind != "reward_forward"
            or score.witness_receipt.kind != "reward_forward"
        ):
            raise ValueError("F0 anchor score receipts have invalid operation kinds")
        return cls(
            image_sha256=score.image_sha256,
            image_reward=score.reward.detach().to(dtype=torch.float32, device="cpu"),
            topiq_nr=score.topiq_nr.detach().to(dtype=torch.float32, device="cpu"),
            pixel_summary=score.pixel,
            vae_decode_reservation_id=score.decode_receipt.reservation_id,
            image_reward_reservation_id=score.reward_receipt.reservation_id,
            topiq_nr_reservation_id=score.witness_receipt.reservation_id,
        )


@dataclass(frozen=True)
class F0AnchorBlockRecovery:
    """Safely decoded trace and the keys needed to re-seal its score receipts."""

    block: Any
    trace: Any
    score_recovery: F0AnchorScoreRecovery

    @property
    def score(self) -> F0AnchorScoreRecovery:
        """Compatibility alias for callers treating the recovery as a block."""
        return self.score_recovery


def _f0_anchor_score_to_payload(value: F0AnchorScoreRecovery) -> dict[str, Any]:
    if type(value) is not F0AnchorScoreRecovery:
        raise TypeError("score_recovery must be an F0AnchorScoreRecovery")
    normalized = F0AnchorScoreRecovery(
        image_sha256=value.image_sha256,
        image_reward=value.image_reward,
        topiq_nr=value.topiq_nr,
        pixel_summary=value.pixel_summary,
        vae_decode_reservation_id=value.vae_decode_reservation_id,
        image_reward_reservation_id=value.image_reward_reservation_id,
        topiq_nr_reservation_id=value.topiq_nr_reservation_id,
    )
    return {
        "image_sha256": normalized.image_sha256,
        "image_reward": normalized.image_reward,
        "topiq_nr": normalized.topiq_nr,
        "pixel_summary": dict(normalized.pixel_summary),
        "receipt_reservation_ids": {
            "vae_decode": normalized.vae_decode_reservation_id,
            "image_reward": normalized.image_reward_reservation_id,
            "topiq_nr": normalized.topiq_nr_reservation_id,
        },
    }


def _f0_anchor_score_from_payload(value: Any) -> F0AnchorScoreRecovery:
    payload = _f0_anchor_dict(
        value,
        {
            "image_sha256",
            "image_reward",
            "topiq_nr",
            "pixel_summary",
            "receipt_reservation_ids",
        },
        label="F0 anchor score recovery",
    )
    receipts = _f0_anchor_dict(
        payload["receipt_reservation_ids"],
        {"vae_decode", "image_reward", "topiq_nr"},
        label="F0 anchor receipt reservation IDs",
    )
    return F0AnchorScoreRecovery(
        image_sha256=payload["image_sha256"],
        image_reward=_f0_anchor_tensor(
            payload["image_reward"],
            label="F0 anchor image_reward",
            floating=True,
            dtype=torch.float32,
        ),
        topiq_nr=_f0_anchor_tensor(
            payload["topiq_nr"],
            label="F0 anchor topiq_nr",
            floating=True,
            dtype=torch.float32,
        ),
        pixel_summary=payload["pixel_summary"],
        vae_decode_reservation_id=receipts["vae_decode"],
        image_reward_reservation_id=receipts["image_reward"],
        topiq_nr_reservation_id=receipts["topiq_nr"],
    )


def _f0_prompt_block_to_payload(value: Any) -> dict[str, Any]:
    from .methods.runner import PromptRecord, PromptSeedBlock

    if type(value) is not PromptSeedBlock or type(value.record) is not PromptRecord:
        raise TypeError("block must be a PromptSeedBlock")
    record = value.record
    for name in ("record_id", "prompt", "split", "source", "stratum"):
        _f0_anchor_string(getattr(record, name), label=f"F0 anchor block {name}")
    if record.split not in {"train", "validation"}:
        raise ValueError("F0 anchor block split must be train or validation")
    if record.fold is not None:
        _f0_anchor_int(record.fold, label="F0 anchor block fold")
    _f0_anchor_int(
        value.generation_seed, label="F0 anchor block generation_seed"
    )
    stable_id = value.stable_id
    if not isinstance(stable_id, str) or _BLOCK_RE.fullmatch(stable_id) is None:
        raise ValueError("F0 anchor stable_id must be filename-safe")
    return {
        "record": {
            "record_id": record.record_id,
            "prompt": record.prompt,
            "split": record.split,
            "source": record.source,
            "stratum": record.stratum,
            "fold": record.fold,
        },
        "generation_seed": value.generation_seed,
    }


def _f0_prompt_block_from_payload(value: Any) -> Any:
    from .methods.runner import PromptRecord, PromptSeedBlock

    payload = _f0_anchor_dict(
        value,
        {"record", "generation_seed"},
        label="F0 anchor block payload",
    )
    record_payload = _f0_anchor_dict(
        payload["record"],
        {"record_id", "prompt", "split", "source", "stratum", "fold"},
        label="F0 anchor prompt record",
    )
    record = PromptRecord(
        record_id=record_payload["record_id"],
        prompt=record_payload["prompt"],
        split=record_payload["split"],
        source=record_payload["source"],
        stratum=record_payload["stratum"],
        fold=record_payload["fold"],
    )
    block = PromptSeedBlock(
        record=record,
        generation_seed=payload["generation_seed"],
    )
    normalized = _f0_prompt_block_to_payload(block)
    if normalized != payload:
        raise ValueError("F0 anchor block payload is not canonical")
    return block


def _f0_conditioning_to_payload(value: Any) -> dict[str, Any]:
    from .sdxl_adapter import SdxlPromptConditioning

    if type(value) is not SdxlPromptConditioning:
        raise TypeError("F0 anchor conditioning must be SdxlPromptConditioning")
    normalized = SdxlPromptConditioning(
        prompt_ids=tuple(value.prompt_ids),
        prompts=tuple(value.prompts),
        generation_seeds=tuple(value.generation_seeds),
        prompt_embeds=value.prompt_embeds,
        pooled_prompt_embeds=value.pooled_prompt_embeds,
        add_text_embeds=value.add_text_embeds,
        add_time_ids=value.add_time_ids,
        do_classifier_free_guidance=value.do_classifier_free_guidance,
    )
    return {
        "prompt_ids": list(normalized.prompt_ids),
        "prompts": list(normalized.prompts),
        "generation_seeds": list(normalized.generation_seeds),
        "prompt_embeds": _f0_anchor_tensor(
            normalized.prompt_embeds,
            label="F0 anchor prompt_embeds",
            floating=True,
        ),
        "pooled_prompt_embeds": _f0_anchor_tensor(
            normalized.pooled_prompt_embeds,
            label="F0 anchor pooled_prompt_embeds",
            floating=True,
        ),
        "add_text_embeds": _f0_anchor_tensor(
            normalized.add_text_embeds,
            label="F0 anchor add_text_embeds",
            floating=True,
        ),
        "add_time_ids": _f0_anchor_tensor(
            normalized.add_time_ids,
            label="F0 anchor add_time_ids",
            floating=True,
        ),
        "do_classifier_free_guidance": normalized.do_classifier_free_guidance,
    }


def _f0_conditioning_from_payload(value: Any) -> Any:
    from .sdxl_adapter import SdxlPromptConditioning

    payload = _f0_anchor_dict(
        value,
        {
            "prompt_ids",
            "prompts",
            "generation_seeds",
            "prompt_embeds",
            "pooled_prompt_embeds",
            "add_text_embeds",
            "add_time_ids",
            "do_classifier_free_guidance",
        },
        label="F0 anchor conditioning payload",
    )
    for name in ("prompt_ids", "prompts", "generation_seeds"):
        if type(payload[name]) is not list:
            raise ValueError(f"F0 anchor conditioning {name} must be a list")
    if type(payload["do_classifier_free_guidance"]) is not bool:
        raise ValueError("F0 anchor CFG flag must be boolean")
    return SdxlPromptConditioning(
        prompt_ids=tuple(payload["prompt_ids"]),
        prompts=tuple(payload["prompts"]),
        generation_seeds=tuple(payload["generation_seeds"]),
        prompt_embeds=_f0_anchor_tensor(
            payload["prompt_embeds"],
            label="F0 anchor prompt_embeds",
            floating=True,
        ),
        pooled_prompt_embeds=_f0_anchor_tensor(
            payload["pooled_prompt_embeds"],
            label="F0 anchor pooled_prompt_embeds",
            floating=True,
        ),
        add_text_embeds=_f0_anchor_tensor(
            payload["add_text_embeds"],
            label="F0 anchor add_text_embeds",
            floating=True,
        ),
        add_time_ids=_f0_anchor_tensor(
            payload["add_time_ids"],
            label="F0 anchor add_time_ids",
            floating=True,
        ),
        do_classifier_free_guidance=payload["do_classifier_free_guidance"],
    )


def _f0_state_to_payload(value: Any) -> dict[str, Any]:
    from .sdxl_adapter import SdxlEulerState

    if type(value) is not SdxlEulerState:
        raise TypeError("F0 anchor state must be SdxlEulerState")
    normalized = SdxlEulerState(
        latents=value.latents,
        conditioning=value.conditioning,
        step_index=value.step_index,
        total_steps=value.total_steps,
        split=value.split,
        branch=value.branch,
        prefix=value.prefix,
        checkpoint_hash=value.checkpoint_hash,
        action_history=value.action_history,
    )
    return {
        "latents": _f0_anchor_tensor(
            normalized.latents, label="F0 anchor state latents", floating=True
        ),
        "conditioning": _f0_conditioning_to_payload(normalized.conditioning),
        "step_index": normalized.step_index,
        "total_steps": normalized.total_steps,
        "split": normalized.split,
        "branch": normalized.branch,
        "prefix": normalized.prefix,
        "checkpoint_hash": normalized.checkpoint_hash,
        "action_history": _f0_anchor_json(
            list(normalized.action_history), label="F0 anchor action_history"
        ),
    }


def _f0_state_from_payload(value: Any) -> Any:
    from .sdxl_adapter import SdxlEulerState

    payload = _f0_anchor_dict(
        value,
        {
            "latents",
            "conditioning",
            "step_index",
            "total_steps",
            "split",
            "branch",
            "prefix",
            "checkpoint_hash",
            "action_history",
        },
        label="F0 anchor state payload",
    )
    if type(payload["action_history"]) is not list:
        raise ValueError("F0 anchor action_history must be a list")
    return SdxlEulerState(
        latents=_f0_anchor_tensor(
            payload["latents"], label="F0 anchor state latents", floating=True
        ),
        conditioning=_f0_conditioning_from_payload(payload["conditioning"]),
        step_index=payload["step_index"],
        total_steps=payload["total_steps"],
        split=payload["split"],
        branch=payload["branch"],
        prefix=payload["prefix"],
        checkpoint_hash=payload["checkpoint_hash"],
        action_history=tuple(
            _f0_anchor_json(
                payload["action_history"], label="F0 anchor action_history"
            )
        ),
    )


def _f0_observation_to_payload(value: Any) -> dict[str, Any]:
    from AttentionGuidance.latent_renderer import RendererObservation

    if type(value) is not RendererObservation:
        raise TypeError("F0 anchor observation must be RendererObservation")
    latents = _f0_anchor_tensor(
        value.latents_before_step,
        label="F0 anchor observation latents",
        floating=True,
    )
    original = _f0_anchor_tensor(
        value.pred_original_sample,
        label="F0 anchor pred_original_sample",
        floating=True,
    )
    update = _f0_anchor_tensor(
        value.scheduler_update,
        label="F0 anchor scheduler_update",
        floating=True,
    )
    if latents.ndim != 4 or original.shape != latents.shape or update.shape != latents.shape:
        raise ValueError("F0 anchor observation tensors have inconsistent shapes")
    step_index = _f0_anchor_int(
        value.step_index, label="F0 anchor observation step_index"
    )
    return {
        "latents_before_step": latents,
        "pred_original_sample": original,
        "scheduler_update": update,
        "step_index": step_index,
        "timestep": _f0_anchor_tensor(
            value.timestep, label="F0 anchor observation timestep"
        ),
        "normalized_timestep": _f0_anchor_tensor(
            value.normalized_timestep,
            label="F0 anchor normalized_timestep",
            floating=True,
        ),
        "pooled_prompt_embeds": _f0_anchor_optional_tensor(
            value.pooled_prompt_embeds,
            label="F0 anchor pooled_prompt_embeds",
        ),
    }


def _f0_observation_from_payload(value: Any) -> Any:
    from AttentionGuidance.latent_renderer import RendererObservation

    payload = _f0_anchor_dict(
        value,
        {
            "latents_before_step",
            "pred_original_sample",
            "scheduler_update",
            "step_index",
            "timestep",
            "normalized_timestep",
            "pooled_prompt_embeds",
        },
        label="F0 anchor observation payload",
    )
    result = RendererObservation(
        latents_before_step=_f0_anchor_tensor(
            payload["latents_before_step"],
            label="F0 anchor observation latents",
            floating=True,
        ),
        pred_original_sample=_f0_anchor_tensor(
            payload["pred_original_sample"],
            label="F0 anchor pred_original_sample",
            floating=True,
        ),
        scheduler_update=_f0_anchor_tensor(
            payload["scheduler_update"],
            label="F0 anchor scheduler_update",
            floating=True,
        ),
        step_index=_f0_anchor_int(
            payload["step_index"], label="F0 anchor observation step_index"
        ),
        timestep=_f0_anchor_tensor(
            payload["timestep"], label="F0 anchor observation timestep"
        ),
        normalized_timestep=_f0_anchor_tensor(
            payload["normalized_timestep"],
            label="F0 anchor normalized_timestep",
            floating=True,
        ),
        pooled_prompt_embeds=_f0_anchor_optional_tensor(
            payload["pooled_prompt_embeds"],
            label="F0 anchor pooled_prompt_embeds",
        ),
    )
    latents = result.latents_before_step
    if (
        latents.ndim != 4
        or result.pred_original_sample.shape != latents.shape
        or result.scheduler_update.shape != latents.shape
    ):
        raise ValueError("F0 anchor observation tensors have inconsistent shapes")
    return result


def _f0_condition_to_payload(value: Any) -> dict[str, Any]:
    from AttentionGuidance.latent_renderer import RendererCondition

    if type(value) is not RendererCondition:
        raise TypeError("F0 anchor condition must be RendererCondition")
    bases = _f0_anchor_tensor(
        value.bases, label="F0 anchor condition bases", floating=True
    )
    if bases.ndim != 5:
        raise ValueError("F0 anchor condition bases must be rank five")
    return {
        "bases": bases,
        "prompt_embedding": _f0_anchor_optional_tensor(
            value.prompt_embedding, label="F0 anchor prompt_embedding"
        ),
        "state_features": _f0_anchor_optional_tensor(
            value.state_features, label="F0 anchor state_features"
        ),
    }


def _f0_condition_from_payload(value: Any) -> Any:
    from AttentionGuidance.latent_renderer import RendererCondition

    payload = _f0_anchor_dict(
        value,
        {"bases", "prompt_embedding", "state_features"},
        label="F0 anchor condition payload",
    )
    bases = _f0_anchor_tensor(
        payload["bases"], label="F0 anchor condition bases", floating=True
    )
    if bases.ndim != 5:
        raise ValueError("F0 anchor condition bases must be rank five")
    return RendererCondition(
        bases=bases,
        prompt_embedding=_f0_anchor_optional_tensor(
            payload["prompt_embedding"], label="F0 anchor prompt_embedding"
        ),
        state_features=_f0_anchor_optional_tensor(
            payload["state_features"], label="F0 anchor state_features"
        ),
    )


def _f0_diagnostics_to_payload(value: Any) -> dict[str, Any]:
    from .renderer import EulerFrameDiagnostics

    if type(value) is not EulerFrameDiagnostics:
        raise TypeError("F0 anchor diagnostics must be EulerFrameDiagnostics")
    if type(value.active_mask) is not tuple or any(
        type(item) is not bool for item in value.active_mask
    ):
        raise ValueError("F0 anchor active_mask must contain plain booleans")
    for name in ("gram_hash", "frame_hash"):
        hashes = getattr(value, name)
        if type(hashes) is not tuple:
            raise ValueError(f"F0 anchor {name} must be a tuple")
        for item in hashes:
            _f0_anchor_hash(item, label=f"F0 anchor {name}")
    return {
        "valid": _f0_anchor_tensor(
            value.valid,
            label="F0 anchor frame validity",
            dtype=torch.bool,
        ),
        "active_mask": list(value.active_mask),
        "eigenvalues": _f0_anchor_tensor(
            value.eigenvalues,
            label="F0 anchor frame eigenvalues",
            floating=True,
        ),
        "condition_number": _f0_anchor_tensor(
            value.condition_number,
            label="F0 anchor frame condition_number",
            floating=True,
        ),
        "gram_error": _f0_anchor_tensor(
            value.gram_error,
            label="F0 anchor frame gram_error",
            floating=True,
        ),
        "angle": _f0_anchor_tensor(
            value.angle, label="F0 anchor frame angle", floating=True
        ),
        "angle_cap_multiplier": _f0_anchor_tensor(
            value.angle_cap_multiplier,
            label="F0 anchor frame angle_cap_multiplier",
            floating=True,
        ),
        "scheduler_cap_multiplier": _f0_anchor_tensor(
            value.scheduler_cap_multiplier,
            label="F0 anchor frame scheduler_cap_multiplier",
            floating=True,
        ),
        "mapped_update_ratio": _f0_anchor_tensor(
            value.mapped_update_ratio,
            label="F0 anchor frame mapped_update_ratio",
            floating=True,
        ),
        "gram_hash": list(value.gram_hash),
        "frame_hash": list(value.frame_hash),
    }


def _f0_diagnostics_from_payload(value: Any) -> Any:
    from .renderer import EulerFrameDiagnostics

    payload = _f0_anchor_dict(
        value,
        {
            "valid",
            "active_mask",
            "eigenvalues",
            "condition_number",
            "gram_error",
            "angle",
            "angle_cap_multiplier",
            "scheduler_cap_multiplier",
            "mapped_update_ratio",
            "gram_hash",
            "frame_hash",
        },
        label="F0 anchor frame diagnostics payload",
    )
    for name in ("active_mask", "gram_hash", "frame_hash"):
        if type(payload[name]) is not list:
            raise ValueError(f"F0 anchor diagnostics {name} must be a list")
    if any(type(item) is not bool for item in payload["active_mask"]):
        raise ValueError("F0 anchor active_mask must contain plain booleans")
    for name in ("gram_hash", "frame_hash"):
        for item in payload[name]:
            _f0_anchor_hash(item, label=f"F0 anchor {name}")
    result = EulerFrameDiagnostics(
        valid=_f0_anchor_tensor(
            payload["valid"],
            label="F0 anchor frame validity",
            dtype=torch.bool,
        ),
        active_mask=tuple(payload["active_mask"]),
        eigenvalues=_f0_anchor_tensor(
            payload["eigenvalues"],
            label="F0 anchor frame eigenvalues",
            floating=True,
        ),
        condition_number=_f0_anchor_tensor(
            payload["condition_number"],
            label="F0 anchor frame condition_number",
            floating=True,
        ),
        gram_error=_f0_anchor_tensor(
            payload["gram_error"],
            label="F0 anchor frame gram_error",
            floating=True,
        ),
        angle=_f0_anchor_tensor(
            payload["angle"], label="F0 anchor frame angle", floating=True
        ),
        angle_cap_multiplier=_f0_anchor_tensor(
            payload["angle_cap_multiplier"],
            label="F0 anchor frame angle_cap_multiplier",
            floating=True,
        ),
        scheduler_cap_multiplier=_f0_anchor_tensor(
            payload["scheduler_cap_multiplier"],
            label="F0 anchor frame scheduler_cap_multiplier",
            floating=True,
        ),
        mapped_update_ratio=_f0_anchor_tensor(
            payload["mapped_update_ratio"],
            label="F0 anchor frame mapped_update_ratio",
            floating=True,
        ),
        gram_hash=tuple(payload["gram_hash"]),
        frame_hash=tuple(payload["frame_hash"]),
    )
    return result


def _f0_validate_tensor(
    value: Any,
    *,
    label: str,
    device: torch.device | None = None,
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
    floating: bool = True,
    detached: bool = True,
) -> torch.Tensor:
    """Validate one in-memory F0 tensor without coercing its precision."""
    if type(value) is not torch.Tensor:
        raise ValueError(f"{label} must be a plain tensor")
    if value.layout != torch.strided or value.device.type == "meta" or value.is_complex():
        raise ValueError(f"{label} must be a materialized real strided tensor")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{label} has the wrong rank")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f"{label} has the wrong shape")
    if floating and not value.is_floating_point():
        raise ValueError(f"{label} must use a floating-point dtype")
    if detached and value.requires_grad:
        raise ValueError(f"{label} must be detached")
    if device is not None and value.device != device:
        raise ValueError(f"{label} is on the wrong device")
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError(f"{label} contains non-finite values")
    return value


def _f0_batch_scalar(
    value: Any,
    *,
    label: str,
    batch: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a scalar-or-batch value expanded without hiding device drift."""
    if type(value) is torch.Tensor:
        tensor = _f0_validate_tensor(value, label=label, device=device)
    else:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be a finite scalar or tensor")
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} must be finite")
        tensor = torch.as_tensor(value, device=device, dtype=torch.float32)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    if tensor.ndim != 1 or tensor.numel() not in (1, batch):
        raise ValueError(f"{label} must be scalar or contain one value per batch row")
    if tensor.numel() == 1:
        tensor = tensor.expand(batch)
    return tensor


def _f0_tensor_relation(
    left: Any,
    right: Any,
    *,
    label: str,
    require_dtype: bool = False,
    rtol: float = _F0_TRACE_ROUND_TRIP_RTOL,
    atol: float = _F0_TRACE_ROUND_TRIP_ATOL,
) -> None:
    """Check shape/device/finite tensors, allowing only bounded float casts."""
    if type(left) is not torch.Tensor or type(right) is not torch.Tensor:
        raise ValueError(f"{label} must contain tensors")
    _f0_validate_tensor(left, label=f"{label} (left)")
    _f0_validate_tensor(right, label=f"{label} (right)")
    if left.shape != right.shape:
        raise ValueError(f"{label} has mismatched shapes")
    if left.device != right.device:
        raise ValueError(f"{label} has mismatched devices")
    if require_dtype and left.dtype != right.dtype:
        raise ValueError(f"{label} has mismatched dtypes")
    if left.dtype != right.dtype and not (
        left.is_floating_point() and right.is_floating_point()
    ):
        raise ValueError(f"{label} has incompatible dtypes")
    if left.is_floating_point():
        try:
            equal = torch.allclose(
                left.detach().float(),
                right.detach().float(),
                rtol=float(rtol),
                atol=float(atol),
            )
        except (RuntimeError, TypeError) as exc:
            raise ValueError(f"{label} could not be compared") from exc
        if not bool(equal):
            raise ValueError(f"{label} values differ")
    elif not torch.equal(left, right):
        raise ValueError(f"{label} values differ")


def _f0_validate_frame(value: Any) -> None:
    from .renderer import EulerFrameDiagnostics, EulerFrameState

    if type(value) is not EulerFrameState:
        raise TypeError("F0 anchor frame must be EulerFrameState")
    sample = _f0_validate_tensor(
        value.sample, label="F0 anchor frame sample", ndim=4
    )
    batch = sample.shape[0]
    device = sample.device
    for name in ("clean_latent", "native_update"):
        _f0_validate_tensor(
            getattr(value, name),
            label=f"F0 anchor frame {name}",
            shape=tuple(sample.shape),
            device=device,
        )
    if value.native_model_output is not None:
        _f0_validate_tensor(
            value.native_model_output,
            label="F0 anchor frame native_model_output",
            shape=tuple(sample.shape),
            device=device,
        )
    bases = (
        value.raw_bases,
        value.tangent_bases,
        value.mapped_bases,
        value.clean_bases,
    )
    for name, item in zip(
        ("raw_bases", "tangent_bases", "mapped_bases", "clean_bases"), bases
    ):
        _f0_validate_tensor(
            item,
            label=f"F0 anchor frame {name}",
            ndim=5,
            device=device,
        )
    if any(item.shape != bases[0].shape for item in bases[1:]):
        raise ValueError("F0 anchor frame basis shapes differ")
    if (
        bases[0].shape[0] != batch
        or bases[0].shape[1] <= 0
        or bases[0].shape[2:] != sample.shape[1:]
    ):
        raise ValueError("F0 anchor frame bases do not match its sample")

    for name in ("sigma_from", "sigma_to", "kappa"):
        _f0_batch_scalar(
            getattr(value, name),
            label=f"F0 anchor frame {name}",
            batch=batch,
            device=device,
        )
    sigma_from = _f0_batch_scalar(
        value.sigma_from,
        label="F0 anchor frame sigma_from",
        batch=batch,
        device=device,
    )
    sigma_to = _f0_batch_scalar(
        value.sigma_to,
        label="F0 anchor frame sigma_to",
        batch=batch,
        device=device,
    )
    _f0_batch_scalar(
        value.kappa, label="F0 anchor frame kappa", batch=batch, device=device
    )
    if bool(torch.any(sigma_from <= 0)) or bool(torch.any(sigma_to < 0)) or bool(
        torch.any(sigma_to > sigma_from)
    ):
        raise ValueError("F0 anchor frame sigmas are out of order")
    _f0_validate_tensor(
        value.context,
        label="F0 anchor frame context",
        ndim=2,
        device=device,
    )
    if value.context.shape[0] != batch or value.context.shape[1] <= 0:
        raise ValueError("F0 anchor frame context has the wrong batch shape")

    diagnostics = value.diagnostics
    if type(diagnostics) is not EulerFrameDiagnostics:
        raise TypeError("F0 anchor frame diagnostics must be EulerFrameDiagnostics")
    _f0_validate_tensor(
        diagnostics.valid,
        label="F0 anchor frame diagnostics.valid",
        shape=(batch,),
        device=device,
        floating=False,
    )
    if diagnostics.valid.dtype != torch.bool:
        raise ValueError("F0 anchor frame diagnostics.valid must be boolean")
    active_mask = diagnostics.active_mask
    if not isinstance(active_mask, (tuple, list)) or len(active_mask) != bases[0].shape[1]:
        raise ValueError("F0 anchor frame active mask has the wrong slot count")
    if any(type(item) is not bool for item in active_mask):
        raise ValueError("F0 anchor frame active mask must contain booleans")
    rank = sum(active_mask)
    diagnostic_vectors = {
        "eigenvalues": (batch, rank),
        "condition_number": (batch,),
        "gram_error": (batch,),
        "angle": (batch,),
        "angle_cap_multiplier": (batch,),
        "scheduler_cap_multiplier": (batch,),
        "mapped_update_ratio": (batch,),
    }
    diagnostic_dtype: torch.dtype | None = None
    for name, shape in diagnostic_vectors.items():
        tensor = _f0_validate_tensor(
            getattr(diagnostics, name),
            label=f"F0 anchor frame diagnostics.{name}",
            shape=shape,
            device=device,
        )
        if diagnostic_dtype is None:
            diagnostic_dtype = tensor.dtype
        elif tensor.dtype != diagnostic_dtype:
            raise ValueError("F0 anchor diagnostics dtypes are inconsistent")
    for name in ("gram_hash", "frame_hash"):
        values = getattr(diagnostics, name)
        if not isinstance(values, (tuple, list)) or len(values) != batch:
            raise ValueError(f"F0 anchor frame {name} has the wrong batch count")
        for item in values:
            _f0_anchor_hash(item, label=f"F0 anchor {name}")
    if value.prediction_type not in {
        "epsilon",
        "sample",
        "original_sample",
        "v_prediction",
    }:
        raise ValueError("F0 anchor frame prediction_type is unsupported")


def _f0_validate_observation(
    value: Any,
    *,
    label: str,
    expected_step: int | None = None,
) -> tuple[int, torch.device]:
    """Validate an Euler observation while retaining its native precision."""
    from AttentionGuidance.latent_renderer import RendererObservation

    if type(value) is not RendererObservation:
        raise TypeError(f"{label} must be RendererObservation")
    current = _f0_validate_tensor(
        value.latents_before_step,
        label=f"{label}.latents_before_step",
        ndim=4,
    )
    batch = current.shape[0]
    device = current.device
    for name in ("pred_original_sample", "scheduler_update"):
        _f0_validate_tensor(
            getattr(value, name),
            label=f"{label}.{name}",
            shape=tuple(current.shape),
            device=device,
        )
    for name in ("timestep", "normalized_timestep"):
        tensor = _f0_validate_tensor(
            getattr(value, name), label=f"{label}.{name}", device=device
        )
        if tensor.ndim > 1 or (tensor.ndim == 1 and tensor.numel() not in (1, batch)):
            raise ValueError(f"{label}.{name} must be scalar or batch-shaped")
    if value.pooled_prompt_embeds is not None:
        pooled = _f0_validate_tensor(
            value.pooled_prompt_embeds,
            label=f"{label}.pooled_prompt_embeds",
            device=device,
        )
        if pooled.ndim < 1 or pooled.shape[0] != batch:
            raise ValueError(f"{label}.pooled_prompt_embeds has the wrong batch shape")
    if type(value.step_index) is not int or value.step_index < 0:
        raise ValueError(f"{label}.step_index is invalid")
    if expected_step is not None and value.step_index != expected_step:
        raise ValueError(f"{label}.step_index differs from its transition")
    return batch, device


def _f0_validate_condition(
    value: Any,
    *,
    label: str,
    batch: int,
    device: torch.device,
    latent_shape: tuple[int, ...],
) -> None:
    """Validate basis and compact conditioning tensors for one cached step."""
    from AttentionGuidance.latent_renderer import RendererCondition

    if type(value) is not RendererCondition:
        raise TypeError(f"{label} must be RendererCondition")
    bases = _f0_validate_tensor(
        value.bases, label=f"{label}.bases", ndim=5, device=device
    )
    if (
        bases.shape[0] != batch
        or bases.shape[1] <= 0
        or bases.shape[2:] != latent_shape[1:]
    ):
        raise ValueError(f"{label}.bases has the wrong shape")
    for name in ("prompt_embedding", "state_features"):
        item = getattr(value, name)
        if item is None:
            continue
        tensor = _f0_validate_tensor(
            item, label=f"{label}.{name}", device=device
        )
        if tensor.ndim < 1 or tensor.shape[0] != batch:
            raise ValueError(f"{label}.{name} has the wrong batch shape")


def _f0_validate_step(
    value: Any, *, label: str, expected_step: int | None = None
) -> tuple[int, torch.device]:
    """Validate one cached EulerRolloutStep and return its batch/device."""
    from .collector import EulerRolloutStep

    if type(value) is not EulerRolloutStep:
        raise TypeError(f"{label} must be EulerRolloutStep")
    batch, device = _f0_validate_observation(
        value.observation,
        label=f"{label}.observation",
        expected_step=expected_step,
    )
    current = value.observation.latents_before_step
    _f0_validate_condition(
        value.condition,
        label=f"{label}.condition",
        batch=batch,
        device=device,
        latent_shape=tuple(current.shape),
    )
    for name in ("native_model_output", "native_prev_sample"):
        _f0_validate_tensor(
            getattr(value, name),
            label=f"{label}.{name}",
            shape=tuple(current.shape),
            device=device,
        )
    for name in ("sigma_from", "sigma_to"):
        _f0_batch_scalar(
            getattr(value, name), label=f"{label}.{name}", batch=batch, device=device
        )
    if value.clean_update_gain is not None:
        _f0_batch_scalar(
            value.clean_update_gain,
            label=f"{label}.clean_update_gain",
            batch=batch,
            device=device,
        )
    if value.prediction_type not in {
        "epsilon",
        "sample",
        "original_sample",
        "v_prediction",
    }:
        raise ValueError(f"{label}.prediction_type is unsupported")
    try:
        metadata = _f0_anchor_json(dict(value.metadata), label=f"{label}.metadata")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}.metadata is invalid") from exc
    if type(metadata) is not dict:
        raise ValueError(f"{label}.metadata must be a mapping")
    return batch, device


def _f0_frame_to_payload(value: Any) -> dict[str, Any]:
    _f0_validate_frame(value)
    fields = {
        "clean_latent": _f0_anchor_tensor(
            value.clean_latent, label="F0 anchor frame clean_latent", floating=True
        ),
        "sample": _f0_anchor_tensor(
            value.sample, label="F0 anchor frame sample", floating=True
        ),
        "native_model_output": _f0_anchor_optional_tensor(
            value.native_model_output,
            label="F0 anchor frame native_model_output",
        ),
        "raw_bases": _f0_anchor_tensor(
            value.raw_bases, label="F0 anchor frame raw_bases", floating=True
        ),
        "tangent_bases": _f0_anchor_tensor(
            value.tangent_bases,
            label="F0 anchor frame tangent_bases",
            floating=True,
        ),
        "mapped_bases": _f0_anchor_tensor(
            value.mapped_bases,
            label="F0 anchor frame mapped_bases",
            floating=True,
        ),
        "clean_bases": _f0_anchor_tensor(
            value.clean_bases,
            label="F0 anchor frame clean_bases",
            floating=True,
        ),
        "native_update": _f0_anchor_tensor(
            value.native_update,
            label="F0 anchor frame native_update",
            floating=True,
        ),
        "sigma_from": _f0_anchor_tensor(
            value.sigma_from, label="F0 anchor frame sigma_from", floating=True
        ),
        "sigma_to": _f0_anchor_tensor(
            value.sigma_to, label="F0 anchor frame sigma_to", floating=True
        ),
        "kappa": _f0_anchor_tensor(
            value.kappa, label="F0 anchor frame kappa", floating=True
        ),
        "context": _f0_anchor_tensor(
            value.context, label="F0 anchor frame context", floating=True
        ),
        "diagnostics": _f0_diagnostics_to_payload(value.diagnostics),
        "prediction_type": value.prediction_type,
    }
    return fields


def _f0_frame_from_payload(value: Any) -> Any:
    from .renderer import EulerFrameState

    names = {
        "clean_latent",
        "sample",
        "native_model_output",
        "raw_bases",
        "tangent_bases",
        "mapped_bases",
        "clean_bases",
        "native_update",
        "sigma_from",
        "sigma_to",
        "kappa",
        "context",
        "diagnostics",
        "prediction_type",
    }
    payload = _f0_anchor_dict(
        value, names, label="F0 anchor frame payload"
    )
    result = EulerFrameState(
        clean_latent=_f0_anchor_tensor(
            payload["clean_latent"],
            label="F0 anchor frame clean_latent",
            floating=True,
        ),
        sample=_f0_anchor_tensor(
            payload["sample"], label="F0 anchor frame sample", floating=True
        ),
        native_model_output=_f0_anchor_optional_tensor(
            payload["native_model_output"],
            label="F0 anchor frame native_model_output",
        ),
        raw_bases=_f0_anchor_tensor(
            payload["raw_bases"], label="F0 anchor frame raw_bases", floating=True
        ),
        tangent_bases=_f0_anchor_tensor(
            payload["tangent_bases"],
            label="F0 anchor frame tangent_bases",
            floating=True,
        ),
        mapped_bases=_f0_anchor_tensor(
            payload["mapped_bases"],
            label="F0 anchor frame mapped_bases",
            floating=True,
        ),
        clean_bases=_f0_anchor_tensor(
            payload["clean_bases"],
            label="F0 anchor frame clean_bases",
            floating=True,
        ),
        native_update=_f0_anchor_tensor(
            payload["native_update"],
            label="F0 anchor frame native_update",
            floating=True,
        ),
        sigma_from=_f0_anchor_tensor(
            payload["sigma_from"],
            label="F0 anchor frame sigma_from",
            floating=True,
        ),
        sigma_to=_f0_anchor_tensor(
            payload["sigma_to"],
            label="F0 anchor frame sigma_to",
            floating=True,
        ),
        kappa=_f0_anchor_tensor(
            payload["kappa"], label="F0 anchor frame kappa", floating=True
        ),
        context=_f0_anchor_tensor(
            payload["context"], label="F0 anchor frame context", floating=True
        ),
        diagnostics=_f0_diagnostics_from_payload(payload["diagnostics"]),
        prediction_type=payload["prediction_type"],
    )
    _f0_validate_frame(result)
    return result


def _f0_scalar_or_tensor_to_payload(value: Any, *, label: str) -> Any:
    if type(value) is torch.Tensor:
        return _f0_anchor_tensor(value, label=label, floating=True)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite scalar or tensor")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _f0_step_to_payload(value: Any) -> dict[str, Any]:
    from .collector import EulerRolloutStep

    if type(value) is not EulerRolloutStep:
        raise TypeError("F0 anchor step must be EulerRolloutStep")
    EulerRolloutStep(
        observation=value.observation,
        condition=value.condition,
        native_model_output=value.native_model_output,
        native_prev_sample=value.native_prev_sample,
        sigma_from=value.sigma_from,
        sigma_to=value.sigma_to,
        prediction_type=value.prediction_type,
        clean_update_gain=value.clean_update_gain,
        metadata=value.metadata,
    )
    metadata = _f0_anchor_json(dict(value.metadata), label="F0 anchor step metadata")
    if type(metadata) is not dict:
        raise ValueError("F0 anchor step metadata must be a mapping")
    return {
        "observation": _f0_observation_to_payload(value.observation),
        "condition": _f0_condition_to_payload(value.condition),
        "native_model_output": _f0_anchor_tensor(
            value.native_model_output,
            label="F0 anchor native_model_output",
            floating=True,
        ),
        "native_prev_sample": _f0_anchor_tensor(
            value.native_prev_sample,
            label="F0 anchor native_prev_sample",
            floating=True,
        ),
        "sigma_from": _f0_scalar_or_tensor_to_payload(
            value.sigma_from, label="F0 anchor sigma_from"
        ),
        "sigma_to": _f0_scalar_or_tensor_to_payload(
            value.sigma_to, label="F0 anchor sigma_to"
        ),
        "prediction_type": value.prediction_type,
        "clean_update_gain": _f0_anchor_optional_tensor(
            value.clean_update_gain, label="F0 anchor clean_update_gain"
        ),
        "metadata": metadata,
    }


def _f0_step_from_payload(value: Any) -> Any:
    from .collector import EulerRolloutStep

    payload = _f0_anchor_dict(
        value,
        {
            "observation",
            "condition",
            "native_model_output",
            "native_prev_sample",
            "sigma_from",
            "sigma_to",
            "prediction_type",
            "clean_update_gain",
            "metadata",
        },
        label="F0 anchor step payload",
    )
    metadata = _f0_anchor_json(payload["metadata"], label="F0 anchor step metadata")
    if type(metadata) is not dict:
        raise ValueError("F0 anchor step metadata must be a mapping")
    return EulerRolloutStep(
        observation=_f0_observation_from_payload(payload["observation"]),
        condition=_f0_condition_from_payload(payload["condition"]),
        native_model_output=_f0_anchor_tensor(
            payload["native_model_output"],
            label="F0 anchor native_model_output",
            floating=True,
        ),
        native_prev_sample=_f0_anchor_tensor(
            payload["native_prev_sample"],
            label="F0 anchor native_prev_sample",
            floating=True,
        ),
        sigma_from=_f0_scalar_or_tensor_to_payload(
            payload["sigma_from"], label="F0 anchor sigma_from"
        ),
        sigma_to=_f0_scalar_or_tensor_to_payload(
            payload["sigma_to"], label="F0 anchor sigma_to"
        ),
        prediction_type=payload["prediction_type"],
        clean_update_gain=_f0_anchor_optional_tensor(
            payload["clean_update_gain"], label="F0 anchor clean_update_gain"
        ),
        metadata=metadata,
    )


def _f0_transition_state_to_payload(value: Any) -> dict[str, Any]:
    from AttentionGuidance.latent_renderer import RendererObservation
    from .renderer import EulerFrameState

    if type(value) is RendererObservation:
        return {
            "type": "renderer_observation",
            "value": _f0_observation_to_payload(value),
        }
    if type(value) is EulerFrameState:
        return {"type": "euler_frame_state", "value": _f0_frame_to_payload(value)}
    raise TypeError("F0 anchor transition state has an unsupported type")


def _f0_transition_state_from_payload(value: Any) -> Any:
    payload = _f0_anchor_dict(
        value, {"type", "value"}, label="F0 anchor transition state payload"
    )
    if payload["type"] == "renderer_observation":
        return _f0_observation_from_payload(payload["value"])
    if payload["type"] == "euler_frame_state":
        return _f0_frame_from_payload(payload["value"])
    raise ValueError("F0 anchor transition state type is unsupported")


def _f0_transition_to_payload(value: Any) -> dict[str, Any]:
    from .rollout import Transition

    if type(value) is not Transition:
        raise TypeError("F0 anchor transition must be Transition")
    step_index = _f0_anchor_int(
        value.step_index, label="F0 anchor transition step_index"
    )
    return {
        "state": _f0_transition_state_to_payload(value.state),
        "action": _f0_anchor_tensor(
            value.action, label="F0 anchor transition action", floating=True
        ),
        "next_state": _f0_state_to_payload(value.next_state),
        "native_transition": _f0_anchor_tensor(
            value.native_transition,
            label="F0 anchor native_transition",
            floating=True,
        ),
        "rendered_transition": _f0_anchor_tensor(
            value.rendered_transition,
            label="F0 anchor rendered_transition",
            floating=True,
        ),
        "log_prob": _f0_anchor_optional_tensor(
            value.log_prob, label="F0 anchor transition log_prob"
        ),
        "step_index": step_index,
        "pre_squash": _f0_anchor_optional_tensor(
            value.pre_squash, label="F0 anchor transition pre_squash"
        ),
        "behavior_mean": _f0_anchor_optional_tensor(
            value.behavior_mean, label="F0 anchor transition behavior_mean"
        ),
        "reference_mean": _f0_anchor_optional_tensor(
            value.reference_mean, label="F0 anchor transition reference_mean"
        ),
    }


def _f0_transition_from_payload(value: Any) -> Any:
    from .rollout import Transition

    payload = _f0_anchor_dict(
        value,
        {
            "state",
            "action",
            "next_state",
            "native_transition",
            "rendered_transition",
            "log_prob",
            "step_index",
            "pre_squash",
            "behavior_mean",
            "reference_mean",
        },
        label="F0 anchor transition payload",
    )
    return Transition(
        state=_f0_transition_state_from_payload(payload["state"]),
        action=_f0_anchor_tensor(
            payload["action"], label="F0 anchor transition action", floating=True
        ),
        next_state=_f0_state_from_payload(payload["next_state"]),
        native_transition=_f0_anchor_tensor(
            payload["native_transition"],
            label="F0 anchor native_transition",
            floating=True,
        ),
        rendered_transition=_f0_anchor_tensor(
            payload["rendered_transition"],
            label="F0 anchor rendered_transition",
            floating=True,
        ),
        log_prob=_f0_anchor_optional_tensor(
            payload["log_prob"], label="F0 anchor transition log_prob"
        ),
        step_index=_f0_anchor_int(
            payload["step_index"], label="F0 anchor transition step_index"
        ),
        pre_squash=_f0_anchor_optional_tensor(
            payload["pre_squash"], label="F0 anchor transition pre_squash"
        ),
        behavior_mean=_f0_anchor_optional_tensor(
            payload["behavior_mean"], label="F0 anchor transition behavior_mean"
        ),
        reference_mean=_f0_anchor_optional_tensor(
            payload["reference_mean"], label="F0 anchor transition reference_mean"
        ),
    )


def _f0_stats_to_payload(value: Any) -> dict[str, Any]:
    from .collector import CollectorStats

    if type(value) is not CollectorStats:
        raise TypeError("F0 anchor stats must be CollectorStats")
    result = {}
    for name in (
        "observe_calls",
        "transition_calls",
        "decision_transitions",
        "auxiliary_transitions",
    ):
        result[name] = _f0_anchor_int(
            getattr(value, name), label=f"F0 anchor stats {name}"
        )
    if value.verified_unet_forwards is not None:
        result["verified_unet_forwards"] = _f0_anchor_int(
            value.verified_unet_forwards,
            label="F0 anchor stats verified_unet_forwards",
        )
    else:
        result["verified_unet_forwards"] = None
    return result


def _f0_stats_from_payload(value: Any) -> Any:
    from .collector import CollectorStats

    payload = _f0_anchor_dict(
        value,
        {
            "observe_calls",
            "transition_calls",
            "decision_transitions",
            "auxiliary_transitions",
            "verified_unet_forwards",
        },
        label="F0 anchor collector stats",
    )
    values = {
        name: _f0_anchor_int(
            payload[name], label=f"F0 anchor stats {name}"
        )
        for name in (
            "observe_calls",
            "transition_calls",
            "decision_transitions",
            "auxiliary_transitions",
        )
    }
    verified = payload["verified_unet_forwards"]
    if verified is not None:
        verified = _f0_anchor_int(
            verified, label="F0 anchor stats verified_unet_forwards"
        )
    return CollectorStats(**values, verified_unet_forwards=verified)


def _f0_cache_to_payload(value: Any) -> dict[str, Any]:
    from .collector import CachedEulerDecision

    if type(value) is not CachedEulerDecision:
        raise TypeError("F0 anchor cache must be CachedEulerDecision")
    normalized = CachedEulerDecision(
        decision_index=value.decision_index,
        state=value.state,
        step=value.step,
        frame=value.frame,
        checkpoint_hash=value.checkpoint_hash,
        frame_contract_hash=value.frame_contract_hash,
        calibration_hash=value.calibration_hash,
        run_contract=value.run_contract,
    )
    checkpoint = normalized.checkpoint_hash
    if checkpoint is not None:
        _f0_anchor_hash(checkpoint, label="F0 anchor cache checkpoint_hash")
    return {
        "decision_index": normalized.decision_index,
        "state": _f0_state_to_payload(normalized.state),
        "step": _f0_step_to_payload(normalized.step),
        "frame": _f0_frame_to_payload(normalized.frame),
        "checkpoint_hash": checkpoint,
        "frame_contract_hash": _f0_anchor_hash(
            normalized.frame_contract_hash,
            label="F0 anchor cache frame_contract_hash",
        ),
        "calibration_hash": _f0_anchor_hash(
            normalized.calibration_hash,
            label="F0 anchor cache calibration_hash",
        ),
        "run_contract": _f0_anchor_hash(
            normalized.run_contract, label="F0 anchor cache run_contract"
        ),
    }


def _f0_cache_from_payload(value: Any) -> Any:
    from .collector import CachedEulerDecision

    payload = _f0_anchor_dict(
        value,
        {
            "decision_index",
            "state",
            "step",
            "frame",
            "checkpoint_hash",
            "frame_contract_hash",
            "calibration_hash",
            "run_contract",
        },
        label="F0 anchor cache payload",
    )
    checkpoint = payload["checkpoint_hash"]
    if checkpoint is not None:
        checkpoint = _f0_anchor_hash(
            checkpoint, label="F0 anchor cache checkpoint_hash"
        )
    return CachedEulerDecision(
        decision_index=_f0_anchor_int(
            payload["decision_index"], label="F0 anchor cache decision_index"
        ),
        state=_f0_state_from_payload(payload["state"]),
        step=_f0_step_from_payload(payload["step"]),
        frame=_f0_frame_from_payload(payload["frame"]),
        checkpoint_hash=checkpoint,
        frame_contract_hash=_f0_anchor_hash(
            payload["frame_contract_hash"],
            label="F0 anchor cache frame_contract_hash",
        ),
        calibration_hash=_f0_anchor_hash(
            payload["calibration_hash"],
            label="F0 anchor cache calibration_hash",
        ),
        run_contract=_f0_anchor_hash(
            payload["run_contract"], label="F0 anchor cache run_contract"
        ),
    )


def _f0_values_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is torch.Tensor:
        return left.shape == right.shape and left.dtype == right.dtype and torch.equal(
            left.detach().cpu(), right.detach().cpu()
        )
    if is_dataclass(left) and not isinstance(left, type):
        return all(
            _f0_values_equal(getattr(left, item.name), getattr(right, item.name))
            for item in fields(left)
            if item.init
        )
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(
            _f0_values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            _f0_values_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _f0_validate_state_binding(
    state: Any,
    *,
    block: Any,
    checkpoint_hash: str,
    initial_conditioning: Any,
) -> None:
    from .sdxl_adapter import SdxlEulerState

    if type(state) is not SdxlEulerState:
        raise TypeError("F0 anchor trace contains a non-SDXL state")
    _f0_validate_tensor(
        state.latents,
        label="F0 anchor state latents",
        ndim=4,
    )
    if state.latents.shape[0] != 1:
        raise ValueError("F0 anchor states must contain one prompt-seed row")
    if (
        state.total_steps != 50
        or state.branch != "anchor"
        or state.prefix != 0
        or state.checkpoint_hash != checkpoint_hash
        or state.split != block.record.split
    ):
        raise ValueError("F0 anchor state differs from the bound anchor trajectory")
    conditioning = state.conditioning
    if (
        conditioning.batch_size != 1
        or conditioning.prompt_ids != (block.record.record_id,)
        or conditioning.prompts != (block.record.prompt,)
        or conditioning.generation_seeds != (block.generation_seed,)
        or not conditioning.do_classifier_free_guidance
    ):
        raise ValueError("F0 anchor state differs from its prompt-seed block")
    if not _f0_values_equal(conditioning, initial_conditioning):
        raise ValueError("F0 anchor conditioning changed within the trajectory")
    expected_actions = sum(index < state.step_index for index in (8, 24, 40))
    if len(state.action_history) != expected_actions:
        raise ValueError("F0 anchor action history differs from the frozen schedule")
    for action in state.action_history:
        try:
            action_tensor = torch.as_tensor(action)
        except (TypeError, ValueError) as exc:
            raise ValueError("F0 anchor action history is malformed") from exc
        if (
            action_tensor.shape != (1, 6)
            or not torch.isfinite(action_tensor).all()
            or torch.count_nonzero(action_tensor).item() != 0
        ):
            raise ValueError("F0 anchor action history must contain strict zero actions")


def _f0_validate_trace_binding(
    trace: Any, *, block: Any, contract: Mapping[str, Any], contract_hash: str
) -> None:
    from AttentionGuidance.latent_renderer import RendererObservation
    from .collector import (
        CachedEulerDecision,
        CollectorStats,
        F0AnchorTrace,
    )
    from .renderer import EulerFrameState
    from .rollout import Transition
    from .sdxl_adapter import SdxlEulerState

    if type(trace) is not F0AnchorTrace:
        raise TypeError("trace must be an F0AnchorTrace")
    F0AnchorTrace(
        branch=trace.branch,
        initial_state=trace.initial_state,
        terminal_state=trace.terminal_state,
        cached_decisions=trace.cached_decisions,
        transitions=trace.transitions,
        stats=trace.stats,
        checkpoint_hash=trace.checkpoint_hash,
    )
    if trace.branch != "anchor":
        raise ValueError("F0 anchor trace must use the anchor branch")
    checkpoint_hash = _f0_anchor_hash(
        trace.checkpoint_hash, label="F0 anchor trace checkpoint_hash"
    )
    if checkpoint_hash != contract.get("initial_renderer_state_sha256"):
        raise ValueError("F0 anchor checkpoint differs from the run contract")
    if type(trace.initial_state) is not SdxlEulerState or type(
        trace.terminal_state
    ) is not SdxlEulerState:
        raise TypeError("F0 anchor endpoints must be SdxlEulerState")
    initial_conditioning = trace.initial_state.conditioning
    for endpoint, step_index in (
        (trace.initial_state, 0),
        (trace.terminal_state, 50),
    ):
        _f0_validate_state_binding(
            endpoint,
            block=block,
            checkpoint_hash=checkpoint_hash,
            initial_conditioning=initial_conditioning,
        )
        if endpoint.step_index != step_index:
            raise ValueError("F0 anchor endpoint has the wrong scheduler position")
    if type(trace.stats) is not CollectorStats or trace.stats != CollectorStats(
        50, 50, 3, 47, 50
    ):
        raise ValueError("F0 anchor stats differ from the verified formal budget")
    if type(trace.cached_decisions) is not dict or set(
        trace.cached_decisions
    ) != {8, 24, 40}:
        raise ValueError("F0 anchor cache differs from the frozen schedule")
    if type(trace.transitions) is not tuple or len(trace.transitions) != 50:
        raise ValueError("F0 anchor transitions must be a 50-row tuple")

    expected_frame_hash = contract.get("renderer_frame_contract_hash")
    expected_calibration_hash = contract.get("calibration_hash")
    current_latents = trace.initial_state.latents
    initial_device = current_latents.device
    initial_shape = tuple(current_latents.shape)
    initial_dtype = current_latents.dtype
    _f0_validate_tensor(
        trace.terminal_state.latents,
        label="F0 anchor terminal state latents",
        ndim=4,
    )
    if (
        trace.terminal_state.latents.shape != initial_shape
        or trace.terminal_state.latents.device != initial_device
        or trace.terminal_state.latents.dtype != initial_dtype
    ):
        raise ValueError("F0 anchor endpoint tensor layout differs from its initial state")
    schedule_sha256: str | None = None
    for index, transition in enumerate(trace.transitions):
        if type(transition) is not Transition or transition.step_index != index:
            raise ValueError("F0 anchor transitions are out of order")
        expected_state_type = EulerFrameState if index in {8, 24, 40} else RendererObservation
        if type(transition.state) is not expected_state_type:
            raise TypeError("F0 anchor transition state type differs from its schedule")
        if type(transition.state) is RendererObservation:
            _f0_validate_observation(
                transition.state,
                label=f"F0 anchor transition[{index}].state",
                expected_step=index,
            )
        else:
            _f0_validate_frame(transition.state)
        state_latents = (
            transition.state.sample
            if type(transition.state) is EulerFrameState
            else transition.state.latents_before_step
        )
        if not _f0_values_equal(state_latents, current_latents):
            raise ValueError("F0 anchor transition chain is discontinuous")
        if type(transition.next_state) is not SdxlEulerState:
            raise TypeError("F0 anchor transition next_state must be SdxlEulerState")
        _f0_validate_state_binding(
            transition.next_state,
            block=block,
            checkpoint_hash=checkpoint_hash,
            initial_conditioning=initial_conditioning,
        )
        if transition.next_state.step_index != index + 1:
            raise ValueError("F0 anchor transition next_state is out of order")
        _f0_validate_tensor(
            transition.next_state.latents,
            label=f"F0 anchor transition[{index}].next_state.latents",
            shape=initial_shape,
            device=initial_device,
        )
        if transition.next_state.latents.dtype != initial_dtype:
            raise ValueError("F0 anchor transition next_state has a dtype drift")
        if (
            type(transition.action) is not torch.Tensor
            or transition.action.shape != (1, 6)
            or transition.action.device != initial_device
            or transition.action.requires_grad
            or not transition.action.is_floating_point()
            or transition.action.dtype != initial_dtype
            or not torch.isfinite(transition.action).all()
            or torch.count_nonzero(transition.action).item() != 0
        ):
            raise ValueError("F0 anchor transitions require detached zero actions")
        if any(
            value is not None
            for value in (
                transition.log_prob,
                transition.pre_squash,
                transition.behavior_mean,
                transition.reference_mean,
            )
        ):
            raise ValueError("F0 anchor transitions cannot contain policy capabilities")
        if (
            type(transition.native_transition) is not torch.Tensor
            or type(transition.rendered_transition) is not torch.Tensor
            or transition.native_transition.shape != initial_shape
            or transition.rendered_transition.shape != initial_shape
            or transition.native_transition.device != initial_device
            or transition.rendered_transition.device != initial_device
            or not transition.native_transition.is_floating_point()
            or not transition.rendered_transition.is_floating_point()
            or not torch.isfinite(transition.native_transition).all()
            or not torch.isfinite(transition.rendered_transition).all()
            or not _f0_values_equal(
                transition.native_transition, transition.rendered_transition
            )
        ):
            raise ValueError("F0 anchor transition is not a strict native no-op")
        expected_native_transition = (
            transition.state.native_update
            if type(transition.state) is EulerFrameState
            else transition.state.scheduler_update
        )
        _f0_tensor_relation(
            transition.native_transition,
            expected_native_transition,
            label=(
                "F0 anchor native transition differs from its observation/frame update"
            ),
            require_dtype=False,
        )
        expected_delta = transition.next_state.latents - current_latents
        if not _f0_values_equal(expected_delta, transition.native_transition):
            raise ValueError("F0 anchor transition bytes disagree with its states")
        if index in {8, 24, 40}:
            cached = trace.cached_decisions[index]
            if type(cached) is not CachedEulerDecision:
                raise TypeError("F0 anchor cache has an unsupported record")
            if (
                cached.decision_index != index
                or cached.checkpoint_hash != checkpoint_hash
                or cached.frame_contract_hash != expected_frame_hash
                or cached.calibration_hash != expected_calibration_hash
                or cached.run_contract != contract_hash
            ):
                raise ValueError("F0 anchor cache provenance is mismatched")
            _f0_validate_state_binding(
                cached.state,
                block=block,
                checkpoint_hash=checkpoint_hash,
                initial_conditioning=initial_conditioning,
            )
            if cached.state.step_index != index or not _f0_values_equal(
                cached.state.latents, current_latents
            ):
                raise ValueError("F0 anchor cached state is out of order")
            _f0_validate_step(
                cached.step,
                label=f"F0 anchor cached decision[{index}].step",
                expected_step=index,
            )
            _f0_validate_frame(cached.frame)
            metadata = _f0_anchor_json(
                dict(cached.step.metadata),
                label=f"F0 anchor cached decision[{index}] metadata",
            )
            if set(metadata) != {
                "schema",
                "schedule_sha256",
                "prompt_ids",
                "branch",
                "step_index",
                "physical_unet_forwards",
            }:
                raise ValueError("F0 anchor cached step metadata has an unsupported shape")
            if metadata.get("schema") != "repldm.sdxl_euler_observation.v1":
                raise ValueError("F0 anchor cached step metadata schema is mismatched")
            metadata_schedule = metadata.get("schedule_sha256")
            _f0_anchor_hash(
                metadata_schedule,
                label="F0 anchor cached step schedule_sha256",
            )
            if schedule_sha256 is None:
                schedule_sha256 = metadata_schedule
            elif metadata_schedule != schedule_sha256:
                raise ValueError("F0 anchor cached steps use different schedules")
            if metadata.get("prompt_ids") != [block.record.record_id]:
                raise ValueError("F0 anchor cached step prompt metadata is mismatched")
            if metadata.get("branch") != "anchor" or metadata.get("step_index") != index:
                raise ValueError("F0 anchor cached step metadata is out of order")
            if metadata.get("physical_unet_forwards") != 1:
                raise ValueError("F0 anchor cached step must record one physical U-Net forward")
            if (
                cached.step.observation.step_index != index
                or not _f0_values_equal(
                    cached.step.observation.latents_before_step, current_latents
                )
                or not _f0_values_equal(cached.frame, transition.state)
            ):
                raise ValueError("F0 anchor cached observation or frame is mismatched")
            if not all(
                _f0_values_equal(left, right)
                for left, right in (
                    (cached.frame.sample, cached.step.observation.latents_before_step),
                    (
                        cached.frame.clean_latent,
                        cached.step.observation.pred_original_sample,
                    ),
                    (cached.frame.native_update, cached.step.observation.scheduler_update),
                    (cached.frame.raw_bases, cached.step.condition.bases),
                )
            ):
                raise ValueError("F0 anchor cached frame differs from its observation")
            if cached.frame.native_model_output is None:
                raise ValueError("F0 anchor cached frame is missing its model output")
            _f0_tensor_relation(
                cached.frame.native_model_output,
                cached.step.native_model_output,
                label="F0 anchor cached model output",
                require_dtype=True,
                rtol=0.0,
                atol=0.0,
            )
            _f0_tensor_relation(
                cached.step.native_prev_sample,
                transition.next_state.latents,
                label="F0 anchor cached native_prev_sample successor",
                require_dtype=False,
            )
            _f0_tensor_relation(
                cached.step.observation.scheduler_update,
                cached.step.native_prev_sample - cached.step.observation.latents_before_step,
                label="F0 anchor cached scheduler update",
                require_dtype=False,
            )
            frame_sigma_from = _f0_batch_scalar(
                cached.frame.sigma_from,
                label="F0 anchor cached frame sigma_from",
                batch=cached.frame.sample.shape[0],
                device=cached.frame.sample.device,
            )
            step_sigma_from = _f0_batch_scalar(
                cached.step.sigma_from,
                label="F0 anchor cached step sigma_from",
                batch=cached.frame.sample.shape[0],
                device=cached.frame.sample.device,
            )
            _f0_tensor_relation(
                frame_sigma_from,
                step_sigma_from,
                label="F0 anchor cached sigma_from",
                require_dtype=False,
            )
            frame_sigma_to = _f0_batch_scalar(
                cached.frame.sigma_to,
                label="F0 anchor cached frame sigma_to",
                batch=cached.frame.sample.shape[0],
                device=cached.frame.sample.device,
            )
            step_sigma_to = _f0_batch_scalar(
                cached.step.sigma_to,
                label="F0 anchor cached step sigma_to",
                batch=cached.frame.sample.shape[0],
                device=cached.frame.sample.device,
            )
            _f0_tensor_relation(
                frame_sigma_to,
                step_sigma_to,
                label="F0 anchor cached sigma_to",
                require_dtype=False,
            )
            if cached.frame.prediction_type != cached.step.prediction_type:
                raise ValueError("F0 anchor cached prediction types are mismatched")
            if cached.step.clean_update_gain is not None:
                frame_kappa = _f0_batch_scalar(
                    cached.frame.kappa,
                    label="F0 anchor cached frame kappa",
                    batch=cached.frame.sample.shape[0],
                    device=cached.frame.sample.device,
                )
                step_gain = _f0_batch_scalar(
                    cached.step.clean_update_gain,
                    label="F0 anchor cached clean_update_gain",
                    batch=cached.frame.sample.shape[0],
                    device=cached.frame.sample.device,
                )
                _f0_tensor_relation(
                    frame_kappa,
                    step_gain,
                    label="F0 anchor cached clean update gain",
                    require_dtype=False,
                )
        current_latents = transition.next_state.latents
    if not _f0_values_equal(current_latents, trace.terminal_state.latents):
        raise ValueError("F0 anchor terminal state differs from its transition chain")


def _f0_trace_to_payload(
    value: Any, *, block: Any, contract: Mapping[str, Any], contract_hash: str
) -> dict[str, Any]:
    _f0_validate_trace_binding(
        value, block=block, contract=contract, contract_hash=contract_hash
    )
    return {
        "branch": value.branch,
        "initial_state": _f0_state_to_payload(value.initial_state),
        "terminal_state": _f0_state_to_payload(value.terminal_state),
        "cached_decisions": {
            str(index): _f0_cache_to_payload(value.cached_decisions[index])
            for index in (8, 24, 40)
        },
        "transitions": [
            _f0_transition_to_payload(transition)
            for transition in value.transitions
        ],
        "stats": _f0_stats_to_payload(value.stats),
        "checkpoint_hash": value.checkpoint_hash,
    }


def _f0_trace_from_payload(
    value: Any, *, block: Any, contract: Mapping[str, Any], contract_hash: str
) -> Any:
    from .collector import F0AnchorTrace

    payload = _f0_anchor_dict(
        value,
        {
            "branch",
            "initial_state",
            "terminal_state",
            "cached_decisions",
            "transitions",
            "stats",
            "checkpoint_hash",
        },
        label="F0 anchor trace payload",
    )
    caches = _f0_anchor_dict(
        payload["cached_decisions"],
        {"8", "24", "40"},
        label="F0 anchor cached decisions",
    )
    transitions = payload["transitions"]
    if type(transitions) is not list or len(transitions) != 50:
        raise ValueError("F0 anchor transitions payload must have 50 rows")
    result = F0AnchorTrace(
        branch=payload["branch"],
        initial_state=_f0_state_from_payload(payload["initial_state"]),
        terminal_state=_f0_state_from_payload(payload["terminal_state"]),
        cached_decisions={
            index: _f0_cache_from_payload(caches[str(index)])
            for index in (8, 24, 40)
        },
        transitions=tuple(
            _f0_transition_from_payload(transition) for transition in transitions
        ),
        stats=_f0_stats_from_payload(payload["stats"]),
        checkpoint_hash=_f0_anchor_hash(
            payload["checkpoint_hash"],
            label="F0 anchor trace checkpoint_hash",
        ),
    )
    _f0_validate_trace_binding(
        result, block=block, contract=contract, contract_hash=contract_hash
    )
    return result


@dataclass(frozen=True)
class CheckpointProvenance:
    """Immutable identity of a behavior or reference renderer checkpoint."""

    path: str
    sha256: str
    bytes: int
    renderer_state_sha256: str

    def __post_init__(self) -> None:
        checkpoint = Path(self.path)
        if not checkpoint.is_absolute() or str(checkpoint.resolve()) != self.path:
            raise ValueError("checkpoint provenance path must be normalized and absolute")
        if _HASH_RE.fullmatch(self.sha256) is None:
            raise ValueError("checkpoint sha256 must be a lowercase SHA-256 hash")
        if _HASH_RE.fullmatch(self.renderer_state_sha256) is None:
            raise ValueError("renderer_state_sha256 must be a lowercase SHA-256 hash")
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, int) or self.bytes <= 0:
            raise ValueError("checkpoint byte count must be positive")

    @classmethod
    def capture(
        cls, path: str | Path, *, renderer_state_sha256: str
    ) -> "CheckpointProvenance":
        checkpoint = Path(path).resolve(strict=True)
        digest, size = _hash_file(checkpoint)
        return cls(str(checkpoint), digest, size, renderer_state_sha256)

    def validate_current(self) -> None:
        digest, size = _hash_file(Path(self.path))
        if digest != self.sha256 or size != self.bytes:
            raise RuntimeError("renderer checkpoint bytes changed after provenance capture")


class AtomicRolloutStore:
    """Publish one detached rollout only after data and provenance are durable."""

    def __init__(self, authorization_binding: Any) -> None:
        binding = require_authorization_binding(authorization_binding)
        binding.validate_current()
        self.authorization_binding = binding
        self.contract_hash = binding.contract_hash
        self.root = Path(binding.contract["paths"]["rollout_dir"])
        self._binding_identity = id(binding)

    def _preflight(self) -> None:
        binding = require_authorization_binding(self.authorization_binding)
        if id(binding) != self._binding_identity:
            raise RuntimeError("rollout-store authorization binding was replaced")
        if Path(binding.contract["paths"]["rollout_dir"]) != self.root:
            raise RuntimeError("rollout-store path changed after construction")
        binding.validate_current()

    def _successful_opd_label_receipts(
        self,
    ) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
        """Read and verify the successful label receipts for this run.

        A persisted reservation ID is only meaningful when it resolves to the
        sealed ledger belonging to the same authorization binding.  Returning
        the durable reservation/receipt pairs also avoids treating a caller's
        arbitrary string as a provenance capability.
        """
        from .ledger import QueryLedger

        contract = self.authorization_binding.contract
        ledger = QueryLedger(
            contract["paths"]["ledger_path"],
            contract["query_budget"],
            run_contract=self.contract_hash,
            strict_provenance=True,
            authorization_binding=self.authorization_binding,
        )
        pairs = ledger.successful_receipt_pairs("label")
        result: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for reservation, receipt in pairs:
            reservation_id = reservation.get("reservation_id")
            if not isinstance(reservation_id, str) or not reservation_id:
                raise RuntimeError("the OPD label ledger contains an invalid reservation ID")
            if reservation_id in result:
                raise RuntimeError("the OPD label ledger contains a duplicate reservation ID")
            result[reservation_id] = (reservation, receipt)
        return result

    def _validate_opd_payload(
        self,
        collection: RolloutCollection,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Validate optional detached OPD targets and their JSON provenance."""
        targets = getattr(collection, "opd_teacher_targets", None)
        provenance = getattr(collection, "opd_teacher_label_provenance", None)
        if targets is None and provenance is None:
            if metadata is not None and (
                "teacher_labels" in metadata
                or metadata.get("schema") == "repldm.scored_opd_rollout.v1"
            ):
                raise ValueError("OPD metadata requires persisted teacher targets and provenance")
            return
        if targets is None or provenance is None:
            raise ValueError("OPD targets and provenance must be persisted together")
        if not isinstance(targets, (tuple, list)) or len(targets) != 3:
            raise ValueError("OPD collection must contain exactly three teacher targets")
        if not isinstance(provenance, (tuple, list)) or len(provenance) != 3:
            raise ValueError("OPD collection must contain exactly three target provenance records")
        expected_indices = (8, 24, 40)
        for expected_index, target, record in zip(expected_indices, targets, provenance):
            if (
                not isinstance(target, torch.Tensor)
                or target.ndim != 4
                or not target.is_floating_point()
                or target.requires_grad
                or not torch.isfinite(target).all()
            ):
                raise ValueError("persisted OPD teacher targets must be detached finite NCHW tensors")
            if not isinstance(record, Mapping):
                raise ValueError("persisted OPD target provenance must be mappings")
            try:
                # This also rejects tensors, NaNs, non-string mapping keys, and
                # other values that cannot survive a canonical manifest.
                _canonical(dict(record))
            except (TypeError, ValueError) as exc:
                raise ValueError("persisted OPD target provenance is not canonical JSON") from exc
            if record.get("decision_index") != expected_index:
                raise ValueError("persisted OPD target decision differs from the registered schedule")
            target_hash = record.get("target_sha256")
            raw_hash = record.get("raw_target_sha256", target_hash)
            receipt_hash = record.get("label_receipt_output_sha256")
            for name, value in (
                ("target_sha256", target_hash),
                ("raw_target_sha256", raw_hash),
                ("label_receipt_output_sha256", receipt_hash),
                ("teacher_checkpoint_sha256", record.get("teacher_checkpoint_sha256")),
                ("teacher_state_sha256", record.get("teacher_state_sha256", record.get("teacher_checkpoint_sha256"))),
            ):
                if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
                    raise ValueError(f"persisted OPD provenance {name} is not a SHA-256 hash")
            if target_hash != tensor_sha256(target) or raw_hash != target_hash:
                raise ValueError("persisted OPD target bytes differ from provenance hashes")
            if receipt_hash != operation_output_sha256(target):
                raise ValueError("persisted OPD receipt does not bind the target bytes")
            shape = record.get("target_shape")
            dtype = record.get("target_dtype")
            if shape is not None and list(target.shape) != list(shape):
                raise ValueError("persisted OPD target shape differs from provenance")
            if dtype is not None and str(target.dtype) != dtype:
                raise ValueError("persisted OPD target dtype differs from provenance")
            if "frame_valid" in record and type(record["frame_valid"]) is not bool:
                raise ValueError("persisted OPD frame_valid must be boolean")

        formal_metadata = metadata is not None and (
            "teacher_labels" in metadata
            or metadata.get("schema") == "repldm.scored_opd_rollout.v1"
        )
        if formal_metadata:
            label_receipts = self._successful_opd_label_receipts()
            labels = metadata.get("teacher_labels")
            if not isinstance(labels, (tuple, list)) or len(labels) != 3:
                raise ValueError("OPD metadata must contain exactly three teacher labels")
            for persisted, manifest in zip(provenance, labels):
                if not isinstance(manifest, Mapping):
                    raise ValueError("OPD metadata teacher labels must be mappings")
                if _canonical(dict(persisted)) != _canonical(dict(manifest)):
                    raise ValueError("OPD metadata teacher label differs from payload provenance")
            target_hashes = metadata.get("teacher_target_sha256")
            if target_hashes is not None:
                if list(target_hashes) != [
                    record.get("raw_target_sha256", record.get("target_sha256"))
                    for record in provenance
                ]:
                    raise ValueError("OPD metadata raw target hashes differ from payload")
            file_hash = metadata.get("teacher_checkpoint_file_sha256")
            if file_hash is not None and (
                not isinstance(file_hash, str) or _HASH_RE.fullmatch(file_hash) is None
            ):
                raise ValueError("OPD metadata checkpoint file hash is invalid")
            state_hash = metadata.get("teacher_state_sha256", metadata.get("teacher_checkpoint_sha256"))
            if state_hash is not None and (
                not isinstance(state_hash, str) or _HASH_RE.fullmatch(state_hash) is None
            ):
                raise ValueError("OPD metadata teacher state hash is invalid")
            for record in provenance:
                if file_hash is not None and record.get("teacher_checkpoint_file_sha256") != file_hash:
                    raise ValueError("OPD payload checkpoint file hash differs from metadata")
                if state_hash is not None and record.get("teacher_state_sha256", record.get("teacher_checkpoint_sha256")) != state_hash:
                    raise ValueError("OPD payload teacher state hash differs from metadata")
            # Reuse the typed label validator for the complete receipt/context
            # contract.  Import lazily to avoid making storage part of the
            # framework-neutral objective module's import graph.
            from .methods.common import (
                OpdTeacherLabel,
                _validate_opd_label_binding,
            )

            anchor_transitions = collection.branches["anchor"].transitions
            for target, record, transition in zip(
                targets, provenance, anchor_transitions
            ):
                required = {
                    "decision_index",
                    "target_sha256",
                    "label_receipt_output_sha256",
                    "frame_valid",
                    "teacher_checkpoint_sha256",
                    "teacher_checkpoint_file_sha256",
                    "teacher_state_sha256",
                    "reservation_id",
                    "operation_context",
                    "student_state_sha256",
                    "raw_target_sha256",
                    "target_shape",
                    "target_dtype",
                }
                if not required.issubset(record):
                    raise ValueError("formal OPD provenance is incomplete")
                label = OpdTeacherLabel(
                    decision_index=record["decision_index"],
                    target=target,
                    target_sha256=record["target_sha256"],
                    receipt_output_sha256=record["label_receipt_output_sha256"],
                    frame_valid=record["frame_valid"],
                    teacher_checkpoint_sha256=record["teacher_checkpoint_sha256"],
                    teacher_state_sha256=record.get(
                        "teacher_state_sha256", record["teacher_checkpoint_sha256"]
                    ),
                    teacher_checkpoint_file_sha256=record[
                        "teacher_checkpoint_file_sha256"
                    ],
                    reservation_id=record["reservation_id"],
                    operation_context=record["operation_context"],
                    student_state_sha256=record["student_state_sha256"],
                    raw_target_sha256=record["raw_target_sha256"],
                )
                label.validate(strict=True)
                _validate_opd_label_binding(
                    label,
                    transition,
                    strict=True,
                    teacher_checkpoint_hash=label.teacher_state_sha256,
                    teacher_checkpoint_file_hash=label.teacher_checkpoint_file_sha256,
                )
                durable_pair = label_receipts.get(label.reservation_id)
                if durable_pair is None:
                    raise ValueError(
                        "formal OPD label reservation has no successful ledger receipt"
                    )
                reservation, receipt_row = durable_pair
                reservation_metadata = reservation.get("metadata")
                result = receipt_row.get("result")
                if not isinstance(reservation_metadata, Mapping) or not isinstance(
                    result, Mapping
                ):
                    raise ValueError("formal OPD label receipt is missing ledger provenance")
                allocation = reservation_metadata.get("method_allocation")
                if allocation != {
                    "kind": "label",
                    "amount": 1,
                    "method": self.authorization_binding.contract.get("method"),
                }:
                    raise ValueError("formal OPD label receipt has the wrong allocation")
                if (
                    reservation.get("kind") != "label"
                    or reservation.get("amount") != 1
                    or receipt_row.get("kind") != "label"
                    or receipt_row.get("amount") != 1
                    or receipt_row.get("success") is not True
                    or receipt_row.get("metadata") != reservation_metadata
                    or result.get("output_hash") != label.receipt_output_sha256
                    or result.get("scalar_or_gradient") != "teacher_transition"
                    or result.get("parent_reservation_id") is not None
                ):
                    raise ValueError("formal OPD label receipt differs from the label provenance")
                context = record["operation_context"]
                context_fields = (
                    "split",
                    "prompt",
                    "seed",
                    "step",
                    "prefix",
                    "branch",
                    "action",
                    "checkpoint_hash",
                    "image_hash",
                    "cached_parent",
                )
                expected_context = {
                    field: reservation_metadata.get(field) for field in context_fields
                }
                if _canonical(expected_context) != _canonical(dict(context)):
                    raise ValueError(
                        "formal OPD label operation context differs from the ledger receipt"
                    )

    def _validate_collection(
        self,
        collection: RolloutCollection,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(collection, RolloutCollection):
            raise TypeError("collection must be a RolloutCollection")
        if collection.preserve_graph:
            raise ValueError("a rollout must be detached before persistent storage")
        if collection.total_steps != 50 or collection.prefix_steps != 8:
            raise ValueError("rollout does not use the registered 50-step shared prefix")
        if set(collection.branches) != {"plus", "minus", "anchor"}:
            raise ValueError("rollout must contain plus, minus, and anchor branches")
        if set(collection.terminal_states) != {"plus", "minus", "anchor"}:
            raise ValueError("rollout terminal states are incomplete")
        if [proposal.step_index for proposal in collection.proposals] != [8, 24, 40]:
            raise ValueError("rollout decisions differ from the registered schedule")
        self._validate_opd_payload(collection, metadata)

    def save(
        self,
        block_id: str,
        collection: RolloutCollection,
        *,
        behavior_checkpoint: CheckpointProvenance,
        reference_checkpoint: CheckpointProvenance,
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Atomically publish a data file followed by its hash-addressed manifest."""
        self._preflight()
        if not isinstance(block_id, str) or _BLOCK_RE.fullmatch(block_id) is None:
            raise ValueError("block_id must be a stable filename-safe identifier")
        if not isinstance(metadata, Mapping):
            raise TypeError("rollout metadata must be a mapping")
        self._validate_collection(collection, metadata)
        if not isinstance(behavior_checkpoint, CheckpointProvenance) or not isinstance(
            reference_checkpoint, CheckpointProvenance
        ):
            raise TypeError("behavior and reference checkpoint provenance are required")
        behavior_checkpoint.validate_current()
        reference_checkpoint.validate_current()
        if reference_checkpoint.renderer_state_sha256 != self.authorization_binding.contract[
            "initial_renderer_state_sha256"
        ]:
            raise ValueError("reference checkpoint is not the frozen initial renderer")
        metadata_value = json.loads(_canonical(dict(metadata)).decode("utf-8"))
        _prepare_directory(self.root)
        manifest_path = self.root / f"{block_id}.json"
        if manifest_path.exists() or manifest_path.is_symlink():
            raise FileExistsError(f"rollout block already exists: {block_id}")

        envelope = {
            "schema": ROLLOUT_ENVELOPE_SCHEMA,
            "run_contract_sha256": self.contract_hash,
            "block_id": block_id,
            "behavior_checkpoint": asdict(behavior_checkpoint),
            "reference_checkpoint": asdict(reference_checkpoint),
            "metadata": metadata_value,
            "collection": collection,
        }
        temporary_data = self.root / f".{block_id}.{uuid.uuid4().hex}.pt.tmp"
        try:
            with temporary_data.open("xb") as handle:
                torch.save(envelope, handle)
                handle.flush()
                os.fsync(handle.fileno())
            data_hash, data_bytes = _hash_file(temporary_data)
            data_path = self.root / f"{block_id}-{data_hash[:16]}.pt"
            if data_path.exists() or data_path.is_symlink():
                raise FileExistsError(f"rollout payload already exists: {data_path.name}")
            temporary_data.replace(data_path)
            _fsync_directory(self.root)
            self._preflight()
            behavior_checkpoint.validate_current()
            reference_checkpoint.validate_current()
            manifest = {
                "schema": ROLLOUT_MANIFEST_SCHEMA,
                "status": "complete",
                "run_contract_sha256": self.contract_hash,
                "block_id": block_id,
                "payload": {
                    "path": str(data_path),
                    "sha256": data_hash,
                    "bytes": data_bytes,
                },
                "behavior_checkpoint": asdict(behavior_checkpoint),
                "reference_checkpoint": asdict(reference_checkpoint),
                "metadata": metadata_value,
            }
            temporary_manifest = self.root / f".{block_id}.{uuid.uuid4().hex}.json.tmp"
            try:
                with temporary_manifest.open("xb") as handle:
                    handle.write(_canonical(manifest) + b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary_manifest.replace(manifest_path)
                _fsync_directory(self.root)
            finally:
                temporary_manifest.unlink(missing_ok=True)
            return manifest
        finally:
            temporary_data.unlink(missing_ok=True)

    def load(
        self,
        block_id: str,
        *,
        behavior_checkpoint: CheckpointProvenance,
        reference_checkpoint: CheckpointProvenance,
    ) -> RolloutCollection:
        """Validate every bound byte before returning a stored collection."""
        self._preflight()
        if not isinstance(block_id, str) or _BLOCK_RE.fullmatch(block_id) is None:
            raise ValueError("block_id must be a stable filename-safe identifier")
        manifest_path = self.root / f"{block_id}.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("rollout manifest is missing or not an ordinary file")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("rollout manifest is unreadable") from exc
        required = {
            "schema",
            "status",
            "run_contract_sha256",
            "block_id",
            "payload",
            "behavior_checkpoint",
            "reference_checkpoint",
            "metadata",
        }
        if not isinstance(manifest, dict) or set(manifest) != required:
            raise ValueError("rollout manifest has an unsupported shape")
        expected = {
            "schema": ROLLOUT_MANIFEST_SCHEMA,
            "status": "complete",
            "run_contract_sha256": self.contract_hash,
            "block_id": block_id,
            "behavior_checkpoint": asdict(behavior_checkpoint),
            "reference_checkpoint": asdict(reference_checkpoint),
        }
        for field, value in expected.items():
            if manifest.get(field) != value:
                raise ValueError(f"rollout manifest field {field} is mismatched")
        behavior_checkpoint.validate_current()
        reference_checkpoint.validate_current()
        payload = manifest["payload"]
        if not isinstance(payload, Mapping) or set(payload) != {"path", "sha256", "bytes"}:
            raise ValueError("rollout payload descriptor is invalid")
        data_path = Path(payload["path"])
        if (
            data_path.parent != self.root
            or not isinstance(payload["sha256"], str)
            or data_path.name != f"{block_id}-{payload['sha256'][:16]}.pt"
        ):
            raise ValueError("rollout payload path is not content-addressed inside rollout_dir")
        digest, size = _hash_file(data_path)
        if digest != payload["sha256"] or size != payload["bytes"]:
            raise RuntimeError("rollout payload bytes differ from the manifest")
        envelope = torch.load(data_path, map_location="cpu", weights_only=False)
        if not isinstance(envelope, Mapping):
            raise ValueError("rollout payload is not a mapping")
        for field, value in (
            ("schema", ROLLOUT_ENVELOPE_SCHEMA),
            ("run_contract_sha256", self.contract_hash),
            ("block_id", block_id),
            ("behavior_checkpoint", asdict(behavior_checkpoint)),
            ("reference_checkpoint", asdict(reference_checkpoint)),
            ("metadata", manifest["metadata"]),
        ):
            if envelope.get(field) != value:
                raise ValueError(f"rollout payload field {field} is mismatched")
        collection = envelope.get("collection")
        self._validate_collection(collection, manifest["metadata"])
        self._preflight()
        return collection


class AtomicF0TargetStore:
    """Publish weights-only F0 targets under one immutable run capability."""

    def __init__(self, authorization_binding: Any) -> None:
        binding = require_authorization_binding(authorization_binding)
        binding.validate_current()
        if binding.contract.get("method") != "f0":
            raise ValueError("F0 target storage requires an F0 run contract")
        run_dir_value = binding.contract.get("paths", {}).get("run_dir")
        if not isinstance(run_dir_value, str):
            raise ValueError("F0 run contract has no run_dir")
        run_dir = Path(run_dir_value)
        if not run_dir.is_absolute() or Path(os.path.abspath(run_dir)) != run_dir:
            raise ValueError("F0 run_dir must be normalized and absolute")
        self.authorization_binding = binding
        self.contract_hash = binding.contract_hash
        self.root = run_dir / "f0_targets"
        self._binding_identity = id(binding)

    def _preflight(self) -> None:
        binding = require_authorization_binding(self.authorization_binding)
        if id(binding) != self._binding_identity:
            raise RuntimeError("F0-target-store authorization binding was replaced")
        if binding.contract_hash != self.contract_hash:
            raise RuntimeError("F0-target-store run contract changed")
        expected = Path(binding.contract["paths"]["run_dir"]) / "f0_targets"
        if expected != self.root or binding.contract.get("method") != "f0":
            raise RuntimeError("F0-target-store contract changed after construction")
        binding.validate_current()

    @staticmethod
    def _stable_id(value: Any) -> str:
        if not isinstance(value, str) or _BLOCK_RE.fullmatch(value) is None:
            raise ValueError("F0 target stable_id must be filename-safe")
        return value

    def save(self, row: Any) -> dict[str, Any]:
        """Atomically publish one content-addressed target and its commit manifest."""
        from .f0_targets import F0TargetRow, f0_target_to_payload

        self._preflight()
        if not isinstance(row, F0TargetRow):
            raise TypeError("row must be an F0TargetRow")
        stable_id = self._stable_id(row.stable_id)
        record_hash = row.record_sha256
        if _HASH_RE.fullmatch(record_hash) is None:
            raise ValueError("F0 target record hash is invalid")
        target_payload = f0_target_to_payload(row)
        if target_payload.get("record_sha256") != record_hash:
            raise RuntimeError("F0 target serialization changed its record hash")
        self._preflight()
        _prepare_f0_directory(self.root, create=True)
        manifest_path = self.root / f"{stable_id}.json"
        if os.path.lexists(manifest_path):
            raise FileExistsError(f"F0 target stable_id already exists: {stable_id}")

        temporary_payload = self.root / f".{stable_id}.{uuid.uuid4().hex}.pt.tmp"
        temporary_manifest = self.root / f".{stable_id}.{uuid.uuid4().hex}.json.tmp"
        data_path: Path | None = None
        payload_published = False
        manifest_published = False
        completed = False
        try:
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(temporary_payload, flags, 0o600)
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as handle:
                    torch.save(
                        {
                            "schema": F0_TARGET_ENVELOPE_SCHEMA,
                            "run_contract_sha256": self.contract_hash,
                            "stable_id": stable_id,
                            "record_sha256": record_hash,
                            "target": target_payload,
                        },
                        handle,
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                os.close(descriptor)
            _payload_bytes, payload_hash, payload_size = _read_f0_ordinary_file(
                temporary_payload, label="temporary F0 target payload"
            )
            data_path = self.root / f"{stable_id}-{payload_hash[:16]}.pt"
            self._preflight()
            _publish_f0_exclusive(
                temporary_payload, data_path, label="F0 target payload"
            )
            payload_published = True

            manifest_core = {
                "schema": F0_TARGET_MANIFEST_SCHEMA,
                "status": "complete",
                "run_contract_sha256": self.contract_hash,
                "stable_id": stable_id,
                "record_sha256": record_hash,
                "payload": {
                    "path": str(data_path),
                    "sha256": payload_hash,
                    "bytes": payload_size,
                },
            }
            manifest = {
                **manifest_core,
                "manifest_sha256": _f0_manifest_hash(manifest_core),
            }
            manifest_bytes = _canonical(manifest) + b"\n"
            manifest_descriptor = os.open(temporary_manifest, flags, 0o600)
            try:
                with os.fdopen(manifest_descriptor, "wb", closefd=False) as handle:
                    handle.write(manifest_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                os.close(manifest_descriptor)
            self._preflight()
            _publish_f0_exclusive(
                temporary_manifest, manifest_path, label="F0 target manifest"
            )
            manifest_published = True
            self._preflight()
            completed = True
            return manifest
        finally:
            temporary_payload.unlink(missing_ok=True)
            temporary_manifest.unlink(missing_ok=True)
            if not completed:
                if manifest_published:
                    manifest_path.unlink(missing_ok=True)
                if payload_published and data_path is not None:
                    data_path.unlink(missing_ok=True)
                if manifest_published or payload_published:
                    _fsync_directory(self.root)

    def load(self, stable_id: str) -> Any:
        """Authenticate manifest and payload bytes before safe target decoding."""
        from .f0_targets import f0_target_from_payload

        self._preflight()
        identifier = self._stable_id(stable_id)
        _prepare_f0_directory(self.root, create=False)
        manifest_path = self.root / f"{identifier}.json"
        manifest_bytes, _manifest_file_hash, _manifest_size = (
            _read_f0_ordinary_file(manifest_path, label="F0 target manifest")
        )
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("F0 target manifest is unreadable") from exc
        required = {
            "schema",
            "status",
            "run_contract_sha256",
            "stable_id",
            "record_sha256",
            "payload",
            "manifest_sha256",
        }
        if not isinstance(manifest, dict) or set(manifest) != required:
            raise ValueError("F0 target manifest has an unsupported shape")
        if manifest_bytes != _canonical(manifest) + b"\n":
            raise RuntimeError("F0 target manifest bytes are not canonical")
        manifest_hash = manifest.get("manifest_sha256")
        manifest_core = {
            key: value for key, value in manifest.items() if key != "manifest_sha256"
        }
        if (
            not isinstance(manifest_hash, str)
            or _HASH_RE.fullmatch(manifest_hash) is None
            or _f0_manifest_hash(manifest_core) != manifest_hash
        ):
            raise RuntimeError("F0 target manifest SHA-256 is invalid")
        expected = {
            "schema": F0_TARGET_MANIFEST_SCHEMA,
            "status": "complete",
            "run_contract_sha256": self.contract_hash,
            "stable_id": identifier,
        }
        for field, value in expected.items():
            if manifest.get(field) != value:
                raise ValueError(f"F0 target manifest field {field} is mismatched")
        record_hash = manifest.get("record_sha256")
        if not isinstance(record_hash, str) or _HASH_RE.fullmatch(record_hash) is None:
            raise ValueError("F0 target manifest record hash is invalid")
        payload = manifest.get("payload")
        if not isinstance(payload, Mapping) or set(payload) != {
            "path",
            "sha256",
            "bytes",
        }:
            raise ValueError("F0 target payload descriptor is invalid")
        path_value = payload.get("path")
        payload_hash = payload.get("sha256")
        payload_size = payload.get("bytes")
        if (
            not isinstance(path_value, str)
            or not isinstance(payload_hash, str)
            or _HASH_RE.fullmatch(payload_hash) is None
            or isinstance(payload_size, bool)
            or not isinstance(payload_size, int)
            or payload_size <= 0
        ):
            raise ValueError("F0 target payload descriptor values are invalid")
        data_path = Path(path_value)
        expected_path = self.root / f"{identifier}-{payload_hash[:16]}.pt"
        if data_path != expected_path:
            raise ValueError("F0 target payload path is not content-addressed")
        payload_bytes, actual_hash, actual_size = _read_f0_ordinary_file(
            data_path, label="F0 target payload"
        )
        if actual_hash != payload_hash or actual_size != payload_size:
            raise RuntimeError("F0 target payload bytes differ from the manifest")
        if "weights_only" not in inspect.signature(torch.load).parameters:
            raise RuntimeError(
                "safe F0 target loading requires torch.load(weights_only=True)"
            )
        envelope = torch.load(
            io.BytesIO(payload_bytes), map_location="cpu", weights_only=True
        )
        if not isinstance(envelope, Mapping) or set(envelope) != {
            "schema",
            "run_contract_sha256",
            "stable_id",
            "record_sha256",
            "target",
        }:
            raise ValueError("F0 target payload envelope has an unsupported shape")
        for field, value in (
            ("schema", F0_TARGET_ENVELOPE_SCHEMA),
            ("run_contract_sha256", self.contract_hash),
            ("stable_id", identifier),
            ("record_sha256", record_hash),
        ):
            if envelope.get(field) != value:
                raise ValueError(f"F0 target payload field {field} is mismatched")
        row = f0_target_from_payload(envelope.get("target"))
        if row.stable_id != identifier or row.record_sha256 != record_hash:
            raise ValueError("restored F0 target differs from its manifest")
        self._preflight()
        return row

    def load_or_none(self, stable_id: str) -> Any | None:
        """Return ``None`` only when no committed target manifest exists."""
        self._preflight()
        identifier = self._stable_id(stable_id)
        try:
            _prepare_f0_directory(self.root, create=False)
        except ValueError as exc:
            if str(exc) == "F0 target directory is unavailable":
                return None
            raise
        manifest_path = self.root / f"{identifier}.json"
        if not os.path.lexists(manifest_path):
            return None
        return self.load(identifier)

    def has(self, stable_id: str) -> bool:
        """Authenticate and decode a completed target before reporting it present."""
        return self.load_or_none(stable_id) is not None


class AtomicF0AnchorStore:
    """Persist complete strict-C0 traces as immutable prompt-seed blocks."""

    def __init__(self, authorization_binding: Any) -> None:
        binding = require_authorization_binding(authorization_binding)
        binding.validate_current()
        if binding.contract.get("method") != "f0":
            raise ValueError("F0 anchor storage requires an F0 run contract")
        run_dir_value = binding.contract.get("paths", {}).get("run_dir")
        if not isinstance(run_dir_value, str):
            raise ValueError("F0 run contract has no run_dir")
        run_dir = Path(run_dir_value)
        if not run_dir.is_absolute() or Path(os.path.abspath(run_dir)) != run_dir:
            raise ValueError("F0 run_dir must be normalized and absolute")
        self.authorization_binding = binding
        self.contract_hash = binding.contract_hash
        self.root = run_dir / "f0_anchor_blocks"
        self._binding_identity = id(binding)

    def _preflight(self) -> None:
        binding = require_authorization_binding(self.authorization_binding)
        if id(binding) != self._binding_identity:
            raise RuntimeError("F0-anchor-store authorization binding was replaced")
        if binding.contract_hash != self.contract_hash:
            raise RuntimeError("F0-anchor-store run contract changed")
        expected = Path(binding.contract["paths"]["run_dir"]) / "f0_anchor_blocks"
        if expected != self.root or binding.contract.get("method") != "f0":
            raise RuntimeError("F0-anchor-store contract changed after construction")
        binding.validate_current()

    @staticmethod
    def _coerce_score(value: Any) -> F0AnchorScoreRecovery:
        if type(value) is F0AnchorScoreRecovery:
            return _f0_anchor_score_from_payload(
                _f0_anchor_score_to_payload(value)
            )
        return F0AnchorScoreRecovery.from_scored_image(value)

    @staticmethod
    def _block_parts(block: Any) -> tuple[str, dict[str, Any]]:
        payload = _f0_prompt_block_to_payload(block)
        return block.stable_id, payload

    def save(
        self,
        block: Any,
        trace: Any = None,
        score_recovery: Any = None,
        *,
        score: Any = None,
    ) -> dict[str, Any]:
        """Atomically commit one trace and only the score's ledger recovery keys.

        ``block`` may also be an ``F0AnchorBlock``; in that form ``trace`` and
        score arguments must be omitted.  Otherwise callers provide a
        ``PromptSeedBlock``, its trace, and either ``score_recovery`` or the
        original ``F0ScoredImage`` through ``score``.
        """
        from .methods.f0 import F0AnchorBlock

        self._preflight()
        if type(block) is F0AnchorBlock:
            if trace is not None or score_recovery is not None or score is not None:
                raise TypeError("an F0AnchorBlock cannot be combined with separate fields")
            anchor = block
            block = anchor.block
            trace = anchor.trace
            score = anchor.score
        if (score_recovery is None) == (score is None):
            raise TypeError("provide exactly one of score_recovery or score")
        recovery = self._coerce_score(
            score if score is not None else score_recovery
        )
        stable_id, block_payload = self._block_parts(block)
        trace_payload = _f0_trace_to_payload(
            trace,
            block=block,
            contract=self.authorization_binding.contract,
            contract_hash=self.contract_hash,
        )
        recovery_payload = _f0_anchor_score_to_payload(recovery)

        self._preflight()
        _prepare_f0_directory(self.root, create=True, label="F0 anchor")
        manifest_path = self.root / f"{stable_id}.json"
        if os.path.lexists(manifest_path):
            raise FileExistsError(
                f"F0 anchor stable_id already exists: {stable_id}"
            )

        temporary_payload = self.root / f".{stable_id}.{uuid.uuid4().hex}.pt.tmp"
        temporary_manifest = self.root / f".{stable_id}.{uuid.uuid4().hex}.json.tmp"
        data_path: Path | None = None
        payload_published = False
        manifest_published = False
        completed = False
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(temporary_payload, flags, 0o600)
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as handle:
                    torch.save(
                        {
                            "schema": F0_ANCHOR_ENVELOPE_SCHEMA,
                            "run_contract_sha256": self.contract_hash,
                            "stable_id": stable_id,
                            "block": block_payload,
                            "trace": trace_payload,
                            "score_recovery": recovery_payload,
                        },
                        handle,
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                os.close(descriptor)
            _payload_bytes, payload_hash, payload_size = _read_f0_ordinary_file(
                temporary_payload, label="temporary F0 anchor payload"
            )
            data_path = self.root / f"{stable_id}-{payload_hash[:16]}.pt"
            self._preflight()
            _publish_f0_exclusive(
                temporary_payload, data_path, label="F0 anchor payload"
            )
            payload_published = True

            manifest_core = {
                "schema": F0_ANCHOR_MANIFEST_SCHEMA,
                "status": "complete",
                "run_contract_sha256": self.contract_hash,
                "stable_id": stable_id,
                "block": block_payload,
                "payload": {
                    "path": str(data_path),
                    "sha256": payload_hash,
                    "bytes": payload_size,
                },
            }
            manifest = {
                **manifest_core,
                "manifest_sha256": _f0_manifest_hash(manifest_core),
            }
            manifest_descriptor = os.open(temporary_manifest, flags, 0o600)
            try:
                with os.fdopen(manifest_descriptor, "wb", closefd=False) as handle:
                    handle.write(_canonical(manifest) + b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                os.close(manifest_descriptor)
            self._preflight()
            _publish_f0_exclusive(
                temporary_manifest, manifest_path, label="F0 anchor manifest"
            )
            manifest_published = True
            self._preflight()
            completed = True
            return manifest
        finally:
            temporary_payload.unlink(missing_ok=True)
            temporary_manifest.unlink(missing_ok=True)
            if not completed:
                if manifest_published:
                    manifest_path.unlink(missing_ok=True)
                if payload_published and data_path is not None:
                    data_path.unlink(missing_ok=True)
                if manifest_published or payload_published:
                    _fsync_directory(self.root)

    def _root_is_available(self) -> bool:
        try:
            _prepare_f0_directory(self.root, create=False, label="F0 anchor")
        except ValueError as exc:
            if str(exc) == "F0 anchor directory is unavailable":
                return False
            raise
        return True

    def _payload_bytes(
        self, block: Any
    ) -> tuple[dict[str, Any], bytes]:
        self._preflight()
        stable_id, block_payload = self._block_parts(block)
        _prepare_f0_directory(self.root, create=False, label="F0 anchor")
        manifest_path = self.root / f"{stable_id}.json"
        manifest_bytes, _manifest_file_hash, _manifest_size = (
            _read_f0_ordinary_file(manifest_path, label="F0 anchor manifest")
        )
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("F0 anchor manifest is unreadable") from exc
        required = {
            "schema",
            "status",
            "run_contract_sha256",
            "stable_id",
            "block",
            "payload",
            "manifest_sha256",
        }
        if not isinstance(manifest, dict) or set(manifest) != required:
            raise ValueError("F0 anchor manifest has an unsupported shape")
        if manifest_bytes != _canonical(manifest) + b"\n":
            raise RuntimeError("F0 anchor manifest bytes are not canonical")
        manifest_hash = manifest.get("manifest_sha256")
        manifest_core = {
            key: value for key, value in manifest.items() if key != "manifest_sha256"
        }
        if (
            not isinstance(manifest_hash, str)
            or _HASH_RE.fullmatch(manifest_hash) is None
            or _f0_manifest_hash(manifest_core) != manifest_hash
        ):
            raise RuntimeError("F0 anchor manifest SHA-256 is invalid")
        expected = {
            "schema": F0_ANCHOR_MANIFEST_SCHEMA,
            "status": "complete",
            "run_contract_sha256": self.contract_hash,
            "stable_id": stable_id,
            "block": block_payload,
        }
        for field, expected_value in expected.items():
            if manifest.get(field) != expected_value:
                raise ValueError(f"F0 anchor manifest field {field} is mismatched")
        payload = manifest.get("payload")
        if not isinstance(payload, Mapping) or set(payload) != {
            "path",
            "sha256",
            "bytes",
        }:
            raise ValueError("F0 anchor payload descriptor is invalid")
        path_value = payload.get("path")
        payload_hash = payload.get("sha256")
        payload_size = payload.get("bytes")
        if (
            not isinstance(path_value, str)
            or not isinstance(payload_hash, str)
            or _HASH_RE.fullmatch(payload_hash) is None
            or type(payload_size) is not int
            or payload_size <= 0
        ):
            raise ValueError("F0 anchor payload descriptor values are invalid")
        data_path = Path(path_value)
        expected_path = self.root / f"{stable_id}-{payload_hash[:16]}.pt"
        if data_path != expected_path:
            raise ValueError("F0 anchor payload path is not content-addressed")
        payload_bytes, actual_hash, actual_size = _read_f0_ordinary_file(
            data_path, label="F0 anchor payload"
        )
        if actual_hash != payload_hash or actual_size != payload_size:
            raise RuntimeError("F0 anchor payload bytes differ from the manifest")
        return manifest, payload_bytes

    def load(self, block: Any) -> F0AnchorBlockRecovery:
        """Authenticate and safely reconstruct one complete anchor block."""
        stable_id, block_payload = self._block_parts(block)
        _manifest, payload_bytes = self._payload_bytes(block)
        if "weights_only" not in inspect.signature(torch.load).parameters:
            raise RuntimeError(
                "safe F0 anchor loading requires torch.load(weights_only=True)"
            )
        envelope = torch.load(
            io.BytesIO(payload_bytes), map_location="cpu", weights_only=True
        )
        if not isinstance(envelope, Mapping) or set(envelope) != {
            "schema",
            "run_contract_sha256",
            "stable_id",
            "block",
            "trace",
            "score_recovery",
        }:
            raise ValueError("F0 anchor payload envelope has an unsupported shape")
        for field, expected_value in (
            ("schema", F0_ANCHOR_ENVELOPE_SCHEMA),
            ("run_contract_sha256", self.contract_hash),
            ("stable_id", stable_id),
            ("block", block_payload),
        ):
            if envelope.get(field) != expected_value:
                raise ValueError(f"F0 anchor payload field {field} is mismatched")
        restored_block = _f0_prompt_block_from_payload(envelope.get("block"))
        if restored_block != block or restored_block.stable_id != stable_id:
            raise ValueError("restored F0 anchor block differs from its manifest")
        trace = _f0_trace_from_payload(
            envelope.get("trace"),
            block=restored_block,
            contract=self.authorization_binding.contract,
            contract_hash=self.contract_hash,
        )
        recovery = _f0_anchor_score_from_payload(envelope.get("score_recovery"))
        self._preflight()
        return F0AnchorBlockRecovery(
            block=restored_block,
            trace=trace,
            score_recovery=recovery,
        )

    def load_or_none(self, block: Any) -> F0AnchorBlockRecovery | None:
        """Return ``None`` only when no committed manifest exists for the block."""
        self._preflight()
        stable_id, _block_payload = self._block_parts(block)
        if not self._root_is_available():
            return None
        manifest_path = self.root / f"{stable_id}.json"
        if not os.path.lexists(manifest_path):
            return None
        return self.load(block)

    def has(self, block: Any) -> bool:
        """Probe for a fully authenticated and decodable completed block."""
        return self.load_or_none(block) is not None


F0AnchorBlockStore = AtomicF0AnchorStore


__all__ = [
    "F0_ANCHOR_ENVELOPE_SCHEMA",
    "F0_ANCHOR_MANIFEST_SCHEMA",
    "F0_TARGET_ENVELOPE_SCHEMA",
    "F0_TARGET_MANIFEST_SCHEMA",
    "ROLLOUT_ENVELOPE_SCHEMA",
    "ROLLOUT_MANIFEST_SCHEMA",
    "AtomicF0AnchorStore",
    "AtomicF0TargetStore",
    "AtomicRolloutStore",
    "CheckpointProvenance",
    "F0AnchorBlockRecovery",
    "F0AnchorBlockStore",
    "F0AnchorScoreRecovery",
]
