"""Contract-bound execution for every budgeted training operation.

Formal renderer runs must reserve a query before touching an expensive model or
changing trainable state.  This module is the single boundary used by the SDXL
adapter, reward adapter, label builders, and optimizer.  A failed callback still
receives a durable receipt; an interrupted callback keeps its reservation charged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re
import time
from typing import Any, Callable, Mapping, Optional, TypeVar

from torch import Tensor

from .authorization import AuthorizationBinding, require_authorization_binding
from .ledger import QueryLedger
from .renderer import tensor_sha256


OPERATION_KINDS = frozenset(
    {
        "unet_forward",
        "scheduler_step",
        "vae_decode",
        "reward_forward",
        "reward_backward",
        "label",
        "optimizer_step",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_T = TypeVar("_T")


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
        raise ValueError("operation provenance must contain finite JSON values") from exc


def _validate_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hash")
    return value


def operation_output_sha256(value: Any) -> str:
    """Hash the supported tensor/JSON outputs without pickle serialization."""
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, Tensor):
            digest.update(b"tensor\0")
            digest.update(tensor_sha256(item).encode("ascii"))
            return
        if item is None or isinstance(item, (bool, int, str)):
            digest.update(b"json\0")
            payload = _canonical(item)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("operation output contains a non-finite float")
            digest.update(b"json\0")
            payload = _canonical(item)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
            return
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("operation output mapping keys must be strings")
            digest.update(b"mapping\0")
            for key in sorted(item):
                encoded = key.encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
                update(item[key])
            return
        if isinstance(item, (tuple, list)):
            digest.update(b"tuple\0" if isinstance(item, tuple) else b"list\0")
            digest.update(len(item).to_bytes(8, "big"))
            for child in item:
                update(child)
            return
        raise TypeError(
            "operation output is not tensor/JSON data; provide an explicit output_hasher"
        )

    update(value)
    return digest.hexdigest()


def tensor_operation_output_sha256(value: str) -> str:
    """Rebuild a tensor operation hash from its canonical tensor hash."""
    tensor_hash = _validate_hash(value, "tensor_hash")
    digest = hashlib.sha256()
    digest.update(b"tensor\0")
    digest.update(tensor_hash.encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class OperationContext:
    """Per-call provenance shared by all physical and logical operations."""

    split: str
    prompt: str
    seed: int
    step: int
    prefix: int
    branch: str
    action: Any
    checkpoint_hash: str
    image_hash: Optional[str] = None
    cached_parent: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("split", "prompt", "branch"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("seed", "step", "prefix"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.action, (list, tuple, Mapping)):
            raise ValueError("action must be a JSON array or object")
        _canonical(self.action)
        _validate_hash(self.checkpoint_hash, "checkpoint_hash")
        if self.image_hash is not None:
            _validate_hash(self.image_hash, "image_hash")
        if self.cached_parent is not None:
            _validate_hash(self.cached_parent, "cached_parent")


@dataclass(frozen=True)
class OperationReceipt:
    """In-process capability for one durable successful operation receipt."""

    reservation_id: str
    kind: str
    output_hash: str
    image_hash: str
    model: str
    role: str
    preprocess_hash: str
    model_config_sha256: str
    model_asset_manifest_sha256: str
    context: OperationContext
    _executor_seal: object = field(repr=False, compare=False)


class LedgeredOperationExecutor:
    """Execute source-controlled callbacks behind one formal query ledger."""

    def __init__(
        self,
        ledger: QueryLedger,
        *,
        authorization_binding: AuthorizationBinding,
    ) -> None:
        if not isinstance(ledger, QueryLedger):
            raise TypeError("ledger must be a QueryLedger")
        binding = require_authorization_binding(authorization_binding)
        if ledger.authorization_binding is not binding:
            raise ValueError("operation ledger and executor must share one authorization binding")
        if not ledger.strict_provenance:
            raise ValueError("formal operation execution requires strict provenance")
        binding.validate_current()
        self.ledger = ledger
        self.authorization_binding = binding
        self._binding_identity = id(binding)
        self._contract = binding.contract
        self._receipt_seal = object()
        self._code_hash = hashlib.sha256(
            self._contract["code_commit"].encode("ascii")
        ).hexdigest()

    def _preflight(self) -> None:
        binding = require_authorization_binding(self.authorization_binding)
        if id(binding) != self._binding_identity:
            raise RuntimeError("operation authorization binding was replaced")
        if self.ledger.authorization_binding is not binding:
            raise RuntimeError("operation ledger authorization binding was replaced")
        binding.validate_current()

    def verify_receipt(self, receipt: Any) -> OperationReceipt:
        """Return an authentic successful receipt issued by this executor.

        Callers can validate parent/child provenance without reading ledger
        implementation fields or reconstructing the executor's private seal.
        """
        self._preflight()
        if (
            type(receipt) is not OperationReceipt
            or receipt._executor_seal is not self._receipt_seal
        ):
            raise TypeError("receipt was not issued by this operation executor")
        if receipt.kind not in OPERATION_KINDS:
            raise RuntimeError("operation receipt has an unsupported kind")
        for field_name in (
            "output_hash",
            "image_hash",
            "preprocess_hash",
            "model_config_sha256",
            "model_asset_manifest_sha256",
        ):
            _validate_hash(
                getattr(receipt, field_name), f"receipt.{field_name}"
            )
        if not isinstance(receipt.model, str) or not receipt.model:
            raise RuntimeError("operation receipt has no model identity")
        if not isinstance(receipt.role, str) or not receipt.role:
            raise RuntimeError("operation receipt has no role identity")
        if not isinstance(receipt.context, OperationContext):
            raise RuntimeError("operation receipt has no operation context")
        return receipt

    def _metadata(
        self,
        kind: str,
        amount: int,
        context: OperationContext,
        *,
        model: str,
        role: str,
        preprocess_hash: str,
        model_config_sha256: str,
        model_asset_manifest_sha256: str,
    ) -> dict[str, Any]:
        return {
            "code_hash": self._code_hash,
            "data_hash": self._contract["data_manifest_sha256"],
            "checkpoint_hash": context.checkpoint_hash,
            "method_allocation": {
                "kind": kind,
                "amount": amount,
                "method": self._contract["method"],
            },
            "split": context.split,
            "prompt": context.prompt,
            "seed": context.seed,
            "step": context.step,
            "prefix": context.prefix,
            "branch": context.branch,
            "action": context.action,
            "model": model,
            "role": role,
            "preprocess_hash": preprocess_hash,
            "model_config_sha256": model_config_sha256,
            "model_asset_manifest_sha256": model_asset_manifest_sha256,
            "run_contract_hash": self.authorization_binding.contract_hash,
            "renderer_frame_contract_hash": self._contract[
                "renderer_frame_contract_hash"
            ],
            "calibration_hash": self._contract["calibration_hash"],
            "data_manifest_sha256": self._contract["data_manifest_sha256"],
            "reward_config_sha256": self._contract["reward_config_sha256"],
        }

    def _receipt(
        self,
        *,
        output_hash: str,
        context: OperationContext,
        scalar_or_gradient: str,
        wall_seconds: float,
        failure: Optional[Mapping[str, str]],
        model: str,
        role: str,
        preprocess_hash: str,
        model_config_sha256: str,
        model_asset_manifest_sha256: str,
        parent_reservation_id: Optional[str] = None,
    ) -> dict[str, Any]:
        return {
            # The legacy ledger schema calls this field image_hash.  Before a
            # terminal image exists it binds the exact tensor/JSON output; the
            # explicit output_hash field removes that ambiguity for analysis.
            "image_hash": context.image_hash or output_hash,
            "output_hash": output_hash,
            "reward_preprocess_hash": preprocess_hash,
            "model": model,
            "role": role,
            "model_config_sha256": model_config_sha256,
            "model_asset_manifest_sha256": model_asset_manifest_sha256,
            "scalar_or_gradient": scalar_or_gradient,
            "wall_seconds": wall_seconds,
            "cached_parent": context.cached_parent,
            "parent_reservation_id": parent_reservation_id,
            "failure": None if failure is None else dict(failure),
            "run_contract_hash": self.authorization_binding.contract_hash,
            "renderer_frame_contract_hash": self._contract[
                "renderer_frame_contract_hash"
            ],
            "calibration_hash": self._contract["calibration_hash"],
            "data_manifest_sha256": self._contract["data_manifest_sha256"],
            "reward_config_sha256": self._contract["reward_config_sha256"],
        }

    def _validate_operation_identity(
        self,
        *,
        kind: str,
        model: str,
        role: str,
        preprocess_hash: str,
    ) -> tuple[str, str]:
        if model == "pyiqa:topiq_nr" or role == "independent_f0_witness":
            if (
                kind != "reward_forward"
                or model != "pyiqa:topiq_nr"
                or role != "independent_f0_witness"
                or self._contract.get("method") != "f0"
                or preprocess_hash
                != self._contract.get("witness_preprocess_sha256")
            ):
                raise ValueError("TOPIQ-NR identity differs from the authorized F0 witness")
            config_hash = self._contract.get("witness_config_sha256")
            asset_hash = self._contract.get("witness_asset_manifest_sha256")
        elif kind in {"reward_forward", "reward_backward"}:
            if (
                model != "ImageReward-v1.0"
                or role != "training_reward"
                or preprocess_hash != self._contract["reward_preprocess_sha256"]
            ):
                raise ValueError("reward identity differs from the authorized ImageReward")
            config_hash = self._contract.get("reward_config_sha256")
            asset_hash = self._contract.get("reward_asset_manifest_sha256")
        else:
            if (
                model != "repldm-runtime"
                or role != kind
                or preprocess_hash != self._contract["reward_preprocess_sha256"]
            ):
                raise ValueError("operation identity differs from the registered runtime")
            config_hash = self._contract.get("reward_config_sha256")
            asset_hash = self._contract.get("reward_asset_manifest_sha256")
        return (
            _validate_hash(config_hash, "model_config_sha256"),
            _validate_hash(asset_hash, "model_asset_manifest_sha256"),
        )

    def execute_with_receipt(
        self,
        kind: str,
        context: OperationContext,
        callback: Callable[[], _T],
        *,
        amount: int = 1,
        scalar_or_gradient: str = "tensor",
        output_hasher: Callable[[_T], str] = operation_output_sha256,
        parent_receipt: Optional[OperationReceipt] = None,
        model: Optional[str] = None,
        role: Optional[str] = None,
        preprocess_hash: Optional[str] = None,
    ) -> tuple[_T, OperationReceipt]:
        """Execute once and return the value plus its durable receipt token."""
        self._preflight()
        if kind not in OPERATION_KINDS:
            raise ValueError(f"unsupported operation kind {kind!r}")
        if kind not in self.ledger.budget:
            raise ValueError(f"operation kind {kind!r} is absent from the run budget")
        if not isinstance(context, OperationContext):
            raise TypeError("context must be an OperationContext")
        if not callable(callback) or not callable(output_hasher):
            raise TypeError("callback and output_hasher must be callable")
        if not isinstance(scalar_or_gradient, str) or not scalar_or_gradient:
            raise ValueError("scalar_or_gradient must be a non-empty string")
        if model is None:
            model = (
                "ImageReward-v1.0"
                if kind in {"reward_forward", "reward_backward"}
                else "repldm-runtime"
            )
        if role is None:
            role = (
                "training_reward"
                if kind in {"reward_forward", "reward_backward"}
                else kind
            )
        if not isinstance(model, str) or not model:
            raise ValueError("operation model must be a non-empty string")
        if not isinstance(role, str) or not role:
            raise ValueError("operation role must be a non-empty string")
        if preprocess_hash is None:
            preprocess_hash = self._contract["reward_preprocess_sha256"]
        _validate_hash(preprocess_hash, "preprocess_hash")
        model_config_sha256, model_asset_manifest_sha256 = (
            self._validate_operation_identity(
                kind=kind,
                model=model,
                role=role,
                preprocess_hash=preprocess_hash,
            )
        )
        parent_reservation_id = None
        if parent_receipt is not None:
            parent_receipt = self.verify_receipt(parent_receipt)
            parent_reservation_id = parent_receipt.reservation_id
            if kind == "reward_forward" and (
                parent_receipt.kind != "vae_decode"
                or context.image_hash is None
                or context.cached_parent != parent_receipt.output_hash
                or tensor_operation_output_sha256(context.image_hash)
                != parent_receipt.output_hash
            ):
                raise ValueError(
                    "reward-forward parent does not bind the exact VAE image"
                )
        metadata = self._metadata(
            kind,
            amount,
            context,
            model=model,
            role=role,
            preprocess_hash=preprocess_hash,
            model_config_sha256=model_config_sha256,
            model_asset_manifest_sha256=model_asset_manifest_sha256,
        )
        reservation = self.ledger.reserve(kind, amount, metadata=metadata)
        started = time.perf_counter()
        try:
            value = callback()
            output_hash = _validate_hash(output_hasher(value), "output_hash")
        except BaseException as exc:
            wall_seconds = time.perf_counter() - started
            failure = {
                "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "message": str(exc),
            }
            output_hash = hashlib.sha256(_canonical(failure)).hexdigest()
            self.ledger.receipt(
                reservation,
                result=self._receipt(
                    output_hash=output_hash,
                    context=context,
                    scalar_or_gradient=scalar_or_gradient,
                    wall_seconds=wall_seconds,
                    failure=failure,
                    model=model,
                    role=role,
                    preprocess_hash=preprocess_hash,
                    model_config_sha256=model_config_sha256,
                    model_asset_manifest_sha256=model_asset_manifest_sha256,
                    parent_reservation_id=parent_reservation_id,
                ),
                success=False,
            )
            raise
        wall_seconds = time.perf_counter() - started
        self.ledger.receipt(
            reservation,
            result=self._receipt(
                output_hash=output_hash,
                context=context,
                scalar_or_gradient=scalar_or_gradient,
                wall_seconds=wall_seconds,
                failure=None,
                model=model,
                role=role,
                preprocess_hash=preprocess_hash,
                model_config_sha256=model_config_sha256,
                model_asset_manifest_sha256=model_asset_manifest_sha256,
                parent_reservation_id=parent_reservation_id,
            ),
            success=True,
        )
        return value, OperationReceipt(
            reservation_id=reservation.reservation_id,
            kind=kind,
            output_hash=output_hash,
            image_hash=context.image_hash or output_hash,
            model=model,
            role=role,
            preprocess_hash=preprocess_hash,
            model_config_sha256=model_config_sha256,
            model_asset_manifest_sha256=model_asset_manifest_sha256,
            context=context,
            _executor_seal=self._receipt_seal,
        )

    def validate_receipt(
        self,
        receipt: OperationReceipt,
        *,
        kind: str,
        output_hash: str,
        context: OperationContext,
        scalar_or_gradient: str,
        amount: int = 1,
        model: Optional[str] = None,
        role: Optional[str] = None,
        preprocess_hash: Optional[str] = None,
        parent_reservation_id: Optional[str] = None,
    ) -> None:
        """Verify one executor capability against its durable ledger records."""
        self._preflight()
        receipt = self.verify_receipt(receipt)
        if kind not in OPERATION_KINDS or kind not in self.ledger.budget:
            raise ValueError("receipt kind is absent from the formal operation budget")
        if not isinstance(context, OperationContext):
            raise TypeError("receipt context must be an OperationContext")
        if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
            raise ValueError("receipt amount must be a positive integer")
        if not isinstance(scalar_or_gradient, str) or not scalar_or_gradient:
            raise ValueError("scalar_or_gradient must be a non-empty string")
        output_hash = _validate_hash(output_hash, "output_hash")
        if model is None:
            model = (
                "ImageReward-v1.0"
                if kind in {"reward_forward", "reward_backward"}
                else "repldm-runtime"
            )
        if role is None:
            role = (
                "training_reward"
                if kind in {"reward_forward", "reward_backward"}
                else kind
            )
        if not isinstance(model, str) or not model:
            raise ValueError("receipt model must be a non-empty string")
        if not isinstance(role, str) or not role:
            raise ValueError("receipt role must be a non-empty string")
        if preprocess_hash is None:
            preprocess_hash = self._contract["reward_preprocess_sha256"]
        preprocess_hash = _validate_hash(preprocess_hash, "preprocess_hash")
        model_config_sha256, model_asset_manifest_sha256 = (
            self._validate_operation_identity(
                kind=kind,
                model=model,
                role=role,
                preprocess_hash=preprocess_hash,
            )
        )
        parent_was_supplied = parent_reservation_id is not None
        if parent_reservation_id is not None and (
            not isinstance(parent_reservation_id, str) or not parent_reservation_id
        ):
            raise ValueError("parent_reservation_id must be a non-empty string or None")

        matching = [
            pair
            for pair in self.ledger.successful_receipt_pairs(kind)
            if pair[0].get("reservation_id") == receipt.reservation_id
        ]
        if len(matching) != 1:
            raise RuntimeError("operation capability has no unique durable receipt")
        reservation, durable = matching[0]
        result = durable.get("result")
        if not isinstance(result, Mapping):
            raise RuntimeError("durable operation receipt has no result mapping")

        # A legacy reward adapter can reconstruct the operation context with
        # its old list-shaped action after the formal F0 path has recorded the
        # canonical metric mapping.  Authenticate the durable pair first, and
        # allow only that one action-field discrepancy when the caller omitted
        # the optional parent.  Every other field remains an exact capability
        # check, and an explicitly supplied parent never takes this branch.
        durable_parent_id = result.get("parent_reservation_id")
        action_compatibility = False
        provenance_context = context
        if receipt.context != context:
            shared_fields = (
                "split",
                "prompt",
                "seed",
                "step",
                "prefix",
                "branch",
                "checkpoint_hash",
                "image_hash",
                "cached_parent",
            )
            action_compatibility = (
                not parent_was_supplied
                and kind == "reward_forward"
                and isinstance(durable_parent_id, str)
                and bool(durable_parent_id)
                and all(
                    getattr(receipt.context, field) == getattr(context, field)
                    for field in shared_fields
                )
            )
            if action_compatibility:
                provenance_context = receipt.context

        if (
            receipt.kind != kind
            or receipt.output_hash != output_hash
            or receipt.image_hash != (context.image_hash or output_hash)
            or receipt.model != model
            or receipt.role != role
            or receipt.preprocess_hash != preprocess_hash
            or receipt.model_config_sha256 != model_config_sha256
            or receipt.model_asset_manifest_sha256 != model_asset_manifest_sha256
            or (receipt.context != context and not action_compatibility)
        ):
            raise ValueError("operation receipt capability differs from the expected call")

        expected_metadata = self._metadata(
            kind,
            amount,
            provenance_context,
            model=model,
            role=role,
            preprocess_hash=preprocess_hash,
            model_config_sha256=model_config_sha256,
            model_asset_manifest_sha256=model_asset_manifest_sha256,
        )
        if (
            reservation.get("kind") != kind
            or reservation.get("amount") != amount
            or reservation.get("metadata") != expected_metadata
            or durable.get("kind") != kind
            or durable.get("amount") != amount
            or durable.get("metadata") != expected_metadata
            or durable.get("success") is not True
        ):
            raise ValueError("durable operation receipt differs from its capability")
        # Older adapter overrides called ``validate_receipt`` without naming
        # the parent explicitly.  Once the capability has been authenticated,
        # the durable result is the authoritative source for that relationship;
        # infer it only when the caller omitted the optional argument.  An
        # explicitly supplied parent still has to match byte-for-byte.
        durable_parent_id = result.get("parent_reservation_id")
        if parent_reservation_id is None and durable_parent_id is not None:
            parent_reservation_id = durable_parent_id
        if parent_reservation_id is not None and (
            not isinstance(parent_reservation_id, str) or not parent_reservation_id
        ):
            raise ValueError("parent_reservation_id must be a non-empty string or None")
        wall_seconds = result.get("wall_seconds")
        if (
            isinstance(wall_seconds, bool)
            or not isinstance(wall_seconds, (int, float))
            or not math.isfinite(float(wall_seconds))
            or float(wall_seconds) < 0
        ):
            raise RuntimeError("durable operation receipt has invalid timing")
        expected_result = self._receipt(
            output_hash=output_hash,
            context=provenance_context,
            scalar_or_gradient=scalar_or_gradient,
            wall_seconds=float(wall_seconds),
            failure=None,
            model=model,
            role=role,
            preprocess_hash=preprocess_hash,
            model_config_sha256=model_config_sha256,
            model_asset_manifest_sha256=model_asset_manifest_sha256,
            parent_reservation_id=parent_reservation_id,
        )
        durable_result = dict(result)
        durable_result.pop("result_hash", None)
        if durable_result != expected_result:
            raise ValueError("durable operation result differs from its capability")

    def restore_receipt(self, reservation_id: str) -> OperationReceipt:
        """Re-seal one success recovered from the verified durable ledger."""
        self._preflight()
        if not isinstance(reservation_id, str) or not reservation_id:
            raise ValueError("reservation_id must be a non-empty string")
        pairs = self.ledger.successful_receipt_pairs()
        matching = [
            pair
            for pair in pairs
            if pair[0].get("reservation_id") == reservation_id
        ]
        if len(matching) != 1:
            raise RuntimeError(
                "reservation has no unique successful durable operation receipt"
            )
        reservation, durable = matching[0]
        metadata = reservation.get("metadata")
        result = durable.get("result")
        if not isinstance(metadata, Mapping) or not isinstance(result, Mapping):
            raise RuntimeError("durable operation receipt is missing provenance")
        kind = reservation.get("kind")
        amount = reservation.get("amount")
        if (
            kind not in OPERATION_KINDS
            or durable.get("kind") != kind
            or durable.get("amount") != amount
            or durable.get("metadata") != metadata
        ):
            raise ValueError("durable operation kind or metadata is inconsistent")

        output_hash = _validate_hash(result.get("output_hash"), "output_hash")
        image_hash = _validate_hash(result.get("image_hash"), "image_hash")
        cached_parent = result.get("cached_parent")
        if cached_parent is not None:
            cached_parent = _validate_hash(cached_parent, "cached_parent")
        parent_id = result.get("parent_reservation_id")
        if parent_id is not None and (
            not isinstance(parent_id, str) or not parent_id
        ):
            raise ValueError("durable parent reservation id is invalid")

        context = OperationContext(
            split=metadata.get("split"),
            prompt=metadata.get("prompt"),
            seed=metadata.get("seed"),
            step=metadata.get("step"),
            prefix=metadata.get("prefix"),
            branch=metadata.get("branch"),
            action=metadata.get("action"),
            checkpoint_hash=metadata.get("checkpoint_hash"),
            image_hash=(
                image_hash
                if kind in {"reward_forward", "reward_backward"}
                else None
            ),
            cached_parent=cached_parent,
        )

        if parent_id is not None:
            parents = [
                pair
                for pair in pairs
                if pair[0].get("reservation_id") == parent_id
            ]
            if len(parents) != 1:
                raise RuntimeError(
                    "durable operation parent has no unique successful receipt"
                )
            parent_reservation, parent_durable = parents[0]
            parent_metadata = parent_reservation.get("metadata")
            parent_result = parent_durable.get("result")
            if not isinstance(parent_metadata, Mapping) or not isinstance(
                parent_result, Mapping
            ):
                raise RuntimeError("durable operation parent is missing provenance")
            expected_parent_kind = {
                "reward_forward": "vae_decode",
                "reward_backward": "reward_forward",
            }.get(kind)
            if expected_parent_kind is None or parent_reservation.get(
                "kind"
            ) != expected_parent_kind:
                raise ValueError("durable operation has an invalid parent kind")
            shared_context = (
                "split",
                "prompt",
                "seed",
                "step",
                "prefix",
                "branch",
                "action",
                "checkpoint_hash",
            )
            if any(
                metadata.get(field) != parent_metadata.get(field)
                for field in shared_context
            ):
                raise ValueError("durable operation context differs from its parent")
            parent_output = _validate_hash(
                parent_result.get("output_hash"), "parent.output_hash"
            )
            if cached_parent != parent_output:
                raise ValueError("durable operation cached parent hash is inconsistent")
            if kind == "reward_forward" and (
                tensor_operation_output_sha256(image_hash) != parent_output
            ):
                raise ValueError("durable reward image differs from its VAE parent")
            if kind == "reward_backward" and image_hash != parent_result.get(
                "image_hash"
            ):
                raise ValueError("durable reward backward uses a different image")
        elif kind == "reward_backward":
            raise ValueError("durable reward backward is missing its forward parent")

        model = result.get("model")
        role = result.get("role")
        preprocess_hash = result.get("reward_preprocess_hash")
        receipt = OperationReceipt(
            reservation_id=reservation_id,
            kind=kind,
            output_hash=output_hash,
            image_hash=image_hash,
            model=model,
            role=role,
            preprocess_hash=preprocess_hash,
            model_config_sha256=result.get("model_config_sha256"),
            model_asset_manifest_sha256=result.get(
                "model_asset_manifest_sha256"
            ),
            context=context,
            _executor_seal=self._receipt_seal,
        )
        self.validate_receipt(
            receipt,
            kind=kind,
            output_hash=output_hash,
            context=context,
            scalar_or_gradient=result.get("scalar_or_gradient"),
            amount=amount,
            model=model,
            role=role,
            preprocess_hash=preprocess_hash,
            parent_reservation_id=parent_id,
        )
        return receipt

    def execute(
        self,
        kind: str,
        context: OperationContext,
        callback: Callable[[], _T],
        *,
        amount: int = 1,
        scalar_or_gradient: str = "tensor",
        output_hasher: Callable[[_T], str] = operation_output_sha256,
        model: Optional[str] = None,
        role: Optional[str] = None,
        preprocess_hash: Optional[str] = None,
    ) -> _T:
        """Reserve, execute once, and return only the operation value."""
        value, _receipt = self.execute_with_receipt(
            kind,
            context,
            callback,
            amount=amount,
            scalar_or_gradient=scalar_or_gradient,
            output_hasher=output_hasher,
            model=model,
            role=role,
            preprocess_hash=preprocess_hash,
        )
        return value


__all__ = [
    "OPERATION_KINDS",
    "LedgeredOperationExecutor",
    "OperationContext",
    "OperationReceipt",
    "operation_output_sha256",
    "tensor_operation_output_sha256",
]
