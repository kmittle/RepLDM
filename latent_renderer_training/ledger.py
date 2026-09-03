"""Append-only query accounting used to enforce equal experimental budgets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import uuid
from contextlib import contextmanager
from typing import Any, Mapping, Optional


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _json_value(value: Any) -> Any:
    """Normalize JSON-compatible values (including tuples) before persistence."""
    return json.loads(_canonical(value).decode("utf-8"))


@dataclass(frozen=True)
class QueryReservation:
    reservation_id: str
    kind: str
    amount: int
    metadata: dict[str, Any]


_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_RESERVATION_PROVENANCE_FIELDS = (
    "code_hash",
    "data_hash",
    "checkpoint_hash",
    "method_allocation",
    "split",
    "prompt",
    "seed",
    "step",
    "prefix",
    "branch",
    "action",
)
_RECEIPT_PROVENANCE_FIELDS = (
    "image_hash",
    "reward_preprocess_hash",
    "scalar_or_gradient",
    "cached_parent",
)
_HASH_FIELDS = {
    "code_hash",
    "data_hash",
    "checkpoint_hash",
    "image_hash",
    "reward_preprocess_hash",
}

_SEAL_SCHEMA = "repldm.query_ledger_seal.v2"
_TRANSACTION_SCHEMA = "repldm.query_ledger_transaction.v1"


class QueryLedger:
    """Reserve before compute and append a receipt after compute.

    A reservation without a receipt remains charged after a crash.  This is
    intentionally conservative: the ledger never silently gives a failed or
    interrupted method a second budget.  One JSON object is written per line,
    with fsync before the call returns.
    """

    def __init__(
        self,
        path: str | Path,
        budget: Mapping[str, int],
        *,
        run_contract: Any,
        strict_provenance: bool = True,
        authorization_binding: Any = None,
        authorization: Any = None,
        require_authorization: bool = False,
    ) -> None:
        self.path = Path(path)
        self.seal_path = self.path.with_name(self.path.name + ".seal")
        self.transaction_path = self.path.with_name(self.path.name + ".txn")
        self.budget = {}
        for key, value in budget.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("ledger budgets must be non-negative integers")
            self.budget[str(key)] = value
        if not self.budget:
            raise ValueError("ledger budget must contain at least one query kind")
        if not isinstance(strict_provenance, bool):
            raise TypeError("strict_provenance must be boolean")
        self.strict_provenance = strict_provenance

        if authorization is not None:
            raise TypeError(
                "QueryLedger requires authorization_binding; authorization is not accepted"
            )
        binding = authorization_binding
        if binding is not None:
            from .authorization import require_authorization_binding
            from .run_contract import TrainingRunContract

            binding = require_authorization_binding(binding)
            if not strict_provenance:
                raise ValueError(
                    "authorization-bound ledgers require strict provenance"
                )
            if isinstance(run_contract, TrainingRunContract):
                supplied_hash = run_contract.sha256
            elif isinstance(run_contract, Mapping):
                supplied_hash = TrainingRunContract.from_mapping(run_contract).sha256
            else:
                supplied_hash = str(run_contract)
            if supplied_hash != binding.contract_hash:
                raise ValueError("ledger run contract differs from authorization binding")
            binding.validate_current()
            payload = binding.contract
            declared_budget = payload.get("query_budget")
            if dict(declared_budget) != dict(self.budget):
                raise ValueError("ledger query budget differs from the run contract")
            self.authorization_binding = binding
            self._contract_payload = payload
            self.run_contract = binding.contract_hash
        else:
            from .run_contract import RUN_CONTRACT_SCHEMA, TrainingRunContract

            if isinstance(run_contract, TrainingRunContract) or (
                isinstance(run_contract, Mapping)
                and run_contract.get("schema") == RUN_CONTRACT_SCHEMA
            ):
                raise RuntimeError(
                    "formal query ledgers require a validated authorization binding"
                )
            self.authorization_binding = None
            self._contract_payload = None
            self.run_contract = str(run_contract)
            if not self.run_contract:
                raise ValueError("run_contract must be a non-empty string")
        self.require_authorization = bool(require_authorization or binding is not None)
        if self.require_authorization and self.authorization_binding is None:
            raise RuntimeError(
                "formal query-ledger operations require a validated authorization binding"
            )

    def _preflight(self) -> None:
        """Run the authorization gate before any callback or ledger I/O."""
        if self.authorization_binding is None:
            if self.require_authorization:
                raise RuntimeError(
                    "query-ledger operation requires a validated authorization binding"
                )
            return
        self.authorization_binding.validate_current()

    def _validate_bound_fields(self, value: Mapping[str, Any], *, result: bool = False) -> None:
        """Require provenance fields that identify the complete training run."""
        if self.authorization_binding is None:
            return
        payload = self._contract_payload or {}
        required = {
            "run_contract_hash": self.run_contract,
            "renderer_frame_contract_hash": payload["renderer_frame_contract_hash"],
            "calibration_hash": payload["calibration_hash"],
            "data_manifest_sha256": payload["data_manifest_sha256"],
            "reward_config_sha256": payload["reward_config_sha256"],
        }
        for field, expected in required.items():
            if value.get(field) != expected:
                kind = "receipt" if result else "reservation"
                raise ValueError(f"{kind} provenance field {field} differs from the run contract")

    @staticmethod
    def _validate_hash(value: Any, field: str) -> None:
        if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
            raise ValueError(f"{field} must be a lowercase SHA-256 hash")

    def _validate_provenance(
        self, metadata: Mapping[str, Any], *, kind: str, amount: int
    ) -> dict[str, Any]:
        if not isinstance(metadata, Mapping):
            raise TypeError("reservation metadata must be a mapping")
        value = _json_value(dict(metadata))
        if not self.strict_provenance:
            # Even relaxed provenance must remain canonical JSON so a record
            # cannot hide NaN/Inf values from the hash chain.
            self._validate_bound_fields(value)
            return value
        missing = [field for field in _RESERVATION_PROVENANCE_FIELDS if field not in value]
        if missing:
            raise ValueError(f"reservation metadata lacks provenance fields: {missing}")
        for field in _HASH_FIELDS.intersection(_RESERVATION_PROVENANCE_FIELDS):
            self._validate_hash(value[field], field)
        allocation = value["method_allocation"]
        if not isinstance(allocation, Mapping):
            raise ValueError("method_allocation must be a mapping")
        if allocation.get("kind") != kind or allocation.get("amount") != amount:
            raise ValueError("method_allocation must match the reservation kind and amount")
        for field in ("split", "prompt", "branch"):
            if not isinstance(value[field], str) or not value[field]:
                raise ValueError(f"{field} must be a non-empty string")
        for field in ("seed", "step", "prefix"):
            if isinstance(value[field], bool) or not isinstance(value[field], int) or value[field] < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if not isinstance(value["action"], (list, tuple, Mapping)):
            raise ValueError("action must be a JSON array or object")
        # Validate JSON serializability and reject NaN/Inf hidden in nested data.
        self._validate_bound_fields(value)
        return value

    def _validate_receipt_result(
        self, result: Mapping[str, Any]
    ) -> dict[str, Any]:
        if not isinstance(result, Mapping):
            raise TypeError("receipt result must be a mapping")
        value = _json_value(dict(result))
        if self.strict_provenance:
            missing = [field for field in _RECEIPT_PROVENANCE_FIELDS if field not in value]
            if missing:
                raise ValueError(f"receipt result lacks provenance fields: {missing}")
            self._validate_hash(value["image_hash"], "image_hash")
            self._validate_hash(value["reward_preprocess_hash"], "reward_preprocess_hash")
            if not isinstance(value["scalar_or_gradient"], str) or not value["scalar_or_gradient"]:
                raise ValueError("scalar_or_gradient must be a non-empty string")
            parent = value["cached_parent"]
            if parent is not None:
                self._validate_hash(parent, "cached_parent")
        self._validate_bound_fields(value, result=True)
        supplied_hash = value.pop("result_hash", None)
        computed_hash = hashlib.sha256(_canonical(value)).hexdigest()
        if supplied_hash is not None and supplied_hash != computed_hash:
            raise ValueError("result_hash does not match the receipt result")
        value["result_hash"] = computed_hash
        return value

    def _verify_seal(self, records: list[dict[str, Any]]) -> None:
        if not records:
            if self.seal_path.exists():
                raise RuntimeError("query ledger has a seal but no records")
            return
        if self.seal_path.is_symlink() or not self.seal_path.is_file():
            raise RuntimeError("query ledger seal is missing")
        try:
            seal = json.loads(self.seal_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("query ledger seal is invalid") from exc
        if not isinstance(seal, dict):
            raise RuntimeError("query ledger seal is not an object")
        seal_hash = seal.get("seal_hash")
        unsigned = dict(seal)
        unsigned.pop("seal_hash", None)
        if not isinstance(seal_hash, str) or not _HASH_RE.fullmatch(seal_hash):
            raise RuntimeError("query ledger seal hash is invalid")
        if hashlib.sha256(_canonical(unsigned)).hexdigest() != seal_hash:
            raise RuntimeError("query ledger seal hash verification failed")
        if unsigned.get("schema") != _SEAL_SCHEMA:
            raise RuntimeError("query ledger seal schema is incompatible with this ledger")
        expected = {
            "schema": _SEAL_SCHEMA,
            "run_contract": self.run_contract,
            "budget": dict(sorted(self.budget.items())),
            "strict_provenance": self.strict_provenance,
            "record_count": len(records),
            "tip_record_hash": records[-1]["record_hash"],
            "root_record_hash": records[0]["record_hash"],
        }
        if unsigned != expected:
            mismatches = [
                key for key in expected
                if unsigned.get(key) != expected[key]
            ]
            raise RuntimeError(
                "query ledger seal does not match the run or record chain: "
                + ", ".join(mismatches)
            )

    def _seal_for_records(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Build the immutable contract seal for a complete record chain."""
        if not records:
            raise ValueError("cannot seal an empty query ledger")
        unsigned = {
            "schema": _SEAL_SCHEMA,
            "run_contract": self.run_contract,
            "budget": dict(sorted(self.budget.items())),
            "strict_provenance": self.strict_provenance,
            "record_count": len(records),
            "tip_record_hash": records[-1]["record_hash"],
            "root_record_hash": records[0]["record_hash"],
        }
        seal = dict(unsigned)
        seal["seal_hash"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
        return seal

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _write_transaction(self, payload: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.transaction_path.with_name(
            self.transaction_path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.transaction_path)
        self._fsync_directory(self.transaction_path.parent)

    def _recover_pending_transaction(self) -> None:
        """Complete a durable append interrupted between record and seal."""
        if not self.transaction_path.exists():
            return
        if not self.transaction_path.is_file() or self.transaction_path.is_symlink():
            raise RuntimeError("query ledger transaction is not a regular file")
        try:
            transaction = json.loads(self.transaction_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("query ledger transaction is invalid") from exc
        if not isinstance(transaction, dict) or transaction.get("schema") != _TRANSACTION_SCHEMA:
            raise RuntimeError("query ledger transaction schema is invalid")
        if transaction.get("run_contract") != self.run_contract:
            raise RuntimeError("query ledger transaction has a different run contract")
        if transaction.get("budget") != dict(sorted(self.budget.items())):
            raise RuntimeError("query ledger transaction budget does not match the run")
        if transaction.get("strict_provenance") is not self.strict_provenance:
            raise RuntimeError("query ledger transaction strictness does not match the run")
        record = transaction.get("record")
        expected_seal = transaction.get("seal")
        base_count = transaction.get("base_count")
        if (
            not isinstance(record, dict)
            or isinstance(base_count, bool)
            or not isinstance(base_count, int)
            or base_count < 0
            or not isinstance(expected_seal, dict)
        ):
            raise RuntimeError("query ledger transaction fields are invalid")
        records = self._read_chain()
        if len(records) == base_count:
            expected_previous = records[-1]["record_hash"] if records else None
            if record.get("sequence") != base_count + 1 or record.get("previous_record_hash") != expected_previous:
                raise RuntimeError("query ledger transaction does not follow the record chain")
            self._append_record_bytes(record)
            records = self._read_chain()
        elif len(records) == base_count + 1 and records[-1] == record:
            pass
        else:
            raise RuntimeError("query ledger transaction conflicts with the record chain")
        calculated_seal = self._seal_for_records(records)
        if expected_seal != calculated_seal:
            raise RuntimeError("query ledger transaction seal is inconsistent")
        self._publish_seal(calculated_seal)
        self.transaction_path.unlink()
        self._fsync_directory(self.transaction_path.parent)

    def _read_chain(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        if self.path.is_symlink() or not self.path.is_file():
            raise RuntimeError("query ledger is not a regular file")
        records: list[dict[str, Any]] = []
        previous_hash: str | None = None
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid query ledger line {line_number}") from exc
                if not isinstance(record, dict):
                    raise RuntimeError(f"query ledger line {line_number} is not an object")
                if record.get("run_contract") != self.run_contract:
                    raise RuntimeError(
                        f"query ledger line {line_number} has a different run contract"
                    )
                record_hash = record.get("record_hash")
                unsigned = dict(record)
                unsigned.pop("record_hash", None)
                if not isinstance(record_hash, str) or not _HASH_RE.fullmatch(record_hash):
                    raise RuntimeError(
                        f"query ledger line {line_number} has an invalid record hash"
                    )
                if hashlib.sha256(_canonical(unsigned)).hexdigest() != record_hash:
                    raise RuntimeError(
                        f"query ledger line {line_number} failed hash verification"
                    )
                sequence = record.get("sequence")
                if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence != len(records) + 1:
                    raise RuntimeError(f"query ledger line {line_number} has an invalid sequence")
                if record.get("previous_record_hash") != previous_hash:
                    raise RuntimeError(f"query ledger line {line_number} breaks the hash chain")
                row_type = record.get("type")
                reservation_id = record.get("reservation_id")
                if not isinstance(reservation_id, str) or not reservation_id:
                    raise RuntimeError(
                        f"query ledger line {line_number} has an invalid reservation id"
                    )
                kind = record.get("kind")
                amount = record.get("amount")
                if (
                    kind not in self.budget
                    or isinstance(amount, bool)
                    or not isinstance(amount, int)
                    or amount <= 0
                ):
                    raise RuntimeError(
                        f"query ledger line {line_number} has an invalid query amount"
                    )
                if row_type == "reservation":
                    if not isinstance(record.get("metadata"), dict):
                        raise RuntimeError(
                            f"query ledger line {line_number} has invalid metadata"
                        )
                    if self.strict_provenance:
                        try:
                            self._validate_provenance(
                                record["metadata"],
                                kind=kind,
                                amount=amount,
                            )
                        except (TypeError, ValueError) as exc:
                            raise RuntimeError(
                                f"query ledger line {line_number} has invalid reservation provenance"
                            ) from exc
                elif row_type == "receipt":
                    if not isinstance(record.get("success"), bool) or not isinstance(
                        record.get("result"), dict
                    ):
                        raise RuntimeError(
                            f"query ledger line {line_number} has invalid receipt fields"
                        )
                    if "metadata" in record and not isinstance(record["metadata"], dict):
                        raise RuntimeError(
                            f"query ledger line {line_number} has invalid receipt metadata"
                        )
                    if self.strict_provenance:
                        try:
                            self._validate_receipt_result(record["result"])
                        except (TypeError, ValueError) as exc:
                            raise RuntimeError(
                                f"query ledger line {line_number} has invalid receipt provenance"
                            ) from exc
                else:
                    raise RuntimeError(
                        f"query ledger line {line_number} has unknown record type"
                    )
                records.append(record)
                previous_hash = record_hash
        reservations: dict[str, dict[str, Any]] = {}
        receipts: set[str] = set()
        for record in records:
            reservation_id = record["reservation_id"]
            if record["type"] == "reservation":
                if reservation_id in reservations:
                    raise RuntimeError(f"duplicate reservation id {reservation_id}")
                reservations[reservation_id] = record
                continue
            if reservation_id in receipts:
                raise RuntimeError(f"duplicate receipt for reservation {reservation_id}")
            source = reservations.get(reservation_id)
            if source is None:
                raise RuntimeError(f"receipt has no reservation {reservation_id}")
            if record["kind"] != source["kind"] or record["amount"] != source["amount"]:
                raise RuntimeError(f"receipt does not match reservation {reservation_id}")
            if "metadata" in record and record["metadata"] != source["metadata"]:
                raise RuntimeError(f"receipt metadata does not match reservation {reservation_id}")
            receipts.add(reservation_id)
        return records

    def _records(self) -> list[dict[str, Any]]:
        self._recover_pending_transaction()
        records = self._read_chain()
        self._verify_seal(records)
        return records

    def _append_record_bytes(self, record: Mapping[str, Any]) -> None:
        """Publish a complete ledger file atomically, avoiding torn JSON lines."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and (self.path.is_symlink() or not self.path.is_file()):
            raise RuntimeError("query ledger is not a regular file")
        temporary = self.path.with_name(
            self.path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        payload = (
            json.dumps(dict(record), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            + "\n"
        ).encode("utf-8")
        try:
            with temporary.open("wb") as handle:
                if self.path.exists():
                    with self.path.open("rb") as source:
                        shutil.copyfileobj(source, handle)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(self.path)
            self._fsync_directory(self.path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def _publish_seal(self, seal: Mapping[str, Any]) -> None:
        self.seal_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_seal = self.seal_path.with_name(
            self.seal_path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        with temporary_seal.open("w", encoding="utf-8") as handle:
            json.dump(dict(seal), handle, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_seal.replace(self.seal_path)
        self._fsync_directory(self.seal_path.parent)

    def _append(self, record: dict[str, Any]) -> None:
        record = dict(record)
        if "record_hash" in record:
            raise ValueError("record_hash is managed by QueryLedger")
        existing = self._records()
        record["sequence"] = len(existing) + 1
        record["previous_record_hash"] = (
            existing[-1]["record_hash"] if existing else None
        )
        record["run_contract"] = self.run_contract
        record["record_hash"] = hashlib.sha256(_canonical(record)).hexdigest()
        expected_records = [*existing, record]
        seal = self._seal_for_records(expected_records)
        transaction = {
            "schema": _TRANSACTION_SCHEMA,
            "run_contract": self.run_contract,
            "budget": dict(sorted(self.budget.items())),
            "strict_provenance": self.strict_provenance,
            "base_count": len(existing),
            "record": record,
            "seal": seal,
        }
        # The transaction intent is durable before the append.  A later read
        # can therefore distinguish an interrupted writer from tampering.
        self._write_transaction(transaction)
        self._append_record_bytes(record)
        self._publish_seal(seal)
        self.transaction_path.unlink()
        self._fsync_directory(self.transaction_path.parent)

    @contextmanager
    def _exclusive_lock(self):
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        with lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _used_from_records(records: list[dict[str, Any]], kind: str) -> int:
        total = 0
        for record in records:
            if record.get("type") == "reservation" and record.get("kind") == kind:
                total += int(record.get("amount", 0))
        return total

    def used(self, kind: str) -> int:
        self._preflight()
        if kind not in self.budget:
            raise KeyError(kind)
        with self._exclusive_lock():
            return self._used_from_records(self._records(), kind)

    def remaining(self, kind: str) -> int:
        self._preflight()
        if kind not in self.budget:
            raise KeyError(kind)
        with self._exclusive_lock():
            return self.budget[kind] - self._used_from_records(self._records(), kind)

    def reserve(self, kind: str, amount: int, *, metadata: Optional[Mapping[str, Any]] = None) -> QueryReservation:
        self._preflight()
        if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
            raise ValueError("reservation amount must be positive")
        if kind not in self.budget:
            raise KeyError(f"unknown query kind {kind!r}")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("reservation metadata must be a mapping")
        with self._exclusive_lock():
            records = self._records()
            remaining = self.budget[kind] - self._used_from_records(records, kind)
            if amount > remaining:
                raise RuntimeError(f"query budget exceeded for {kind}: requested {amount}, remaining {remaining}")
            reservation_metadata = self._validate_provenance(
                {} if metadata is None else metadata, kind=kind, amount=amount
            )
            reservation = QueryReservation(
                str(uuid.uuid4()), kind, amount, reservation_metadata
            )
            self._append({
                "type": "reservation", "reservation_id": reservation.reservation_id,
                "kind": kind, "amount": amount, "metadata": reservation.metadata,
                "wall_time": time.time(),
            })
        return reservation

    def receipt(self, reservation: QueryReservation, *, result: Mapping[str, Any], success: bool) -> None:
        self._preflight()
        if not isinstance(reservation, QueryReservation):
            raise TypeError("receipt requires a QueryReservation")
        if not isinstance(success, bool):
            raise TypeError("receipt success must be boolean")
        receipt_result = self._validate_receipt_result(result)
        with self._exclusive_lock():
            records = self._records()
            reservations = {
                row["reservation_id"]: row
                for row in records
                if row["type"] == "reservation"
            }
            receipts = {
                row["reservation_id"] for row in records if row["type"] == "receipt"
            }
            stored = reservations.get(reservation.reservation_id)
            if stored is None:
                raise ValueError("receipt refers to an unknown reservation")
            if reservation.reservation_id in receipts:
                raise ValueError("reservation already has a receipt")
            if (
                reservation.kind != stored["kind"]
                or reservation.amount != stored["amount"]
                or reservation.metadata != stored["metadata"]
            ):
                raise ValueError("receipt reservation does not match the stored reservation")
            self._append({
                "type": "receipt", "reservation_id": reservation.reservation_id,
                "kind": stored["kind"], "amount": stored["amount"],
                "metadata": dict(stored["metadata"]), "success": success,
                "result": receipt_result, "wall_time": time.time(),
            })

    def summary(self) -> dict[str, Any]:
        self._preflight()
        with self._exclusive_lock():
            records = self._records()
            reserved = {
                kind: self._used_from_records(records, kind) for kind in self.budget
            }
            receipt_rows = [row for row in records if row.get("type") == "receipt"]
            completed = len(receipt_rows)
            receipt_ids = {
                item.get("reservation_id")
                for item in records
                if item.get("type") == "receipt"
            }
            unfinished = sum(
                1 for row in records
                if row.get("type") == "reservation"
                and row.get("reservation_id") not in receipt_ids
            )
            successes = {
                kind: sum(
                    int(row["amount"])
                    for row in receipt_rows
                    if row["kind"] == kind and row["success"]
                )
                for kind in self.budget
            }
            failures = {
                kind: sum(
                    int(row["amount"])
                    for row in receipt_rows
                    if row["kind"] == kind and not row["success"]
                )
                for kind in self.budget
            }
            wall_seconds = {
                kind: sum(
                    float(row["result"].get("wall_seconds", 0.0))
                    for row in receipt_rows
                    if row["kind"] == kind
                )
                for kind in self.budget
            }
            return {
                "budget": dict(self.budget),
                "reserved": reserved,
                "remaining": {
                    k: self.budget[k] - reserved[k] for k in self.budget
                },
                "completed_receipts": completed,
                "successful_amount": successes,
                "failed_amount": failures,
                "wall_seconds": wall_seconds,
                "unfinished_reservations": unfinished,
                "record_count": len(records),
                "root_record_hash": records[0]["record_hash"] if records else None,
                "tip_record_hash": records[-1]["record_hash"] if records else None,
            }

    def verified_records(self) -> tuple[dict[str, Any], ...]:
        """Return a defensive snapshot after verifying the complete sealed chain."""
        self._preflight()
        with self._exclusive_lock():
            records = self._records()
            return tuple(_json_value(record) for record in records)

    def successful_receipt_pairs(
        self, kind: Optional[str] = None
    ) -> tuple[tuple[dict[str, Any], dict[str, Any]], ...]:
        """Return successful reservation/receipt pairs in durable receipt order."""
        if kind is not None and kind not in self.budget:
            raise KeyError(kind)
        records = self.verified_records()
        reservations = {
            row["reservation_id"]: row
            for row in records
            if row["type"] == "reservation"
        }
        return tuple(
            (
                _json_value(reservations[row["reservation_id"]]),
                _json_value(row),
            )
            for row in records
            if row["type"] == "receipt"
            and row["success"] is True
            and (kind is None or row["kind"] == kind)
        )

    def successful_output_hashes(self, kind: str) -> tuple[str, ...]:
        """Return successful output hashes in durable receipt order."""
        self._preflight()
        if kind not in self.budget:
            raise KeyError(kind)
        with self._exclusive_lock():
            records = self._records()
            values = []
            for row in records:
                if (
                    row.get("type") != "receipt"
                    or row.get("kind") != kind
                    or row.get("success") is not True
                ):
                    continue
                output_hash = row["result"].get("output_hash")
                self._validate_hash(output_hash, "output_hash")
                values.append(output_hash)
            return tuple(values)
