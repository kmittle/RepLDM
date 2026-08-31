"""Transactional scorer for the frozen HPSv2 relational-renderer experiment.

Reads a run's manifest, runs the configured Scorers (self-contained metric modules
under scorers/, each decoupled from Sana), and writes scores.jsonl. Each metric is
declared in a yaml config (configs/eval_common.yaml). Weights are validated up front
so a missing metric is SKIPPED with a warning instead of crashing the run.

Resume is additive when enabled metrics and their execution provenance still match.
Strict runs bind source, package/runtime, model asset, and preprocessing metadata;
any drift invalidates the row before old columns can be reused. Progress is written
only to an attempt file; canonical scores and their receipt are published at success.

  /home/bycao/miniforge3/envs/repldm_eval/bin/python \
      eval-pipeline/score_hpsv2_relational_renderer.py \
      --run_dir outputs/hpsv2_relational_renderer/full_v1 \
      --config eval-pipeline/configs/hpsv2_full_scoring_v1.yaml \
      --device cuda:2 --strict --require-scorer-provenance \
      --require-exclusive-gpu
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
import threading
import uuid

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)

import yaml  # noqa: E402
import scorers  # noqa: E402,F401  (import registers all metrics)
import scorers.base as scorer_base  # noqa: E402
from scorers.base import REGISTRY  # noqa: E402
from scorer_provenance import (  # noqa: E402
    SCORER_PROVENANCE_SCHEMA,
    build_scorer_provenance,
    registered_scorer_provenance_contract,
    validate_hardened_score_rows,
)
from s7_provenance import (  # noqa: E402
    PROVENANCE_SCHEMA,
    json_sha256,
    validate_design_rows,
    validate_run_contract,
    validate_scores_against_manifest,
    validate_sidecar,
)


EXCLUSIVE_CUDA_EXECUTION_SCHEMA = "repldm_exclusive_cuda_execution_v1"
EXCLUSIVE_CUDA_POLL_SECONDS = 2.0
SCORING_SUCCESS_SCHEMA = "repldm_scoring_success_v1"
SCORING_SUCCESS_NAME = "scoring_success.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def exclusive_cuda_execution_contract(device):
    """Describe the fail-closed process monitor used by one scoring run."""
    normalized_device = str(device)
    if re.fullmatch(r"cuda:\d+", normalized_device) is None:
        raise ValueError("exclusive CUDA execution requires an explicit cuda:N device")
    return {
        "schema": EXCLUSIVE_CUDA_EXECUTION_SCHEMA,
        "exclusive_cuda_process": True,
        "device": normalized_device,
        "poll_seconds": EXCLUSIVE_CUDA_POLL_SECONDS,
        "query_backend": "nvidia-smi",
        "allowed_process_policy": "scorer_pid_only",
    }


def validate_exclusive_cuda_execution_contract(
    contract, observed_sha256, *, expected_device
):
    """Require the exact watchdog contract and its canonical hash."""
    expected = exclusive_cuda_execution_contract(expected_device)
    if contract != expected:
        raise ValueError("exclusive CUDA scoring execution contract differs")
    digest = json_sha256(expected)
    if observed_sha256 != digest:
        raise ValueError("exclusive CUDA scoring execution hash differs")
    return digest


def _task_id_binding(rows, *, label):
    """Bind row count and ordered task identities without copying all IDs."""
    task_ids = []
    observed = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{label} contains a non-mapping row")
        task_id = row.get("id")
        if not isinstance(task_id, str) or not task_id or task_id in observed:
            raise ValueError(f"{label} contains an empty or duplicate task id")
        observed.add(task_id)
        task_ids.append(task_id)
    return {
        "row_count": len(task_ids),
        "task_ids_sha256": json_sha256(task_ids),
    }


def _require_sha256(value, *, label, optional=False):
    if optional and value is None:
        return
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} is not a lowercase SHA-256")


def build_scoring_success_receipt(
    *,
    scores_sha256,
    score_rows,
    manifest_sha256,
    manifest_rows,
    run_config_sha256,
    run_contract_sha256,
    scoring_config_sha256,
    scoring_config,
    metric_names,
    params,
    strict,
    required_scorer_provenance_schema,
    scorer_provenance,
    scorer_provenance_sha256,
    cuda_execution_provenance,
    cuda_execution_provenance_sha256,
):
    """Build the deterministic receipt for one fully published score table."""
    _require_sha256(scores_sha256, label="scores SHA-256")
    _require_sha256(manifest_sha256, label="manifest SHA-256")
    _require_sha256(
        run_config_sha256, label="run config SHA-256", optional=True
    )
    _require_sha256(
        run_contract_sha256, label="run contract SHA-256", optional=True
    )
    _require_sha256(scoring_config_sha256, label="scoring config SHA-256")
    if type(strict) is not bool:
        raise ValueError("strict scoring flag must be boolean")
    if not isinstance(scoring_config, dict) or not isinstance(params, dict):
        raise ValueError("scoring config and params must be mappings")
    metrics = list(metric_names)
    if any(not isinstance(name, str) or not name for name in metrics):
        raise ValueError("metric names must be non-empty strings")

    scorer_contract = (
        dict(scorer_provenance) if scorer_provenance is not None else None
    )
    if (scorer_contract is None) != (scorer_provenance_sha256 is None):
        raise ValueError("scorer provenance payload and hash must appear together")
    if scorer_contract is not None:
        _require_sha256(
            scorer_provenance_sha256, label="scorer provenance SHA-256"
        )
        if json_sha256(scorer_contract) != scorer_provenance_sha256:
            raise ValueError("scorer provenance receipt binding differs")

    cuda_contract = (
        dict(cuda_execution_provenance)
        if cuda_execution_provenance is not None
        else None
    )
    if (cuda_contract is None) != (cuda_execution_provenance_sha256 is None):
        raise ValueError("CUDA execution payload and hash must appear together")
    if cuda_contract is not None:
        _require_sha256(
            cuda_execution_provenance_sha256,
            label="CUDA execution provenance SHA-256",
        )
        if json_sha256(cuda_contract) != cuda_execution_provenance_sha256:
            raise ValueError("CUDA execution receipt binding differs")

    scores_binding = _task_id_binding(score_rows, label="scores")
    manifest_binding = _task_id_binding(manifest_rows, label="manifest")
    return {
        "schema": SCORING_SUCCESS_SCHEMA,
        "scores": {"sha256": scores_sha256, **scores_binding},
        "manifest": {"sha256": manifest_sha256, **manifest_binding},
        "run_contract": {
            "config_sha256": run_config_sha256,
            "run_contract_sha256": run_contract_sha256,
        },
        "scoring_config": {
            "sha256": scoring_config_sha256,
            "payload": dict(scoring_config),
            "payload_sha256": json_sha256(dict(scoring_config)),
            "effective": {
                "metrics": metrics,
                "params": dict(params),
                "strict": strict,
                "required_scorer_provenance_schema": (
                    required_scorer_provenance_schema
                ),
            },
        },
        "scorer_provenance": {
            "contract": scorer_contract,
            "sha256": scorer_provenance_sha256,
        },
        "cuda_execution_provenance": {
            "contract": cuda_contract,
            "sha256": cuda_execution_provenance_sha256,
        },
    }


def scoring_success_receipt_bytes(receipt):
    """Serialize a receipt canonically for byte-for-byte recomputation."""
    return (
        json.dumps(
            dict(receipt),
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def validate_scoring_success_receipt(receipt, **expected_inputs):
    """Rebuild and compare a scoring receipt from current artifacts."""
    if not isinstance(receipt, dict):
        raise ValueError("scoring success receipt must be a mapping")
    expected = build_scoring_success_receipt(**expected_inputs)
    if dict(receipt) != expected:
        raise ValueError("scoring success receipt differs from current artifacts")
    return expected


CFG_SCORING_SCHEMA = "cfg_scoring_contract_v1"
CFG_ACTION_SCHEMA = "cfg_baselines_v1"
CFG_SCORING_METRICS = ("pixel", "clip", "hps", "iqa")
CFG_SCORING_PARAMS = {
    "patch_crops": 5,
    "clip_model": "ViT-B/32",
    "clipscore_w": 2.5,
}
CFG_REQUIRED_SCORE_KEYS = (
    "colorfulness",
    "laplacian_sharpness",
    "clipped_fraction",
    "mean_saturation",
    "contrast_std",
    "clip_cosine",
    "clipscore",
    "hpsv2",
    "topiq_nr",
)
CFG_SCORING_FIELDS = {"metrics", "strict", "params", "required_score_keys"}


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _unique_rows(rows, label):
    result = {}
    for row in rows:
        row_id = str(row.get("id", ""))
        if not row_id or row_id in result:
            raise ValueError(f"{label} contains duplicate or empty id {row_id!r}")
        result[row_id] = row
    return result


def resolve_device(device, cuda_available):
    """Normalize CLI GPU indices to the Torch ``cuda:N`` device form."""
    value = str(device).strip()
    if value.isdecimal():
        value = f"cuda:{value}"
    if value != "cpu" and not cuda_available:
        return "cpu"
    return value


def finite_numeric(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def manifest_uses_hash_binding(manifest):
    """Require all-or-none image/run hashes and return whether they are present."""
    required = ("image_sha256", "run_contract_sha256")
    presence = [any(key in row for key in required) for row in manifest]
    if any(presence) and not all(presence):
        raise RuntimeError("manifest mixes hash-bound and unbound rows")
    if not any(presence):
        return False
    for row in manifest:
        for key in required:
            value = row.get(key)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise RuntimeError(
                    f"{row.get('id')}: manifest {key} is not a lowercase SHA-256"
                )
    return True


def load_hash_bound_image_bytes(run_dir, row):
    """Read one regular image once and bind the returned bytes to its manifest hash."""
    root = os.path.realpath(os.path.abspath(run_dir))
    relative = row.get("image_path")
    if not isinstance(relative, str) or not relative:
        raise RuntimeError(f"{row.get('id')}: hash-bound manifest lacks image_path")
    candidate = os.path.abspath(os.path.join(root, relative))
    try:
        contained = os.path.commonpath((root, candidate)) == root
    except ValueError:
        contained = False
    if not contained or os.path.realpath(candidate) != candidate:
        raise RuntimeError(f"{row.get('id')}: hash-bound image path is unsafe")

    descriptor = -1
    try:
        descriptor = os.open(
            candidate,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"{row.get('id')}: hash-bound image path is unsafe")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise RuntimeError(f"{row.get('id')}: hash-bound image path is unsafe") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    identity = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity):
        raise RuntimeError(f"{row.get('id')}: image changed while it was read")
    payload = b"".join(chunks)
    if hashlib.sha256(payload).hexdigest() != row.get("image_sha256"):
        raise RuntimeError(f"{row.get('id')}: image changed after manifest validation")
    return payload


def validate_hash_bound_images(run_dir, manifest):
    """Verify every bound image before score reuse and again after scoring."""
    for row in manifest:
        load_hash_bound_image_bytes(run_dir, row)


def _nvidia_smi(arguments):
    try:
        result = subprocess.run(
            ["nvidia-smi", *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("cannot query NVIDIA compute-process state") from exc
    if result.returncode != 0:
        raise RuntimeError(
            "cannot query NVIDIA compute-process state: " + result.stderr.strip()
        )
    return result.stdout


def _exclusive_cuda_conflicts(device, allowed_pid):
    if "CUDA_VISIBLE_DEVICES" in os.environ or "CUDA_DEVICE_ORDER" in os.environ:
        raise RuntimeError("exclusive GPU scoring requires unmasked physical CUDA ids")
    matched = re.fullmatch(r"cuda:(\d+)", str(device))
    if matched is None:
        raise RuntimeError("exclusive GPU scoring requires an explicit cuda:N device")
    physical_index = int(matched.group(1))
    inventory = _nvidia_smi(
        ["--query-gpu=index,uuid", "--format=csv,noheader,nounits"]
    )
    matches = []
    for line in inventory.splitlines():
        fields = [field.strip() for field in line.split(",", 1)]
        if len(fields) == 2 and fields[0] == str(physical_index):
            matches.append(fields[1])
    if len(matches) != 1:
        raise RuntimeError("exclusive scoring GPU is absent or ambiguous")
    target_uuid = matches[0]
    processes = _nvidia_smi(
        [
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    conflicts = []
    for line in processes.splitlines():
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4 or fields[0] != target_uuid:
            continue
        try:
            pid = int(fields[1])
        except ValueError:
            pid = -1
        if pid != int(allowed_pid):
            conflicts.append(
                {
                    "gpu_uuid": fields[0],
                    "pid": pid,
                    "process_name": fields[2],
                    "used_memory_mib": fields[3],
                }
            )
    return conflicts


class ExclusiveCudaWatchdog:
    """Continuously fail scoring if a foreign process appears on its GPU."""

    def __init__(
        self, device, enabled, poll_seconds=EXCLUSIVE_CUDA_POLL_SECONDS
    ):
        self.device = device
        self.enabled = bool(enabled)
        self.poll_seconds = float(poll_seconds)
        self.pid = os.getpid()
        self.stop_event = threading.Event()
        self.failures = []
        self.thread = None

    def _check_now(self):
        conflicts = _exclusive_cuda_conflicts(self.device, self.pid)
        if conflicts:
            raise RuntimeError(
                "foreign compute process appeared during scoring: "
                + json.dumps(conflicts, sort_keys=True)
            )

    def _monitor(self):
        while not self.stop_event.wait(self.poll_seconds):
            try:
                self._check_now()
            except BaseException as exc:
                self.failures.append(exc)
                self.stop_event.set()
                return

    def assert_healthy(self):
        if self.failures:
            raise RuntimeError("exclusive GPU scoring monitor failed") from self.failures[0]

    def __enter__(self):
        if self.enabled:
            self._check_now()
            self.thread = threading.Thread(
                target=self._monitor,
                name="exclusive-cuda-watchdog",
                daemon=True,
            )
            self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback_value):
        if not self.enabled:
            return False
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(5.0, self.poll_seconds * 2.0))
            if self.thread.is_alive() and exc_type is None:
                raise RuntimeError("exclusive GPU scoring monitor did not stop")
        if exc_type is None:
            self._check_now()
            self.assert_healthy()
        return False


def exclusive_cuda_required(cfg, cli_required=False):
    """Resolve the watchdog requirement from config and the CLI."""
    configured = cfg.get("exclusive_cuda_process", False)
    if type(configured) is not bool:
        raise RuntimeError("exclusive_cuda_process must be true or false")
    return configured or bool(cli_required)


def required_scorer_provenance_schema(cfg, run_config, cli_required=False):
    """Resolve an optional fail-closed provenance requirement."""
    requested = []
    provenance_config = cfg.get("scorer_provenance")
    if provenance_config is not None:
        if not isinstance(provenance_config, dict) or set(provenance_config) != {
            "required_schema"
        }:
            raise RuntimeError(
                "scorer_provenance config must contain only required_schema"
            )
        requested.append(provenance_config.get("required_schema"))
    run_requirement = run_config.get("required_scorer_provenance_schema")
    if run_requirement is not None:
        requested.append(run_requirement)
    if cli_required:
        requested.append(SCORER_PROVENANCE_SCHEMA)
    if not requested:
        return None
    if any(value != SCORER_PROVENANCE_SCHEMA for value in requested):
        raise RuntimeError(
            f"unsupported scorer provenance requirement; expected "
            f"{SCORER_PROVENANCE_SCHEMA!r}"
        )
    return SCORER_PROVENANCE_SCHEMA


@contextmanager
def scoring_output_lock(run_dir):
    """Exclude generation and concurrent scoring from one run directory."""
    # Generation uses the same lock file, so scoring cannot consume a manifest
    # while a resumed generator may still replace images or sidecars.
    lock_path = os.path.join(os.path.abspath(run_dir), ".generate.lock")
    handle = open(lock_path, "a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "generation or scoring is already running for "
                f"{os.path.abspath(run_dir)}"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


@contextmanager
def atomic_text_writer(path):
    """Publish a text file through a unique same-directory temporary file."""
    destination = os.path.abspath(path)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination)}.",
        suffix=".tmp",
        dir=os.path.dirname(destination),
    )
    handle = None
    try:
        handle = os.fdopen(fd, "w", encoding="utf-8")
        with handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        if handle is None:
            os.close(fd)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _fsync_directory(path):
    """Durably publish directory-entry changes."""
    descriptor = os.open(os.path.abspath(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_regular_bytes(path, label):
    """Read one regular file through a no-follow descriptor."""
    descriptor = -1
    try:
        descriptor = os.open(
            os.path.abspath(path),
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"{label} is not a regular file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise RuntimeError(f"{label} is missing or unsafe") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    identity = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity):
        raise RuntimeError(f"{label} changed while it was read")
    return b"".join(chunks)


def _jsonl_from_bytes(payload, label):
    rows = []
    try:
        text = payload.decode("utf-8")
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{label} line {line_number} is not a mapping")
            rows.append(row)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSONL") from exc
    return rows


def _scoring_receipt_context(
    *,
    run_dir,
    scoring_config_path,
    scoring_config,
    manifest,
    metric_names,
    params,
    strict,
    required_scorer_provenance_schema,
    scorer_provenance,
    scorer_provenance_sha256,
    cuda_execution_provenance,
    cuda_execution_provenance_sha256,
):
    """Capture every non-score input required to rebuild a success receipt."""
    manifest_payload = _read_regular_bytes(
        os.path.join(run_dir, "manifest.jsonl"), "manifest"
    )
    scoring_config_payload = _read_regular_bytes(
        scoring_config_path, "scoring config"
    )
    run_config_path = os.path.join(run_dir, "config.json")
    run_config_sha256 = None
    run_contract_sha256 = None
    if os.path.lexists(run_config_path):
        run_config_payload = _read_regular_bytes(run_config_path, "run config")
        run_config_sha256 = hashlib.sha256(run_config_payload).hexdigest()
        try:
            current_run_config = json.loads(run_config_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("run config is not valid JSON") from exc
        if not isinstance(current_run_config, dict):
            raise RuntimeError("run config must contain a mapping")
        run_contract_sha256 = current_run_config.get("run_contract_sha256")
    return {
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "manifest_rows": list(manifest),
        "run_config_sha256": run_config_sha256,
        "run_contract_sha256": run_contract_sha256,
        "scoring_config_sha256": hashlib.sha256(
            scoring_config_payload
        ).hexdigest(),
        "scoring_config": dict(scoring_config),
        "metric_names": list(metric_names),
        "params": dict(params),
        "strict": bool(strict),
        "required_scorer_provenance_schema": (
            required_scorer_provenance_schema
        ),
        "scorer_provenance": scorer_provenance,
        "scorer_provenance_sha256": scorer_provenance_sha256,
        "cuda_execution_provenance": cuda_execution_provenance,
        "cuda_execution_provenance_sha256": (
            cuda_execution_provenance_sha256
        ),
    }


def load_verified_score_publication(run_dir, receipt_context):
    """Load canonical rows only when their recomputed receipt is exact."""
    scores_path = os.path.join(run_dir, "scores.jsonl")
    receipt_path = os.path.join(run_dir, SCORING_SUCCESS_NAME)
    if not os.path.lexists(scores_path) and not os.path.lexists(receipt_path):
        return []
    if not os.path.lexists(scores_path) or not os.path.lexists(receipt_path):
        raise ValueError("canonical scores and success receipt must appear together")
    scores_payload = _read_regular_bytes(scores_path, "canonical scores")
    receipt_payload = _read_regular_bytes(receipt_path, "scoring success receipt")
    score_rows = _jsonl_from_bytes(scores_payload, "canonical scores")
    try:
        receipt = json.loads(receipt_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("scoring success receipt is not valid JSON") from exc
    expected = validate_scoring_success_receipt(
        receipt,
        scores_sha256=hashlib.sha256(scores_payload).hexdigest(),
        score_rows=score_rows,
        **receipt_context,
    )
    if receipt_payload != scoring_success_receipt_bytes(expected):
        raise ValueError("scoring success receipt is not canonical")
    return score_rows


def _clear_canonical_scoring_outputs(run_dir):
    """Invalidate the receipt first, then remove both canonical artifacts."""
    changed = False
    for name in (SCORING_SUCCESS_NAME, "scores.jsonl"):
        path = os.path.join(run_dir, name)
        if not os.path.lexists(path):
            continue
        try:
            os.unlink(path)
        except IsADirectoryError as exc:
            raise RuntimeError(f"canonical scoring artifact is unsafe: {path}") from exc
        changed = True
    if changed:
        _fsync_directory(run_dir)


def _attempt_scores_path(run_dir):
    return os.path.join(
        os.path.abspath(run_dir),
        f".scores.attempt-{os.getpid()}-{uuid.uuid4().hex}.jsonl",
    )


def _publish_scoring_attempt(run_dir, attempt_path, manifest, receipt_context):
    """Atomically publish scores, then their receipt, or leave neither resumable."""
    scores_path = os.path.join(run_dir, "scores.jsonl")
    receipt_path = os.path.join(run_dir, SCORING_SUCCESS_NAME)
    attempt_payload = _read_regular_bytes(attempt_path, "scoring attempt")
    score_rows = _jsonl_from_bytes(attempt_payload, "scoring attempt")
    if [row.get("id") for row in score_rows] != [row.get("id") for row in manifest]:
        raise RuntimeError("scoring attempt task order differs from the manifest")
    receipt = build_scoring_success_receipt(
        scores_sha256=hashlib.sha256(attempt_payload).hexdigest(),
        score_rows=score_rows,
        **receipt_context,
    )
    published_scores = False
    try:
        os.replace(attempt_path, scores_path)
        published_scores = True
        _fsync_directory(run_dir)
        if _read_regular_bytes(scores_path, "canonical scores") != attempt_payload:
            raise RuntimeError("canonical scores changed during publication")
        receipt_payload = scoring_success_receipt_bytes(receipt)
        with atomic_text_writer(receipt_path) as handle:
            handle.write(receipt_payload.decode("ascii"))
        _fsync_directory(run_dir)
        load_verified_score_publication(run_dir, receipt_context)
    except BaseException:
        if published_scores:
            _clear_canonical_scoring_outputs(run_dir)
        raise
    return score_rows


def cfg_scoring_contract(run_config, metric_names, params, strict):
    """Load and enforce the scoring recipe bound by a registered CFG run."""
    if run_config.get("cfg_baseline_registered") is not True:
        return None
    actions_path = run_config.get("actions_yaml")
    actions_hash = run_config.get("actions_sha256")
    if not isinstance(actions_path, str) or not os.path.isfile(actions_path):
        raise RuntimeError("registered CFG run actions YAML is unavailable for scoring")
    with open(actions_path, "rb") as handle:
        actions_bytes = handle.read()
    if hashlib.sha256(actions_bytes).hexdigest() != actions_hash:
        raise RuntimeError("registered CFG run actions YAML changed before scoring")
    actions_config = yaml.safe_load(actions_bytes) or {}
    if not isinstance(actions_config, dict):
        raise RuntimeError("registered CFG actions YAML must contain a mapping")
    if actions_config.get("schema") != CFG_ACTION_SCHEMA:
        raise RuntimeError("registered CFG actions YAML has the wrong schema")
    scoring = actions_config.get("scoring")
    if not isinstance(scoring, dict):
        raise RuntimeError("registered CFG actions lack a scoring contract")
    if set(scoring) != CFG_SCORING_FIELDS:
        raise RuntimeError("registered CFG scoring fields differ from the v1 contract")
    registered_metrics = scoring.get("metrics")
    registered_params = scoring.get("params")
    if registered_metrics != list(CFG_SCORING_METRICS):
        raise RuntimeError("registered CFG YAML metrics differ from the v1 contract")
    if list(metric_names) != list(CFG_SCORING_METRICS):
        raise RuntimeError(
            f"registered CFG scoring requires metrics {list(CFG_SCORING_METRICS)}, "
            f"got {list(metric_names)}"
        )
    if strict is not True or scoring.get("strict") is not True:
        raise RuntimeError("registered CFG scoring requires --strict")
    if json_sha256(registered_params) != json_sha256(CFG_SCORING_PARAMS):
        raise RuntimeError("registered CFG YAML params differ from the v1 contract")
    if json_sha256(params) != json_sha256(CFG_SCORING_PARAMS):
        raise RuntimeError("scoring config params differ from CFG registration")
    if scoring.get("required_score_keys") != list(CFG_REQUIRED_SCORE_KEYS):
        raise RuntimeError(
            "registered CFG required score keys differ from the v1 contract"
        )
    return {
        "schema": CFG_SCORING_SCHEMA,
        "action_schema": CFG_ACTION_SCHEMA,
        "metrics": list(CFG_SCORING_METRICS),
        "strict": True,
        "params": dict(CFG_SCORING_PARAMS),
        "required_score_keys": list(CFG_REQUIRED_SCORE_KEYS),
        "actions_sha256": actions_hash,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--config", default=os.path.join(THIS, "configs", "eval_common.yaml"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--metrics", default=None, help="comma-separated override of config 'metrics'")
    ap.add_argument(
        "--strict", action="store_true",
        help="fail if any requested scorer cannot load or score an image",
    )
    ap.add_argument(
        "--require-scorer-provenance",
        action="store_true",
        help="require the hardened scorer provenance schema (implies --strict)",
    )
    ap.add_argument(
        "--require-exclusive-gpu",
        action="store_true",
        help="stop if another compute PID appears on the explicit physical cuda:N",
    )
    args = ap.parse_args()
    if args.require_scorer_provenance:
        args.strict = True

    with scoring_output_lock(args.run_dir):
        return _score_run(args, ap)


def _score_run(args, ap):
    with open(args.config) as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise RuntimeError("scoring config must contain a YAML mapping")
    import torch
    device = resolve_device(args.device, torch.cuda.is_available())
    try:
        torch.device(device)
    except (RuntimeError, TypeError) as exc:
        ap.error(f"invalid --device {args.device!r}: {exc}")
    watchdog = ExclusiveCudaWatchdog(
        device,
        exclusive_cuda_required(
            cfg, getattr(args, "require_exclusive_gpu", False)
        ),
    )
    attempt_path = _attempt_scores_path(args.run_dir)
    try:
        with watchdog:
            publication = _score_run_impl(
                args, ap, cfg, torch, device, watchdog, attempt_path
            )
        watchdog.assert_healthy()
        if publication is None:
            return None
        current_context = _scoring_receipt_context(
            **publication["receipt_context_args"]
        )
        if current_context != publication["receipt_context"]:
            raise RuntimeError("scoring inputs changed before publication")
        _publish_scoring_attempt(
            args.run_dir,
            attempt_path,
            publication["manifest"],
            current_context,
        )
        print(
            f"scores -> {os.path.join(args.run_dir, 'scores.jsonl')}",
            flush=True,
        )
        print(
            f"receipt -> {os.path.join(args.run_dir, SCORING_SUCCESS_NAME)}",
            flush=True,
        )
        return None
    except BaseException:
        _clear_canonical_scoring_outputs(args.run_dir)
        raise
    finally:
        try:
            os.unlink(attempt_path)
        except FileNotFoundError:
            pass


def _score_run_impl(args, ap, cfg, torch, device, watchdog, attempt_path):
    metric_names = args.metrics.split(",") if args.metrics else cfg.get("metrics", [])
    params = dict(cfg.get("params", {}))
    exclusive_execution = (
        exclusive_cuda_execution_contract(device) if watchdog.enabled else None
    )
    exclusive_execution_sha256 = (
        json_sha256(exclusive_execution) if exclusive_execution is not None else None
    )

    manifest = load_jsonl(os.path.join(args.run_dir, "manifest.jsonl"))
    if not manifest:
        raise RuntimeError("manifest.jsonl is empty or missing")
    manifest_by_id = _unique_rows(manifest, "manifest")
    run_config_path = os.path.join(args.run_dir, "config.json")
    run_config = {}
    if os.path.isfile(run_config_path):
        with open(run_config_path) as handle:
            run_config = json.load(handle)
    if (
        run_config.get("split_role") == "engineering_smoke"
        or run_config.get("engineering_only") is True
    ):
        raise RuntimeError(
            "structural engineering smoke forbids quality scoring"
        )
    required_provenance_schema = required_scorer_provenance_schema(
        cfg,
        run_config,
        getattr(args, "require_scorer_provenance", False),
    )
    if required_provenance_schema is not None and not args.strict:
        raise RuntimeError("hardened scorer provenance requires --strict")
    s7_run = any(
        run_config.get(flag) is True
        for flag in (
            "trajectory_registered",
            "scheduler_baseline_registered",
            "cfg_baseline_registered",
            "native_renderer_registered",
        )
    ) or any(
        row.get("provenance_schema") == PROVENANCE_SCHEMA for row in manifest
    )
    if s7_run:
        contract_hash = validate_run_contract(run_config)
        action_ids = [str(action.get("id")) for action in run_config.get("actions", [])]
        seeds = [int(value) for value in run_config.get("seeds", [])]
        validate_design_rows(
            manifest,
            expected_action_ids=action_ids or None,
            expected_seeds=seeds or None,
        )
        for row in manifest:
            if row.get("provenance_schema") != PROVENANCE_SCHEMA:
                raise RuntimeError(f"{row.get('id')}: missing S7 provenance schema")
            validate_sidecar(
                row,
                args.run_dir,
                expected_contract_sha256=contract_hash,
            )
    hash_bound_run = manifest_uses_hash_binding(manifest)
    if hash_bound_run:
        validate_hash_bound_images(args.run_dir, manifest)
    registered_scoring = cfg_scoring_contract(
        run_config, metric_names, params, args.strict
    )
    registered_scoring_sha256 = (
        json_sha256(registered_scoring) if registered_scoring is not None else None
    )
    # instantiate scorers (validate weights first; skip cleanly if missing/broken)
    active = []
    unavailable = []
    for name in metric_names:
        if name not in REGISTRY:
            message = f"unknown metric '{name}'"
            print(f"[warn] {message}, skipping", flush=True)
            unavailable.append(message)
            continue
        cls = REGISTRY[name]
        ready, msg = cls.weights_status(**params)
        if not ready:
            message = f"'{name}' weights missing -> {msg}"
            print(f"[skip] {message}", flush=True)
            unavailable.append(message)
            continue
        try:
            active.append((name, cls(device=device, **params)))
            print(f"[ok] loaded scorer '{name}' on {device}", flush=True)
            watchdog.assert_healthy()
        except Exception as e:
            message = f"'{name}' failed to init -> {e}"
            print(f"[skip] {message}", flush=True)
            unavailable.append(message)
    if args.strict and unavailable:
        raise RuntimeError("requested scorers unavailable: " + "; ".join(unavailable))
    if not active:
        _clear_canonical_scoring_outputs(args.run_dir)
        print("no active scorers; nothing to do", flush=True)
        return None

    scorer_provenance = None
    scorer_provenance_sha256 = None
    if args.strict:
        scorer_provenance, scorer_provenance_sha256 = build_scorer_provenance(
            active,
            params=params,
            device=device,
            runner_path=__file__,
            base_path=scorer_base.__file__,
            source_root=THIS,
        )
    registered_scorer_contract = None
    registered_scorer_hash = None
    for registration in (cfg, run_config):
        candidate_contract, candidate_hash = registered_scorer_provenance_contract(
            registration
        )
        if candidate_contract is not None:
            if (
                registered_scorer_contract is not None
                and candidate_contract != registered_scorer_contract
            ):
                raise RuntimeError("registered scorer provenance contracts disagree")
            registered_scorer_contract = candidate_contract
        if candidate_hash is not None:
            if (
                registered_scorer_hash is not None
                and candidate_hash != registered_scorer_hash
            ):
                raise RuntimeError("registered scorer provenance hashes disagree")
            registered_scorer_hash = candidate_hash
    if registered_scorer_hash is not None:
        if scorer_provenance is None:
            raise RuntimeError(
                "a registered scorer provenance contract requires --strict"
            )
        if scorer_provenance_sha256 != registered_scorer_hash:
            raise RuntimeError(
                "loaded scorer provenance differs from the registered contract"
            )
        if registered_scorer_contract is not None and (
            scorer_provenance != registered_scorer_contract
        ):
            raise RuntimeError(
                "loaded scorer provenance differs from the registered payload"
            )

    receipt_context_args = {
        "run_dir": args.run_dir,
        "scoring_config_path": args.config,
        "scoring_config": cfg,
        "manifest": manifest,
        "metric_names": metric_names,
        "params": params,
        "strict": args.strict,
        "required_scorer_provenance_schema": required_provenance_schema,
        "scorer_provenance": scorer_provenance,
        "scorer_provenance_sha256": scorer_provenance_sha256,
        "cuda_execution_provenance": exclusive_execution,
        "cuda_execution_provenance_sha256": exclusive_execution_sha256,
    }
    receipt_context = _scoring_receipt_context(**receipt_context_args)
    try:
        existing_rows = load_verified_score_publication(
            args.run_dir, receipt_context
        )
    except (RuntimeError, ValueError) as exc:
        print(
            f"[warn] canonical scores are not resumable -> {exc}",
            flush=True,
        )
        existing_rows = []
    _clear_canonical_scoring_outputs(args.run_dir)
    existing = _unique_rows(existing_rows, "scores")
    if hash_bound_run and existing_rows:
        try:
            validate_scores_against_manifest(manifest, existing_rows)
        except ValueError:
            # A valid receipt cannot make rows for replaced images reusable.
            existing = {}

    output_keys = [key for _, scorer in active for key, _ in scorer.OUTPUT_KEYS]
    if len(output_keys) != len(set(output_keys)):
        raise RuntimeError("active scorers declare duplicate output keys")
    need_keys = set(output_keys)
    if registered_scoring is not None and need_keys != set(CFG_REQUIRED_SCORE_KEYS):
        raise RuntimeError(
            "registered CFG scorer outputs differ from required_score_keys"
        )

    def score_provenance_is_current(row, score):
        if (
            score.get("scoring_execution_provenance") != exclusive_execution
            or score.get("scoring_execution_provenance_sha256")
            != exclusive_execution_sha256
        ):
            return False
        if hash_bound_run:
            if (
                score.get("image_sha256") != row.get("image_sha256")
                or score.get("run_contract_sha256")
                != row.get("run_contract_sha256")
            ):
                return False
        if s7_run:
            if registered_scoring is not None and not (
                score.get("scoring_contract") == registered_scoring
                and score.get("scoring_contract_sha256")
                == registered_scoring_sha256
            ):
                return False
        if scorer_provenance is not None:
            return (
                score.get("scorer_provenance") == scorer_provenance
                and score.get("scorer_provenance_sha256")
                == scorer_provenance_sha256
            )
        return True

    def score_is_current(row):
        if row["id"] not in existing or not need_keys.issubset(existing[row["id"]].keys()):
            return False
        score = existing[row["id"]]
        if not score_provenance_is_current(row, score):
            return False
        return not (args.strict or hash_bound_run) or all(
            finite_numeric(score.get(key)) for key in need_keys
        )

    todo = [r for r in manifest if not score_is_current(r)]
    print(f"{len(manifest)} images; {len(todo)} to (re)score with {[n for n, _ in active]}", flush=True)

    def flush():
        with atomic_text_writer(attempt_path) as out:
            for r in manifest:
                if r["id"] in existing:
                    out.write(
                        json.dumps(
                            existing[r["id"]],
                            allow_nan=not args.strict and registered_scoring is None,
                        )
                        + "\n"
                    )

    from PIL import Image
    for i, r in enumerate(todo):
        watchdog.assert_healthy()
        if hash_bound_run:
            image_bytes = load_hash_bound_image_bytes(args.run_dir, r)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            image_path = os.path.join(args.run_dir, r["image_path"])
            img = Image.open(image_path).convert("RGB")
        metadata_keys = (
            "id", "prompt_index", "bucket", "seed", "scale", "action_id",
            "action_type", "band_scales", "image_path",
        )
        prior = existing.get(r["id"])
        rec = (
            prior
            if prior is not None and score_provenance_is_current(r, prior)
            else {key: r[key] for key in metadata_keys if key in r}
        )
        if hash_bound_run:
            rec.update(
                {
                    "image_sha256": r["image_sha256"],
                    "run_contract_sha256": r["run_contract_sha256"],
                }
            )
        if s7_run:
            rec.update(
                {
                    "provenance_schema": PROVENANCE_SCHEMA,
                    "action_sha256": r.get("action_sha256"),
                }
            )
            if registered_scoring is not None:
                rec.update(
                    {
                        "scoring_contract": registered_scoring,
                        "scoring_contract_sha256": registered_scoring_sha256,
                    }
                )
        if scorer_provenance is not None:
            rec.update(
                {
                    "scorer_provenance": scorer_provenance,
                    "scorer_provenance_sha256": scorer_provenance_sha256,
                }
            )
        if exclusive_execution is not None:
            rec.update(
                {
                    "scoring_execution_provenance": exclusive_execution,
                    "scoring_execution_provenance_sha256": (
                        exclusive_execution_sha256
                    ),
                }
            )
        for name, sc in active:
            scorer_keys = {key for key, _ in sc.OUTPUT_KEYS}
            if scorer_keys.issubset(rec) and (
                not (args.strict or s7_run)
                or all(finite_numeric(rec.get(key)) for key in scorer_keys)
            ):
                continue
            for key in scorer_keys:
                rec.pop(key, None)
            try:
                scored = sc.score_image(img, r["prompt"])
                if registered_scoring is not None and (
                    not isinstance(scored, dict) or set(scored) != scorer_keys
                ):
                    raise ValueError(
                        f"{name} returned fields outside its registered output contract"
                    )
                if (args.strict or hash_bound_run) and any(
                    not finite_numeric(scored.get(key)) for key in scorer_keys
                ):
                    raise ValueError(f"{name} returned non-finite strict scores")
                rec.update(scored)
            except Exception as e:
                print(f"[warn] {name} failed on {r['id']}: {e}", flush=True)
                if args.strict:
                    existing[r["id"]] = rec
                    flush()
                    raise RuntimeError(f"{name} failed on {r['id']}") from e
            watchdog.assert_healthy()
        existing[r["id"]] = rec
        if (i + 1) % 50 == 0 or i == len(todo) - 1:
            flush()
            print(f"  scored {i + 1}/{len(todo)}", flush=True)
    flush()
    watchdog.assert_healthy()
    if hash_bound_run:
        validate_hash_bound_images(args.run_dir, manifest)
        validate_scores_against_manifest(manifest, [existing[row["id"]] for row in manifest])
    if scorer_provenance is not None:
        validate_hardened_score_rows(
            [existing[row["id"]] for row in manifest],
            required_schema=required_provenance_schema
            or SCORER_PROVENANCE_SCHEMA,
            expected_sha256=registered_scorer_hash,
            expected_contract=registered_scorer_contract,
        )
    return {
        "manifest": manifest,
        "receipt_context_args": receipt_context_args,
        "receipt_context": receipt_context,
    }


if __name__ == "__main__":
    main()
