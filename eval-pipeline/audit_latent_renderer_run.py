"""Audit a latent-renderer run without inspecting action quality rankings."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import yaml
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from AttentionGuidance import (  # noqa: E402
    LAZY_LATENT_STRUCTURE_BASIS_ALIASES as _BASIS_ALIASES,
    LAZY_LATENT_STRUCTURE_BASIS_NAMES as _BASIS_NAMES,
    LAZY_LATENT_STRUCTURE_PROVIDER_ID as _LAZY_PROVIDER_ID,
    LATENT_STRUCTURE_PROVIDER_IMPLEMENTATION_ALIASES,
    normalize_latent_structure_bases as _normalize_bases,
    normalize_latent_structure_provider_implementation,
)
from scorer_provenance import (
    SCORER_PROVENANCE_SCHEMA,
    registered_scorer_provenance_contract,
    validate_hardened_score_rows,
)
from s7_provenance import validate_run_contract


DEFAULT_SCORE_KEYS = (
    "colorfulness",
    "laplacian_sharpness",
    "mean_saturation",
    "clipped_fraction",
    "contrast_std",
    "clip_cosine",
    "clipscore",
    "hpsv2",
    "imagereward",
    "patch_ir_mean",
    "patch_ir_std",
    "patch_ir_n",
    "aesthetic",
    "topiq_nr",
)
SPLIT_NAMES = {
    "development": "development",
    "train_search": "train",
    "validation_confirmation": "validation",
    "test_final": "test",
}
LATENT_RENDERER_SCHEDULER_MAPPINGS = {
    "legacy_unit",
    "euler_clean_endpoint",
}
LATENT_RENDERER_BASIS_NORMALIZATIONS = {
    "legacy_l2_to_rms",
    "match_rms",
}
LAZY_LATENT_STRUCTURE_BASIS_NAMES = tuple(_BASIS_NAMES)
LAZY_LATENT_STRUCTURE_BASIS_ALIASES = dict(_BASIS_ALIASES)
LAZY_LATENT_STRUCTURE_PROVIDER_ID = _LAZY_PROVIDER_ID

# A scheduler-native run must be an independently reviewed executable copy of
# the registration template.  Keep these literals local to this dependency-
# light auditor so it can validate a run without importing the GPU generator.
NATIVE_RENDERER_SCHEMA = "scheduler_native_fixed_headroom_actions_v1"
NATIVE_RENDERER_SOURCE_TEMPLATE = (
    "eval-pipeline/configs/scheduler_native_fixed_headroom_development.yaml"
)
NATIVE_RENDERER_AUTH_SCOPE = "development_only_fixed_headroom"
NATIVE_RENDERER_AUTH_FIELDS = {
    "reviewer",
    "reviewed_commit",
    "source_template",
    "source_template_sha256",
    "scope",
    "gpu_generation",
    "scoring",
    "method_selection",
}
NATIVE_RENDERER_SCORER_SCHEMA = "repldm_scorer_provenance_v1"


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _native_design_body(value: Dict[str, Any]) -> Dict[str, Any]:
    """Remove review/status metadata before comparing two native designs."""
    body = copy.deepcopy(value)
    for key in (
        "schema",
        "status",
        "authorization",
        "blocking_conditions",
        "registration_source",
        "scoring",
        "registered_scorer_provenance",
        "registered_scorer_provenance_sha256",
    ):
        body.pop(key, None)
    provider = body.get("required_provider")
    if isinstance(provider, dict):
        # The implementation status is the one field expected to change when
        # a registration-only template becomes an executable copy.
        provider.pop("implementation_status", None)
    return body


def _load_yaml_mapping(path: str | os.PathLike[str], *, label: str) -> Dict[str, Any]:
    try:
        with open(path) as handle:
            value = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"{label} is not readable YAML") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a YAML mapping")
    return value


def _validate_native_registration(
    source_actions_path: str | os.PathLike[str],
    source: Dict[str, Any],
    config: Dict[str, Any],
    *,
    registration_actions_path: str | os.PathLike[str] | None = None,
) -> Dict[str, str]:
    """Validate the reviewed executable copy and its two action hashes.

    ``actions_sha256`` always identifies the executable YAML supplied to the
    generator.  The separate ``native_renderer_source_template_sha256`` binds
    that copy to the frozen registration template.  Keeping both identities is
    important: passing the frozen registration file as the audit source would
    otherwise make a valid executable run look like action-hash drift.
    """
    if source.get("schema") != NATIVE_RENDERER_SCHEMA:
        raise ValueError(
            "registered native renderer must use the reviewed executable schema"
        )
    if source.get("status") != "authorized_development":
        raise ValueError("registered native renderer executable is not authorized")
    authorization = source.get("authorization")
    if not isinstance(authorization, dict):
        raise ValueError("registered native renderer executable lacks authorization")
    if set(authorization) != NATIVE_RENDERER_AUTH_FIELDS:
        raise ValueError("native renderer authorization fields differ from frozen v1")
    if not str(authorization.get("reviewer", "")).strip():
        raise ValueError("native renderer authorization reviewer is missing")
    if authorization.get("scope") != NATIVE_RENDERER_AUTH_SCOPE:
        raise ValueError("native renderer authorization scope differs from frozen v1")
    if authorization.get("source_template") != NATIVE_RENDERER_SOURCE_TEMPLATE:
        raise ValueError("native renderer authorization source_template differs from frozen v1")
    if authorization.get("gpu_generation") is not True:
        raise ValueError("native renderer executable is not authorized for generation")
    if authorization.get("scoring") is not True:
        raise ValueError("native renderer executable is not authorized for scoring")
    if authorization.get("method_selection") is not False:
        raise ValueError("native renderer executable cannot authorize method selection")

    reviewed_commit = str(authorization.get("reviewed_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", reviewed_commit):
        raise ValueError("native renderer reviewed_commit must be full lowercase 40-hex")
    template_hash = str(authorization.get("source_template_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", template_hash):
        raise ValueError("native renderer source_template_sha256 must be lowercase 64-hex")

    scoring = source.get("scoring")
    if not isinstance(scoring, dict):
        raise ValueError("registered native renderer executable lacks scoring provenance")
    scorer_hash = scoring.get("registered_scorer_provenance_sha256")
    if not isinstance(scorer_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", scorer_hash):
        raise ValueError("native renderer scorer provenance hash is invalid")
    if scoring.get("required_schema") != NATIVE_RENDERER_SCORER_SCHEMA:
        raise ValueError("native renderer scorer provenance schema differs from frozen v1")

    repo_root = ROOT.resolve()
    template_path = (repo_root / NATIVE_RENDERER_SOURCE_TEMPLATE).resolve()
    try:
        if os.path.commonpath((str(repo_root), str(template_path))) != str(repo_root):
            raise ValueError("native renderer source template escapes the repository")
    except ValueError:
        raise ValueError("native renderer source template escapes the repository")
    if not template_path.is_file():
        raise ValueError(f"native renderer source template is missing: {template_path}")
    if sha256_file(template_path) != template_hash:
        raise ValueError("native renderer source template hash differs from current bytes")

    # Verify that the reviewed commit really contained the bytes being used.
    try:
        head_commit = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        ancestry = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "merge-base",
                "--is-ancestor",
                reviewed_commit,
                head_commit,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if ancestry.returncode != 0:
            raise ValueError("native renderer reviewed_commit is not an ancestor of HEAD")
        reviewed_bytes = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo_root),
                "show",
                f"{reviewed_commit}:{NATIVE_RENDERER_SOURCE_TEMPLATE}",
            ],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("cannot verify native renderer reviewed template") from exc
    if hashlib.sha256(reviewed_bytes).hexdigest() != template_hash:
        raise ValueError("native renderer source template hash differs from reviewed Git bytes")
    try:
        reviewed_template = yaml.safe_load(reviewed_bytes) or {}
    except yaml.YAMLError as exc:
        raise ValueError("native renderer reviewed template is invalid YAML") from exc
    if not isinstance(reviewed_template, dict):
        raise ValueError("native renderer reviewed template must be a mapping")
    if _native_design_body(source) != _native_design_body(reviewed_template):
        raise ValueError("native renderer executable differs from the frozen design")

    executable_hash = sha256_file(source_actions_path)
    if config.get("actions_sha256") != executable_hash:
        raise ValueError("run config executable actions hash differs from audited YAML")
    if config.get("native_renderer_executable_actions_sha256") != executable_hash:
        raise ValueError("run config native executable actions hash is missing or stale")
    if config.get("native_renderer_source_template") != NATIVE_RENDERER_SOURCE_TEMPLATE:
        raise ValueError("run config native source template path is missing or stale")
    if config.get("native_renderer_source_template_sha256") != template_hash:
        raise ValueError("run config native source template hash is missing or stale")
    if config.get("native_renderer_authorization") != authorization:
        raise ValueError("run config native authorization differs from executable YAML")

    registration_hash = template_hash
    if registration_actions_path is not None:
        registration_path = Path(registration_actions_path).resolve()
        if not registration_path.is_file():
            raise ValueError(f"registered source template is missing: {registration_path}")
        registration_hash = sha256_file(registration_path)
        if registration_hash != template_hash:
            raise ValueError("registered source template hash differs from authorization")
        registered_copy = _load_yaml_mapping(
            registration_path, label="registered source template"
        )
        if _native_design_body(registered_copy) != _native_design_body(reviewed_template):
            raise ValueError("registered source template differs from reviewed Git bytes")
    return {
        "executable_actions_sha256": executable_hash,
        "source_template_sha256": registration_hash,
    }


def load_jsonl(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _unique_by_id(rows: Iterable[Dict[str, Any]], label: str) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("id", ""))
        if not row_id or row_id in result:
            raise ValueError(f"{label} contains an empty or duplicate id {row_id!r}")
        result[row_id] = row
    return result


def _finite_values(record: Dict[str, Any], key: str) -> List[float]:
    values = record.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"renderer diagnostic {key!r} must be a non-empty list")
    flattened = []

    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if isinstance(value, bool):
            raise ValueError(f"renderer diagnostic {key!r} contains a boolean")
        try:
            flattened.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"renderer diagnostic {key!r} contains a non-numeric value"
            ) from exc

    visit(values)
    converted = flattened
    if not all(math.isfinite(value) for value in converted):
        raise ValueError(f"renderer diagnostic {key!r} contains non-finite values")
    return converted


def _registered_renderer_provider(
    defaults: Dict[str, Any], action: Dict[str, Any], *, action_id: str
) -> Dict[str, Any]:
    provider = dict(defaults)
    for mapping, label in (
        (defaults, "defaults"),
        (action.get("provider"), "provider"),
        (action, "action"),
    ):
        if not isinstance(mapping, dict):
            continue
        for key in (
            "implementation",
            "provider_id",
            "provider_provenance_id",
            "provenance_id",
            "requested_bases",
            "required_hook_names",
            "scheduler_mapping",
            "basis_normalization",
        ):
            if key in mapping and mapping[key] is None:
                raise ValueError(f"{action_id}: {label}.{key} cannot be null")
    def _provider_ids(mapping: Any, label: str) -> list[str]:
        if not isinstance(mapping, dict):
            return []
        values = []
        for key in ("provider_id", "provider_provenance_id", "provenance_id"):
            if key not in mapping:
                continue
            value = mapping[key]
            if value is None or not str(value):
                raise ValueError(
                    f"{action_id}: {label}.{key} must be a non-empty string"
                )
            values.append(str(value))
        if values and any(value != values[0] for value in values[1:]):
            raise ValueError(
                f"{action_id}: {label} provider provenance id fields disagree"
            )
        return values

    action_provider_id_values = _provider_ids(action, "action")
    if action.get("provider") is None and "provider" in action:
        raise ValueError(f"{action_id}: registered provider cannot be null")
    overrides = action.get("provider", {}) or {}
    if not isinstance(overrides, dict):
        raise ValueError(f"{action_id}: registered provider must be a mapping")
    nested_provider_id_values = _provider_ids(overrides, "provider")
    default_provider_id_values = _provider_ids(defaults, "defaults")
    provider.update(overrides)
    # Registration manifests may put the lazy mechanism contract directly on
    # each action so heterogeneous basis subsets can share one defaults map.
    for key in (
        "implementation",
        "provider_id",
        "provider_provenance_id",
        "provenance_id",
        "requested_bases",
        "required_hook_names",
        "scheduler_mapping",
        "basis_normalization",
    ):
        if key in action:
            provider[key] = action[key]
    explicit_contract = any(
        key in provider
        for key in (
            "implementation",
            "provider_id",
            "provider_provenance_id",
            "provenance_id",
            "requested_bases",
            "required_hook_names",
        )
    )
    implementation = normalize_latent_structure_provider_implementation(
        provider.get("implementation", LAZY_LATENT_STRUCTURE_PROVIDER_ID)
    )
    if implementation not in {
        LAZY_LATENT_STRUCTURE_PROVIDER_ID,
        "structural_unet_basis_v1",
    }:
        raise ValueError(f"{action_id}: registered provider implementation is invalid")
    mapping = str(provider.get("scheduler_mapping", "legacy_unit"))
    if mapping not in LATENT_RENDERER_SCHEDULER_MAPPINGS:
        raise ValueError(f"{action_id}: registered scheduler_mapping is invalid")
    normalization = str(
        provider.get("basis_normalization", "legacy_l2_to_rms")
    )
    if normalization not in LATENT_RENDERER_BASIS_NORMALIZATIONS:
        raise ValueError(f"{action_id}: registered basis_normalization is invalid")
    requested_bases = _normalize_requested_bases(provider.get("requested_bases"))
    feature_block = str(provider.get("feature_block", "up_blocks.0"))
    semantic_layer = provider.get(
        "semantic_layer",
        "up_blocks.0.attentions.0.transformer_blocks.0.attn1",
    )
    semantic_layer = None if semantic_layer is None else str(semantic_layer)
    expected_hooks = _required_hook_names(
        requested_bases,
        semantic_layer=semantic_layer,
        feature_block=feature_block,
    )
    raw_hooks = provider.get("required_hook_names")
    if raw_hooks is None:
        required_hooks = expected_hooks
    elif isinstance(raw_hooks, list):
        required_hooks = [str(value) for value in raw_hooks]
        if required_hooks != expected_hooks:
            raise ValueError(f"{action_id}: registered required_hook_names drifted")
    else:
        raise ValueError(f"{action_id}: registered required_hook_names must be a list")
    if (
        implementation == "structural_unet_basis_v1"
        and tuple(requested_bases) != tuple(LAZY_LATENT_STRUCTURE_BASIS_NAMES)
    ):
        raise ValueError(
            f"{action_id}: structural_unet_basis_v1 requires all six canonical bases"
        )
    default_provider_id = (
        LAZY_LATENT_STRUCTURE_PROVIDER_ID
        if implementation == LAZY_LATENT_STRUCTURE_PROVIDER_ID
        else "structural_unet_basis_v1"
    )
    provider_id_values = (
        action_provider_id_values
        or nested_provider_id_values
        or default_provider_id_values
    )
    provider_id = provider_id_values[0] if provider_id_values else default_provider_id
    provider["feature_block"] = feature_block
    provider["semantic_layer"] = semantic_layer
    provider["provider_id"] = provider_id
    provider["provider_provenance_id"] = provider_id
    provider.pop("provenance_id", None)
    provider["implementation"] = implementation
    provider["requested_bases"] = requested_bases
    provider["required_hook_names"] = required_hooks
    provider["scheduler_mapping"] = mapping
    provider["basis_normalization"] = normalization
    provider["_strict_contract"] = bool(explicit_contract)
    return provider


def _provider_field(provider: Dict[str, Any], key: str) -> Any:
    """Resolve fields that were implicit in legacy LR-1 run configs."""
    if key in provider:
        value = provider[key]
    elif key == "scheduler_mapping":
        value = "legacy_unit"
    elif key == "basis_normalization":
        value = "legacy_l2_to_rms"
    elif key == "implementation":
        value = LAZY_LATENT_STRUCTURE_PROVIDER_ID
    elif key == "requested_bases":
        value = list(LAZY_LATENT_STRUCTURE_BASIS_NAMES)
    elif key in {"provider_id", "provider_provenance_id"}:
        implementation = str(
            provider.get("implementation", LAZY_LATENT_STRUCTURE_PROVIDER_ID)
        )
        value = provider.get("provider_provenance_id") or provider.get("provenance_id") or (
            "structural_unet_basis_v1"
            if implementation in {"structural", "structural_unet", "structural_unet_basis_v1"}
            else LAZY_LATENT_STRUCTURE_PROVIDER_ID
        )
    elif key == "required_hook_names":
        requested = _normalize_requested_bases(provider.get("requested_bases"))
        value = _required_hook_names(
            requested,
            semantic_layer=provider.get(
                "semantic_layer",
                "up_blocks.0.attentions.0.transformer_blocks.0.attn1",
            ),
            feature_block=provider.get("feature_block", "up_blocks.0"),
        )
    else:
        value = None
    if key == "implementation":
        return normalize_latent_structure_provider_implementation(value)
    if key == "requested_bases":
        return _normalize_requested_bases(value)
    if key == "required_hook_names" and value is not None:
        return [str(item) for item in value]
    if key in {"provider_id", "provider_provenance_id"}:
        return str(value)
    return value


def _normalize_requested_bases(value: Any) -> list[str]:
    """Normalize aliases using the same canonical action order as generation."""
    try:
        return list(_normalize_bases(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc


def _required_hook_names(
    requested_bases: list[str], *, semantic_layer: Any, feature_block: Any
) -> list[str]:
    hooks = []
    if "freeu" in requested_bases:
        hooks.append(f"{feature_block}_backbone_skip")
    if "semantic" in requested_bases and semantic_layer is not None:
        hooks.append(f"{semantic_layer}_qk")
    return hooks


def _audit_provider_diagnostics(
    provider: Dict[str, Any],
    expected_provider: Dict[str, Any] | None,
    *,
    expected_id: str,
    strict: bool,
) -> None:
    """Check the per-step lazy-provider contract, including unused slots."""
    if expected_provider is None or not strict:
        return
    expected_fields = {
        "implementation": expected_provider["implementation"],
        "provider_id": expected_provider["provider_id"],
        "provider_provenance_id": expected_provider["provider_provenance_id"],
        "requested_bases": expected_provider["requested_bases"],
        "constructed_bases": expected_provider["requested_bases"],
        "registered_hook_names": expected_provider["required_hook_names"],
        "required_hook_names": expected_provider["required_hook_names"],
        "scheduler_mapping": expected_provider["scheduler_mapping"],
        "basis_normalization": expected_provider["basis_normalization"],
    }
    for key, expected in expected_fields.items():
        if key not in provider:
            raise ValueError(f"{expected_id}: native provider field {key!r} is missing")
        observed = provider.get(key)
        if observed != expected:
            raise ValueError(f"{expected_id}: native provider field {key!r} drifted")

    basis_rms = provider.get("basis_rms")
    if (
        not isinstance(basis_rms, list)
        or len(basis_rms) != 1
        or not isinstance(basis_rms[0], list)
        or len(basis_rms[0]) != len(LAZY_LATENT_STRUCTURE_BASIS_NAMES)
    ):
        raise ValueError(f"{expected_id}: native basis_rms must have shape (1, 6)")
    values = [float(value) for value in basis_rms[0]]
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError(f"{expected_id}: native basis_rms is invalid")
    requested = set(expected_provider["requested_bases"])
    for index, name in enumerate(LAZY_LATENT_STRUCTURE_BASIS_NAMES):
        if name not in requested and values[index] != 0.0:
            raise ValueError(f"{expected_id}: unrequested basis {name!r} is non-zero")


def _validate_native_action_contract(
    source: Dict[str, Any], source_actions: List[Dict[str, Any]]
) -> tuple[str, str]:
    """Validate the role-level primary and zero-identity rules."""
    if source.get("schema") != NATIVE_RENDERER_SCHEMA:
        raise ValueError("native renderer source uses an executable schema different from v1")
    by_id = {str(action.get("id")): action for action in source_actions}
    analysis = source.get("analysis")
    if not isinstance(analysis, dict):
        raise ValueError("native renderer source lacks analysis metadata")
    baseline_id = str(analysis.get("baseline_action", ""))
    identity_id = str(analysis.get("identity_control", ""))
    if not baseline_id or not identity_id or baseline_id == identity_id:
        raise ValueError(
            "native renderer source must register distinct baseline and identity actions"
        )
    if baseline_id not in by_id or identity_id not in by_id:
        raise ValueError("native renderer baseline/identity action is not registered")
    if by_id[baseline_id].get("type") != "none":
        raise ValueError("native renderer baseline action must be a scheduler no-op")
    identity = by_id[identity_id]
    if identity.get("type") != "latent_renderer_fixed":
        raise ValueError("native renderer identity control must be latent_renderer_fixed")
    coefficients = identity.get("coefficients")
    if (
        not isinstance(coefficients, list)
        or len(coefficients) != len(LAZY_LATENT_STRUCTURE_BASIS_NAMES)
        or any(float(value) != 0.0 for value in coefficients)
    ):
        raise ValueError("native renderer identity control must have all-zero coefficients")
    if identity.get("requested_bases") not in ([], None):
        raise ValueError("native renderer identity control must request no bases")
    if identity.get("required_hook_names") not in ([], None):
        raise ValueError("native renderer identity control must register no hooks")

    primary_ids = [
        str(action.get("id"))
        for action in source_actions
        if action.get("role") == "non_attention_primary"
    ]
    family = analysis.get("primary_holm_family")
    registered_primary = family.get("actions") if isinstance(family, dict) else None
    if (
        not isinstance(registered_primary, list)
        or len(primary_ids) != 4
        or set(registered_primary) != set(primary_ids)
    ):
        raise ValueError("native renderer primary Holm family differs from action roles")

    allowed = {"spectral_low", "spectral_mid", "spectral_high", "laplacian"}
    canonical_index = {
        name: index for index, name in enumerate(LAZY_LATENT_STRUCTURE_BASIS_NAMES)
    }
    for action in source_actions:
        if action.get("role") != "non_attention_primary":
            continue
        action_id = str(action.get("id"))
        requested = action.get("requested_bases")
        if isinstance(requested, str):
            requested = [requested]
        if not isinstance(requested, list) or len(requested) != 1:
            raise ValueError(
                f"{action_id}: non-attention primary must request exactly one basis"
            )
        try:
            normalized = list(_normalize_bases(requested))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{action_id}: invalid primary requested basis") from exc
        if len(normalized) != 1 or normalized[0] not in allowed:
            raise ValueError(
                f"{action_id}: non-attention primary must be one spectral/Laplacian basis"
            )
        if action.get("required_hook_names") != []:
            raise ValueError(f"{action_id}: non-attention primary must register no hooks")
        coefficients = action.get("coefficients")
        if not isinstance(coefficients, list) or len(coefficients) != len(
            LAZY_LATENT_STRUCTURE_BASIS_NAMES
        ):
            raise ValueError(f"{action_id}: primary coefficient vector is malformed")
        nonzero = [
            index for index, value in enumerate(coefficients) if float(value) != 0.0
        ]
        expected_index = canonical_index[normalized[0]]
        if nonzero != [expected_index]:
            raise ValueError(
                f"{action_id}: primary coefficient must select only its requested basis"
            )
        if action.get("selection_eligible") is not True:
            raise ValueError(f"{action_id}: primary action must be selection-eligible")
    return baseline_id, identity_id


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_native_schedule(
    record: Dict[str, Any],
    ledger: list[Dict[str, Any]],
    *,
    expected_steps: int,
    expected_id: str,
    strict: bool,
) -> bool:
    """Validate the frozen Euler schedule and its post-run ledger bindings."""
    schedule_keys = {
        "scheduler_schedule_sha256",
        "scheduler_timesteps",
        "scheduler_sigmas",
    }
    present = schedule_keys & set(record)
    if not present:
        if strict:
            raise ValueError(f"{expected_id}: native scheduler schedule provenance is missing")
        return False
    if present != schedule_keys:
        raise ValueError(f"{expected_id}: native scheduler schedule provenance is incomplete")
    raw_timesteps = record.get("scheduler_timesteps")
    raw_sigmas = record.get("scheduler_sigmas")
    if not isinstance(raw_timesteps, list) or not isinstance(raw_sigmas, list):
        raise ValueError(f"{expected_id}: native scheduler schedule arrays are malformed")
    try:
        timesteps = [float(value) for value in raw_timesteps]
        sigmas = [float(value) for value in raw_sigmas]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{expected_id}: native scheduler schedule is non-numeric") from exc
    if len(timesteps) != expected_steps or len(sigmas) != expected_steps + 1:
        raise ValueError(f"{expected_id}: native scheduler schedule length is invalid")
    if not all(math.isfinite(value) for value in timesteps + sigmas):
        raise ValueError(f"{expected_id}: native scheduler schedule is non-finite")
    if any(sigmas[index] <= 0 for index in range(expected_steps)) or sigmas[-1] < 0:
        raise ValueError(f"{expected_id}: native scheduler sigma schedule is invalid")
    payload = {"timesteps": timesteps, "sigmas": sigmas}
    if record.get("scheduler_schedule_sha256") != _json_sha256(payload):
        raise ValueError(f"{expected_id}: native scheduler schedule hash is invalid")

    for step_index, step in enumerate(ledger):
        if not math.isclose(
            float(step["timestep"]), timesteps[step_index], rel_tol=1e-6, abs_tol=1e-5
        ):
            raise ValueError(f"{expected_id}: native timestep schedule drifted")
        if not math.isclose(
            float(step["sigma_from"]), sigmas[step_index], rel_tol=1e-6, abs_tol=1e-6
        ) or not math.isclose(
            float(step["sigma_to"]), sigmas[step_index + 1], rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError(f"{expected_id}: native sigma schedule drifted")

    # These fields are emitted by current registered runs.  Keep them optional
    # for pre-contract synthetic fixtures, while validating them whenever any
    # of the config-hash ledger is present.
    config_hash_keys = {
        "scheduler_config_sha256_v2",
        "active_scheduler_config_sha256_v2",
        "scheduler_order",
        "scheduler_solver_order",
        "scheduler_construction_init_noise_sigma",
        "scheduler_effective_init_noise_sigma",
    }
    if strict and not config_hash_keys.issubset(record):
        raise ValueError(f"{expected_id}: native scheduler config provenance is incomplete")
    if config_hash_keys & set(record):
        if not config_hash_keys.issubset(record):
            raise ValueError(f"{expected_id}: native scheduler config provenance is incomplete")
        for key in ("scheduler_config_sha256_v2", "active_scheduler_config_sha256_v2"):
            value = record.get(key)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{expected_id}: native scheduler config hash is invalid")
        if record["scheduler_config_sha256_v2"] != record["active_scheduler_config_sha256_v2"]:
            raise ValueError(f"{expected_id}: native scheduler config hash drifted")
        if record.get("scheduler_order") != 1 or record.get("scheduler_solver_order") is not None:
            raise ValueError(f"{expected_id}: native scheduler order contract drifted")
        for key in (
            "scheduler_construction_init_noise_sigma",
            "scheduler_effective_init_noise_sigma",
        ):
            value = float(record[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{expected_id}: native scheduler initial sigma is invalid")
    return True


def _native_schedule_signature(record: Dict[str, Any], *, expected_id: str) -> str | None:
    """Return the complete schedule identity for cross-record consistency checks."""
    keys = (
        "scheduler_schedule_sha256",
        "scheduler_timesteps",
        "scheduler_sigmas",
        "scheduler_config_sha256_v2",
        "active_scheduler_config_sha256_v2",
        "scheduler_order",
        "scheduler_solver_order",
        "scheduler_construction_init_noise_sigma",
        "scheduler_effective_init_noise_sigma",
    )
    present = [key in record for key in keys]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(f"{expected_id}: native scheduler provenance is incomplete")
    payload = {key: record.get(key) for key in keys}
    try:
        return _json_sha256(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{expected_id}: native scheduler provenance is not canonical") from exc


def _audit_native_step_ledger(
    record: Dict[str, Any],
    *,
    expected_steps: int,
    bound: float,
    expected_id: str,
    expected_provider: Dict[str, Any] | None = None,
    strict_schedule: bool = False,
    strict_contract: bool = False,
) -> tuple[float, float, float]:
    calls = record.get("unet_calls_per_step")
    if calls != [1] * expected_steps or record.get("extra_unet_calls") != 0:
        raise ValueError(f"{expected_id}: native renderer violates one-UNet-call NFE")
    if record.get("scheduler_name") != "EulerDiscreteScheduler":
        raise ValueError(f"{expected_id}: native renderer used a non-Euler scheduler")
    ledger = record.get("latent_renderer_step_diagnostics")
    if not isinstance(ledger, list) or len(ledger) != expected_steps:
        raise ValueError(
            f"{expected_id}: native renderer ledger must contain {expected_steps} steps"
        )

    _validate_native_schedule(
        record,
        ledger,
        expected_steps=expected_steps,
        expected_id=expected_id,
        strict=strict_schedule,
    )

    max_ratio = 0.0
    max_mean_error = 0.0
    max_variance_error = 0.0
    for step_index, step in enumerate(ledger):
        if not isinstance(step, dict):
            raise ValueError(f"{expected_id}: native step {step_index} is not a mapping")
        if step.get("step_index") != step_index:
            raise ValueError(f"{expected_id}: native renderer step indices drifted")
        if step.get("scheduler_step_index") != step_index:
            raise ValueError(f"{expected_id}: native scheduler step indices drifted")
        try:
            timestep = float(step["timestep"])
            sigma_from = float(step["sigma_from"])
            sigma_to = float(step["sigma_to"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{expected_id}: native step {step_index} lacks sigma provenance"
            ) from exc
        if not all(math.isfinite(value) for value in (timestep, sigma_from, sigma_to)):
            raise ValueError(f"{expected_id}: native step provenance is non-finite")
        if sigma_from <= 0 or sigma_to < 0 or sigma_to > sigma_from:
            raise ValueError(f"{expected_id}: native step sigma order is invalid")
        if step.get("prediction_type") not in {
            "epsilon",
            "sample",
            "v_prediction",
        }:
            raise ValueError(f"{expected_id}: native prediction type is invalid")
        gain = _finite_values(step, "clean_update_gain")
        applied_ratio = _finite_values(step, "applied_update_ratio")
        mean_error = _finite_values(step, "mean_error")
        variance_error = _finite_values(step, "variance_error")
        if any(len(values) != 1 for values in (gain, applied_ratio, mean_error, variance_error)):
            raise ValueError(f"{expected_id}: native diagnostics must have batch size one")
        expected_gain = 1.0 - sigma_to / sigma_from
        if not 0 < gain[0] <= 1 or not math.isclose(
            gain[0], expected_gain, rel_tol=1e-6, abs_tol=1e-7
        ):
            raise ValueError(f"{expected_id}: native clean-update gain is invalid")
        if applied_ratio[0] < 0 or applied_ratio[0] > bound + 1e-6:
            raise ValueError(f"{expected_id}: native applied trust bound is violated")
        if abs(mean_error[0]) > 1e-4:
            raise ValueError(f"{expected_id}: native mean preservation is violated")
        if abs(variance_error[0]) > 1e-3:
            raise ValueError(f"{expected_id}: native variance preservation is violated")
        provider = step.get("provider_diagnostics")
        if not isinstance(provider, dict):
            raise ValueError(f"{expected_id}: native provider ledger is missing")
        _finite_values(provider, "semantic_entropy")
        if not (
            strict_contract
            or (expected_provider and expected_provider.get("_strict_contract"))
        ):
            _finite_values(provider, "basis_rms")
        _audit_provider_diagnostics(
            provider,
            expected_provider,
            expected_id=expected_id,
            strict=bool(strict_contract or (expected_provider and expected_provider.get("_strict_contract"))),
        )
        max_ratio = max(max_ratio, applied_ratio[0])
        max_mean_error = max(max_mean_error, abs(mean_error[0]))
        max_variance_error = max(max_variance_error, abs(variance_error[0]))

    last_diagnostics = record.get("latent_renderer_diagnostics")
    for key, value in last_diagnostics.items():
        if ledger[-1].get(key) != value:
            raise ValueError(
                f"{expected_id}: last renderer diagnostics differ from native ledger"
            )
    if ledger[-1].get("provider_diagnostics") != record.get(
        "latent_renderer_provider_diagnostics"
    ):
        raise ValueError(
            f"{expected_id}: last provider diagnostics differ from native ledger"
        )
    return max_ratio, max_mean_error, max_variance_error


def audit_run(
    run_dir: str | os.PathLike[str],
    prompts_path: str | os.PathLike[str],
    source_actions_path: str | os.PathLike[str],
    *,
    split_role: str,
    required_score_keys: Iterable[str] = DEFAULT_SCORE_KEYS,
    verify_images: bool = True,
    require_distinct_actions: bool = True,
    registration_actions_path: str | os.PathLike[str] | None = None,
) -> Dict[str, Any]:
    """Reject incomplete, unpaired, malformed, or numerically unsafe runs."""
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    manifest_path = run_dir / "manifest.jsonl"
    scores_path = run_dir / "scores.jsonl"
    for path in (config_path, manifest_path, scores_path):
        if not path.is_file():
            raise ValueError(f"required run file is missing: {path}")

    with config_path.open() as handle:
        config = json.load(handle)
    source = _load_yaml_mapping(source_actions_path, label="source actions")
    prompts = pd.read_csv(prompts_path)
    manifest = load_jsonl(manifest_path)
    scores = load_jsonl(scores_path)
    strict_registered_contract = bool(config.get("native_renderer_registered"))
    run_contract_sha256 = None
    if "run_contract" in config or "run_contract_sha256" in config:
        try:
            run_contract_sha256 = validate_run_contract(config)
        except ValueError as exc:
            raise ValueError(f"run config contract is invalid: {exc}") from exc
    elif strict_registered_contract:
        raise ValueError("registered native renderer run lacks a run contract")
    manifest_by_id = _unique_by_id(manifest, "manifest")
    scores_by_id = _unique_by_id(scores, "scores")

    split_seeds = source.get("split_seeds")
    if not isinstance(split_seeds, dict) or split_role not in split_seeds:
        raise ValueError(f"source actions do not register split role {split_role!r}")
    seeds = [int(value) for value in split_seeds[split_role]]
    if [int(value) for value in config.get("seeds", [])] != seeds:
        raise ValueError("run config seeds differ from the registered split role")
    if config.get("split_role") not in (None, split_role):
        raise ValueError("run config split_role differs from the requested audit role")
    recorded_prompt_hash = config.get("prompts_sha256")
    if recorded_prompt_hash is not None and recorded_prompt_hash != sha256_file(prompts_path):
        raise ValueError("run config prompt hash differs from the audited prompt CSV")
    recorded_action_hash = config.get("actions_sha256")
    if (
        recorded_action_hash is not None
        and recorded_action_hash != sha256_file(source_actions_path)
    ):
        raise ValueError("run config executable action hash differs from the audited source YAML")

    expected_split = SPLIT_NAMES.get(split_role)
    if not {"index", "TEXT"}.issubset(prompts.columns):
        raise ValueError("prompt CSV must contain index and TEXT columns")
    if expected_split and (
        "split" not in prompts or set(prompts["split"].astype(str)) != {expected_split}
    ):
        raise ValueError(f"prompt CSV is not explicitly marked split={expected_split}")
    if prompts["index"].duplicated().any():
        raise ValueError("prompt CSV contains duplicate indices")
    prompt_text = {
        int(row["index"]): str(row["TEXT"]) for _, row in prompts.iterrows()
    }

    source_actions = source.get("actions")
    configured_actions = config.get("actions")
    if not isinstance(source_actions, list) or not isinstance(configured_actions, list):
        raise ValueError("source and run config must both contain action lists")
    action_ids = [str(action.get("id", "")) for action in source_actions]
    configured_ids = [str(action.get("id", "")) for action in configured_actions]
    if not all(action_ids) or len(action_ids) != len(set(action_ids)):
        raise ValueError("source actions contain empty or duplicate ids")
    if configured_ids != action_ids:
        raise ValueError("run action order/set differs from source actions")
    source_by_id = {str(action["id"]): action for action in source_actions}
    if "latent_renderer_provider" in source:
        raw_provider_defaults = source.get("latent_renderer_provider")
        if raw_provider_defaults is None:
            raise ValueError("source latent_renderer_provider cannot be null")
    else:
        raw_provider_defaults = source.get("required_provider", {})
    if raw_provider_defaults is None:
        raise ValueError("source required_provider cannot be null")
    provider_defaults = dict(raw_provider_defaults)
    if not isinstance(provider_defaults, dict):
        raise ValueError("source required_provider must be a mapping")
    renderer_providers: Dict[str, Dict[str, Any]] = {}
    for registered, configured in zip(source_actions, configured_actions):
        action_id = str(registered["id"])
        if configured.get("type") != registered.get("type"):
            raise ValueError(f"{action_id}: run action type differs from source")
        if registered.get("type") == "latent_renderer_fixed":
            expected = [float(value) for value in registered.get("coefficients", [])]
            observed = [float(value) for value in configured.get("coefficients", [])]
            if observed != expected:
                raise ValueError(f"{action_id}: run coefficients differ from source")
            expected_provider = _registered_renderer_provider(
                provider_defaults, registered, action_id=action_id
            )
            renderer_providers[action_id] = expected_provider
            configured_provider = configured.get("latent_renderer_provider", {}) or {}
            if not isinstance(configured_provider, dict):
                raise ValueError(f"{action_id}: run provider must be a mapping")
            for key in (
                "implementation",
                "provider_id",
                "provider_provenance_id",
                "requested_bases",
                "required_hook_names",
                "scheduler_mapping",
                "basis_normalization",
            ):
                if _provider_field(configured_provider, key) != _provider_field(
                    expected_provider, key
                ):
                    raise ValueError(
                        f"{action_id}: run provider field {key!r} differs from source"
                    )
    native_renderer_run = any(
        provider["scheduler_mapping"] == "euler_clean_endpoint"
        for provider in renderer_providers.values()
    )
    native_v1_contract = (
        source.get("schema") == NATIVE_RENDERER_SCHEMA
        or config.get("action_schema") == NATIVE_RENDERER_SCHEMA
    )
    if native_v1_contract and (
        not native_renderer_run
        or not strict_registered_contract
        or config.get("action_schema") != NATIVE_RENDERER_SCHEMA
    ):
        raise ValueError(
            "scheduler-native renderer actions require the strict v1 registered contract"
        )
    identity_pair: tuple[str, str] | None = None
    native_registration_provenance: Dict[str, str] | None = None
    if native_renderer_run and native_v1_contract and strict_registered_contract:
        native_registration_provenance = _validate_native_registration(
            source_actions_path,
            source,
            config,
            registration_actions_path=registration_actions_path,
        )
        identity_pair = _validate_native_action_contract(source, source_actions)
    if native_renderer_run and native_v1_contract and strict_registered_contract:
        registration_source = source.get("registration_source")
        if not isinstance(registration_source, dict) or set(registration_source) != {
            "path",
            "sha256",
        }:
            raise ValueError(
                "registered native renderer source lacks registration_source metadata"
            )
        registration_path = (ROOT / str(registration_source["path"])).resolve()
        if os.path.commonpath((str(ROOT.resolve()), str(registration_path))) != str(
            ROOT.resolve()
        ) or not registration_path.is_file():
            raise ValueError("registered native renderer registration_source path is invalid")
        if sha256_file(registration_path) != registration_source["sha256"]:
            raise ValueError(
                "registered native renderer frozen registration hash differs"
            )
    if native_renderer_run and native_v1_contract and strict_registered_contract:
        source_sampling = source.get("sampling")
        registered_sampling = config.get("registered_sampling")
        if not isinstance(source_sampling, dict) or not isinstance(
            registered_sampling, dict
        ):
            raise ValueError(
                "registered native renderer run lacks sampling provenance"
            )
        for key in (
            "model",
            "model_revision",
            "scheduler",
            "extra_unet_calls",
            "scheduler_churn",
        ):
            if source_sampling.get(key) != registered_sampling.get(key):
                raise ValueError(
                    f"registered native sampling field {key!r} differs from source"
                )
        if source_sampling.get("scheduler") != "EulerDiscreteScheduler":
            raise ValueError(
                "registered native renderer run must use EulerDiscreteScheduler"
            )
        if int(source_sampling.get("extra_unet_calls", -1)) != 0:
            raise ValueError(
                "registered native renderer run requires zero extra U-Net calls"
            )
        if float(source_sampling.get("scheduler_churn", 0.0)) != 0.0:
            raise ValueError(
                "registered native renderer run requires zero scheduler churn"
            )
    registered_schedule_hash = None
    registered_scheduler_config_hashes = {}
    for registration in (source, config):
        sampling = registration.get("sampling")
        if not isinstance(sampling, dict):
            sampling = registration.get("registered_sampling")
        if not isinstance(sampling, dict):
            continue
        for key in ("scheduler_schedule_sha256", "schedule_sha256"):
            candidate = sampling.get(key)
            if candidate is None:
                continue
            candidate = str(candidate)
            if registered_schedule_hash is not None and candidate != registered_schedule_hash:
                raise ValueError("registered native scheduler schedule hashes disagree")
            registered_schedule_hash = candidate
        for key in ("scheduler_config_sha256_v2", "active_scheduler_config_sha256_v2"):
            candidate = sampling.get(key)
            if candidate is None:
                continue
            candidate = str(candidate)
            prior = registered_scheduler_config_hashes.get(key)
            if prior is not None and candidate != prior:
                raise ValueError(f"registered native scheduler config hash {key!r} disagrees")
            registered_scheduler_config_hashes[key] = candidate
    scorer_provenance_sha256 = None
    registered_scorer_contract = None
    registered_scorer_sha256 = None
    for registration in (source, config):
        try:
            candidate_contract, candidate_hash = registered_scorer_provenance_contract(
                registration
            )
        except ValueError as exc:
            raise ValueError(f"registered scorer provenance is invalid: {exc}") from exc
        if candidate_contract is not None:
            if (
                registered_scorer_contract is not None
                and candidate_contract != registered_scorer_contract
            ):
                raise ValueError("registered scorer provenance contracts disagree")
            registered_scorer_contract = candidate_contract
        if candidate_hash is not None:
            if registered_scorer_sha256 is not None and candidate_hash != registered_scorer_sha256:
                raise ValueError("registered scorer provenance hashes disagree")
            registered_scorer_sha256 = candidate_hash
    if native_renderer_run:
        scorer_provenance_sha256 = validate_hardened_score_rows(
            scores,
            required_schema=SCORER_PROVENANCE_SCHEMA,
            expected_sha256=registered_scorer_sha256
            if registered_scorer_sha256 is not None
            else None,
            expected_contract=registered_scorer_contract,
        )
        if strict_registered_contract:
            if registered_scorer_sha256 is None:
                raise ValueError(
                    "registered native renderer run lacks scorer provenance registration"
                )
            if not isinstance(run_contract_sha256, str):
                raise ValueError("registered native renderer run lacks a contract hash")
            for score in scores:
                if score.get("run_contract_sha256") != run_contract_sha256:
                    raise ValueError(
                        f"{score.get('id')}: scorer row is not bound to the run contract"
                    )
    expected_cutoffs = [float(value) for value in source.get("frequency_band_cutoffs", [])]
    observed_cutoffs = [float(value) for value in config.get("frequency_band_cutoffs", [])]
    if observed_cutoffs != expected_cutoffs:
        raise ValueError("run frequency-band cutoffs differ from source")

    expected_design = {
        (prompt_index, seed, action_id)
        for prompt_index in prompt_text
        for seed in seeds
        for action_id in action_ids
    }
    records_by_design: Dict[tuple, Dict[str, Any]] = {}
    for record in manifest:
        key = (
            int(record.get("prompt_index", -1)),
            int(record.get("seed", -1)),
            str(record.get("action_id", "")),
        )
        if key in records_by_design:
            raise ValueError(f"manifest duplicates design cell {key}")
        records_by_design[key] = record
    observed_design = set(records_by_design)
    if observed_design != expected_design:
        missing = len(expected_design - observed_design)
        extra = len(observed_design - expected_design)
        raise ValueError(f"incomplete design: {missing} missing and {extra} extra cells")
    if set(scores_by_id) != set(manifest_by_id):
        raise ValueError("score ids do not exactly match the complete manifest")

    required_score_keys = tuple(required_score_keys)
    max_update_ratio = 0.0
    max_mean_error = 0.0
    max_variance_error = 0.0
    image_hashes: Dict[tuple, str] = {}
    block_devices: Dict[tuple, set] = {}
    block_ranks: Dict[tuple, set] = {}
    native_schedule_signature: str | None = None
    run_root = run_dir.resolve()
    resolution = int(config.get("resolution", 0))
    for design, record in records_by_design.items():
        prompt_index, seed, action_id = design
        expected_id = f"p{prompt_index}_seed{seed}_a{action_id}"
        if record.get("id") != expected_id:
            raise ValueError(f"record id differs from design cell: {record.get('id')!r}")
        if str(record.get("prompt")) != prompt_text[prompt_index]:
            raise ValueError(f"prompt text drift at prompt index {prompt_index}")
        if record.get("action_type") != source_by_id[action_id].get("type"):
            raise ValueError(f"{expected_id}: record action type differs from source")
        if strict_registered_contract and record.get("run_contract_sha256") != run_contract_sha256:
            raise ValueError(f"{expected_id}: sidecar is not bound to the run contract")
        if strict_registered_contract:
            if record.get("provenance_schema") != "s7_trajectory_provenance_v1":
                raise ValueError(f"{expected_id}: sidecar provenance schema is missing")
            sidecar_action = record.get("action")
            if not isinstance(sidecar_action, dict) or record.get("action_sha256") != _json_sha256(
                sidecar_action
            ):
                raise ValueError(f"{expected_id}: sidecar action hash is invalid")
            configured_action = next(
                action
                for action in configured_actions
                if str(action.get("id")) == action_id
            )
            if sidecar_action != configured_action:
                raise ValueError(f"{expected_id}: sidecar action differs from run config")

        block = (prompt_index, seed)
        block_devices.setdefault(block, set()).add(str(record.get("device", "")))
        block_ranks.setdefault(block, set()).add(int(record.get("execution_rank", -1)))
        image_path = (run_dir / str(record.get("image_path", ""))).resolve()
        if os.path.commonpath((run_root, image_path)) != str(run_root):
            raise ValueError(f"{expected_id}: image path escapes the run directory")
        if not image_path.is_file():
            raise ValueError(f"{expected_id}: image is missing")
        image_hashes[design] = sha256_file(image_path)
        if verify_images:
            with Image.open(image_path) as image:
                if image.format != "PNG" or image.mode != "RGB":
                    raise ValueError(f"{expected_id}: expected an RGB PNG")
                if image.size != (resolution, resolution):
                    raise ValueError(f"{expected_id}: image dimensions differ from config")
                image.verify()

        score = scores_by_id[expected_id]
        for key in required_score_keys:
            try:
                value = float(score[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{expected_id}: missing/non-numeric score {key!r}") from exc
            if not math.isfinite(value):
                raise ValueError(f"{expected_id}: non-finite score {key!r}")

        if source_by_id[action_id].get("type") == "latent_renderer_fixed":
            diagnostics = record.get("latent_renderer_diagnostics")
            if not isinstance(diagnostics, dict):
                raise ValueError(f"{expected_id}: renderer diagnostics are missing")
            bound = float(record.get("action", {}).get("max_update_ratio", 0.05))
            if not math.isfinite(bound) or bound < 0:
                raise ValueError(f"{expected_id}: renderer trust bound is invalid")
            provider_diagnostics = record.get("latent_renderer_provider_diagnostics")
            if not isinstance(provider_diagnostics, dict):
                raise ValueError(f"{expected_id}: provider diagnostics are missing")
            _finite_values(provider_diagnostics, "semantic_entropy")
            expected_provider = renderer_providers[action_id]
            if not (strict_registered_contract or expected_provider.get("_strict_contract")):
                _finite_values(provider_diagnostics, "basis_rms")
            expected_mapping = expected_provider["scheduler_mapping"]
            observed_mapping = record.get("latent_renderer_scheduler_mapping")
            record_provider = record.get("action", {}).get(
                "latent_renderer_provider", {}
            )
            if not isinstance(record_provider, dict):
                raise ValueError(f"{expected_id}: sidecar provider is missing")
            for key in ("scheduler_mapping", "basis_normalization"):
                if _provider_field(record_provider, key) != expected_provider[key]:
                    raise ValueError(
                        f"{expected_id}: sidecar provider field {key!r} drifted"
                    )

            _audit_provider_diagnostics(
                provider_diagnostics,
                expected_provider,
                expected_id=expected_id,
                strict=bool(
                    strict_registered_contract
                    or expected_provider.get("_strict_contract")
                ),
            )

            if expected_mapping == "euler_clean_endpoint":
                if observed_mapping != expected_mapping:
                    raise ValueError(
                        f"{expected_id}: native runtime scheduler mapping drifted"
                    )
                expected_steps = int(config.get("num_inference_steps", 0))
                if expected_steps <= 0 or record.get("num_inference_steps") != expected_steps:
                    raise ValueError(
                        f"{expected_id}: native renderer inference-step contract drifted"
                    )
                ratio, mean_error, variance_error = _audit_native_step_ledger(
                    record,
                    expected_steps=expected_steps,
                    bound=bound,
                    expected_id=expected_id,
                    expected_provider=expected_provider,
                    strict_schedule=bool(config.get("native_renderer_registered")),
                    strict_contract=strict_registered_contract,
                )
                schedule_signature = _native_schedule_signature(
                    record, expected_id=expected_id
                )
                if registered_schedule_hash is not None and record.get(
                    "scheduler_schedule_sha256"
                ) != registered_schedule_hash:
                    raise ValueError(
                        f"{expected_id}: native scheduler schedule differs from registration"
                    )
                for key, expected_hash in registered_scheduler_config_hashes.items():
                    if record.get(key) != expected_hash:
                        raise ValueError(
                            f"{expected_id}: native scheduler config differs from registration"
                        )
                if schedule_signature is not None:
                    if (
                        native_schedule_signature is not None
                        and schedule_signature != native_schedule_signature
                    ):
                        raise ValueError(
                            f"{expected_id}: native scheduler schedule differs across records"
                        )
                    native_schedule_signature = schedule_signature
                max_update_ratio = max(max_update_ratio, ratio)
                max_mean_error = max(max_mean_error, mean_error)
                max_variance_error = max(max_variance_error, variance_error)
            else:
                if observed_mapping not in (None, "legacy_unit"):
                    raise ValueError(
                        f"{expected_id}: legacy runtime scheduler mapping drifted"
                    )
                if record.get("latent_renderer_step_diagnostics") is not None:
                    raise ValueError(
                        f"{expected_id}: legacy renderer has a native step ledger"
                    )
                update_ratio = _finite_values(diagnostics, "update_ratio")
                mean_error = _finite_values(diagnostics, "mean_error")
                variance_error = _finite_values(diagnostics, "variance_error")
                if min(update_ratio) < 0 or max(update_ratio) > bound + 1e-6:
                    raise ValueError(f"{expected_id}: renderer trust bound is violated")
                if max(map(abs, mean_error)) > 1e-4:
                    raise ValueError(
                        f"{expected_id}: renderer mean preservation is violated"
                    )
                if max(map(abs, variance_error)) > 1e-3:
                    raise ValueError(
                        f"{expected_id}: renderer variance preservation is violated"
                    )
                max_update_ratio = max(max_update_ratio, max(update_ratio))
                max_mean_error = max(
                    max_mean_error, max(map(abs, mean_error))
                )
                max_variance_error = max(
                    max_variance_error, max(map(abs, variance_error))
                )
        elif (
            record.get("latent_renderer_diagnostics") is not None
            or record.get("latent_renderer_provider_diagnostics") is not None
            or record.get("latent_renderer_scheduler_mapping") is not None
            or record.get("latent_renderer_step_diagnostics") is not None
        ):
            raise ValueError(f"{expected_id}: non-renderer action has stale diagnostics")

    expected_ranks = set(range(len(action_ids)))
    identity_pairs = set()
    baseline_ids = {
        str(action.get("id"))
        for action in source_actions
        if action.get("role") == "nominal_scheduler_baseline"
        or action.get("type") == "none"
    }
    for action in source_actions:
        if action.get("role") != "implementation_identity_control":
            continue
        identity_id = str(action.get("id"))
        if len(baseline_ids) != 1:
            raise ValueError(
                "an implementation identity control requires exactly one nominal baseline"
            )
        coefficients = action.get("coefficients")
        if coefficients is not None and any(float(value) != 0.0 for value in coefficients):
            raise ValueError(
                f"{identity_id}: implementation identity control must have zero coefficients"
            )
        identity_pairs.add(frozenset((identity_id, next(iter(baseline_ids)))))
    if identity_pair is not None:
        # The strict native contract names exactly one pair.  Do not let a
        # stale or extra implementation-identity role silently widen it.
        identity_pairs = {frozenset(identity_pair)}
    observed_identity_pairs = set()
    for block, devices in block_devices.items():
        if len(devices) != 1 or "" in devices:
            raise ValueError(f"prompt/seed block {block} spans or lacks devices")
        if block_ranks[block] != expected_ranks:
            raise ValueError(f"prompt/seed block {block} has invalid execution ranks")
        hashes_by_id = {
            action_id: image_hashes[(block[0], block[1], action_id)]
            for action_id in action_ids
        }
        if identity_pair is not None:
            baseline_id, identity_id = identity_pair
            if hashes_by_id[baseline_id] != hashes_by_id[identity_id]:
                raise ValueError(
                    f"prompt/seed block {block} violates zero-identity PNG parity"
                )
            # Exactly this pair may collide.  Any duplicate involving a
            # non-identity action remains a hard failure.
            non_identity_ids = [
                action_id for action_id in action_ids if action_id != identity_id
            ]
            if len({hashes_by_id[action_id] for action_id in non_identity_ids}) != len(
                non_identity_ids
            ):
                raise ValueError(
                    f"prompt/seed block {block} has duplicate non-identity PNG hashes"
                )
            observed_identity_pairs.add(tuple(sorted(identity_pair)))
        elif require_distinct_actions:
            by_hash = {}
            for action_id, image_hash in hashes_by_id.items():
                by_hash.setdefault(image_hash, set()).add(action_id)
            for duplicate_ids in by_hash.values():
                if len(duplicate_ids) == 1:
                    continue
                pair = frozenset(duplicate_ids)
                if pair in identity_pairs and len(duplicate_ids) == 2:
                    observed_identity_pairs.add(tuple(sorted(duplicate_ids)))
                    continue
                raise ValueError(
                    f"prompt/seed block {block} has unexpected identical PNG hashes "
                    f"for actions {sorted(duplicate_ids)}"
                )

    warnings = []
    if not native_renderer_run:
        warnings.append(
            "legacy LR-1 score rows are not required to carry hardened scorer provenance"
        )
    if config.get("split_role") is None:
        warnings.append("run config predates explicit split_role recording")
    return {
        "passed": True,
        "split_role": split_role,
        "records": len(manifest),
        "prompts": len(prompt_text),
        "seeds": seeds,
        "actions": action_ids,
        "blocks": len(block_devices),
        "devices": sorted({next(iter(values)) for values in block_devices.values()}),
        "required_score_keys": list(required_score_keys),
        "max_update_ratio": max_update_ratio,
        "max_abs_mean_error": max_mean_error,
        "max_abs_variance_error": max_variance_error,
        "scorer_provenance_schema": (
            SCORER_PROVENANCE_SCHEMA if native_renderer_run else None
        ),
        "scorer_provenance_sha256": scorer_provenance_sha256,
        "all_action_png_hashes_distinct_within_block": (
            False if identity_pair is not None else require_distinct_actions
        ),
        "allowed_identity_pairs": [
            sorted(pair)
            for pair in sorted(
                identity_pairs, key=lambda value: tuple(sorted(value))
            )
        ],
        "observed_identity_pairs": [
            sorted(pair)
            for pair in sorted(
                observed_identity_pairs, key=lambda value: tuple(sorted(value))
            )
        ],
        "registered_identity_pair": list(identity_pair) if identity_pair else None,
        "identity_pair_png_hashes_equal": identity_pair is not None,
        "warnings": warnings,
        "provenance": {
            "config_sha256": sha256_file(config_path),
            "manifest_sha256": sha256_file(manifest_path),
            "scores_sha256": sha256_file(scores_path),
            "prompts_sha256": sha256_file(prompts_path),
            "source_actions_sha256": sha256_file(source_actions_path),
            "source_template_sha256": (
                native_registration_provenance["source_template_sha256"]
                if native_registration_provenance is not None
                else None
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--source_actions", required=True)
    parser.add_argument(
        "--registration_actions",
        default="",
        help="optional frozen registration YAML to verify against an executable copy",
    )
    parser.add_argument("--split_role", required=True, choices=tuple(SPLIT_NAMES))
    parser.add_argument("--output", default="")
    parser.add_argument("--skip_image_verify", action="store_true")
    parser.add_argument("--allow_duplicate_action_hashes", action="store_true")
    args = parser.parse_args()

    report = audit_run(
        args.run_dir,
        args.prompts,
        args.source_actions,
        split_role=args.split_role,
        verify_images=not args.skip_image_verify,
        require_distinct_actions=not args.allow_duplicate_action_hashes,
        registration_actions_path=args.registration_actions or None,
    )
    output = Path(args.output) if args.output else Path(args.run_dir) / "run_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"audit -> {output}")


if __name__ == "__main__":
    main()
