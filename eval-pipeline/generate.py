"""Generation stage of the RepLDM eval pipeline (env: `repldm`).

Generates an image for every (prompt x seed x guidance action) combination and
writes a lossless PNG + a per-image sidecar JSON manifest recording the *full*
guidance config. Scoring is a separate stage (score.py, env `repldm_eval`) that
reads these PNGs + manifests — see eval-pipeline/README.md.

Design notes (grounded in the actual pipeline, EXPERIMENT_PLAN §4/§13):
  * Stage 1 remains the default. High-resolution Stage 2 requires the explicit
    `--stage2` opt-in and uses task-seeded resampling noise for paired actions.
    Attention Guidance acts only in Stage 1 in both regimes.
  * `--scales` preserves the constant-scalar sweep. `--actions` accepts a YAML
    grid containing no-AG, legacy schedule, scalar, and three-band actions.
  * Attention Guidance schedules are indexed by t_index and are reentrant, so
    the same pipeline/controller can safely serve repeated rollouts.
  * Multi-GPU with one queue per device. Every action for a (prompt, seed)
    block stays on one device, including after an interrupted run. Complete
    PNG + JSON pairs are skipped.

Example:
  conda run -n diff_attn python eval-pipeline/generate.py \
      --devices 1 --prompts eval-pipeline/prompts/eval_v1.csv \
      --out_dir outputs/frequency_action_pilot \
      --actions eval-pipeline/configs/frequency_action_pilot.yaml \
      --seeds 0,42,123
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager, nullcontext
import fcntl
import glob
import hashlib
import json
import math
import os
import platform
import queue
import re
import subprocess
import sys
import tempfile
import time
import traceback

import pandas as pd
import torch
import torch.multiprocessing as tmp
import yaml
import diffusers
from diffusers import EulerAncestralDiscreteScheduler

# Keep optional classes lazy so legacy environments that predate UniPC still
# run non-scheduler actions and fail with a clear action-validation error only
# when a missing reference is requested.
DPMSolverMultistepScheduler = getattr(diffusers, "DPMSolverMultistepScheduler", None)
UniPCMultistepScheduler = getattr(diffusers, "UniPCMultistepScheduler", None)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from AttentionGuidance import (
    ConstantGuidanceController,
    DIFFUSERS_FREEU_IMPLEMENTATION,
    FREEU_IMPLEMENTATIONS,
    LAZY_LATENT_STRUCTURE_BASIS_NAMES,
    LAZY_LATENT_STRUCTURE_PROVIDER_ID,
    MOMENT_TANGENT_MODES,
    LazyLatentStructureBasisProvider,
    LATENT_STRUCTURE_PROVIDER_IMPLEMENTATION_ALIASES,
    RESIDUAL_MODES,
    SEMANTIC_TRANSPORT_MODES,
    StructuralUNetBasisProvider,
    build_fixed_coefficient_renderer,
    FreeUSchedule,
    PAPER_FREEU_IMPLEMENTATION,
    PAPER_FREEU_PORT_DIFFUSERS_VERSION,
    PAPER_FREEU_SOURCE_COMMIT,
    TrajectoryCorrectionConfig,
    installed_freeu_implementation,
    latent_structure_required_hook_names,
    normalize_latent_structure_bases,
    normalize_latent_structure_provider_implementation,
)
from AttentionGuidance.attention_baselines import (
    ATTENTION_BASELINE_LAYER_GROUPS,
    ATTENTION_PROBABILITY_DTYPES,
    GAG_EQ13_IMPLEMENTATION,
    GAG_INHERITED_LAYER_POLICY,
    GAG_INHERITED_PROBABILITY_DTYPE,
    GAG_PAPER_EQUATIONS,
    GAG_PAPER_ID,
    PLADIS_OPERATOR_PORT_IMPLEMENTATION,
    PLADIS_PINNED_PROBABILITY_DTYPE,
    PLADIS_PINNED_SDXL_GROUP_COUNTS,
    PLADIS_PINNED_SDXL_LAYERS,
    PLADIS_PINNED_SDXL_PROCESSOR_COUNT,
    PLADIS_PINNED_SDXL_PROCESSOR_NAMES_SHA256,
    PLADIS_PORT_DIFFUSERS_VERSION,
    PLADIS_SOURCE_COMMIT,
    installed_attention_baseline,
)
from InferencePipelines import RepLDMSDXLPipeline
from generation_environment import validate_environment_lock
from s7_provenance import (
    PROVENANCE_SCHEMA,
    action_sha256,
    image_sha256,
    json_sha256,
    resolve_frequency_band_cutoffs,
    validate_design_rows,
    validate_run_contract,
    validate_sidecar,
)

DEFAULT_CACHE_DIR = "/mnt/miah204/bycao/RepLDM/pretrained_ckpts"
DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_NEG = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"
FRLA_SELECTION_SCHEMAS = {
    "frla_relational_actions_v1",
    "frla_relational_validation_v1",
}
TRAJECTORY_SCHEMAS = {
    "trajectory_correction_actions_v1",
    "trajectory_correction_validation_v1",
}
CFG_BASELINE_SCHEMA = "cfg_baselines_v1"
NATIVE_RENDERER_SCHEMA = "scheduler_native_fixed_headroom_actions_v1"
NATIVE_RENDERER_AUTH_SOURCE_TEMPLATE = (
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
LATENT_RENDERER_SCHEDULER_MAPPINGS = {
    "legacy_unit",
    "euler_clean_endpoint",
}
LATENT_RENDERER_BASIS_NORMALIZATIONS = {
    "legacy_l2_to_rms",
    "match_rms",
}
# Keep the old module symbol patchable for downstream tests and launchers while
# selecting the lazy implementation by default.  A monkeypatch of either name
# can therefore inject a test provider without changing the runtime contract.
_DEFAULT_STRUCTURAL_PROVIDER_CLASS = StructuralUNetBasisProvider
CFG_BASELINE_ACTION_SCALES = {
    "cfg_2p5": 2.5,
    "cfg_5p0": 5.0,
    "cfg_7p5": 7.5,
    "cfg_10p0": 10.0,
    "cfg_15p0": 15.0,
}
CFG_BASELINE_SAMPLING_KEYS = {
    "model",
    "model_revision",
    "base_scheduler",
    "resolution",
    "num_inference_steps",
    "default_cfg_scale",
    "negative_prompt",
    "power_calibrate",
    "stage2",
    "extra_unet_calls",
    "initialization",
    "cfg_source",
    "cfg_pipeline_argument",
    "guidance_rescale",
}
CFG_BASELINE_EXECUTION_ORDER = {
    "version": "action-order-v1",
    "grouping": ["prompt_index", "seed"],
    "digest": "sha256",
    "input": "action-order-v1:{prompt_index}:{seed}:{action_id}",
    "sort": "ascending_raw_digest",
    "single_device_per_group": True,
}
CFG_BASELINE_PROMPTS = "eval-pipeline/prompts/s5_development.csv"
CFG_BASELINE_PROMPTS_SHA256 = (
    "cf9ae37f2c066e5a35712e2aaf6ea637d85281dc7cd2b104fbc07fe1d8e5201e"
)
CFG_BASELINE_SPLIT_SEEDS = {"development": [0, 42, 123]}
CFG_BASELINE_AUTH_SOURCE_TEMPLATE = "eval-pipeline/configs/cfg_baselines_v1.yaml"
CFG_BASELINE_AUTH_SCOPE = "development_only_cfg_selection"
CFG_BASELINE_AUTH_FIELDS = {
    "reviewer",
    "reviewed_commit",
    "source_template",
    "source_template_sha256",
    "scope",
}
CFG_BASELINE_MODEL_REVISION = "462165984030d82259a11f4367a4eed129e94a7b"
CFG_BASELINE_SCHEDULER_RUNTIME = {
    "config_sha256_v2": "6bf0509f22d8d3a06d6493c2291af8655f6f74846b2d2eae3cf71b5cda000102",
    "num_inference_steps": 50,
    "schedule_sha256": "302d2452f411bf3eea64f8dd3530e232b95c23aed7b818ed6697982a4428c144",
    "construction_init_noise_sigma": 14.648818969726562,
    "effective_init_noise_sigma": 13.158469200134277,
}
STRUCTURAL_CONTROL_SCHEMA = "scheduler_native_structural_controls_actions_v1"
STRUCTURAL_CONTROL_REGISTRATION_SCHEMA = "scheduler_native_structural_controls_v1"
STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE = (
    "eval-pipeline/configs/"
    "scheduler_native_structural_controls_development_registration_v1.yaml"
)
STRUCTURAL_CONTROL_AUTH_SCOPE = "development_only_baseline_calibration"
STRUCTURAL_CONTROL_AUTH_FIELDS = {
    "reviewer",
    "reviewed_commit",
    "source_template",
    "source_template_sha256",
    "scope",
    "gpu_generation",
    "scoring",
    "method_selection",
    "result_access_before_freeze",
}
STRUCTURAL_CONTROL_IMPLEMENTATION_PATHS = (
    "AttentionGuidance/__init__.py",
    "AttentionGuidance/ancestral_correction.py",
    "AttentionGuidance/attention_baselines.py",
    "AttentionGuidance/attention_guidance.py",
    "AttentionGuidance/controller.py",
    "AttentionGuidance/freeu.py",
    "AttentionGuidance/latent_renderer.py",
    "AttentionGuidance/semantic_transport.py",
    "AttentionGuidance/types.py",
    "InferencePipelines/FreeScale/pipeline_freescale_sdxl.py",
    "InferencePipelines/FreeScale/scale_attention.py",
    "InferencePipelines/__init__.py",
    "InferencePipelines/cfg_batch.py",
    "InferencePipelines/RepLDM/pipeline_repldm_sdxl.py",
    "InferencePipelines/RepLDM/pipeline_repldm_sdxl_controlnet.py",
    "eval-pipeline/generate.py",
    "eval-pipeline/generation_environment.py",
    "eval-pipeline/s7_provenance.py",
)
STRUCTURAL_CONTROL_ANALYSIS_SCHEMA = "structural_control_analysis_implementation_v1"
STRUCTURAL_CONTROL_ANALYSIS_PATHS = (
    "AttentionGuidance/__init__.py",
    "eval-pipeline/audit_latent_renderer_run.py",
    "eval-pipeline/audit_structural_control_run.py",
    "eval-pipeline/compare_actions.py",
    "eval-pipeline/evaluate_structural_control_run.py",
    "eval-pipeline/generate.py",
    "eval-pipeline/generation_environment.py",
    "eval-pipeline/s7_provenance.py",
    "eval-pipeline/scorer_provenance.py",
)
STRUCTURAL_CONTROL_SAMPLING_KEYS = {
    "model",
    "model_revision",
    "pipeline",
    "resolution",
    "num_inference_steps",
    "default_cfg_scale",
    "cfg_source",
    "cfg_pipeline_argument",
    "negative_prompt",
    "power_calibrate",
    "guidance_rescale",
    "scheduler",
    "prediction_type",
    "scheduler_churn",
    "initialization",
    "stage2",
    "extra_unet_calls",
    "torch_dtype",
    "variant",
    "local_files_only",
    "low_vram",
    "batch_size",
    "num_images_per_prompt",
    "attention_mask_policy",
}
STRUCTURAL_CONTROL_ACTION_IDS = (
    "no_op_cfg7p5",
    "cfg_only_5",
    "conference_tfsa",
    "freeu_diffusers_historical",
    "freeu_diffusers_paper_parameters",
    "freeu_paper_adaptive",
    "pladis_operator_port",
    "gag_eq13_reimplementation",
)
STRUCTURAL_CONTROL_PROMPTS = (
    "eval-pipeline/prompts/scheduler_native_fixed_headroom_development.csv"
)
STRUCTURAL_CONTROL_PROMPTS_SHA256 = (
    "065a86b95200eb89dc367ffe0f9f8c2d0a64fdab827e4004a1b629b169a6173f"
)
STRUCTURAL_CONTROL_SMOKE_PROMPTS = (
    "eval-pipeline/prompts/scheduler_native_fixed_headroom_smoke.csv"
)
STRUCTURAL_CONTROL_SMOKE_PROMPTS_SHA256 = (
    "60a3bf278689165bcb7a4bdf1f18c7ab91d8bf7a77eb287369a60fe40b6ef4e1"
)
STRUCTURAL_CONTROL_SPLIT_SEEDS = {
    "engineering_smoke": [1798464083],
    "development": [1932556753, 1065503757, 201635682]
}
STRUCTURAL_CONTROL_ENGINEERING_SMOKE = {
    "role": "engineering_only",
    "engineering_only": True,
    "prompts": STRUCTURAL_CONTROL_SMOKE_PROMPTS,
    "prompts_sha256": STRUCTURAL_CONTROL_SMOKE_PROMPTS_SHA256,
    "expected_prompt_count": 11,
    "expected_challenges": 11,
    "seeds": STRUCTURAL_CONTROL_SPLIT_SEEDS["engineering_smoke"],
    "action_count": 8,
    "expected_task_count": 88,
    "require_all_actions_distinct_within_block": True,
    "quality_scoring": False,
    "formal_matrix_evidence": False,
    "quality_claim_allowed": False,
    "method_selection_allowed": False,
}
STRUCTURAL_CONTROL_SMOKE_EVIDENCE_SCOPE = {
    key: STRUCTURAL_CONTROL_ENGINEERING_SMOKE[key]
    for key in (
        "engineering_only",
        "formal_matrix_evidence",
        "quality_claim_allowed",
        "method_selection_allowed",
    )
}
STRUCTURAL_CONTROL_EVIDENCE_SCOPE_KEYS = tuple(
    STRUCTURAL_CONTROL_SMOKE_EVIDENCE_SCOPE
)
STRUCTURAL_CONTROL_FREEU_RESOLUTION_COUNTS = [3, 3, 3]
STRUCTURAL_CONTROL_FREEU_CHANNEL_COUNTS = {"1280": 4, "640": 3, "320": 2}
STRUCTURAL_CONTROL_FREEU_JOINT_COUNTS = {
    "0:1280": 3,
    "1:1280": 1,
    "1:640": 2,
    "2:640": 1,
    "2:320": 2,
}
STRUCTURAL_CONTROL_FREEU_CONSTANT_EFFECT_COUNTS = {
    "b1_s1": 3,
    "b2_s2": 3,
    "no_op": 3,
}
STRUCTURAL_CONTROL_FREEU_PAPER_EFFECT_COUNTS = {
    "b1_s1": 4,
    "b2_s2": 3,
    "no_op": 2,
}

# Keep this map explicit: scheduler names come from experiment YAML and must
# never be resolved through eval/importlib. The kwargs are part of the action
# hash after load_actions normalizes them.
SCHEDULER_BASELINE_CLASSES = {
    "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
}
if DPMSolverMultistepScheduler is not None:
    SCHEDULER_BASELINE_CLASSES["DPMSolverMultistepScheduler"] = DPMSolverMultistepScheduler
if UniPCMultistepScheduler is not None:
    SCHEDULER_BASELINE_CLASSES["UniPCMultistepScheduler"] = UniPCMultistepScheduler
SCHEDULER_BASELINE_DEFAULT_KWARGS = {
    "EulerAncestralDiscreteScheduler": {},
    "DPMSolverMultistepScheduler": {
        "solver_order": 2,
        "algorithm_type": "dpmsolver++",
        "solver_type": "midpoint",
        "lower_order_final": True,
    },
    "UniPCMultistepScheduler": {
        "solver_order": 2,
        "predict_x0": True,
        "solver_type": "bh2",
        "lower_order_final": True,
    },
}
SCHEDULER_BASELINE_ALLOWED_KWARGS = {
    "EulerAncestralDiscreteScheduler": set(),
    "DPMSolverMultistepScheduler": {
        "solver_order",
        "algorithm_type",
        "solver_type",
        "lower_order_final",
    },
    "UniPCMultistepScheduler": {
        "solver_order",
        "predict_x0",
        "solver_type",
        "lower_order_final",
    },
}


def _require_int(value, *, action_id: str, field: str, expected: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"{action_id}: {field} must be integer {expected}")


@contextmanager
def generation_output_lock(out_dir: str):
    """Hold a non-blocking process lock for one generation output directory."""
    output_dir = os.path.abspath(out_dir)
    os.makedirs(output_dir, exist_ok=True)
    lock_path = os.path.join(output_dir, ".generate.lock")
    lock_handle = open(lock_path, "a+")
    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"generation output directory is already locked: {output_dir}"
            ) from exc
        yield lock_path
    finally:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            lock_handle.close()


@contextmanager
def atomic_text_writer(path: str):
    """Write one text artifact through a unique same-directory temp file."""
    destination = os.path.abspath(path)
    directory = os.path.dirname(destination)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination)}.",
        suffix=".tmp",
        dir=directory,
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


def atomic_write_json(path: str, value, *, indent: int | None = None) -> None:
    with atomic_text_writer(path) as handle:
        json.dump(value, handle, indent=indent)


def atomic_save_png(image, path: str) -> None:
    """Save and publish a PNG without exposing a partially encoded image."""
    destination = os.path.abspath(path)
    directory = os.path.dirname(destination)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination)}.",
        suffix=".tmp",
        dir=directory,
    )
    os.close(fd)
    try:
        image.save(temporary, format="PNG")
        with open(temporary, "r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def generation_artifacts_exist(img_dir: str) -> bool:
    return any(
        name.endswith((".png", ".json")) for name in os.listdir(img_dir)
    )


def task_id(prompt_index: int, seed: int, action: dict, legacy_scale_id: bool = False) -> str:
    if legacy_scale_id:
        return f"p{prompt_index}_seed{seed}_s{action['scale']:.4f}"
    return f"p{prompt_index}_seed{seed}_a{action['id']}"


def build_tasks(prompts: pd.DataFrame, seeds, actions, legacy_scale_ids: bool = False) -> list:
    tasks = []
    for _, row in prompts.iterrows():
        pid = int(row["index"])
        for seed in seeds:
            for action in actions:
                task = {
                    "id": task_id(pid, seed, action, legacy_scale_ids),
                    "prompt_index": pid,
                    "prompt": str(row["TEXT"]),
                    "bucket": str(row.get("bucket", "")),
                    "seed": int(seed),
                    "action_id": action["id"],
                    "action_type": action["type"],
                    "action": action,
                    "action_sha256": action_sha256(action),
                }
                if "scale" in action:
                    task["scale"] = float(action["scale"])
                if "band_scales" in action:
                    task["band_scales"] = list(action["band_scales"])
                task["residual_mode"] = action.get("residual_mode", "raw")
                tasks.append(task)
    return tasks


def scale_actions(scales):
    return [
        {
            "id": f"scale_{scale:.4f}",
            "type": "none" if scale == 0 else "scalar",
            "scale": float(scale),
        }
        for scale in scales
    ]


def has_scheduler_native_renderer(actions) -> bool:
    """Return whether a matrix needs the strict native-renderer run contract."""
    return any(
        action.get("type") == "latent_renderer_fixed"
        and action.get("latent_renderer_provider", {}).get("scheduler_mapping")
        == "euler_clean_endpoint"
        for action in actions
    )


def native_renderer_step_diagnostics_required(cfg: dict, action: dict) -> bool:
    """Return whether an action must carry the scheduler-native step ledger."""
    return bool(
        cfg.get("native_renderer_registered")
        and action.get("type") == "latent_renderer_fixed"
        and action.get("latent_renderer_provider", {}).get("scheduler_mapping")
        == "euler_clean_endpoint"
    )


def strict_registered_run(cfg: dict) -> bool:
    """Return whether resume/consolidation must use the strict S7 contract."""
    return bool(
        cfg.get("trajectory_registered")
        or cfg.get("scheduler_baseline_registered")
        or cfg.get("cfg_baseline_registered")
        or cfg.get("native_renderer_registered")
        or cfg.get("structural_control_registered")
    )


def structural_control_evidence_scope(split_role: str | None) -> dict:
    """Return the explicit non-evidentiary scope attached to smoke artifacts."""
    if split_role == "engineering_smoke":
        return dict(STRUCTURAL_CONTROL_SMOKE_EVIDENCE_SCOPE)
    return {}


def validate_structural_intervention_runtime(
    record: dict, action: dict, *, num_inference_steps: int
) -> None:
    """Validate actual activation ledgers for one structural-control action."""
    steps = int(num_inference_steps)
    action_type = action.get("type")
    if action_type == "legacy":
        delay_steps = int(action.get("delay_steps", 0))
        expected_density = [1] * (steps - delay_steps) + [0] * delay_steps
        expected_decay = list(action.get("decay") or []) or None
        if (
            record.get("attn_guidance_scale") != float(action.get("scale", 0.0))
            or record.get("attn_guidance_density") != expected_density
            or record.get("attn_guidance_decay") != expected_decay
        ):
            raise ValueError("structural-control TFSA runtime arguments differ")
        ledger = record.get("attention_guidance_runtime")
        active_steps = steps - delay_steps
        if not isinstance(ledger, list) or len(ledger) != steps:
            raise ValueError("structural-control TFSA runtime ledger is incomplete")
        for step_index, item in enumerate(ledger):
            t_index = steps - 1 - step_index
            active = step_index >= delay_steps
            if not isinstance(item, dict) or item.get("step_index") != step_index:
                raise ValueError("structural-control TFSA step index differs")
            if item.get("t_index") != t_index or item.get("active") is not active:
                raise ValueError("structural-control TFSA activation schedule differs")
            observed_scale = item.get("applied_scale")
            if isinstance(observed_scale, bool) or not isinstance(
                observed_scale, (int, float)
            ) or not math.isfinite(float(observed_scale)):
                raise ValueError("structural-control TFSA scale ledger is invalid")
            expected_scale = 0.0
            if active:
                rank = step_index - delay_steps
                phase = 0.0 if active_steps == 1 else rank / (active_steps - 1)
                expected_scale = float(action["scale"]) * (
                    (math.cos(math.pi * phase) + 1.0) / 2.0
                ) ** float(action["decay"][2])
                expected_scale = max(expected_scale, float(action["decay"][1]))
            if not math.isclose(
                float(observed_scale), expected_scale, rel_tol=5e-3, abs_tol=5e-6
            ):
                raise ValueError("structural-control TFSA applied scale differs")
    else:
        if (
            record.get("attn_guidance_scale") != 0.0
            or record.get("attn_guidance_density") != "all"
            or record.get("attn_guidance_decay") is not None
            or record.get("attention_guidance_runtime") != []
        ):
            raise ValueError("non-TFSA action has stale TFSA runtime provenance")

    if action_type == "freeu":
        schedule = freeu_runtime(action)
        expected_runtime = [
            {
                "step_index": step_index,
                "parameters": list(
                    schedule.at(step_index / max(steps - 1, 1)).as_tuple()
                ),
            }
            for step_index in range(steps)
        ]
        if record.get("freeu_runtime") != expected_runtime:
            raise ValueError("structural-control FreeU activation ledger differs")
        expected_calls_per_step = action.get("expected_operator_calls_per_step")
        expected_resolution_counts = action.get(
            "expected_resolution_idx_call_counts_per_step"
        )
        expected_channel_counts = action.get(
            "expected_hidden_channel_call_counts_per_step"
        )
        expected_joint_counts = action.get(
            "expected_resolution_channel_call_counts_per_step"
        )
        expected_effect_counts = action.get(
            "expected_operator_effect_call_counts_per_step"
        )
        if not isinstance(expected_calls_per_step, int) or not isinstance(
            expected_resolution_counts, list
        ):
            raise ValueError("structural-control FreeU call topology is missing")
        expected_operator_runtime = {
            "implementation": action.get("implementation"),
            "operator_calls_total": expected_calls_per_step * steps,
            "resolution_idx_call_counts": {
                str(index): int(count) * steps
                for index, count in enumerate(expected_resolution_counts)
            },
            "hidden_channel_call_counts": {
                str(channel): int(count) * steps
                for channel, count in expected_channel_counts.items()
            },
            "resolution_channel_call_counts": {
                str(key): int(count) * steps
                for key, count in expected_joint_counts.items()
            },
            "operator_effect_call_counts": {
                str(effect): int(count) * steps
                for effect, count in expected_effect_counts.items()
            },
        }
        if record.get("freeu_operator_runtime") != expected_operator_runtime:
            raise ValueError("structural-control FreeU operator calls differ")
    elif record.get("freeu_runtime") != [] or record.get(
        "freeu_operator_runtime"
    ) is not None:
        raise ValueError("non-FreeU action has stale FreeU runtime provenance")


def validate_structural_control_sidecar(
    record: dict, cfg: dict, *, expected_task: dict | None = None
) -> None:
    """Reject a structural-control sidecar that cannot be safely resumed."""
    if not cfg.get("structural_control_registered"):
        return
    expected_evidence_scope = structural_control_evidence_scope(cfg.get("split_role"))
    config_evidence_scope = {
        key: cfg[key] for key in STRUCTURAL_CONTROL_EVIDENCE_SCOPE_KEYS if key in cfg
    }
    record_evidence_scope = {
        key: record[key]
        for key in STRUCTURAL_CONTROL_EVIDENCE_SCOPE_KEYS
        if key in record
    }
    if config_evidence_scope != expected_evidence_scope:
        raise ValueError("structural-control run evidence scope differs")
    if record_evidence_scope != expected_evidence_scope:
        raise ValueError("structural-control sidecar evidence scope differs")
    sampling = cfg.get("registered_sampling")
    scheduler_runtime = cfg.get("scheduler_runtime")
    runtime = cfg.get("runtime_provenance")
    if not all(isinstance(value, dict) for value in (sampling, scheduler_runtime, runtime)):
        raise ValueError("structural-control resume contract is incomplete")
    if record.get("registered_sampling") != sampling or record.get(
        "scheduler_runtime"
    ) != scheduler_runtime:
        raise ValueError("structural-control sidecar sampling contract differs")
    if record.get("num_inference_steps") != 50 or record.get(
        "unet_calls_per_step"
    ) != [1] * 50 or record.get("extra_unet_calls") != 0:
        raise ValueError("structural-control sidecar violates matched 50x1 U-Net calls")

    if record.get("scheduler_name") != "EulerDiscreteScheduler" or record.get(
        "base_scheduler_name"
    ) != "EulerDiscreteScheduler":
        raise ValueError("structural-control sidecar scheduler class differs")
    expected_config_hash = scheduler_runtime.get("config_sha256_v2")
    if any(
        record.get(key) != expected_config_hash
        for key in (
            "scheduler_config_sha256_v2",
            "active_scheduler_config_sha256_v2",
        )
    ):
        raise ValueError("structural-control sidecar Euler config differs")
    for payload_key, digest_key in (
        ("scheduler_config", "scheduler_config_sha256_v2"),
        ("active_scheduler_config", "active_scheduler_config_sha256_v2"),
    ):
        payload = record.get(payload_key)
        try:
            observed_hash = scheduler_config_payload_sha256(payload)
        except ValueError as exc:
            raise ValueError(
                "structural-control sidecar scheduler config payload differs"
            ) from exc
        if observed_hash != record.get(digest_key):
            raise ValueError("structural-control sidecar scheduler config hash differs")
    timesteps = record.get("scheduler_timesteps")
    sigmas = record.get("scheduler_sigmas")
    if (
        not isinstance(timesteps, list)
        or len(timesteps) != 50
        or not isinstance(sigmas, list)
        or len(sigmas) != 51
        or record.get("scheduler_schedule_sha256")
        != scheduler_runtime.get("schedule_sha256")
        or json_sha256({"timesteps": timesteps, "sigmas": sigmas})
        != scheduler_runtime.get("schedule_sha256")
    ):
        raise ValueError("structural-control sidecar Euler schedule differs")
    scheduler_values = {
        "scheduler_construction_init_noise_sigma": scheduler_runtime.get(
            "construction_init_noise_sigma"
        ),
        "scheduler_effective_init_noise_sigma": scheduler_runtime.get(
            "effective_init_noise_sigma"
        ),
        "scheduler_init_noise_sigma": scheduler_runtime.get(
            "construction_init_noise_sigma"
        ),
        "scheduler_order": 1,
        "scheduler_kwargs": {},
    }
    if any(record.get(key) != value for key, value in scheduler_values.items()):
        raise ValueError("structural-control sidecar Euler ledger differs")

    device = str(record.get("device", ""))
    devices = cfg.get("devices")
    worker_device = record.get("worker_device_provenance")
    required_device_fields = {
        "gpu",
        "compute_capability",
        "total_memory_bytes",
        "requested_device",
        "logical_device_index",
        "physical_device_index",
        "gpu_uuid",
        "pci_bus_id",
        "cuda_visible_devices",
    }
    if (
        not re.fullmatch(r"cuda:\d+", device)
        or not isinstance(devices, list)
        or device not in devices
        or not isinstance(worker_device, dict)
        or set(worker_device) != required_device_fields
        or worker_device.get("requested_device") != device
        or worker_device.get("logical_device_index") != int(device.split(":", 1)[1])
        or isinstance(worker_device.get("physical_device_index"), bool)
        or not isinstance(worker_device.get("physical_device_index"), int)
        or worker_device["physical_device_index"] < 0
        or not re.fullmatch(
            r"GPU-[0-9a-fA-F-]{36}", str(worker_device.get("gpu_uuid", ""))
        )
        or not str(worker_device.get("pci_bus_id", "")).strip()
        or isinstance(worker_device.get("total_memory_bytes"), bool)
        or not isinstance(worker_device.get("total_memory_bytes"), int)
        or worker_device["total_memory_bytes"] <= 0
    ):
        raise ValueError("structural-control sidecar worker device identity differs")
    expected_hardware = runtime.get("generation_environment_hardware")
    if not isinstance(expected_hardware, dict) or any(
        worker_device.get(key) != expected_hardware.get(key)
        for key in ("gpu", "compute_capability")
    ):
        raise ValueError("structural-control sidecar worker hardware differs")
    expected_determinism = runtime.get("generation_environment_determinism")
    if not isinstance(expected_determinism, dict) or record.get(
        "worker_determinism_provenance"
    ) != expected_determinism:
        raise ValueError("structural-control sidecar worker determinism differs")
    model_load = record.get("model_load_provenance")
    expected_model_load = {
        "torch_dtype": sampling.get("torch_dtype"),
        "variant": sampling.get("variant"),
        "local_files_only": sampling.get("local_files_only"),
        "revision": sampling.get("model_revision"),
    }
    if model_load != expected_model_load:
        raise ValueError("structural-control sidecar model-load provenance differs")
    for key, expected in runtime.items():
        if record.get(key) != expected:
            raise ValueError(
                f"structural-control sidecar runtime provenance differs for {key!r}"
            )
    expected_run_fields = {
        "height": int(sampling["resolution"]),
        "width": int(sampling["resolution"]),
        "power_calibrate": int(sampling["power_calibrate"]),
        "frequency_band_cutoffs": cfg.get("frequency_band_cutoffs"),
        "stage": cfg.get("stage_name"),
        "stage2_enabled": bool(sampling["stage2"]),
        "models_to_cpu": cfg.get("models_to_cpu"),
        "multi_encoder": cfg.get("multi_encoder"),
        "multi_decoder": cfg.get("multi_decoder"),
        "num_resample_timesteps": cfg.get("num_resample_timesteps"),
        "init_rates": cfg.get("init_rates"),
        "stage2_noise_source": cfg.get("stage2_noise_source"),
        "model_name": sampling["model"],
        "model_revision": sampling["model_revision"],
    }
    if any(record.get(key) != expected for key, expected in expected_run_fields.items()):
        raise ValueError("structural-control sidecar run settings differ")

    action = record.get("action")
    if expected_task is not None:
        action = expected_task.get("action")
    if not isinstance(action, dict):
        raise ValueError("structural-control sidecar action is missing")
    expected_cfg = float(action.get("cfg_scale", float("nan")))
    if record.get("guidance_scale") != expected_cfg or record.get(
        "guidance_rescale"
    ) != 0.0:
        raise ValueError("structural-control sidecar CFG provenance differs")
    action_type = action.get("type")
    validate_structural_intervention_runtime(
        record, action, num_inference_steps=int(sampling["num_inference_steps"])
    )
    if action_type == "freeu":
        expected_freeu = {
            "freeu_schedule": action.get("freeu_schedule"),
            "freeu_implementation": action.get("implementation"),
            "freeu_source_commit": action.get("source_commit"),
            "freeu_implementation_diffusers_version": action.get(
                "implementation_diffusers_version"
            ),
            "freeu_preserve_moments": False,
        }
        if any(record.get(key) != value for key, value in expected_freeu.items()):
            raise ValueError("structural-control sidecar FreeU provenance differs")
    elif any(
        record.get(key) is not None
        for key in (
            "freeu_schedule",
            "freeu_implementation",
            "freeu_source_commit",
            "freeu_implementation_diffusers_version",
        )
    ) or record.get("freeu_preserve_moments") is not False:
        raise ValueError("structural-control sidecar has stale FreeU provenance")
    if action_type == "attention_baseline":
        expected_attention = {
            "attention_baseline_implementation": action.get("implementation"),
            "attention_baseline_source_commit": action.get("source_commit"),
            "attention_baseline_paper_id": action.get("paper_id"),
            "attention_baseline_topology": {
                "group_counts": action.get("expected_processor_group_counts"),
                "processor_count": action.get("expected_processor_count"),
                "processor_names_sha256": action.get(
                    "expected_processor_names_sha256"
                ),
                "processors_called": action.get("expected_processor_count"),
                "processor_calls_total": action.get("expected_processor_count")
                * int(sampling["num_inference_steps"]),
                "processor_call_count_min": int(sampling["num_inference_steps"]),
                "processor_call_count_max": int(sampling["num_inference_steps"]),
            },
        }
        if any(record.get(key) != value for key, value in expected_attention.items()):
            raise ValueError("structural-control sidecar attention provenance differs")
    elif any(
        record.get(key) is not None
        for key in (
            "attention_baseline_implementation",
            "attention_baseline_source_commit",
            "attention_baseline_paper_id",
            "attention_baseline_topology",
        )
    ):
        raise ValueError("structural-control sidecar has stale attention provenance")


def scheduler_isolation_required(cfg: dict) -> bool:
    """Return whether each action requires a fresh scheduler instance."""
    return bool(
        cfg.get("scheduler_baseline_registered")
        or cfg.get("cfg_baseline_registered")
        or cfg.get("native_renderer_registered")
        or cfg.get("structural_control_registered")
    )


def group_tasks_by_pair(tasks):
    """Keep every action for a (prompt, seed) pair on one worker/device."""
    groups = {}
    for task in tasks:
        key = (task["prompt_index"], task["seed"])
        groups.setdefault(key, []).append(task)
    return list(groups.values())


def task_is_complete(
    task: dict,
    img_dir: str,
    *,
    run_contract_sha256: str | None = None,
    structural_config: dict | None = None,
) -> bool:
    stem = os.path.join(img_dir, task["id"])
    if not (os.path.exists(stem + ".png") and os.path.exists(stem + ".json")):
        return False
    # Legacy, non-registered sweeps predate the S7 provenance contract.  Keep
    # their resume behavior unchanged; registered runs always pass a contract
    # hash and take the strict path below.
    if run_contract_sha256 is None:
        return True
    try:
        with open(stem + ".json") as handle:
            record = json.load(handle)
        validate_sidecar(
            record,
            os.path.dirname(img_dir),
            expected_task=task,
            expected_contract_sha256=run_contract_sha256,
        )
        if structural_config is not None:
            validate_structural_control_sidecar(
                record, structural_config, expected_task=task
            )
        if "execution_rank" in task and record.get("execution_rank") != task[
            "execution_rank"
        ]:
            raise ValueError("sidecar execution_rank differs from the frozen order")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def worker_resume_contract_sha256(cfg: dict) -> str | None:
    """Return the strict contract for registered workers and None for legacy runs."""
    return cfg["run_contract_sha256"] if strict_registered_run(cfg) else None


def recorded_group_device(
    task_group: list,
    img_dir: str,
    *,
    run_contract_sha256: str | None = None,
    structural_config: dict | None = None,
):
    """Return a valid block device; ignore crash debris for registered runs."""
    devices = set()
    for task in task_group:
        json_path = os.path.join(img_dir, task["id"] + ".json")
        if not os.path.exists(json_path):
            continue
        try:
            with open(json_path) as handle:
                record = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            if run_contract_sha256 is not None:
                continue
            raise ValueError(f"cannot read existing sidecar {json_path}: {exc}") from exc
        if run_contract_sha256 is not None:
            try:
                validate_sidecar(
                    record,
                    os.path.dirname(img_dir),
                    expected_task=task,
                    expected_contract_sha256=run_contract_sha256,
                )
                if structural_config is not None:
                    validate_structural_control_sidecar(
                        record, structural_config, expected_task=task
                    )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        if record.get("device"):
            devices.add(str(record["device"]))

    if len(devices) > 1:
        key = (task_group[0]["prompt_index"], task_group[0]["seed"])
        raise ValueError(
            f"existing prompt/seed block {key} spans devices {sorted(devices)}; "
            "paired inference is invalid, so use a new output directory"
        )
    return next(iter(devices), None)


def assign_tasks_to_devices(
    tasks: list,
    devices: list,
    img_dir: str,
    *,
    run_contract_sha256: str | None = None,
    structural_config: dict | None = None,
) -> dict:
    """Assign pending tasks without changing block placement on resume.

    The fallback round-robin index is computed over all blocks, not only the
    unfinished subset. An existing sidecar takes precedence over that fallback.
    """
    assignments = {device: [] for device in devices}
    for group_index, task_group in enumerate(group_tasks_by_pair(tasks)):
        recorded_device = recorded_group_device(
            task_group,
            img_dir,
            run_contract_sha256=run_contract_sha256,
            structural_config=structural_config,
        )
        ordered_group = sorted(
            task_group,
            key=lambda task: hashlib.sha256(
                (
                    f"action-order-v1:{task['prompt_index']}:{task['seed']}:"
                    f"{task['action_id']}"
                ).encode("utf-8")
            ).digest(),
        )
        for execution_rank, task in enumerate(ordered_group):
            task["execution_rank"] = execution_rank
        pending = [
            task
            for task in ordered_group
            if not task_is_complete(
                task,
                img_dir,
                run_contract_sha256=run_contract_sha256,
                structural_config=structural_config,
            )
        ]
        if not pending:
            continue
        device = recorded_device or devices[group_index % len(devices)]
        if device not in assignments:
            key = (task_group[0]["prompt_index"], task_group[0]["seed"])
            raise ValueError(
                f"prompt/seed block {key} must resume on {device}, which is not in "
                f"--devices {','.join(devices)}"
            )
        assignments[device].extend(pending)
    return {device: pending for device, pending in assignments.items() if pending}


def load_actions(path: str, num_inference_steps: int):
    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    action_schema = config.get("schema")
    if config.get("schema") in {
        "latent_renderer_registration_v1",
        "structural_control_registration_v1",
    }:
        raise ValueError(
            "registration manifests are not generation action configs; "
            "an independently reviewed executable YAML is required"
        )
    actions = config.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("action config must contain a non-empty 'actions' list")
    if config.get("schema") == "trajectory_correction_validation_v1":
        selected_action = str(config.get("selected_action", ""))
        if not selected_action:
            raise ValueError(
                "trajectory-correction validation requires selected_action frozen from development"
            )
        reference_ids = {
            str(action.get("id"))
            for action in actions
            if not bool(action.get("selection_eligible", True))
        }
        selected_ids = {"no_correction", selected_action} | reference_ids
        actions = [action for action in actions if str(action.get("id")) in selected_ids]
        if len(actions) < 2 or selected_action not in {
            str(action.get("id")) for action in actions
        }:
            raise ValueError(
                "selected_action must identify one registered non-baseline action"
            )

    seen = set()
    default_layer = str(
        config.get(
            "semantic_transport_layer",
            "up_blocks.0.attentions.0.transformer_blocks.0.attn1",
        )
    )
    default_topk = int(config.get("semantic_transport_topk", 16))
    if default_topk <= 0:
        raise ValueError("semantic_transport_topk must be positive")
    if "latent_renderer_provider" in config:
        provider_defaults = config.get("latent_renderer_provider")
        if provider_defaults is None:
            raise ValueError("latent_renderer_provider cannot be null")
    else:
        provider_defaults = config.get("required_provider", {})
    if provider_defaults is None:
        raise ValueError("required_provider cannot be null")
    if not isinstance(provider_defaults, dict):
        raise ValueError("latent_renderer_provider must be a mapping")
    normalized = []
    for raw in actions:
        if not isinstance(raw, dict):
            raise ValueError("each action must be a mapping")
        action = dict(raw)
        action_id = str(action.get("id", ""))
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", action_id):
            raise ValueError(f"invalid action id {action_id!r}; use letters, digits, '.', '_' or '-'")
        if action_id in seen:
            raise ValueError(f"duplicate action id {action_id!r}")
        seen.add(action_id)
        action_type = action.get("type")
        if action_schema == CFG_BASELINE_SCHEMA:
            unknown_fields = set(action) - {
                "id",
                "type",
                "cfg_scale",
            }
            if unknown_fields:
                raise ValueError(
                    f"{action_id}: cfg_baselines_v1 action has unsupported fields "
                    f"{sorted(unknown_fields)}"
                )
            if action_type != "none":
                raise ValueError(
                    f"{action_id}: cfg_baselines_v1 actions must use type 'none'"
                )
            if "cfg_scale" not in action:
                raise ValueError(
                    f"{action_id}: cfg_baselines_v1 requires an explicit cfg_scale"
                )
        if action_type not in {
            "none",
            "scalar",
            "legacy",
            "frequency_bands",
            "clean_transport",
            "attention_baseline",
            "latent_renderer_fixed",
            "freeu",
            "trajectory_correction",
            "scheduler_baseline",
        }:
            raise ValueError(f"unsupported action type {action_type!r} for {action_id}")

        if action_type == "none":
            action["scale"] = 0.0
        elif action_type in {"scalar", "legacy"}:
            action["scale"] = float(action["scale"])
            if action["scale"] < 0:
                raise ValueError(f"{action_id}: scale must be non-negative")
        elif action_type == "frequency_bands":
            action["band_scales"] = [float(value) for value in action["band_scales"]]
            if len(action["band_scales"]) != 3 or min(action["band_scales"]) < 0:
                raise ValueError(f"{action_id}: band_scales must be three non-negative values")
        elif action_type == "clean_transport":
            mode = str(action.get("transport_mode", ""))
            if mode not in SEMANTIC_TRANSPORT_MODES:
                raise ValueError(f"{action_id}: unsupported transport_mode {mode!r}")
            angle = float(action.get("angle", -1.0))
            if not math.isfinite(angle) or angle < 0:
                raise ValueError(f"{action_id}: angle must be finite and non-negative")
            topk = int(action.get("topk", default_topk))
            if topk <= 0:
                raise ValueError(f"{action_id}: topk must be positive")
            action["transport_mode"] = mode
            action["angle"] = angle
            action["topk"] = topk
            action["semantic_transport_layer"] = str(
                action.get("semantic_transport_layer", default_layer)
            )
            action["permutation_seed"] = int(action.get("permutation_seed", 1729))
            action["scale"] = 0.0
        elif action_type == "attention_baseline":
            baseline = str(action.get("attention_baseline", ""))
            if baseline not in {"pladis", "gag"}:
                raise ValueError(f"{action_id}: attention_baseline must be pladis or gag")
            alpha = float(action.get("alpha", 1.5))
            baseline_scale = float(action.get("baseline_scale", -1.0))
            if alpha != 1.5 or not math.isfinite(baseline_scale) or baseline_scale < 0:
                raise ValueError(
                    f"{action_id}: alpha must be 1.5 and baseline_scale non-negative"
                )
            action["attention_baseline"] = baseline
            action["alpha"] = alpha
            action["baseline_scale"] = baseline_scale
            action["eta"] = float(action.get("eta", 15.0))
            action["zeta"] = float(action.get("zeta", 0.0))
            if baseline == "gag" and (
                action["eta"] <= 0 or not 0 <= action["zeta"] <= 1
            ):
                raise ValueError(f"{action_id}: invalid GAG eta/zeta")
            if "applied_layers" in action:
                raw_layers = action["applied_layers"]
                if isinstance(raw_layers, (str, bytes)) or not isinstance(
                    raw_layers, list
                ):
                    raise ValueError(
                        f"{action_id}: applied_layers must be a list of layer groups"
                    )
                layers = [str(value) for value in raw_layers]
                if not layers or len(layers) != len(set(layers)):
                    raise ValueError(
                        f"{action_id}: applied_layers must be non-empty and unique"
                    )
                unknown_layers = set(layers) - ATTENTION_BASELINE_LAYER_GROUPS
                if unknown_layers:
                    raise ValueError(
                        f"{action_id}: unsupported attention layer groups "
                        f"{sorted(unknown_layers)}"
                    )
                action["applied_layers"] = layers
            if "probability_dtype" in action:
                probability_dtype = str(action["probability_dtype"])
                if probability_dtype not in ATTENTION_PROBABILITY_DTYPES:
                    raise ValueError(
                        f"{action_id}: probability_dtype must be one of "
                        f"{sorted(ATTENTION_PROBABILITY_DTYPES)}"
                    )
                action["probability_dtype"] = probability_dtype
            if "implementation" in action:
                implementation = str(action["implementation"])
                expected_group_counts = action.get("expected_processor_group_counts")
                if expected_group_counts != PLADIS_PINNED_SDXL_GROUP_COUNTS:
                    raise ValueError(
                        f"{action_id}: expected_processor_group_counts must be "
                        f"{PLADIS_PINNED_SDXL_GROUP_COUNTS}"
                    )
                expected_processor_count = action.get("expected_processor_count")
                if (
                    isinstance(expected_processor_count, bool)
                    or expected_processor_count != PLADIS_PINNED_SDXL_PROCESSOR_COUNT
                ):
                    raise ValueError(
                        f"{action_id}: expected_processor_count must be "
                        f"{PLADIS_PINNED_SDXL_PROCESSOR_COUNT}"
                    )
                if (
                    action.get("expected_processor_names_sha256")
                    != PLADIS_PINNED_SDXL_PROCESSOR_NAMES_SHA256
                ):
                    raise ValueError(
                        f"{action_id}: expected_processor_names_sha256 differs from "
                        "the pinned SDXL topology"
                    )
                action["expected_processor_group_counts"] = dict(
                    expected_group_counts
                )
                action["expected_processor_count"] = expected_processor_count
                action["expected_processor_names_sha256"] = (
                    PLADIS_PINNED_SDXL_PROCESSOR_NAMES_SHA256
                )
                if action.get("attention_mask_policy") != "none":
                    raise ValueError(
                        f"{action_id}: pinned attention baseline requires "
                        "attention_mask_policy 'none'"
                    )
                if baseline == "pladis":
                    if implementation != PLADIS_OPERATOR_PORT_IMPLEMENTATION:
                        raise ValueError(
                            f"{action_id}: unsupported PLADIS implementation "
                            f"{implementation!r}"
                        )
                    if action.get("source_commit") != PLADIS_SOURCE_COMMIT:
                        raise ValueError(
                            f"{action_id}: pinned PLADIS must bind source_commit "
                            f"{PLADIS_SOURCE_COMMIT}"
                        )
                    if tuple(action.get("applied_layers", ())) != PLADIS_PINNED_SDXL_LAYERS:
                        raise ValueError(
                            f"{action_id}: pinned PLADIS applied_layers must be "
                            f"{list(PLADIS_PINNED_SDXL_LAYERS)}"
                        )
                    if action.get("probability_dtype") != PLADIS_PINNED_PROBABILITY_DTYPE:
                        raise ValueError(
                            f"{action_id}: pinned PLADIS probability_dtype must be "
                            f"{PLADIS_PINNED_PROBABILITY_DTYPE!r}"
                        )
                    runtime_diffusers = str(getattr(diffusers, "__version__", "unknown"))
                    if runtime_diffusers != PLADIS_PORT_DIFFUSERS_VERSION:
                        raise ValueError(
                            f"{action_id}: PLADIS operator port requires diffusers "
                            f"{PLADIS_PORT_DIFFUSERS_VERSION}, got {runtime_diffusers}; "
                            "the upstream PLADIS repository itself binds 0.33.1"
                        )
                else:
                    if implementation != GAG_EQ13_IMPLEMENTATION:
                        raise ValueError(
                            f"{action_id}: unsupported GAG implementation "
                            f"{implementation!r}"
                        )
                    required_provenance = {
                        "paper_id": GAG_PAPER_ID,
                        "official_code_available": False,
                        "implementation_origin": "independent_reimplementation",
                        "equations": list(GAG_PAPER_EQUATIONS),
                        "alpha_source": "inferred_from_pladis",
                        "layer_policy_source": "inherited_from_pladis",
                        "paper_license": "CC-BY-4.0",
                        "software_license": "Apache-2.0",
                    }
                    for field, expected in required_provenance.items():
                        if action.get(field) != expected:
                            raise ValueError(
                                f"{action_id}: GAG provenance field {field} must be "
                                f"{expected!r}"
                            )
                    if tuple(action.get("applied_layers", ())) != GAG_INHERITED_LAYER_POLICY:
                        raise ValueError(
                            f"{action_id}: GAG inherited applied_layers must be "
                            f"{list(GAG_INHERITED_LAYER_POLICY)}"
                        )
                    if action.get("probability_dtype") != GAG_INHERITED_PROBABILITY_DTYPE:
                        raise ValueError(
                            f"{action_id}: GAG inherited probability_dtype must be "
                            f"{GAG_INHERITED_PROBABILITY_DTYPE!r}"
                        )
                    if (
                        action["baseline_scale"] != 10.0
                        or action["eta"] != 15.0
                        or action["zeta"] != 0.0
                    ):
                        raise ValueError(
                            f"{action_id}: GAG Eq. 13 reproduction requires "
                            "lambda=10, eta=15, zeta=0"
                        )
                action["implementation"] = implementation
            action["scale"] = 0.0
        elif action_type == "latent_renderer_fixed":
            coefficients = [float(value) for value in action.get("coefficients", [])]
            if len(coefficients) != len(LAZY_LATENT_STRUCTURE_BASIS_NAMES) or not all(
                math.isfinite(value) for value in coefficients
            ):
                raise ValueError(
                    f"{action_id}: latent_renderer_fixed requires six finite coefficients "
                    "in canonical basis order"
                )
            coefficient_bound = float(
                action.get("coefficient_bound", provider_defaults.get("coefficient_bound", 1.0))
            )
            if not math.isfinite(coefficient_bound) or coefficient_bound <= 0:
                raise ValueError(f"{action_id}: coefficient_bound must be positive")
            if any(abs(value) >= coefficient_bound for value in coefficients):
                raise ValueError(
                    f"{action_id}: coefficients must be strictly inside coefficient_bound"
                )
            provider = dict(provider_defaults)
            raw_provider_overrides = action.get("provider")
            if raw_provider_overrides is None and "provider" in action:
                raise ValueError(f"{action_id}: provider cannot be null")
            provider_overrides = raw_provider_overrides or {}
            if not isinstance(provider_overrides, dict):
                raise ValueError(f"{action_id}: provider must be a mapping")
            for mapping, label in (
                (provider_defaults, "defaults"),
                (provider_overrides, "provider"),
                (action, "action"),
            ):
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

            def _provider_ids(mapping, label):
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

            default_provider_id_candidates = _provider_ids(provider_defaults, "defaults")
            nested_provider_id_candidates = _provider_ids(provider_overrides, "provider")
            action_provider_id_candidates = _provider_ids(action, "action")
            provider.update(provider_overrides)
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
            semantic_mode = str(provider.get("semantic_mode", "reciprocal_semantic"))
            if semantic_mode not in {
                "clean_tfsa",
                "reciprocal_latent",
                "reciprocal_semantic",
                "reciprocal_semantic_permuted",
            }:
                raise ValueError(f"{action_id}: unsupported latent renderer semantic_mode")
            semantic_topk = int(provider.get("semantic_topk", 16))
            if semantic_topk <= 0:
                raise ValueError(f"{action_id}: semantic_topk must be positive")
            prompt_dim = int(provider.get("prompt_dim", 0))
            state_dim = int(provider.get("state_dim", 0))
            if prompt_dim != 0 or state_dim != 0:
                raise ValueError(
                    f"{action_id}: fixed renderer actions require prompt_dim=state_dim=0"
                )
            feature_block = str(provider.get("feature_block", "up_blocks.0"))
            semantic_layer = provider.get(
                "semantic_layer",
                "up_blocks.0.attentions.0.transformer_blocks.0.attn1",
            )
            semantic_layer = None if semantic_layer is None else str(semantic_layer)
            # Basis and hook requests are accepted at action level for compact
            # grids, with the provider mapping retained as a shared default.
            requested_bases_raw = action.get(
                "requested_bases", provider.get("requested_bases")
            )
            try:
                requested_bases = normalize_latent_structure_bases(
                    requested_bases_raw
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{action_id}: invalid requested_bases: {exc}"
                ) from exc
            expected_hook_names = latent_structure_required_hook_names(
                requested_bases,
                semantic_layer=semantic_layer,
                feature_block=feature_block,
            )
            raw_hook_names = action.get(
                "required_hook_names", provider.get("required_hook_names")
            )
            if raw_hook_names is None:
                required_hook_names = expected_hook_names
            else:
                if isinstance(raw_hook_names, str) or not isinstance(
                    raw_hook_names, (list, tuple)
                ):
                    raise ValueError(
                        f"{action_id}: required_hook_names must be a list"
                    )
                required_hook_names = tuple(str(value) for value in raw_hook_names)
                if required_hook_names != expected_hook_names:
                    raise ValueError(
                        f"{action_id}: required_hook_names do not match requested_bases"
                    )
            implementation = str(
                action.get(
                    "implementation",
                    provider.get("implementation", LAZY_LATENT_STRUCTURE_PROVIDER_ID),
                )
            )
            implementation = normalize_latent_structure_provider_implementation(
                implementation
            )
            if implementation not in {
                LAZY_LATENT_STRUCTURE_PROVIDER_ID,
                "structural_unet_basis_v1",
            }:
                raise ValueError(f"{action_id}: unsupported provider implementation")
            if (
                implementation == "structural_unet_basis_v1"
                and tuple(requested_bases) != tuple(LAZY_LATENT_STRUCTURE_BASIS_NAMES)
            ):
                raise ValueError(
                    f"{action_id}: structural_unet_basis_v1 requires all six canonical "
                    "bases; use the lazy provider for requested subsets"
                )
            default_provider_id = (
                LAZY_LATENT_STRUCTURE_PROVIDER_ID
                if implementation == LAZY_LATENT_STRUCTURE_PROVIDER_ID
                else "structural_unet_basis_v1"
            )
            provider_id_candidates = (
                action_provider_id_candidates
                if action_provider_id_candidates
                else nested_provider_id_candidates
                if nested_provider_id_candidates
                else default_provider_id_candidates
            )
            if any(not value for value in provider_id_candidates):
                raise ValueError(
                    f"{action_id}: provider provenance id must be non-empty"
                )
            if provider_id_candidates and any(
                value != provider_id_candidates[0]
                for value in provider_id_candidates[1:]
            ):
                raise ValueError(f"{action_id}: provider provenance id fields disagree")
            provider_id = (
                provider_id_candidates[0]
                if provider_id_candidates
                else default_provider_id
            )
            scheduler_mapping = str(
                provider.get("scheduler_mapping", "legacy_unit")
            )
            if scheduler_mapping not in LATENT_RENDERER_SCHEDULER_MAPPINGS:
                raise ValueError(
                    f"{action_id}: scheduler_mapping must be legacy_unit or "
                    "euler_clean_endpoint"
                )
            basis_normalization = str(
                provider.get("basis_normalization", "legacy_l2_to_rms")
            )
            if basis_normalization not in LATENT_RENDERER_BASIS_NORMALIZATIONS:
                raise ValueError(
                    f"{action_id}: basis_normalization must be legacy_l2_to_rms "
                    "or match_rms"
                )
            action["coefficients"] = coefficients
            action["coefficient_bound"] = coefficient_bound
            action["max_update_ratio"] = float(
                action.get("max_update_ratio", 0.05)
            )
            if (
                not math.isfinite(action["max_update_ratio"])
                or action["max_update_ratio"] < 0
            ):
                raise ValueError(f"{action_id}: max_update_ratio must be non-negative")
            action["latent_renderer_provider"] = {
                "feature_block": feature_block,
                "semantic_layer": semantic_layer,
                "semantic_mode": semantic_mode,
                "semantic_topk": semantic_topk,
                "permutation_seed": int(provider.get("permutation_seed", 1729)),
                "prompt_dim": prompt_dim,
                "state_dim": state_dim,
                "provider_id": provider_id,
                "provider_provenance_id": provider_id,
                "implementation": implementation,
                "requested_bases": list(requested_bases),
                "required_hook_names": list(required_hook_names),
                "scheduler_mapping": scheduler_mapping,
                "basis_normalization": basis_normalization,
            }
            action["scale"] = 0.0
        elif action_type == "freeu":
            if "parameters" in action and "knots" in action:
                raise ValueError(f"{action_id}: provide either parameters or knots, not both")
            if "parameters" in action:
                knots = ((0.0, action["parameters"]), (1.0, action["parameters"]))
            else:
                raw_knots = action.get("knots")
                if raw_knots is None and isinstance(action.get("schedule"), dict):
                    raw_knots = action["schedule"].get("knots")
                if not isinstance(raw_knots, list):
                    raise ValueError(f"{action_id}: freeu requires parameters or a knots list")
                knots = []
                for raw_knot in raw_knots:
                    if not isinstance(raw_knot, dict):
                        raise ValueError(f"{action_id}: each FreeU knot must be a mapping")
                    knots.append((raw_knot.get("position"), raw_knot.get("parameters")))
            try:
                schedule = FreeUSchedule(knots)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{action_id}: invalid FreeU schedule: {exc}") from exc
            preserve_moments = action.get("preserve_moments", False)
            if not isinstance(preserve_moments, bool):
                raise ValueError(f"{action_id}: preserve_moments must be a boolean")
            implementation = str(
                action.get("implementation", DIFFUSERS_FREEU_IMPLEMENTATION)
            )
            if implementation not in FREEU_IMPLEMENTATIONS:
                raise ValueError(
                    f"{action_id}: unsupported FreeU implementation {implementation!r}"
                )
            if implementation == PAPER_FREEU_IMPLEMENTATION:
                if preserve_moments:
                    raise ValueError(
                        f"{action_id}: paper FreeU cannot be combined with moment preservation"
                    )
                if action.get("source_commit") != PAPER_FREEU_SOURCE_COMMIT:
                    raise ValueError(
                        f"{action_id}: paper FreeU must bind source_commit "
                        f"{PAPER_FREEU_SOURCE_COMMIT}"
                    )
                if (
                    action.get("implementation_diffusers_version")
                    != PAPER_FREEU_PORT_DIFFUSERS_VERSION
                ):
                    raise ValueError(
                        f"{action_id}: paper FreeU must bind diffusers "
                        f"{PAPER_FREEU_PORT_DIFFUSERS_VERSION}"
                    )
            if "implementation" in action:
                action["implementation"] = implementation
            expected_calls = action.get("expected_operator_calls_per_step")
            expected_resolution_counts = action.get(
                "expected_resolution_idx_call_counts_per_step"
            )
            expected_channel_counts = action.get(
                "expected_hidden_channel_call_counts_per_step"
            )
            expected_joint_counts = action.get(
                "expected_resolution_channel_call_counts_per_step"
            )
            expected_effect_counts = action.get(
                "expected_operator_effect_call_counts_per_step"
            )
            topology_values = (
                expected_calls,
                expected_resolution_counts,
                expected_channel_counts,
                expected_joint_counts,
                expected_effect_counts,
            )
            if any(value is None for value in topology_values) and any(
                value is not None for value in topology_values
            ):
                raise ValueError(
                    f"{action_id}: FreeU call topology fields must be provided together"
                )
            if expected_calls is not None:
                if (
                    isinstance(expected_calls, bool)
                    or not isinstance(expected_calls, int)
                    or expected_calls <= 0
                    or not isinstance(expected_resolution_counts, list)
                    or not expected_resolution_counts
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value <= 0
                        for value in expected_resolution_counts
                    )
                    or sum(expected_resolution_counts) != expected_calls
                    or not isinstance(expected_channel_counts, dict)
                    or not isinstance(expected_joint_counts, dict)
                    or not isinstance(expected_effect_counts, dict)
                    or any(
                        not isinstance(key, str)
                        or isinstance(value, bool)
                        or not isinstance(value, int)
                        or value <= 0
                        for mapping in (
                            expected_channel_counts,
                            expected_joint_counts,
                            expected_effect_counts,
                        )
                        for key, value in mapping.items()
                    )
                    or any(
                        sum(mapping.values()) != expected_calls
                        for mapping in (
                            expected_channel_counts,
                            expected_joint_counts,
                            expected_effect_counts,
                        )
                    )
                ):
                    raise ValueError(f"{action_id}: invalid FreeU call topology")
            action["freeu_schedule"] = schedule.to_record()
            action["freeu_preserve_moments"] = preserve_moments
            action.pop("parameters", None)
            action.pop("knots", None)
            action.pop("schedule", None)
            action.pop("preserve_moments", None)
            action["scale"] = 0.0
        elif action_type == "trajectory_correction":
            mix = float(action.get("mix", -1.0))
            if not math.isfinite(mix) or not 0.0 <= mix <= 1.0:
                raise ValueError(f"{action_id}: mix must be finite and in [0, 1]")
            noise_mode = str(action.get("noise_mode", "sqrt"))
            if noise_mode not in {"sqrt", "linear", "none"}:
                raise ValueError(
                    f"{action_id}: noise_mode must be sqrt, linear, or none"
                )
            max_ratio = action.get("max_correction_ratio")
            if max_ratio is not None:
                max_ratio = float(max_ratio)
                if not math.isfinite(max_ratio) or max_ratio < 0:
                    raise ValueError(
                        f"{action_id}: max_correction_ratio must be finite and non-negative"
                    )
            action["mix"] = mix
            action["noise_mode"] = noise_mode
            action["max_correction_ratio"] = max_ratio
            action["scale"] = 0.0
        elif action_type == "scheduler_baseline":
            scheduler_class = str(
                action.get("scheduler_class", "EulerAncestralDiscreteScheduler")
            )
            if scheduler_class not in SCHEDULER_BASELINE_CLASSES:
                raise ValueError(
                    f"{action_id}: unsupported scheduler baseline {scheduler_class!r}; "
                    f"choose one of {sorted(SCHEDULER_BASELINE_CLASSES)}"
                )
            supplied_kwargs = action.get("scheduler_kwargs", {}) or {}
            if not isinstance(supplied_kwargs, dict):
                raise ValueError(f"{action_id}: scheduler_kwargs must be a mapping")
            unknown_kwargs = set(supplied_kwargs) - SCHEDULER_BASELINE_ALLOWED_KWARGS[
                scheduler_class
            ]
            if unknown_kwargs:
                raise ValueError(
                    f"{action_id}: unsupported {scheduler_class} kwargs: "
                    f"{sorted(unknown_kwargs)}"
                )
            scheduler_kwargs = dict(SCHEDULER_BASELINE_DEFAULT_KWARGS[scheduler_class])
            scheduler_kwargs.update(supplied_kwargs)
            if scheduler_class == "DPMSolverMultistepScheduler":
                _require_int(
                    scheduler_kwargs["solver_order"],
                    action_id=action_id,
                    field="DPM++ solver_order",
                    expected=2,
                )
                if not isinstance(scheduler_kwargs["algorithm_type"], str):
                    raise ValueError(f"{action_id}: DPM++ algorithm_type must be a string")
                if scheduler_kwargs["algorithm_type"] != "dpmsolver++":
                    raise ValueError(f"{action_id}: only deterministic dpmsolver++ is registered")
                if not isinstance(scheduler_kwargs["solver_type"], str):
                    raise ValueError(f"{action_id}: DPM++ solver_type must be a string")
                if scheduler_kwargs["solver_type"] not in {"midpoint", "heun"}:
                    raise ValueError(f"{action_id}: DPM++ solver_type must be midpoint or heun")
                if not isinstance(scheduler_kwargs["lower_order_final"], bool):
                    raise ValueError(f"{action_id}: lower_order_final must be boolean")
            elif scheduler_class == "UniPCMultistepScheduler":
                _require_int(
                    scheduler_kwargs["solver_order"],
                    action_id=action_id,
                    field="UniPC solver_order",
                    expected=2,
                )
                if not isinstance(scheduler_kwargs["predict_x0"], bool):
                    raise ValueError(f"{action_id}: predict_x0 must be boolean")
                if not isinstance(scheduler_kwargs["solver_type"], str):
                    raise ValueError(f"{action_id}: UniPC solver_type must be a string")
                if scheduler_kwargs["solver_type"] not in {"bh1", "bh2"}:
                    raise ValueError(f"{action_id}: UniPC solver_type must be bh1 or bh2")
                if not isinstance(scheduler_kwargs["lower_order_final"], bool):
                    raise ValueError(f"{action_id}: lower_order_final must be boolean")
            action["scheduler_class"] = scheduler_class
            action["scheduler_kwargs"] = scheduler_kwargs
            # Reference actions are reported but never selected by the fixed
            # correction gate unless explicitly marked eligible.
            action["selection_eligible"] = bool(action.get("selection_eligible", False))
            action["scale"] = 0.0

        residual_mode = str(action.get("residual_mode", "raw"))
        if residual_mode not in RESIDUAL_MODES:
            raise ValueError(f"{action_id}: unsupported residual_mode {residual_mode!r}")
        if action_type != "scalar" and residual_mode != "raw":
            raise ValueError(f"{action_id}: only scalar actions support non-raw residual modes")
        action["residual_mode"] = residual_mode

        if "cfg_scale" in action:
            cfg_scale = float(action["cfg_scale"])
            if not math.isfinite(cfg_scale) or cfg_scale < 0:
                raise ValueError(f"{action_id}: cfg_scale must be finite and non-negative")
            action["cfg_scale"] = cfg_scale

        delay_steps = int(action.get("delay_steps", 0))
        if not 0 <= delay_steps < num_inference_steps:
            raise ValueError(f"{action_id}: delay_steps must be in [0, {num_inference_steps})")
        action["delay_steps"] = delay_steps
        if "decay" in action and action["decay"] is not None:
            if len(action["decay"]) != 3:
                raise ValueError(f"{action_id}: decay must have three values")
            action["decay"] = list(action["decay"])
        if "max_update_ratio" in action and action["max_update_ratio"] is not None:
            action["max_update_ratio"] = float(action["max_update_ratio"])
            if action["max_update_ratio"] < 0:
                raise ValueError(f"{action_id}: max_update_ratio must be non-negative")
            if residual_mode in MOMENT_TANGENT_MODES:
                raise ValueError(
                    f"{action_id}: max_update_ratio is not defined for moment-tangent updates"
                )
        normalized.append(action)

    cutoffs = resolve_frequency_band_cutoffs(config)
    return normalized, cutoffs


def validate_split_seed_role(path: str, split_role, seeds) -> None:
    """Enforce registered search/validation/test seeds when a YAML defines them."""
    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    registered = config.get("split_seeds")
    if registered is None:
        if split_role is not None:
            raise ValueError("--split_role requires an action config with split_seeds")
        return
    if not isinstance(registered, dict) or not registered:
        raise ValueError("split_seeds must be a non-empty mapping")
    if split_role is None:
        raise ValueError(
            "this action config registers split_seeds; pass --split_role explicitly"
        )
    if split_role not in registered:
        raise ValueError(
            f"unknown --split_role {split_role!r}; expected one of {sorted(registered)}"
        )
    expected = [int(value) for value in registered[split_role]]
    if list(seeds) != expected:
        raise ValueError(
            f"--seeds {list(seeds)} do not match {split_role} registered seeds {expected}"
        )


def validate_scheduler_baseline_authorization(path: str) -> None:
    """Require an explicit human authorization for scheduler reference runs."""
    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    if config.get("schema") != "scheduler_baselines_v1":
        return
    if config.get("status") != "authorized":
        raise ValueError(
            "scheduler_baselines_v1 is CPU-audited but not authorized; "
            "a reviewer must freeze status: authorized before generation"
        )
    authorization = config.get("authorization")
    if not isinstance(authorization, dict):
        raise ValueError(
            "authorized scheduler baseline config requires an authorization mapping"
        )
    if not str(authorization.get("reviewer", "")).strip():
        raise ValueError("authorized scheduler baseline config requires authorization.reviewer")
    if not str(authorization.get("reviewed_commit", "")).strip():
        raise ValueError(
            "authorized scheduler baseline config requires authorization.reviewed_commit"
        )


def validate_cfg_baseline_authorization(
    path: str, *, repository_root: str | None = None
) -> None:
    """Bind CFG authorization to an immutable, reviewed Git template."""
    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    if config.get("schema") != CFG_BASELINE_SCHEMA:
        return
    if config.get("status") != "authorized":
        raise ValueError(
            "cfg_baselines_v1 is not authorized; a reviewer must freeze "
            "status: authorized before generation"
        )
    authorization = config.get("authorization")
    if not isinstance(authorization, dict):
        raise ValueError(
            "authorized CFG baseline config requires an authorization mapping"
        )
    if set(authorization) != CFG_BASELINE_AUTH_FIELDS:
        raise ValueError(
            "CFG authorization fields differ from the frozen authorization schema"
        )
    if not str(authorization.get("reviewer", "")).strip():
        raise ValueError(
            "authorized CFG baseline config requires authorization.reviewer"
        )
    if authorization.get("source_template") != CFG_BASELINE_AUTH_SOURCE_TEMPLATE:
        raise ValueError(
            "CFG authorization source_template must be "
            f"{CFG_BASELINE_AUTH_SOURCE_TEMPLATE}"
        )
    if authorization.get("scope") != CFG_BASELINE_AUTH_SCOPE:
        raise ValueError(
            f"CFG authorization scope must be {CFG_BASELINE_AUTH_SCOPE}"
        )
    reviewed_commit = str(authorization.get("reviewed_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", reviewed_commit):
        raise ValueError(
            "CFG authorization reviewed_commit must be a full 40-hex Git commit"
        )
    template_hash = str(authorization.get("source_template_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", template_hash):
        raise ValueError(
            "CFG authorization source_template_sha256 must be 64 lowercase hex"
        )

    repo = os.path.abspath(repository_root or ROOT)
    try:
        head_commit = subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            stderr=subprocess.PIPE,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot resolve repository HEAD for CFG authorization: {exc}") from exc
    ancestry = subprocess.run(
        ["git", "-C", repo, "merge-base", "--is-ancestor", reviewed_commit, head_commit],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if ancestry.returncode == 1:
        raise ValueError("CFG authorization reviewed_commit is not an ancestor of HEAD")
    if ancestry.returncode != 0:
        detail = ancestry.stderr.strip() or "git merge-base failed"
        raise ValueError(f"cannot verify CFG authorization ancestry: {detail}")
    try:
        template_bytes = subprocess.check_output(
            [
                "git",
                "-C",
                repo,
                "show",
                f"{reviewed_commit}:{CFG_BASELINE_AUTH_SOURCE_TEMPLATE}",
            ],
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "cannot read the reviewed CFG source template from Git"
        ) from exc
    if hashlib.sha256(template_bytes).hexdigest() != template_hash:
        raise ValueError(
            "CFG authorization source_template_sha256 differs from reviewed Git bytes"
        )
    try:
        reviewed_template = yaml.safe_load(template_bytes) or {}
    except yaml.YAMLError as exc:
        raise ValueError("reviewed CFG source template is not valid YAML") from exc
    if not isinstance(reviewed_template, dict):
        raise ValueError("reviewed CFG source template must be a mapping")

    def unsigned_body(value: dict) -> dict:
        return {
            key: item
            for key, item in value.items()
            if key not in {"status", "authorization"}
        }

    if unsigned_body(config) != unsigned_body(reviewed_template):
        raise ValueError(
            "authorized CFG config differs from the reviewed source template"
        )


def validate_native_renderer_authorization(
    path: str, *, repository_root: str | None = None
) -> None:
    """Bind an executable native-renderer copy to its frozen registration.

    The registration YAML is deliberately not executable.  This check makes a
    separate reviewed copy explicit: the design body (prompts, actions,
    sampling and gates) must remain byte-equivalent after removing only the
    registration status/authorization metadata and the scorer contract.
    """
    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    if config.get("schema") != NATIVE_RENDERER_SCHEMA:
        return
    if config.get("status") != "authorized_development":
        raise ValueError(
            f"{NATIVE_RENDERER_SCHEMA} is not authorized; reviewer must freeze "
            "status: authorized_development"
        )
    authorization = config.get("authorization")
    if not isinstance(authorization, dict):
        raise ValueError(
            "authorized native-renderer config requires an authorization mapping"
        )
    if set(authorization) != NATIVE_RENDERER_AUTH_FIELDS:
        raise ValueError(
            "native-renderer authorization fields differ from the frozen schema"
        )
    if not str(authorization.get("reviewer", "")).strip():
        raise ValueError("native-renderer authorization requires reviewer")
    if authorization.get("scope") != NATIVE_RENDERER_AUTH_SCOPE:
        raise ValueError(
            "native-renderer authorization scope must be "
            f"{NATIVE_RENDERER_AUTH_SCOPE}"
        )
    if authorization.get("source_template") != NATIVE_RENDERER_AUTH_SOURCE_TEMPLATE:
        raise ValueError(
            "native-renderer authorization source_template must be "
            f"{NATIVE_RENDERER_AUTH_SOURCE_TEMPLATE}"
        )
    template_hash = str(authorization.get("source_template_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", template_hash):
        raise ValueError(
            "native-renderer source_template_sha256 must be 64 lowercase hex"
        )
    registration_source = config.get("registration_source")
    if not isinstance(registration_source, dict) or set(registration_source) != {
        "path",
        "sha256",
    }:
        raise ValueError(
            "authorized native-renderer config requires a frozen registration_source"
        )
    if registration_source["path"] != NATIVE_RENDERER_AUTH_SOURCE_TEMPLATE:
        raise ValueError(
            "native-renderer registration_source.path differs from source_template"
        )
    if not isinstance(registration_source["sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", registration_source["sha256"]
    ):
        raise ValueError("native-renderer registration_source.sha256 is invalid")
    if registration_source["sha256"] != template_hash:
        raise ValueError(
            "native-renderer registration_source hash differs from authorization"
        )
    if authorization.get("gpu_generation") is not True:
        raise ValueError("native-renderer development generation is not authorized")
    if authorization.get("scoring") is not True:
        raise ValueError("native-renderer development scoring is not authorized")
    if authorization.get("method_selection") is not False:
        raise ValueError(
            "native-renderer development authorization cannot authorize method selection"
        )
    reviewed_commit = str(authorization.get("reviewed_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", reviewed_commit):
        raise ValueError(
            "native-renderer reviewed_commit must be a full 40-hex Git commit"
        )
    scoring = config.get("scoring")
    if not isinstance(scoring, dict):
        raise ValueError("authorized native-renderer config requires scoring metadata")
    scorer_hash = scoring.get("registered_scorer_provenance_sha256")
    if not isinstance(scorer_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", scorer_hash):
        raise ValueError(
            "native-renderer scoring must register a scorer provenance SHA-256"
        )
    if scoring.get("required_schema") != NATIVE_RENDERER_SCORER_SCHEMA:
        raise ValueError(
            "native-renderer scoring must require the hardened scorer provenance schema"
        )

    repo = os.path.abspath(repository_root or ROOT)
    try:
        head_commit = subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            stderr=subprocess.PIPE,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            f"cannot resolve repository HEAD for native-renderer authorization: {exc}"
        ) from exc
    ancestry = subprocess.run(
        ["git", "-C", repo, "merge-base", "--is-ancestor", reviewed_commit, head_commit],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if ancestry.returncode == 1:
        raise ValueError(
            "native-renderer reviewed_commit is not an ancestor of HEAD"
        )
    if ancestry.returncode != 0:
        detail = ancestry.stderr.strip() or "git merge-base failed"
        raise ValueError(f"cannot verify native-renderer authorization ancestry: {detail}")

    template_path = os.path.join(repo, NATIVE_RENDERER_AUTH_SOURCE_TEMPLATE)
    try:
        with open(template_path, "rb") as handle:
            template_bytes = handle.read()
        reviewed_bytes = subprocess.check_output(
            [
                "git",
                "-C",
                repo,
                "show",
                f"{reviewed_commit}:{NATIVE_RENDERER_AUTH_SOURCE_TEMPLATE}",
            ],
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "cannot read the reviewed native-renderer source template"
        ) from exc
    if hashlib.sha256(template_bytes).hexdigest() != template_hash:
        raise ValueError(
            "native-renderer source_template_sha256 differs from current Git bytes"
        )
    if hashlib.sha256(reviewed_bytes).hexdigest() != template_hash:
        raise ValueError(
            "native-renderer source_template_sha256 differs from reviewed Git bytes"
        )
    try:
        reviewed_template = yaml.safe_load(reviewed_bytes) or {}
    except yaml.YAMLError as exc:
        raise ValueError("reviewed native-renderer source template is invalid YAML") from exc
    if not isinstance(reviewed_template, dict):
        raise ValueError("reviewed native-renderer source template must be a mapping")

    def design_body(value: dict) -> dict:
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
            provider.pop("implementation_status", None)
        return body

    if design_body(config) != design_body(reviewed_template):
        raise ValueError(
            "authorized native-renderer config differs from the frozen source template"
        )
    provider = config.get("required_provider")
    if not isinstance(provider, dict) or provider.get("implementation_status") != "implemented":
        raise ValueError(
            "authorized native-renderer config must record the implemented lazy provider"
        )


def validate_structural_control_authorization(
    path: str,
    *,
    repository_root: str | None = None,
    require_clean: bool = True,
    verify_current_source: bool = True,
) -> None:
    """Bind an executable structural-control matrix to reviewed source bytes."""
    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    if config.get("schema") != STRUCTURAL_CONTROL_SCHEMA:
        return
    if config.get("registration_schema") != STRUCTURAL_CONTROL_REGISTRATION_SCHEMA:
        raise ValueError("structural-control registration_schema differs")
    if config.get("status") != "authorized_development":
        raise ValueError("structural-control executable is not authorized_development")

    authorization = config.get("authorization")
    if not isinstance(authorization, dict) or set(
        authorization
    ) != STRUCTURAL_CONTROL_AUTH_FIELDS:
        raise ValueError(
            "structural-control authorization fields differ from the frozen schema"
        )
    if not str(authorization.get("reviewer", "")).strip():
        raise ValueError("structural-control authorization requires reviewer")
    if authorization.get("scope") != STRUCTURAL_CONTROL_AUTH_SCOPE:
        raise ValueError("structural-control authorization scope differs")
    if authorization.get("source_template") != STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE:
        raise ValueError("structural-control authorization source_template differs")
    if authorization.get("gpu_generation") is not True:
        raise ValueError("structural-control generation is not authorized")
    if authorization.get("scoring") is not True:
        raise ValueError("structural-control scoring is not authorized")
    if authorization.get("method_selection") is not False:
        raise ValueError("structural controls cannot authorize method selection")
    if authorization.get("result_access_before_freeze") is not False:
        raise ValueError("structural controls must remain result-blind before freeze")
    reviewed_commit = str(authorization.get("reviewed_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", reviewed_commit):
        raise ValueError("structural-control reviewed_commit must be full 40-hex")
    template_hash = str(authorization.get("source_template_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", template_hash):
        raise ValueError("structural-control source_template_sha256 is invalid")

    registration_source = config.get("registration_source")
    if not isinstance(registration_source, dict) or set(registration_source) != {
        "path",
        "sha256",
    }:
        raise ValueError("structural-control registration_source is invalid")
    if registration_source != {
        "path": STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE,
        "sha256": template_hash,
    }:
        raise ValueError("structural-control registration_source differs from authorization")

    scoring = config.get("scoring")
    if not isinstance(scoring, dict) or set(scoring) != {
        "required_schema",
        "registered_scorer_provenance_sha256",
    }:
        raise ValueError("structural-control scoring contract is invalid")
    if scoring.get("required_schema") != NATIVE_RENDERER_SCORER_SCHEMA or not re.fullmatch(
        r"[0-9a-f]{64}",
        str(scoring.get("registered_scorer_provenance_sha256", "")),
    ):
        raise ValueError("structural-control scorer provenance binding is invalid")

    implementation_source = config.get("implementation_source")
    if not isinstance(implementation_source, dict) or set(implementation_source) != {
        "reviewed_commit",
        "files",
    }:
        raise ValueError("structural-control implementation_source is invalid")
    if implementation_source.get("reviewed_commit") != reviewed_commit:
        raise ValueError("implementation_source reviewed_commit differs from authorization")
    source_files = implementation_source.get("files")
    if not isinstance(source_files, dict) or set(source_files) != set(
        STRUCTURAL_CONTROL_IMPLEMENTATION_PATHS
    ):
        raise ValueError("structural-control implementation source paths differ")
    if any(
        not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
        for digest in source_files.values()
    ):
        raise ValueError("structural-control implementation source hash is invalid")

    analysis_implementation = config.get("analysis_implementation")
    if not isinstance(analysis_implementation, dict) or set(
        analysis_implementation
    ) != {"schema", "files"}:
        raise ValueError("structural-control analysis_implementation is invalid")
    if analysis_implementation.get("schema") != STRUCTURAL_CONTROL_ANALYSIS_SCHEMA:
        raise ValueError("structural-control analysis implementation schema differs")
    analysis_files = analysis_implementation.get("files")
    if not isinstance(analysis_files, dict) or set(analysis_files) != set(
        STRUCTURAL_CONTROL_ANALYSIS_PATHS
    ):
        raise ValueError("structural-control analysis implementation paths differ")
    if any(
        not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
        for digest in analysis_files.values()
    ):
        raise ValueError("structural-control analysis implementation hash is invalid")

    repo = os.path.realpath(repository_root or ROOT)
    executable_path = os.path.realpath(path)
    try:
        if os.path.commonpath((repo, executable_path)) != repo:
            raise ValueError(
                "structural-control executable must be inside the reviewed repository"
            )
    except ValueError as exc:
        raise ValueError(
            "structural-control executable must be inside the reviewed repository"
        ) from exc
    executable_relative_path = os.path.relpath(executable_path, repo)
    try:
        head_commit = subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            stderr=subprocess.PIPE,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot resolve repository HEAD: {exc}") from exc
    ancestry = subprocess.run(
        ["git", "-C", repo, "merge-base", "--is-ancestor", reviewed_commit, head_commit],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if ancestry.returncode == 1:
        raise ValueError("structural-control reviewed_commit is not an ancestor of HEAD")
    if ancestry.returncode != 0:
        raise ValueError(
            "cannot verify structural-control reviewed commit: "
            f"{ancestry.stderr.strip() or 'git merge-base failed'}"
        )
    try:
        subprocess.check_output(
            [
                "git",
                "-C",
                repo,
                "ls-files",
                "--error-unmatch",
                "--",
                executable_relative_path,
            ],
            stderr=subprocess.PIPE,
        )
        committed_executable_bytes = subprocess.check_output(
            [
                "git",
                "-C",
                repo,
                "show",
                f"{head_commit}:{executable_relative_path}",
            ],
            stderr=subprocess.PIPE,
        )
        with open(executable_path, "rb") as handle:
            current_executable_bytes = handle.read()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "structural-control executable must be tracked at repository HEAD"
        ) from exc
    if current_executable_bytes != committed_executable_bytes:
        raise ValueError(
            "structural-control executable bytes differ from repository HEAD"
        )
    if require_clean and not verify_current_source:
        raise ValueError("clean-tree validation requires current-source verification")
    if require_clean:
        try:
            dirty = subprocess.check_output(
                [
                    "git",
                    "-C",
                    repo,
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                ],
                stderr=subprocess.PIPE,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(f"cannot verify clean source tree: {exc}") from exc
        if dirty:
            raise ValueError(
                "structural-control generation requires a clean repository worktree"
            )

    try:
        reviewed_template_bytes = subprocess.check_output(
            [
                "git",
                "-C",
                repo,
                "show",
                f"{reviewed_commit}:{STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE}",
            ],
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("cannot read reviewed structural-control template") from exc
    if verify_current_source:
        template_path = os.path.join(repo, STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE)
        try:
            with open(template_path, "rb") as handle:
                current_template_bytes = handle.read()
        except OSError as exc:
            raise ValueError("cannot read current structural-control template") from exc
        if hashlib.sha256(current_template_bytes).hexdigest() != template_hash:
            raise ValueError("current structural-control template hash differs")
    if hashlib.sha256(reviewed_template_bytes).hexdigest() != template_hash:
        raise ValueError("reviewed structural-control template hash differs")
    try:
        reviewed_template = yaml.safe_load(reviewed_template_bytes) or {}
    except yaml.YAMLError as exc:
        raise ValueError("reviewed structural-control template is invalid YAML") from exc

    def design_body(value: dict) -> dict:
        body = copy.deepcopy(value)
        for key in (
            "schema",
            "status",
            "authorization",
            "blocking_conditions",
            "registration_source",
            "implementation_source",
        ):
            body.pop(key, None)
        return body

    if design_body(config) != design_body(reviewed_template):
        raise ValueError(
            "authorized structural-control config differs from the frozen template"
        )
    for relative_path in STRUCTURAL_CONTROL_IMPLEMENTATION_PATHS:
        expected_hash = source_files[relative_path]
        current_path = os.path.join(repo, relative_path)
        try:
            reviewed_bytes = subprocess.check_output(
                ["git", "-C", repo, "show", f"{reviewed_commit}:{relative_path}"],
                stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(
                f"cannot read reviewed implementation source {relative_path}"
            ) from exc
        if verify_current_source:
            try:
                with open(current_path, "rb") as handle:
                    current_bytes = handle.read()
            except OSError as exc:
                raise ValueError(
                    f"cannot read current implementation source {relative_path}"
                ) from exc
            if hashlib.sha256(current_bytes).hexdigest() != expected_hash:
                raise ValueError(
                    f"current implementation source hash differs for {relative_path}"
                )
        if hashlib.sha256(reviewed_bytes).hexdigest() != expected_hash:
            raise ValueError(
                f"reviewed implementation source hash differs for {relative_path}"
            )
    for relative_path in STRUCTURAL_CONTROL_ANALYSIS_PATHS:
        expected_hash = analysis_files[relative_path]
        current_path = os.path.join(repo, relative_path)
        try:
            reviewed_bytes = subprocess.check_output(
                ["git", "-C", repo, "show", f"{reviewed_commit}:{relative_path}"],
                stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(
                f"cannot read reviewed analysis source {relative_path}"
            ) from exc
        if verify_current_source:
            try:
                with open(current_path, "rb") as handle:
                    current_bytes = handle.read()
            except OSError as exc:
                raise ValueError(
                    f"cannot read current analysis source {relative_path}"
                ) from exc
            if hashlib.sha256(current_bytes).hexdigest() != expected_hash:
                raise ValueError(
                    f"current analysis source hash differs for {relative_path}"
                )
        if hashlib.sha256(reviewed_bytes).hexdigest() != expected_hash:
            raise ValueError(
                f"reviewed analysis source hash differs for {relative_path}"
            )


def validate_structural_control_design(
    path: str,
    *,
    prompts_path: str,
    actions: list,
    seeds: list,
    model_name: str,
    resolution: int,
    num_inference_steps: int,
    guidance_scale: float,
    negative_prompt: str,
    power_calibrate: int,
    stage2_enabled: bool,
    split_role: str | None,
    low_vram: bool = False,
) -> None:
    """Reject any execution drift from the frozen structural-control matrix."""
    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    if config.get("schema") != STRUCTURAL_CONTROL_SCHEMA:
        return
    sampling = config.get("sampling")
    if not isinstance(sampling, dict) or set(sampling) != STRUCTURAL_CONTROL_SAMPLING_KEYS:
        raise ValueError("structural-control sampling fields differ from the schema")
    analysis_implementation = config.get("analysis_implementation")
    if not isinstance(analysis_implementation, dict) or set(
        analysis_implementation
    ) != {"schema", "files"}:
        raise ValueError("structural-control analysis_implementation is invalid")
    if analysis_implementation.get("schema") != STRUCTURAL_CONTROL_ANALYSIS_SCHEMA:
        raise ValueError("structural-control analysis implementation schema differs")
    analysis_files = analysis_implementation.get("files")
    if not isinstance(analysis_files, dict) or set(analysis_files) != set(
        STRUCTURAL_CONTROL_ANALYSIS_PATHS
    ) or any(
        not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
        for digest in analysis_files.values()
    ):
        raise ValueError("structural-control analysis implementation files differ")
    expected_sampling = {
        "model": str(model_name),
        "model_revision": CFG_BASELINE_MODEL_REVISION,
        "pipeline": "RepLDMSDXLPipeline",
        "resolution": int(resolution),
        "num_inference_steps": int(num_inference_steps),
        "default_cfg_scale": 7.5,
        "cfg_source": "action.cfg_scale",
        "cfg_pipeline_argument": "guidance_scale",
        "negative_prompt": DEFAULT_NEG,
        "power_calibrate": 0,
        "guidance_rescale": 0.0,
        "scheduler": "EulerDiscreteScheduler",
        "prediction_type": "epsilon",
        "scheduler_churn": 0.0,
        "initialization": "scheduler_native_init_sigma",
        "stage2": bool(stage2_enabled),
        "extra_unet_calls": 0,
        "torch_dtype": "float16",
        "variant": "fp16",
        "local_files_only": True,
        "low_vram": False,
        "batch_size": 1,
        "num_images_per_prompt": 1,
        "attention_mask_policy": "none",
    }
    for key, expected in expected_sampling.items():
        observed = sampling.get(key)
        if isinstance(expected, bool):
            same = isinstance(observed, bool) and observed is expected
        elif isinstance(expected, int):
            same = not isinstance(observed, bool) and isinstance(observed, int) and observed == expected
        elif isinstance(expected, float):
            same = (
                not isinstance(observed, bool)
                and isinstance(observed, (int, float))
                and float(observed) == expected
            )
        else:
            same = observed == expected
        if not same:
            raise ValueError(
                f"structural-control sampling.{key} differs: expected "
                f"{expected!r}, got {observed!r}"
            )
    if float(guidance_scale) != 7.5:
        raise ValueError("structural controls require --guidance_scale 7.5")
    if negative_prompt != DEFAULT_NEG:
        raise ValueError("structural controls require the registered negative prompt")
    if isinstance(power_calibrate, bool) or int(power_calibrate) != 0:
        raise ValueError("structural controls require --power_calibrate 0")
    if bool(low_vram):
        raise ValueError("structural controls require low_vram=False")
    if config.get("scheduler_runtime") != CFG_BASELINE_SCHEDULER_RUNTIME:
        raise ValueError("structural-control Euler runtime contract differs")
    if config.get("frequency_band_cutoffs") != [0.08, 0.25]:
        raise ValueError("structural-control frequency-band cutoffs differ")
    if config.get("execution_order") != CFG_BASELINE_EXECUTION_ORDER:
        raise ValueError("structural-control execution_order differs")
    if config.get("split_role") != "development":
        raise ValueError("structural controls must retain the development result role")
    if config.get("split_seeds") != STRUCTURAL_CONTROL_SPLIT_SEEDS:
        raise ValueError("structural-control split seeds differ")
    if config.get("engineering_smoke") != STRUCTURAL_CONTROL_ENGINEERING_SMOKE:
        raise ValueError("structural-control engineering smoke profile differs")
    if split_role not in STRUCTURAL_CONTROL_SPLIT_SEEDS:
        raise ValueError("structural controls require engineering_smoke or development")
    if list(seeds) != STRUCTURAL_CONTROL_SPLIT_SEEDS[split_role]:
        raise ValueError("structural-control CLI seeds differ")
    if config.get("failure_policy") != "shared_abort_after_first_task_error":
        raise ValueError("structural-control failure policy differs")

    source = config.get("source_manifest")
    expected_source = {
        "path": "eval-pipeline/prompts/scheduler_native_fixed_headroom_manifest.json",
        "sha256": "5373acf08f0e28d586732909f38787f8180a5d12c1ed58c2e1881134c10b6d5f",
        "prompts": STRUCTURAL_CONTROL_PROMPTS,
        "prompts_sha256": STRUCTURAL_CONTROL_PROMPTS_SHA256,
        "expected_prompt_count": 33,
        "expected_challenges": 11,
    }
    if source != expected_source:
        raise ValueError("structural-control source manifest registration differs")
    prompt_profile = (
        STRUCTURAL_CONTROL_ENGINEERING_SMOKE
        if split_role == "engineering_smoke"
        else expected_source
    )
    expected_prompts_path = os.path.abspath(
        os.path.join(ROOT, str(prompt_profile["prompts"]))
    )
    if os.path.abspath(prompts_path) != expected_prompts_path:
        raise ValueError("structural-control prompts path differs")
    if sha256_file(prompts_path) != prompt_profile["prompts_sha256"]:
        raise ValueError("structural-control prompts hash differs")
    manifest_path = os.path.abspath(os.path.join(ROOT, expected_source["path"]))
    if sha256_file(manifest_path) != expected_source["sha256"]:
        raise ValueError("structural-control source manifest hash differs")
    prompts = pd.read_csv(prompts_path)
    if (
        len(prompts) != int(prompt_profile["expected_prompt_count"])
        or prompts["source_challenge"].nunique()
        != int(prompt_profile["expected_challenges"])
    ):
        raise ValueError("structural-control prompt/challenge counts differ")

    expected_design = {
        "role": "development_only_baseline_calibration",
        "prompt_count": 33,
        "seed_count": 3,
        "action_count": 8,
        "expected_task_count": 792,
        "paired_block_policy": "one_prompt_seed_block_per_gpu",
        "method_selection": False,
        "confirmation": False,
        "reuse_scope": "same_development_split_as_headroom_screen",
    }
    if config.get("design") != expected_design:
        raise ValueError("structural-control design metadata differs")
    if [str(action.get("id")) for action in actions] != list(
        STRUCTURAL_CONTROL_ACTION_IDS
    ):
        raise ValueError("structural-control action IDs/order differ")
    expected_types = {
        "no_op_cfg7p5": "none",
        "cfg_only_5": "none",
        "conference_tfsa": "legacy",
        "freeu_diffusers_historical": "freeu",
        "freeu_diffusers_paper_parameters": "freeu",
        "freeu_paper_adaptive": "freeu",
        "pladis_operator_port": "attention_baseline",
        "gag_eq13_reimplementation": "attention_baseline",
    }
    expected_cfg = {action_id: 7.5 for action_id in STRUCTURAL_CONTROL_ACTION_IDS}
    expected_cfg.update(
        {
            "cfg_only_5": 5.0,
            "pladis_operator_port": 5.0,
            "gag_eq13_reimplementation": 5.0,
        }
    )
    for action in actions:
        action_id = str(action["id"])
        if action.get("type") != expected_types[action_id]:
            raise ValueError(f"{action_id}: structural-control action type differs")
        if float(action.get("cfg_scale", float("nan"))) != expected_cfg[action_id]:
            raise ValueError(f"{action_id}: structural-control cfg_scale differs")
    freeu_actions = [action for action in actions if action["type"] == "freeu"]
    if any(
        action.get("implementation_diffusers_version")
        != PAPER_FREEU_PORT_DIFFUSERS_VERSION
        for action in freeu_actions
    ):
        raise ValueError("structural-control FreeU diffusers binding differs")
    for action in freeu_actions:
        expected_effect_counts = (
            STRUCTURAL_CONTROL_FREEU_PAPER_EFFECT_COUNTS
            if action.get("implementation") == PAPER_FREEU_IMPLEMENTATION
            else STRUCTURAL_CONTROL_FREEU_CONSTANT_EFFECT_COUNTS
        )
        if (
            action.get("expected_operator_calls_per_step") != 9
            or action.get("expected_resolution_idx_call_counts_per_step")
            != STRUCTURAL_CONTROL_FREEU_RESOLUTION_COUNTS
            or action.get("expected_hidden_channel_call_counts_per_step")
            != STRUCTURAL_CONTROL_FREEU_CHANNEL_COUNTS
            or action.get("expected_resolution_channel_call_counts_per_step")
            != STRUCTURAL_CONTROL_FREEU_JOINT_COUNTS
            or action.get("expected_operator_effect_call_counts_per_step")
            != expected_effect_counts
        ):
            raise ValueError("structural-control FreeU call topology differs")

    analysis = config.get("analysis")
    if not isinstance(analysis, dict) or analysis.get("primary_metric") != "topiq_nr":
        raise ValueError("structural-control primary analysis differs")
    families = analysis.get("multiplicity_families")
    if not isinstance(families, dict) or set(families) != {
        "freeu_vs_no_op",
        "freeu_mechanism_contrasts",
        "attention_vs_cfg5",
    }:
        raise ValueError("structural-control multiplicity families differ")
    for family in families.values():
        if family.get("holm_alpha") != 0.05 or family.get(
            "point_estimate_screen_delta"
        ) != 0.005:
            raise ValueError("structural-control family threshold differs")


def validate_cfg_baseline_design(
    path: str,
    *,
    prompts_path: str,
    actions: list,
    seeds: list,
    model_name: str,
    resolution: int,
    num_inference_steps: int,
    guidance_scale: float,
    negative_prompt: str,
    power_calibrate: int,
    stage2_enabled: bool,
) -> None:
    """Reject drift from an authorized ordinary-CFG reference matrix."""
    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    if config.get("schema") != CFG_BASELINE_SCHEMA:
        return

    sampling = config.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("cfg_baselines_v1 requires a sampling mapping")
    if set(sampling) != CFG_BASELINE_SAMPLING_KEYS:
        missing = sorted(CFG_BASELINE_SAMPLING_KEYS - set(sampling))
        extra = sorted(set(sampling) - CFG_BASELINE_SAMPLING_KEYS)
        raise ValueError(
            "cfg_baselines_v1 sampling fields differ from the frozen schema: "
            f"missing={missing}, extra={extra}"
        )
    if float(guidance_scale) != 7.5:
        raise ValueError("cfg_baselines_v1 requires --guidance_scale 7.5")
    if negative_prompt != DEFAULT_NEG:
        raise ValueError("cfg_baselines_v1 requires the repository default negative prompt")
    if (
        isinstance(power_calibrate, bool)
        or not isinstance(power_calibrate, int)
        or power_calibrate != 0
    ):
        raise ValueError("cfg_baselines_v1 requires --power_calibrate 0")
    expected_sampling = {
        "model": str(model_name),
        "model_revision": CFG_BASELINE_MODEL_REVISION,
        "base_scheduler": "EulerDiscreteScheduler",
        "resolution": int(resolution),
        "num_inference_steps": int(num_inference_steps),
        "default_cfg_scale": 7.5,
        "negative_prompt": DEFAULT_NEG,
        "power_calibrate": 0,
        "stage2": bool(stage2_enabled),
        "extra_unet_calls": 0,
        "initialization": "scheduler_native_init_sigma",
        "cfg_source": "action.cfg_scale",
        "cfg_pipeline_argument": "guidance_scale",
        "guidance_rescale": 0.0,
    }
    for key, observed in expected_sampling.items():
        registered = sampling.get(key)
        if isinstance(observed, bool):
            same = isinstance(registered, bool) and registered is observed
        elif isinstance(observed, float):
            same = (
                not isinstance(registered, bool)
                and isinstance(registered, (int, float))
                and float(registered) == observed
            )
        elif isinstance(observed, int):
            same = (
                not isinstance(registered, bool)
                and isinstance(registered, int)
                and registered == observed
            )
        else:
            same = registered == observed
        if not same:
            raise ValueError(
                f"CFG baseline sampling.{key} differs: "
                f"registered {registered!r}, observed {observed!r}"
            )
    if config.get("scheduler_runtime") != CFG_BASELINE_SCHEDULER_RUNTIME:
        raise ValueError(
            "cfg_baselines_v1 scheduler_runtime differs from the frozen Euler contract"
        )

    source = config.get("source_manifest")
    if not isinstance(source, dict):
        raise ValueError("cfg_baselines_v1 requires a source_manifest mapping")
    registered_prompts = source.get("prompts")
    if not isinstance(registered_prompts, str) or not registered_prompts:
        raise ValueError("CFG baseline source_manifest.prompts is required")
    if registered_prompts != CFG_BASELINE_PROMPTS:
        raise ValueError(
            "CFG baseline prompt registration must use "
            f"{CFG_BASELINE_PROMPTS}"
        )
    registered_abs = os.path.abspath(os.path.join(ROOT, registered_prompts))
    actual_abs = os.path.abspath(prompts_path)
    if registered_abs != actual_abs:
        raise ValueError(
            "CFG baseline prompt path differs: "
            f"registered {registered_abs!r}, observed {actual_abs!r}"
        )
    if source.get("prompts_sha256") != CFG_BASELINE_PROMPTS_SHA256:
        raise ValueError("CFG baseline registered prompt SHA-256 is not frozen v1")
    if sha256_file(prompts_path) != CFG_BASELINE_PROMPTS_SHA256:
        raise ValueError("CFG baseline prompt SHA-256 differs from registration")

    split_seeds = config.get("split_seeds")
    if split_seeds != CFG_BASELINE_SPLIT_SEEDS:
        raise ValueError(
            "cfg_baselines_v1 split_seeds must be exactly development: [0, 42, 123]"
        )
    if list(seeds) != CFG_BASELINE_SPLIT_SEEDS["development"]:
        raise ValueError(
            f"CFG baseline seeds {list(seeds)} differ from frozen development seeds"
        )

    design = config.get("design")
    if not isinstance(design, dict):
        raise ValueError("cfg_baselines_v1 requires a design mapping")
    if config.get("execution_order") != CFG_BASELINE_EXECUTION_ORDER:
        raise ValueError("cfg_baselines_v1 execution_order differs from the contract")
    prompts = pd.read_csv(prompts_path)
    observed_counts = {
        "prompt_count": len(prompts),
        "action_count": len(actions),
        "seed_count": len(seeds),
        "task_count": len(prompts) * len(actions) * len(seeds),
        "block_count": len(prompts) * len(seeds),
    }
    count_fields = {
        "prompt_count": "expected_prompt_count",
        "action_count": "expected_action_count",
        "seed_count": "expected_seed_count",
        "task_count": "expected_task_count",
        "block_count": "expected_block_count",
    }
    for observed_name, registered_name in count_fields.items():
        registered = design.get(registered_name)
        if isinstance(registered, bool) or not isinstance(registered, int):
            raise ValueError(
                f"CFG baseline design.{registered_name} must be an integer"
            )
        if registered != observed_counts[observed_name]:
            raise ValueError(
                f"CFG baseline design {observed_name} differs: "
                f"registered {registered}, observed {observed_counts[observed_name]}"
            )

    expected_action_ids = list(CFG_BASELINE_ACTION_SCALES)
    registered_scales = design.get("cfg_scales")
    if not isinstance(registered_scales, list) or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in registered_scales
    ):
        raise ValueError("CFG baseline design.cfg_scales must be a numeric list")
    if [float(value) for value in registered_scales] != list(
        CFG_BASELINE_ACTION_SCALES.values()
    ):
        raise ValueError("CFG baseline design.cfg_scales differs from the frozen grid")
    if design.get("baseline_action_id") != "cfg_7p5":
        raise ValueError("CFG baseline design.baseline_action_id must be cfg_7p5")
    if design.get("action_ids") != expected_action_ids:
        raise ValueError(
            "CFG baseline registered action IDs differ from the frozen five-arm matrix"
        )
    observed_action_ids = [str(action.get("id")) for action in actions]
    if observed_action_ids != expected_action_ids:
        raise ValueError(
            "CFG baseline action IDs differ: "
            f"expected {expected_action_ids!r}, observed {observed_action_ids!r}"
        )
    for action in actions:
        action_id = str(action.get("id"))
        if action.get("type") != "none":
            raise ValueError(f"CFG baseline action {action_id!r} must use type none")
        if float(action.get("cfg_scale", float("nan"))) != CFG_BASELINE_ACTION_SCALES[
            action_id
        ]:
            raise ValueError(f"CFG baseline action {action_id!r} has wrong cfg_scale")

    expected_hashes = design.get("action_sha256")
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != set(
        expected_action_ids
    ):
        raise ValueError(
            "CFG baseline design.action_sha256 must exactly cover the five actions"
        )
    for action in actions:
        action_id = str(action["id"])
        if expected_hashes[action_id] != action_sha256(action):
            raise ValueError(f"CFG baseline action hash differs for {action_id!r}")


def validate_scheduler_baseline_design(
    path: str,
    *,
    prompts_path: str,
    actions: list,
    seeds: list,
    model_name: str,
    resolution: int,
    num_inference_steps: int,
    guidance_scale: float,
    stage2_enabled: bool,
) -> None:
    """Reject sampling, provenance, and design drift in a frozen manifest."""
    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    if config.get("schema") != "scheduler_baselines_v1":
        return
    sampling = config.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("scheduler_baselines_v1 requires a sampling mapping")
    expected_sampling = {
        "model": model_name,
        "base_scheduler": "EulerDiscreteScheduler",
        "resolution": int(resolution),
        "num_inference_steps": int(num_inference_steps),
        "cfg_scale": float(guidance_scale),
        "stage2": bool(stage2_enabled),
        "extra_unet_calls": 0,
        "initialization": "scheduler_native_init_sigma",
    }
    for key, observed in expected_sampling.items():
        registered = sampling.get(key)
        if isinstance(observed, float):
            same = isinstance(registered, (int, float)) and float(registered) == observed
        else:
            same = registered == observed
        if not same:
            raise ValueError(
                f"scheduler baseline sampling.{key} differs: "
                f"registered {registered!r}, observed {observed!r}"
            )
    source = config.get("source_manifest")
    if not isinstance(source, dict):
        raise ValueError("scheduler_baselines_v1 requires a source_manifest mapping")
    registered_prompts = source.get("prompts")
    if not isinstance(registered_prompts, str) or not registered_prompts:
        raise ValueError("scheduler baseline source_manifest.prompts is required")
    registered_abs = os.path.abspath(os.path.join(ROOT, registered_prompts))
    actual_abs = os.path.abspath(prompts_path)
    if registered_abs != actual_abs:
        raise ValueError(
            "scheduler baseline prompt path differs: "
            f"registered {registered_abs!r}, observed {actual_abs!r}"
        )
    registered_prompt_hash = source.get("prompts_sha256")
    if registered_prompt_hash != sha256_file(prompts_path):
        raise ValueError("scheduler baseline prompt SHA-256 differs from registration")
    design = config.get("design")
    if not isinstance(design, dict):
        raise ValueError("scheduler_baselines_v1 requires a design count mapping")
    prompts = pd.read_csv(prompts_path)
    observed = {
        "prompt_count": len(prompts),
        "action_count": len(actions),
        "seed_count": len(seeds),
        "task_count": len(prompts) * len(actions) * len(seeds),
    }
    expected = {
        "prompt_count": int(design.get("expected_prompt_count", -1)),
        "action_count": int(design.get("expected_action_count", -1)),
        "seed_count": int(design.get("expected_seed_count", -1)),
        "task_count": int(design.get("expected_task_count", -1)),
    }
    for key in ("prompt_count", "action_count", "seed_count", "task_count"):
        if observed[key] != expected[key]:
            raise ValueError(
                f"scheduler baseline design {key} differs: "
                f"registered {expected[key]}, observed {observed[key]}"
            )
    expected_action_ids = design.get("action_ids")
    observed_action_ids = [str(action.get("id")) for action in actions]
    if expected_action_ids != observed_action_ids:
        raise ValueError(
            "scheduler baseline action IDs differ: "
            f"registered {expected_action_ids!r}, observed {observed_action_ids!r}"
        )
    if any(bool(action.get("selection_eligible", True)) for action in actions):
        raise ValueError("scheduler baseline references must all be selection-ineligible")
    expected_types = {
        "no_correction": ("none", None, {}),
        "euler_ancestral_reference": (
            "scheduler_baseline",
            "EulerAncestralDiscreteScheduler",
            {},
        ),
        "dpmpp_2m_reference": (
            "scheduler_baseline",
            "DPMSolverMultistepScheduler",
            SCHEDULER_BASELINE_DEFAULT_KWARGS["DPMSolverMultistepScheduler"],
        ),
        "unipc_2_reference": (
            "scheduler_baseline",
            "UniPCMultistepScheduler",
            SCHEDULER_BASELINE_DEFAULT_KWARGS["UniPCMultistepScheduler"],
        ),
    }
    for action in actions:
        action_id = str(action.get("id"))
        expected_type, expected_class, expected_kwargs = expected_types.get(
            action_id, (None, None, None)
        )
        if action.get("type") != expected_type:
            raise ValueError(f"scheduler baseline action {action_id!r} has wrong type")
        if expected_class is not None and action.get("scheduler_class") != expected_class:
            raise ValueError(f"scheduler baseline action {action_id!r} has wrong class")
        if dict(action.get("scheduler_kwargs", {})) != dict(expected_kwargs):
            raise ValueError(f"scheduler baseline action {action_id!r} has wrong kwargs")
    expected_hashes = design.get("action_sha256")
    if not isinstance(expected_hashes, dict):
        raise ValueError("scheduler baseline design requires action_sha256 mapping")
    for action in actions:
        action_id = str(action.get("id"))
        expected_hash = expected_hashes.get(action_id)
        if expected_hash != action_sha256(action):
            raise ValueError(
                f"scheduler baseline action hash differs for {action_id!r}"
            )


def validate_registered_trajectory_design(
    path: str,
    *,
    prompts_path: str,
    resolution: int,
    num_inference_steps: int,
    guidance_scale: float,
    stage2_enabled: bool,
    model_name: str | None = None,
    scheduler_name: str | None = None,
    extra_unet_calls: int | None = None,
) -> None:
    """Reject accidental protocol drift for the frozen S7 development gate."""
    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    if config.get("schema") not in {
        "trajectory_correction_actions_v1",
        "trajectory_correction_validation_v1",
    }:
        return
    if config.get("schema") == "trajectory_correction_validation_v1" and not config.get(
        "selected_action"
    ):
        raise ValueError(
            "trajectory-correction validation requires selected_action frozen from development"
        )
    sampling = config.get("sampling") or {}
    if not isinstance(sampling, dict):
        raise ValueError("trajectory-correction sampling must be a mapping")
    registered_model = sampling.get("model")
    registered_scheduler = sampling.get("scheduler")
    registered_extra_calls = sampling.get("extra_unet_calls")
    if not isinstance(registered_model, str) or not registered_model:
        raise ValueError("trajectory-correction registration must declare sampling.model")
    if not isinstance(registered_scheduler, str) or not registered_scheduler:
        raise ValueError(
            "trajectory-correction registration must declare sampling.scheduler"
        )
    if isinstance(registered_extra_calls, bool) or not isinstance(
        registered_extra_calls, int
    ):
        raise ValueError(
            "trajectory-correction registration must declare integer extra_unet_calls"
        )
    if registered_extra_calls != 0:
        raise ValueError(
            "trajectory-correction actions require extra_unet_calls == 0"
        )
    if model_name is not None and str(model_name) != registered_model:
        raise ValueError(
            f"trajectory-correction model is registered as {registered_model!r}, "
            f"got {model_name!r}"
        )
    if scheduler_name is not None and str(scheduler_name) != registered_scheduler:
        raise ValueError(
            f"trajectory-correction scheduler is registered as {registered_scheduler!r}, "
            f"got {scheduler_name!r}"
        )
    if extra_unet_calls is not None and int(extra_unet_calls) != registered_extra_calls:
        raise ValueError(
            "trajectory-correction extra_unet_calls differs from registration"
        )
    registered_noise_mode = sampling.get("noise_mode")
    if registered_noise_mode is not None and str(registered_noise_mode) not in {
        "sqrt",
        "linear",
        "none",
    }:
        raise ValueError("trajectory-correction sampling.noise_mode is invalid")
    for action in config.get("actions", []):
        if action.get("type") != "trajectory_correction":
            continue
        mode = str(action.get("noise_mode", "sqrt"))
        if mode not in {"sqrt", "linear", "none"}:
            raise ValueError(
                f"trajectory-correction action {action.get('id')!r} has invalid noise_mode"
            )
    expected = {
        "resolution": int(sampling.get("resolution", resolution)),
        "num_inference_steps": int(
            sampling.get("num_inference_steps", num_inference_steps)
        ),
        "guidance_scale": float(sampling.get("cfg_scale", guidance_scale)),
        "stage2": sampling.get("stage2", stage2_enabled),
    }
    actual = {
        "resolution": int(resolution),
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "stage2": bool(stage2_enabled),
    }
    if not isinstance(expected["stage2"], bool):
        raise ValueError("trajectory-correction registration sampling.stage2 must be boolean")
    if expected != actual:
        raise ValueError(
            "trajectory-correction sampling is registered as "
            f"{expected}, got {actual}"
        )
    source_manifest = config.get("source_manifest") or {}
    registered_prompts = source_manifest.get("prompts")
    if registered_prompts is not None:
        expected_path = os.path.abspath(
            os.path.join(ROOT, str(registered_prompts))
        )
        if os.path.abspath(prompts_path) != expected_path:
            raise ValueError(
                "trajectory-correction action config requires prompts file "
                f"{expected_path}"
            )
    expected_hash = source_manifest.get("prompts_sha256")
    if expected_hash is not None and expected_hash != sha256_file(prompts_path):
        raise ValueError("trajectory-correction prompt hash does not match registration")


def generation_contract(
    cfg: dict,
    *,
    actions: list,
    seeds: list[int],
    prompts_sha256: str,
    actions_sha256: str | None,
) -> dict:
    """Return the immutable fields that define a resumable generation run."""
    contract = {
        "schema": PROVENANCE_SCHEMA,
        "action_schema": cfg.get("action_schema"),
        "actions_sha256": actions_sha256,
        "actions": actions,
        "prompts_sha256": prompts_sha256,
        "seeds": [int(value) for value in seeds],
        "model_name": cfg["model_name"],
        "resolution": int(cfg["resolution"]),
        "num_inference_steps": int(cfg["num_inference_steps"]),
        "guidance_scale": float(cfg["guidance_scale"]),
        "negative_prompt": cfg["negative_prompt"],
        "power_calibrate": int(cfg["power_calibrate"]),
        "stage_name": cfg["stage_name"],
        "stage2_enabled": bool(cfg["stage2_enabled"]),
        "models_to_cpu": bool(cfg["models_to_cpu"]),
        "multi_encoder": bool(cfg["multi_encoder"]),
        "multi_decoder": bool(cfg["multi_decoder"]),
        "num_resample_timesteps": int(cfg["num_resample_timesteps"]),
        "init_rates": list(cfg["init_rates"]),
        "frequency_band_cutoffs": list(cfg["frequency_band_cutoffs"]),
        "split_role": cfg.get("split_role"),
        "git_commit": cfg.get("git_commit"),
        "runtime_provenance": cfg.get("runtime_provenance", {}),
    }
    return contract


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scheduler_config_sha256(scheduler) -> str:
    """Hash the legacy JSON-normalized scheduler config.

    Keep this byte-for-byte compatible with existing S7 sidecars. New
    scheduler-baseline records additionally use ``scheduler_config_sha256_v2``
    below, which canonicalizes diffusers bookkeeping.
    """
    config = getattr(scheduler, "config", {})
    payload = json.dumps(dict(config), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def scheduler_config_payload(scheduler) -> dict:
    """Return the complete JSON-safe scheduler config with stable metadata."""
    config = dict(getattr(scheduler, "config", {}))
    if isinstance(config.get("_use_default_values"), list):
        config["_use_default_values"] = sorted(config["_use_default_values"])
    return json.loads(json.dumps(config, sort_keys=True, default=str))


def scheduler_config_payload_sha256(payload: dict) -> str:
    """Hash one recorded scheduler-config payload using the v2 byte contract."""
    if not isinstance(payload, dict):
        raise ValueError("scheduler config payload must be a mapping")
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def scheduler_config_sha256_v2(scheduler) -> str:
    """Hash effective scheduler config with stable default bookkeeping."""
    config = scheduler_config_payload(scheduler)
    return scheduler_config_payload_sha256(config)


def scheduler_baseline_runtime(action: dict, base_scheduler):
    """Construct one registered scheduler baseline from the base config."""
    if action.get("type") != "scheduler_baseline":
        return None
    base_name = type(base_scheduler).__name__
    if base_name != "EulerDiscreteScheduler":
        raise ValueError(
            "scheduler baselines require an EulerDiscreteScheduler base, "
            f"got {base_name}"
        )
    scheduler_class_name = str(action.get("scheduler_class", ""))
    scheduler_class = SCHEDULER_BASELINE_CLASSES.get(scheduler_class_name)
    if scheduler_class is None:
        raise ValueError(f"unsupported scheduler baseline {scheduler_class_name!r}")
    scheduler_kwargs = dict(action.get("scheduler_kwargs", {}))
    return scheduler_class.from_config(base_scheduler.config, **scheduler_kwargs)


def native_scheduler_runtime(base_scheduler):
    """Construct a clean copy of the model's native scheduler.

    RepLDM's pipeline mutates scheduler state in ``set_timesteps`` and during
    stepping.  Registered matched-NFE blocks must not share that state across
    actions, otherwise provenance (and potentially paired noise scaling) can
    depend on execution order.
    """
    return type(base_scheduler).from_config(base_scheduler.config)


def scheduler_schedule_payload(scheduler) -> dict:
    """Serialize the active timestep/sigma schedule without scheduler state."""
    timesteps = getattr(scheduler, "timesteps", None)
    sigmas = getattr(scheduler, "sigmas", None)

    def values(tensor):
        if tensor is None:
            return None
        if torch.is_tensor(tensor):
            tensor = tensor.detach().cpu().reshape(-1).tolist()
        return [float(value) for value in tensor]

    return {"timesteps": values(timesteps), "sigmas": values(sigmas)}


def prepare_scheduler_schedule_provenance(
    scheduler, num_inference_steps: int, device: str
) -> dict:
    """Freeze the exact schedule and both meanings of initial noise scale."""
    construction_sigma = float(scheduler.init_noise_sigma)
    scheduler.set_timesteps(int(num_inference_steps), device=device)
    effective_sigma = float(scheduler.init_noise_sigma)
    payload = scheduler_schedule_payload(scheduler)
    timesteps = payload["timesteps"] or []
    sigmas = payload["sigmas"] or []
    if len(timesteps) != int(num_inference_steps):
        raise ValueError("scheduler produced an unexpected timestep count")
    return {
        # Retain the old field as a construction-state alias for compatibility.
        "scheduler_init_noise_sigma": construction_sigma,
        "scheduler_construction_init_noise_sigma": construction_sigma,
        "scheduler_effective_init_noise_sigma": effective_sigma,
        "scheduler_timesteps": timesteps,
        "scheduler_sigmas": sigmas or None,
        "scheduler_schedule_sha256": json_sha256(payload),
    }


def validate_cfg_scheduler_runtime(
    registered_runtime: dict,
    *,
    config_sha256_v2: str,
    schedule_provenance: dict | None = None,
) -> None:
    """Reject any Euler implementation or schedule drift before CFG sampling."""
    if registered_runtime != CFG_BASELINE_SCHEDULER_RUNTIME:
        raise RuntimeError("CFG-baseline scheduler runtime contract drifted")
    if config_sha256_v2 != registered_runtime["config_sha256_v2"]:
        raise RuntimeError("CFG-baseline Euler config differs from the frozen runtime")
    if schedule_provenance is None:
        return
    observed = {
        "num_inference_steps": len(schedule_provenance["scheduler_timesteps"]),
        "schedule_sha256": schedule_provenance["scheduler_schedule_sha256"],
        "construction_init_noise_sigma": schedule_provenance[
            "scheduler_construction_init_noise_sigma"
        ],
        "effective_init_noise_sigma": schedule_provenance[
            "scheduler_effective_init_noise_sigma"
        ],
    }
    expected = {
        key: registered_runtime[key]
        for key in (
            "num_inference_steps",
            "schedule_sha256",
            "construction_init_noise_sigma",
            "effective_init_noise_sigma",
        )
    }
    if observed != expected:
        raise RuntimeError("CFG-baseline Euler schedule differs from the frozen runtime")


def validate_structural_control_scheduler_runtime(
    registered_runtime: dict,
    *,
    config_sha256_v2: str,
    schedule_provenance: dict | None = None,
) -> None:
    """Reject Euler config/schedule drift for every structural-control arm."""
    if registered_runtime != CFG_BASELINE_SCHEDULER_RUNTIME:
        raise RuntimeError("structural-control scheduler runtime contract drifted")
    if config_sha256_v2 != registered_runtime["config_sha256_v2"]:
        raise RuntimeError("structural-control Euler config differs")
    if schedule_provenance is None:
        return
    observed = {
        "num_inference_steps": len(schedule_provenance["scheduler_timesteps"]),
        "schedule_sha256": schedule_provenance["scheduler_schedule_sha256"],
        "construction_init_noise_sigma": schedule_provenance[
            "scheduler_construction_init_noise_sigma"
        ],
        "effective_init_noise_sigma": schedule_provenance[
            "scheduler_effective_init_noise_sigma"
        ],
    }
    expected = {
        key: registered_runtime[key]
        for key in (
            "num_inference_steps",
            "schedule_sha256",
            "construction_init_noise_sigma",
            "effective_init_noise_sigma",
        )
    }
    if observed != expected:
        raise RuntimeError("structural-control Euler schedule differs")


def scheduler_provenance_record(
    action: dict,
    *,
    include: bool,
    base_config_sha256_v2: str,
    active_config_sha256_v2: str,
    order: int,
    solver_order,
    init_noise_sigma: float,
    schedule_provenance: dict | None = None,
) -> dict:
    """Return the versioned scheduler ledger for registered reference runs."""
    if not include:
        return {}
    record = {
        "scheduler_config_sha256_v2": base_config_sha256_v2,
        "active_scheduler_config_sha256_v2": active_config_sha256_v2,
        "scheduler_kwargs": dict(action.get("scheduler_kwargs", {})),
        "scheduler_order": int(order),
        "scheduler_solver_order": solver_order,
        "scheduler_init_noise_sigma": float(init_noise_sigma),
    }
    if schedule_provenance:
        record.update(schedule_provenance)
    return record


def validate_one_unet_call_per_step(
    observed_calls: list[int], num_inference_steps: int, *, run_label: str
) -> None:
    """Enforce the matched-NFE contract for a registered generation run."""
    expected_steps = int(num_inference_steps)
    if len(observed_calls) != expected_steps or any(
        isinstance(value, bool) or int(value) != 1 for value in observed_calls
    ):
        raise RuntimeError(
            f"{run_label} requires exactly one U-Net call per denoising step"
        )


def validate_structural_control_worker_result(
    cfg: dict, images, observed_calls: list[int]
) -> None:
    """Enforce one image and matched NFE for every structural-control action."""
    if not cfg.get("structural_control_registered"):
        return
    if not isinstance(images, (list, tuple)) or len(images) != 1:
        raise RuntimeError("structural controls require exactly one image per prompt")
    validate_one_unet_call_per_step(
        observed_calls,
        cfg["num_inference_steps"],
        run_label="structural controls",
    )


def validate_final_test_authorization(
    authorization_path: str, actions_path: str, seeds
) -> None:
    """Require an independently hashed LR-1 validation/review authorization."""
    with open(authorization_path) as handle:
        authorization = json.load(handle)
    if authorization.get("schema") != "latent_renderer_final_authorization_v1":
        raise ValueError("invalid final-test authorization schema")
    if authorization.get("status") != "authorized_final_test":
        raise ValueError("final-test authorization is not active")
    expected_seeds = [int(value) for value in authorization.get("test_seeds", [])]
    if list(seeds) != expected_seeds:
        raise ValueError(
            f"--seeds {list(seeds)} do not match authorized final-test seeds {expected_seeds}"
        )
    source_hash = authorization.get("source_actions_sha256")
    if source_hash != sha256_file(actions_path):
        raise ValueError("final-test authorization hash does not match --actions")
    summary = authorization.get("review_summary", {})
    if summary.get("passed") is not True:
        raise ValueError("final-test authorization lacks a passing blinded review")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def validate_registered_environment_lock(config: dict) -> dict | None:
    """Validate an optional repository-relative environment-lock registration."""

    registration = config.get("environment_lock")
    if registration is None:
        return None
    if not isinstance(registration, dict) or set(registration) != {"path", "sha256"}:
        raise ValueError("environment_lock must exactly define path and sha256")
    relative_path = registration["path"]
    expected_sha256 = registration["sha256"]
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("environment_lock.path must be a non-empty repository path")
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise ValueError("environment_lock.sha256 must be a lowercase SHA-256 digest")
    absolute_path = os.path.abspath(os.path.join(ROOT, relative_path))
    if os.path.commonpath((ROOT, absolute_path)) != ROOT:
        raise ValueError("environment_lock.path must stay inside the repository")
    record = validate_environment_lock(
        absolute_path, expected_sha256=expected_sha256
    )
    record["path"] = os.path.relpath(absolute_path, ROOT)
    return record


def runtime_provenance(environment_lock: dict | None = None) -> dict:
    """Return package/runtime versions that can affect generated pixels."""
    record = {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "diffusers_version": str(getattr(diffusers, "__version__", "unknown")),
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if environment_lock is not None:
        record.update(
            {
                "generation_environment_lock_id": environment_lock["lock_id"],
                "generation_environment_lock_path": environment_lock["path"],
                "generation_environment_lock_sha256": environment_lock["sha256"],
                "generation_environment_packages": environment_lock["observed"][
                    "packages"
                ],
                "generation_environment_platform": environment_lock["observed"][
                    "platform"
                ],
                "generation_environment_hardware": environment_lock["observed"][
                    "hardware"
                ],
                "generation_environment_determinism": environment_lock["observed"][
                    "determinism"
                ],
            }
        )
    return record


def worker_determinism_provenance() -> dict:
    """Sample determinism controls inside the spawned generation worker."""
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def cuda_device_identity(device_index: int, device_properties) -> dict:
    """Resolve a CUDA logical device to its physical NVIDIA identity."""
    torch_uuid = str(getattr(device_properties, "uuid", "")).lower()
    normalized_torch_uuid = torch_uuid.removeprefix("gpu-")
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot resolve the selected CUDA device identity") from exc
    matches = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        physical_index, gpu_uuid, pci_bus_id = fields
        if gpu_uuid.lower().removeprefix("gpu-") == normalized_torch_uuid:
            matches.append((physical_index, gpu_uuid, pci_bus_id))
    if len(matches) != 1:
        raise RuntimeError("selected CUDA device UUID is absent or ambiguous in nvidia-smi")
    physical_index, gpu_uuid, pci_bus_id = matches[0]
    try:
        physical_index_value = int(physical_index)
    except ValueError as exc:
        raise RuntimeError("selected CUDA physical index is invalid") from exc
    return {
        "requested_device": f"cuda:{int(device_index)}",
        "logical_device_index": int(device_index),
        "physical_device_index": physical_index_value,
        "gpu_uuid": gpu_uuid,
        "pci_bus_id": pci_bus_id,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def validate_model_cache(model_name: str, cache_dir: str) -> None:
    if os.path.isdir(model_name):
        if not os.path.isfile(os.path.join(model_name, "model_index.json")):
            raise FileNotFoundError(f"local model directory lacks model_index.json: {model_name}")
        return
    repo_dir = os.path.join(cache_dir, "models--" + model_name.replace("/", "--"))
    snapshots = glob.glob(os.path.join(repo_dir, "snapshots", "*", "model_index.json"))
    if not snapshots:
        raise FileNotFoundError(
            f"no local snapshot for {model_name!r} under {cache_dir!r}; "
            f"expected {repo_dir}/snapshots/*/model_index.json"
        )


def generation_stage_settings(
    stage2_enabled: bool,
    resolution: int,
    low_vram: bool,
) -> dict:
    """Validate the requested stage and return recorded pipeline settings."""
    if resolution <= 0 or resolution % 8:
        raise ValueError("resolution must be a positive multiple of 8")
    if stage2_enabled and resolution <= 1024:
        raise ValueError("--stage2 requires --resolution greater than 1024")
    if not stage2_enabled and resolution > 1024:
        raise ValueError("resolution above 1024 requires the explicit --stage2 opt-in")

    init_rates = [0.8]
    if stage2_enabled and resolution >= 4096:
        init_rates = [0.9, 0.8]
    return {
        "stage2_enabled": stage2_enabled,
        "stage_name": (
            f"stage2_{resolution}" if stage2_enabled else f"stage1_{resolution}"
        ),
        "models_to_cpu": bool(low_vram or stage2_enabled),
        "multi_encoder": bool(stage2_enabled),
        "multi_decoder": bool(stage2_enabled and resolution > 2048),
        "num_resample_timesteps": 50,
        "init_rates": init_rates,
        "stage2_noise_source": "task_generator" if stage2_enabled else None,
    }


def guidance_runtime(action: dict, num_inference_steps: int):
    """Translate a validated action record into pipeline guidance arguments."""
    action_type = action["type"]
    residual_mode = action.get("residual_mode", "raw")
    controller = None
    attn_scale = float(action.get("scale", 0.0))
    attn_density = "all"
    attn_decay = None
    if action_type == "legacy":
        delay_steps = action.get("delay_steps", 0)
        attn_density = tuple(
            [1] * (num_inference_steps - delay_steps) + [0] * delay_steps
        )
        attn_decay = tuple(action["decay"]) if action.get("decay") else None
    elif action_type == "frequency_bands":
        controller = ConstantGuidanceController(
            band_scales=tuple(action["band_scales"]),
            max_update_ratio=action.get("max_update_ratio"),
        )
    elif action_type == "scalar" and (
        residual_mode != "raw" or action.get("max_update_ratio") is not None
    ):
        controller = ConstantGuidanceController(
            scale=attn_scale,
            max_update_ratio=action.get("max_update_ratio"),
            residual_mode=residual_mode,
        )
        attn_scale = 0.0
    return controller, attn_scale, attn_density, attn_decay


def clean_transport_runtime(action: dict):
    """Translate a validated clean-transport action for the pipeline."""
    if action["type"] != "clean_transport":
        return None
    return {
        "mode": action["transport_mode"],
        "angle": action["angle"],
        "topk": action["topk"],
        "layer_name": action["semantic_transport_layer"],
        "permutation_seed": action["permutation_seed"],
    }


def attention_baseline_runtime(action: dict):
    """Translate a validated PLADIS/GAG action for the worker context."""
    if action["type"] != "attention_baseline":
        return None
    return {
        "kind": action["attention_baseline"],
        "attention_scale": action["baseline_scale"],
        "alpha": action["alpha"],
        "eta": action["eta"],
        "zeta": action["zeta"],
        "applied_layers": action.get("applied_layers"),
        "probability_dtype": action.get("probability_dtype", "float32"),
        "expected_group_counts": action.get("expected_processor_group_counts"),
        "expected_processor_names_sha256": action.get(
            "expected_processor_names_sha256"
        ),
        "attention_mask_policy": action.get("attention_mask_policy", "supported"),
    }


def freeu_runtime(action: dict):
    """Translate a normalized FreeU action into a re-entrant schedule."""
    if action["type"] != "freeu":
        return None
    record = action.get("freeu_schedule")
    if not isinstance(record, dict) or not isinstance(record.get("knots"), list):
        raise ValueError("normalized FreeU action lacks a knots record")
    return FreeUSchedule(
        (item["position"], item["parameters"]) for item in record["knots"]
    )


def freeu_implementation_runtime(action: dict) -> str:
    """Return the explicitly named FreeU tensor operator for one action."""
    if action["type"] != "freeu":
        return DIFFUSERS_FREEU_IMPLEMENTATION
    return str(action.get("implementation", DIFFUSERS_FREEU_IMPLEMENTATION))


def trajectory_correction_runtime(action: dict):
    """Translate a normalized trajectory-correction action."""
    if action["type"] != "trajectory_correction":
        return None
    return TrajectoryCorrectionConfig(
        mix=action["mix"],
        noise_mode=action["noise_mode"],
        max_correction_ratio=action.get("max_correction_ratio"),
    )


def latent_renderer_runtime(action: dict, pipe, device: str, guidance_scale: float):
    """Construct one fixed LR-1 renderer/provider pair for a worker device."""
    if action["type"] != "latent_renderer_fixed":
        return None, None
    renderer = build_fixed_coefficient_renderer(
        action["coefficients"],
        latent_channels=4,
        coefficient_bound=action["coefficient_bound"],
        max_update_ratio=action["max_update_ratio"],
        preserve_moments=True,
        basis_normalization=action["latent_renderer_provider"][
            "basis_normalization"
        ],
    ).to(device)
    provider_config = action["latent_renderer_provider"]
    implementation = normalize_latent_structure_provider_implementation(
        provider_config.get("implementation", LAZY_LATENT_STRUCTURE_PROVIDER_ID)
    )
    common_kwargs = {
        "batch_size": 1,
        "do_classifier_free_guidance": float(
            action.get("cfg_scale", guidance_scale)
        )
        > 1.0,
        "latent_channels": 4,
        "semantic_mode": provider_config["semantic_mode"],
        "semantic_topk": provider_config["semantic_topk"],
        "semantic_layer": provider_config["semantic_layer"],
        "feature_block": provider_config["feature_block"],
        "permutation_seed": provider_config["permutation_seed"],
        "prompt_dim": provider_config["prompt_dim"],
        "state_dim": provider_config["state_dim"],
    }
    if implementation == LAZY_LATENT_STRUCTURE_PROVIDER_ID:
        # ``StructuralUNetBasisProvider`` remains a patchable compatibility
        # symbol for older launchers/tests.  The registered implementation is
        # lazy unless that symbol has explicitly been replaced.
        provider_class = LazyLatentStructureBasisProvider
        if StructuralUNetBasisProvider is not _DEFAULT_STRUCTURAL_PROVIDER_CLASS:
            provider_class = StructuralUNetBasisProvider
        provider = provider_class(
            pipe.unet,
            **common_kwargs,
            requested_bases=provider_config["requested_bases"],
            required_hook_names=provider_config["required_hook_names"],
            scheduler_mapping=provider_config["scheduler_mapping"],
            basis_normalization=provider_config["basis_normalization"],
            provider_id=provider_config["provider_id"],
            provider_provenance_id=provider_config.get(
                "provider_provenance_id", provider_config["provider_id"]
            ),
        )
    elif implementation == "structural_unet_basis_v1":
        provider = StructuralUNetBasisProvider(
            pipe.unet,
            **common_kwargs,
            requested_bases=provider_config["requested_bases"],
            required_hook_names=provider_config["required_hook_names"],
            scheduler_mapping=provider_config["scheduler_mapping"],
            basis_normalization=provider_config["basis_normalization"],
            provider_id=provider_config["provider_id"],
            provider_provenance_id=provider_config.get(
                "provider_provenance_id", provider_config["provider_id"]
            ),
        )
    else:
        raise ValueError(f"unsupported latent renderer provider implementation {implementation!r}")
    renderer.eval()
    return renderer, provider


def latent_renderer_pipeline_kwargs(action: dict, renderer, provider) -> dict:
    """Return the complete renderer kwargs passed to one pipeline invocation."""
    if action["type"] != "latent_renderer_fixed":
        if renderer is not None or provider is not None:
            raise ValueError("non-renderer action received latent renderer runtime objects")
        return {
            "latent_renderer": None,
            "latent_renderer_basis_provider": None,
            "latent_renderer_scheduler_mapping": "legacy_unit",
        }
    if renderer is None or provider is None:
        raise ValueError("latent renderer action is missing runtime objects")
    provider_config = action.get("latent_renderer_provider")
    if not isinstance(provider_config, dict):
        raise ValueError("latent renderer action lacks normalized provider config")
    mapping = str(provider_config.get("scheduler_mapping", ""))
    if mapping not in LATENT_RENDERER_SCHEDULER_MAPPINGS:
        raise ValueError("latent renderer action has invalid scheduler_mapping")
    return {
        "latent_renderer": renderer,
        "latent_renderer_basis_provider": provider,
        "latent_renderer_scheduler_mapping": mapping,
    }


def validate_latent_renderer_scheduler(action: dict, scheduler) -> str | None:
    """Fail before sampling when a native renderer has the wrong scheduler."""
    if action["type"] != "latent_renderer_fixed":
        return None
    provider_config = action.get("latent_renderer_provider")
    if not isinstance(provider_config, dict):
        raise RuntimeError("latent renderer action lacks normalized provider config")
    mapping = str(provider_config.get("scheduler_mapping", ""))
    if mapping not in LATENT_RENDERER_SCHEDULER_MAPPINGS:
        raise RuntimeError("latent renderer scheduler mapping is not registered")
    if (
        mapping == "euler_clean_endpoint"
        and type(scheduler).__name__ != "EulerDiscreteScheduler"
    ):
        raise RuntimeError(
            "euler_clean_endpoint requires exactly EulerDiscreteScheduler"
        )
    return mapping


def _json_safe_record(value, *, label: str) -> dict:
    if value is None:
        raise RuntimeError(f"{label} are missing")
    if hasattr(value, "to_record"):
        value = value.to_record()
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a mapping")
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} are not JSON-safe") from exc


def _finite_diagnostic_values(record: dict, key: str, *, step_index: int) -> list[float]:
    values = record.get(key)
    if not isinstance(values, list) or len(values) != 1:
        raise RuntimeError(
            f"native renderer step {step_index} field {key!r} must contain "
            "one batch value"
        )
    converted = [float(value) for value in values]
    if not all(math.isfinite(value) for value in converted):
        raise RuntimeError(
            f"native renderer step {step_index} field {key!r} is non-finite"
        )
    return converted


def _validate_provider_diagnostics_contract(
    provider_diagnostics: dict,
    provider_contract: dict,
    *,
    step_index: int,
) -> None:
    """Require the complete normalized provider contract at every step."""
    expected_provider_fields = {
        "implementation": provider_contract.get(
            "implementation", LAZY_LATENT_STRUCTURE_PROVIDER_ID
        ),
        "provider_id": provider_contract.get("provider_id"),
        "provider_provenance_id": provider_contract.get(
            "provider_provenance_id", provider_contract.get("provider_id")
        ),
        "requested_bases": provider_contract.get("requested_bases"),
        "constructed_bases": provider_contract.get("requested_bases"),
        "registered_hook_names": provider_contract.get("required_hook_names", []),
        "required_hook_names": provider_contract.get("required_hook_names", []),
        "scheduler_mapping": provider_contract.get("scheduler_mapping"),
        "basis_normalization": provider_contract.get("basis_normalization"),
    }
    for key, expected in expected_provider_fields.items():
        if expected is None:
            continue
        observed = provider_diagnostics.get(key)
        if observed != expected:
            raise RuntimeError(
                f"scheduler-native renderer step {step_index} provider field {key!r} drifted"
            )
    basis_rms = provider_diagnostics.get("basis_rms")
    if (
        not isinstance(basis_rms, list)
        or len(basis_rms) != 1
        or not isinstance(basis_rms[0], list)
        or len(basis_rms[0]) != len(LAZY_LATENT_STRUCTURE_BASIS_NAMES)
    ):
        raise RuntimeError(
            f"scheduler-native renderer step {step_index} basis_rms must have shape (1, 6)"
        )
    values = [float(value) for value in basis_rms[0]]
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise RuntimeError(
            f"scheduler-native renderer step {step_index} basis_rms is invalid"
        )
    requested = set(provider_contract.get("requested_bases") or [])
    for index, name in enumerate(LAZY_LATENT_STRUCTURE_BASIS_NAMES):
        if name not in requested and values[index] != 0.0:
            raise RuntimeError(
                f"scheduler-native renderer step {step_index} unrequested basis {name!r} is non-zero"
            )


def validate_native_renderer_step_diagnostics(
    records,
    num_inference_steps: int,
    *,
    max_update_ratio: float,
    schedule_provenance: dict | None = None,
    provider_contract: dict | None = None,
) -> list[dict]:
    """Validate the complete scheduler-native per-step renderer ledger."""
    expected_steps = int(num_inference_steps)
    if not isinstance(records, list) or len(records) != expected_steps:
        raise RuntimeError(
            "scheduler-native renderer diagnostics must contain exactly one "
            "record per denoising step"
        )
    try:
        normalized = json.loads(json.dumps(records, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "scheduler-native renderer diagnostics are not JSON-safe"
        ) from exc
    if schedule_provenance is not None:
        expected_timesteps = schedule_provenance.get("scheduler_timesteps")
        expected_sigmas = schedule_provenance.get("scheduler_sigmas")
        expected_hash = schedule_provenance.get("scheduler_schedule_sha256")
        payload = {
            "timesteps": expected_timesteps,
            "sigmas": expected_sigmas,
        }
        if (
            not isinstance(expected_timesteps, list)
            or not isinstance(expected_sigmas, list)
            or len(expected_timesteps) != expected_steps
            or len(expected_sigmas) != expected_steps + 1
            or expected_hash != json_sha256(payload)
        ):
            raise RuntimeError("registered native scheduler schedule provenance is invalid")
        for schedule_index, schedule_record in enumerate(normalized):
            if not math.isclose(
                float(schedule_record["timestep"]),
                float(expected_timesteps[schedule_index]),
                rel_tol=1e-6,
                abs_tol=1e-5,
            ) or not math.isclose(
                float(schedule_record["sigma_from"]),
                float(expected_sigmas[schedule_index]),
                rel_tol=1e-6,
                abs_tol=1e-6,
            ) or not math.isclose(
                float(schedule_record["sigma_to"]),
                float(expected_sigmas[schedule_index + 1]),
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                raise RuntimeError(
                    f"scheduler-native renderer step {schedule_index} differs from the registered schedule"
                )
    for expected_index, record in enumerate(normalized):
        if not isinstance(record, dict):
            raise RuntimeError("scheduler-native renderer step record must be a mapping")
        if record.get("step_index") != expected_index:
            raise RuntimeError("scheduler-native renderer step indices are incomplete")
        if record.get("scheduler_step_index") != expected_index:
            raise RuntimeError("scheduler-native scheduler step indices drifted")
        try:
            timestep = float(record["timestep"])
            sigma_from = float(record["sigma_from"])
            sigma_to = float(record["sigma_to"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"scheduler-native renderer step {expected_index} lacks sigma provenance"
            ) from exc
        if not all(math.isfinite(value) for value in (timestep, sigma_from, sigma_to)):
            raise RuntimeError(
                f"scheduler-native renderer step {expected_index} has non-finite provenance"
            )
        if sigma_from <= 0 or sigma_to < 0 or sigma_to > sigma_from:
            raise RuntimeError(
                f"scheduler-native renderer step {expected_index} has invalid sigma order"
            )
        if record.get("prediction_type") not in {
            "epsilon",
            "sample",
            "v_prediction",
        }:
            raise RuntimeError(
                f"scheduler-native renderer step {expected_index} has invalid prediction type"
            )
        gain = _finite_diagnostic_values(
            record, "clean_update_gain", step_index=expected_index
        )
        applied_ratio = _finite_diagnostic_values(
            record, "applied_update_ratio", step_index=expected_index
        )
        mean_error = _finite_diagnostic_values(
            record, "mean_error", step_index=expected_index
        )
        variance_error = _finite_diagnostic_values(
            record, "variance_error", step_index=expected_index
        )
        expected_gain = 1.0 - sigma_to / sigma_from
        if not 0 < gain[0] <= 1 or not math.isclose(
            gain[0], expected_gain, rel_tol=1e-6, abs_tol=1e-7
        ):
            raise RuntimeError(
                f"scheduler-native renderer step {expected_index} has invalid clean gain"
            )
        if applied_ratio[0] < 0 or applied_ratio[0] > max_update_ratio + 1e-6:
            raise RuntimeError(
                f"scheduler-native renderer step {expected_index} violates the applied trust bound"
            )
        if abs(mean_error[0]) > 1e-4:
            raise RuntimeError(
                f"scheduler-native renderer step {expected_index} violates mean preservation"
            )
        if abs(variance_error[0]) > 1e-3:
            raise RuntimeError(
                f"scheduler-native renderer step {expected_index} violates variance preservation"
            )
        provider_diagnostics = record.get("provider_diagnostics")
        if not isinstance(provider_diagnostics, dict):
            raise RuntimeError(
                f"scheduler-native renderer step {expected_index} lacks provider diagnostics"
            )
        if provider_contract is not None:
            _validate_provider_diagnostics_contract(
                provider_diagnostics,
                provider_contract,
                step_index=expected_index,
            )
    return normalized


def latent_renderer_sidecar_fields(
    action: dict,
    pipe,
    observed_unet_calls: list[int],
    num_inference_steps: int,
    *,
    schedule_provenance: dict | None = None,
) -> dict:
    """Build and validate renderer-specific sidecar provenance."""
    if action["type"] != "latent_renderer_fixed":
        if any(
            getattr(pipe, field, None) is not None
            for field in (
                "_last_latent_renderer_diagnostics",
                "_last_latent_renderer_provider_diagnostics",
                "_last_latent_renderer_scheduler_mapping",
            )
        ) or getattr(pipe, "_last_latent_renderer_step_diagnostics", None) not in (
            None,
            [],
        ):
            raise RuntimeError("non-renderer action retained stale renderer diagnostics")
        return {
            "latent_renderer_scheduler_mapping": None,
            "latent_renderer_diagnostics": None,
            "latent_renderer_provider_diagnostics": None,
            "latent_renderer_step_diagnostics": None,
        }

    expected_mapping = action["latent_renderer_provider"]["scheduler_mapping"]
    runtime_mapping = getattr(
        pipe, "_last_latent_renderer_scheduler_mapping", None
    )
    if runtime_mapping != expected_mapping:
        raise RuntimeError("pipeline latent renderer scheduler mapping drifted")
    diagnostics = _json_safe_record(
        getattr(pipe, "_last_latent_renderer_diagnostics", None),
        label="latent renderer diagnostics",
    )
    provider_diagnostics = _json_safe_record(
        getattr(pipe, "_last_latent_renderer_provider_diagnostics", None),
        label="latent renderer provider diagnostics",
    )
    provider_config = action.get("latent_renderer_provider", {})
    if not isinstance(provider_config, dict):
        raise RuntimeError("latent renderer action has invalid provider config")
    # New lazy-provider actions carry an explicit mechanism contract.  Keep
    # legacy sidecars readable when these fields were not present in their
    # frozen action, but enforce exact propagation for every normalized action.
    contract_present = any(
        key in provider_config
        for key in (
            "implementation",
            "provider_id",
            "provider_provenance_id",
            "requested_bases",
            "required_hook_names",
        )
    )
    provider_contract = provider_config if contract_present else None
    if provider_contract is not None:
        _validate_provider_diagnostics_contract(
            provider_diagnostics,
            provider_contract,
            step_index=-1,
        )
    step_diagnostics = None
    if expected_mapping == "euler_clean_endpoint":
        validate_one_unet_call_per_step(
            observed_unet_calls,
            num_inference_steps,
            run_label="scheduler-native latent renderer",
        )
        step_diagnostics = validate_native_renderer_step_diagnostics(
            getattr(pipe, "_last_latent_renderer_step_diagnostics", None),
            num_inference_steps,
            max_update_ratio=float(action["max_update_ratio"]),
            provider_contract=provider_contract,
            schedule_provenance=schedule_provenance,
        )
        for key, value in diagnostics.items():
            if step_diagnostics[-1].get(key) != value:
                raise RuntimeError(
                    "last-step renderer diagnostics differ from the native ledger"
                )
        if step_diagnostics[-1].get("provider_diagnostics") != provider_diagnostics:
            raise RuntimeError(
                "last-step provider diagnostics differ from the native ledger"
            )
    return {
        "latent_renderer_scheduler_mapping": runtime_mapping,
        "latent_renderer_diagnostics": diagnostics,
        "latent_renderer_provider_diagnostics": provider_diagnostics,
        "latent_renderer_step_diagnostics": step_diagnostics,
    }


def worker_process(cfg: dict, device: str, task_queue, error_queue, abort_event=None):
    torch.cuda.set_device(device)
    device_index = torch.device(device).index
    if device_index is None:
        device_index = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(device_index)
    worker_device_provenance = {
        "gpu": str(device_properties.name),
        "compute_capability": (
            f"{int(device_properties.major)}.{int(device_properties.minor)}"
        ),
        "total_memory_bytes": int(device_properties.total_memory),
        **cuda_device_identity(device_index, device_properties),
    }
    if worker_device_provenance["requested_device"] != str(device):
        raise RuntimeError("selected CUDA logical device differs from worker assignment")
    worker_determinism = worker_determinism_provenance()
    registered_sampling = cfg.get("registered_sampling") or {}
    load_kwargs = {
        "torch_dtype": torch.float16,
        "variant": "fp16",
        "cache_dir": cfg["cache_dir"],
        "local_files_only": True,
    }
    if (
        cfg.get("cfg_baseline_registered")
        or cfg.get("native_renderer_registered")
        or cfg.get("structural_control_registered")
    ):
        model_revision = str(registered_sampling.get("model_revision", ""))
        if cfg.get("cfg_baseline_registered") and model_revision != CFG_BASELINE_MODEL_REVISION:
            raise RuntimeError("CFG-baseline model revision differs from registration")
        if cfg.get("native_renderer_registered") and not model_revision:
            raise RuntimeError("scheduler-native run lacks a registered model revision")
        if cfg.get("structural_control_registered") and (
            model_revision != CFG_BASELINE_MODEL_REVISION
        ):
            raise RuntimeError("structural-control model revision differs")
        if cfg.get("model_revision") != model_revision:
            raise RuntimeError("registered run config has model revision drift")
        load_kwargs["revision"] = model_revision
    pipe = RepLDMSDXLPipeline.from_pretrained(
        cfg["model_name"],
        **load_kwargs,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    base_scheduler = pipe.scheduler
    base_scheduler_name = type(base_scheduler).__name__
    base_scheduler_hash = scheduler_config_sha256(base_scheduler)
    base_scheduler_hash_v2 = scheduler_config_sha256_v2(base_scheduler)
    base_scheduler_config = scheduler_config_payload(base_scheduler)
    model_load_provenance = None
    if cfg.get("structural_control_registered"):
        expected_hardware = cfg.get("runtime_provenance", {}).get(
            "generation_environment_hardware"
        )
        if not isinstance(expected_hardware, dict):
            raise RuntimeError("structural-control hardware provenance is missing")
        observed_hardware = {
            key: worker_device_provenance[key]
            for key in ("gpu", "compute_capability")
        }
        registered_hardware = {
            key: expected_hardware.get(key)
            for key in ("gpu", "compute_capability")
        }
        if observed_hardware != registered_hardware:
            raise RuntimeError(
                "selected structural-control worker GPU differs from environment lock"
            )
        expected_determinism = cfg.get("runtime_provenance", {}).get(
            "generation_environment_determinism"
        )
        if worker_determinism != expected_determinism:
            raise RuntimeError(
                "structural-control worker determinism differs from environment lock"
            )
        if type(pipe).__name__ != "RepLDMSDXLPipeline":
            raise RuntimeError("structural-control pipeline class differs")
        try:
            observed_model_dtype = str(next(pipe.unet.parameters()).dtype).removeprefix(
                "torch."
            )
        except (AttributeError, StopIteration) as exc:
            raise RuntimeError("cannot determine structural-control U-Net dtype") from exc
        if observed_model_dtype != registered_sampling.get("torch_dtype"):
            raise RuntimeError("structural-control U-Net dtype differs")
        model_load_provenance = {
            "torch_dtype": observed_model_dtype,
            "variant": load_kwargs.get("variant"),
            "local_files_only": load_kwargs.get("local_files_only"),
            "revision": load_kwargs.get("revision"),
        }
    if cfg.get("trajectory_registered"):
        expected_scheduler = str(registered_sampling.get("scheduler", ""))
        if base_scheduler_name != expected_scheduler:
            raise RuntimeError(
                "trajectory-correction run loaded the wrong scheduler: "
                f"registered {expected_scheduler!r}, got {base_scheduler_name!r}"
            )
        if str(cfg.get("model_name")) != str(registered_sampling.get("model")):
            raise RuntimeError("trajectory-correction model differs from registration")
        if int(registered_sampling.get("extra_unet_calls", -1)) != 0:
            raise RuntimeError("trajectory-correction requires zero extra U-Net calls")
    if cfg.get("scheduler_baseline_registered"):
        expected_scheduler = str(registered_sampling.get("base_scheduler", ""))
        if base_scheduler_name != expected_scheduler:
            raise RuntimeError(
                "scheduler-baseline run loaded the wrong base scheduler: "
                f"registered {expected_scheduler!r}, got {base_scheduler_name!r}"
            )
        if str(cfg.get("model_name")) != str(registered_sampling.get("model")):
            raise RuntimeError("scheduler-baseline model differs from registration")
        if int(registered_sampling.get("extra_unet_calls", -1)) != 0:
            raise RuntimeError("scheduler-baseline requires zero extra U-Net calls")
    if cfg.get("cfg_baseline_registered"):
        expected_scheduler = str(registered_sampling.get("base_scheduler", ""))
        if base_scheduler_name != expected_scheduler:
            raise RuntimeError(
                "CFG-baseline run loaded the wrong base scheduler: "
                f"registered {expected_scheduler!r}, got {base_scheduler_name!r}"
            )
        if str(cfg.get("model_name")) != str(registered_sampling.get("model")):
            raise RuntimeError("CFG-baseline model differs from registration")
        if int(registered_sampling.get("extra_unet_calls", -1)) != 0:
            raise RuntimeError("CFG-baseline requires zero extra U-Net calls")
        if str(registered_sampling.get("cfg_source")) != "action.cfg_scale":
            raise RuntimeError("CFG-baseline requires action.cfg_scale as its CFG source")
        if str(registered_sampling.get("cfg_pipeline_argument")) != "guidance_scale":
            raise RuntimeError(
                "CFG-baseline must pass each action scale through guidance_scale"
            )
        if cfg.get("guidance_rescale") != 0.0 or registered_sampling.get(
            "guidance_rescale"
        ) != 0.0:
            raise RuntimeError("CFG-baseline guidance_rescale must be exactly 0.0")
        validate_cfg_scheduler_runtime(
            cfg.get("scheduler_runtime"),
            config_sha256_v2=base_scheduler_hash_v2,
        )
    if cfg.get("native_renderer_registered"):
        expected_scheduler = str(registered_sampling.get("scheduler", ""))
        if expected_scheduler != "EulerDiscreteScheduler":
            raise RuntimeError(
                "scheduler-native registration must name EulerDiscreteScheduler"
            )
        if base_scheduler_name != expected_scheduler:
            raise RuntimeError(
                "scheduler-native renderer run requires EulerDiscreteScheduler"
            )
        if str(cfg.get("model_name")) != str(registered_sampling.get("model", "")):
            raise RuntimeError("scheduler-native model differs from registration")
        if int(registered_sampling.get("extra_unet_calls", -1)) != 0:
            raise RuntimeError("scheduler-native renderer requires zero extra U-Net calls")
        if float(registered_sampling.get("scheduler_churn", 0.0)) != 0.0:
            raise RuntimeError("scheduler-native Euler mapping requires zero scheduler churn")
    if cfg.get("structural_control_registered"):
        if base_scheduler_name != registered_sampling.get("scheduler"):
            raise RuntimeError("structural-control scheduler class differs")
        if str(cfg.get("model_name")) != str(registered_sampling.get("model")):
            raise RuntimeError("structural-control model differs")
        if base_scheduler.config.get("prediction_type") != registered_sampling.get(
            "prediction_type"
        ):
            raise RuntimeError("structural-control prediction_type differs")
        if registered_sampling.get("cfg_source") != "action.cfg_scale" or (
            registered_sampling.get("cfg_pipeline_argument") != "guidance_scale"
        ):
            raise RuntimeError("structural-control CFG routing differs")
        if cfg.get("guidance_rescale") != 0.0 or registered_sampling.get(
            "guidance_rescale"
        ) != 0.0:
            raise RuntimeError("structural-control guidance_rescale differs")
        if registered_sampling.get("attention_mask_policy") != "none":
            raise RuntimeError("structural-control attention-mask policy differs")
        if (
            registered_sampling.get("variant") != "fp16"
            or registered_sampling.get("local_files_only") is not True
            or registered_sampling.get("batch_size") != 1
            or registered_sampling.get("num_images_per_prompt") != 1
            or int(registered_sampling.get("extra_unet_calls", -1)) != 0
            or float(registered_sampling.get("scheduler_churn", float("nan"))) != 0.0
            or registered_sampling.get("initialization")
            != "scheduler_native_init_sigma"
        ):
            raise RuntimeError("structural-control runtime policy differs")
        validate_structural_control_scheduler_runtime(
            cfg.get("scheduler_runtime"),
            config_sha256_v2=base_scheduler_hash_v2,
        )

    img_dir = os.path.join(cfg["out_dir"], "images")
    commit = cfg["git_commit"]
    worker_contract_sha256 = worker_resume_contract_sha256(cfg)
    latent_runtime_cache = {}
    n_done = 0
    while True:
        if abort_event is not None and abort_event.is_set():
            break
        try:
            task = task_queue.get_nowait()
        except queue.Empty:
            break
        if task is None:
            break
        png_path = os.path.join(img_dir, task["id"] + ".png")
        json_path = os.path.join(img_dir, task["id"] + ".json")
        if task_is_complete(
            task,
            img_dir,
            run_contract_sha256=worker_contract_sha256,
            structural_config=cfg if cfg.get("structural_control_registered") else None,
        ):
            continue
        try:
            generator = torch.Generator(device).manual_seed(task["seed"])
            action = task["action"]
            controller, attn_scale, attn_density, attn_decay = guidance_runtime(
                action, cfg["num_inference_steps"]
            )
            clean_config = clean_transport_runtime(action)
            baseline_config = attention_baseline_runtime(action)
            freeu_schedule = freeu_runtime(action)
            freeu_implementation = freeu_implementation_runtime(action)
            trajectory_correction = trajectory_correction_runtime(action)
            # The diffusers FreeU implementation stores mutable attributes on
            # every up block.  Clear them before every action to preserve exact
            # paired comparisons, including after a failed task.
            pipe.unet.disable_freeu()
            if cfg["stage2_enabled"] and action["type"] == "latent_renderer_fixed":
                raise ValueError("latent renderer is registered for Stage 1 only")
            if action["type"] == "latent_renderer_fixed":
                latent_renderer, latent_provider = latent_runtime_cache.get(action["id"], (None, None))
                if latent_renderer is None:
                    latent_renderer, latent_provider = latent_renderer_runtime(
                        action, pipe, device, cfg["guidance_scale"]
                    )
                    latent_runtime_cache[action["id"]] = (latent_renderer, latent_provider)
            else:
                latent_renderer, latent_provider = None, None
            renderer_pipeline_kwargs = latent_renderer_pipeline_kwargs(
                action, latent_renderer, latent_provider
            )
            if cfg["stage2_enabled"] and clean_config is not None:
                raise ValueError("clean transport is registered for Stage 1 only")
            cfg_scale = float(action.get("cfg_scale", cfg["guidance_scale"]))
            guidance_rescale = float(cfg.get("guidance_rescale", 0.0))
            if cfg.get("cfg_baseline_registered"):
                expected_cfg_scale = CFG_BASELINE_ACTION_SCALES.get(action["id"])
                if expected_cfg_scale is None or cfg_scale != expected_cfg_scale:
                    raise RuntimeError(
                        f"CFG-baseline action {action['id']!r} has runtime scale drift"
                    )
            if cfg.get("structural_control_registered"):
                expected_cfg_scale = (
                    5.0
                    if action["id"]
                    in {
                        "cfg_only_5",
                        "pladis_operator_port",
                        "gag_eq13_reimplementation",
                    }
                    else 7.5
                )
                if cfg_scale != expected_cfg_scale:
                    raise RuntimeError(
                        f"structural-control action {action['id']!r} has CFG drift"
                    )
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            start_time = time.perf_counter()
            baseline_context = (
                installed_attention_baseline(pipe.unet, **baseline_config)
                if baseline_config is not None
                else nullcontext()
            )
            freeu_context = (
                installed_freeu_implementation(freeu_implementation)
                if freeu_schedule is not None
                else nullcontext()
            )
            scheduler_reference = action["type"] == "scheduler_baseline"
            # A registered matrix isolates every action from the mutable
            # scheduler state left by the previous pipeline call.  In
            # particular, Euler's ``init_noise_sigma`` property changes after
            # ``set_timesteps`` for the legacy SDXL scheduler config.
            scheduler_isolated = scheduler_isolation_required(cfg)
            scheduler_runtime = None
            if scheduler_isolated:
                scheduler_runtime = (
                    scheduler_baseline_runtime(action, base_scheduler)
                    if scheduler_reference
                    else native_scheduler_runtime(base_scheduler)
                )
                pipe.scheduler = scheduler_runtime
            elif scheduler_reference:
                scheduler_runtime = scheduler_baseline_runtime(action, base_scheduler)
                pipe.scheduler = scheduler_runtime
            active_scheduler = (
                scheduler_runtime if scheduler_runtime is not None else base_scheduler
            )
            scheduler_name = type(active_scheduler).__name__
            active_scheduler_hash = scheduler_config_sha256(active_scheduler)
            active_scheduler_hash_v2 = scheduler_config_sha256_v2(active_scheduler)
            active_scheduler_config = scheduler_config_payload(active_scheduler)
            active_scheduler_order = int(getattr(active_scheduler, "order", 1))
            active_solver_order = active_scheduler.config.get("solver_order")
            active_init_noise_sigma = float(active_scheduler.init_noise_sigma)
            validate_latent_renderer_scheduler(action, active_scheduler)
            schedule_provenance = {}
            if scheduler_isolated:
                schedule_provenance = prepare_scheduler_schedule_provenance(
                    active_scheduler,
                    cfg["num_inference_steps"],
                    device,
                )
                if cfg.get("cfg_baseline_registered"):
                    validate_cfg_scheduler_runtime(
                        cfg["scheduler_runtime"],
                        config_sha256_v2=active_scheduler_hash_v2,
                        schedule_provenance=schedule_provenance,
                    )
                if cfg.get("structural_control_registered"):
                    validate_structural_control_scheduler_runtime(
                        cfg["scheduler_runtime"],
                        config_sha256_v2=active_scheduler_hash_v2,
                        schedule_provenance=schedule_provenance,
                    )
            try:
                with baseline_context as attention_baseline_topology, freeu_context as freeu_operator_runtime:
                    images = pipe(
                        task["prompt"],
                        negative_prompt=cfg["negative_prompt"],
                        generator=generator,
                        height=cfg["resolution"], width=cfg["resolution"],
                        num_inference_steps=cfg["num_inference_steps"],
                        guidance_scale=cfg_scale,
                        guidance_rescale=guidance_rescale,
                        show_image=False,
                        multi_decoder=cfg["multi_decoder"],
                        multi_encoder=cfg["multi_encoder"],
                        models_to_cpu=cfg["models_to_cpu"],
                        num_resample_timesteps=cfg["num_resample_timesteps"],
                        init_rates=cfg["init_rates"],
                        attn_type="vanilla",
                        attn_guidance_scale=attn_scale,
                        attn_guidance_density=attn_density,
                        attn_guidance_decay=attn_decay,
                        power_calibrate=cfg["power_calibrate"],
                        attn_guidance_filter=None,
                        attn_guidance_controller=controller,
                        attn_guidance_band_cutoffs=tuple(cfg["frequency_band_cutoffs"]),
                        semantic_transport_config=clean_config,
                        freeu_schedule=freeu_schedule,
                        freeu_preserve_moments=bool(action.get("freeu_preserve_moments", False)),
                        trajectory_correction=trajectory_correction,
                        **renderer_pipeline_kwargs,
                    )
                if schedule_provenance:
                    observed_schedule_hash = json_sha256(
                        scheduler_schedule_payload(active_scheduler)
                    )
                    if observed_schedule_hash != schedule_provenance[
                        "scheduler_schedule_sha256"
                    ]:
                        raise RuntimeError(
                            "pipeline scheduler schedule differs from the registered ledger"
                        )
            finally:
                if scheduler_runtime is not None:
                    pipe.scheduler = base_scheduler
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
                peak_memory = int(torch.cuda.max_memory_allocated(device))
            else:
                peak_memory = None
            elapsed = time.perf_counter() - start_time
            observed_unet_calls = [
                int(value)
                for value in getattr(pipe, "_last_unet_calls_per_step", [])
            ]
            if trajectory_correction is not None:
                validate_one_unet_call_per_step(
                    observed_unet_calls,
                    cfg["num_inference_steps"],
                    run_label="trajectory correction",
                )
            if cfg.get("scheduler_baseline_registered"):
                validate_one_unet_call_per_step(
                    observed_unet_calls,
                    cfg["num_inference_steps"],
                    run_label="scheduler baselines",
                )
            if cfg.get("cfg_baseline_registered"):
                validate_one_unet_call_per_step(
                    observed_unet_calls,
                    cfg["num_inference_steps"],
                    run_label="CFG baselines",
                )
            validate_structural_control_worker_result(
                cfg, images, observed_unet_calls
            )
            if native_renderer_step_diagnostics_required(cfg, action):
                validate_one_unet_call_per_step(
                    observed_unet_calls,
                    cfg["num_inference_steps"],
                    run_label="scheduler-native renderer matrix",
                )
            observed_extra_unet_calls = max(
                (value - 1 for value in observed_unet_calls), default=0
            )
            diagnostics = getattr(pipe, "_last_guidance_diagnostics", None)
            renderer_sidecar = latent_renderer_sidecar_fields(
                action,
                pipe,
                observed_unet_calls,
                cfg["num_inference_steps"],
                schedule_provenance=schedule_provenance,
            )
            if native_renderer_step_diagnostics_required(cfg, action):
                validate_native_renderer_step_diagnostics(
                    renderer_sidecar["latent_renderer_step_diagnostics"],
                    cfg["num_inference_steps"],
                    max_update_ratio=float(action.get("max_update_ratio", 0.05)),
                    schedule_provenance=schedule_provenance,
                    provider_contract=action.get("latent_renderer_provider"),
                )
            atomic_save_png(images[-1], png_path)
            generated_image_sha256 = image_sha256(png_path)
            scheduler_reference_provenance = scheduler_provenance_record(
                action,
                include=bool(
                    scheduler_reference
                    or cfg.get("scheduler_baseline_registered")
                    or cfg.get("cfg_baseline_registered")
                    or cfg.get("native_renderer_registered")
                    or cfg.get("structural_control_registered")
                ),
                base_config_sha256_v2=base_scheduler_hash_v2,
                active_config_sha256_v2=active_scheduler_hash_v2,
                order=active_scheduler_order,
                solver_order=active_solver_order,
                init_noise_sigma=active_init_noise_sigma,
                schedule_provenance=schedule_provenance,
            )
            if cfg.get("cfg_baseline_registered") or cfg.get(
                "structural_control_registered"
            ):
                scheduler_reference_provenance.update(
                    {
                        "scheduler_config": base_scheduler_config,
                        "active_scheduler_config": active_scheduler_config,
                    }
                )
            record = {
                **task,
                "provenance_schema": PROVENANCE_SCHEMA,
                "run_contract_sha256": cfg["run_contract_sha256"],
                "config_sha256": cfg["run_contract_sha256"],
                "image_sha256": generated_image_sha256,
                "image_path": os.path.relpath(png_path, cfg["out_dir"]),
                "height": cfg["resolution"], "width": cfg["resolution"],
                "num_inference_steps": cfg["num_inference_steps"],
                "guidance_scale": cfg_scale,
                "guidance_rescale": guidance_rescale,
                "inference_seconds": elapsed,
                "peak_gpu_memory_bytes": peak_memory,
                "power_calibrate": cfg["power_calibrate"],
                "attn_guidance_scale": attn_scale,
                "attn_guidance_density": attn_density,
                "attn_guidance_decay": attn_decay,
                "attention_guidance_runtime": getattr(
                    pipe, "_last_attention_guidance_runtime", []
                ),
                "frequency_band_cutoffs": cfg["frequency_band_cutoffs"],
                "stage": cfg["stage_name"],
                "stage2_enabled": cfg["stage2_enabled"],
                "models_to_cpu": cfg["models_to_cpu"],
                "multi_encoder": cfg["multi_encoder"],
                "multi_decoder": cfg["multi_decoder"],
                "num_resample_timesteps": cfg["num_resample_timesteps"],
                "init_rates": cfg["init_rates"],
                "stage2_noise_source": cfg["stage2_noise_source"],
                "model_name": cfg["model_name"],
                "model_revision": cfg.get("model_revision"),
                "model_load_provenance": model_load_provenance,
                "scheduler_name": scheduler_name,
                "base_scheduler_name": base_scheduler_name,
                "scheduler_config_sha256": base_scheduler_hash,
                "active_scheduler_config_sha256": active_scheduler_hash,
                "scheduler_reference": scheduler_reference,
                "git_commit": commit,
                "registered_sampling": registered_sampling or None,
                "scheduler_runtime": cfg.get("scheduler_runtime"),
                "extra_unet_calls": observed_extra_unet_calls,
                "unet_calls_per_step": observed_unet_calls,
                **cfg["runtime_provenance"],
                "device": device,
                "worker_device_provenance": worker_device_provenance,
                "worker_determinism_provenance": worker_determinism,
                **structural_control_evidence_scope(
                    cfg.get("split_role")
                    if cfg.get("structural_control_registered")
                    else None
                ),
                **renderer_sidecar,
                "freeu_schedule": getattr(pipe, "_last_freeu_schedule", None),
                "freeu_runtime": getattr(pipe, "_last_freeu_runtime", []),
                "freeu_operator_runtime": freeu_operator_runtime,
                "freeu_preserve_moments": getattr(pipe, "_last_freeu_preserve_moments", False),
                "freeu_implementation": (
                    freeu_implementation if freeu_schedule is not None else None
                ),
                "freeu_source_commit": (
                    action.get("source_commit") if freeu_schedule is not None else None
                ),
                "freeu_implementation_diffusers_version": (
                    action.get("implementation_diffusers_version")
                    if freeu_schedule is not None
                    else None
                ),
                "attention_baseline_implementation": (
                    action.get("implementation")
                    if action["type"] == "attention_baseline"
                    else None
                ),
                "attention_baseline_source_commit": (
                    action.get("source_commit")
                    if action["type"] == "attention_baseline"
                    else None
                ),
                "attention_baseline_paper_id": (
                    action.get("paper_id")
                    if action["type"] == "attention_baseline"
                    else None
                ),
                "attention_baseline_topology": attention_baseline_topology,
                "trajectory_correction": getattr(
                    pipe, "_last_trajectory_correction", None
                ),
                "trajectory_correction_diagnostics": getattr(
                    pipe, "_last_trajectory_correction_diagnostics", None
                ),
                **scheduler_reference_provenance,
            }
            if diagnostics:
                record.update(diagnostics)
            else:
                record.update(
                    {
                        "semantic_transport_mode": None,
                        "semantic_transport_layer": None,
                        "semantic_transport_angle": None,
                        "semantic_transport_topk": None,
                        "semantic_transport_steps": 0,
                        "mean_normalized_affinity_entropy": None,
                        "mean_transport_confidence": None,
                        "mean_transport_scheduler_update_norm_ratio": None,
                        "mean_transport_update_norm": None,
                        "mean_scheduler_update_norm": None,
                    }
                )
            atomic_write_json(json_path, record)
            del images
            n_done += 1
        except Exception:
            error = traceback.format_exc()
            error_queue.put((task["id"], error))
            print(f"[{device}] FAILED task {task['id']}:\n{error}", flush=True)
            if cfg.get("structural_control_registered"):
                if abort_event is not None:
                    abort_event.set()
                break
    print(f"[{device}] done, generated {n_done} images", flush=True)


def worker_process_entry(cfg, device, task_queue, error_queue, abort_event=None):
    """Propagate worker-startup failures to sibling structural workers."""
    try:
        worker_process(cfg, device, task_queue, error_queue, abort_event)
    except BaseException:
        if abort_event is not None:
            abort_event.set()
        error_queue.put((f"worker:{device}", traceback.format_exc()))
        raise


def consolidate_manifest(
    out_dir: str,
    expected_ids=None,
    *,
    expected_tasks=None,
    run_contract_sha256: str | None = None,
    strict: bool = False,
    allow_partial: bool = False,
    structural_config: dict | None = None,
):
    img_dir = os.path.join(out_dir, "images")
    rows = []
    expected_task_map = {
        str(task["id"]): task for task in (expected_tasks or [])
    }
    expected_set = set(expected_ids or expected_task_map)
    observed_set = set()
    artifact_names = sorted(os.listdir(img_dir))
    if strict:
        unexpected_pngs = [
            fn
            for fn in artifact_names
            if fn.endswith(".png") and expected_set and fn[:-4] not in expected_set
        ]
        if unexpected_pngs:
            raise ValueError(
                f"unexpected PNG for unregistered task: {unexpected_pngs[0]}"
            )
    for fn in artifact_names:
        if fn.endswith(".json"):
            stem = fn[:-5]
            if expected_set and stem not in expected_set:
                if strict:
                    raise ValueError(f"unexpected sidecar for unregistered task: {fn}")
                continue
            with open(os.path.join(img_dir, fn)) as f:
                record = json.load(f)
            if record.get("id") != stem:
                raise ValueError(f"sidecar id does not match filename: {fn}")
            if run_contract_sha256 is not None:
                validate_sidecar(
                    record,
                    out_dir,
                    expected_task=expected_task_map.get(stem),
                    expected_contract_sha256=run_contract_sha256,
                )
                if structural_config is not None:
                    validate_structural_control_sidecar(
                        record,
                        structural_config,
                        expected_task=expected_task_map.get(stem),
                    )
            rows.append(record)
            observed_set.add(stem)
    if strict and not allow_partial and expected_set and observed_set != expected_set:
        raise ValueError(
            f"incomplete manifest: {len(expected_set - observed_set)} missing sidecars"
        )
    if strict and not allow_partial and expected_tasks and run_contract_sha256 is not None:
        validate_design_rows(
            rows,
            expected_action_ids=sorted({task["action_id"] for task in expected_tasks}),
            expected_seeds=sorted({int(task["seed"]) for task in expected_tasks}),
        )
    with atomic_text_writer(os.path.join(out_dir, "manifest.jsonl")) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def run_generation_locked(
    ap,
    args,
    *,
    stage_settings: dict,
    seeds: list[int],
    actions: list,
    band_cutoffs: list[float],
    legacy_scale_ids: bool,
    action_config: dict,
    devices: list[str],
    expected_prompts_sha256: str,
    expected_actions_sha256: str | None,
) -> None:
    """Execute all output reads and writes while the caller holds its lock."""
    prompts = pd.read_csv(args.prompts)
    observed_prompts_sha256 = sha256_file(args.prompts)
    if observed_prompts_sha256 != expected_prompts_sha256:
        ap.error("prompts file changed after locked validation")
    observed_actions_sha256 = sha256_file(args.actions) if args.actions else None
    if observed_actions_sha256 != expected_actions_sha256:
        ap.error("actions file changed after locked validation")
    assert "index" in prompts.columns and "TEXT" in prompts.columns, (
        "prompts CSV needs index,TEXT[,bucket]"
    )
    if prompts["index"].duplicated().any():
        ap.error("prompt indices must be unique")
    validate_model_cache(args.model_name, args.cache_dir)
    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)

    cfg = {
        "out_dir": args.out_dir,
        "resolution": args.resolution,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "power_calibrate": args.power_calibrate,
        "negative_prompt": args.negative_prompt,
        "cache_dir": args.cache_dir,
        "model_name": args.model_name,
        "low_vram": args.low_vram,
        "frequency_band_cutoffs": band_cutoffs,
        "git_commit": git_commit(),
        "split_role": args.split_role,
        "prompts_csv": os.path.abspath(args.prompts),
        "prompts_sha256": observed_prompts_sha256,
        "actions_yaml": os.path.abspath(args.actions) if args.actions else None,
        "actions_sha256": observed_actions_sha256,
        "runtime_provenance": runtime_provenance(
            action_config.get("_validated_environment_lock")
        ),
        **stage_settings,
    }
    action_schema = action_config.get("schema") if args.actions else None
    trajectory_registered = action_schema in TRAJECTORY_SCHEMAS
    scheduler_baseline_registered = action_schema == "scheduler_baselines_v1"
    cfg_baseline_registered = action_schema == CFG_BASELINE_SCHEMA
    structural_control_registered = action_schema == STRUCTURAL_CONTROL_SCHEMA
    # The normalized mapping is the capability contract. This keeps a future
    # native protocol strict even before it receives a dedicated frozen schema.
    native_renderer_registered = has_scheduler_native_renderer(actions)
    if native_renderer_registered:
        if action_schema != NATIVE_RENDERER_SCHEMA:
            ap.error(
                "scheduler-native renderer actions require the independently reviewed "
                f"{NATIVE_RENDERER_SCHEMA} executable schema"
            )
        sampling_registration = action_config.get("sampling")
        if not isinstance(sampling_registration, dict):
            ap.error(
                "scheduler-native renderer actions require a registered sampling mapping"
            )
        for required_key in (
            "model",
            "model_revision",
            "scheduler",
            "extra_unet_calls",
            "scheduler_churn",
        ):
            if required_key not in sampling_registration:
                ap.error(
                    "scheduler-native sampling registration is missing "
                    f"{required_key!r}"
                )
        authorization = action_config.get("authorization")
        if not isinstance(authorization, dict):
            ap.error("scheduler-native executable authorization metadata is missing")
        cfg["native_renderer_authorization"] = copy.deepcopy(authorization)
        cfg["native_renderer_registration_schema"] = NATIVE_RENDERER_SCHEMA
        cfg["native_renderer_source_template"] = authorization.get("source_template")
        cfg["native_renderer_source_template_sha256"] = authorization.get(
            "source_template_sha256"
        )
        # ``actions_sha256`` remains the executable-copy hash.  Record it under
        # a distinct name so an audit can retain both the executable and frozen
        # template identities without overloading the S7 contract field.
        cfg["native_renderer_executable_actions_sha256"] = observed_actions_sha256
    if structural_control_registered:
        authorization = action_config.get("authorization")
        implementation_source = action_config.get("implementation_source")
        analysis_implementation = action_config.get("analysis_implementation")
        if not isinstance(authorization, dict) or not isinstance(
            implementation_source, dict
        ) or not isinstance(analysis_implementation, dict):
            ap.error("structural-control authorization/source metadata is missing")
        cfg["structural_control_authorization"] = copy.deepcopy(authorization)
        cfg["structural_control_registration_schema"] = (
            STRUCTURAL_CONTROL_REGISTRATION_SCHEMA
        )
        cfg["structural_control_source_template"] = authorization.get(
            "source_template"
        )
        cfg["structural_control_source_template_sha256"] = authorization.get(
            "source_template_sha256"
        )
        cfg["structural_control_implementation_source"] = copy.deepcopy(
            implementation_source
        )
        cfg["structural_control_analysis_implementation"] = copy.deepcopy(
            analysis_implementation
        )
        cfg["structural_control_executable_actions_sha256"] = (
            observed_actions_sha256
        )
        cfg["structural_control_failure_policy"] = action_config.get(
            "failure_policy"
        )
        cfg.update(structural_control_evidence_scope(args.split_role))
    cfg["action_schema"] = action_schema
    cfg["trajectory_registered"] = trajectory_registered
    cfg["scheduler_baseline_registered"] = scheduler_baseline_registered
    cfg["cfg_baseline_registered"] = cfg_baseline_registered
    cfg["native_renderer_registered"] = native_renderer_registered
    cfg["structural_control_registered"] = structural_control_registered
    # A scheduler-native run is not auditable unless its scoring provenance is
    # registered independently.  The executable scoring config may provide the
    # expected payload/hash; the frozen registration files intentionally do not.
    cfg["scorer_provenance_binding_required"] = bool(
        native_renderer_registered or structural_control_registered
    )
    for key in (
        "registered_scorer_provenance",
        "expected_scorer_provenance",
        "registered_scorer_provenance_sha256",
        "expected_scorer_provenance_sha256",
        "scorer_provenance_sha256",
        "scoring",
        "scoring_contract",
        "executable_scoring",
    ):
        if key in action_config:
            cfg[key] = action_config[key]
    cfg["registered_sampling"] = (
        dict(action_config.get("sampling") or {})
        if trajectory_registered
        or scheduler_baseline_registered
        or cfg_baseline_registered
        or native_renderer_registered
        or structural_control_registered
        else {}
    )
    cfg["model_revision"] = (
        cfg["registered_sampling"].get("model_revision")
        if cfg_baseline_registered
        or native_renderer_registered
        or structural_control_registered
        else None
    )
    cfg["guidance_rescale"] = (
        float(cfg["registered_sampling"]["guidance_rescale"])
        if cfg_baseline_registered
        or native_renderer_registered
        or structural_control_registered
        else 0.0
    )
    cfg["scheduler_runtime"] = (
        dict(action_config.get("scheduler_runtime") or {})
        if cfg_baseline_registered or structural_control_registered
        else None
    )
    cfg["run_contract"] = generation_contract(
        cfg,
        actions=actions,
        seeds=seeds,
        prompts_sha256=cfg["prompts_sha256"],
        actions_sha256=cfg["actions_sha256"],
    )
    cfg["run_contract_sha256"] = json_sha256(cfg["run_contract"])
    # Native renderer and other registered runs require strict sidecar/contract
    # validation on resume. Existing legacy LR-1 sweeps retain their original
    # PNG+JSON completion semantics.
    registered_run = strict_registered_run(cfg)
    resume_contract_sha256 = cfg["run_contract_sha256"] if registered_run else None

    tasks = build_tasks(prompts, seeds, actions, legacy_scale_ids)
    expected_ids = {task["id"] for task in tasks}
    img_dir = os.path.join(args.out_dir, "images")
    config_path = os.path.join(args.out_dir, "config.json")
    if registered_run and os.path.exists(config_path):
        try:
            with open(config_path) as handle:
                previous_config = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            ap.error(f"cannot read existing run config: {exc}")
        try:
            validate_run_contract(previous_config)
        except ValueError as exc:
            ap.error(f"existing registered run config is invalid: {exc}")
        if previous_config.get("run_contract_sha256") != cfg["run_contract_sha256"]:
            ap.error(
                "existing registered run config differs from the requested "
                "model/scheduler/action/sampling contract; use a new output directory"
            )
        if native_renderer_registered:
            # Keep the legacy run-contract schema stable, but bind native
            # authorization metadata separately on resume.
            native_metadata_keys = (
                "native_renderer_registered",
                "native_renderer_registration_schema",
                "native_renderer_authorization",
                "native_renderer_source_template",
                "native_renderer_source_template_sha256",
                "native_renderer_executable_actions_sha256",
                "scorer_provenance_binding_required",
                "scoring",
            )
            for key in native_metadata_keys:
                if previous_config.get(key) != cfg.get(key):
                    ap.error(
                        "existing native renderer authorization metadata differs "
                        f"for {key!r}; use a new output directory"
                    )
        if structural_control_registered:
            structural_metadata_keys = (
                "structural_control_registered",
                "structural_control_registration_schema",
                "structural_control_authorization",
                "structural_control_source_template",
                "structural_control_source_template_sha256",
                "structural_control_implementation_source",
                "structural_control_analysis_implementation",
                "structural_control_executable_actions_sha256",
                "structural_control_failure_policy",
                "scorer_provenance_binding_required",
                "scoring",
            )
            for key in structural_metadata_keys:
                if previous_config.get(key) != cfg.get(key):
                    ap.error(
                        "existing structural-control authorization metadata differs "
                        f"for {key!r}; use a new output directory"
                    )
    elif (
        registered_run
        and not os.path.exists(config_path)
        and generation_artifacts_exist(img_dir)
    ):
        ap.error(
            "registered artifacts exist without config.json; use a new output directory"
        )
    cfg["devices"] = list(devices)
    structural_config = cfg if structural_control_registered else None
    device_tasks = assign_tasks_to_devices(
        tasks,
        devices,
        img_dir,
        run_contract_sha256=resume_contract_sha256,
        structural_config=structural_config,
    )
    # Assignment freezes each action's deterministic execution rank. Compute
    # completion only after that field exists so registered resumes reject a
    # sidecar whose order provenance drifted.
    todo = [
        task
        for task in tasks
        if not task_is_complete(
            task,
            img_dir,
            run_contract_sha256=resume_contract_sha256,
            structural_config=structural_config,
        )
    ]
    worker_count = len(device_tasks)
    print(
        f"{len(prompts)} prompts x {len(seeds)} seeds x {len(actions)} actions = "
        f"{len(tasks)} tasks; {len(tasks) - len(todo)} already done, "
        f"{len(todo)} to generate on {worker_count} GPU(s).",
        flush=True,
    )

    if todo:
        atomic_write_json(
            os.path.join(args.out_dir, "config.json"),
            {**cfg, "seeds": seeds, "actions": actions, "devices": devices},
            indent=2,
        )
        tmp.set_start_method("spawn", force=True)
        manager = tmp.Manager()
        error_queue = manager.Queue()
        abort_event = manager.Event() if structural_control_registered else None
        active_devices = list(device_tasks)
        task_queues = [manager.Queue() for _ in active_devices]
        for device, task_queue in zip(active_devices, task_queues):
            for task in device_tasks[device]:
                task_queue.put(task)
        procs = []
        for device, task_queue in zip(active_devices, task_queues):
            process = tmp.Process(
                target=worker_process_entry,
                args=(cfg, device, task_queue, error_queue, abort_event),
            )
            process.start()
            procs.append((device, process))
        for _, process in procs:
            process.join()

        worker_failures = [
            (device, process.exitcode)
            for device, process in procs
            if process.exitcode != 0
        ]
        task_failures = []
        while True:
            try:
                task_failures.append(error_queue.get_nowait())
            except Exception:
                break
        if worker_failures or task_failures:
            completed = consolidate_manifest(
                args.out_dir,
                expected_ids,
                expected_tasks=tasks,
                run_contract_sha256=resume_contract_sha256,
                strict=registered_run,
                allow_partial=True,
                structural_config=structural_config,
            )
            examples = ", ".join(task_id for task_id, _ in task_failures[:5])
            raise RuntimeError(
                f"generation failed: workers={worker_failures}, "
                f"task_failures={len(task_failures)} ({examples}); "
                f"preserved {completed} completed records for resume"
            )

    completed = consolidate_manifest(
        args.out_dir,
        expected_ids,
        expected_tasks=tasks,
        run_contract_sha256=resume_contract_sha256,
        strict=registered_run,
        structural_config=structural_config,
    )
    print(
        f"manifest.jsonl written with {completed} records -> "
        f"{os.path.join(args.out_dir, 'manifest.jsonl')}",
        flush=True,
    )


def prepare_generation_locked(ap, args, stage_settings: dict, seeds: list[int]) -> None:
    """Load and validate every file-backed input while holding the output lock."""
    if args.actions and args.scales:
        ap.error("--actions and --scales are mutually exclusive")
    expected_prompts_sha256 = sha256_file(args.prompts)
    expected_actions_sha256 = sha256_file(args.actions) if args.actions else None
    action_config = {}
    if args.actions:
        actions, band_cutoffs = load_actions(args.actions, args.num_inference_steps)
        try:
            with open(args.actions) as action_handle:
                action_config = yaml.safe_load(action_handle) or {}
            action_config["_validated_environment_lock"] = (
                validate_registered_environment_lock(action_config)
            )
            validate_scheduler_baseline_authorization(args.actions)
            validate_cfg_baseline_authorization(args.actions)
            validate_native_renderer_authorization(args.actions)
            validate_structural_control_authorization(args.actions)
            validate_split_seed_role(args.actions, args.split_role, seeds)
            validate_structural_control_design(
                args.actions,
                prompts_path=args.prompts,
                actions=actions,
                seeds=seeds,
                model_name=args.model_name,
                resolution=args.resolution,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                negative_prompt=args.negative_prompt,
                power_calibrate=args.power_calibrate,
                stage2_enabled=args.stage2,
                split_role=args.split_role,
                low_vram=args.low_vram,
            )
            validate_cfg_baseline_design(
                args.actions,
                prompts_path=args.prompts,
                actions=actions,
                seeds=seeds,
                model_name=args.model_name,
                resolution=args.resolution,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                negative_prompt=args.negative_prompt,
                power_calibrate=args.power_calibrate,
                stage2_enabled=args.stage2,
            )
            validate_scheduler_baseline_design(
                args.actions,
                prompts_path=args.prompts,
                actions=actions,
                seeds=seeds,
                model_name=args.model_name,
                resolution=args.resolution,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                stage2_enabled=args.stage2,
            )
            validate_registered_trajectory_design(
                args.actions,
                prompts_path=args.prompts,
                resolution=args.resolution,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                stage2_enabled=args.stage2,
                model_name=args.model_name,
            )
            if (
                args.split_role == "test_final"
                and action_config.get("schema") == "latent_renderer_actions_v1"
            ):
                if not args.authorization:
                    raise ValueError(
                        "latent-renderer test_final requires --authorization from the validation gate"
                    )
                validate_final_test_authorization(
                    args.authorization, args.actions, seeds
                )
        except ValueError as exc:
            ap.error(str(exc))
        legacy_scale_ids = False
    else:
        if args.split_role is not None:
            ap.error("--split_role requires --actions")
        scale_values = args.scales or "0,0.001,0.002,0.003,0.005"
        scales = [float(s) for s in scale_values.split(",") if s != ""]
        actions = scale_actions(scales)
        band_cutoffs = [0.08, 0.25]
        legacy_scale_ids = True
    devices = [
        f"cuda:{value.strip()}"
        for value in args.devices.split(",")
        if value.strip()
    ]
    if not devices:
        ap.error("no devices")
    if len(devices) != len(set(devices)):
        ap.error("--devices must not contain duplicates")
    if sha256_file(args.prompts) != expected_prompts_sha256:
        ap.error("prompts file changed during locked validation")
    if args.actions and sha256_file(args.actions) != expected_actions_sha256:
        ap.error("actions file changed during locked validation")

    run_generation_locked(
        ap,
        args,
        stage_settings=stage_settings,
        seeds=seeds,
        actions=actions,
        band_cutoffs=band_cutoffs,
        legacy_scale_ids=legacy_scale_ids,
        action_config=action_config,
        devices=devices,
        expected_prompts_sha256=expected_prompts_sha256,
        expected_actions_sha256=expected_actions_sha256,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="CSV with columns index,TEXT,bucket")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--devices", default="0", help="comma-separated GPU indices, e.g. 6,7")
    ap.add_argument("--scales", default=None,
                    help="comma-separated constant attn_guidance_scale values")
    ap.add_argument("--actions", default=None,
                    help="YAML action grid; mutually exclusive with --scales")
    ap.add_argument(
        "--split_role",
        default=None,
        help="registered split_seeds role, e.g. train_search or validation_confirmation",
    )
    ap.add_argument(
        "--authorization",
        default=None,
        help="final-test authorization JSON required for latent-renderer test_final runs",
    )
    ap.add_argument("--seeds", default="0,42,123", help="comma-separated seeds")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument(
        "--stage2",
        action="store_true",
        help="explicitly enable RepLDM high-resolution resampling for resolution > 1024",
    )
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--power_calibrate", type=int, default=0)
    ap.add_argument("--negative_prompt", default=DEFAULT_NEG)
    ap.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--model_name", default=DEFAULT_MODEL)
    ap.add_argument("--low_vram", action="store_true",
                    help="offload models to CPU between phases (models_to_cpu=True); use on busy GPUs")
    args = ap.parse_args()

    try:
        stage_settings = generation_stage_settings(
            args.stage2, args.resolution, args.low_vram
        )
    except ValueError as exc:
        ap.error(str(exc))
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    if len(seeds) != len(set(seeds)):
        ap.error("--seeds must not contain duplicates")

    with generation_output_lock(args.out_dir):
        prepare_generation_locked(ap, args, stage_settings, seeds)


if __name__ == "__main__":
    main()
