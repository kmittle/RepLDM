"""Run a small, result-blind component smoke for the native renderer.

This script exercises the reviewed provider contract on a real SDXL forward.
It is intentionally not a scorer or an action selector: the only claims it
can establish are zero-action parity, requested-basis laziness, one U-Net call
per step, finite Euler mapping diagnostics, and a non-zero plumbing probe. It
tests only two nonzero actions and does not replace the executable grid audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from typing import Dict

import torch
from diffusers import EulerDiscreteScheduler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from AttentionGuidance import (  # noqa: E402
    LAZY_LATENT_STRUCTURE_BASIS_NAMES,
    LazyLatentStructureBasisProvider,
    build_fixed_coefficient_renderer,
)
from InferencePipelines import RepLDMSDXLPipeline  # noqa: E402


NEGATIVE_PROMPT = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"
ACTION_COEFFICIENTS = {
    "lazy_zero_identity": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "spectral_low_native": (0.0, 0.02, 0.0, 0.0, 0.0, 0.0),
    "laplacian_native": (0.0, 0.0, 0.0, 0.0, 0.0, 0.02),
}
ACTION_BASES = {
    "lazy_zero_identity": (),
    "spectral_low_native": ("spectral_low",),
    "laplacian_native": ("laplacian",),
}


def image_hash(path: str) -> str:
    """Hash the saved PNG bytes, matching the strict run auditor."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fresh_scheduler(base_scheduler):
    """Reset mutable Euler state before every paired action."""
    return EulerDiscreteScheduler.from_config(base_scheduler.config)


def run_action(pipe, base_scheduler, action_id: str, prompt: str, seed: int, steps: int):
    pipe.scheduler = fresh_scheduler(base_scheduler)
    device = pipe._execution_device
    generator = torch.Generator(device).manual_seed(seed)
    if action_id == "no_op":
        renderer = None
        provider = None
        mapping = "legacy_unit"
    else:
        coefficients = ACTION_COEFFICIENTS[action_id]
        renderer = build_fixed_coefficient_renderer(
            coefficients,
            latent_channels=4,
            coefficient_bound=1.0,
            max_update_ratio=0.05,
            preserve_moments=True,
            basis_normalization="match_rms",
        ).to(device)
        requested = ACTION_BASES[action_id]
        provider = LazyLatentStructureBasisProvider(
            pipe.unet,
            batch_size=1,
            do_classifier_free_guidance=True,
            latent_channels=4,
            requested_bases=requested,
            required_hook_names=None,
            semantic_mode="reciprocal_semantic",
            semantic_topk=16,
            prompt_dim=0,
            state_dim=0,
            scheduler_mapping="euler_clean_endpoint",
            basis_normalization="match_rms",
            provider_id="lazy_action_requested_bases_v1",
        )
        mapping = "euler_clean_endpoint"
    image = pipe(
        prompt,
        negative_prompt=NEGATIVE_PROMPT,
        generator=generator,
        height=1024,
        width=1024,
        num_inference_steps=steps,
        guidance_scale=7.5,
        show_image=False,
        multi_decoder=False,
        models_to_cpu=False,
        multi_encoder=False,
        attn_guidance_scale=0.0,
        latent_renderer=renderer,
        latent_renderer_basis_provider=provider,
        latent_renderer_scheduler_mapping=mapping,
    )[-1]
    calls = list(getattr(pipe, "_last_unet_calls_per_step", []))
    image_path = os.path.join(run_action.out_dir, f"{action_id}.png")
    image.save(image_path)
    record = {
        "action_id": action_id,
        "hash": image_hash(image_path),
        "unet_calls_per_step": calls,
        "provider_diagnostics": (
            None if provider is None else provider.last_diagnostics
        ),
        "step_diagnostics": (
            None
            if provider is None
            else list(getattr(pipe, "_last_latent_renderer_step_diagnostics", []))
        ),
        "runtime_mapping": getattr(pipe, "_last_latent_renderer_scheduler_mapping", None),
    }
    return record


def validate_record(record: dict, steps: int) -> None:
    """Apply only engineering guards; no score or ranking is read here."""
    action_id = record["action_id"]
    if record["unet_calls_per_step"] != [1] * steps:
        raise RuntimeError(f"{action_id}: expected one U-Net call per step")
    if action_id == "no_op":
        return
    provider = record["provider_diagnostics"]
    if not isinstance(provider, dict):
        raise RuntimeError(f"{action_id}: provider diagnostics are missing")
    expected = list(ACTION_BASES[action_id])
    if provider.get("requested_bases") != expected:
        raise RuntimeError(f"{action_id}: requested basis set drifted")
    if provider.get("constructed_bases") != expected:
        raise RuntimeError(f"{action_id}: constructed basis set drifted")
    if provider.get("registered_hook_names") != []:
        raise RuntimeError(f"{action_id}: non-attention smoke registered a hook")
    if provider.get("required_hook_names") != []:
        raise RuntimeError(f"{action_id}: non-attention smoke requires a hook")
    basis_rms = provider.get("basis_rms")
    if not isinstance(basis_rms, list) or len(basis_rms) != 1:
        raise RuntimeError(f"{action_id}: basis_rms shape is invalid")
    expected_index = {
        name: index for index, name in enumerate(LAZY_LATENT_STRUCTURE_BASIS_NAMES)
    }
    for index, value in enumerate(basis_rms[0]):
        if not torch.isfinite(torch.tensor(float(value))) or float(value) < 0:
            raise RuntimeError(f"{action_id}: non-finite basis RMS")
        if LAZY_LATENT_STRUCTURE_BASIS_NAMES[index] not in expected:
            if float(value) != 0.0:
                raise RuntimeError(f"{action_id}: unused basis slot is non-zero")
    ledger = record["step_diagnostics"]
    if not isinstance(ledger, list) or len(ledger) != steps:
        raise RuntimeError(f"{action_id}: native step ledger is incomplete")
    for step_index, step in enumerate(ledger):
        if step.get("step_index") != step_index:
            raise RuntimeError(f"{action_id}: step index drifted")
        gain = float(step["clean_update_gain"][0])
        ratio = float(step["applied_update_ratio"][0])
        sigma_from = float(step["sigma_from"])
        sigma_to = float(step["sigma_to"])
        if not (0.0 < gain <= 1.0) or abs(gain - (1.0 - sigma_to / sigma_from)) > 1e-5:
            raise RuntimeError(f"{action_id}: invalid Euler clean gain")
        if not (0.0 <= ratio <= 0.05 + 1e-6):
            raise RuntimeError(f"{action_id}: trust ratio exceeded")
        if abs(float(step["mean_error"][0])) > 1e-4:
            raise RuntimeError(f"{action_id}: mean preservation failed")
        if abs(float(step["variance_error"][0])) > 1e-3:
            raise RuntimeError(f"{action_id}: variance preservation failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--cache_dir", default="pretrained_ckpts")
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--prompt", default="a red kite above a quiet alpine lake at dawn")
    parser.add_argument("--seed", type=int, default=1798464083)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--out_dir",
        default="outputs/latent_renderer/scheduler_native_fixed_headroom_smoke_v1",
    )
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the scheduler-native smoke requires CUDA")
    os.makedirs(args.out_dir, exist_ok=True)
    pipe = RepLDMSDXLPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=args.cache_dir,
        local_files_only=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    base_scheduler = pipe.scheduler
    run_action.out_dir = args.out_dir
    records: Dict[str, dict] = {}
    with torch.no_grad():
        records["no_op"] = run_action(
            pipe, base_scheduler, "no_op", args.prompt, args.seed, args.steps
        )
        for action_id in ACTION_COEFFICIENTS:
            records[action_id] = run_action(
                pipe, base_scheduler, action_id, args.prompt, args.seed, args.steps
            )
    for record in records.values():
        validate_record(record, args.steps)
    if records["lazy_zero_identity"]["hash"] != records["no_op"]["hash"]:
        raise RuntimeError("lazy zero identity is not byte-identical to no-op")
    if records["spectral_low_native"]["hash"] == records["no_op"]["hash"]:
        raise RuntimeError("spectral probe did not change the decoded image")
    if records["laplacian_native"]["hash"] == records["no_op"]["hash"]:
        raise RuntimeError("laplacian probe did not change the decoded image")
    report = {
        "schema": "scheduler_native_fixed_headroom_smoke_v1",
        "git_commit": subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(
                ["git", "-C", ROOT, "status", "--porcelain"], text=True
            ).strip()
        ),
        "model": args.model,
        "device": str(device),
        "prompt": args.prompt,
        "seed": args.seed,
        "steps": args.steps,
        "records": records,
        "zero_matches_no_op": True,
        "nonzero_probes_differ": True,
        "quality_claim_allowed": False,
        "formal_matrix_evidence": False,
    }
    with open(os.path.join(args.out_dir, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
