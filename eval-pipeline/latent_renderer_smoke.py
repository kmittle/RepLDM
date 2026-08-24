"""Run a single-prompt Stage-1 renderer wiring smoke with cached SDXL.

This is an engineering check only.  It compares no renderer, a zero
renderer, and a deterministic non-zero probe renderer at the same seed and
NFE.  It does not select coefficients, train a policy, or support a quality
claim.  The model must already exist in the local Hugging Face cache.
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from AttentionGuidance import (  # noqa: E402
    LatentRendererConfig,
    RendererCondition,
    RendererObservation,
    StructuralLatentRenderer,
    build_laplacian_basis,
    build_spectral_bases,
)
from InferencePipelines import RepLDMSDXLPipeline  # noqa: E402


class SpectralLaplacianProvider:
    """Use deterministic placeholder semantic/FreeU bases for wiring only."""

    def __call__(self, observation: RendererObservation) -> RendererCondition:
        x0 = observation.pred_original_sample
        spectral = build_spectral_bases(x0)
        laplacian = build_laplacian_basis(x0)[:, 0]
        placeholder = torch.zeros_like(laplacian)
        bases = torch.cat(
            (
                placeholder[:, None],
                spectral,
                placeholder[:, None],
                laplacian[:, None],
            ),
            dim=1,
        )
        return RendererCondition(bases=bases)


def _hash_image(image) -> str:
    import numpy as np

    return hashlib.sha256(np.asarray(image).tobytes()).hexdigest()


def _make_renderer(device: torch.device, probe: bool) -> StructuralLatentRenderer:
    renderer = StructuralLatentRenderer(
        LatentRendererConfig(
            num_bases=6,
            latent_channels=4,
            hidden_dim=256,
            depth=2,
            spatial_hidden_dim=32,
            max_update_ratio=0.05,
        )
    ).to(device)
    if probe:
        # A fixed, tiny non-zero probe verifies the scheduler injection path;
        # it is not a learned checkpoint and must never be scored as a method.
        renderer.policy[-1].bias.data.copy_(
            torch.tensor([0.02, -0.01, 0.01, 0.015, 0.01, -0.01], device=device)
        )
        renderer.spatial_head["output"].bias.data.fill_(0.005)
    return renderer


def _run(pipe, prompt: str, seed: int, steps: int, renderer=None):
    generator = torch.Generator(pipe._execution_device).manual_seed(seed)
    return pipe(
        prompt,
        negative_prompt="blurry, ugly, duplicate, poorly drawn, deformed, mosaic",
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
        latent_renderer_basis_provider=SpectralLaplacianProvider()
        if renderer is not None
        else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--cache_dir", default="pretrained_ckpts")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--prompt", default="a red kite above a quiet alpine lake at dawn")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--out_dir", default="outputs/latent_renderer/wiring_smoke_v1")
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the wiring smoke requires a CUDA device")
    os.makedirs(args.out_dir, exist_ok=True)
    pipe = RepLDMSDXLPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=args.cache_dir,
        local_files_only=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    baseline = _run(pipe, args.prompt, args.seed, args.steps)
    zero_renderer = _make_renderer(device, probe=False)
    zero = _run(pipe, args.prompt, args.seed, args.steps, zero_renderer)
    probe_renderer = _make_renderer(device, probe=True)
    probe = _run(pipe, args.prompt, args.seed, args.steps, probe_renderer)
    images = {
        "no_renderer": baseline[-1],
        "zero_renderer": zero[-1],
        "probe_renderer": probe[-1],
    }
    hashes: Dict[str, str] = {}
    for name, image in images.items():
        image.save(os.path.join(args.out_dir, name + ".png"))
        hashes[name] = _hash_image(image)
    diagnostics = pipe._last_latent_renderer_diagnostics
    diagnostic_summary = None
    if diagnostics is not None:
        diagnostic_summary = diagnostics.to_record()
    report = {
        "schema": "latent_renderer_wiring_smoke_v1",
        "git_commit": subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "--short", "HEAD"]
        ).decode().strip(),
        "model": args.model,
        "device": str(device),
        "prompt": args.prompt,
        "seed": args.seed,
        "steps": args.steps,
        "hashes": hashes,
        "zero_matches_no_renderer": hashes["zero_renderer"] == hashes["no_renderer"],
        "probe_differs_from_no_renderer": hashes["probe_renderer"] != hashes["no_renderer"],
        "last_step_diagnostics": diagnostic_summary,
    }
    with open(os.path.join(args.out_dir, "report.json"), "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["zero_matches_no_renderer"] or not report["probe_differs_from_no_renderer"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
