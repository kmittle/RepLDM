"""Run the CPU-only LR-0 synthetic mechanism audit.

This command exercises the renderer contract without loading SDXL weights or
looking at image-quality scores.  It is a correctness gate, not evidence of a
method gain.  The JSON report is suitable for attaching to an experiment
manifest and records the exact tensor shape, seed, device, and capacity counts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from AttentionGuidance import (  # noqa: E402
    LatentRendererConfig,
    StructuralLatentRenderer,
    build_graph_transport_basis,
    build_spectral_bases,
    inject_rendered_clean_update,
)


def run_audit(
    seed: int = 1729, device: str = "cpu", height: int = 16, width: int = 16
) -> Dict:
    if height <= 1 or width <= 1:
        raise ValueError("height and width must be greater than one")
    torch.manual_seed(seed)
    target = torch.device(device)
    latent = torch.randn(2, 4, height, width, device=target)
    scheduler_update = torch.randn_like(latent)
    spectral = build_spectral_bases(latent)
    identity_graph = torch.eye(height * width, device=target).expand(2, -1, -1)
    semantic = build_graph_transport_basis(latent, identity_graph, height, width)[:, 0]
    laplacian = latent - torch.nn.functional.avg_pool2d(
        torch.nn.functional.pad(latent, (1, 1, 1, 1), mode="reflect"), 3, 1
    )
    bases = torch.cat(
        (
            semantic[:, None],
            spectral,
            torch.zeros_like(semantic[:, None]),
            laplacian[:, None],
        ),
        dim=1,
    )
    # The fifth basis is a deterministic zero placeholder for a projected
    # backbone-minus-skip feature in this synthetic audit; the sixth is the
    # Laplacian basis. This preserves the registered basis order.
    renderer = StructuralLatentRenderer(
        LatentRendererConfig(
            num_bases=6,
            prompt_dim=32,
            state_dim=16,
            hidden_dim=256,
            depth=2,
            max_update_ratio=0.05,
        )
    ).to(target)
    renderer.policy[-1].bias.data.copy_(
        torch.tensor([0.20, -0.10, 0.05, 0.15, 0.08, -0.06], device=target)
    )
    prompt = torch.randn(2, 32, device=target)
    state = torch.randn(2, 16, device=target)
    output = renderer(
        latent,
        bases,
        timestep=torch.tensor([0.2, 0.8], device=target),
        prompt_embedding=prompt,
        state_features=state,
        scheduler_update=scheduler_update,
    )
    flipped = renderer(
        torch.flip(latent, dims=(-1,)),
        torch.flip(bases, dims=(-1,)),
        timestep=torch.tensor([0.2, 0.8], device=target),
        prompt_embedding=prompt,
        state_features=state,
        scheduler_update=torch.flip(scheduler_update, dims=(-1,)),
    )
    identity_renderer = StructuralLatentRenderer(
        LatentRendererConfig(num_bases=6, max_update_ratio=0.05)
    ).to(target)
    identity_output = identity_renderer(
        latent,
        bases,
        scheduler_update=scheduler_update,
    )
    injected = inject_rendered_clean_update(
        latent, latent, output.guided_x0
    )
    partition_error = float((spectral.sum(dim=1) - latent).abs().max().cpu())
    flip_error = float(
        (
            flipped.guided_x0 - torch.flip(output.guided_x0, dims=(-1,))
        ).abs().max().cpu()
    )
    identity_error = float((identity_output.guided_x0 - latent).abs().max().cpu())
    injection_error = float((injected - output.guided_x0).abs().max().cpu())
    report = {
        "schema": "latent_renderer_lr0_report_v1",
        "seed": int(seed),
        "device": str(target),
        "shape": list(latent.shape),
        "partition_error": partition_error,
        "flip_equivariance_error": flip_error,
        "zero_initialization_error": identity_error,
        "scheduler_injection_error": injection_error,
        "max_update_ratio": float(output.diagnostics.update_ratio.max().cpu()),
        "max_mean_error": float(output.diagnostics.mean_error.max().cpu()),
        "max_variance_error": float(output.diagnostics.variance_error.max().cpu()),
        "parameter_count": renderer.parameter_count,
        "finite": bool(torch.isfinite(output.guided_x0).all().cpu()),
        "checks": {
            "spectral_partition": partition_error <= 1e-4,
            "flip_equivariance": flip_error <= 1e-4,
            "zero_initialization": identity_error == 0.0,
            "scheduler_injection": injection_error <= 1e-6,
            "trust_region": float(output.diagnostics.update_ratio.max().cpu()) <= 0.05001,
            "moment_mean": float(output.diagnostics.mean_error.max().cpu()) <= 1e-4,
            "moment_variance": float(output.diagnostics.variance_error.max().cpu()) <= 1e-3,
            "finite": bool(torch.isfinite(output.guided_x0).all().cpu()),
        },
    }
    report["passed"] = all(report["checks"].values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--output", help="optional JSON report path")
    args = parser.parse_args()
    report = run_audit(args.seed, args.device, args.height, args.width)
    if args.output:
        parent = os.path.dirname(os.path.abspath(args.output))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.output, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
