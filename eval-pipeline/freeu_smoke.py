"""Compare SDXL no-op and static FreeU feature reweighting.

This is an engineering/diagnostic runner.  It uses the same seed and prompt
for every setting and records image hashes plus simple pixel diagnostics.  It
does not select a paper result; the action grid belongs in a separately frozen
evaluation manifest once a setting has shown a signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from InferencePipelines import RepLDMSDXLPipeline  # noqa: E402


SETTINGS: Dict[str, Optional[Tuple[float, float, float, float]]] = {
    "no_freeu": None,
    # Historical diffusers constant-gain preset. This is neither the adaptive
    # paper operator nor the FreeU README's SDXL parameter set.
    "freeu_sdxl": (0.6, 0.4, 1.1, 1.2),
    "freeu_balanced": (0.8, 0.8, 1.1, 1.1),
}


def image_hash(image) -> str:
    return hashlib.sha256(np.asarray(image).tobytes()).hexdigest()


def pixel_summary(image) -> dict:
    array = np.asarray(image).astype(np.float32) / 255.0
    clipped = ((array <= 0.0) | (array >= 1.0)).mean()
    saturation = (array.max(axis=-1) - array.min(axis=-1)).mean()
    return {"clipped_fraction": float(clipped), "mean_saturation": float(saturation)}


def run(pipe, prompt: str, seed: int, steps: int, setting):
    pipe.unet.disable_freeu()
    if setting is not None:
        pipe.unet.enable_freeu(*setting)
    generator = torch.Generator(pipe._execution_device).manual_seed(seed)
    result = pipe(
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
    )
    pipe.unet.disable_freeu()
    return result[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--cache_dir", default="pretrained_ckpts")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--prompt", default="a red kite above a quiet alpine lake at dawn")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--out_dir", default="outputs/freeu_smoke_v1")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    pipe = RepLDMSDXLPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=args.cache_dir,
        local_files_only=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    report = {
        "schema": "freeu_smoke_v1",
        "prompt": args.prompt,
        "seed": args.seed,
        "steps": args.steps,
        "device": str(device),
        "settings": {},
    }
    for name, setting in SETTINGS.items():
        image = run(pipe, args.prompt, args.seed, args.steps, setting)
        image.save(os.path.join(args.out_dir, name + ".png"))
        report["settings"][name] = {
            "parameters": setting,
            "sha256": image_hash(image),
            **pixel_summary(image),
        }
    with open(os.path.join(args.out_dir, "report.json"), "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
