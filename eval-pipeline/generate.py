"""Stage-1 generation stage of the RepLDM eval pipeline (env: `repldm`).

Generates a Stage-1 (1024^2) image for every (prompt x seed x guidance-scale)
combination and writes a lossless PNG + a per-image sidecar JSON manifest
recording the *full* guidance config. Scoring is a separate stage (score.py,
env `sana_cby`) that reads these PNGs + manifests — see eval-pipeline/README.md.

Design notes (grounded in the actual pipeline, EXPERIMENT_PLAN §4/§13):
  * Stage-1-only: height=width=1024 -> `height*width <= 1024**2` so Stage 2 is
    skipped (the pipeline returns just the Stage-1 image). Guidance acts
    only in Stage 1, so this is the regime we evaluate.
  * Constant per-step scale: attn_guidance_density="all" (every step) +
    attn_guidance_decay=None. attn_guidance_scale=0 reproduces the no-guidance
    baseline exactly (the pipeline applies guidance only when scale > 0).
  * AttnGuidance is re-instantiated inside every __call__, so the
    one-shot-iterator non-reentrancy (RISK-12) does NOT affect eval: each image
    is an independent generation.
  * Multi-GPU via a shared task queue (one worker per device), mirroring
    InferCases/RepLDM/SDXL/t2i_infer_imgs.py. Resume-safe: a (prompt,seed,scale)
    whose PNG already exists is skipped.

Example:
  conda run -n repldm python eval-pipeline/generate.py \
      --devices 6 --prompts eval-pipeline/prompts/eval_v1.csv \
      --out_dir outputs/exp1.1_scale_sweep/pilot \
      --scales 0,0.001,0.002,0.003,0.005 --seeds 0,42,123 --low_vram
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import traceback

import pandas as pd
import torch
import torch.multiprocessing as tmp

from InferencePipelines import RepLDMSDXLPipeline

DEFAULT_CACHE_DIR = "/mnt/miah204/bycao/RepLDM/pretrained_ckpts"
DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_NEG = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"


def task_id(prompt_index: int, seed: int, scale: float) -> str:
    return f"p{prompt_index}_seed{seed}_s{scale:.4f}"


def build_tasks(prompts: pd.DataFrame, seeds, scales) -> list:
    tasks = []
    for _, row in prompts.iterrows():
        pid = int(row["index"])
        for seed in seeds:
            for scale in scales:
                tasks.append(
                    {
                        "id": task_id(pid, seed, scale),
                        "prompt_index": pid,
                        "prompt": str(row["TEXT"]),
                        "bucket": str(row.get("bucket", "")),
                        "seed": int(seed),
                        "scale": float(scale),
                    }
                )
    return tasks


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def worker_process(cfg: dict, tasks_done_marker_dir: str, device: str, task_queue):
    torch.cuda.set_device(device)
    pipe = RepLDMSDXLPipeline.from_pretrained(
        cfg["model_name"], torch_dtype=torch.float16, variant="fp16",
        cache_dir=cfg["cache_dir"], local_files_only=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    img_dir = os.path.join(cfg["out_dir"], "images")
    commit = cfg["git_commit"]
    n_done = 0
    while True:
        try:
            task = task_queue.get_nowait()
        except Exception:
            break
        if task is None:
            break
        png_path = os.path.join(img_dir, task["id"] + ".png")
        json_path = os.path.join(img_dir, task["id"] + ".json")
        if os.path.exists(png_path) and os.path.exists(json_path):
            continue
        try:
            generator = torch.Generator(device).manual_seed(task["seed"])
            images = pipe(
                task["prompt"],
                negative_prompt=cfg["negative_prompt"],
                generator=generator,
                height=cfg["resolution"], width=cfg["resolution"],
                num_inference_steps=cfg["num_inference_steps"],
                guidance_scale=cfg["guidance_scale"],
                show_image=False,
                multi_decoder=False, multi_encoder=False,
                models_to_cpu=cfg["low_vram"],
                attn_type="vanilla",
                attn_guidance_scale=task["scale"],
                attn_guidance_density="all",
                attn_guidance_decay=None,
                power_calibrate=cfg["power_calibrate"],
                attn_guidance_filter=None,
            )
            images[-1].save(png_path)  # lossless PNG
            record = {
                **task,
                "image_path": os.path.relpath(png_path, cfg["out_dir"]),
                "height": cfg["resolution"], "width": cfg["resolution"],
                "num_inference_steps": cfg["num_inference_steps"],
                "guidance_scale": cfg["guidance_scale"],
                "power_calibrate": cfg["power_calibrate"],
                "attn_guidance_density": "all",
                "attn_guidance_decay": None,
                "stage": "stage1_1024",
                "model_name": cfg["model_name"],
                "git_commit": commit,
                "device": device,
            }
            with open(json_path, "w") as f:
                json.dump(record, f)
            del images
            n_done += 1
        except Exception:
            print(f"[{device}] FAILED task {task['id']}:\n{traceback.format_exc()}", flush=True)
    print(f"[{device}] done, generated {n_done} images", flush=True)


def consolidate_manifest(out_dir: str):
    img_dir = os.path.join(out_dir, "images")
    rows = []
    for fn in sorted(os.listdir(img_dir)):
        if fn.endswith(".json"):
            with open(os.path.join(img_dir, fn)) as f:
                rows.append(json.load(f))
    with open(os.path.join(out_dir, "manifest.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="CSV with columns index,TEXT,bucket")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--devices", default="0", help="comma-separated GPU indices, e.g. 6,7")
    ap.add_argument("--scales", default="0,0.001,0.002,0.003,0.005",
                    help="comma-separated constant attn_guidance_scale values")
    ap.add_argument("--seeds", default="0,42,123", help="comma-separated seeds")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--power_calibrate", type=int, default=0)
    ap.add_argument("--negative_prompt", default=DEFAULT_NEG)
    ap.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--model_name", default=DEFAULT_MODEL)
    ap.add_argument("--low_vram", action="store_true",
                    help="offload models to CPU between phases (models_to_cpu=True); use on busy GPUs")
    args = ap.parse_args()

    assert args.resolution <= 1024, "eval is Stage-1-only; keep resolution <= 1024 so Stage 2 is skipped"
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    scales = [float(s) for s in args.scales.split(",") if s != ""]
    devices = [f"cuda:{d.strip()}" for d in args.devices.split(",") if d.strip() != ""]
    assert devices, "no devices"

    prompts = pd.read_csv(args.prompts)
    assert "index" in prompts.columns and "TEXT" in prompts.columns, "prompts CSV needs index,TEXT[,bucket]"
    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)

    cfg = {
        "out_dir": args.out_dir, "resolution": args.resolution,
        "num_inference_steps": args.num_inference_steps, "guidance_scale": args.guidance_scale,
        "power_calibrate": args.power_calibrate, "negative_prompt": args.negative_prompt,
        "cache_dir": args.cache_dir, "model_name": args.model_name, "low_vram": args.low_vram,
        "git_commit": git_commit(),
    }

    tasks = build_tasks(prompts, seeds, scales)
    img_dir = os.path.join(args.out_dir, "images")
    todo = [t for t in tasks
            if not (os.path.exists(os.path.join(img_dir, t["id"] + ".png"))
                    and os.path.exists(os.path.join(img_dir, t["id"] + ".json")))]
    print(f"{len(prompts)} prompts x {len(seeds)} seeds x {len(scales)} scales = {len(tasks)} tasks; "
          f"{len(tasks) - len(todo)} already done, {len(todo)} to generate on {len(devices)} GPU(s).", flush=True)

    if todo:
        with open(os.path.join(args.out_dir, "config.json"), "w") as f:
            json.dump({**cfg, "seeds": seeds, "scales": scales,
                       "prompts_csv": os.path.abspath(args.prompts)}, f, indent=2)
        tmp.set_start_method("spawn", force=True)
        manager = tmp.Manager()
        task_queue = manager.Queue()
        for t in todo:
            task_queue.put(t)
        procs = []
        for device in devices:
            p = tmp.Process(target=worker_process, args=(cfg, img_dir, device, task_queue))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

    n = consolidate_manifest(args.out_dir)
    print(f"manifest.jsonl written with {n} records -> {os.path.join(args.out_dir, 'manifest.jsonl')}", flush=True)


if __name__ == "__main__":
    main()
