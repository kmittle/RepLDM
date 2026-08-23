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
import glob
import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import traceback

import pandas as pd
import torch
import torch.multiprocessing as tmp
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from AttentionGuidance import (
    ConstantGuidanceController,
    MOMENT_TANGENT_MODES,
    RESIDUAL_MODES,
)
from InferencePipelines import RepLDMSDXLPipeline

DEFAULT_CACHE_DIR = "/mnt/miah204/bycao/RepLDM/pretrained_ckpts"
DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_NEG = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"


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


def group_tasks_by_pair(tasks):
    """Keep every action for a (prompt, seed) pair on one worker/device."""
    groups = {}
    for task in tasks:
        key = (task["prompt_index"], task["seed"])
        groups.setdefault(key, []).append(task)
    return list(groups.values())


def task_is_complete(task: dict, img_dir: str) -> bool:
    stem = os.path.join(img_dir, task["id"])
    return os.path.exists(stem + ".png") and os.path.exists(stem + ".json")


def recorded_group_device(task_group: list, img_dir: str):
    """Return a block's recorded device and reject already-invalid pairing."""
    devices = set()
    for task in task_group:
        json_path = os.path.join(img_dir, task["id"] + ".json")
        if not os.path.exists(json_path):
            continue
        try:
            with open(json_path) as handle:
                record = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read existing sidecar {json_path}: {exc}") from exc
        if record.get("device"):
            devices.add(str(record["device"]))

    if len(devices) > 1:
        key = (task_group[0]["prompt_index"], task_group[0]["seed"])
        raise ValueError(
            f"existing prompt/seed block {key} spans devices {sorted(devices)}; "
            "paired inference is invalid, so use a new output directory"
        )
    return next(iter(devices), None)


def assign_tasks_to_devices(tasks: list, devices: list, img_dir: str) -> dict:
    """Assign pending tasks without changing block placement on resume.

    The fallback round-robin index is computed over all blocks, not only the
    unfinished subset. An existing sidecar takes precedence over that fallback.
    """
    assignments = {device: [] for device in devices}
    for group_index, task_group in enumerate(group_tasks_by_pair(tasks)):
        recorded_device = recorded_group_device(task_group, img_dir)
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
        pending = [task for task in ordered_group if not task_is_complete(task, img_dir)]
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
    actions = config.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("action config must contain a non-empty 'actions' list")

    seen = set()
    normalized = []
    for raw in actions:
        action = dict(raw)
        action_id = str(action.get("id", ""))
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", action_id):
            raise ValueError(f"invalid action id {action_id!r}; use letters, digits, '.', '_' or '-'")
        if action_id in seen:
            raise ValueError(f"duplicate action id {action_id!r}")
        seen.add(action_id)
        action_type = action.get("type")
        if action_type not in {"none", "scalar", "legacy", "frequency_bands"}:
            raise ValueError(f"unsupported action type {action_type!r} for {action_id}")

        if action_type == "none":
            action["scale"] = 0.0
        elif action_type in {"scalar", "legacy"}:
            action["scale"] = float(action["scale"])
            if action["scale"] < 0:
                raise ValueError(f"{action_id}: scale must be non-negative")
        else:
            action["band_scales"] = [float(value) for value in action["band_scales"]]
            if len(action["band_scales"]) != 3 or min(action["band_scales"]) < 0:
                raise ValueError(f"{action_id}: band_scales must be three non-negative values")

        residual_mode = str(action.get("residual_mode", "raw"))
        if residual_mode not in RESIDUAL_MODES:
            raise ValueError(f"{action_id}: unsupported residual_mode {residual_mode!r}")
        if action_type != "scalar" and residual_mode != "raw":
            raise ValueError(f"{action_id}: only scalar actions support non-raw residual modes")
        action["residual_mode"] = residual_mode

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

    cutoffs = [float(value) for value in config.get("frequency_band_cutoffs", [0.08, 0.25])]
    if len(cutoffs) != 2 or not 0 < cutoffs[0] < cutoffs[1] < 0.5:
        raise ValueError("frequency_band_cutoffs must satisfy 0 < low < mid < 0.5")
    return normalized, cutoffs


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


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


def worker_process(cfg: dict, device: str, task_queue, error_queue):
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
        except queue.Empty:
            break
        if task is None:
            break
        png_path = os.path.join(img_dir, task["id"] + ".png")
        json_path = os.path.join(img_dir, task["id"] + ".json")
        if os.path.exists(png_path) and os.path.exists(json_path):
            continue
        try:
            generator = torch.Generator(device).manual_seed(task["seed"])
            action = task["action"]
            controller, attn_scale, attn_density, attn_decay = guidance_runtime(
                action, cfg["num_inference_steps"]
            )
            images = pipe(
                task["prompt"],
                negative_prompt=cfg["negative_prompt"],
                generator=generator,
                height=cfg["resolution"], width=cfg["resolution"],
                num_inference_steps=cfg["num_inference_steps"],
                guidance_scale=cfg["guidance_scale"],
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
            )
            images[-1].save(png_path)  # lossless PNG
            record = {
                **task,
                "image_path": os.path.relpath(png_path, cfg["out_dir"]),
                "height": cfg["resolution"], "width": cfg["resolution"],
                "num_inference_steps": cfg["num_inference_steps"],
                "guidance_scale": cfg["guidance_scale"],
                "power_calibrate": cfg["power_calibrate"],
                "attn_guidance_density": attn_density,
                "attn_guidance_decay": attn_decay,
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
                "git_commit": commit,
                "device": device,
            }
            with open(json_path, "w") as f:
                json.dump(record, f)
            del images
            n_done += 1
        except Exception:
            error = traceback.format_exc()
            error_queue.put((task["id"], error))
            print(f"[{device}] FAILED task {task['id']}:\n{error}", flush=True)
    print(f"[{device}] done, generated {n_done} images", flush=True)


def consolidate_manifest(out_dir: str, expected_ids=None):
    img_dir = os.path.join(out_dir, "images")
    rows = []
    for fn in sorted(os.listdir(img_dir)):
        if fn.endswith(".json"):
            stem = fn[:-5]
            if expected_ids is not None and stem not in expected_ids:
                continue
            with open(os.path.join(img_dir, fn)) as f:
                record = json.load(f)
            if record.get("id") != stem:
                raise ValueError(f"sidecar id does not match filename: {fn}")
            rows.append(record)
    with open(os.path.join(out_dir, "manifest.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="CSV with columns index,TEXT,bucket")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--devices", default="0", help="comma-separated GPU indices, e.g. 6,7")
    ap.add_argument("--scales", default=None,
                    help="comma-separated constant attn_guidance_scale values")
    ap.add_argument("--actions", default=None,
                    help="YAML action grid; mutually exclusive with --scales")
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
    if args.actions and args.scales:
        ap.error("--actions and --scales are mutually exclusive")
    if args.actions:
        actions, band_cutoffs = load_actions(args.actions, args.num_inference_steps)
        legacy_scale_ids = False
    else:
        scale_values = args.scales or "0,0.001,0.002,0.003,0.005"
        scales = [float(s) for s in scale_values.split(",") if s != ""]
        actions = scale_actions(scales)
        band_cutoffs = [0.08, 0.25]
        legacy_scale_ids = True
    devices = [f"cuda:{d.strip()}" for d in args.devices.split(",") if d.strip() != ""]
    assert devices, "no devices"
    if len(devices) != len(set(devices)):
        ap.error("--devices must not contain duplicates")

    prompts = pd.read_csv(args.prompts)
    assert "index" in prompts.columns and "TEXT" in prompts.columns, "prompts CSV needs index,TEXT[,bucket]"
    if prompts["index"].duplicated().any():
        ap.error("prompt indices must be unique")
    validate_model_cache(args.model_name, args.cache_dir)
    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)

    cfg = {
        "out_dir": args.out_dir, "resolution": args.resolution,
        "num_inference_steps": args.num_inference_steps, "guidance_scale": args.guidance_scale,
        "power_calibrate": args.power_calibrate, "negative_prompt": args.negative_prompt,
        "cache_dir": args.cache_dir, "model_name": args.model_name, "low_vram": args.low_vram,
        "frequency_band_cutoffs": band_cutoffs,
        "git_commit": git_commit(),
        **stage_settings,
    }

    tasks = build_tasks(prompts, seeds, actions, legacy_scale_ids)
    expected_ids = {task["id"] for task in tasks}
    img_dir = os.path.join(args.out_dir, "images")
    todo = [task for task in tasks if not task_is_complete(task, img_dir)]
    device_tasks = assign_tasks_to_devices(tasks, devices, img_dir)
    worker_count = len(device_tasks)
    print(f"{len(prompts)} prompts x {len(seeds)} seeds x {len(actions)} actions = {len(tasks)} tasks; "
          f"{len(tasks) - len(todo)} already done, {len(todo)} to generate on {worker_count} GPU(s).", flush=True)

    if todo:
        with open(os.path.join(args.out_dir, "config.json"), "w") as f:
            json.dump({**cfg, "seeds": seeds, "actions": actions,
                       "devices": devices,
                       "prompts_csv": os.path.abspath(args.prompts)}, f, indent=2)
        tmp.set_start_method("spawn", force=True)
        manager = tmp.Manager()
        error_queue = manager.Queue()
        active_devices = list(device_tasks)
        task_queues = [manager.Queue() for _ in active_devices]
        for device, queue in zip(active_devices, task_queues):
            for task in device_tasks[device]:
                queue.put(task)
        procs = []
        for device, task_queue in zip(active_devices, task_queues):
            p = tmp.Process(target=worker_process, args=(cfg, device, task_queue, error_queue))
            p.start()
            procs.append((device, p))
        for _, p in procs:
            p.join()

        worker_failures = [(device, p.exitcode) for device, p in procs if p.exitcode != 0]
        task_failures = []
        while True:
            try:
                task_failures.append(error_queue.get_nowait())
            except Exception:
                break
        if worker_failures or task_failures:
            n = consolidate_manifest(args.out_dir, expected_ids)
            examples = ", ".join(task_id for task_id, _ in task_failures[:5])
            raise RuntimeError(
                f"generation failed: workers={worker_failures}, "
                f"task_failures={len(task_failures)} ({examples}); "
                f"preserved {n} completed records for resume"
            )

    n = consolidate_manifest(args.out_dir, expected_ids)
    print(f"manifest.jsonl written with {n} records -> {os.path.join(args.out_dir, 'manifest.jsonl')}", flush=True)


if __name__ == "__main__":
    main()
