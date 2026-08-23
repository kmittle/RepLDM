# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

RepLDM (NeurIPS 2025 Spotlight) is a **training-free** method for high-quality, high-resolution image generation (up to 8K) that reprograms pretrained latent diffusion models — currently SDXL. This is a research repository: there is no test suite, lint config, or CI. Code is delivered as two installable packages plus runnable example scripts/notebooks.

Two core ideas drive everything:

1. **Attention Guidance** (`AttentionGuidance/`) — a reusable, model-agnostic module that nudges latents toward a more layout-consistent state after each denoising step using Training-Free Self-Attention (TFSA): `z̃ = γ·TFSA(z) + (1−γ)·z`, where `TFSA(z) = reshape⁻¹(softmax(f(z)f(z)ᵀ/λ)·f(z))`. Controls color richness and detail.
2. **Two-stage high-res pipeline** (`InferencePipelines/RepLDM/`) — Stage 1 generates at the model's training resolution (~1024²) with Attention Guidance; Stage 2 progressively upsamples (pixel interpolation → VAE re-encode → partial "diffusion-denoising" restart loop) to reach the target resolution.

## Setup & commands

```bash
conda create -n repldm python=3.9
conda activate repldm
pip install -e .          # installs the AttentionGuidance + InferencePipelines packages
```

There are no build/lint/test commands — running is done by executing the example scripts/notebooks under `InferCases/`.

**Single-image text-to-image** (edit the config constants at the top of the file, then run):
```bash
jupyter notebook InferCases/RepLDM/SDXL/t2i_generate_images.ipynb   # edit the config cell at the top, then run
```

**Batch text-to-image across GPUs** (reads prompts from a CSV, one worker process per GPU via a task queue):
```bash
python InferCases/RepLDM/SDXL/t2i_infer_imgs.py --devices 0,1 --resolution 3072 --data_path <prompts.csv>
```

**ControlNet inference**: see `InferCases/RepLDM/SDXL/controlnet_infer.ipynb`; condition images (canny/depth) are produced by `controlnet_preprocess.py`.

## Working conventions

- **Background processes run in a new tmux window** of the current session (e.g. `tmux new-window`), not detached or backgrounded inline. This keeps long-running generation/eval jobs inspectable and their output attached.
- **Clean up after smoke tests / sanity runs.** Once a smoke test or sanity run finishes, delete its scripts, caches, and outputs so the project tree stays clean — leave behind only what's intentionally part of the repo.

### Model loading — important
Every `from_pretrained` call uses `local_files_only=True` with `torch_dtype=torch.float16, variant="fp16"`. **Models are never auto-downloaded** — the HuggingFace weights (`stabilityai/stable-diffusion-xl-base-1.0`, `diffusers/controlnet-canny-sdxl-1.0`, `diffusers/controlnet-depth-sdxl-1.0`, `Intel/dpt-hybrid-midas`) must already exist in the cache dir, which defaults to a **relative** `../huggingface_models` (or `../../huggingface_models` from notebooks). Adjust `cache_dir` to your environment. (`eval-pipeline/generate.py` instead defaults to an absolute `pretrained_ckpts` cache dir.)

## Architecture

### Package layout
- `AttentionGuidance/` — installable package exposing the single `AttnGuidance` class. No diffusers/pipeline dependency; it operates purely on latent tensors.
- `InferencePipelines/` — installable package exposing three diffusers pipelines via its `__init__.py`:
  - `RepLDMSDXLPipeline` — text-to-image (`RepLDM/pipeline_repldm_sdxl.py`)
  - `RepLDMSDXLControlNetPipeline` — text-to-image + ControlNet (`RepLDM/pipeline_repldm_sdxl_controlnet.py`)
  - `FreeScaleSDXLPipeline` — the [FreeScale](https://github.com/ali-vilab/FreeScale) baseline augmented with Attention Guidance (`FreeScale/`)
- `InferCases/` — example scripts and notebooks. **Not a package**, not importable; entry points only.

### `AttnGuidance` (`AttentionGuidance/attention_guidance.py`)
Constructed once per generation with `(dtype, device, num_total_steps, h, w, attn_type, guidance_scale, guidance_density, guidance_scale_decay, power_calibrate, guidance_filter, attn_scaling)`. It precomputes which steps get guidance (`guidance_density`), a per-step scale schedule (`guidance_scale_decay`: linear/cosine/exp), and an optional FFT low-pass `guidance_filter` window that widens over steps. Called as `attn_guidance(t_index, latents, alpha_t=None, scale=None)`.

**Critical convention:** `t_index` is the *reverse* step index, not the timestep. For T=50, timesteps t=50…1 map to `t_index`=49…0. Pipelines call it as `attn_guidance(num_timesteps - 1 - i, latents, alpha)`. Only `attn_type="vanilla"` is implemented.

### Two-stage RepLDM flow (in each pipeline's `__call__`)
- **Stage 1** — full denoising loop at training resolution (aspect-ratio-preserved ~1024²). Attention Guidance is applied **after the scheduler step** each iteration (only when `attn_guidance_scale > 0`). Latent mean/std are captured as an "anchor."
- **Stage 2** — runs only when `height*width > 1024²`. For each upsampling stage: interpolate the image up, VAE-encode (tiled if large), restart the diffusion at a partial timestep determined by `init_rates` (add noise, then denoise from there), and renormalize latents to the Stage-1 anchor statistics. **Attention Guidance is disabled in Stage 2** (and in FreeScale's multi-scale loop) — it runs at base resolution only.

### Key `__call__` parameters (RepLDM pipelines)
- `height`, `width` — target resolution (e.g. 3072, 6144, 8192).
- `init_rates` (list) — restart fraction per Stage-2 upsampling stage; **its length = number of upsampling stages**. Convention from the examples: `[0.8]` for <4096, `[0.9, 0.8]` for ≥4096. Higher rate = less re-denoising (more faithful to prior stage).
- `num_resample_timesteps` — denoising steps used in Stage-2 stages (typically 50).
- `multi_decoder` / `multi_encoder` — tiled VAE decode/encode to fit large resolutions in VRAM (`multi_decoder` is auto-enabled for resolution > 2048 in the batch script). `models_to_cpu` / `lowvram` — offload models between steps for low-VRAM runs.
- Attention guidance knobs: `attn_type='vanilla'`, `attn_guidance_scale` (RepLDM default 0.001, FreeScale 0.005; ~0.004–0.005 typical for tuning), `attn_guidance_density` (a per-step on/off tuple, e.g. `tuple([1]*47+[0]*3)` or `[0]*31+[1]*15+[0]*4`), `attn_guidance_decay` (e.g. `('cosine',0,3)` or `None`), `power_calibrate` (0/1/2, or None), `attn_guidance_filter` (`None` or a filter spec).
- **Return value**: a list of PIL images (one per stage). `images[-1]` is the final high-res result.

ControlNet pipeline adds `condition_image`, `controlnet_conditioning_scale`, `guess_mode`, `control_guidance_start/end`; the condition image is interpolated to match each Stage-2 resolution and fed through every stage.

### FreeScale (`InferencePipelines/FreeScale/`)
A multi-scale, restart-based high-res baseline. `pipeline_freescale_sdxl.py` swaps each `BasicTransformerBlock.forward` between `ori_forward` (base res) and `scale_forward` (windowed multi-scale self-attention) from `scale_attention.py` via descriptor binding. Attention Guidance is layered into its base-resolution denoising loop. Key params: `resolutions_list`, `restart_steps`, `cosine_scale`, `dilate_tau`, `fast_mode`. Its attention-guidance knobs use FreeScale-specific names — `attn_guidance_scale_decay` and `attn_guidance_power_calibrate` (vs RepLDM's `attn_guidance_decay` / `power_calibrate`).

## Branches
- `main` — current code, includes modifications beyond the paper.
- `base` — reproduces the original paper results. Use this branch when comparing against the published method.
