# Repository Guidelines

## Project Structure & Module Organization

`AttentionGuidance/` contains the reusable latent-attention implementation. `InferencePipelines/RepLDM/` provides the SDXL and ControlNet pipelines, while `InferencePipelines/FreeScale/` contains the FreeScale integration. Runnable scripts and notebooks live under `InferCases/RepLDM/SDXL/`; they are examples, not an importable package. The decoupled experiment harness is in `eval-pipeline/`, with metric plugins in `scorers/`, YAML configuration in `configs/`, and prompt sets in `prompts/`. Keep paper figures in `fig/`. Checkpoints and generated artifacts belong in `pretrained_ckpts/` and `outputs/` and must remain untracked.

## Build, Test, and Development Commands

Use the supported Python 3.9 environment:

```bash
conda create -n repldm python=3.9
conda activate repldm
pip install -e .
```

The editable install exposes both Python packages. Run batch inference with:

```bash
python InferCases/RepLDM/SDXL/t2i_infer_imgs.py --devices 0 --resolution 3072 --data_path prompts.csv
```

For a minimal evaluation generation, use `eval-pipeline/prompts/smoke.csv` and an ignored output directory; detailed scoring and aggregation commands are documented in `eval-pipeline/README.md`. Before submitting Python changes, run:

```bash
python -m compileall AttentionGuidance InferencePipelines eval-pipeline
```

## Coding Style & Naming Conventions

Follow existing Python conventions: four-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and descriptive module names. Add type hints to new or changed public interfaces and concise docstrings for non-obvious tensor shapes, schedules, and pipeline parameters. There is no configured formatter or linter, so keep edits focused and avoid reformatting the large pipeline files wholesale.

## Testing Guidelines

This repository currently has no automated test suite, coverage threshold, or CI. Validate imports or syntax with `compileall`, then exercise the affected script or notebook with the smallest practical prompt, seed, scale, and resolution. GPU runs require cached model weights; generation code is intended to use local checkpoints rather than download them implicitly. Remove temporary smoke artifacts when finished.

## Commit & Pull Request Guidelines

Recent commits use short scoped summaries such as `inspect(iter 13): fix ControlNet Stage-2 trigger`. Prefer an imperative, specific subject (`eval: handle missing scorer weights`) over the older generic `update` style. Pull requests should describe the behavior changed, exact validation commands, required hardware/checkpoints, and any output-quality or VRAM impact. Link relevant issues or experiments and include representative images when generation behavior changes. Use `main` for current behavior; consult `base` when reproducing paper results.

## GPU Reservation During Idle Periods

GPUs 4-7 are scarce shared resources. When no formal experiment is running, reserve those cards with the marked utility below so another job does not take them unexpectedly:

```bash
python scripts/gpu_reservation.py start --devices 4 5 6 7
python scripts/gpu_reservation.py status --devices 4 5 6 7
```

Each worker allocates nearly all available memory and continuously performs an FP16 matrix multiply. This is only a resource lock, not an experiment, and its processes are labeled `repldm-gpu-reservation-v1`. Before starting generation, training, or scoring, stop only these workers and verify the cards with `nvidia-smi`:

```bash
python scripts/gpu_reservation.py stop --devices 4 5 6 7
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

After a run finishes, restart the reservation while debugging or waiting for the next approved job. Do not kill unrelated GPU processes; the utility keeps PID files and logs under `/tmp/repldm_gpu_reservation`.
