# RepLDM Evaluation Pipeline

The harness separates generation from scoring through lossless PNG files and JSON sidecars. This lets expensive generations be scored repeatedly without coupling SDXL dependencies to reward-model dependencies.

```text
generate.py       [diff_attn]    prompt x seed x action -> PNG + sidecar -> manifest.jsonl
score.py          [repldm_eval]  manifest + PNG          -> scores.jsonl
compare_actions.py[repldm_eval]  manifest + scores       -> action_comparisons.csv
```

## Generate

Stage-1 experiments are limited to 1024² so Stage 2 is skipped. `--scales` retains the legacy constant-scale sweep; `--actions` accepts no-AG, conference-expert, scalar, and low/mid/high-frequency actions from YAML.

```bash
/home/bycao/miniforge3/envs/diff_attn/bin/python eval-pipeline/generate.py \
  --devices 1,2,3,4 \
  --prompts eval-pipeline/prompts/eval_v1.csv \
  --out_dir outputs/exp_spectral_headroom/pilot_12prompt_3seed_v1 \
  --actions eval-pipeline/configs/frequency_action_pilot.yaml \
  --seeds 0,42,123
```

All actions for one `(prompt, seed)` block run on the same GPU. Blocks use deterministic device placement and deterministic shuffled action order. On resume, an existing sidecar's device takes precedence; an already cross-device block is rejected. A task is complete only when both PNG and JSON exist. Worker and per-task failures make the command fail after preserving completed records.

Keep CFG, `power_calibrate`, model, resolution, negative prompt, and step count fixed within a run. Use a new output directory whenever any of these or the action definitions change.

`configs/frequency_amplitude_followup.yaml` is explicitly post-hoc: it checks whether the 0.004 pilot simply used too much scalar or mid-band guidance. Treat it as search data and validate any selected amplitude on new prompts.

## Prepare Scorers

The scoring environment is a clone of `diff_attn` with independent evaluation packages:

```bash
/home/bycao/miniforge3/envs/repldm_eval/bin/python -m pip install \
  pyiqa==0.1.15.post2 hpsv2==1.2.0 openai-clip==1.0.1 fairscale==0.4.13

HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  /home/bycao/miniforge3/envs/repldm_eval/bin/python eval-pipeline/prestage_weights.py
```

ImageReward additionally uses the source checkout at `/mnt/miah204/bycao/ImageReward`. The pre-stage command caches its checkpoint, media config, and BERT tokenizer. Generation and formal scoring run offline after weights are staged.

## Score

```bash
CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/bycao/miniforge3/envs/repldm_eval/bin/python eval-pipeline/score.py \
  --run_dir outputs/exp_spectral_headroom/pilot_12prompt_3seed_v1 \
  --device cuda:0 --strict
```

`--strict` is required for reported experiments: it fails if any configured scorer cannot initialize or score an image. Scoring is additive; complete existing scorer outputs are not recomputed.

Configured outputs are ImageReward and native crops, pixel witnesses, CLIPScore, HPSv2, LAION aesthetic, and TOPIQ-NR. CLIP, HPSv2, and aesthetic are correlated 224px model families, not independent confirmations. Patch-IR and Laplacian variance can reward local texture or noise; use them only as diagnostics.

## Compare Actions

```bash
/home/bycao/miniforge3/envs/repldm_eval/bin/python eval-pipeline/compare_actions.py \
  --run_dir outputs/exp_spectral_headroom/pilot_12prompt_3seed_v1 \
  --baseline no_ag \
  --metrics topiq_nr,hpsv2,imagereward,patch_ir_mean,clip_cosine,aesthetic,clipped_fraction
```

The comparison rejects missing or cross-device pairing metadata. It reports paired deltas, crossed prompt/seed bootstrap 95% intervals, prompt-level sign-flip tests, and both within-metric and global Holm corrections. Missing prompt×seed cells are rejected rather than silently converted into an unbalanced comparison.

`aggregate.py` and `visualize.py` remain available for the legacy scalar sweep. Their plots are descriptive and must not be used for the invalidated cross-device pilot in `EXPERIMENT_RESULTS.md`.

## Layout

```text
configs/                 scorer and action YAML
prompts/                 prompt CSV files
scorers/                 independent metric plugins
generate.py              grouped multi-GPU generation
score.py                 additive scoring runner
compare_actions.py       paired inference and multiplicity correction
aggregate.py             legacy scalar-sweep diagnostics
visualize.py             legacy montage and witness plots
prestage_weights.py      one-time scorer weight setup
```
