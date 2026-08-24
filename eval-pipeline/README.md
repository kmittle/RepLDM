# RepLDM Evaluation Pipeline

The harness separates generation from scoring through lossless PNG files and JSON sidecars. This lets expensive generations be scored repeatedly without coupling SDXL dependencies to reward-model dependencies.

```text
generate.py       [diff_attn]    prompt x seed x action -> PNG + sidecar -> manifest.jsonl
score.py          [repldm_eval]  manifest + PNG          -> scores.jsonl
compare_actions.py[repldm_eval]  manifest + scores       -> action_comparisons.csv
analyze_adaptivity.py[repldm_eval] scores + held-out seed -> adaptivity CSVs
```

## Generate

Stage 1 at up to 1024² is the default. A resolution above 1024 requires the explicit `--stage2` opt-in, which enables the repository's high-resolution resampling path and records all phase settings. `--scales` retains the legacy constant-scale sweep; `--actions` accepts no-AG, conference-expert, scalar, and low/mid/high-frequency actions from YAML. Scalar actions may set `residual_mode` to `raw`, `mean_centered`, `moment_tangent`, `moment_tangent_rescaled`, `trajectory_cone_tangent`, or `trajectory_cone_tangent_rescaled`; omitted mode means byte-compatible `raw`. Trajectory-cone modes project against the scheduler update already passed to the controller. Fixed-moment modes cannot use frequency gains or the additive `max_update_ratio` cap because either operation would invalidate their geometry.

```bash
/home/bycao/miniforge3/envs/diff_attn/bin/python eval-pipeline/generate.py \
  --devices 1,2,3,4 \
  --prompts eval-pipeline/prompts/eval_v1.csv \
  --out_dir outputs/exp_spectral_headroom/pilot_12prompt_3seed_v1 \
  --actions eval-pipeline/configs/frequency_action_pilot.yaml \
  --seeds 0,42,123
```

Run the registered Stage-2 engineering smoke before any 2048² batch:

```bash
/home/bycao/miniforge3/envs/diff_attn/bin/python eval-pipeline/generate.py \
  --devices 1 \
  --prompts eval-pipeline/prompts/stage2_smoke.csv \
  --out_dir outputs/exp_stage2_transfer/engineering_smoke_v1 \
  --actions eval-pipeline/configs/stage2_engineering_smoke.yaml \
  --seeds 0 --resolution 2048 --stage2
```

Stage-2 resampling noise is drawn from the per-task generator, so paired actions share both Stage-1 and Stage-2 randomness. With phase offload enabled, normal decoding explicitly restores the VAE and latent to the execution device. The engineering smoke's `no_ag` and `no_ag_repeat` final PNGs must be identical; `conference_expert` must differ.

All actions for one `(prompt, seed)` block run on the same GPU. Blocks use deterministic device placement and deterministic shuffled action order. On resume, an existing sidecar's device takes precedence; an already cross-device block is rejected. A task is complete only when both PNG and JSON exist. Worker and per-task failures make the command fail after preserving completed records.

Keep CFG, `power_calibrate`, model, resolution, negative prompt, and step count fixed within a run. Use a new output directory whenever any of these or the action definitions change.

`configs/frequency_amplitude_followup.yaml` is explicitly post-hoc: it checks whether the 0.004 pilot simply used too much scalar or mid-band guidance. Treat it as search data and validate any selected amplitude on new prompts.

`configs/moment_tangent_smoke.yaml` is registered in `MODEL_ITERATIONS.md`. Run it on `prompts/smoke.csv` with seed `0` before freezing a larger development grid; its two prompts may reject broken or catastrophic actions but cannot support an efficacy claim.

`configs/moment_tangent_development.yaml` is the action grid frozen from that range check. Its 12-prompt, 3-seed output is development evidence; do not reuse those prompts for confirmation if an action passes the registered gate.

`configs/trajectory_cone_smoke.yaml` is the registered S3 range check. Its hypothesis and action-removal rule are fixed in `MODEL_ITERATIONS.md`; do not add scales after viewing its scores.

`configs/trajectory_cone_development.yaml` freezes the complete non-catastrophic S3 intervals. As with S2, its 12-prompt set is development-only.

`configs/stage2_engineering_smoke.yaml` is correctness-only. `configs/stage2_transfer_pilot.yaml` freezes the five-action 2048² mechanistic ladder after S3; it reuses development prompts to audit target-domain mismatch and cannot support a confirmation claim. That registered pilot was negative and is closed in `MODEL_ITERATIONS.md`; do not rerun it with new scales or schedules.

## S5 Registered Prompts and Actions

S5 uses `prompts/s5_development.csv` and `prompts/s5_smoke.csv`, which were frozen before implementation and are disjoint from every earlier prompt CSV. The provenance source is PartiPrompts at commit `5a657978134374ce28973948331b319adef164bd`. Treat the TSV rows as zero-based after the header. With the fixed key `repldm-s5-v1`, rank each row in a challenge by

```text
SHA256("repldm-s5-v1:development:" + challenge + ":" + row_index + ":" + Prompt)
```

and retain the two smallest digests from each of `Complex`, `Fine-grained Detail`, `Properties & Positioning`, `Quantity`, `Writing & Symbols`, and `Perspective`. Exclude those 12 rows, then rank the remaining rows in those challenges by `SHA256("repldm-s5-v1:smoke:" + row_index + ":" + Prompt)` and retain the two smallest. The recorded source rows are `359,367,636,424,997,979,1045,1042,1621,1549,909,929` for development and `450,996` for smoke. Do not replace a prompt after seeing an image or score.

`configs/s5_smoke.yaml` is a correctness and catastrophic-range check, not a tuning set. It fixes the self-attention layer `up_blocks.0.attentions.0.transformer_blocks.0.attn1`, mutual top-k `16`, and semantic angle candidates `0.005, 0.01, 0.02, 0.04`. It also freezes raw noisy-latent TFSA, clean latent controls, a jointly permuted semantic graph, CFG-only 5.0, official PLADIS, and official GAG. Run it as:

```bash
/home/bycao/miniforge3/envs/diff_attn/bin/python eval-pipeline/generate.py \
  --devices 1 --prompts eval-pipeline/prompts/s5_smoke.csv \
  --out_dir outputs/exp_s5/engineering_smoke_v1 \
  --actions eval-pipeline/configs/s5_smoke.yaml --seeds 0
```

Only the registered catastrophic thresholds may remove an extreme contiguous angle. Freeze a new development YAML before generating `s5_development.csv`; never use smoke scores to choose a winner.

For a provenance-clean engineering replay, the same command was run in
`outputs/exp_s5/engineering_smoke_v2` at commit `cb8eddd`. It reproduced the
earlier smoke hashes and metadata exactly; smoke remains correctness evidence,
not efficacy evidence.

`configs/s5_development.yaml` is the frozen development grid produced after the smoke gate. In `engineering_smoke_v1`, no registered catastrophic condition occurred, so the complete contiguous semantic interval `0.005, 0.01, 0.02, 0.04` is retained. Generate it only after freezing this file:

```bash
/home/bycao/miniforge3/envs/diff_attn/bin/python eval-pipeline/generate.py \
  --devices 1,2,3,4 --prompts eval-pipeline/prompts/s5_development.csv \
  --out_dir outputs/exp_s5/development_12prompt_3seed_v1 \
  --actions eval-pipeline/configs/s5_development.yaml --seeds 0,42,123
```

The provenance-clean development run is `outputs/exp_s5/development_12prompt_3seed_v2`
(`cb8eddd`), with 504/504 records and strict scores. Reproduce the paired
comparison with:

```bash
/home/bycao/miniforge3/envs/repldm_eval/bin/python eval-pipeline/compare_actions.py \
  --run_dir outputs/exp_s5/development_12prompt_3seed_v2 --baseline no_ag \
  --metrics topiq_nr,hpsv2,imagereward,clip_cosine,clipped_fraction,mean_saturation
```

No semantic action reached the `+0.005` TOPIQ gate or showed a stable
structural gain in the fixed montage. S5 is therefore a registered null and
must not be followed by an angle/top-k/layer/reward sweep or RL training.

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

## Analyze Adaptivity Headroom

Use leave-one-seed-out selection to test whether per-prompt action choices are stable beyond the seeds used to choose them:

```bash
/home/bycao/miniforge3/envs/repldm_eval/bin/python eval-pipeline/analyze_adaptivity.py \
  --run_dir outputs/exp_spectral_headroom/pilot_12prompt_3seed_v1 \
  --selection_metric topiq_nr
```

For each fold, the script selects a global action and one action per prompt on the other seeds, then evaluates both on the held-out seed. `no_ag` is always a candidate, so the analysis cannot create a gain by forcing guidance. It writes `adaptivity_comparisons.csv` and `adaptivity_selections.csv`, including the candidate set, objective direction, inference seed, and resampling counts. It rejects incomplete or cross-device blocks and reports the per-prompt-minus-global headroom with the same paired inference used above. This is a cross-seed consistency test on known prompts, not evidence of generalization to unseen prompts.

`aggregate.py` and `visualize.py` remain available for the legacy scalar sweep. Their plots are descriptive and must not be used for the invalidated cross-device pilot in `EXPERIMENT_RESULTS.md`.

## Layout

```text
configs/                 scorer and action YAML
prompts/                 prompt CSV files
scorers/                 independent metric plugins
generate.py              grouped multi-GPU generation
score.py                 additive scoring runner
compare_actions.py       paired inference and multiplicity correction
analyze_adaptivity.py    leave-one-seed-out action-selection headroom
aggregate.py             legacy scalar-sweep diagnostics
visualize.py             legacy montage and witness plots
prestage_weights.py      one-time scorer weight setup
```
