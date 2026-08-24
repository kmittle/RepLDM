# RepLDM Evaluation Pipeline

The harness separates generation from scoring through lossless PNG files and JSON sidecars. This lets expensive generations be scored repeatedly without coupling SDXL dependencies to reward-model dependencies.

```text
generate.py       [diff_attn]    prompt x seed x action -> PNG + sidecar -> manifest.jsonl
score.py          [repldm_eval]  manifest + PNG          -> scores.jsonl
compare_actions.py[repldm_eval]  manifest + scores       -> action_comparisons.csv
analyze_adaptivity.py[repldm_eval] scores + held-out seed -> adaptivity CSVs
```

## Generate

Stage 1 at up to 1024² is the default. A resolution above 1024 requires the explicit `--stage2` opt-in, which enables the repository's high-resolution resampling path and records all phase settings. `--scales` retains the legacy constant-scale sweep; `--actions` accepts no-AG, conference-expert, scalar, low/mid/high-frequency, standalone `trajectory_correction`, and scheduler-reference actions from YAML. Scalar actions may set `residual_mode` to `raw`, `mean_centered`, `moment_tangent`, `moment_tangent_rescaled`, `trajectory_cone_tangent`, or `trajectory_cone_tangent_rescaled`; omitted mode means byte-compatible `raw`. Trajectory-cone modes project against the scheduler update already passed to the controller. Fixed-moment modes cannot use frequency gains or the additive `max_update_ratio` cap because either operation would invalidate their geometry. A trajectory correction interpolates an Euler step toward its analytical ancestral transition, supports `mix` in `[0,1]`, and records per-step norm diagnostics; only `mix=0` and the uncapped `mix=1,sqrt` endpoint have exact scheduler semantics, while intermediate values are ablation controls. The hook is Stage-1-only and cannot be combined with another intervention. Scheduler references replace only the sampler, record the base scheduler config hash, and are marked `selection_eligible: false` when they are controls rather than proposed methods.

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

The scheduler-correction development gate is intentionally small and is not a
final test. After checking the exact prompt/config hashes, run it with:

```bash
/home/bycao/miniforge3/envs/diff_attn/bin/python eval-pipeline/generate.py \
  --devices 1 --prompts eval-pipeline/prompts/trajectory_correction_heldout_v1.csv \
  --out_dir outputs/trajectory_correction/development_v2 \
  --actions eval-pipeline/configs/trajectory_correction_development.yaml \
  --split_role development --seeds 0,42
```

Score the resulting manifest with the ordinary `score.py` and compare paired
actions against `no_correction`. The `euler_ancestral_reference` action is the
native `EulerAncestralDiscreteScheduler` control; it is reported but excluded
from fixed-action selection. The registered `noise_mode=none` actions are the
deterministic-drift ablation for the `sqrt` ancestral actions. Any correction
must be interpreted against this native sampler and later against DPM-Solver++
and UniPC same-NFE controls before a paper claim. Do not select a winner or
start renderer/RL training from this 11-prompt development gate; a larger,
separately frozen validation split is required.

After scoring, the preregistered S7 gate can be audited with:

```bash
/home/bycao/miniforge3/envs/repldm_eval/bin/python eval-pipeline/select_trajectory_correction.py \
  --run_dir outputs/trajectory_correction/development_v2 \
  --actions eval-pipeline/configs/trajectory_correction_development.yaml
```

The selector requires a complete paired design and returns `no_correction` when
no candidate reaches the primary/guard thresholds.

If a candidate passes, freeze the validation action in a new file (the template
itself is immutable):

```bash
/home/bycao/miniforge3/envs/repldm_eval/bin/python eval-pipeline/freeze_trajectory_correction_validation.py \
  --selection outputs/trajectory_correction/development_v2/trajectory_correction_selection.json \
  --source-actions eval-pipeline/configs/trajectory_correction_development.yaml \
  --output eval-pipeline/configs/trajectory_correction_validation_v1.yaml
```

`configs/trajectory_correction_validation_template.yaml` and
`prompts/trajectory_correction_validation_v1.csv` are that larger split
(44 prompts, three confirmation seeds). Keep `selected_action: null` until the
development report is archived; then freeze the selected development action and
run the confirmation command with `--split_role validation_confirmation` and
seeds `11,29,101`. Do not add a new mix or noise mode after looking at those
validation scores.

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

## Latent Renderer Registration

The post-S5 latent-renderer direction is a separate hypothesis. Read
`LATENT_RENDERER_PROTOCOL.md` and `RL_RESEARCH_DESIGN.md` before running it.
`configs/latent_renderer_mechanism_audit.yaml` freezes the LR-0 mechanism
audit and is registration-only; `generate.py` intentionally rejects it. The
reusable CPU/GPU-safe primitives are in
`AttentionGuidance/latent_renderer.py`, with focused tests under
`tests/test_latent_renderer.py`. No RL or renderer checkpoint is authorized
until the fixed-action LR-1 gate and the search-then-distill comparison pass.

Run the synthetic correctness audit without model weights:

```bash
/home/bycao/miniforge3/envs/diff_attn/bin/python \
  eval-pipeline/audit_latent_renderer.py \
  --output outputs/latent_renderer/lr0_cpu_report.json
```

The command must report `"passed": true`; its output is engineering evidence
only and cannot substitute for the prompt-disjoint LR-1 quality experiment.

With the local SDXL cache, the inference-only pipeline wiring smoke is:

```bash
/home/bycao/miniforge3/envs/diff_attn/bin/python \
  eval-pipeline/latent_renderer_smoke.py --device cuda:1
```

It requires the zero renderer to reproduce the no-renderer PNG exactly and a
fixed non-zero probe to change it. The probe is deliberately not a learned
checkpoint and its image must not enter coefficient selection or scoring.

The structural-provider smoke exercises the real UNet feature path used by the
next LR-1 implementation:

```bash
/home/bycao/miniforge3/envs/diff_attn/bin/python \
  eval-pipeline/latent_renderer_structural_smoke.py --device cuda:1
```

It captures one ordinary `up_blocks.0` backbone/skip pair and the registered
self-attention Q/K layer, then checks the same no-op/probe hash conditions.
This remains plumbing evidence; it is not a scored development run.

The first fixed-action LR-1 search uses the frozen train split (validation is
used only after the action is frozen):

```bash
/home/bycao/miniforge3/envs/diff_attn/bin/python eval-pipeline/generate.py \
  --devices 1,2,3,4 \
  --prompts eval-pipeline/prompts/latent_renderer_train.csv \
  --out_dir outputs/latent_renderer/lr1_fixed_train_searchseeds_v2 \
  --actions eval-pipeline/configs/latent_renderer_fixed_lr1.yaml \
  --split_role train_search --seeds 7,19,73
```

`latent_renderer_fixed` emits constant six-dimensional coefficients through
the same moment/trust-region renderer and records provider diagnostics in each
sidecar. Do not select an action from the test split or treat this generation
command as evidence until the registered TOPIQ-NR and non-inferiority gates
are evaluated.

After strict scoring, apply the frozen train-only proxy rule mechanically:

```bash
/home/bycao/miniforge3/envs/repldm_eval/bin/python eval-pipeline/select_fixed_action.py \
  --run_dir outputs/latent_renderer/lr1_fixed_train_searchseeds_v2 \
  --prompts eval-pipeline/prompts/latent_renderer_train.csv
```

The selector requires a positive paired HPSv2 mean and 95% interval relative
to `no_ag`, then applies CLIP/pixel guards, finite renderer diagnostics, and
the recorded YAML order. It never reads TOPIQ-NR or any test row;
`fixed_action_selection.json` is the only action authorization for the
validation run. It also rejects any seed, coefficient, provider, action-set, or
input-hash drift from the registered YAML. A `no_ag` result closes LR-1 without
a post-hoc search.

Before selection, run the result-blind integrity audit on the complete scored
design:

```bash
/home/bycao/miniforge3/envs/repldm_eval/bin/python \
  eval-pipeline/audit_latent_renderer_run.py \
  --run_dir outputs/latent_renderer/lr1_fixed_train_searchseeds_v2 \
  --prompts eval-pipeline/prompts/latent_renderer_train.csv \
  --source_actions eval-pipeline/configs/latent_renderer_fixed_lr1.yaml \
  --split_role train_search
```

It rejects missing/extra cells, duplicate IDs, cross-device blocks, malformed
PNGs, identical action images, missing or non-finite strict scores, stale
diagnostics, and moment/trust violations. `run_audit.json` contains only
integrity summaries and hashes, never action-quality rankings.

If and only if a non-baseline action is selected, freeze the sole validation
configuration and its preregistered controls:

```bash
/home/bycao/miniforge3/envs/repldm_eval/bin/python \
  eval-pipeline/freeze_latent_renderer_validation.py \
  --selection outputs/latent_renderer/lr1_fixed_train_searchseeds_v2/fixed_action_selection.json \
  --output eval-pipeline/configs/latent_renderer_validation_lr1.yaml
```

The freezer rejects changed selection metrics, thresholds, resampling counts,
seeds, or candidate sets. It emits `no_ag`, the one train winner, the frozen
conference expert, and a preregistered Rademacher direction matched to the
winner's coefficient L2 norm. Generate validation exactly once with:

```bash
/home/bycao/miniforge3/envs/diff_attn/bin/python eval-pipeline/generate.py \
  --devices 1,2,3,4 \
  --prompts eval-pipeline/prompts/latent_renderer_validation.csv \
  --out_dir outputs/latent_renderer/lr1_fixed_validation_v1 \
  --actions eval-pipeline/configs/latent_renderer_validation_lr1.yaml \
  --split_role validation_confirmation --seeds 11,29,101
```

Do not create the emitted YAML when selection returns `no_ag`, and do not run
the final-test split unless every registered validation gate passes.

After strict scoring and the result-blind run audit, apply the frozen
statistical gate:

```bash
/home/bycao/miniforge3/envs/repldm_eval/bin/python \
  eval-pipeline/evaluate_latent_renderer_validation.py \
  --run_dir outputs/latent_renderer/lr1_fixed_validation_v1 \
  --frozen_actions eval-pipeline/configs/latent_renderer_validation_lr1.yaml \
  --audit outputs/latent_renderer/lr1_fixed_validation_v1/run_audit.json
```

The selected action must pass TOPIQ comparisons against `no_ag`, the conference
expert, and the matched-random control with one Holm-corrected family. HPSv2
and CLIP use CI-based non-inferiority against `no_ag`; pixel guards use paired
mean deltas. Even a statistical pass returns `qualitative_review_required`:
the frozen 24-prompt, seed-11 blinded review must still be completed before a
validation pass or final-test authorization can be recorded.

Create the review package only after the validation run passes the statistical
gate. Share `montage.png`, the individual pair PNGs, and `review_prompts.csv`;
keep `review_key.json` private until all reviewers submit their forms:

```bash
/home/bycao/miniforge3/envs/repldm_eval/bin/python \
  eval-pipeline/make_latent_renderer_blind_montage.py \
  --run_dir outputs/latent_renderer/lr1_fixed_validation_v1 \
  --prompts eval-pipeline/prompts/latent_renderer_validation.csv \
  --frozen_actions eval-pipeline/configs/latent_renderer_validation_lr1.yaml \
  --output_dir outputs/latent_renderer/lr1_fixed_validation_v1/blind_review
```

Complete one copy of `review_form_template.csv` per reviewer and finalize the
gate only when the preregistered preference and dimension thresholds pass:

```bash
/home/bycao/miniforge3/envs/repldm_eval/bin/python \
  eval-pipeline/finalize_latent_renderer_validation.py \
  --frozen_actions eval-pipeline/configs/latent_renderer_validation_lr1.yaml \
  --validation_gate outputs/latent_renderer/lr1_fixed_validation_v1/validation_gate.json \
  --review_key outputs/latent_renderer/lr1_fixed_validation_v1/blind_review/review_key.json \
  --review_forms outputs/latent_renderer/lr1_fixed_validation_v1/blind_review/reviewer_1.csv \
                 outputs/latent_renderer/lr1_fixed_validation_v1/blind_review/reviewer_2.csv \
  --output_actions eval-pipeline/configs/latent_renderer_final_test_lr1.yaml \
  --output_authorization outputs/latent_renderer/latent_renderer_final_test_authorization.json
```

This emits a four-action test config only after statistical and blinded gates
pass. The original ten-action search grid is never valid for final test. The
generator requires the emitted authorization and exact action-YAML hash for
`--split_role test_final`:

```bash
/home/bycao/miniforge3/envs/diff_attn/bin/python eval-pipeline/generate.py \
  --devices 1,2,3,4 \
  --prompts eval-pipeline/prompts/latent_renderer_test.csv \
  --out_dir outputs/latent_renderer/lr1_fixed_final_test_v1 \
  --actions eval-pipeline/configs/latent_renderer_final_test_lr1.yaml \
  --split_role test_final --seeds 0,42,123 \
  --authorization outputs/latent_renderer/latent_renderer_final_test_authorization.json
```

## Layout

```text
configs/                 scorer and action YAML
prompts/                 prompt CSV files
scorers/                 independent metric plugins
generate.py              grouped multi-GPU generation
score.py                 additive scoring runner
compare_actions.py       paired inference and multiplicity correction
analyze_adaptivity.py    leave-one-seed-out action-selection headroom
select_fixed_action.py   frozen train-only LR-1 action selection
freeze_latent_renderer_validation.py  one-shot validation config freezer
audit_latent_renderer_run.py  result-blind design and numerical audit
evaluate_latent_renderer_validation.py  frozen LR-1 statistical gate
make_latent_renderer_blind_montage.py  deterministic blinded review package
finalize_latent_renderer_validation.py  review-to-final authorization
aggregate.py             legacy scalar-sweep diagnostics
visualize.py             legacy montage and witness plots
prestage_weights.py      one-time scorer weight setup
```
