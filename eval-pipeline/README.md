# RepLDM Evaluation Pipeline

The harness separates generation from scoring through lossless PNG files and JSON sidecars. This lets expensive generations be scored repeatedly without coupling SDXL dependencies to reward-model dependencies.

```text
generate.py       [diff_attn]    prompt x seed x action -> PNG + sidecar -> manifest.jsonl
score.py          [repldm_eval]  manifest + PNG          -> scores.jsonl
compare_actions.py[repldm_eval]  manifest + scores       -> action_comparisons.csv
analyze_adaptivity.py[repldm_eval] scores + held-out seed -> adaptivity CSVs
```

## Full GenEval

Formal renderer checkpoints use the frozen Sana metadata at
`/mnt/miah204/bycao/Sana/diffusion/post_training/dataset/geneval/test_metadata.jsonl`.
It contains 553 prompts and four samples per prompt, so one setting is exactly
2,212 images. The four rows for a prompt are not four independent prompts;
confidence intervals are clustered by prompt. The numeric seeds are a
registration choice, not a GenEval requirement. The config registers them as
one ordered `seed_cohort` (`id` plus SHA-256), and every method in a comparison
must use the same cohort. The formal CLI accepts only cohort IDs registered in
the reviewed code revision. To change seeds, add a new versioned registration
and rerun every method; changing only one method is rejected. A seed list that
differs from the selected config, a missing cohort binding, or a mixed cohort
is rejected by every CLI stage.

The stages below are resumable and keep every file under one run directory.
Run them from the repository root with the `repldm_eval` environment:

```bash
PY=/home/bycao/miniforge3/envs/repldm_eval/bin/python

# 1. Validate generator records and publish geneval/input_manifest.jsonl.
$PY eval-pipeline/geneval_full.py validate-input \
  --run-dir outputs/renderer/opd_c0 \
  --records outputs/renderer/opd_c0/manifest.jsonl \
  --checkpoint-id opd-c0 \
  --checkpoint-sha256 <64-hex-checkpoint-hash> \
  --method opd \
  --run-contract-sha256 <64-hex-contract-hash>

# 2. Create the official 00000/samples/0000.png layout.
$PY eval-pipeline/geneval_full.py prepare-layout \
  --run-dir outputs/renderer/opd_c0

# 3. Run the reviewed local Sana evaluator and seal the result.
$PY eval-pipeline/geneval_full.py run \
  --run-dir outputs/renderer/opd_c0 \
  --evaluator-python /opt/geneval/bin/python \
  --evaluator-script /mnt/miah204/bycao/Sana/tools/metrics/geneval/evaluation/evaluate_images.py \
  --model-path /mnt/miah204/bycao/Sana/output/pretrained_models/geneval

# Existing raw output can be normalized without rerunning the detector.
$PY eval-pipeline/geneval_full.py aggregate \
  --run-dir outputs/renderer/opd_c0 \
  --raw-results outputs/renderer/opd_c0/geneval/raw_results.jsonl \
  --evaluator-python /opt/geneval/bin/python \
  --evaluator-script /mnt/miah204/bycao/Sana/tools/metrics/geneval/evaluation/evaluate_images.py \
  --model-path /mnt/miah204/bycao/Sana/output/pretrained_models/geneval
```

The evaluator, model tree, input manifest, layout, raw JSONL, scores, config,
checkpoint, run contract, and shared seed cohort are all hashed into
`geneval/summary.json`.
`validate-summary --summary ...` must pass before a score is copied into an
experiment table. The upstream evaluator is intentionally kept outside this
repository; use a local, reviewed checkout with network access disabled.

## Generate

Stage 1 at up to 1024² is the default. A resolution above 1024 requires the explicit `--stage2` opt-in, which enables the repository's high-resolution resampling path and records all phase settings. `--scales` retains the legacy constant-scale sweep; `--actions` accepts no-AG, conference-expert, scalar, low/mid/high-frequency, standalone `trajectory_correction`, and scheduler-reference actions from YAML. Scalar actions may set `residual_mode` to `raw`, `mean_centered`, `moment_tangent`, `moment_tangent_rescaled`, `trajectory_cone_tangent`, or `trajectory_cone_tangent_rescaled`; omitted mode means byte-compatible `raw`. Trajectory-cone modes project against the scheduler update already passed to the controller. Fixed-moment modes cannot use frequency gains or the additive `max_update_ratio` cap because either operation would invalidate their geometry. A trajectory correction interpolates an Euler step toward its analytical ancestral transition, supports `mix` in `[0,1]`, and records per-step norm diagnostics; only `mix=0` and the uncapped `mix=1,sqrt` endpoint have exact scheduler semantics, while intermediate values are ablation controls. Strict paired RNG parity requires a single `torch.Generator`; generator lists are rejected for this intervention. The hook is Stage-1-only and cannot be combined with another intervention. Scheduler references replace only the sampler, record the base scheduler config hash, and are marked `selection_eligible: false` when they are controls rather than proposed methods.

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

Matched-NFE scheduler controls are descriptive and never selection-eligible.
Validate their complete effective schedules before scoring, then validate score
provenance after strict scoring:

```bash
/home/bycao/miniforge3/envs/repldm_eval/bin/python \
  eval-pipeline/validate_scheduler_baseline_run.py \
  --run-dir outputs/scheduler_baselines/development_v5 \
  --actions eval-pipeline/configs/scheduler_baselines_development_authorized_20260825.yaml \
  --prompts eval-pipeline/prompts/trajectory_correction_heldout_v1.csv \
  --kind manifest --output outputs/scheduler_baselines/development_v5/run_audit_manifest.json
```

Repeat with `--kind scores` and a distinct output after `score.py --strict`.
The validator rejects an incomplete grid, stale PNG or score hashes, action or
contract drift, non-finite metrics, extra U-Net calls, and missing or mutated
timestep/sigma schedules.

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
(44 prompts, three confirmation seeds). The validation loader runs three actions
per prompt/seed: `no_correction`, the native scheduler reference, and the one
frozen selected action. Keep `selected_action: null` until the development report
is archived; then freeze the selected development action and run the confirmation
command with `--split_role validation_confirmation` and seeds `11,29,101`. Do
not add a new mix or noise mode after looking at those validation scores.

For unattended, resumable execution of this registered sequence, use
`run_trajectory_correction_queue.sh`. It takes an exclusive lock, records input
hashes and stage state under `outputs/trajectory_correction/queue_v1/`, waits
for a GPU with at least 22 GiB free, and resumes incomplete generation or
scoring. At every GPU stage it also waits for any older S7 watcher or scoring
process targeting the same development run, so a handoff cannot start two
owners on one device. A failed development selector writes an auditable `null_route.json`
and stops. Only a passing selector can freeze the validation YAML and queue the
44-prompt x 3-seed x 3-action confirmation run (baseline, native reference,
selected action); after validation scoring it stops in `awaiting_review` and
never starts a renderer or RL job.

```bash
bash eval-pipeline/run_trajectory_correction_queue.sh
```

Use `--status` to inspect the state, or `--dry-run` with
`S7_DRY_RUN_SELECTION=ancestral_mix_050` to exercise the pass branch without
touching a GPU. The queue never terminates or signals an unrelated process.

`configs/frequency_amplitude_followup.yaml` is explicitly post-hoc: it checks whether the 0.004 pilot simply used too much scalar or mid-band guidance. Treat it as search data and validate any selected amplitude on new prompts.

`configs/moment_tangent_smoke.yaml` is registered in `../doc/research/MODEL_ITERATIONS.md`. Run it on `prompts/smoke.csv` with seed `0` before freezing a larger development grid; its two prompts may reject broken or catastrophic actions but cannot support an efficacy claim.

`configs/moment_tangent_development.yaml` is the action grid frozen from that range check. Its 12-prompt, 3-seed output is development evidence; do not reuse those prompts for confirmation if an action passes the registered gate.

`configs/trajectory_cone_smoke.yaml` is the registered S3 range check. Its hypothesis and action-removal rule are fixed in `../doc/research/MODEL_ITERATIONS.md`; do not add scales after viewing its scores.

`configs/trajectory_cone_development.yaml` freezes the complete non-catastrophic S3 intervals. As with S2, its 12-prompt set is development-only.

`configs/stage2_engineering_smoke.yaml` is correctness-only. `configs/stage2_transfer_pilot.yaml` freezes the five-action 2048² mechanistic ladder after S3; it reuses development prompts to audit target-domain mismatch and cannot support a confirmation claim. That registered pilot was negative and is closed in `../doc/research/MODEL_ITERATIONS.md`; do not rerun it with new scales or schedules.

## S5 Registered Prompts and Actions

S5 uses `prompts/s5_development.csv` and `prompts/s5_smoke.csv`, which were frozen before implementation and are disjoint from every earlier prompt CSV. The provenance source is PartiPrompts at commit `5a657978134374ce28973948331b319adef164bd`. Treat the TSV rows as zero-based after the header. With the fixed key `repldm-s5-v1`, rank each row in a challenge by

```text
SHA256("repldm-s5-v1:development:" + challenge + ":" + row_index + ":" + Prompt)
```

and retain the two smallest digests from each of `Complex`, `Fine-grained Detail`, `Properties & Positioning`, `Quantity`, `Writing & Symbols`, and `Perspective`. Exclude those 12 rows, then rank the remaining rows in those challenges by `SHA256("repldm-s5-v1:smoke:" + row_index + ":" + Prompt)` and retain the two smallest. The recorded source rows are `359,367,636,424,997,979,1045,1042,1621,1549,909,929` for development and `450,996` for smoke. Do not replace a prompt after seeing an image or score.

`configs/s5_smoke.yaml` is a correctness and catastrophic-range check, not a tuning set. It fixes the self-attention layer `up_blocks.0.attentions.0.transformer_blocks.0.attn1`, mutual top-k `16`, and semantic angle candidates `0.005, 0.01, 0.02, 0.04`. It also freezes raw noisy-latent TFSA, clean latent controls, a jointly permuted semantic graph, CFG-only 5.0, and historical PLADIS/GAG reproductions. The action IDs containing `official` are preserved only for artifact provenance and must not be cited as official implementations; see `../doc/audits/BASELINE_PROVENANCE.md`. Run it as:

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

## Pin the Generation Runtime

Formal generation uses the repository lock rather than the older broad package
declarations. Validate the active interpreter before launching a registered
control:

```bash
/home/bycao/miniforge3/envs/diff_attn/bin/python \
  eval-pipeline/generation_environment.py \
  --lock eval-pipeline/configs/generation_environment_diff_attn_20260825.yaml \
  --expected_sha256 8f7b38ccb770880537f5080b1d3b4eb426a294458ea644ec8a4ef6b61f771da4
```

The lock pins the pixel-affecting generation packages, platform, CUDA/cuDNN/GPU
stack, and PyTorch determinism/TF32 flags. Each newly registered YAML must bind
its lock SHA-256. Registered structural-control workers also verify the selected
GPU model and compute capability; generated sidecars record the actual device.

## Register Structural Controls

`configs/scheduler_native_structural_controls_development_registration_v1.yaml`
is a result-blind design record, not an action file. `generate.py` rejects its
`structural_control_registration_v1` schema. Do not make it executable by merely
renaming the schema.

An independently reviewed executable must use
`scheduler_native_structural_controls_actions_v1`, preserve the registration's
prompt, seed, action, sampling, execution-order, scoring, and analysis bodies,
and bind both the template hash and a reviewed implementation commit. Its source
manifest covers the generation pipeline and every local Attention Guidance,
FreeU, PLADIS, and GAG implementation file. Generation fails when those bytes
drift or the Git worktree is dirty. Every arm is scheduler-isolated and must
record exactly 50 one-call denoising steps. These development controls calibrate
baselines only; they cannot select a renderer or authorize RL.

After an independent reviewer issued and committed
`configs/scheduler_native_structural_controls_development_authorized_v1.yaml`,
the engineering profile was run first at commit `e0b323f`. It reused the exact
eight actions and sampling contract from that YAML at 1024px and 50 steps; it
did not permit quality scoring:

```bash
ACTIONS=eval-pipeline/configs/scheduler_native_structural_controls_development_authorized_v1.yaml
/home/bycao/miniforge3/envs/diff_attn/bin/python eval-pipeline/generate.py \
  --devices 7 \
  --prompts eval-pipeline/prompts/scheduler_native_fixed_headroom_smoke.csv \
  --out_dir outputs/structural_controls/engineering_smoke_v1 \
  --actions "$ACTIONS" --split_role engineering_smoke \
  --seeds 1798464083 --resolution 1024 --num_inference_steps 50
```

The frozen smoke reported `88/88`, complete runtime ledgers, and eight distinct
PNGs in every prompt block. A shared abort signal stopped sibling workers after
the first task failure. Its config and every sidecar bind
`engineering_only=true`, `formal_matrix_evidence=false`,
`quality_claim_allowed=false`, and `method_selection_allowed=false`; the scorer
rejects this scope before loading metric models. The audit also recomputes each
action's deterministic execution rank and requires the disk sidecar to equal
its manifest row.

The amended v2 auditor is formal-development only and rejects
`--engineering_smoke` before reading run artifacts or taking a lock. Reproduce
the historical gate only in a detached `e0b323f` worktree and a fresh output
directory, using that commit's auditor; never point it at the canonical run.
The already-passing frozen smoke permitted the formal development generation
below. Do not score or inspect outcome artifacts yet:

```bash
ACTIONS=eval-pipeline/configs/scheduler_native_structural_controls_development_authorized_v1.yaml
REGISTRATION=eval-pipeline/configs/scheduler_native_structural_controls_development_registration_v1.yaml
AMENDMENT=eval-pipeline/configs/scheduler_native_structural_controls_analysis_amendment_v1.yaml
RUN=outputs/structural_controls/development_v1
SEAL="$RUN/structural_control_pre_score_seal.json"

/home/bycao/miniforge3/envs/diff_attn/bin/python eval-pipeline/generate.py \
  --devices 7 \
  --prompts eval-pipeline/prompts/scheduler_native_fixed_headroom_development.csv \
  --out_dir "$RUN" \
  --actions "$ACTIONS" --split_role development \
  --seeds 1932556753,1065503757,201635682 \
  --resolution 1024 --num_inference_steps 50
```

Keep `ACTIONS`, `REGISTRATION`, `AMENDMENT`, `RUN`, and `SEAL` defined in the
same shell for every command below.

The analysis amendment has a two-commit authorization protocol. Commit A adds
the final auditor/evaluator hashes but keeps `status: blocked_pending_independent_review`.
An independent reviewer must review commit A without accessing outcomes. Commit
B may change only the amendment authorization fields: set
`status: authorized_pre_score` and bind `reviewed_commit` to commit A. The
amendment keeps method selection, validation, RL, and publication authorization
false. Do not create the seal from the blocked candidate.

After commit B, create the canonical one-shot pre-score seal. This command must
run before `scores.jsonl`, `run_audit.json`, any evaluation output, or hidden
evaluation staging directory exists:

```bash
/home/bycao/miniforge3/envs/repldm_eval/bin/python \
  eval-pipeline/audit_structural_control_run.py --create-pre-score-seal \
  --run_dir "$RUN" --actions "$ACTIONS" --registration "$REGISTRATION" \
  --analysis-amendment "$AMENDMENT" --pre-score-seal "$SEAL"
```

Only a successfully validated seal permits strict offline scoring. Formal
scoring must use the sealed launcher below; invoking `score.py` directly is
prohibited:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/bycao/miniforge3/envs/repldm_eval/bin/python \
  eval-pipeline/audit_structural_control_run.py --score-sealed \
  --run_dir "$RUN" --actions "$ACTIONS" --registration "$REGISTRATION" \
  --analysis-amendment "$AMENDMENT" --pre-score-seal "$SEAL"
```

While holding the generation/scoring lock, the launcher validates the canonical
run, amendment, seal, scorer runner, and scoring config; requires all score,
temporary, attempt, and receipt artifacts absent; then exclusively creates and
fsyncs `$RUN/structural_control_scoring_attempt.json`. The fixed offline child
must authenticate the launcher process and one-use capability before its first
metric computation. The launcher never removes the attempt marker: a launcher
error, child error, process termination, or post-score validation failure
permanently consumes the attempt and forbids retry. Only a zero-exit child with
exactly 792 manifest-bound, finite, provenance-valid score rows receives the
fsynced `$RUN/structural_control_scoring_success.json` receipt.

Then run the dedicated audit before any metric comparison. The formal audit is
schema `scheduler_native_structural_control_audit_v2`; it binds amendment,
seal, scoring-attempt, and scoring-success hashes and must report
`auditor_scope: formal_development_only` and `outcome_details_disclosed: false`.
Before parsing outcome rows, the CLI
exclusively creates and fsyncs
`$RUN/structural_control_audit_attempt.json`; the report binds that marker's
SHA256. A failed audit attempt consumes the one allowed attempt, so do not remove
the marker or retry:

```bash
/home/bycao/miniforge3/envs/repldm_eval/bin/python \
  eval-pipeline/audit_structural_control_run.py \
  --run_dir "$RUN" \
  --prompts eval-pipeline/prompts/scheduler_native_fixed_headroom_development.csv \
  --actions "$ACTIONS" --registration "$REGISTRATION" \
  --analysis-amendment "$AMENDMENT" --pre-score-seal "$SEAL" \
  --output "$RUN/run_audit.json"
```

Only a warning-free audit may enter the one-shot evaluator:

```bash
/home/bycao/miniforge3/envs/repldm_eval/bin/python \
  eval-pipeline/evaluate_structural_control_run.py \
  --run-dir "$RUN" --actions "$ACTIONS" --audit "$RUN/run_audit.json" \
  --analysis-amendment "$AMENDMENT" --pre-score-seal "$SEAL"

/home/bycao/miniforge3/envs/repldm_eval/bin/python \
  eval-pipeline/evaluate_structural_control_run.py --verify-bundle \
  --run-dir "$RUN" --actions "$ACTIONS" --audit "$RUN/run_audit.json" \
  --analysis-amendment "$AMENDMENT" --pre-score-seal "$SEAL"
```

The evaluator reports every registered action and guard, reruns the dedicated
audit, requires its valid one-shot marker, and never emits a selected action.
While holding the evaluation lock, it first creates and fsyncs the canonical
`$RUN/structural_control_evaluation_attempt.json` with exclusive, no-follow
creation. This happens before reading score, audit, or other outcome bytes. The
marker permanently consumes the CLI attempt: it remains after write/fsync
failure, analysis failure, or process termination, and its presence forbids a
rerun even when no bundle was published.

A successful attempt publishes exactly one canonical directory,
`$RUN/structural_control_evaluation_bundle/`, containing only
`structural_control_evaluation.json` and `structural_control_contrasts.csv`.
JSON binds the CSV filename, schema, scope, row count, and SHA256; every CSV row
carries the non-authorization envelope and all 14 input hashes, including the
scoring attempt/success, audit attempt, and evaluation attempt artifacts.
`--verify-bundle` requires all four but neither creates nor removes them. Under
the lock it rehashes the 14 current inputs, reruns the formal
audit and frozen statistics, deterministically rebuilds the complete JSON and
CSV, and requires both files to match byte for byte. It then rechecks all four
scoring/audit/evaluation evidence artifacts and the canonical bundle's directory
identity, exact regular-file
entries, and bytes. Thus input drift or a coordinated replacement during replay
cannot validate rewritten contrasts, action summaries, provenance, or CSV
bindings.

Stop on any nonzero exit, warning, missing artifact, schema/hash mismatch, input
drift, or unexpected legacy output at `$RUN/structural_control_evaluation.json`
or `$RUN/structural_control_contrasts.csv`. Do not delete, rewrite, reseal, rerun
sealed scoring or the one-shot audit/evaluator, or remove any attempt/receipt
artifact after a failure. The
low-level bundle publisher has atomic recovery tests, but this does not authorize
a second formal CLI attempt. Do not inspect outcomes, select a method, begin
validation, or authorize RL after a failure. An action pair identical across all
99 formal blocks is an intervention-activation failure; pair identities/counts
are not disclosed in the v2 result-blind audit.

## Prepare Scorers

The scoring environment is a clone of `diff_attn` with independent evaluation packages:

```bash
/home/bycao/miniforge3/envs/repldm_eval/bin/python -m pip install \
  pyiqa==0.1.15.post2 hpsv2==1.2.0 openai-clip==1.0.1 fairscale==0.4.13

HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  /home/bycao/miniforge3/envs/repldm_eval/bin/python eval-pipeline/prestage_weights.py
```

ImageReward additionally uses the source checkout at `/mnt/miah204/bycao/ImageReward`. The pre-stage command caches its checkpoint, media config, and the complete `bert-base-uncased` closure (`vocab.txt`, `tokenizer_config.json`, `tokenizer.json`, and `config.json`). It also stages TOPIQ-NR's `cfanet_nr_koniq_res50-9a73138b.pth` and the `timm/resnet50.a1_in1k` `model.safetensors` backbone. The command exits nonzero and reports the failed steps if any asset check fails; `PRESTAGE COMPLETE` is printed only after all steps pass. Generation and formal scoring run offline after weights are staged.

## Score

```bash
CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/bycao/miniforge3/envs/repldm_eval/bin/python eval-pipeline/score.py \
  --run_dir outputs/exp_spectral_headroom/pilot_12prompt_3seed_v1 \
  --device cuda:0 --strict
```

`--strict` is required for reported experiments: it fails on unavailable scorers
or non-finite outputs and writes a hardened provenance contract into every score
row. The contract binds scorer/runner source hashes, package and runtime versions,
model identifiers and resolved revisions, checkpoint hashes, and preprocessing.
Resume reuses existing values only when this complete contract still matches.
Preregistered runs can fail closed by adding
`scorer_provenance.required_schema: repldm_scorer_provenance_v1` to the scoring
YAML or by passing `--require-scorer-provenance`.

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

`aggregate.py` and `visualize.py` remain available for the legacy scalar sweep. Their plots are descriptive and must not be used for the invalidated cross-device pilot in `../doc/research/EXPERIMENT_RESULTS.md`.

## Latent Renderer Registration

The post-S5 latent-renderer direction is a separate hypothesis. Read
`../doc/protocols/LATENT_RENDERER_PROTOCOL.md` and `../doc/research/RL_RESEARCH_DESIGN.md` before running it.
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
