# S8 CFG-EC Smoke Design

Status: interface audit and dry-run design only. No S8 implementation, model
generation, scoring, or merge is authorized by this document.

Worktree: `/tmp/repldm-cfgec`

Base: detached `rl-version` commit `029c59f55075b0e60cee3018893f20689f6533a5`

The main `rl-version` worktree and the S7 queue are out of scope. The only
related change found in the repository is commit `7930f92` (`cfg-batch-fix`),
which is not merged into this base. It is a prerequisite for any multi-prompt
CFG-EC smoke.

## Proposed Contract

CFG-EC is a proposed correction of the classifier-free guided noise prediction
using the current and immediately preceding conditional/unconditional pair:

```text
eps_u_t, eps_c_t = current UNet prediction split
eps_cfg_t = eps_u_t + w * (eps_c_t - eps_u_t)
eps_ec_t = cfg_ec_correction(
    eps_u_t, eps_c_t,
    eps_u_prev, eps_c_prev,
    timestep_t, timestep_prev,
    scheduler_state,
    cfg_ec_config,
)
```

The exact correction operator and its time normalization must be frozen before
the smoke. A raw subtraction of predictions from different timesteps is not an
acceptable implementation contract. The correction must either use a registered
sigma/timestep normalization or explicitly define why the scheduler's prediction
parameterization makes the subtraction valid.

The first denoising step has no history. It must use ordinary CFG and then seed
the history cache. History is per pipeline invocation, effective sample, action,
and scheduler phase; it must never cross a prompt, seed, action, scheduler, or
worker boundary.

The default S8 design is history-only. It must not add a U-Net call, scheduler
step, random draw, or VAE call. An extra-call variant is a different action and
requires an equal-cost extra-call control; it cannot be silently introduced as
an optimization of the history-only action.

## Interface Audit

| Location | Existing interface | CFG-EC requirement |
| --- | --- | --- |
| `InferencePipelines/RepLDM/pipeline_repldm_sdxl.py:791-846` | `RepLDMSDXLPipeline.__call__` exposes CFG, scheduler, and intervention arguments. | Add one optional typed/configured CFG-EC argument only when implementation begins. Reset state at entry and reject incompatible Stage-2 or other intervention combinations unless explicitly registered. |
| `.../pipeline_repldm_sdxl.py:1118-1215` | `do_classifier_free_guidance = guidance_scale > 1.0`; negative and positive embeddings are concatenated before the loop. | Require `guidance_scale > 1`; preserve the existing embedding order and record the effective batch size. A CFG-EC action with CFG disabled must be rejected or be an exact baseline no-op, not silently reinterpret the action. |
| `.../pipeline_repldm_sdxl.py:1298-1349` | Stage-1 loop expands latents, calls U-Net once, splits predictions, applies CFG, then calls the scheduler. | Capture the pair immediately after the split, apply the registered correction before the existing rescale/scheduler boundary, then update the cache only after the current pair is accepted. |
| `.../pipeline_repldm_sdxl.py:1627-1650` | Stage-2 loop has a second, separate U-Net/CFG path. | Prefer Stage-1-only registration for the smoke. If Stage 2 is later enabled, it needs a separate cache reset and the same invariants; do not accidentally carry Phase-1 history into Phase 2. |
| `eval-pipeline/generate.py:187-460` | YAML action validation normalizes known action types and rejects unknown types. | Add a dedicated `cfg_ec` type, finite coefficient/range checks, a fixed history policy, and an explicit `extra_unet_calls` field. Do not overload `scalar` or `trajectory_correction`. |
| `eval-pipeline/generate.py:719-727` | Runtime helpers translate normalized actions into pipeline configs. | Add a pure constructor for the CFG-EC config; it must not allocate a cache or consume RNG. |
| `eval-pipeline/generate.py:789-866` | Worker creates one generator per task and calls the pipeline once per action. | Construct/reset CFG-EC state inside that task call. Never reuse a previous task's prediction pair, even when the worker keeps one pipeline instance. |
| `eval-pipeline/generate.py:884-932` | Sidecars record action, scheduler, timing, memory, and existing diagnostics. | Record `cfg_ec`, `history_valid_steps`, `history_reset_reason`, `extra_unet_calls`, cache dtype/bytes, and correction norm ratios. A missing or stale history record is a failed task, not a null value. |
| `InferencePipelines/cfg_batch.py` from `7930f92` | Provides `expand_cfg_latents` and `split_cfg_noise_pred` for all-negative/all-positive order. | Treat this unmerged commit as a hard prerequisite for batched CFG-EC. Do not implement history on top of the old interleaved order. |

## Batch Order

For effective batch size `B = batch_size * num_images_per_prompt`, the intended
rows are:

```text
prompt_embeds:       [u_0, ..., u_(B-1), c_0, ..., c_(B-1)]
latent_model_input:  [x_0, ..., x_(B-1), x_0, ..., x_(B-1)]
UNet output:         [eps_u_0, ..., eps_u_(B-1), eps_c_0, ..., eps_c_(B-1)]
split:               noise_pred.chunk(2)
```

The current `rl-version` code uses `latents.repeat_interleave(2)` and
`noise_pred[::2]`/`noise_pred[1::2]`. That happens to look correct for `B=1`,
but it mispairs rows for `B>1` because the embeddings are concatenated by
branch, not interleaved. Commit `7930f92` changes this to `torch.cat((latents,
latents), dim=0)` and `chunk(2)`, with a regression test. The S8 smoke must use
that fix or remain explicitly `B=1` while the batch contract is tested in a
CPU-only unit test.

There is a second batch audit point: the Stage-1 loop currently creates a
two-row `add_time_ids` tensor at `pipeline_repldm_sdxl.py:1298`. A multi-prompt
smoke must verify that micro-conditioning rows are expanded to `2B` in the same
order; fixing latent rows alone is insufficient.

The history cache must have shape `(B, C, H, W)` for each of
`eps_u_prev` and `eps_c_prev`, never `(2B, C, H, W)` with implicit interleaving.
The cache update and correction must operate row-wise so prompt `i` cannot read
prompt `j` history.

## History Cache and Boundaries

The minimum auditable state is:

```text
valid: bool
previous_timestep: scalar
previous_unconditional: Tensor[B,C,H,W]
previous_conditional: Tensor[B,C,H,W]
previous_scheduler_signature: immutable hash/identifier
```

Implementation requirements:

1. Allocate no history until CFG-EC is enabled and the first pair is available.
2. Store detached tensors in the prediction dtype/device; do not retain an
   autograd graph. A clone is required if the next U-Net or scheduler can mutate
   the source storage.
3. Reset at every `__call__`, at every scheduler phase boundary, and after a
   failed task. Reset when the scheduler class/configuration changes.
4. On the first step or after a reset, return ordinary CFG exactly and mark the
   returned diagnostics as `history_valid=false`.
5. Update history after correction diagnostics are captured, using the current
   unmodified conditional and unconditional predictions. Never cache the already
   guided tensor in place of the pair.
6. If the correction is disabled (`beta=0` or an explicit no-op action), bypass
   cache allocation and correction arithmetic. This is required for exact PNG
   hash parity, not only numerical closeness.

The correction hook should have a pure shape-checked interface similar to:

```text
correct_cfg_prediction(
    current_unconditional,
    current_conditional,
    previous_unconditional,
    previous_conditional,
    timestep,
    previous_timestep,
    scheduler_signature,
    config,
) -> (corrected_prediction, diagnostics)
```

It must reject mismatched batch, dtype, device, timestep sequence, or scheduler
signature rather than broadcasting or silently falling back to a different
history. Non-finite correction norms are task failures.

## Cost and Parity Accounting

The existing baseline performs one batched U-Net evaluation per denoising step
(two CFG rows per effective sample), one scheduler step, and no intervention RNG
draw. A history-only CFG-EC action must report:

```text
nfe = num_inference_steps
extra_unet_calls = 0
extra_scheduler_steps = 0
extra_rng_draws = 0
```

For SDXL Stage 1 at 1024 resolution (`C=4`, `H=W=128`, fp16), one cached
conditional/unconditional pair is approximately 256 KiB for `B=1`:
`2 * 4 * 128 * 128 * 2` bytes. Scale this linearly with `B`, spatial area, and
element size; report the measured peak allocation rather than relying only on
the estimate. Storing current and previous pairs simultaneously roughly doubles
the transient cache footprint.

An extra-call design must report the number and location of extra calls, the
effective NFE, wall time, peak memory, and RNG behavior. It requires a matched
baseline that makes the same extra call with the same prompt, seed, scheduler,
and batch order. A cached historical pair must not be recomputed merely to make
the accounting look symmetric.

RNG parity rules:

- History capture uses `detach`/`clone` only and never calls `randn`, a scheduler
  noise helper, or a new `torch.Generator`.
- The ordinary generator stream is consumed at the same points as no-AG.
- A no-op action must take the baseline branch and produce the exact baseline PNG
  hash and sidecar-relevant scheduler metadata. It must not allocate history,
  even if the configured coefficient is numerically zero.
- Repeated no-op runs with the same seed must remain byte-identical. Any hash
  mismatch blocks the smoke.

## StructFlow-Inspired Source-Prior Candidate

This is a separate candidate, not an alternative implementation of CFG-EC.
The reference is *Spatially-Grounded Flow Matching: Structured Source
Distributions for Image Generation* (arXiv:2608.15452v1). That work changes the
flow-matching source distribution and trains, or progressively post-trains, the
model on the changed distribution. It is evidence for a source-prior hypothesis,
not evidence that an inference-only swap is valid for an SDXL checkpoint trained
on an iid Gaussian source.

The proposed transfer would act only when the initial Stage-1 latent is created
(the current hook is `prepare_latents`, at
`InferencePipelines/RepLDM/pipeline_repldm_sdxl.py:571-586`). For a frozen spatial
mask `M` and region anchors `z_k`, a concrete, auditable variant is:

```text
eta[i,j] ~ N(z_M[i,j], lambda^2 I)
epsilon = eta / sqrt(1 + lambda^2)
latents_0 = scheduler.init_noise_sigma * epsilon
```

The mask source (fixed library mask, deterministic grid/Voronoi proxy, or a
segmentation model), latent-resolution downsampling, `lambda`, and whether
anchors are per-channel must be frozen before a smoke. A per-sample guard may
check finite values, empirical mean/variance, and an analytic KL to the iid
Gaussian. It must not form a dense covariance matrix at SDXL latent dimension.
For the region-wise Gaussian above, the KL can be evaluated from the block
eigenvalues. If anchors are independent per channel, with region sizes `n_k`
and `rho = 1 / (1 + lambda^2)`, the normalized covariance has eigenvalues
`1-rho` (multiplicity `n_k-1`) and `1+(n_k-1)*rho` (multiplicity one) per
channel, so the zero-mean KL to `N(0,I)` is:

```text
KL = -0.5 * C * sum_k ((n_k - 1) * log(1-rho)
                       + log(1 + (n_k - 1)*rho))
```

This is an analytic prior-level diagnostic, not a substitute for the
per-sample empirical moment checks. At `lambda=0` the covariance is singular
and the KL to a full-rank iid Gaussian is infinite. The guard must therefore
reject that setting or define a documented regularized/pseudo-KL rather than
silently reporting a finite value. CISA or other cross-channel anchors require
a separately derived covariance; reusing this formula would be incorrect.
Guard failures should fail the task with diagnostics; unbounded resampling or
adaptive retries would make RNG accounting non-reproducible.

### Smoke Cost and Controls

| Metric | History-only CFG-EC | Structured source prior |
| --- | --- | --- |
| U-Net calls / scheduler steps | Baseline NFE; zero extra calls/steps. | Baseline NFE; zero extra calls/steps once the initial latent is built. |
| RNG behavior | No intervention draws; `detach`/`clone` only. | Usually changes the initial draw pattern (anchor plus residual), unless a deterministic transform of one baseline noise tensor is specified; log draw count and generator state. |
| Extra work | Per-step tensor arithmetic and a pair cache (about 256 KiB for one fp16 1024px effective sample, before transient copies). | One-time mask/anchor/residual construction and O(`B*C*H*W`) moment/KL checks; mask generation or host/device transfer can dominate the non-UNet overhead. |
| CPU smoke | Shape/order, first-step reset, row-wise history, no-op hash plumbing, and correction norm tests. | Distribution sampler, mask-to-latent alignment, analytic guard, lambda-to-iid limit, per-sample isolation, and fixed-seed RNG tests; no model is needed. |
| Required image controls | `no_correction`, exact `cfg_ec_zero`, and (if applicable) matched extra-call control. | iid initial latent, exact iid/no-op branch, fixed-mask structured prior, and a distinct mask/proxy control. Keep all denoising settings fixed. |

The source-prior candidate is computationally cheaper than an extra-call
CFG-EC variant and comparable to history-only CFG-EC in U-Net NFE. It is not
automatically cheaper in wall time: SLIC/proxy construction, dtype conversions,
and guard reductions occur before the first denoising step. The initial latent
is consumed by every later step, so a failed guard or a changed RNG stream can
invalidate paired-image comparisons even though the reported NFE is unchanged.
For `lambda -> infinity`, use an explicit baseline branch to obtain exact PNG
hash parity; a merely large finite `lambda` is not an identity control.

### Implementation Risk and Review Novelty

The source-prior path has a smaller code diff but a larger semantic risk than
history-only CFG-EC:

1. SDXL's scheduler and checkpoint expect the usual latent scale and iid source;
   injecting correlation changes the initial distribution before any learned
   denoising correction. The StructFlow paper's post-training result does not
   remove this mismatch for RepLDM.
2. Mask layout, latent downsampling, batch indexing, and Stage-1/Stage-2 phase
   boundaries all need explicit tests. A single shared mask or broadcast guard
   can make samples look plausible while leaking structure across effective
   batch rows.
3. Moment checks are noisy at small latent sizes, while exact KL is sensitive to
   region sizes and singular limits. Tolerances, covariance parameterization,
   and fail/adjust behavior must be frozen and emitted in the sidecar. A guard
   that rescales each sample after construction may erase the intended
   correlation and must be treated as a different operator.
4. A source-prior action can be paired fairly only with common initial-noise
   provenance. If it consumes a different number of generator values, the
   baseline comparison needs a declared common-random-number protocol rather
   than relying on equal integer seeds.

From a review-novelty perspective, a direct hierarchical correlated-noise
transfer to SDXL is close to the cited StructFlow contribution and should be
framed as a transfer/robustness baseline, not as a new method. A per-sample
moment/KL guard is useful engineering for distribution safety, but is not by
itself a novelty claim unless its statistic, correction rule, and benefit are
derived and ablated. CFG-EC changes the denoising-time guidance operator and is
more orthogonal to StructFlow, but it still needs a frozen formula and controls
against ordinary CFG, temporal prediction extrapolation, and any extra-call
implementation. Combining both in the first smoke would confound attribution
and should be deferred.

**Recommendation:** keep history-only CFG-EC as the primary S8 smoke because
its cost and invariants are bounded, and register the source-prior transfer as
an independent exploratory action (or later S8-SP smoke) with its own output
directory, mask provenance, guard diagnostics, and iid identity control. Neither
candidate should be selected for RL or ranked from a two-prompt smoke alone.

## Proposed Smoke Controls

The following is a design matrix, not a frozen action file:

| Action | Purpose | Expected cost |
| --- | --- | --- |
| `no_correction` | Ordinary CFG baseline at the registered CFG scale. | 1 U-Net call/step, no cache. |
| `cfg_ec_zero` | Identity plumbing control; must hash exactly to `no_correction`. | 1 U-Net call/step, cache bypassed. |
| `cfg_ec_history` | The single pre-registered history correction candidate. | 1 U-Net call/step, pair cache. |
| `cfg_ec_history_shuffled` | Negative control that permutes previous pairs across `B>1`; detects accidental row-insensitive behavior. | Same as candidate. |
| `cfg_ec_extra_call_control` | Required only if the candidate uses an extra prediction. | Matched extra-call NFE. |
| `cfg_scale_5` | Existing CFG-only scale control, if retained in the S8 protocol. | Same one-call NFE. |

Keep scheduler, model, resolution, VAE, negative prompt, NFE, initial noise,
prompt CSV, and seed fixed. Do not include S7 trajectory-correction actions in
the CFG-EC smoke. `cfg_ec_history_shuffled` is a diagnostic control, not a
selection winner.

Required smoke assertions:

- `B=2` CPU tensor test proves all-negative/all-positive pairing and row-wise
  history alignment.
- First-step `history_valid=false`; later steps have exactly one previous pair.
- `cfg_ec_zero` PNG hashes equal `no_correction` hashes for every prompt/seed.
- No-op and candidate sidecars expose NFE, extra calls, cache bytes, and history
  reset reason.
- Candidate output is finite and non-empty; correction norms and scheduler-step
  norms are finite.
- The candidate cannot be selected or taken to RL from this smoke alone.

## Safe Dry-Run Commands

There is currently no `generate.py --dry-run` flag. The following commands are
safe because they inspect code, validate syntax, or exercise CPU tensors only;
none loads SDXL weights or starts a GPU worker.

```bash
cd /tmp/repldm-cfgec
git status --short
git rev-parse HEAD
git show --stat --oneline 7930f92
/home/bycao/miniforge3/envs/repldm_eval/bin/python -m compileall -q \
  AttentionGuidance InferencePipelines eval-pipeline
/home/bycao/miniforge3/envs/repldm_eval/bin/python -m pytest -q \
  tests/test_eval_pipeline.py tests/test_semantic_transport.py
/home/bycao/miniforge3/envs/repldm_eval/bin/python eval-pipeline/generate.py --help
```

The following CPU-only pairing check is the minimum pre-implementation dry run:

```bash
cd /tmp/repldm-cfgec
/home/bycao/miniforge3/envs/repldm_eval/bin/python - <<'PY'
import torch

latents = torch.tensor([[10.0], [20.0]])
expanded = torch.cat((latents, latents), dim=0)
uncond, cond = expanded.chunk(2)
assert torch.equal(uncond, latents)
assert torch.equal(cond, latents)

prediction = torch.tensor([[100.0], [200.0], [300.0], [400.0]])
eps_u, eps_c = prediction.chunk(2)
assert torch.equal(eps_u, torch.tensor([[100.0], [200.0]]))
assert torch.equal(eps_c, torch.tensor([[300.0], [400.0]]))
print("CFG row-order dry run passed")
PY
```

Do not run a future command of the form
`generate.py --devices ... --actions cfg_ec_smoke.yaml` until the action schema,
operator, no-op hash test, and provenance fields are implemented and reviewed.
That would be a GPU experiment, not a dry run.

## Go/No-Go

S8-CFG-EC is not ready for generation on this audit. Before any GPU smoke,
merge or otherwise explicitly apply the isolated CFG batch-order fix, freeze the
correction formula and history policy, add CPU tests for `B=1` and `B=2`, and
define the matched extra-call control if the implementation cannot remain
history-only. A no-op hash mismatch, any cross-row history read, any unexpected
RNG draw, or an NFE discrepancy closes the smoke without ranking the candidate.
