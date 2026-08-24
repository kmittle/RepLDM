# FRLA / Structural Residual Audit

## Scope and provenance

This is a CPU-only preparation audit for the registered FRLA hypothesis.  It is
based on `origin/rl-version` at `029c59f` and includes the isolated CFG fixes
`7930f92`, `3a59466`, and `ad3bc14`.  It does not alter the S7 YAML files or
queue, start a GPU job, or claim an experiment result.

## Representation finding

The current `StructuralUNetBasisProvider` emits six rank-compatible slots:

1. reciprocal semantic transport;
2. spectral low, mid, and high bands;
3. normalized FreeU-style backbone-minus-skip difference;
4. Laplacian residual.

This is a generic residual coordinate system, not FRLA.  The proposal requires
a detached decoder feature, a deterministic reduction and resize to `16 x 16`,
five local cosine relation lags `[(1,0), (0,1), (1,1), (2,0), (0,2)]`, and one
latent-only gradient step on the relation discrepancy.  None of those
relation-gradient semantics are implemented by the six slots.  Therefore LR-1
may use this provider as a fixed-basis control, but must not be labelled FRLA,
distillation, or RL.  A learned renderer remains gated on S7 and on the fixed
FRLA comparison in `frla_relational.md`.

## Scheduler and trust cap

The pipeline order is: one ordinary U-Net call, scheduler step, optional
trajectory/semantic correction, renderer evaluation on `pred_original_sample`,
then `inject_rendered_clean_update(prev_sample, pred_original_sample, guided_x0)`.
The renderer currently projects a residual to the fixed-moment tangent, applies
`max_update_ratio` against the scheduler update, maps it with a geodesic, and
casts `guided_x0` back to the latent dtype.

In float32 the geodesic chord is no longer than its tangent arc, so the final
float residual remains below the declared cap.  A reproducible dtype boundary
exists after the cast: with `torch.manual_seed(21)`, tensors of shapes
`latent=(2,4,8,8)`, `raw=(2,4,8,8)`, `scheduler=(2,4,8,8)` in `bfloat16`, and
`max_update_ratio=0.1`, the pre-cast ratios are `[0.0999577, 0.0999543]`,
while the cast residual ratios are `[0.1003298, 0.1001001]`.  This is
quantization, not a geodesic or per-channel-radius violation.

The hardening branch adds an opt-in `enforce_post_cast_cap` action/config flag.
Diagnostics now retain pre-cast and post-cast norms/ratios, the observed
pre-correction overrun, the final overrun, and per-sample correction/fallback
flags.  In strict mode an overrun sample is scaled in tangent space and
retracted again; if quantization still exceeds the cap it falls back to the
exact original latent.  Thus `update_ratio` remains the actual injected
post-cast ratio, while the uncorrected overrun is still auditable.  The flag is
normalized into each action and written to `config.json` and task sidecars, so
provenance cannot silently change the safety contract.  The default remains
`false` for byte-compatible historical runs.

This contract covers the renderer residual (`guided_x0 - pred_original_sample`).
The later pipeline addition `prev_sample + guided_x0 - pred_original_sample`
performs another low-precision arithmetic step; an end-to-end hard bound would
need a separate injection-side check and is intentionally not claimed here.

## Identity and CFG contracts

Both the policy output layer and optional spatial output projection are
zero-initialized, making a fresh renderer exactly identity.  Tests must check
`guided_x0 == latent`, zero residual, and trainability after a nonzero update;
loading a checkpoint is the only way to enable an action.

CFG rows are required to be `[all negative, all positive]`.  The isolated CFG
fixes use `torch.cat((latents, latents))`, split predictions with `chunk(2)`,
and expand SDXL time IDs by branch blocks.  `StructuralUNetFeatureCapture`
selects the positive half (`value[batch_size:]`).  The audit regression uses
`B=2` synthetic rows `[u0,u1,c0,c1]`; an interleaved batch would select the
wrong prompt and invalidates any batched FRLA signal.

## Queue gate and verification

Before any FRLA run, freeze config/prompt-manifest hashes, seed and scheduler
state, reset renderer/provider state per task, and record the exact feature row,
residual norm, dtype, and RNG provenance.  Keep one completed job and one
validated next job in the queue, but do not bypass the S7 gate.  The local CPU
checks are:

```bash
python -m unittest tests.test_frla_structural_audit
python -m unittest tests.test_latent_renderer
python -m compileall AttentionGuidance InferencePipelines eval-pipeline
```

The broader suite still requires optional `diffusers` and `numpy` packages.
