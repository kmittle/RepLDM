# CFG-EC CPU Proxy Formula

This note freezes the interface-only operator in
`AttentionGuidance/cfg_ec.py`. It is not a pipeline registration and does not
authorize a GPU smoke.

## Registered Inputs

For each effective sample `b`, provide current predictions `u_t`, `c_t` and
the immediately preceding pair `u_prev`, `c_prev`, all with shape `(B, ...)`.
`current_time` and `previous_time` are scalar, finite, monotonically decreasing
normalized timestep/sigma coordinates. `CFGECConfig` also supplies CFG scale
`w`, alignment threshold `tau` in `[0, 1]`, and correction strength `blend` in
`[0, 1]`. The pre-registered `max_extrapolation_ratio` guard defaults to
`4.0` (and must be at least `1.0` so the legacy normalized API remains valid).

The first denoising step passes both previous tensors as `None`; it returns
ordinary CFG exactly and does not inspect a history cache. A partial history is
an error. History is always row-wise and is never broadcast between samples.

## Operator

The CFG-OEC paper (arXiv:2511.14075, Eq. 10--15) uses an equal-step proxy
`hat{p} = 2 p_t - p_prev`. We write that proxy as a normalized-time finite
difference to make the assumption explicit:

```text
dq      = q_t - q_prev
d_p     = (p_t - p_prev) / dq
hat{p}  = p_t + d_p * dq
A       = c_t - hat{c}
B       = u_t - hat{u}
B_perp  = B - <A,B>/<A,A> A
u_bar   = hat{u} + B_perp
s       = <A,B> / (||A|| ||B||)
```

For rows with nonzero proxy errors, non-negative `s`, and `s < tau`, the
dynamic OEC candidate is `u_oec = (1-s) u_bar + s u_t`. Negative-cosine rows
are explicitly skipped and recorded as a guard: the paper notes that observed
values are predominantly non-negative but does not provide a universal bound
for the extrapolating case. The registered `blend` scales this complete
correction:

```text
u_out   = u_t + blend * (u_oec - u_t)
eps_out = u_out + w * (c_t - u_out)
```

Rows failing the threshold, or with a degenerate proxy error, are exact
ordinary CFG rows. Cosine values are clamped to `[-1, 1]` only after the
finite norm check; no random draw or extra model call is made.

## Scheduler-Aware Sigma Variant

`correct_cfg_prediction_sigma` is a separate, explicitly registered proxy for
three deterministic sigma points. It uses the next scheduler gap to define the
extrapolation horizon:

```text
h_prev  = sigma_prev - sigma_cur
h_next  = sigma_cur - sigma_next
r       = h_next / h_prev
hat{p}  = p_cur + r * (p_cur - p_prev)
```

The API rejects non-finite or negative sigmas, non-monotone points, and gaps at
or below a combined tolerance
`time_tolerance + relative_time_tolerance * max(abs(sigmas))`. The default
absolute and relative tolerances are both `1e-6`. It also rejects `r` above the
pre-registered `max_extrapolation_ratio=4.0`; this bound covers the measured
`1.9563` maximum among diffusers `EulerDiscreteScheduler` 5/10/20/30/50-step
Karras and non-Karras smoke schedules. That observation was obtained with
`/home/bycao/miniforge3/envs/repldm_eval/bin/python`, diffusers `0.32.1`,
`num_train_timesteps=1000`, default beta configuration, and
`use_karras_sigmas` toggled; it is not a universal scheduler guarantee and
must be rechecked for any deployed schedule. Equal spacing gives `r=1`, exactly
reducing to the normalized proxy above; a non-unit `r` is a distinct
sigma-horizon action, not an implicit claim that the published OEC paper
defined this scheduler rule.
`CFGECDiagnostics.extrapolation_ratio` records `r` for every history-valid
call (and `0` for identity/no-history branches). The same row-wise projection,
negative-alignment guard, blend, and finite checks are then applied.

## Time Caveat

The paper does not define a scheduler-specific extrapolation horizon for
nonuniform Euler sigma schedules. To avoid silently inventing one, the default
configuration accepts only a unit normalized interval
`abs(previous_time-current_time) == 1` (within tolerance). Setting
`allow_normalized_time_proxy=True` explicitly opts into the same local-step
formula at a non-unit normalized interval; this is an interface smoke ablation,
not a claim of physical scheduler equivalence. A future pipeline integration
must derive and register its scheduler-specific horizon before use.

**TODO / no-go for integration:** do not pass raw Euler or Euler-Ancestral
sigmas into this proxy as if the local equal-step extrapolation were physically
valid. A scheduler-specific derivation, prediction-parameterization audit, and
matched control are required before this function can be wired into a denoising
loop. The pure API does not validate scheduler class/configuration hashes,
prediction type, churn, integer step index/cache provenance, latent trust caps,
or the repository's CFG batch-order fix; all are integration blockers and must
be checked at the pipeline boundary.

## Identity and Diagnostics

`blend=0` bypasses all history arithmetic and returns ordinary CFG exactly.
Missing history does the same with `history_valid=false`. Diagnostics contain
finite, per-row alignment, negative-guard, applied-gate, correction-ratio, and
effective-blend values, plus the normalized time delta, extrapolation ratio,
and a reason string. Shape, dtype, device, ordering, sigma-gap, and finite-value
mismatches raise instead of broadcasting or silently falling back.
