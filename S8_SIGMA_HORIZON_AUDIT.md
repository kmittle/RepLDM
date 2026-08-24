# S8 Sigma-Horizon CFG-EC Audit

Status: mathematical and novelty audit only. This note does not authorize a
pipeline change, GPU run, or validation action. The current CPU proxy remains
an equal-local-interval interface and must not be passed raw Euler sigmas.

## 1. Correct Ratio Direction

Index the denoising states by `i`, with
`sigma_(i-1) > sigma_i >= sigma_(i+1)`. The cached pair is evaluated at the
previous, higher-noise state. Define positive adjacent step lengths

```text
h_prev = sigma_(i-1) - sigma_i
h_next = sigma_i - sigma_(i+1)
r_sigma = h_next / h_prev
```

For a branch prediction `p(sigma)` that is locally linear, extrapolating the
current prediction to the *next scheduler point* gives

```text
p_hat_i = p_i + r_sigma * (p_i - p_(i-1))
```

The equivalent signed expression is
`(sigma_(i+1)-sigma_i)/(sigma_i-sigma_(i-1))`; both numerator and denominator
are negative, so the ratio is positive. The inverse ratio is wrong. A minimal
unit test uses sigmas `[10, 6, 3]` and `p=sigma`: `r_sigma=3/4` predicts `3`
exactly, while `4/3` does not.

This horizon is a design choice, not a consequence of CFG-OEC. The paper's
proxy is `2*p_i-p_(i-1)`, i.e. `r=1`, and uses the difference as an error
surrogate at the current step. A next-point sigma horizon therefore defines a
new `CFG-EC-sigma` variant. It must be compared with the registered `r=1`
proxy, and cannot be described as an exact CFG-OEC reproduction. At the final
point (`sigma_(i+1)=0`) this definition gives `r_sigma=0` and no proxy error;
that endpoint behavior must be reported or disabled by a frozen rule.

If the intended target is one *previous-sized* local interval rather than the
next scheduler point, the ratio is deliberately `r=1`; there is no principled
reason to insert a sigma ratio merely because the schedule is nonuniform.
This ambiguity must be settled before tuning `tau` or `blend`.

## 2. Parameterization and Scheduler Scope

The first candidate should be restricted to deterministic
`EulerDiscreteScheduler` with `prediction_type="epsilon"`, no churn, and one
ordinary scheduler transition per denoising step. In this setting

```text
x_(i+1) = x_i + epsilon_i * (sigma_(i+1) - sigma_i)
```

up to the scheduler's native dtype conversion. The correction can therefore
be measured in the same latent transition space. Raw prediction differences
are not interchangeable across epsilon, v-prediction, and x0/sample
parameterizations. Supporting v-prediction requires converting each branch at
its own sigma before extrapolation and converting the corrected result back; it
is a separate registered variant, not a free generalization.

Use the scheduler's integer `step_index` and `sigmas` tensor, not float
timestep equality. Before applying a correction, require:

* `sigmas[step_index-1] > sigmas[step_index] >= sigmas[step_index+1]`, all
  finite, with nonzero `h_prev` and finite positive `r_sigma`;
* the cache signature matches the scheduler class, config hash, sigma schedule,
  prediction type, action id, prompt/seed task, and current phase;
* the cached prediction is from exactly `step_index-1`, with no skipped or
  repeated scheduler step;
* `s_churn == 0` (otherwise the model sees `sigma_hat`, not nominal sigma, and
  fresh noise changes the trajectory); and
* correction is applied before `rescale_noise_cfg`, with that ordering frozen
  and recorded.

Reset the pair cache at every pipeline call, `set_timesteps`, scheduler swap,
Stage-1/Stage-2 boundary, denoising truncation, failed task, or changed action.
Reject CFG-EC when CFG is disabled. Do not enable it for
`EulerAncestralDiscreteScheduler` by inheritance: every step adds fresh noise,
so the previous branch pair belongs to a different stochastic trajectory.
An ancestral arm needs a separate derivation, common-noise protocol, and native
ancestral control.

## 3. High-Order Solver Overlap

The sigma-ratio finite difference is close to what multistep solvers already
do. For example, diffusers' second-order DPM-Solver uses log-SNR coordinates
`lambda`, with

```text
h  = lambda_(i+1) - lambda_i
h0 = lambda_i - lambda_(i-1)
r0 = h0 / h
D1 = (1 / r0) * (m_i - m_(i-1)) = (h / h0) * (m_i - m_(i-1))
```

UniPC, DEIS, LMS, PNDM, and related multistep methods likewise use
nonuniform-history coefficients. Thus a sigma-horizon branch extrapolation is
not a new high-order integrator. It changes the CFG branch before an Euler
step; the defensible claim, if any, is a scheduler-specific error
orthogonalization, not improved numerical integration in general.

If the candidate is evaluated with DPM-Solver/UniPC, the scheduler's own
history and the CFG-EC cache will be two coupled extrapolators. Attribution is
not identifiable without a native high-order solver + ordinary CFG arm and an
explicit interaction arm. Do not report a gain over Euler alone as evidence
for the correction.

## 4. Required Matched Controls

Freeze the same model, VAE, prompts, seeds, resolution, CFG scale, NFE, and
initial noise provenance. The minimum development matrix is:

| Arm | Purpose |
| --- | --- |
| Euler + ordinary CFG | Primary same-scheduler baseline. |
| Euler + exact index CFG-OEC (`r=1`) | Separates the published proxy from sigma horizon. |
| Euler + sigma-horizon CFG-EC (`r_sigma`) | Candidate under audit. |
| Euler + horizon-only extrapolation | Uses the same `r_sigma` but no OEC projection; tests whether the gain is only prediction extrapolation. |
| Euler + projection-only (`r=1`) | Tests orthogonalization without horizon adaptation. |
| Native DPM-Solver++(2M), UniPC/DEIS/LMS (ordinary CFG) | Matched high-order-history controls at declared NFE. |
| Euler + matched dummy residual | Equal update norm/cap, shuffled branch or random direction; detects magnitude/sharpness shortcuts. |
| Exact no-op (`blend=0`) | Must preserve PNG/hash and scheduler metadata byte-for-byte. |

The raw-paper dynamic mix (`s` used directly) and the conservative negative-
cosine skip are distinct arms. `tau` is selected on development only and then
frozen. Report per-step `r_sigma`, cosine, gate rate, negative skips, correction
norm, scheduler-update ratio, cap rate, and endpoint skips; a single composite
image score is insufficient.

## 5. Trust and Failure Guards

For deterministic Euler, let `eps_cfg` be ordinary CFG and `eps_ec` the
post-OEC prediction. Before the scheduler call, compute

```text
delta_x = (eps_ec - eps_cfg) * (sigma_(i+1) - sigma_i)
native  = eps_cfg * (sigma_(i+1) - sigma_i)
```

Apply a fixed per-row trust cap to `||delta_x||/max(||native||, eps)` (with a
separately registered absolute-latent fallback when `native` is near zero).
Scale only the correction residual, never the whole CFG prediction, and record
`cap_applied`, pre/post ratios, and non-finite failures. A cap in epsilon space
alone is not scheduler invariant. If `r_sigma` exceeds a frozen finite bound,
skip or fail with a diagnostic; do not silently clip it after looking at test
images.

The no-op path must bypass history allocation and all sigma arithmetic. The
candidate must add zero U-Net calls, zero scheduler steps, and zero RNG draws.
For any ancestral or churn experiment, report the native noise draw stream and
use a matched scheduler control; it is not a history-only comparison.

## 6. Novelty Verdict

The strongest defensible boundary is: **a bounded, scheduler-aware adaptation
of CFG-OEC for deterministic Euler epsilon prediction**. The sigma ratio alone
is a finite-difference convention already present in multistep solver theory;
it is not a new solver, latent representation, renderer, or RL algorithm. If
the result only beats ordinary Euler, loses to exact CFG-OEC or horizon-only
extrapolation, or disappears against native high-order controls, close this
route. A positive result must survive all controls, independent structural
witnesses, crossed prompt/seed uncertainty, and the scheduler guards above
before it can motivate a later renderer/RL study.
