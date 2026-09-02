# OPD, DPO, and RL Latent Renderer Protocol

Status: implementation contract. It does not authorize training. Training remains blocked
until a selected data view has `training_ready=true`, the implementation passes an
independent `$check 1`, and the reviewed commit is pushed.

## Research Claim

The journal method is a small policy that renders a better clean latent while SDXL stays
frozen. The defensible contribution is not a new optimizer. It is the combination of:

- a low-dimensional, model-structure-derived action space;
- an exact SDXL Euler clean-endpoint mapping;
- fixed moment geometry and scheduler-update caps; and
- one shared interface for OPD, renderer-DPO, and RL.

This claim survives only if the structured action space beats equal-norm random and
parameter-matched free-residual controls, and if learned state feedback beats an open-loop
prompt/timestep policy.

## Use Precise Names

- **External-teacher OPD:** the student rolls out first. A frozen external teacher labels
  states that the current student actually visits.
- **OPSD-style teacher:** a reward gradient constructs bounded positive and negative clean
  targets. This is a teacher-construction baseline, not a new OPD algorithm.
- **Renderer-DPO:** preference optimization over a stochastic renderer action with a valid
  `log pi(a | s)` and a frozen reference policy.
- **Reference-anchored RL:** short-horizon policy optimization with grouped branches, exact
  action log-probability, and a reference/KL penalty.

Reward-gradient targets overlap directly with DiffusionOPSD; shared-parent preferences
overlap with LPO; shared-prefix branches overlap with BranchGRPO; and a frozen diffusion
backbone plus a learned latent controller overlaps with Diffusion Controller. None of these
ingredients is a novelty claim.

## Shared Renderer Contract

All training methods use the same coefficient-only `StructuralLatentRenderer` with the same
active bases, state encoder, parameter count, coefficient limits, prompt set, seeds, 50-step
zero-churn Euler schedule, CFG, VAE, and SDXL checkpoint. The first round disables the D4
spatial head. A probability objective that covers only coefficients while another trainable
head changes the latent would be invalid.

The training API must split the current forward pass into:

```text
state = prepare_state(observation, condition, bases)
mu = action_parameters(state)
action = action_distribution(mu).sample()  # or mu for deterministic deployment
guided_x0, diagnostics = apply_coefficients(observation, bases, action)
```

Deployment keeps the existing public pipeline under `no_grad()` and loads the policy mean.
Training uses a separate package; it must not copy the production pipeline or remove its
`no_grad()` boundary.

For state `x_i`, guided noise `epsilon_g`, and Euler sigmas:

```text
x0_hat = x_i - sigma_i * epsilon_g
epsilon(Y) = (x_i - Y) / sigma_i
E_i(Y) = x_i + (sigma_next - sigma_i) * epsilon(Y)
kappa_i = 1 - sigma_next / sigma_i
E_i(Y) - E_i(x0_hat) = kappa_i * (Y - x0_hat)
```

Zero action must reuse the native `epsilon_g` bytes and be exactly no-op. A nonzero endpoint
is converted back to the scheduler's native prediction before exactly one scheduler step.

### EulerNativeFrameV1

The main structural hypothesis is `EulerNativeFrameV1`. It converts six model-derived raw
bases into a low-dimensional action frame measured in the scheduler's actual update
coordinates. OPD, DPO, and RL must use this same frame; changing it between methods would
confound the optimizer comparison.

The raw slots have one canonical order:

```text
semantic, spectral_low, spectral_mid, spectral_high, freeu, laplacian
```

Use `lazy_latent_structure_basis_v1`, reciprocal-semantic top-16 relations from
`up_blocks.0.attentions.0.transformer_blocks.0.attn1`, FreeU features from `up_blocks.0`,
and spectral cutoffs `(0.08, 0.25)`. These settings and every provider source hash belong in
the run contract. The first renderer is frozen as:

```text
num_bases=6, latent_channels=4
hidden_dim=256, depth=2
prompt_dim=32, state_dim=16, timestep_dim=16
spatial_hidden_dim=0
coefficient_bound=1.0, max_update_ratio=0.05
preserve_moments=true, normalize_bases=false
trainable_parameter_count=91,654
```

The count is the complete two-hidden-layer coefficient MLP. Prompt and state compression
are deterministic; an extra trainable encoder would violate the registered model. Independent
basis normalization after frame construction is forbidden because it would destroy the
scheduler metric.

For nominal clean latent `x`, detached raw basis `R_k`, native scheduler update
`N = native_prev_sample - current_sample`, and `kappa = 1 - sigma_next / sigma_current`,
project each channel onto the fixed-mean, fixed-variance tangent:

```text
xc = x - mean_hw(x)
P_x(v) = v - mean_hw(v)
         - <v - mean_hw(v), xc> / max(||xc||^2, eps) * xc
q_k = P_x(R_k)
d_k = kappa * q_k
s = ||N||_F
D_k = vec(d_k) / s
G = D D^T
```

Require `kappa >= 1e-6` and `s >= 1e-6`. Before any reward output is viewed, freeze one
global mask from the 576 strict-no-op states defined by all 64 training prompts, all 32
validation prompts, their two registered seeds, and decision indices `{8, 24, 40}`. In
canonical slot order, float64 two-pass modified Gram-Schmidt retains slot `k` only if
`||D_k||^2 >= 1e-8` and its residual-energy ratio after earlier retained slots is at least
`1e-6` in every calibration state. Freeze and hash that mask; require global rank at least
two. There is no per-state rank shrinkage after training starts.

At every later state, recompute the same diagnostics for all globally retained slots. If any
retained slot fails, reject the whole decision, take the registered no-op fallback, give the
record zero loss weight, and charge all reserved queries. Globally inactive slots are exact
zeros. Sampling and density evaluation therefore live on one fixed `R^rank` transformed-
Gaussian space, with a fixed Dirac mass on inactive coordinates. OPD, DPO, and RL cannot use
different masks or state-dependent probability dimensions.

The probability convention is part of the contract: active coordinates use Lebesgue measure
and inactive coordinates use a unit point mass at zero. Every policy density is therefore the
`rank`-dimensional density on active coordinates only. A sample with a nonzero inactive
coordinate, a changed mask hash, or a NaN/Inf density is invalid, receives zero loss, and
still consumes its reserved query.

For the active principal Gram matrix, use the symmetric inverse square root, never an
eigenvector action basis:

```text
G_A = 0.5 * (G_A + G_A^T) = V diag(lambda) V^T
lambda_floor = max(1e-12, 1e-6 * max(lambda))
S = V diag(max(lambda, lambda_floor)^(-1/2)) V^T
F_A = S D_A
```

Require positive diagonal alignment between `F_A` and `D_A` and
`||F_A F_A^T - I||_2 <= 1e-3`; otherwise reject the state. Symmetric whitening keeps the
canonical slots invariant to eigenvector signs, ordering, and repeated-eigenvalue rotations.

With `K=6`, `rho=max_update_ratio`, and `a_max=coefficient_bound`, set:

```text
beta = 0.999 * rho / (a_max * sqrt(K))
mapped_basis_k = beta * s * F_k
clean_basis_k = mapped_basis_k / kappa
a_k = active_k * a_max * tanh(u_k)
v = sum_k a_k * clean_basis_k
```

Then `||kappa*v||_F / ||N||_F <= 0.999*rho` for every coefficient-box corner. Keep
`sqrt(K)`, not `sqrt(rank)`, so physical action scale cannot jump when rank changes. Apply
the operations in this exact order: per-basis tangent projection, Euler mapping, whitening,
inverse Euler mapping, coefficient mixture, a second tangent projection for numerical
idempotence, the existing scheduler cap, a clean-latent angle cap, fixed-moment geodesic,
native-output conversion, and one scheduler step. The angle cap is fixed at
`theta_max=0.05` radians. For tangent update `v` and centered clean latent `xc`, multiply `v`
by `min(1, tan(theta_max) * ||xc||_F / max(||v||_F, 1e-12))` before the geodesic. This guard
remains effective when `kappa` is small. Normally both cap multipliers must be one. A zero
action takes an
explicit fast path that returns the original clean latent and native model output byte for
byte. Every trajectory stores the active mask, eigenvalues, condition number, Gram/frame
hashes, both cap multipliers, realized geodesic angle, and mapped-update ratio.

The first causal control is a matched Haar-random tangent frame. Generate `rank` Gaussian
tangent vectors from the immutable key `(run_contract_hash, prompt_id, generation_seed,
step_index, "haar_tangent_v1")`, apply the same projection and sign-fixed thin QR, place the
columns into the fixed active slots, and use the same scale. It reuses only the native
frame's global mask, never its directions.

Haar noise alone is not a sufficient control because it has a different spectrum. The
primary structure control is `PhaseMatchedFrameV1`: take the real 2-D FFT of each structured
mapped basis and channel, keep every magnitude exactly, randomize only conjugate-symmetric
phase with a frozen counter-based key, invert the FFT, then apply the same tangent projection,
whitening, box-corner scale, scheduler cap, and angle cap. Preserving Fourier magnitude also
preserves per-channel autocorrelation. Additional controls are: the same raw bases without
whitening, scaled by exact enumeration of all at most 64 coefficient-box corners; a
decision-index-matched derangement that moves structured bases across prompt states; and a
parameter-matched learned free-residual frame built from six trainable `4 x 16 x 16`
templates and the same renderer API. Its total trainable parameter count must be within 1%
of 91,654. Structured-frame superiority is unsupported unless it beats the phase-matched and
free-residual controls under identical prompts, actions, queries, optimizer steps, and
physical compute. Haar and unwhitened results remain mechanism ablations, not the only
causal evidence.

## Data Gate

The protected index contains 49,393 prompt rows, 46,619 normalized unique prompts, 39,712
image-bearing rows, and 37,160 unique local benchmark images. It includes all locally
available 4KLSDB validation and test metadata and their x4 HR images. Exact prompt exclusion
uses both `caption` and every nonempty `cogvlm_caption`.

The first selected view contains 64 training and 32 validation records from two sources and
eight fixed strata:

```text
nature, urban, people, food, artwork, cgi, animals, architecture
```

For every `source x stratum` cell, select exactly four training and two validation records;
thus each split is half 4KLSDB and half PixVerve. Rank candidates by SHA-256 of one frozen
selection seed and stable record ID. The four training records in every cell map one-to-one
to the four cross-fitting folds. There is no reseeding, prompt rewriting, tokenizer
relaxation, or manual replacement; one insufficient cell rejects the whole view.

PixVerve uses `short_caption` as `model_prompt` and preserves `long_caption` as
`raw_prompt`; both fields receive exact and semantic leakage checks. 4KLSDB uses `caption`
for both fields. Both SDXL tokenizers at the pinned model revision must fit `model_prompt`
with special tokens in at most 77 tokens and without truncation. A preliminary read-only
audit found about 1,911 eligible 4KLSDB captions and 95,306 eligible PixVerve short captions;
these counts show only pool feasibility, not that every stratum has enough clean rows.

Before setting `training_ready=true`, every selected row must bind raw-file SHA-256,
canonical decoded-pixel SHA-256, pHash, image dimensions, tokenizer provenance, and exact
and semantic prompt-leakage results. The selected images must also be checked against all
available benchmark images. Missing models, incomplete protected indexes, uncertain
licenses, symlinks, decode failures, or unresolved near duplicates fail closed.

The selected-view config must additionally bind the parent catalog/hash, source field maps,
license evidence, classifier model/revision/files, class templates, confidence margin and
tie rule, selection seed, both tokenizer manifests, decoder/EXIF/ICC behavior, semantic
model and calibrated threshold, pHash definition, image-embedding model and calibrated
threshold, and every protected-index hash. Each accepted row stores its stratum, fold,
selection rank, token counts, nearest protected text/image neighbor, similarity/distance,
threshold, and model hash. These classifier and semantic/image threshold artifacts are not
yet frozen; until they are locally available and hashed, the builder must emit
`training_ready=false` and must not fall back to exact matching alone.

## Feasibility Experiment F0

F0 asks one question: does the bounded structure space contain an outcome-improving action
that the small renderer can realize? It is not a benchmark shortcut and not yet an OPD,
DPO, or RL result.

Use 64 training-only prompts, generation seeds `{2026090101, 2026090102}`, and zero-based
decision indices `{8, 24, 40}`: 384 logical teacher states. At decision index `i`, the U-Net
for step `i` has already run and the action changes that step's scheduler transition. The
train and validation behavior/anchor is the same strict no-op checkpoint `C0`; it never
refreshes during F0. `T_OPSD` is used only for the separate realization suffix. The following
constants are frozen before the first decode:

```text
eta_target=0.25, target_steps=2, trust_radius_u=0.50
backtracking=(1.0, 0.5, 0.25, 0.125), epsilon_grad=1e-12
branch_coefficient=1.0, fit_steps_per_fold=1
vae_scaling_factor=0.13025
decoder_output=(decode(z / 0.13025)[0] / 2 + 0.5).clamp(0, 1)
```

The configured `vae_scaling_factor` must equal the pinned SDXL VAE config; a mismatch fails
before any reward query. At each target step, recompute the reward gradient at the detached
current target, normalize it by `max(||g||_2, epsilon_grad)`, take a step of length
`eta_target / target_steps`, and project `u-u_bar` to the trust ball of radius
`trust_radius_u`. Backtracking may choose only the first finite candidate satisfying the
fixed action, moment, scheduler, angle, and cap checks. It may not inspect a reward, ranking,
or image-quality outcome. If all four candidates fail, reject the state and charge its query.
At each student state, take the gradient in the renderer's active pre-squash action coordinates:

```text
u_bar = stopgrad(mu_ema(s_i))
a(u) = a_max * tanh(u)
Y(u) = apply_coefficients(s_i, a(u))
g_u = grad_u R_local(decode(Y(u))) at the detached current target
u_plus/minus = TargetStep(u_bar, +/- g_u, frozen constants)
a_plus/minus = a(u_plus/minus)
```

`apply_coefficients` includes the frozen moment geometry and scheduler cap, so every target
remains inside the deployed renderer action manifold. Only active dimensions participate.
Any predeclared backtracking responds to action/moment/cap validity, never to an observed
outcome. Evaluate `+/-` with a common prefix and common suffix randomness. F0 pairs may
initialize controls, but they cease to be on-policy labels after a checkpoint changes and
must not be silently reused by OPD, DPO, or RL.

For fitting, let `P_phi` be the scheduler transition produced by the current renderer mean,
and let `T_plus/T_minus` be the exact native Euler transitions of the detached targets.
Construct implicit branches `u_phi+ = u_bar + branch_coefficient*(mu_phi-u_bar)` and
`u_phi- = u_bar - branch_coefficient*(mu_phi-u_bar)` and map both through the real renderer.
The frozen F0 objective is:

```text
L_F0 = w * mse_tensor(P_phi+, stopgrad(T_plus)) / max(mse_tensor(N, 0), 1e-12)
     + (1-w) * mse_tensor(P_phi-, stopgrad(T_minus)) / max(mse_tensor(N, 0), 1e-12)
     + 0.10 * mean_active((mu_phi-u_ref)^2)
```

`w=clip(0.5 + 0.5*(r_endpoint-r_prompt_mean)/reward_scale, 0, 1)` is computed only from
the frozen round-0 anchor distribution. Fitting retains no decoder or reward graph and uses
one optimizer update per fold; action-coordinate regression is not substituted for the
native-transition loss.

Split the 64 prompts into four deterministic folds of 16, stratified by source and prompt
stratum before any output. For each fold, fit on the other 48 prompts and generate the held-
out fold's renderer action. Concatenate these out-of-fold actions for the realization test.
After that test passes, fit one final teacher on all 64 prompts with the same optimizer
settings; do not use its in-sample reward as evidence.

Define the frozen quantities for every logical state:

```text
G_target   = (R_select(plus) - R_select(anchor)) / reward_scale
G_realized = (R_select(out_of_fold_student) - R_select(anchor)) / reward_scale
direction  = 1 if R_select(plus) > R_select(minus) + 1e-6
             0 if R_select(plus) < R_select(minus) - 1e-6
             0.5 otherwise
```

`reward_scale` is the round-0 anchor scale defined under Frozen Statistics and Guards.
Target and realized suffixes use the same registered suffix randomness but distinct branch
IDs. F0 has a maximum of 35,200 physical U-Net calls, 1,664 VAE decodes, 2,944 reward
forwards, 384 reward backwards, 44 total active GPU-hours, and 22.5 GiB peak memory per GPU.
This includes 9,600 U-Net calls, 384 decodes, and 768 forwards for the out-of-fold student
suffix under `R_select` and the independent witness. The four fold fits and final all-prompt
fit each use 200 optimizer steps with the frozen main optimizer, for 1,000 fitting steps.
Fold optimizer seeds are `{2026090100, 2026090101, 2026090102, 2026090103}` and the final
all-prompt teacher seed is `2026090104`; a failed fit is reported and never retried with a
different seed.
The ledger records actual values. A gradient query and a scalar query are never reported as
equivalent merely because both count as one forward.

F0 is an OPSD-style feasibility cost reported once as shared teacher-construction cost. If
it passes, `T_OPSD`, its full receipts, and its immutable checkpoint become common inputs to
every main arm. `T_OPSD` is also a deployable baseline: run its deterministic mean on the
complete HPSv2 and GenEval suites, and report its 91,654 parameters, latency, memory, and
all safety guards against strict no-op and `conference_settings`. No main arm receives a
private teacher, gradient, target, or branch. Each arm's round data contain the same fields
when they are physically queried; the ledger records which fields each objective actually uses.

F0 passes only when all conditions hold:

- no-op parity, scheduler parity, finite-value, moment, and hard-cap violations are zero;
- valid nonzero-gradient coverage is at least 0.80 in every decision stratum;
- terminal `+` versus `-` direction accuracy is at least 0.55 and its prompt-level 95%
  lower bound is above 0.50;
- mean `G_target` is at least 0.10 with a 95% lower bound above zero;
- a reward not used to construct or select the targets improves by at least its frozen
  minimum effect, with a 95% lower bound above zero and all pixel guards passing;
- fewer than 1% of targets sit at 98% or more of the scheduler-update cap; and
- out-of-fold mean `G_realized` has a positive 95% lower bound and
  `mean(G_realized) / mean(G_target) >= 0.25`.

Power for the minimum effects must be frozen before any F0 output is viewed. A failed F0
closes that basis family; DPO or RL cannot rescue it.

After the train gate passes, run one confirmation on all 32 validation prompts with seeds
`{2026090191, 2026090192}` and the same three decision indices. This confirmation has 192
states, at most 17,600 U-Net forwards, 832 decodes, 1,472 reward forwards, and 192 reward
backwards, including a separate final-teacher realization suffix. It is used once for the
frozen teacher decision and never for fitting or hyperparameter selection. The same target,
realization, witness, cap, and pixel gates apply at prompt level; failure rejects the teacher.

## Training Arms

### Frozen First-Run Schedule

Run exactly two online rounds and one hyperparameter setting. In round 1, all arms start from
the same strict no-op checkpoint. In round 2, each arm starts from its own round-1 EMA
checkpoint; policies are expected to differ, so each arm recollects its own states with the
same prompt/seed manifest. A round has 32 optimizer updates, each with a fresh batch of four
prompt-seed blocks. The 128 blocks are assigned by one frozen permutation, so every block is
used exactly once per round. The behavior checkpoint is frozen for collection and is updated
by EMA only after that update. Consequently, external-teacher OPD performs a new rollout
after every parameter update; a state collected before an update is never reused as an OPD
state. A separate one-collection/32-fit experiment is named `offline_teacher_distill` and is
not reported as OPD.

At each of the three decisions on an evolving branch, draw a registered standard-normal
vector `epsilon_t` and use fixed pre-squash `sigma=0.25`:

```text
u_plus  = mu_behavior(s_plus)  + 0.25 * epsilon_t
u_minus = mu_behavior(s_minus) - 0.25 * epsilon_t
a_branch = a_max * tanh(u_branch)
```

The two branches use antithetic vectors from one versioned RNG manifest. At index 8 their
states are identical; at indices 24 and 40 their states differ because earlier sampled
actions changed the trajectory. The deterministic anchor takes the behavior mean at all
three decisions. OPD, DPO, RL, and search-distill use this same proposal distribution; they
cannot change `sigma`, use IID instead of antithetic noise, or add candidates.

For each 64-prompt x 2-seed round, the non-transferable allocation for the sampled-pair arms
(`search_distill`, `renderer_dpo`, and `reference_rl`) is:

- 128 prompt-seed blocks, 384 deterministic anchor states, and 768 sampled branch actions;
- one anchor and two sampled terminal branches per block: 384 physical training-reward
  forwards and 384 terminal VAE decodes;
- zero training-reward backwards;
- 16,896 physical U-Net forwards: indices `0..8` run once, then the anchor and two sampled
  branches each run indices `9..49`, so `128 * (9 + 3 * 41) = 16,896`;
- at most 42 active GPU-hours and 22.5 GiB peak memory per GPU.

The external-teacher OPD and `offline_teacher_distill` arms receive the same 128 blocks plus
384 detached `T_OPSD` labels and one independently rolled-out teacher branch per block. Their
allocation is 512 reward forwards/decodes and 22,144 U-Net forwards
(`128 * (9 + 4 * 41)`); this teacher-label cost is reported separately and is not silently
charged to the sampled-pair cohort. Every round uses exactly 32 optimizer updates with batch
size four, AdamW learning rate `1e-4`, betas `(0.9, 0.999)`, weight decay `0.01`, gradient-norm
cap `1.0`, and EMA decay `0.995`. There is no early stopping, learning-rate search,
objective-weight search, or retry with a new seed. Independent optimization seeds are
`{202609011, 202609012, 202609013}` and are reported as three runs, not pooled as extra
prompt samples.

Failed, rejected, and fallback branches remain charged. Reward/decode/U-Net allocations
cannot move between methods, rounds, or forward/backward categories. A stronger teacher or
extra proposal is a compute-unmatched upper bound and cannot enter the primary comparison.

After round 2, use all 32 validation prompts and both validation seeds once. Each sampled-pair
arm receives 64 blocks, 384 sampled action decisions, 192 terminal reward forwards/decodes,
and 8,448 U-Net forwards; each teacher-label arm receives 64 blocks, 192 `T_OPSD` labels,
256 teacher/branch reward forwards/decodes, and 11,072 U-Net forwards. Validation never
updates weights. Both round-end checkpoints are frozen artifacts and therefore receive
complete HPSv2 and GenEval evaluation.

Across two rounds plus validation, a sampled-pair optimization run receives 960 terminal
reward forwards/decodes and 42,240 U-Net forwards; a teacher-label run receives 960 teacher
labels, 1,280 reward forwards/decodes, and 55,360 U-Net forwards. The active-GPU cap is 105
hours per seed. These totals apply separately to each of the three optimization seeds; they
are never pooled or transferred. Every arm reports teacher target gain
`(R_select(teacher_branch)-R_select(anchor))/reward_scale` separately from its own
deterministic checkpoint realization gain.

### External-Teacher OPD

The OPD arm is disabled until F0 produces a frozen, validation-passing OPSD-trained renderer
checkpoint `T_OPSD`. Teacher construction is reported once with all gradient queries, GPU
time, and failed targets. `T_OPSD` uses the same coefficient action API and has at most 1M
parameters. All arms start from the same strict no-op student and receive the same
`T_OPSD` labels; no arm is initialized from the teacher.

For every one of the 32 updates, the student first rolls out the current four-block batch and
stores the three states it actually visited. `T_OPSD` is evaluated on those exact detached
states, not on teacher-generated states. Let `T_i` be the teacher scheduler transition and
`N_i = E_i(x0_hat_i) - x_i` be the native transition. With the initial no-op policy as fixed
reference, `mse_tensor` means the elementwise mean over channels and spatial dimensions,
then the outer mean is over the three decisions and four-block batch:

```text
L_OPD = mean_i mse_tensor(P_phi(s_i), stopgrad(T_i))
                 / max(mse_tensor(N_i, 0), 1e-12)
        + 0.10 * mean_active((mu_phi(s_i) - u_ref(s_i))^2)
```

`P_phi` is the exact native Euler transition after the renderer mean is mapped through the
fixed frame, caps, and scheduler conversion. The target is detached and no decoder/reward
graph enters the fit. One optimizer update follows each fresh collection; then the behavior
checkpoint is updated by EMA. Teacher inference calls, fresh-rollout calls, and all failures
are recorded separately. Reusing old student states after an update is
`offline_teacher_distill`, not OPD. The direct deployable `T_OPSD` baseline is evaluated
before any student result is inspected.

### Search-Distill Baseline

Select the better of the same two terminal sampled branches and regress only its native
scheduler transitions with exactly the normalized transition loss and active-dimension
anchor used by OPD. Action-coordinate regression is not an option. This is the winner-only
supervised baseline for DPO. It is not external-teacher OPD, even though it labels current-
policy branches. It uses the same fresh four-block collection/update cadence as DPO and RL;
its target is detached after branch selection and ties receive zero loss weight.

### Renderer-DPO

Use a fixed-variance transformed Gaussian:

```text
u ~ Normal(mu(s), sigma)
a = a_max * tanh(u)
```

For chosen and rejected three-action trajectories, define the per-decision reference-relative
log-ratio (the states are the states actually visited by each branch):

```text
Delta_phi = mean_t [log pi_phi(a_chosen_t | s_chosen_t)
                    - log pi_phi(a_rejected_t | s_rejected_t)]
Delta_ref = the same mean under the frozen initial no-op reference
L_DPO = -log sigmoid(0.10 * (Delta_phi - Delta_ref))
```

The implementation includes the exact `tanh` change-of-variables Jacobian, fixed variance,
fixed global mask, and numerical boundary handling. The reference never refreshes. The
chosen/rejected label is determined once from the frozen training reward normalization and
is shared by all pair-based arms. Compare exact DPO with search-distill, shuffled preferences,
and the frozen reference. If DPO does not beat search-distill, report preference distillation,
not a DPO contribution.

### Reference-Anchored RL

The policy samples at all three decisions on each evolving branch. Earlier actions change the
later state and later action distribution. Use the DDPO-style per-decision ratio, rather than
one ratio for an entire trajectory. For a paired terminal reward `r_b`, define the frozen
leave-one-out group advantage at every decision:

```text
A_b,t = (r_b - mean(r_{-b})) / max(|r_+ - r_-|, 1e-6)
ratio_b,t = exp(log pi_phi(a_b,t | s_b,t)
               - log pi_behavior(a_b,t | s_b,t))
L_RL = mean_b,t[-min(ratio_b,t * A_b,t,
                     clip(ratio_b,t, 0.8, 1.2) * A_b,t)]
       + 0.01 * KL_ref
KL_ref = mean_b,t,active 0.5 * ((mu_phi(s_b,t) - mu_ref(s_b,t)) / 0.25)^2
```

The behavior policy is the frozen round-start checkpoint and the reference is the frozen
initial no-op policy. `mean(r_{-b})` is the other antithetic branch for the two-branch group;
for the optional four-branch sensitivity audit it is the mean of the other three branches.
Advantages are zero for ties. Ratios are clipped only as shown; there is no reward shaping or
hidden entropy bonus. Both policies have fixed `sigma=0.25` and the same bijective `tanh`
transform, so `KL_ref` is the exact Gaussian KL on the fixed active coordinates, averaged over
branches, decisions, and active dimensions. Report the per-decision ratio and advantage
ledger. Compare state-conditioned RL with prompt/timestep-only open loop, shuffled state,
and the reference. A sequential RL claim requires a pre-registered nonzero early-action
counterfactual effect and a gain over open loop; otherwise report contextual-bandit policy
optimization.

## Frozen Statistics and Guards

`R_local` and `R_select` are the same frozen ImageReward v1.0 checkpoint and preprocessing.
The implementation config must bind the model revision, every checkpoint file SHA-256,
preprocessing code hash, package versions, input range, resize, and normalization before any
decode. No ensemble, crop expansion, HPS weight, or online reward update is allowed in the
first contract. TOPIQ-NR is the independent F0 witness; HPSv2, CLIP, and official GenEval
never feed target construction, preference labels, advantages, stopping, or hyperparameters.

Compute the training-reward location and scale once from the 128 round-0 deterministic anchor
images, before reading any branch outcome: median and `max(IQR / 1.349, 1e-6)`. Reuse these
numbers for every arm, round, and seed. No per-arm or post-outcome normalization is allowed.

The statistical unit is the prompt. Average generation seeds and decision indices within a
prompt, then run 10,000 prompt-cluster bootstrap resamples with seed `20260901`. Report the
paired mean and percentile 95% interval. Direction accuracy is also aggregated to one value
per prompt. The independent witness is TOPIQ-NR with minimum mean delta `+0.005` and a 95%
lower bound above zero. The clipped-fraction interval upper delta must be at most `+0.001`,
mean-saturation interval upper delta at most `+0.005`, and contrast geometric-ratio interval
inside `[0.95, 1.05]`. HPSv2 and CLIP lower bounds must exceed `-0.005` where those metrics
are used. A power artifact for 64 train and 32 validation prompts is frozen before output;
insufficient power is a no-go, not permission to add prompts after seeing results.

For a terminal ImageReward difference within `1e-6`, search-distill uses the deterministic
anchor target, DPO retains the record with zero loss weight, and RL uses zero advantages.
The query remains charged and cannot be replaced. Non-ties use the same chosen/rejected label
in every arm.

## Query Ledger

`QueryLedger` is append-only and reserves budget before compute. An unfinished reservation
after a crash is conservatively charged. Every receipt binds the run contract, code/data/
checkpoint hashes, method allocation, split, prompt, seed, step, prefix, branch, action,
image hash, reward/preprocess hash, scalar-or-gradient mode, result hash, wall time, and
cached parent.

Report at least:

- reward forward and backward examples by model;
- logical labels and label-reuse events;
- VAE decodes, physical U-Net forwards, scheduler transitions, and suffix NFE;
- optimizer steps, GPU-hours, peak memory, failures, retries, and discarded branches.

Caching may reduce physical compute, but each method also receives a separate counterfactual
allocation so that shared labels do not create a hidden budget advantage.

## Engineering and Tests

Add a separate `latent_renderer_training/` package with contracts, policy/action mapping,
rollout, storage, tensor-native rewards, teachers, objectives, ledger, trainer, and one CLI.
Public logic has one implementation; method differences live in versioned configuration.

Before a GPU run, tests must cover action/forward equivalence, exact no-op, transformed-
Gaussian log-probability and KL, all Euler sigma pairs, CFG batch order, one U-Net call per
step, shared-prefix replay, reward gradients only to the clean latent/policy, OPD target
stop-gradient, DPO label-swap direction, an RL toy problem, ledger crash recovery and budget
rejection, checkpoint resume, and deployment loading parity.

Frame-specific tests must cover fixed-moment orthogonality, native-metric Gram identity,
zero/duplicate/linear-combination rank masks, batch-independence, eigensolver sign/order
invariance, all 64 coefficient-box corners, pre/post-geodesic cap bounds, cap multiplier one,
byte-exact zero action, inactive-dimension probability removal, frame-hash replay, deterministic
Haar keys and QR signs, and fail-closed zero update, invalid `kappa`, NaN, all-degenerate, or
excess-orthogonality-error states. A reference-order test must fail if cap or geodesic is moved
before frame mixing.

## Evaluation and Stop Rules

Every frozen checkpoint is evaluated on complete HPSv2 (3,200 images) and official GenEval
(553 prompts times four images, 2,212 total). Training rewards cannot be the only primary
metric. Report three independent training seeds separately, then paired prompt-level
confidence intervals; do not multiply the apparent sample size by seeds or settings.

A round-2 method is compared pairwise with both `strict_no_op` and the frozen
`conference_settings` baseline, using the same prompt, generation seed, sampler, and initial
noise. TOPIQ-NR is scored on the complete 3,200-image HPSv2 output set. For each metric,
average the three optimization-seed results within prompt, then bootstrap prompts while
keeping all three model seeds together. All three seeds must pass safety guards, and at least
two of three seed-level point estimates must have the required direction; no best-seed
selection is allowed.

The aggregate method passes the first complete-benchmark efficacy gate only if, relative to
`strict_no_op`, TOPIQ-NR improves by at least `+0.005` with a positive 95% lower bound and
either:

- HPSv2 improves by at least `+0.005` with a positive prompt-level 95% lower bound while
  GenEval's lower bound is at least `-0.01`; or
- official GenEval improves by at least `+0.01` with a positive prompt-level 95% lower bound
  while HPSv2's lower bound is at least `-0.005`.

The clipping, saturation, contrast, CLIP, finite-value, moment, and cap guards must also pass.
Relative to `conference_settings`, TOPIQ-NR must have a positive lower bound and HPSv2 and
GenEval must meet the same non-inferiority bounds. Apply Holm correction to the four primary
arms (OPD, search-distill, DPO, RL) within each metric family. Round 1 is reported but cannot
change round 2 or any hyperparameter.

Only a candidate that passes the full first gate may add DPG-Bench, higher resolution,
another backbone, ControlNet, or the VAE frequency/equivariance study. The final paper also
requires blinded human comparison, matched random/free-residual controls, latency and memory,
and direct mechanism evidence. Good proxy scores without these checks are not a TPAMI claim.

## Required Related Work

At minimum, compare or delimit DiffusionOPD, Flow-OPD, DiffusionOPSD, LPO,
Diffusion-DPO, DDPO, DPOK, DRaFT, Flow-GRPO, BranchGRPO, Diffusion Controller, RVM,
FreeU, [FreSca](https://arxiv.org/abs/2504.02154),
[FDG](https://arxiv.org/abs/2506.19713), [SPA](https://arxiv.org/abs/2607.22091),
Improving the Diffusability of Autoencoders, and EQ-VAE. Frequency separation, tangent
projection, low-dimensional latent control, OPD, DPO, and RL are occupied ingredients. The
conference-to-journal link is the shared idea of model-native latent structure, not an
unsupported claim that the optimizer or any one frame operation is Attention Guidance.
