# S7 Sampler and RL Gate Audit

**Audit date:** 2026-08-25. Titles, first-submission dates, and mechanism
summaries were checked against official arXiv abstract pages. The papers are
novelty boundaries and controls, not evidence that their results transfer to
SDXL.

## Sampler-Independent Boundary

The registered S7 action is a bounded post-step correction on an
`EulerDiscreteScheduler`. It must be reported as a fixed sampler ablation. A
gain over the Euler baseline is not a method gain until the same frozen U-Net,
CFG, initial noise, resolution, VAE, and denoiser-call budget are compared with
the following native references:

| Control | Direct 2025-26 precedent | What it rules out |
|---|---|---|
| Native Euler-Ancestral | [LC-GRPO, 2608.05600](https://arxiv.org/abs/2608.05600) (2026-08-06) aligns an ODE Euler step before Langevin exploration; [CPS, 2509.05952](https://arxiv.org/abs/2509.05952) (2025-09-07) analyzes excess SDE noise. | A stochastic-looking S7 mix cannot be called a new SDE or RL sampler. Use the native endpoint with independently recorded noise. |
| DPM-Solver++ | [Diffusion-Sharpening, 2502.12146](https://arxiv.org/abs/2502.12146) (2025-02-17) optimizes whole denoising trajectories and compares inference scaling. | A trajectory improvement may be an ordinary high-order solver effect. Freeze the exact DPM-Solver++ order/timestep schedule and count every U-Net call. |
| UniPC | [Precise, 2605.23522](https://arxiv.org/abs/2605.23522) (2026-05-22) treats finite-step SDE discretization as part of the policy. | Solver discretization and stochasticity are confounds; report UniPC at equal NFE, not only equal nominal steps. |
| ODE/SDE interpolation | [MixGRPO, 2507.21802](https://arxiv.org/abs/2507.21802) (2025-07-29) confines SDE exploration to a sliding window; [LC-GRPO](https://arxiv.org/abs/2608.05600) adds Langevin correction after an ODE step. | `mix` is an interpolation knob unless derived from a valid SDE and tested against these controls. |
| Mean-preserving guidance | [NoiseTilt, 2606.18066](https://arxiv.org/abs/2606.18066) (2026-06-16) leaves the reverse mean fixed and tilts only the noise term. | A clean-latent/mean residual must be separated from noise guidance; otherwise the claimed source of improvement is ambiguous. |
| No-extra-NFE compensation | [DC-Solver, 2409.03755](https://arxiv.org/abs/2409.03755) (2024-09-05) dynamically compensates predictor-corrector error. | "No extra denoiser calls" is not novelty; include a matched compensation control or state why it cannot run on SDXL. |

### Minimal S7 comparison

Freeze the action and all settings before validation. The minimum paired block
is:

| ID | Action | Selection status |
|---|---|---|
| S0 | EulerDiscrete, no correction | baseline |
| S1 | The frozen selector output (use `drift_mix_025` only if that is the selected action) | candidate |
| S2 | Native Euler-Ancestral, same timesteps and seeded initial noise | reference, never selected by S7 selector |
| S3 | DPM-Solver++ (fixed order/config, equal denoiser evaluations) | reference |
| S4 | UniPC (fixed order/config, equal denoiser evaluations) | reference |
| S5 | Drift-only versus noise-only/energy-matched variant, only if already registered | mechanism diagnostic, not a post-hoc search |
| S6 | DC-Solver-style no-extra-NFE compensation, if an exact SDXL-compatible implementation is available | matched control |

For S2-S4, "equal NFE" means equal actual U-Net evaluations, not equal loop
length. Record solver order, timestep list, model-output type, churn/noise
draws, scheduler config hash, wall time, and peak memory. Use the same
`(prompt, seed)` on one device and preserve action execution order in the
manifest. A selected S7 action that only beats S0 but loses to S2-S4 is a
sampler result and closes the S7-to-renderer route.

## Search-Then-Distill Is the RL Null Model

Several recent works make a learned RL controller an especially high burden:

| Work | Verified mechanism | Gate implication |
|---|---|---|
| [LatSearch, 2603.14526](https://arxiv.org/abs/2603.14526) (2026-03-15) | Latent reward-guided search with intermediate signals for video inference scaling. | Per-sample search is a required oracle/cost baseline; report the quality-versus-number-of-new-prompts crossover. |
| [Diffusion-Sharpening, 2502.12146](https://arxiv.org/abs/2502.12146) (2025-02-17) | Path-integral trajectory selection amortized into a fine-tuned model. | Whole-trajectory search plus amortization is already established. |
| [CRD, 2603.14128](https://arxiv.org/abs/2603.14128) (2026-03-14) | Centered reward distillation with KL-regularized forward-process training. | Distillation and reference anchoring are mandatory before online policy gradients. |
| [RTDMD, 2605.26108](https://arxiv.org/abs/2605.26108) (2026-05-25) | Reward-tilted distribution matching for few-step generators. | Reward maximization plus distribution matching is not an RL novelty. |
| [REST, 2608.09226](https://arxiv.org/abs/2608.09226) (2026-08-10) | Distills scored RL trajectories into a decoupled few-step student. | Reusing scored trajectories for a deterministic student is a direct control. |
| [GDRO, 2601.02036](https://arxiv.org/abs/2601.02036) (2026-01-05) | Offline group-level reward optimization for deterministic flow models. | Online stochastic rollouts must justify their extra cost and variance. |
| [JAGG, 2607.17572](https://arxiv.org/abs/2607.17572) (2026-07-20) | Aggregates/interpolates Jacobians to reduce diffusion-RL backward cost. | Cached gradients or shared prefixes do not establish estimator novelty. |

The null model is therefore explicit:

```text
fixed basis search -> prompt/bucket labels -> tiny deterministic distiller
```

Search uses common random numbers, a predeclared coefficient/action grid, and
training prompts only. Distillation uses the same renderer inputs and parameter
budget as RL, but no backpropagation through the diffusion rollout. Freeze the
distiller once and evaluate it on unseen prompts. Per-prompt search remains an
oracle and must be charged its reward and denoiser calls; the distiller is the
deployment baseline and must be charged its training cost amortized over the
number of prompts.

## Falsifiable RL Gates

1. **Headroom gate.** On a prompt-disjoint validation set, RL must beat the
   frozen search-then-distill policy on TOPIQ-NR by the preregistered minimum
   effect (use `+0.005` only if retained from the existing protocol), with a
   crossed prompt/seed 95% interval above zero and Holm-adjusted prompt
   sign-flip significance. Matching the interval or losing on HPSv2/CLIP
   rejects RL.
2. **Oracle/adaptivity gate.** Estimate the per-prompt oracle gap between the
   best fixed action and the best allowed action using a separate reward/metric
   from selection. If the gap is within crossed seed uncertainty, collapse to a
   global or 2-3-stage action; a sequential policy is unjustified.
3. **Cost gate.** Report search denoiser calls, reward calls, training GPU-hours,
   parameter/FLOP count, deployment latency, and peak memory. RL must recover
   its training cost after a predeclared number of new prompts; otherwise the
   simpler distiller wins even if a point metric is marginally higher.
4. **Robustness gate.** The RL gain must persist under an independent reward
   family, native-resolution patch/multi-crop evaluation, color-normalized
   scores, artifact/clipping/saturation guards, diversity, and blinded human
   preference. A scalar reward increase alone is reward hacking.
5. **Reference/KL gate.** Compare RL from scratch, KL/data-anchored RL, and
   antithetic shared-prefix rollouts. If the anchored or deterministic
   distiller is non-inferior at lower cost, do not claim online RL necessity.

## Smallest Executable Matrix After S7

Run this only if S7 passes its existing fixed-action gate; it does not add
actions to the running S7 queue:

| Stage | Arms | Minimum evidence |
|---|---|---|
| M0 sampler | S0-S4 above, plus S5/S6 when registered | Native solver comparison, NFE/cost accounting, no pixel-safety regression. |
| M1 fixed renderer | no-op; best six-basis coefficient vector; best 2-3-stage vector; L2-matched random basis; shuffled structural features; equal-cost extra U-Net call | Fixed basis beats no-op/conference TFSA and random/shuffled controls on held-out quality and structure; moments/trust cap hold. |
| M2 distillation | search labels -> ridge/2-layer MLP; context-free schedule; state/prompt-conditioned distiller | Conditioned distiller recovers a predeclared fraction of the oracle gap on unseen prompts without extra denoiser calls. |
| M3 RL necessity | same renderer as M2, antithetic shared-prefix RL; KL/data-anchored variant; offline/group baseline | RL beats M2 on independent witnesses and amortized cost; otherwise stop and report M2. |

Use the existing disjoint train/validation/test manifests and reserve final
seeds. Do not tune solver settings, action caps, reward weights, or stopping
rules after validation scores. Any S7 failure terminates M1-M3; no learned
controller or paper framing can rescue it.
