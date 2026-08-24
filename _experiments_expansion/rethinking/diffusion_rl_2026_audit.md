# 2026 Diffusion-RL Literature Audit

## Why This Audit Exists

The repository has not yet found a fixed update that beats `no_ag` with the
registered guards. It would therefore be a mistake to start RL just because
recent papers report large reward gains. This note records what those papers
actually solve and what remains distinct for RepLDM.

The abstracts below were checked from the arXiv API on 25 August 2026. They are
research context, not evidence about this repository's image quality.

## What Recent Work Changes

| Work | Main idea | Consequence for RepLDM |
|---|---|---|
| SGPO, arXiv:2608.06768 | Assigns different objectives to early, middle and late denoising stages using SNR and semantic change. | Stage-specific credit is a useful control, but copying its RL objective would not make a frozen latent renderer novel. |
| CRD, arXiv:2603.14128 | Centers rewards within each prompt and anchors the policy to a CFG-guided reference to limit drift. | Any later learner needs within-prompt centering and a trust/KL anchor; a raw ImageReward objective is not acceptable. |
| GeMPO, arXiv:2603.10250 | Generalizes sample reweighting and can use signed weights to repel poor actions. | Reweighting is an optimizer baseline, not our contribution; negative weights must be compared with antithetic controls. |
| MARBLE, arXiv:2605.06507 | Balances several reward gradients by solving a small quadratic program. | Multi-reward conflicts need to be measured, not hidden in one weighted sum. |
| JAGG, arXiv:2607.17572 | Aggregates Jacobians across nearly linear consecutive steps to reduce GRPO backward passes. | Useful for cost accounting, but the approximation must be tested against exact gradients before use. |
| RTDMD / REST, arXiv:2605.26108 / 2608.09226 | Combines reward optimization with distribution-matching or scored-trajectory distillation. | Search-then-distill is a mandatory baseline before online RL for a small renderer. |
| DiffRGD, arXiv:2606.28417 | Uses Riemannian latent updates to preserve a Gaussian latent structure. | It overlaps our moment/geodesic idea; we cannot claim the geometry as new. |
| LatSearch, arXiv:2603.14526 | Uses intermediate latent rewards for resampling and pruning. | Intermediate reward can reduce sparse credit, but it adds candidates/compute and must be a matched-cost control. |
| Path-space view, arXiv:2608.14430 | Unifies reverse-trajectory and forward-matching diffusion-RL objectives. | The estimator family is not a novelty opportunity; report which variance-reduction choice is used. |

## Revised Hypothesis

The only defensible journal extension remains a **scheduler-consistent,
low-capacity residual renderer** attached to the frozen RepLDM trajectory. Its
output is a bounded coefficient vector over fixed structural bases, injected
through the scheduler's predicted-clean latent. It has a soft conceptual link
to the conference attention guidance (both reprogram a frozen latent
trajectory), but it is not presented as a successful rescue of the old TFSA
residual.

The sequence is deliberately strict:

1. Finish S7 fixed-action development and validation. If no action passes the
   frozen TOPIQ, HPS/CLIP and pixel gates, close this route and report the
   negative result.
2. If S7 passes, compare a fixed structural renderer with a no-op, conference
   action, native scheduler, random matched-capacity residual, and an equal-cost
   extra-UNet control. Do not train it yet.
3. Only after positive static headroom, collect bounded teacher coefficients
   with shared-prefix antithetic search and fit a tiny deterministic renderer.
4. Allow online RL only if it beats fixed search and distillation on unseen
   prompts while preserving a reference-KL/trust bound and a held-out witness.

## Reward and Evaluation Controls

The existing ImageReward/HPS/aesthetic family is mostly evaluated through
224-pixel encoders. It is therefore a weak witness for fine detail and can
reward colour or saturation changes. A later learner must use patch or
multi-crop structural witnesses and full-resolution guards, while keeping the
primary test metric completely held out. Within-prompt reward centering,
antithetic actions, a reference trust region, and explicit clipping/saturation
constraints are required. A composite reward is logged component by component;
it is never used as the sole publication endpoint.

This audit does not authorize an RL run. It narrows the next experiment and
prevents a sophisticated optimizer from hiding a failed fixed operator.
