# S5 Related-Work Audit

This audit was frozen before implementing or generating S5. It is a rejection-risk screen, not a claim that attention transport is broadly new. Full texts were checked through 24 August 2026.

## Closest Inference-Time Guidance

| Line of work | Prior coverage | Required consequence for S5 |
|---|---|---|
| SAG (2210.00939), PAG (2403.17377), SEG (2408.00760), and ASAG (2511.07499) | Build an extra degraded or perturbed self-attention prediction and contrast denoiser outputs. | Treat these as higher-compute, extra-NFE upper bounds. A one-pass claim must report NFE, latency, and memory rather than calling itself cost-free. |
| NAG (2505.21179) | Extrapolates normalized positive/negative attention outputs, especially for negative guidance. | Attention extrapolation and normalization are not novel components. |
| PLADIS (2503.07677) | Within one forward pass, contrasts dense and alpha-entmax cross-attention with no training or extra NFE. | This is the closest same-NFE baseline. Pin the author source, use scale `2.0`, query-dtype probabilities, and its default SDXL `[up, down]` layers. A diffusers `0.32.1` port is not an end-to-end reproduction of the stated `0.33.1` environment. |
| GAG (2603.02531v2) | Decomposes PLADIS's sparse-dense contrast and retains a capped retrieval-aligned component. | No author code was verified as of 2026-08-25. Use an explicitly paper-derived Eq. 12/13 implementation with lambda `10`, eta `15`, and zeta `0`; disclose that alpha and layer policy are inherited from PLADIS. |
| Self-Guidance (2306.00986), Diffusion Self-Guidance (2412.05827), S2-Guidance (2508.12880), EMAG (2512.17303), and Internal Guidance (2512.24176) | Obtain guidance from pretrained-model predictions, attention, features, or internal dynamics without an external classifier. | Neither predicted clean samples nor internal semantic features are independently new. |

## Control and Editing Methods

PnP (2211.12572), MasaCtrl (2304.08465), FreeControl (2312.07536), Universal Guidance (2302.07121), and Readout Guidance (2312.02150) use feature injection, attention control, external objectives, or learned readouts for editing and conditional control. They establish that internal features can encode spatial correspondence. S5 is evaluated as unconditional quality guidance with no reference image, external loss, backpropagation, or trained readout; it must not borrow an editing claim from these methods.

## Adaptive and RL Methods

AdaNAT (2409.00342), AdaGen (2603.06993), Dynamic CFG (2509.16131), LeSAMP (2607.23488), BranchGRPO (2509.06040), DRM (2605.25661), and SGPO (2608.06768) already cover learned, prompt-conditioned, state-dependent, timestep-dependent, branching, or stage-wise generation policies. Therefore a learned schedule or RL controller is not an S5 contribution. RL is prohibited unless a fixed static S5 action first clears the registered quality, alignment, and pixel-safety gates on unseen prompts.

## Narrow Claim Under Test

The only admissible S5 distinction is the complete operator: read standard conditional self-attention without modifying the UNet; construct a reciprocal, mutual-neighbor spatial graph; transport the scheduler-returned predicted clean sample on a fixed-moment manifold; and inject the clean-sample displacement as `prev_sample + u` without backpropagation or another UNet call.

The claim is rejected if PLADIS or GAG wins, if a spatially permuted equal-compute graph performs similarly, if raw or clean four-channel affinity matches semantic affinity, or if gains come from clipping, saturation, contrast, or prompt reuse. Even a development pass requires a prompt-disjoint confirmation set, high-resolution transfer, and blinded human comparison before publication language is justified.

The original S5 YAML action IDs containing `official` are immutable historical
identifiers. Their actual implementations and corrected claim boundaries are
recorded in `../audits/BASELINE_PROVENANCE.md`; they must not be cited as official code
reproductions.
