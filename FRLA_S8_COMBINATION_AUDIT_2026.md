# FRLA/S8 Combination-Space Audit

**Cutoff:** 2026-08-25. **Status:** literature and protocol audit only; no GPU
run, score, or implementation result is added here. The arXiv API query
`https://export.arxiv.org/api/query?id_list=2503.08253,2512.10794,2605.16949,2605.20808,2512.05394,2606.03578,2602.00883,2606.28417,2603.11509,2607.21326,2608.16513&max_results=30`
returned the records below (feed updated `2026-08-24T22:59:02Z`).

## Bottom Line

FRLA's components are covered separately. SARA, iREPA, sREPA, SGA, and SSVAE
already use autocorrelation, cosine self-similarity, Gram structure, or local
correlation to shape diffusion features or latents. DIAMOND, DiffRGD, MOG,
DUNE, SlerpFlow, and the new MLLM correction already cover clean-latent or
internal-feature trajectory intervention, geometric projection, and
mid-generation latent correction. The exact pairing of a **fixed,
rank-compatible local descriptor** with a **one-call scheduler-coordinate
clean-latent injection** was not found in this API audit, but that pairing is a
combination hypothesis, not component novelty. Risk is high.

The only defensible FRLA claim is therefore a positive, preregistered
descriptor-by-injection interaction at matched NFE and compute. A point metric
gain, a new name for a Gram/autocorrelation loss, or a trust cap alone is not a
contribution. If the interaction is absent, report FRLA as an assembled
inference control and close the route before distillation or RL.

## Overlap Audit

| Record (arXiv version) | What is already covered | Consequence for FRLA |
|---|---|---|
| [SARA, 2503.08253v1](https://arxiv.org/abs/2503.08253v1), 2025-03-11 | Training-time hierarchical alignment includes patch matching, **autocorrelation-matrix alignment**, and adversarial distribution alignment. | Do not claim first autocorrelation/structural alignment. Use a fixed local-autocorrelation target and a patch/Gram control; distinguish inference-only SDXL from training-time ImageNet evidence. |
| [iREPA, 2512.10794v1](https://arxiv.org/abs/2512.10794v1), 2025-12-11 | Shows pairwise cosine spatial structure predicts alignment quality and uses convolutional projection plus spatial normalization. | Do not claim first pairwise-cosine spatial descriptor or spatially aware projection. Include pointwise, cosine, and shuffled-spatial controls. |
| [sREPA, 2605.16949v1](https://arxiv.org/abs/2605.16949v1), 2026-05-16 | Explicitly matches relational geometry/similarity distributions of teacher and diffusion feature maps during training. | A relational feature target is directly overlapped. No "first relational geometry" claim; an inference-time detached feature test must show a causal regime difference. |
| [SGA, 2605.20808v1](https://arxiv.org/abs/2605.20808v1), 2026-05-20 | Aligns internal self-similarities of intermediate diffusion features and VAE latents while avoiding pointwise manifold distortion. | The phrase "non-invasive spatial Gram constraint" is already used. Reproduce a Gram/self-similarity arm or explain why the fixed four-channel lag descriptor tests a different rank and compute regime. |
| [SSVAE, 2512.05394v3](https://arxiv.org/abs/2512.05394v3), 2025-12-05 (revised 2026-07-24) | Uses local-correlation regularization and latent spectral/eigenspectrum shaping to improve video diffusability. | Local correlation is not new, and a correlation proxy cannot be presented as a proven image mechanism. Keep the SDXL VAE frozen and report correlation, DCT, moments, and quality separately. |
| [Diffusing in the Right Space, 2606.03578v1](https://arxiv.org/abs/2606.03578v1), 2026-06-02 | Systematically evaluates latent spatial structure, spectral smoothness, equivariance, and velocity ambiguity; VIV is a trajectory diagnostic. | Do not select a relation target from one proxy or call VIV causal evidence on SDXL Euler. Use scheduler-agnostic trajectory diagnostics as witnesses only. |
| [DIAMOND, 2602.00883v1](https://arxiv.org/abs/2602.00883v1), 2026-01-31 | Reconstructs a clean sample and applies an artifact-detector gradient during every inference step. | Clean-latent gradient trajectory correction and artifact steering are covered. Include a matched detector/dummy-gradient arm and charge backward cost. |
| [DiffRGD, 2606.28417v2](https://arxiv.org/abs/2606.28417v2), submitted 2026-06-25/revised 2026-07-02 | Uses Riemannian gradient descent on a Gaussian-induced spherical latent manifold. | Tangent projection, Gaussian-moment preservation, and constrained latent guidance are not FRLA novelty. Compare Euclidean, spherical, MOG, and covariance controls. |
| [MOG/Auto-MOG, 2603.11509v1](https://arxiv.org/abs/2603.11509v1), 2026-03-12 | Provides closed-form Riemannian CFG guidance and dynamic energy balancing without retraining. | Neither manifold-aware guidance nor adaptive energy scheduling can be claimed. Freeze fixed-energy and Auto-MOG references. |
| [SlerpFlow, 2607.21326v1](https://arxiv.org/abs/2607.21326v1), 2026-07-23 | Uses spherical interpolation to correct rectified-flow velocity directions and caches corrected velocity without extra training. | Spherical trajectory correction, curvature-aware direction repair, and cached correction are covered, even though its task is inversion/editing. Include spherical/velocity controls and do not call scheduler-coordinate geometry new. |
| [MLLM-Guided Semantic Correction, 2608.16513v1](https://arxiv.org/abs/2608.16513v1), 2026-08-17 | Performs training-free mid-generation semantic assessment and controllable latent trajectory intervention, using preview frames and MLLM feedback. | Mid-generation latent intervention and semantic trajectory correction are not firsts. The extra MLLM/preview calls are a cost upper bound and a semantic-control reference, not a matched one-call baseline. |

The records are adjacent rather than identical: the representation-alignment
papers train a model or VAE, while FRLA is inference-only; DIAMOND and MLLM
correction use task/reward feedback, while FRLA uses one detached U-Net feature;
SlerpFlow targets inversion. Those differences support a narrow combination
test, but they do not restore component-level novelty.

## Required FRLA Factorial

Before reading held-out scores, freeze the feature block, deterministic
four-channel reduction and projection seed, `16 x 16` grid, five lags, `eta`,
trust cap, covariance retraction, scheduler, CFG, resolution, timestep list,
and initial-noise policy. Use common random numbers and a prompt-disjoint
development/validation/test manifest.

The development block must separate descriptor and injection effects:

| Axis | Required levels |
|---|---|
| Descriptor | no-op; pointwise feature statistic; fixed five-lag cosine descriptor; full Gram/self-similarity descriptor (rank-matched implementation); shuffled feature tokens; L2-matched random direction |
| Injection | no injection; Euclidean clean-latent residual; scheduler-coordinate `prev_sample + (guided_x0 - pred_original_sample)`; spherical/MOG/DiffRGD projection |
| Cost upper bounds | one equal-NFE extra-U-Net call; one same-wall-time FFT/backward control; no-gradient and detached-feature controls |

Let `Y(A,B)` denote the paired score for descriptor level `A` and injection
level `B`; `descriptor-only` is `Y(relational,no-injection)` and
`scheduler-injection-only` is `Y(no-descriptor,scheduler-coordinate)`. The
primary statistic is the interaction contrast

```text
I = (FRLA - descriptor-only) - (scheduler-injection-only - no-op)
```

with prompt and seed treated as crossed factors. A positive FRLA-minus-control
mean is insufficient if `I` is zero or negative. Record actual U-Net calls,
feature-hook/reduction time, backward time, FFTs, reward calls, wall time, peak
memory, update ratio, descriptor error, channel moments, DCT/SEC, clipping,
saturation, off-target drift, and structural/detail witnesses. Report the
extra-call arm as an upper bound, never as a fair one-call method.

## One Conditional Alternative: FSTP

FRLA is high-risk because its descriptor is directly covered. Register **one**
alternative now, before any S8 development or validation score is read:
**Feature-Scheduler Tangent Projection (FSTP)**. It may run only on a fresh
prompt-disjoint split if the predeclared FRLA development gate fails or its
interaction is indistinguishable from the descriptor/injection controls. It is
not a post-hoc rescue after held-out validation. FSTP is a fixed hypothesis,
not a claim of projection or FreeU novelty. Running it would require a separate
registration amendment; this audit does not override the current S8 stop rule
or authorize a GPU job.

At one ordinary U-Net call, capture detached backbone and skip features from
the same decoder block. Deterministically reduce and resize them to latent
shape, normalize each sample, and form the FreeU-style feature difference

```text
f_t = normalize(reduce(backbone_t)) - normalize(reduce(skip_t))
u_t = normalize(scheduler_update_t)
r_t = f_t - dot(f_t, u_t) * u_t
guided_x0 = MomentRetraction(pred_original_sample + eta * r_t)
x_prev = prev_sample + (guided_x0 - pred_original_sample)
```

The dot product is a per-sample flattened Frobenius product. Normalize `r_t`
and scale it by the frozen scheduler-update norm before applying `eta`, so the
trust cap is meaningful across timesteps.

Apply the registered scheduler-update trust cap and channel-moment retraction.
The orthogonal projection removes the part already supplied by the native
scheduler; it uses no relational descriptor, gradient, learned parameter, or
extra U-Net call. Freeze `eta`, block, reduction seed, normalization, and cap.
FSTP does not claim a new FreeU, projection, tangent, or scheduler injection;
its only testable question is whether a model-native feature direction adds
guarded information not already in the scheduler direction.

FSTP controls are no-op, FRLA (only if FRLA passes), parallel projection
`dot(f_t,u_t)u_t`, norm-matched FreeU amplitude, shuffled features, L2-matched
random direction, Euclidean clean-latent injection, and the equal-NFE extra-call
upper bound. Use the same seeds, timesteps, CFG, VAE, scheduler, and execution
order. FSTP has no backward pass, so report its lower cost rather than hiding
the difference; the same-wall-time FFT/no-op control prevents a speed-only
claim.

## Falsification and Stop Rules

S8 remains conditional on the registered S7 fixed-action gate. On a new,
prompt-disjoint validation split, FRLA or FSTP must satisfy all of the
following before any search, distillation, or RL:

1. TOPIQ-NR delta `>= +0.005` against no-op, with a crossed prompt/seed 95%
   interval above zero and Holm-adjusted prompt sign-flip `p < 0.05`.
2. Non-inferiority on HPSv2 and CLIP, plus a positive structural/detail witness
   (OCR/count/layout/patch or blinded human preference) without a color,
   contrast, saturation, or sharpness shortcut.
3. Mean clipped-fraction delta `<= +0.001`, saturation delta `<= +0.005`,
   finite outputs, channel-moment drift `<= 1%`, and every update below the
   frozen trust cap; report CIB-Med-inspired per-step and endpoint off-target
   drift on predeclared nuisance axes.
4. FRLA must show a positive interaction `I` and beat shuffled/random and
   pointwise controls. FSTP must beat no-op, FreeU, shuffled/random controls,
   and any eligible FRLA action at matched NFE; a compute-only win is not a
   quality claim.

Failure of the fixed gate closes the representation route. Do not tune lags,
Gram rank, feature block, projection seed, `eta`, reward, or controller to
rescue it. If both FRLA and the single FSTP alternative fail, stop S8 and do
not train an RL controller. If FRLA passes quality but lacks interaction,
write it as a reproducible assembled baseline; do not use prose to upgrade it
to a novel relational-renderer mechanism.
