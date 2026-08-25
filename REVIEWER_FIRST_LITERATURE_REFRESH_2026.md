# Reviewer-First Literature Refresh (2026)

**Cutoff:** 2026-08-25. This is a literature and protocol audit only; no
training, generation, scoring, or S7 queue change is authorized by this note.

## Decision

The paper should stop presenting any individual ingredient as a first. Recent
work already covers latent rendering, frozen feature intervention, spectral
correction, relational alignment, geometric guidance, ODE/SDE correction,
adaptive sampling, dense rewards, and trajectory correction. The defensible
hypothesis is narrower: a fixed rank-compatible descriptor paired with one
denoiser-call scheduler-coordinate clean-latent injection and a zero-init trust
cap. This is a combination claim, valid only if its preregistered interaction
survives the controls below.

## Two Representation Papers: Correct Boundary

[Improving the Diffusability of Autoencoders (2502.14831)](https://arxiv.org/abs/2502.14831)
finds excess high-frequency latent energy and uses scale-equivariant decoder
regularization. The experiments fine-tune pretrained AEs and also train AE
families from scratch; downstream diffusion models are trained/evaluated on
the changed latent distribution. It does not establish a zero-shot DCT/FFT
residual on a frozen SDXL VAE and U-Net.

[EQ-VAE (2502.09509)](https://arxiv.org/abs/2502.09509) studies spatial
transformation **equivariance**, not invariance. It fine-tunes SD-VAE/SDXL-VAE
and then trains or evaluates a generator in that new latent space. Its
reconstruction, moment, and equivariance results are useful upper bounds, but
they are not evidence for frozen inference-time latent intervention.

## Claims To Abandon

Drop "first spectral/coarse-to-fine latent correction," "first equivariant or
relational latent guidance," "first frozen internal intervention," "first
Riemannian/geodesic or SDE-consistent sampler," "first latent renderer," and
"first dense-reward/adaptive diffusion-RL controller." A gain over deterministic
Euler alone is a sampler result; a gain from a spectral proxy alone is a
representation result. Neither supports a renderer claim.

## Required S7/FRLA Controls

At matched actual U-Net evaluations and wall-clock accounting, compare native
Euler-Ancestral, deterministic Euler, registered interpolation, DPM-Solver++,
UniPC, and an available CPS/Precise/LC-GRPO-style ODE-plus-noise control. Add
FreeU, SPA, fixed DCT and wavelet projections, shuffled features, L2-matched
random directions, MOG/DiffRGD geometry, pointwise versus relational
descriptors, and an equal-NFE extra-U-Net upper bound. Freeze priors,
detectors, feature blocks, seeds, and action caps before held-out scoring.

The fixed action must clear TOPIQ-NR delta `>= +0.005`, a crossed prompt/seed
confidence interval above zero, Holm-adjusted sign-flip significance,
HPSv2/CLIP non-inferiority, structural/detail witnesses, and moment, trust,
clipping, saturation, diversity, and off-target-drift guards. Test the
descriptor-by-injection interaction, not just the endpoint mean. If it fails,
do not introduce distillation or RL.

## Static-Operator Recommendation

Do not register another static operator now. The existing conditional FSTP
fallback is sufficiently distinct from FRLA (feature direction orthogonalized
against the scheduler update, no descriptor gradient, no extra U-Net call) and
is already documented in `FRLA_S8_COMBINATION_AUDIT_2026.md`. Run it only on a
fresh split after a predeclared FRLA failure. If FRLA and FSTP both fail, close
the route; if either only beats deterministic Euler or a spectral control,
report that simpler explanation rather than expanding the novelty claim.
