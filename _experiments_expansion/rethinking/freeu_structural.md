# FreeU Structural Intervention

## Motivation

The earlier TFSA, semantic-transport, and fixed latent-renderer searches did not
show held-out headroom, so an RL controller was not justified. This experiment
tested a different update source suggested by the conference paper's broader
principle: modify a frozen diffusion model through its native structure. FreeU
([Si et al., 2023](https://arxiv.org/abs/2309.11497)) reweights U-Net backbone and
skip features and is an established baseline, not a novelty claim.

Two latent-representation papers motivated the safeguards. *Improving the
Diffusability of Autoencoders* ([2502.14831](https://arxiv.org/abs/2502.14831))
argues that excessive latent high frequencies disrupt coarse-to-fine denoising;
*EQ-VAE* ([2502.09509](https://arxiv.org/abs/2502.09509)) reduces latent
complexity with transformation-equivariance regularization. These works train
the autoencoder; they do not justify blindly filtering SDXL latents at inference.

## What Changed

`AttentionGuidance/freeu.py` adds validated, re-entrant piecewise-linear FreeU
schedules. The SDXL pipeline applies them only during Stage 1 and resets mutable
U-Net state between paired actions. The evaluation harness records the exact
schedule and supports constant or knot-based actions. A follow-up controller
applies the same feature transform while matching per-channel feature mean and
RMS, isolating structural redistribution from amplitude/contrast changes. A
naive alternative that projected every scheduler latent back to the previous
step's moments was rejected in a two-prompt smoke: all projected images became
black or invalid, so it was never scored.

## Experiments

`outputs/freeu_conservative_search_v1` contains 8 actions, 24 prompts, and 2
seeds (384/384 records). `outputs/freeu_moment_followup_v1` repeats the paired
ordinary and moment-preserving controls with the same size and was generated at
commit `29dcf0f`. Both runs used 30 steps, 1024px Stage 1, and strict scoring.
The primary endpoint was TOPIQ-NR; HPSv2 and CLIP were preference/alignment
witnesses, while clipping, saturation, contrast, colorfulness, and sharpness
were guards or diagnostics. Inference used crossed prompt/seed bootstrap CIs,
prompt sign-flip tests, and Holm correction.

| action | Δ TOPIQ-NR [95% CI] | Δ HPSv2 | Δ clipped | Δ saturation |
|---|---:|---:|---:|---:|
| ordinary backbone-only FreeU | +0.008447 [+0.003323,+0.013826] | +0.005473 | +0.006656 | +0.018090 |
| ordinary early/low window | +0.004131 [+0.001053,+0.007595] | +0.001488 | +0.001583 | +0.005257 |
| moment-preserving backbone-only | +0.000609 [-0.001100,+0.002540] | +0.000506 | -0.000074 | -0.000469 |
| moment-preserving low window | +0.001445 [-0.000894,+0.003758] | -0.000346 | -0.000042 | -0.000032 |

The ordinary quality signal is inseparable from large contrast, colorfulness,
clipping, and saturation increases. Once feature moments are constrained, those
side effects disappear, but so does the quality signal. No action reached the
registered `+0.005` TOPIQ gate while satisfying the `+0.001` clipping and `+0.005`
saturation limits. The moment-preserving variants also did not improve CLIP or
ImageReward consistently.

## Decision and Next Step

This is a negative development result, not a validation or test result. FreeU
is retained as a strong structural baseline and as evidence of metric shortcut
risk, but this action family is closed: no more FreeU scale/window tuning,
distillation, or RL is authorized. The next hypothesis must obtain its update
from a different, scheduler-consistent source (for example an equivariance or
low-frequency consistency residual) and must first beat `no_freeu` with guards
on a newly registered prompt-disjoint split. If that fixed action also fails,
the journal extension should report the negative result rather than add an RL
controller.

Reproduction commands and scorer definitions are in `eval-pipeline/README.md`.
Generated images and scorer outputs remain under ignored `outputs/`.
