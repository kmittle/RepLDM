# S7 Related-Work and Rejection-Risk Audit

This note was frozen before interpreting the S7 development scores. The
implementation must be described as a fixed SDXL sampler ablation, not as a
new scheduler or RL estimator.

## Direct Overlap

- LC-GRPO ([2608.05600](https://arxiv.org/abs/2608.05600)) uses an
  inference-aligned predictor/corrector pattern with Langevin-style noise.
- DC-Solver ([2409.03755](https://arxiv.org/abs/2409.03755)) introduces dynamic
  compensation for diffusion solver error without extra denoiser evaluations.
- Noise Level Correction ([2412.05488](https://arxiv.org/abs/2412.05488)) and
  Precise ([2605.23522](https://arxiv.org/abs/2605.23522)) analyze the bias and
  excess noise caused by heuristic discretizations.
- MixGRPO ([2507.21802](https://arxiv.org/abs/2507.21802)) and CPS
  ([2509.05952](https://arxiv.org/abs/2509.05952)) explicitly study ODE/SDE
  mixing and coefficient-preserving stochastic transitions. Flow-GRPO,
  DanceGRPO, and DiffusionNFT cover related stochastic-rollout training.

These works make the native `EulerAncestralDiscreteScheduler` a mandatory
control. The intermediate `mix` values in this repository have no independent
SDE derivation; they are bounded interpolation controls. A gain over Euler
alone is therefore insufficient and must be reported as a sampler-baseline
effect unless it also survives the native ancestral, DPM-Solver++, and UniPC
same-NFE references.

## Required Controls

The preregistered development run includes native Euler-Ancestral as a
non-selectable reference. Before any publication claim, use the same prompts,
seeds, CFG, NFE, and initial-noise policy for drift-only, noise-only,
energy-matched, early-SDE/late-ODE, DPM-Solver++, and UniPC controls. Report
latency, peak memory, per-step noise diagnostics, and blind human preference;
TOPIQ/HPS/CLIP alone cannot establish structural improvement. If a fixed
correction does not beat the known sampler references on held-out prompts, the
route closes and no renderer or RL controller is justified.
