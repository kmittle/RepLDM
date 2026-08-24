# FRLA Relational Latent Operator (Proposal Only)

## Status and Motivation

This is a preregistered follow-up hypothesis, not an experiment result. It may
run only after the S7 scheduler-correction gate is resolved. Earlier TFSA,
fixed-renderer, and FreeU searches did not provide guarded headroom, so this
proposal changes the source of the update rather than adding another scale
search or an RL controller.

The idea is **Frozen Relational Latent Alignment (FRLA)**. iREPA
([2512.10794](https://arxiv.org/abs/2512.10794)), SARA
([2503.08253](https://arxiv.org/abs/2503.08253)), sREPA
([2605.16949](https://arxiv.org/abs/2605.16949)), and *Diffusing in the Right
Space* ([2606.03578](https://arxiv.org/abs/2606.03578)) suggest that spatial
self-similarity and relational structure can be more stable signals than a
single frequency statistic. This is a soft connection to RepLDM: both methods
reprogram a frozen model's native latent trajectory, but FRLA is not TFSA.
LaRender ([2508.07647](https://arxiv.org/abs/2508.07647)) and SATeCo
([2403.17000](https://arxiv.org/abs/2403.17000)) are additional novelty
boundaries: they use object-conditioned rendering or feature adapters, whereas
FRLA has no object mask, prompt branch, or second denoiser.
DUNE ([2607.09753](https://arxiv.org/abs/2607.09753)), DIAMOND
([2602.00883](https://arxiv.org/abs/2602.00883)), InfSplign
([2512.17851](https://arxiv.org/abs/2512.17851)), and PMG/LGDM
([2506.00327](https://arxiv.org/abs/2506.00327)) already cover frozen-U-Net
feature interventions or clean-latent gradient corrections. SSVAE
([2512.05394](https://arxiv.org/abs/2512.05394)) also studies local correlation
regularization. These are direct baselines and novelty limits: FRLA cannot claim
the gradient, tangent projection, local relation loss, or frozen manipulation
itself as new. Covariance is a deterministic metric only; covariance-aware
noise is excluded because it has been reported to hurt VAE latent diffusion.

## Fixed Operator

During the ordinary conditional U-Net forward, capture one decoder feature map
and detach it. Deterministically reduce it to four channels and resize both it
and `pred_original_sample` to a `16 x 16` grid. For five fixed lags
`(1,0),(0,1),(1,1),(2,0),(0,2)`, compute local cosine autocorrelations and take
one latent-only gradient step that reduces their squared discrepancy. The
channel reduction, lag boundary handling, step size `eta`, and trust-cap ratio
must be fixed in the YAML before scoring (use a recorded fixed projection seed,
not a learned reduction). Because the pipeline is globally `no_grad`, use
`torch.enable_grad()` only around a cloned, detached clean latent and the
relation loss; never backpropagate through the U-Net. Inject the bounded
clean-latent residual after the scheduler step, followed by a scheduler-update
trust cap and a shrinkage channel-covariance retraction. This is a covariance
control, not a claim that DiffRGD's isotropic sphere is anisotropic. The full
token Gram version is an ablation only: the four-channel latent and a wider
feature map otherwise have a rank mismatch. Record backward time, FLOPs, peak
memory, residual norms, descriptor error, and latent-prior drift.

The gradient step is an implementation control, not a new optimizer claim:
DiffRGD/MPGD/DPS already cover test-time latent-gradient actions. FRLA asks
only whether this U-Net-native relational direction has fixed quality headroom.
If it does, a later contribution would be a bounded small renderer that
predicts or distills the direction; that later model is not authorized here.

## Falsification Gate

On a new prompt-disjoint split with identical seeds, CFG, scheduler, NFE, and
initial noise, compare no-op, conference TFSA, SPA, FreeU, FRLA, shuffled
feature tokens, a feature-gradient/no-gradient ablation, pointwise feature
matching, relation-preserving tangent, isotropic shell, Mahalanobis shell, a
matched dummy latent-gradient control, a same-NFE extra-U-Net upper bound, and a
same-wall-time no-op/FFT control.
Require TOPIQ-NR delta `>= +0.005`, crossed-bootstrap CI above zero, Holm-adjusted
prompt sign-flip `p < 0.05`, HPSv2/CLIP non-inferiority, and clipping/saturation
guards (`+0.001`/`+0.005`). Also report OCR/count/spatial accuracy,
LPIPS/diversity, latent moments, DCT/SEC, runtime, and qualitative crops.

If FRLA fails this fixed gate, close the representation route. No parameter
search, distillation, or RL is allowed. If it passes, freeze one action on
development data before any validation or learned controller; DiffRGD
([2606.28417](https://arxiv.org/abs/2606.28417)) and LatSearch
([2603.14526](https://arxiv.org/abs/2603.14526)) remain computational and
novelty controls, not claims of FRLA novelty.
