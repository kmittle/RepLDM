# Reviewer-First Method and Experiment Protocol

## Research Claim Under Test

The journal extension may claim that RepLDM's post-step TFSA residual benefits from content- and state-dependent **spectral allocation**. It may not claim novelty for merely learning a scalar schedule: prompt-conditioned or timestep-varying sampling policies already exist.

For residual `d_t = TFSA(z_t) - z_t`, the candidate update is

```text
u_t = sum_b a_(t,b) P_b(d_t),  b in {low, mid, high}
z_t <- z_t + u_t
||u_t||_2 <= rho_t ||scheduler_update_t||_2
```

`P_b` are smooth masks forming a partition of unity. The norm cap is a safety constraint, not a claimed optimization contribution.

## Journal Narrative and Evidence Boundary

The conference-to-journal relationship may be argued at the level of the
problem and principle: both reprogram a frozen diffusion model through its
native latent trajectory. The journal paper need not preserve the conference
attention-guidance mechanism if a smaller latent renderer realizes that
principle more effectively. Writing may clarify this intellectual continuity,
but it may not convert a new, weakly related module into a technical extension
or hide a failed prerequisite. The manuscript must label inherited ideas, new
mechanisms, and new evidence separately, and every causal or superiority claim
must map to a preregistered ablation, held-out comparison, or transfer test.

## Novelty Threat Matrix

| Closest line | What it already covers | Consequence for this work |
|---|---|---|
| LeSAMP, arXiv:2607.23488 | RL-trained, prompt-conditioned, timestep-varying sampling parameters | A learned scalar schedule is not novel; spectral TFSA allocation must carry the claim |
| AdaGen, arXiv:2409.00342 / 2603.06993 | State-adaptive RL policies for iterative generation schedules | “Adaptive policy” alone is fully overlapped |
| Dynamic CFG, arXiv:2509.16131 | Online feedback and per-sample/per-step CFG selection | Must compare against dynamic scalar guidance and explain the different correction operator |
| DRM, arXiv:2605.25661 | Latent step-wise reward and step-wise GRPO/sampling | Dense credit assignment is not independently novel |
| SGPO, arXiv:2608.06768 | Stage-specific objectives for diffusion RL | Stage-aware optimization is a baseline, not a contribution |
| BranchGRPO, arXiv:2509.06040 | Shared-prefix branches and depth-wise advantages | Generic branching/counterfactual rollout language is overlapped; any estimator claim needs a precise mathematical difference |

## 2026 Literature Audit (正文核对)

The following arXiv versions were checked against their full text on 2026-08-24.
They are rejection-risk controls, not claims that these papers are peer-reviewed
or that their reported gains transfer to SDXL.

| Work | Full-text fact relevant here | Protocol consequence |
|---|---|---|
| Unified path-space view (2608.14430v1) | Reverse-trajectory policy gradients and forward matching are derived from one path-space objective; value-gradient and scale-bounded weights are explicit design axes. | Do not claim a new RL estimator. Report action bounds, variance, and a matched path-space/search baseline. |
| SGPO (2608.06768v1) | SNR and semantic change divide denoising into chaotic, structure-stable, and convergent stages with different objectives. | Stage partitioning is a baseline/analysis variable; any renderer must beat a frozen stage-wise controller. |
| JAGG (2607.17572v3) | Endpoint Jacobians are interpolated to approximate intermediate GRPO gradients, with routing by cosine similarity. | A Jacobian shortcut needs an exact-autograd and SPSA comparison; it is not a novelty claim. |
| DRM (2605.25661) | A diffusion-backed reward model supplies noisy-latent, step-wise rewards for Step-GRPO. | Dense credit assignment is covered; use terminal and independent native-resolution witnesses, not only a learned 224px reward. |
| LeSAMP (2607.23488) | A frozen diffusion/flow model is controlled by a prompt-conditioned LLM policy emitting timestep-varying sampling parameters. | “Frozen backbone + adaptive schedule” is directly overlapped; compare equal-NFE policy capacity and wall-clock cost. |
| CRD (2603.14128v1) | Within-prompt centered reward distillation and KL anchoring explicitly control distribution drift and reward hacking. | Search-then-distill and an anchored reference are mandatory before online RL. |
| GeMPO (2603.10250) | Generalized measure matching allows flexible, even signed, reward reweighting instead of fixed exponential weights. | A reweighting choice is an ablation, not the contribution; retain negative-sample and exploration controls. |
| MARBLE (2605.06507v1) | Per-reward advantages are harmonized in gradient space rather than collapsed into a hand-weighted sum. | If multiple rewards train the renderer, report per-reward gradients and conflict diagnostics. |
| RTDMD (2605.26108v3) / REST (2608.09226) | Reward-tilted distribution matching and RL-native scored-trajectory distillation combine alignment with a simpler student. | Distillation is the required first learning baseline; RL is allowed only after held-out residual headroom. |
| BranchGRPO (2509.06040v5) | Shared-prefix branching and depth-wise advantages reduce rollout cost and terminal-reward sparsity. | Any later black-box controller must use shared prefixes and an equal-budget branch control. |

These overlaps sharpen the journal contribution boundary: the only admissible
claim is a low-capacity, structure-conditioned residual renderer that acts on a
frozen model's native latent trajectory, with scheduler-consistent constraints.
The conference connection is the shared latent-trajectory principle; neither
“RL”, “adaptive”, nor “stage-wise” is sufficient novelty. The LR-1 static gate,
search-then-distill control, held-out metrics, and human evaluation therefore
remain hard prerequisites rather than optional ablations.

This matrix is a rejection-risk screen, not a related-work conclusion. It must be refreshed before submission and supported by full-paper comparison rather than abstract-only wording.

## Baseline Ladder

1. No Attention Guidance.
2. Conference expert schedule.
3. Best constant scalar under equal search budget.
4. Best 2-3 stage scalar schedule.
5. Best 2-3 stage spectral schedule.
6. Prompt-conditioned search-then-distill controller.
7. RL controller, only if it beats 1-6 on held-out prompts and amortized cost.

The 50-step free schedule is admissible only if it beats the 2-3 stage schedule beyond the paired seed CI. RL is admissible only if search-distill leaves measurable headroom.

## Sequential Kill Gates

| Gate | Required evidence | Kill decision |
|---|---|---|
| G0 Correctness | legacy parity, reentrancy, gradient and trust-cap tests | Any failure blocks experiments |
| G1 Spectral headroom | held-out spectral action beats no-AG, expert and best scalar without guard degradation | Otherwise abandon spectral/RL path |
| G2 Granularity | free schedule beats 2-3 stages beyond CI | Otherwise use a segment controller |
| G3 Adaptivity | prompt-conditioned oracle gap survives seed CV on at least one quality and one preference metric | Otherwise use one global schedule |
| G4 RL necessity | RL beats search-distill and equal-budget black-box search | Otherwise report the simpler method |
| G5 Transfer | frozen winner survives 1024² to high-resolution Stage-2, ControlNet and a second backbone | Otherwise limit the paper claim explicitly |

## Statistical Protocol

Use prompt-disjoint search, validation and test sets. Freeze actions, cutoffs, reward weights and stopping rules before test generation. Every `(prompt, seed)` action block must run on one GPU; deterministically shuffle action order within blocks and record device metadata. Treat prompt and seed as crossed factors, report paired mean differences with crossed-bootstrap 95% CIs, prompt-level sign-flip p-values and Holm correction. Report all failures and all registered actions.

TOPIQ-NR is the pilot primary endpoint, but it cannot establish semantics. Confirm with HPSv2 and text-image alignment; use ImageReward and patch-IR as diagnostics. Pixel statistics are hacking guards, never optimization targets. The final paper additionally requires blinded human pairwise evaluation with randomized left/right order and prompt-level confidence intervals.

## RL Entry Condition

If G1-G3 pass, start with shared-state antithetic actions around the static spectral winner and estimate local terminal-reward differences. Compare against identical rollout branches without policy learning. Do not call this estimator novel unless the BranchGRPO/DRM/SGPO distinction survives full related-work review. Stop training when any held-out preference/alignment metric or clipping guard crosses its predeclared non-inferiority bound.

## High-Resolution Target-Domain Exception

S0-S3 use the 1024² Stage-1 image, but the repository's main product claim concerns the final image after high-resolution resampling. One diagnostic 2048² audit is therefore allowed even though G1-G3 failed. This audit may show that Stage 2 changes an already frozen action's ranking; it may not introduce new scales, optimize on high-resolution scores, authorize RL, or count reused prompts as confirmation data.

Stage-2 actions must share both the Stage-1 seed and explicitly seeded resampling noise within each prompt/seed block. A duplicate no-guidance action must reproduce the final PNG exactly before formal generation. Report final-image quality, preference, alignment and pixel guards, plus the fact that Attention Guidance remains Stage-1-only. A positive pilot must still be confirmed on unseen prompts with randomized blinded high-resolution comparisons; a negative pilot closes this exception. S4 was negative, so this exception is now closed and cannot be reopened by scale or schedule tuning.

## Post-S4 S5 Amendment

The spectral-TFSA claim and G1-G4 ladder above are historical and failed in S0-S4. S5 may not reuse their proposal source or enter adaptive control. Its sole admissible static hypothesis is scheduler-consistent reciprocal semantic transport, preregistered in `MODEL_ITERATIONS.md` and bounded by `S5_RELATED_WORK.md`.

For S5, G1 is replaced by a stricter static gate: on new prompts, a candidate must improve TOPIQ-NR by at least `+0.005` with a positive 95% crossed-bootstrap interval and within-metric Holm significance; remain non-inferior on HPSv2 and CLIP; satisfy clipping/saturation guards; beat no-AG, conference TFSA, raw and clean latent controls, the permuted semantic control, CFG 5.0, PLADIS, and GAG; and show a structural/detail gain in a fixed montage. Failure closes the Attention Guidance extension. Success permits only prompt-disjoint confirmation and high-resolution transfer; RL still requires a later, separately registered headroom argument.

The S5 development gate was subsequently run at `cb8eddd` on 12 prompt x 3
seed x 14 action blocks (504 paired records). The best reciprocal-semantic
TOPIQ delta was only `+0.000509` with a 95% interval crossing zero, and no
action passed the registered threshold or fixed-montage structural review.
This is a null result: the Attention Guidance extension is closed, and the
latent-renderer/RL design must be registered as a new hypothesis rather than
trained on S5 outcomes.
