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

## Novelty Threat Matrix

| Closest line | What it already covers | Consequence for this work |
|---|---|---|
| LeSAMP, arXiv:2607.23488 | RL-trained, prompt-conditioned, timestep-varying sampling parameters | A learned scalar schedule is not novel; spectral TFSA allocation must carry the claim |
| AdaGen, arXiv:2409.00342 / 2603.06993 | State-adaptive RL policies for iterative generation schedules | “Adaptive policy” alone is fully overlapped |
| Dynamic CFG, arXiv:2509.16131 | Online feedback and per-sample/per-step CFG selection | Must compare against dynamic scalar guidance and explain the different correction operator |
| DRM, arXiv:2605.25661 | Latent step-wise reward and step-wise GRPO/sampling | Dense credit assignment is not independently novel |
| SGPO, arXiv:2608.06768 | Stage-specific objectives for diffusion RL | Stage-aware optimization is a baseline, not a contribution |
| BranchGRPO, arXiv:2509.06040 | Shared-prefix branches and depth-wise advantages | Generic branching/counterfactual rollout language is overlapped; any estimator claim needs a precise mathematical difference |

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
