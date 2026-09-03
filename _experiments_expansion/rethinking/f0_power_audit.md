# F0 Statistical-Power Audit

## Why this audit was necessary

F0 is allowed to start only when its sample size can test the frozen minimum
effects. The first power artifact accepted caller-supplied `planned_power`
numbers without enough information to recompute them. That is not evidence: a
run could simply claim `0.90`. This audit therefore checks the hardest discrete
gate before any formal F0 image is generated.

## Frozen question

The protocol currently treats the prompt as the statistical unit, uses 64
training prompts and 32 validation prompts, and requires terminal `+` versus
`-` direction accuracy of at least `0.55` with a 95% lower bound above `0.50`.
Each prompt contains two generation seeds and three decision indices, or six
binary direction observations.

## Observation

Even under the optimistic assumption that all six observations within a prompt
are independent, an exact binomial test has weak power at the minimum effect:

| Phase | Independent observations | Power at alpha 0.05 | Power at lower-tail alpha 0.025 |
|---|---:|---:|---:|
| Train | 384 | 0.610 | 0.488 |
| Validation | 192 | 0.392 | 0.239 |

The `0.025` column matches the lower side of a two-sided 95% interval more
closely. Treating the prompt itself as one correlated binary observation makes
power still lower: about `0.140` for 64 prompts and `0.082` for 32 prompts at
alpha `0.05`. At least 620 fully independent binary observations are needed
for 80% power at alpha `0.05`, and 786 are needed at alpha `0.025`. Positive
within-prompt correlation increases those requirements.

The archived scheduler-native pilot provides useful scale information for the
continuous TOPIQ witness, but it does not repair the direction test. Its score
file has SHA-256
`f6ab5f08c4859774e21ebe551b0af382e13b7f6d4fabf42c74af27fdf56c7acb`
and 13,965,822 bytes. Across 33 prompts, prompt-averaged paired TOPIQ standard
deviations for six nonzero structural controls versus strict no-op range from
`0.0041` to `0.0087`. This can inform a separately bound pilot-based power
model, but it cannot be reused as if it measured F0 direction accuracy or
student realization variance.

## Conclusion

The old F0 power gate is **not valid**, so a 64+32-prompt F0 cannot be treated
as a powered or formal result. Passing unit tests does not change that
conclusion.

Before launch, the power artifact must bind the null, alternative, alpha,
test direction, clustering assumption or prompt-level variance, sample count,
critical value, calculation method, required power, and any pilot file hash.
The protocol must then either increase the prompt count, require a larger
direction effect, or explicitly restrict F0 to engineering selection. The
choice must be frozen before inspecting any F0 output.

## Resolution for the first run

The first 64+32 run is now restricted to a pre-registered engineering
teacher-construction screen. It may reject a basis or decide whether the cost
of complete benchmark evaluation is justified, but it cannot support a paper
claim, a significance claim, or an OPD/DPO/RL comparison. Its bootstrap
intervals are reported only as conservative stability filters.

If the screen produces a frozen `T_OPSD`, that checkpoint must be evaluated on
all 3,200 HPSv2 prompts and all 2,212 official GenEval images. Only those
prompt-disjoint complete benchmarks can provide the first formal efficacy
evidence. The code must reject the obsolete
`repldm.renderer_f0_power.v1` schema and accept only an immutable registration
that explicitly sets `inferential_claim_allowed=false` and binds the complete
benchmark obligation.

## Next experiment

Implement the fail-closed engineering-screen registration and complete the
F0 runner. If `T_OPSD` survives, run the complete HPSv2 and GenEval suites. A
later mechanism study may use an entirely prompt-disjoint calibration sample
to estimate prompt-level variance and within-prompt correlation, then compute
a powered sample size before making an inferential direction-accuracy claim.
