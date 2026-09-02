# Research Documentation

Repository-level research documents are grouped by purpose. Paths mentioned
inside these documents are repository-relative unless a link states otherwise.

## Research Progress

- [`EXPERIMENT_PLAN.md`](research/EXPERIMENT_PLAN.md): historical and current experiment plans.
- [`EXPERIMENT_RESULTS.md`](research/EXPERIMENT_RESULTS.md): quantitative results and gate decisions.
- [`MODEL_ITERATIONS.md`](research/MODEL_ITERATIONS.md): chronological method iterations.
- [`RL_RESEARCH_DESIGN.md`](research/RL_RESEARCH_DESIGN.md): learned-renderer and RL design constraints.
- [`TODO_2026_08_31.md`](research/TODO_2026_08_31.md): ordered execution queue and completion gates.

## Protocols

- [`REVIEWER_PROTOCOL.md`](protocols/REVIEWER_PROTOCOL.md): reviewer-first evidence rules.
- [`MAINSTREAM_EVALUATION_PROTOCOL.md`](protocols/MAINSTREAM_EVALUATION_PROTOCOL.md):
  fixed prompt/seed sizes and paired reporting for formal comparisons.
- [`LATENT_RENDERER_PROTOCOL.md`](protocols/LATENT_RENDERER_PROTOCOL.md): fixed-renderer search protocol.
- [`SCHEDULER_NATIVE_FIXED_HEADROOM_PROTOCOL.md`](protocols/SCHEDULER_NATIVE_FIXED_HEADROOM_PROTOCOL.md): scheduler-native screen.
- [`ADAPTIVE_ORACLE_PROTOCOL.md`](protocols/ADAPTIVE_ORACLE_PROTOCOL.md): current adaptive-oracle protocol.
- [`OPD_DPO_RL_LATENT_RENDERER_PROTOCOL.md`](protocols/OPD_DPO_RL_LATENT_RENDERER_PROTOCOL.md):
  current shared renderer, data, budget, training, and evaluation contract.

## Literature And Audits

`literature/` contains related-work surveys. `audits/` records baseline
provenance, representation-side limits, and frozen novelty assessments.

- [`BENCHMARK_AND_PREFERENCE_LEARNING_SURVEY_2026_08_31.md`](literature/BENCHMARK_AND_PREFERENCE_LEARNING_SURVEY_2026_08_31.md): full-benchmark sizes, preference-data risks, and the OPD renderer route.
