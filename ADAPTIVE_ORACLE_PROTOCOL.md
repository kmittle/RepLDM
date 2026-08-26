# Adaptive Oracle Protocol

**Status (2026-08-25): `blocked_registration_only`.** This document authorizes
no GPU use, image generation, scoring, renderer training, or RL. It defines the
evidence required before an executable registration can be reviewed. Formal
outputs and scores from other studies are out of scope.

## Claim Boundary and Historical Nulls

S5 found no registered Attention Guidance action with stable structural
headroom. LR-1 was a closed null for its unit-gain post-step nudge, whose later
scheduler audit also showed that it was not a native Euler clean-endpoint map.
The scheduler-native fixed-headroom screen returned `null_route`. These results
forbid tuning their angles, layers, spectral-mid amplitudes, rewards, or policies.
They do not test the family below.

The fresh hypothesis is narrower: non-attention local relations from a frozen
U-Net may define a useful, bounded transport direction in exact scheduler
coordinates. It uses new prompt and seed namespaces, a different feature,
operator, action bank, and causal design. Any paper connection to the conference
version is the principle of model-native latent-trajectory reprogramming, not a
claim that this operator rescues or extends a positive historical action.

## Frozen Candidate Operator

At Euler step `t`, a kwargs-aware forward pre-hook on `up_blocks.0` reads only
its `hidden_states` input from the single ordinary U-Net call. For registered
SDXL CFG it must have shape `[2B,1280,32,32]`, ordered unconditional then
conditional; the second `B` rows define \(h_t^c\). Shape, row order, hook count,
or module-path drift fails closed. Both \(h_t^c\) and \(\hat x_{0,t}\) are
stop-gradient tensors. No attention probabilities, Q/K tensors, backbone
autograd graph, extra U-Net call, or VAE decode is allowed. Fixed adaptive
average pooling gives \(h_t\in\mathbb R^{1280\times16\times16}\). On the 16x16
grid define three D4-complete offset orbits:

\[
\mathcal O_{a1}=\{(\pm1,0),(0,\pm1)\},\quad
\mathcal O_{d1}=\{(\pm1,\pm1)\},\quad
\mathcal O_{a2}=\{(\pm2,0),(0,\pm2)\}.
\]

Only in-bounds neighbours are present; indexing never wraps. Let
`epsilon_h=epsilon_a=1e-6`. A pooled feature-vector norm at or below
`epsilon_h` is undefined and fails the record; it is never silently treated as
cosine zero. For node `i` and valid neighbour `j` in orbit `o`, define

\[
a^o_{ij}=\epsilon_a+\frac{1+\operatorname{clip}(\cos(h_{t,i},h_{t,j}),-1,1)}{2},\qquad
W^o_{ij}=a^o_{ij}/\sum_{k\in N_o(i)}a^o_{ik}.
\]

All other entries are zero. Thus each row sums to one without a temperature or
other quality-tuned affinity parameter. Let \(z_t=D_{16}(\hat x_{0,t})\), using
adaptive average pooling, and let `U16` be bilinear upsampling with
`align_corners=False`. The three relational bases are exactly

\[
b^o_t=U_{16}(W^o_t z_t-z_t).
\]

For each latent channel, remove the spatial mean from `b` and project away its
component along \(q=\hat x_0-\operatorname{mean}(\hat x_0)\). This is the tangent
projection \(v=T_{\hat x_0}(b)\). The candidate endpoint follows the fixed-moment
sphere geodesic

\[
y_c=\mu_c+q_c\cos\theta+\|q_c\|_2
       \frac{s v_c}{\|v_c\|_2}\sin\theta,\quad s\in\{-1,+1\}.
\]

Zero or non-finite tangent norms fail closed. A common nonnegative `theta` is
found deterministically so the mapped update ratio is nominally `0.02`. For
deterministic, no-churn `EulerDiscreteScheduler`, production reconstructs the
native epsilon corresponding to `y` and calls `scheduler.step` exactly once;
equivalently, with current state fixed,

\[
u_t=(1-\sigma_{t+1}/\sigma_t)(y_t-\hat x_{0,t}),\qquad
\|u_t\|_2/(\|\bar x_{t-1}-x_t\|_2+10^{-12})=0.02.
\]

The measured ratio has a hard `0.05` cap after scheduler mapping. A cap hit,
moment drift above 1%, native-round-trip mismatch, or any non-finite value is a
record failure. Other schedulers are unsupported, not silently approximated.

## Actions and the Nondegenerate Factorial

Round one has no stage search or per-step policy. One trajectory-global action
is chosen from the context available at the first ordinary forward, then its
same orbit/sign is applied at all 50 steps (`envelope[t] = 1`); `W`, `x0`, and
the direction are recomputed on that action's own trajectory. Once step 0 is
intervened on, the trajectory differs. Therefore no complete no-op/base cache
may supply later U-Net features or scheduler states; every action reruns the
post-intervention trajectory with common initial noise.

The primary logical bank is `{P0, +/-a1, +/-d1, +/-a2}`. Its seven members are
no-op and the six signed geodesics above. The seven-member random bank is
`{R0, +/-r_a1, +/-r_d1, +/-r_a2}`. `R0` is the same generated record as `P0`,
so the complete matrix has 13 unique executions, not two knowingly identical
no-ops. Each `r_o` uses the same orbit, 16x16 grid, pooling, bilinear
upsampling, tangent projection, and geodesic as `P`, but replaces feature
cosine by a counter-keyed symmetric random edge affinity fixed before outcomes.
It is independent of `h`, prompt text, images, and metrics; its injection path
is forbidden from reading a relational basis or its norm. All `P` and `R`
actions solve the same `0.02` scheduler-coordinate target from only `x0` and
the nominal Euler update, so geodesic displacement is direction-norm matched
analytically. The negative endpoint direction is the registered tangent-
antithetic pair. The edge RNG, field resolution, keys, and matching tolerances
are hashed before generation.

The random affinity is exactly \(a^{r,o}_{ij}=\epsilon_a+U_{ij}\). Its counter
is a JSON object whose `schema` field is `ao-random-edge-counter-v1`, followed
by `experiment_id`, `split_role`, `prompt_row_id`, integer `seed`, integer
`step_index`, `orbit_name`, and sorted integer node ids `edge_low,edge_high`.
All string values are restricted to `[A-Za-z0-9._:-]+`. Serialize with Python
`json.dumps(counter, ensure_ascii=True, sort_keys=True, separators=(",", ":"))`
and encode as UTF-8 with no BOM or trailing newline. One complete byte-string is:

```json
{"edge_high":18,"edge_low":17,"experiment_id":"ao-search-v1","orbit_name":"axis-r1","prompt_row_id":"search-0001","schema":"ao-random-edge-counter-v1","seed":123456789,"split_role":"search","step_index":0}
```

SHA-256 of those bytes supplies `k` from its first three bytes interpreted as
an unsigned big-endian integer, and \(U=k/2^{24}\in[0,1-2^{-24}]\). Both
directions of one undirected edge reuse the same draw; `+/-` actions reuse the
same graph. No process RNG or floating-point hash conversion is permitted.
Repeated `k` or `U` values are valid; uniqueness is required only for counter
tuples and the registered prompt/seed namespaces.

The causal 2x2 crosses **descriptor source** with **injection basis**:

| | Relational injection | Norm-matched random injection |
|---|---|---|
| Structural descriptor | first-step local-relation summaries select `P` | the same summaries select `R` |
| Prompt-only descriptor | pooled text selects `P` | pooled text selects `R` |

These are four active selectors, not a descriptor-only no-op. A fixed
context-free selector, described below, is an additional control. The primary
interaction is
\(I=[S_{rel}-P_{rel}]-[S_{rnd}-P_{rnd}]\); it must be positive with a
simultaneous interval above zero.

## Fresh Data and Outcome Isolation

Separate deterministic namespaces are required for `ao-search-v1`,
`ao-replay-v1`, and `ao-test-v1`, each for both prompts and seeds. A builder
must pin the source revision and bytes, normalize prompts by Unicode NFKC,
collapsed whitespace, and casefolding, and exclude every normalized text and
explicit or text-derived source row in every repository prompt manifest. It
must also scan all historical run configs, manifests, and sidecars for used or
reserved seeds. The manifest records inventories, collision lists (which must
be empty), selection digests, split hashes, and globally unique seeds. Search,
replay, and test are prompt- and seed-disjoint. Engineering prompts/seeds are
retired immediately. Formal splits are stratified for OCR, exact counting, and
spatial relations, with answer annotations frozen and independently verified.

Replay is an outer evaluation, not more search. The selector weights,
hyperparameters, action bank, random keys, and decision rule are frozen before
any replay generation. For each replay block, its choice is written to a
hash-chained decision ledger before image decode or scoring; the chooser cannot
read that block's image, metric, or any replay outcome. The test namespace stays
sealed until a distilled checkpoint and its analysis are frozen.

## Oracle, Selector, and Controls

The search split renders the complete paired 13-action matrix. Feasibility has
three non-interchangeable levels. A **record** is valid only when its image,
hook, scheduler, moment, cap, and call ledgers pass. Within each training fold,
an **action** is eligible only when aggregate HPSv2, CLIP, pixel, and diversity
guards computed from that fold pass. A **selector** is acceptable only when its
OOF or replay outputs pass the full simultaneous inference guards. Collection-
level guards are never used as per-record ridge labels. For valid records, the
frozen utility is the minimum of the two margin-standardized co-primary deltas;
no-op has utility zero. Scales, missing-record behavior, and tie order are
frozen in the power artifact.

The **seed-CV oracle** is a diagnostic upper bound: for every prompt and held
search seed, select its action using only the other search seeds and score it on
the held seed. Its comparator is the best global action selected, including
eligibility and tie-breaking, from all prompts at only those same non-held
seeds. Neither selection may use any record from the held seed. No positive
oracle gap on both co-primary endpoints means stop before fitting any selector.

For each action, a ridge model predicts record utility. Template/source groups
are atomic and assigned to challenge-stratified prompt folds before outcomes.
For held prompt fold `k` and held seed `s`, training is exactly
`{(p,s'): fold(p) != k and s' != s}` and evaluation is exactly
`{(p,s): fold(p) == k}`. Thus every prompt-seed record is evaluated once with
both its prompt group and seed absent from training. Preprocessing and ridge-
penalty selection are nested inside that training set.
For every outer pair `(k,s)`, all OOF comparators use that identical training
set `T_{k,s}`. Their action eligibility, preprocessing, hyperparameters,
coefficients, tie-breaking, and action choice are selected or fitted only in
`T_{k,s}` before evaluation on the held fold. This applies to best global,
intercept-only/context-free ridge, prompt-only ridge, fixed within-challenge
shuffled structural descriptors, and equal-outcome-query-budget random search
with a preregistered counter RNG. OOF must first beat the cross-fitted global
action on both co-primary endpoints and pass its selector-level guards. Only
then may each complete procedure refit on all search data; the chosen
preprocessing, penalty, tie rule, eligible-action rule, and one deterministic
ridge per action are serialized and hash-frozen for replay. Query counts,
failed queries, candidate exposure, and failure accounting must match. Training
has a common preregistered compute ceiling rather than artificial numerical
equality; actual fit time, FLOPs, and peak memory are reported. No control may
query replay outcomes.

## Endpoints, Guards, and Inference

An independent verifier, with separately hashed code and no selector-training
role, owns labels, strict scoring, the result-blind audit, and the one-shot
evaluator. For the full structural-relational selector, co-primary endpoints
are (1) TOPIQ-NR delta versus no-op, requiring at least `+0.005`, and (2) an
equal-category macro of OCR exact/normalized accuracy, exact counting accuracy,
and spatial-relation accuracy. The frozen power artifact also sets nonzero
adaptivity margins versus best global for both endpoints. Their simultaneous
95% lower bounds, and the 2x2 interaction bounds, must exceed the registered
margins rather than merely zero.

Against both no-op and best global, HPSv2 and CLIP lower bounds must be at least
`-0.005`. Pixel guards retain clipped-fraction `+0.001`, saturation `+0.005`,
and contrast-ratio `[0.95,1.05]` margins. Inter-seed LPIPS/DINO diversity ratios
must remain in `[0.95,1.10]`. Every step requires applied ratio at most `0.05`,
absolute channel-mean error at most `1e-4`, relative channel-variance error at
most `1e-3`, and channel-covariance drift at most 1%. The high-frequency energy
ratio may not exceed no-op by 10%; coarse-to-fine ordering and end-to-end D4
excess-drift margins are frozen by the power artifact. Require 50x1 U-Net and
scheduler calls, zero backbone backward/decode, fewer than 1M trainable selector
parameters, median latency at most `1.10x`, P95 latency at most `1.15x`, and peak
VRAM at most `1.10x` no-op. Report every raw endpoint and guard; correlated
preference metrics cannot substitute for the structural macro.

The statistical unit is the prompt after averaging its registered seeds.
Crossed prompt/seed bootstrap intervals use 10,000 fixed-seed resamples; prompt-
level paired sign flips use 100,000 draws (or exact enumeration when smaller).
Max-T gives simultaneous intervals across action/selector contrasts, and Holm
controls the preregistered endpoint families at FWER 0.05. Challenge results are
sensitivity analyses, not independent samples.

Sample counts are not guessed here. A CPU simulator starts from candidate-blind
historical no-op variance, then spans a preregistered conservative grid of
treatment/no-op variance ratios, paired correlations, action heterogeneity,
selector error, and all covariance terms needed for the difference-in-
differences. It chooses the worst-case sample count by directly simulating the
final max-T/Holm rejection rule. If these inputs cannot bound paired or
interaction variance, an independently held, blinded `ao-variance-pilot-v1`
may release only covariance and sample-size artifacts, never rankings; its
prompts/seeds are retired and cannot train or select the method. For TOPIQ
`0.005`, the reliability-calibrated structural effect, and registered
adaptivity margins, search, replay, and test are powered separately for at
least 90% joint efficacy power and 80% guard power. Inputs, sensitivity grid,
simulation code, pilot firewall if used, chosen counts, and hashes form
`power_analysis.json`; infeasible power leaves the study blocked.

## One-Shot Artifacts and Sequential Gates

All writers use locks, atomic creation, and refusal when an output exists.
`adaptive_oracle_prompt_manifest_v1.json`, `power_analysis.json`, a non-
executable registration YAML, implementation/environment manifests, and an
independent review authorization are committed first. The executable binds
their SHA-256 values and the reviewed commit. Every run binds config, prompt,
action, code, environment, model, scheduler, decision-ledger, PNG, sidecar, and
strict-scorer hashes.

1. **CPU gate:** synthetic affine-cosine constants, zero-feature fail-closed,
   orbit/D4/no-wrap/row-sum, both-input-detach, antithetic norm parity,
   tangent/geodesic moments, bitwise no-op, native Euler round trip,
   trajectory-isolation, counter-tuple and namespace collision audit (not
   uniqueness of truncated random values), and power analysis must produce one
   warning-free `cpu_audit.json`.
2. **Engineering gate:** only after independent authorization, run the 11 fresh
   challenge smoke prompts x one retired-on-use seed x 13 unique actions at
   1024, Euler, 50 steps. Scoring is forbidden. Require 143 PNG/sidecar pairs,
   distinct active outputs, complete ledgers, bounds, and a warning-free
   one-shot `engineering_audit.json`.
3. **Formal search gate:** run the powered full matrix, strict offline scoring,
   result-blind `search_run_audit.json`, then one-shot `search_evaluation.json`.
   Incomplete cells, provenance/mechanism warnings, an absent seed-CV oracle
   gap, OOF failure versus global, or a failed OOF interaction close this
   family. An individual action's aggregate guard failure makes that action
   ineligible under the frozen fold rule and is fully reported; it is not
   silently converted into a family-wide failure or omitted after outcomes.
4. **Outer replay gate:** freeze `selector_freeze.json` and its decision-ledger
   schema first; run only the frozen selectors and controls on powered replay
   data. Audit before the independent one-shot evaluator. Both co-primaries,
   the relational interaction, and all guards must pass versus best-global,
   context-free, prompt-only, shuffled, random-bank, and equal-budget controls.
5. **Authorization boundary:** a replay pass may emit only
   `search_then_distill_authorization.json` for supervised distillation of the
   frozen search teacher and one frozen test evaluation. It does not authorize
   online rewards, policy gradients, PPO/GRPO, or any RL. RL requires a new
   registration, untouched data, and demonstrated residual headroom over the
   distilled model.

Any integrity/mechanism failure, or any registered efficacy or guard failure of
the frozen full selector at its formal gate, is terminal for this family.
Control and individual-action failures are reported and handled only by the
predeclared eligibility rule. No lag, orbit, feature hook, ratio, cap, envelope,
metric, margin, seed, or exclusion may be changed after outcomes; a changed
hypothesis starts a new namespace and registration.

## Missing Implementation Checklist

- Exact conditional `up_blocks.0` hook, 16x16 reducer, orbit/transport operator,
  random bank, moment geodesic, and native-Euler integration with diagnostics.
- Trajectory-isolated 13-action generation and hash-chained pre-outcome choices.
- Fresh prompt/seed builder, historical collision scanner, and power simulator.
- OOF ridge/oracle/equal-budget controls and the genuine 2x2 interaction.
- Provenance-locked OCR, counting, spatial, spectral, equivariance, diversity,
  compute scorers, independent auditor/evaluator, freezes, and authorizer.
- Clean reviewed commit, environment lock, source hashes, and independent
  authorization. Until all exist and the CPU gate passes, status remains
  `blocked_registration_only`: no GPU and no RL.
