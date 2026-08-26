# Adaptive Oracle Protocol

**Status (2026-08-26): `blocked_registration_only`.** This document authorizes
no GPU use, image generation, scoring, renderer training, or RL. It defines the
evidence required before an executable authorization can be reviewed. Formal
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

DUNE is the closest training-free intervention at the same SDXL `h`-space. The
current engineering smoke does not compare efficacy, but any later formal
search must add a preregistered, same-hook and compute-matched temporal-anomaly
control. Otherwise a positive result cannot distinguish local relations from a
generic benefit of detecting abrupt feature changes over time.

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
`epsilon_h=1e-6`, `epsilon_z=1e-6`, and `epsilon_a=1e-6` be independently
registered constants. A pooled feature-vector norm at or below
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

Remove the spatial mean independently from every channel of `b` and
\(\hat x_0\), then flatten the spatial dimensions to obtain matrices
\(B,Q\in\mathbb R^{4\times HW}\). Let \(G_Q=QQ^T\) and project the basis away
from the complete channel row space,

\[
V=B-BQ^T G_Q^{-1}Q.
\]

Both \(G_Q\) and \(G_V=VV^T\) must be positive definite. With lower Cholesky
factors \(G_Q=L_Q L_Q^T\) and \(G_V=L_V L_V^T\), define
\(D=L_Q L_V^{-1}V\). Then \(DQ^T=0\) and \(DD^T=QQ^T\), so the full-Gram
fixed-moment geodesic is

\[
Y=\mu+Q\cos\theta+sD\sin\theta,\quad s\in\{-1,+1\}.
\]

This preserves every channel mean and the complete channel covariance matrix,
not only its diagonal variances, while retaining
\(\|Y-Q\|_F=2\|Q\|_F\sin(\theta/2)\). Singular or non-finite channel/tangent
Gram matrices fail closed. A common nonnegative `theta` is
found deterministically so the mapped update ratio is nominally `0.02`. For
deterministic, no-churn `EulerDiscreteScheduler`, production reconstructs the
native epsilon corresponding to `y` and calls `scheduler.step` exactly once;
equivalently, with current state fixed,

\[
u_t=(1-\sigma_{t+1}/\sigma_t)(y_t-\hat x_{0,t}),\qquad
\|u_t\|_2/(\|\bar x_{t-1}-x_t\|_2+10^{-12})=0.02.
\]

After the one scheduler call, the implementation analytically reconstructs the
native no-op `prev_sample`, including Euler's output-dtype cast, and measures
the actual intervention as `guided_prev_sample - native_prev_sample`. This
post-mapping ratio must remain within `5e-4` of `0.02` and has a hard `0.05`
cap. A cap hit,
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
so the primary matrix has 13 unique executions, not two knowingly identical
no-ops. Each `r_o` uses the same orbit, 16x16 grid, pooling, bilinear
upsampling, tangent projection, and geodesic as `P`, but replaces feature
cosine by a counter-keyed symmetric random edge affinity fixed before outcomes.
It is independent of `h`, prompt text, images, and metrics; its injection path
is forbidden from reading a relational basis or its norm. All `P` and `R`
actions solve the same `0.02` scheduler-coordinate target from only `x0` and
the nominal Euler update, so geodesic displacement is direction-norm matched
analytically. The negative endpoint direction is the registered tangent-
antithetic pair. The edge RNG, field resolution, keys, and matching tolerances
are hashed before generation. Because every signed pair starts from the same
prompt, seed, and initial latent, its `+/-` trajectories must report an identical
step-0 basis hash for each `P/R` orbit. Later basis hashes may diverge only after
the signed interventions separate their trajectories.

The random affinity is exactly \(a^{r,o}_{ij}=\epsilon_a+U_{ij}\). Its counter
is a JSON object whose `schema` field is `ao-random-edge-counter-v1`, followed
by `experiment_id`, `split_role`, `prompt_row_id`, integer `seed`, integer
`step_index`, `orbit_name`, and sorted integer node ids `edge_low,edge_high`.
All string values are restricted to `[A-Za-z0-9._:-]+`. To preserve the same
D4 symmetry as the structural operator, an actual undirected edge first maps to
its canonical D4 representative: apply all eight rotations/reflections of the
16x16 node grid, sort the two node ids within each image, and take the
lexicographically smallest pair. The counter's `edge_low,edge_high` are this
canonical pair. D4-related actual edges intentionally reuse one counter and
weight; actual undirected edges themselves must remain unique within an orbit.
Serialize with Python
`json.dumps(counter, ensure_ascii=True, sort_keys=True, separators=(",", ":"))`
and encode as UTF-8 with no BOM or trailing newline. One complete byte-string is:

```json
{"edge_high":18,"edge_low":17,"experiment_id":"ao-search-v1","orbit_name":"axis-r1","prompt_row_id":"search-0001","schema":"ao-random-edge-counter-v1","seed":123456789,"split_role":"search","step_index":0}
```

SHA-256 of those bytes supplies `k` from its first three bytes interpreted as
an unsigned big-endian integer, and \(U=k/2^{24}\in[0,1-2^{-24}]\). Both
directions of one undirected edge, all D4 images of that edge, and `+/-` actions
reuse the same draw. No process RNG or floating-point hash conversion is
permitted. Repeated `k` or `U` values for distinct canonical counters are valid;
uniqueness is required for actual edge tuples and registered prompt/seed
namespaces, while canonical-counter reuse must be explained exactly by D4.
For each `(prompt, seed, step, orbit)`, sort the unique canonical pairs, prefix
each exact counter byte-string by its unsigned four-byte big-endian length, and
SHA-256 the concatenation. The resulting `random_counter_set_sha256` is
independent of action id and sign. The auditor requires the `R+` and `R-`
trajectories for an orbit to report the same value at every step even though
their transported bases diverge after their latent trajectories separate.

Two nested controls test whether any gain is specific to the hooked feature
relations. They are not independently searched action banks. The **uniform**
control sets every in-bounds affinity in the selected orbit to
\(a^{u,o}_{ij}=1\). The **predicted-clean** control uses
\(z=D_{16}(\hat x_{0,t})\), fails when any token norm is at or below
`epsilon_z`, and sets

\[
a^{x,o}_{ij}=\epsilon_a+
\frac{1+\operatorname{clip}(\cos(z_i,z_j),-1,1)}{2}.
\]

For each held OOF block, the structural selector chooses `P0` or one signed
orbit without using that block's outcomes. A masked freezer writes this choice
and the operator/config `P/U/X` hashes to the decision ledger before either new
control is generated. Those hashes exclude PNG, sidecar, and score artifacts.
For an active choice, `U` and `X` rerun separate complete trajectories with the
identical orbit, sign, initial noise, geodesic, ratio, and guards; each
recomputes its own graph and scheduler state at every step. For `P0`, all three
reuse the one no-op record but the audit emits logical `U/X` rows with
`P-U=P-X=0`. Every block remains in the intention-to-treat estimand; active-only
analysis is forbidden. For `N` blocks and `N_active` active choices, search
therefore executes exactly `13N + 2N_active` trajectories, not 25 full banks.
The predicted-clean control uses the model's current \(\hat x_0\), not a ground-
truth VAE latent, future state, decoded image, or SPARE reproduction.

The causal 2x2 crosses **descriptor source** with **injection basis**:

| | Relational injection | Norm-matched random injection |
|---|---|---|
| Structural descriptor | first-step local-relation summaries select `P` | the same summaries select `R` |
| Prompt-only descriptor | pooled text selects `P` | pooled text selects `R` |

These are four active selectors, not a descriptor-only no-op. A fixed
context-free selector, described below, is an additional control. The primary
interaction is
\(I=[S_{rel}-P_{rel}]-[S_{rnd}-P_{rnd}]\); it must be positive with a
frozen selection bound at search and a confirmatory simultaneous interval above
zero at replay.

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

The search split first renders the complete paired 13-action primary matrix.
Feasibility has
three non-interchangeable levels. A **record** is valid only when its image,
hook, scheduler, moment, cap, and call ledgers pass. Within each training fold,
an **action** is eligible only when aggregate HPSv2, CLIP, pixel, and diversity
guards computed from that fold pass. A **selector** advances from search only
when its frozen OOF selection guards pass and is confirmatory only when replay
passes the simultaneous inference guards. Collection-level guards are never
used as per-record ridge labels. For valid records, the
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
with a preregistered counter RNG.

After primary scoring, the independent masked freezer receives, for each
`(k,s)`, only `T_{k,s}` scores, held pre-outcome descriptors, and reviewed
operator/config hashes. It cannot read held PNGs, sidecars, or scores. It seals
all OOF choices and logical `P0` rows before the preliminary evaluator runs.
Only then may that evaluator read held primary outcomes; it emits only an
authorization or terminal bit bound to the existing ledger hash and releases
no metric. The generator can read only that bit, the ledger, and generation
inputs, never any score file. If authorized, it generates the frozen nested
`U/X` trajectories above.

Search OOF is a development selection and futility gate, not confirmatory
inference. Its primary, interaction, `P-U`, and `P-X` intervals are selection
statistics with no advertised coverage or paper efficacy claim. A full search
pass permits refitting for replay, but every preprocessing statistic, penalty,
eligibility decision, coefficient, tie rule, and action choice in that refit
may read only the original 13-action primary matrix. `U/X` scores are isolated
inputs to the independent gate evaluator and never train or alter a procedure.
The refitted procedures are serialized and hash-frozen for replay. Query
counts, failed queries, candidate exposure, and failure accounting must match.
Training has a common preregistered compute ceiling rather than artificial
numerical equality; actual fit time, FLOPs, and peak memory are reported. No
control may query replay outcomes.

## Endpoints, Guards, and Inference

An independent verifier, with separately hashed code and no selector-training
role, owns labels, strict scoring, the result-blind audit, and the one-shot
evaluator. For the full structural-relational selector, co-primary endpoints
are (1) TOPIQ-NR delta versus no-op, requiring at least `+0.005`, and (2) an
equal-category macro of OCR exact/normalized accuracy, exact counting accuracy,
and spatial-relation accuracy. The frozen power artifact also sets nonzero
adaptivity margins versus best global for both endpoints. At confirmatory
replay, their simultaneous 95% lower bounds, the 2x2 interaction bounds, and
the intention-to-treat `P-U` and `P-X` bounds on both endpoints must exceed
their power-registered nonzero margins rather than merely zero. Identically
computed search bounds are selection diagnostics only.

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
For confirmatory replay, Max-T gives simultaneous intervals across all
action/selector and `P-U/P-X` contrasts, and Holm controls the preregistered
endpoint families at FWER 0.05. Search uses the same estimators only as a frozen
selection rule. Challenge results are sensitivity analyses, not independent
samples.

Sample counts are not guessed here. A CPU simulator starts from candidate-blind
historical no-op variance, then spans a preregistered conservative grid of
treatment/no-op variance ratios, paired correlations, action heterogeneity,
selector error, selector no-op rates, and all covariance terms needed for the
difference-in-differences and nested controls. For search it simulates the
entire masked-freezer, preliminary authorization, conditional `U/X` generation,
ITT-zero-row, and final futility rule to report selection operating
characteristics without a Type-I or coverage claim. For replay it chooses the
worst-case sample count by directly simulating the final max-T/Holm rejection
rule. If these inputs cannot bound paired, interaction, or nested-control
variance, an independently held, blinded `ao-variance-pilot-v1`
may release only covariance and sample-size artifacts, never rankings; its
prompts/seeds are retired and cannot train or select the method. For TOPIQ
`0.005`, the reliability-calibrated structural effect, and registered
adaptivity margins, search selection and replay/test confirmation are powered
separately for at least 90% joint pass probability and 80% guard power under
their respective rules. Inputs, sensitivity grid, simulation code, pilot
firewall if used, chosen counts, and hashes form
`power_analysis.json`; infeasible power leaves the study blocked.

## One-Shot Artifacts and Sequential Gates

All writers use locks, atomic creation, and refusal when an output exists.
`adaptive_oracle_prompt_manifest_v1.json`, a non-executable engineering
registration YAML, implementation/environment manifests, and an independent
engineering review authorization are committed before GPU smoke. The
executable binds their SHA-256 values and the reviewed commit. `power_analysis`
and selector/freezer artifacts are separate formal-search prerequisites and are
not inputs to the no-scoring engineering gate. Every later formal run binds
config, prompt, action, code, environment, model, scheduler, decision-ledger,
PNG, sidecar, and strict-scorer hashes.

The executable authorization has one canonical repository path. Its exact
bytes must equal the tracked blob at the authorization-carrying `HEAD`, whose
commit is recorded in the attempt, run config, and success receipt; the source
`reviewed_commit` must be its ancestor, and every frozen implementation/input
must still equal its blob at that reviewed commit. This two-commit binding
avoids an impossible self-reference in which an authorization file would need
to contain the hash of the commit that contains it. The textual `reviewer`
identity is an organizational attestation, not a cryptographic signature;
stronger identity assurance requires a separately specified signed-attestation
scheme.

1. **CPU gate:** synthetic affine-cosine constants, zero-feature and zero-`z`
   fail-closed, uniform/predicted-clean/random row sums, orbit/D4/no-wrap,
   both-input-detach, antithetic norm parity,
   tangent/geodesic moments, bitwise no-op, native Euler round trip,
   trajectory-isolation, actual-edge/canonical-D4-counter and namespace
   collision audit (not uniqueness of truncated random values), and a real
   50-step `EulerDiscreteScheduler` round trip must produce one warning-free
   `cpu_audit.json`. Reconstructed-clean relative L2 error is capped at `1e-2`
   after native model-output quantization, while expected-previous-sample
   relative L2 error is capped at `1e-3`; max absolute error is diagnostic.
2. **Engineering gate:** only after independent authorization, run the 11 fresh
   challenge smoke prompts x one retired-on-use seed x 13 primary actions plus
   one pre-outcome, counter-cycled active `U/X` pair per prompt at 1024, Euler,
   50 steps. The cycle covers all signed orbits. Scoring is forbidden. Require
   165 PNG/sidecar pairs, distinct active outputs, complete ledgers, bounds, and
   a warning-free one-shot `engineering_audit.json`. This is runtime stress
   coverage only: images are never inspected or scored, prompt categories are
   not samples for an efficacy claim, and a pass authorizes only construction
   of the separately powered formal-search registration.

   For every task, the pipeline records the timesteps, sigmas, construction and
   post-`set_timesteps` init-noise sigma from its actual scheduler invocation,
   the hash of the prepared initial latent, and direct hashes immediately before
   and after every scheduler step. Callback hashes are only a cross-check and
   cannot supply or shift the pre-step ledger. Python warnings, process-local
   logging records, and stderr are captured in canonical runtime evidence;
   counts and hashes are independently recomputed by the engineering auditor,
   and any warning evidence fails the warning-free gate.
3. **Formal search gate:** run the powered 13-action primary matrix, strict
   offline scoring, and result-blind `search_primary_audit.json`. The masked
   freezer first seals every OOF choice from fold-permitted inputs. Only then
   may an independent preliminary evaluator close the family or emit a ledger-
   bound authorization bit; it releases no outcomes. Generate and strictly
   score the selected controls, audit the complete ITT search, then run one
   `search_selection.json` futility evaluator. Incomplete cells,
   provenance/mechanism warnings, an absent seed-CV oracle gap, selection-rule
   failure versus global, a failed interaction, or either failed `P-U/P-X`
   selection contrast closes this family. None is confirmatory evidence. An
   individual action's aggregate guard failure makes that action ineligible
   under the frozen fold rule and is fully reported; it is not silently
   converted into a family-wide failure or omitted after outcomes.
4. **Outer replay gate:** freeze `selector_freeze.json` and its decision-ledger
   schema first; run only the frozen selectors and controls on powered replay
   data, including separately generated `U/X` trajectories under the same
   chosen signed orbit; `P0` contributes audited ITT zero rows. Audit before the
   independent one-shot confirmatory evaluator. Both co-primaries, the
   relational interaction, `P-U`, `P-X`, and all guards must pass versus best-
   global, context-free, prompt-only, shuffled, random-bank, and equal-budget
   controls under the registered simultaneous inference.
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

## Implementation and Authorization Checklist

Implemented and still subject to the one-shot CPU audit are the exact
conditional `up_blocks.0` hook, 16x16 local operator, D4-canonical random-edge,
uniform and predicted-clean controls, fixed-ratio moment geodesic, real Euler
round-trip diagnostics, fresh engineering prompt/seed collision scanner,
15-action generator, sidecar contract, warning capture, strict PNG auditor,
and non-executable registration. Model loading is rooted only through a held
directory descriptor exposed as `/proc/self/fd/<n>`; recursive pre/post tree
signatures bind file bytes and file/directory identity metadata, so replacement
or mutation fails the attempt. This is fail-closed detection, not a claim of
noninterference against another same-UID process. The warning-as-error CPU suite
currently passes `240/240`; the remaining legacy suite passes `333/333`.
Prompt assets replay byte-for-byte from the pinned PartiPrompts
source using the committed exclusion inventory without reading `outputs/`.
That inventory freezes sorted, unique per-file prompt, source-row, and seed
projections for the historical metadata corpus and separately inventories
forbidden score/quality paths. An unmasked physical `cuda:1` environment-only
probe matches the pinned GPU/runtime lock. These are implementation checks, not
generation or efficacy evidence.

GPU smoke remains blocked until a clean reviewed implementation commit,
warning-free one-shot `cpu_audit.json`, and later independent executable
authorization all exist and agree by hash. The executable must use physical
GPU 1-4 with `CUDA_VISIBLE_DEVICES` unset, recheck authorized bytes across
runtime import and generation, and record the physical GPU identity.

The result-blind selected-control freezer, power simulator, OOF ridge/oracle and
equal-budget controls, 2x2 interaction, provenance-locked quality/structural
scorers, and formal auditors/evaluators are required before formal search, not
before engineering smoke. They remain blockers for any efficacy experiment,
distillation, or RL. Until the engineering blockers above close, status remains
`blocked_registration_only`: no GPU. RL remains unauthorized at every gate in
this document.
