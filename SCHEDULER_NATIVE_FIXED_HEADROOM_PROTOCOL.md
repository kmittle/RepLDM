# Scheduler-Native Fixed Headroom Protocol

## Scope and Claim Boundary

This registration asks one narrow screening question: under the frozen 50-step
SDXL Euler trajectory, does a fixed, scheduler-native structural clean-endpoint
direction produce enough quantitative signal to justify independent review?
It does not by itself establish a practically meaningful effect or population
generalization, and it does not authorize learned renderers, distillation, RL,
validation, or final testing. Its conceptual link
to conference Attention Guidance is the use of structure exposed during one
frozen denoiser call; the primary hypothesis deliberately tests whether
attention is unnecessary by using latent-only spectral and Laplacian bases.

The historical `StructuralUNetBasisProvider` could not isolate that hypothesis:
it installed hooks and constructed unused bases. The reviewed executable now
uses `LazyLatentStructureBasisProvider`, constructs only action-requested bases,
and records exact hook and basis sets. Spectral and Laplacian primary actions
register no Q/K or U-Net feature hook. The historical smoke remains wiring-only
and is not quality evidence.

## Frozen Data and Seeds

The source is Google Research Parti commit
`5a657978134374ce28973948331b319adef164bd`, file SHA-256
`fab29e41bb512a169b56acab4cf2a41dcb675e285df2efcde6640c7dd3c440eb`.
The builder verifies both Git HEAD and file bytes. It NFKC-normalizes, collapses
whitespace, and casefolds text before excluding every prompt in the repository's
ten pre-existing CSV files. Explicit and text-derived exclusions cover 177
unique texts and 165 source rows. Within each of all 11 Challenges, candidates
are ordered by the frozen namespace hash, then assigned as follows:

| Split | Prompts | Per Challenge | Reserved Seeds |
|---|---:|---:|---|
| smoke | 11 | 1 | `1798464083` |
| development | 33 | 3 | `1932556753,1065503757,201635682` |
| validation | 22 | 2 | `1071459191,1417413784,649810593` |
| final | 22 | 2 | `1134005626,1510693625,1730014245` |

Generated seeds `0,7,19,42,73,123` and previously registered seeds
`11,29,101` are retired. In particular, `0,42,123` are development-consumed and
must never again be described as unseen final seeds. All new seed sets are
deterministically selected, globally disjoint, and unused in the audited output
inventory.

The prompt manifest is
`eval-pipeline/prompts/scheduler_native_fixed_headroom_manifest.json`, SHA-256
`5373acf08f0e28d586732909f38787f8180a5d12c1ed58c2e1881134c10b6d5f`.
It records source, exclusions, selection digests, split entries, CSV hashes, and
seed digests. Reproduce or verify it with:

```bash
python eval-pipeline/build_scheduler_native_fixed_headroom_prompts.py --check
```

## Frozen Actions and Inference

Both registrations require `EulerDiscreteScheduler`, epsilon prediction, zero
churn, 50 NFE, `euler_clean_endpoint`, `match_rms`, CFG 7.5, 1024px Stage 1,
and one U-Net call per step. The smoke has four actions: scheduler no-op,
historical-provider zero identity, low-frequency probe, and Laplacian probe. It
is wiring-only (`11 x 1 x 4 = 44` outputs if separately authorized) and cannot
be scored for selection.

The reviewed development grid has eight actions (`33 x 3 x 8 = 792` outputs):
no-op, lazy-provider zero identity, four non-attention primary actions
(low/mid/high spectral and Laplacian), and semantic/FreeU mechanism ablations.
One coefficient magnitude (`0.02`) is frozen to avoid an amplitude search. The
primary Holm family contains exactly four actions. The two mechanism ablations
form a separate Holm family and are not selection-eligible; their success cannot
authorize a non-attention claim.

The frozen files and hashes are:

- `eval-pipeline/configs/scheduler_native_fixed_headroom_smoke.yaml`:
  `609fbd984cb7ab17b49fc4793d6ddfbb020ce276ff3bde6deb1f1f83b7269a0a`
- `eval-pipeline/configs/scheduler_native_fixed_headroom_development.yaml`:
  `aa42ab90e9dfc7993d0d8e5e0dcbcdfd80e3426cb3400c33dd3d2db1122fdb61`
- `eval-pipeline/configs/scheduler_native_fixed_headroom_actions_v1.yaml`:
  `9b64c5773b6b4494507036b677193d1e596665ce92393434802a9e01c01f3393`
- `eval-pipeline/configs/scheduler_native_fixed_headroom_evaluation_v1.yaml`:
  `5a1cb85e7ea30deef89bd0ae353bba2ac380dc12ad78b8138a22299e5ad55f80`

The immutable registration remains fail-closed. The distinct executable copy,
`scheduler_native_fixed_headroom_actions_v1.yaml`, was independently reviewed
and preserves its prompt, seed, action, coefficient, sampling, and analysis
fields. Generation binds both files by SHA-256.

## Gates and Stop Rules

The zero identity must be byte-identical to no-op. Every nonzero run must record
50 finite per-step mapping gains, applied update ratios, moment errors, requested
and constructed bases, registered hooks, and exactly one U-Net call. Scores must
bind scorer code, package, preprocessing, and checkpoint provenance.

The registered screen triggers only when one primary action has mean TOPIQ-NR
delta at least `+0.005`, a positive crossed prompt/seed bootstrap lower bound,
prompt-level sign-flip significance after four-action Holm correction, and all
registered HPSv2, CLIP, clipping, saturation, and contrast guards. The `+0.005`
condition is a point-estimate threshold, not a confidence bound against a
`+0.005` null. Holm correction covers only the four TOPIQ zero-null tests; guard
intervals are not simultaneous across actions. Prompts are the inference units
for the sign-flip test, while challenge summaries are descriptive. Therefore a
trigger returns only `independent_review_required`; it does not establish
practical headroom, choose a winner, or authorize validation.

If no primary triggers, the result closes only these four positive `+0.02`
spectral/Laplacian actions and their direct distillation or RL route. It does not
falsify signed, scheduled, target-spectrum, or learned renderers. Any
confirmation must use a newly frozen split, simultaneous one-sided bounds across
carried actions and guards, a preregistered tie-break or carry-all rule, and a
blinded qualitative rubric fixed before image access.

The repository already contains readable validation and final prompts and
seeds, so those files are reproducibility fixtures rather than a genuinely
hidden TPAMI holdout. A new confirmation split should be hash-committed and held
by an independent custodian until the method and analysis are frozen. The
existing 22-prompt final split is not
large or diverse enough for a TPAMI-level broad efficacy claim. Such a claim
still requires larger external prompt suites, multiple training seeds and model
settings, strong same-NFE baselines, randomized blinded human evaluation, and
complete runtime, VRAM, failure, and compute reporting.

## Completed Development Result

The authorized run
`outputs/latent_renderer/scheduler_native_fixed_headroom_development_v2`
completed `792/792` generations and strict score rows. The result-blind audit
passed without warnings, and the one-shot evaluator returned `decision=null_route`.
Relative to no-op, primary TOPIQ-NR deltas were `-0.001154` for spectral-low,
`+0.004027` for spectral-mid, `-0.004024` for spectral-high, and `-0.002463`
for Laplacian. Spectral-mid had a positive interval and passed its Holm-adjusted
zero-null test, but missed the preregistered `+0.005` point screen and failed the
clipped-fraction interval guard. It is descriptive evidence, not a winner, and
must not be tuned post hoc.

The sealed config, manifest, scores, audit, evaluation JSON, and evaluation CSV
SHA-256 values are respectively
`4566993c...bf1de`, `badab464...af319`, `f6ab5f08...c7acb`,
`c8d2dedf...462a4`, `890f7906...fe9e`, and `dadcf5b5...1c90`.
No validation, distillation, method selection, learned renderer, or RL run is
authorized by this result.
