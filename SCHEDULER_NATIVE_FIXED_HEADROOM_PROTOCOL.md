# Scheduler-Native Fixed Headroom Protocol

## Scope and Claim Boundary

This registration asks one narrow question: under the frozen 50-step SDXL
Euler trajectory, does a fixed, scheduler-native structural clean-endpoint
direction improve held-out quality? It does not authorize generation, learned
renderers, distillation, RL, validation, or final testing. Its conceptual link
to conference Attention Guidance is the use of structure exposed during one
frozen denoiser call; the primary hypothesis deliberately tests whether
attention is unnecessary by using latent-only spectral and Laplacian bases.

The current `StructuralUNetBasisProvider` cannot establish that hypothesis. It
registers a Q/K hook and captures decoder features in its constructor, then
constructs semantic, spectral, FreeU, and Laplacian bases on every call,
regardless of which coefficient is nonzero. Therefore
`scheduler_native_fixed_headroom_smoke.yaml` is a historical-provider wiring
ablation only. Development is blocked until a lazy provider constructs only the
action-requested bases and records its exact hook and basis sets. Spectral and
Laplacian primary actions must register no Q/K or U-Net feature hook.

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

The blocked development grid has eight actions (`33 x 3 x 8 = 792` outputs):
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

They use the existing `latent_renderer_registration_v1` fail-closed schema, so
the current generator rejects them. A later executable copy requires an
independent reviewed authorization and must preserve every prompt, seed, action,
coefficient, sampling, and analysis field byte-for-byte.

## Gates and Stop Rules

The zero identity must be byte-identical to no-op. Every nonzero run must record
50 finite per-step mapping gains, applied update ratios, moment errors, requested
and constructed bases, registered hooks, and exactly one U-Net call. Scores must
bind scorer code, package, preprocessing, and checkpoint provenance.

A primary action passes development only with mean TOPIQ-NR delta at least
`+0.005`, positive crossed prompt/seed bootstrap lower bound, prompt-level
sign-flip significance after four-action Holm correction, and all registered
HPSv2, CLIP, clipping, saturation, and contrast guards. Otherwise the fixed
renderer, distillation, and RL route closes without validation. A winner must be
frozen before one validation run; final remains untouched until validation and
blinded qualitative gates pass.

The 22-prompt final split is only a static-headroom confirmation gate. It is not
large or diverse enough for a TPAMI-level broad efficacy claim. Such a claim
still requires larger external prompt suites, multiple training seeds and model
settings, strong same-NFE baselines, randomized blinded human evaluation, and
complete runtime, VRAM, failure, and compute reporting.
