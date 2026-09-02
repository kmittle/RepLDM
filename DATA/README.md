# Latent Renderer Data Catalog

This directory contains generated JSONL indexes for OPD, DPO, and RL latent-renderer
research. Images, model weights, feature caches, and run logs remain in the five source
directories; records reference payloads by absolute path.

The checked-in configuration is intentionally a development candidate inventory, not a
formal training release. The plain command below is the final command only after every
artifact hash has been populated and a selected payload view has been bound:

```bash
python eval-pipeline/build_data_catalog.py
python eval-pipeline/build_data_catalog.py --validate-only
```

Until then, use an explicitly named development directory. It is evidence for auditing
only and must never feed training:

```bash
python eval-pipeline/build_data_catalog.py --allow-dirty --no-verify-paths \
  --output-dir DATA/dev-catalog-YYYYMMDD
```

The formal command fails closed while `expected_artifact_hashes` is empty; this is
intentional. After selected payload SHA-256 manifests and the frozen configuration are
committed, rebuild from the clean pushed commit and run `--validate-only` before use.

Development builds may use `--allow-dirty --no-verify-paths`, but they never update
`DATA/current` and must not feed an experiment. `DATA/current` changes only after the new
content-addressed release passes full validation.

A formal build still requires local `HEAD` to equal its upstream tip. Later validation of
that immutable release accepts normal upstream fast-forwards because the recorded commit
remains an ancestor of the recorded upstream ref. A missing ref or rewritten/divergent
history fails validation.

## Payload Integrity Boundary

The catalog verifies metadata and referenced paths; it does not claim that image bytes are
immutable. None of the five roots provides a checksum manifest covering the actual
4KLSDB, PixVerve, Aesthetic-4K, extracted Style30K, or 179 GB feature-shard payloads.
Therefore `training_ready` and `complete` remain `false`. Before training, materialize a
selected view and hash every selected payload with SHA-256. The
`--require-training-ready` guard intentionally rejects this candidate catalog.

## Audited Sources

| Physical directory | Catalog decision |
| --- | --- |
| `/mnt/miah204/bycao/Pixel-Render` | Inventory 5,058,099 manifest rows: 3,287,110 files exist and 1,770,989 are missing. None enters training. Aesthetic-4K contributes 12,009 train images and Style30K contributes 25,868; Aesthetic eval and OmniStyle are excluded. |
| `/mnt/miah204/bycao/Sana` | 4KLSDB and 50,000 GenEval train prompts are candidates. PickScore and OCR lack data-specific local license evidence, so all their rows are excluded. DrawBench and test splits are held out. |
| `/mnt/miah209/bycao/Pixel-Render` | 129,484 4KLSDB and 95,735 PixVerve source rows are cataloged; 95,733 PixVerve rows remain training-eligible after the two exact benchmark matches are excluded. PixVerve-Bench, MJHQ-30K, and all 3,984 local 4KLSDB validation/test images are protected evaluation data. |
| `/mnt/miah209/bycao/sana_data` | Eight feature/mask shard pairs are auxiliary cache records only. Size and ID order are checked, but shard bytes remain unbound. |
| `/mnt/miah209/bycao/sana_runs` | Run provenance only; contributes zero samples. |

The frozen candidate total is 313,094 rows: 275,217 contain prompts and 263,094 reference
images. Source counts, splits, modalities, protected-source hashes, and candidate statistics
are pinned by the versioned config. Artifact byte counts and hashes are recorded in each
release manifest; `expected_artifact_hashes` is intentionally empty for this development-only
catalog, so those values are not yet a formal acceptance contract. Rows marked `review_required`,
`noncommercial_research`, or `benchmark_only` are excluded by a fail-closed allowlist.

The historical expanded development release under
`DATA/dev-catalog-protected-20260901/` is retained for audit comparison only. It predates
the current source-containment checks and is superseded; rebuild a newly named development
release after any catalog-code change. No historical development release authorizes training.

Missing manifest entries remain auditable as `missing_image_reference` rows with a null
`image_path`, `source_image_missing`, and their declared path in `source_record`.

Every row follows `repldm.data_record.v1`. `training_candidates.jsonl` is verified
byte-for-byte as the eligible-row derivation of configured sources. The 49,393-row
`benchmark_holdouts.jsonl` contains 46,619 normalized unique prompts, 39,712 image-bearing
rows, and 37,160 unique images. It includes HPSv2, GenEval test, DrawBench, MJHQ-30K,
PixVerve-Bench, both Aesthetic-4K evaluation splits, all local 4KLSDB validation/test rows,
and frozen RepLDM prompts. The 4KLSDB metadata IDs must map one-to-one to four validation
and four test x4 HR shards. Exact text matching is enforced; semantic text leakage and image
near-duplicate removal are still required before a paper training set is frozen.

The `payload_integrity` field distinguishes prompt-only rows, missing references,
unbound image paths, and unbound auxiliary shards. No `unbound_*` row is sufficient to
authorize a training run without the selected-payload checksum step above.

`training_views.jsonl` defines method-specific filters. Its DPO view is only a prompt pool:
these directories do not contain matched preference pairs. Generate candidates and labels
under a separately versioned DPO collection protocol.
