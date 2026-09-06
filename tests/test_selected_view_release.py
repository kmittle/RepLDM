from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from unittest import mock

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "eval-pipeline"
sys.path.insert(0, str(EVAL_ROOT))

import data_catalog.selected as selected_module
from data_catalog.io import canonical_json_bytes
from data_catalog.schema import make_record, normalize_prompt
from data_catalog.selected import (
    PARENT_ARTIFACT_ORDER,
    SELECTED_CONFIG_SCHEMA,
    SELECTED_ROW_SCHEMA,
    SELECTED_VIEW_SCHEMA,
    STRATA,
    THRESHOLD_CALIBRATION_SCHEMA,
    decode_image_payload,
    dct_phash_v1,
    model_binding_sha256,
    selected_release_id,
    validate_selected_view_release,
)

SHA256 = "a" * 64
COMMIT = "b" * 40
SOURCES = ("four_k_lsdb", "pixverve_95k")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(path: Path) -> dict[str, object]:
    return {
        "path": str(path.absolute()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _calibration(
    path: Path,
    *,
    metric: str,
    selected_value: float | int,
    comparison: str,
    model_hash: str,
) -> dict[str, object]:
    source_path = path.with_name(f"{path.stem}-source.jsonl")
    source_path.write_bytes(
        f"calibration source for {metric}\n".encode("utf-8")
    )
    _write_json(
        path,
        {
            "schema": THRESHOLD_CALIBRATION_SCHEMA,
            "metric": metric,
            "selected_value": selected_value,
            "comparison": comparison,
            "sample_count": 32,
            "positive_count": 16,
            "negative_count": 16,
            "source": _binding(source_path),
            "model_binding_sha256": model_hash,
        },
    )
    return _binding(path)


def _descriptor(path: Path, **extra: object) -> dict[str, object]:
    return {**_binding(path), **extra}


class SelectedReleaseFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.catalogs = root / "DATA" / "catalogs"
        self.selected_root = root / "DATA" / "selected-views"
        self.assets = root / "assets"
        self.catalogs.mkdir(parents=True)
        self.selected_root.mkdir(parents=True)
        self.assets.mkdir()
        self.parent = self._write_parent()
        self.config, self.rows = self._selection_inputs()
        self.release = self._write_child()

    def _write_parent(self) -> Path:
        release = self.catalogs / ("catalog-" + "1" * 20)
        release.mkdir()
        artifacts = []
        for name in PARENT_ARTIFACT_ORDER:
            path = release / name
            if name == "training_candidates.jsonl":
                continue
            if name == "benchmark_holdouts.jsonl":
                path.write_bytes(
                    canonical_json_bytes(
                        {"id": "protected-0", "prompt": "a protected fixture prompt"}
                    )
                )
            else:
                path.write_text(f"fixture {name}\n", encoding="utf-8")
            artifacts.append(_descriptor(path))

        candidate_path = release / "training_candidates.jsonl"
        candidate_path.touch()
        artifacts.append(_descriptor(candidate_path))
        by_name = {row["path"].rsplit("/", 1)[-1]: row for row in artifacts}
        ordered = [
            {
                "path": name,
                "bytes": by_name[name]["bytes"],
                "sha256": by_name[name]["sha256"],
            }
            for name in PARENT_ARTIFACT_ORDER
        ]
        manifest = {
            "schema": "repldm.data_catalog.v1",
            "release_id": release.name,
            "complete": False,
            "candidate_catalog_complete": True,
            "development_build": False,
            "training_ready": False,
            "verify_paths": True,
            "git": {
                "commit": COMMIT,
                "dirty": False,
                "pushed": True,
                "upstream_commit": COMMIT,
                "upstream_ref": "refs/remotes/origin/fixture",
            },
            "protected_normalized_unique_prompts": 46619,
            "protected_unique_images": 37160,
            "artifacts": ordered,
        }
        _write_json(release / "manifest.json", manifest)
        return release

    def _selection_inputs(self) -> tuple[dict[str, object], list[dict[str, object]]]:
        model_file = self.assets / "model.bin"
        model_file.write_bytes(b"frozen-model")
        model = {
            "id": "fixture/model",
            "revision": "d" * 40,
            "files": [_binding(model_file)],
        }
        model_hash = model_binding_sha256(model)

        tokenizer_models = []
        for index in (1, 2):
            tokenizer_file = self.assets / f"tokenizer-{index}.json"
            tokenizer_file.write_bytes(f"frozen-tokenizer-{index}".encode("ascii"))
            tokenizer_models.append(
                {
                    "id": "fixture/sdxl",
                    "revision": "e" * 40,
                    "files": [_binding(tokenizer_file)],
                }
            )

        license_file = self.assets / "LICENSE.txt"
        license_file.write_text("fixture license\n", encoding="utf-8")
        index_bindings = {}
        for name in ("semantic_text", "phash", "image_embedding"):
            path = self.assets / f"protected-{name}.jsonl"
            _write_json(path, {"id": "protected-0"})
            index_bindings[name] = _binding(path)

        phash_definition = {
            "implementation": "dct_phash_v1",
            "hash_bits": 64,
            "resize": 32,
            "low_frequency_size": 8,
            "exclude_dc": True,
        }
        phash_hash = hashlib.sha256(canonical_json_bytes(phash_definition)).hexdigest()
        classifier_calibration = _calibration(
            self.assets / "classifier-calibration.json",
            metric="classifier_confidence_margin",
            selected_value=0.05,
            comparison="reject_below_margin",
            model_hash=model_hash,
        )
        semantic_calibration = _calibration(
            self.assets / "semantic-calibration.json",
            metric="cosine_similarity",
            selected_value=0.8,
            comparison="reject_at_or_above",
            model_hash=model_hash,
        )
        phash_calibration = _calibration(
            self.assets / "phash-calibration.json",
            metric="hamming_distance",
            selected_value=4,
            comparison="reject_at_or_below",
            model_hash=phash_hash,
        )
        image_calibration = _calibration(
            self.assets / "image-calibration.json",
            metric="cosine_similarity",
            selected_value=0.9,
            comparison="reject_at_or_above",
            model_hash=model_hash,
        )
        parent_manifest = self.parent / "manifest.json"
        config: dict[str, object] = {
            "schema": SELECTED_CONFIG_SCHEMA,
            "view_id": "balanced_latent_renderer_v1",
            "parent_catalog": {
                "release_id": self.parent.name,
                "manifest_sha256": _sha256(parent_manifest),
            },
            "sources": {
                "four_k_lsdb": {
                    "model_prompt_field": "prompt",
                    "raw_prompt_field": "prompt",
                    "license": "CC-BY-4.0",
                    "license_status": "verified_from_dataset_card",
                    "license_evidence": [_binding(license_file)],
                },
                "pixverve_95k": {
                    "model_prompt_field": "source_record.model_prompt",
                    "raw_prompt_field": "source_record.raw_prompt",
                    "license": "Apache-2.0",
                    "license_status": "verified_from_dataset_card",
                    "license_evidence": [_binding(license_file)],
                },
            },
            "strata": list(STRATA),
            "quotas": {
                "train_per_source_stratum": 4,
                "validation_per_source_stratum": 2,
                "train_total": 64,
                "validation_total": 32,
            },
            "selection": {
                "seed": "fixture-selection-seed",
                "algorithm": "sha256_seed_nul_record_id_v1",
                "tie_rule": "reject",
            },
            "classifier": {
                "model": model,
                "class_templates": {
                    name: [f"an image of {name}"] for name in STRATA
                },
                "confidence_margin": 0.05,
                "tie_rule": "reject",
                "calibration": classifier_calibration,
            },
            "tokenizers": [
                {
                    "id": "sdxl_tokenizer",
                    "model": tokenizer_models[0],
                    "max_tokens": 77,
                    "add_special_tokens": True,
                    "truncation": False,
                },
                {
                    "id": "sdxl_tokenizer_2",
                    "model": tokenizer_models[1],
                    "max_tokens": 77,
                    "add_special_tokens": True,
                    "truncation": False,
                },
            ],
            "clip_tokenizer_id": "sdxl_tokenizer",
            "decoder": {
                "library": "Pillow",
                "version": "11.0.0",
                "littlecms_version": "2.12",
                "max_image_pixels": 400_000_000,
                "exif_transpose": True,
                "icc_to_srgb": True,
                "output_mode": "RGB",
                "pixel_hash": "sha256_rgb_u64be_width_height_bytes_v1",
            },
            "semantic_text": {
                "model": model,
                "threshold": 0.8,
                "comparison": "reject_at_or_above",
                "calibration": semantic_calibration,
            },
            "phash": {
                **phash_definition,
                "threshold": 4,
                "comparison": "reject_at_or_below",
                "calibration": phash_calibration,
            },
            "image_embedding": {
                "model": model,
                "threshold": 0.9,
                "comparison": "reject_at_or_above",
                "calibration": image_calibration,
            },
            "protected_index": {
                "holdout_rows": 49393,
                "normalized_unique_prompts": 46619,
                "unique_images": 37160,
                **index_bindings,
            },
        }

        rows = []
        candidates = []
        seed = config["selection"]["seed"]  # type: ignore[index]
        for source in SOURCES:
            for stratum in STRATA:
                cell_rows = []
                for index in range(6):
                    image = self.assets / f"{source}-{stratum}-{index}.png"
                    source_number = SOURCES.index(source)
                    stratum_number = STRATA.index(stratum)
                    pixels = Image.new(
                        "RGB",
                        (8, 8),
                        (
                            20 + source_number * 90 + index,
                            10 + stratum_number * 25,
                            30 + index * 20,
                        ),
                    )
                    pixels.putpixel(
                        (index % 8, stratum_number % 8),
                        (200 - index, 150 + source_number, 80 + stratum_number),
                    )
                    pixels.save(image)
                    prompt = f"A {source} {stratum} fixture {index}"
                    source_record = (
                        {}
                        if source == "four_k_lsdb"
                        else {"model_prompt": prompt, "raw_prompt": prompt + " raw"}
                    )
                    candidate = make_record(
                        source=source,
                        stable_key=f"{stratum}-{index}",
                        source_roots=("fixture",),
                        split="train",
                        prompt=prompt,
                        image_path=image.absolute(),
                        width=64,
                        height=64,
                        license_name=config["sources"][source]["license"],  # type: ignore[index]
                        license_status=config["sources"][source][
                            "license_status"
                        ],  # type: ignore[index]
                        modality="image_text",
                        intended_use=("latent_renderer_training",),
                        training_eligible=True,
                        source_record=source_record,
                    )
                    candidates.append(candidate)
                    model_prompt = prompt
                    raw_prompt = prompt if source == "four_k_lsdb" else prompt + " raw"
                    decoded = decode_image_payload(image, config["decoder"])
                    row = {
                        **candidate,
                        "schema": SELECTED_ROW_SCHEMA,
                        "stratum": stratum,
                        "selection_digest": hashlib.sha256(
                            f"{seed}\0{candidate['id']}".encode("utf-8")
                        ).hexdigest(),
                        "model_prompt": model_prompt,
                        "raw_prompt": raw_prompt,
                        "raw_file_sha256": _sha256(image),
                        "decoded_pixel_sha256": decoded.pixel_sha256,
                        "decoded_width": decoded.width,
                        "decoded_height": decoded.height,
                        "phash": dct_phash_v1(decoded),
                        "token_counts": {
                            "sdxl_tokenizer": 12,
                            "sdxl_tokenizer_2": 12,
                        },
                        "classifier_check": {
                            "stratum": stratum,
                            "top_score": 0.9,
                            "runner_up_score": 0.8,
                            "confidence_margin": 0.09999999999999998,
                            "required_margin": 0.05,
                            "model_binding_sha256": model_hash,
                        },
                        "exact_text_checks": {
                            name: {
                                "normalized_sha256": hashlib.sha256(
                                    normalize_prompt(value).encode("utf-8")
                                ).hexdigest(),
                                "protected_matches": [],
                            }
                            for name, value in {
                                "model_prompt": model_prompt,
                                "raw_prompt": raw_prompt,
                            }.items()
                        },
                        "semantic_text_checks": {
                            name: {
                                "nearest_protected_id": "protected-0",
                                "similarity": 0.7,
                                "threshold": 0.8,
                                "model_binding_sha256": model_hash,
                            }
                            for name in ("model_prompt", "raw_prompt")
                        },
                        "nearest_protected_image": {
                            "phash": {
                                "nearest_protected_id": "protected-0",
                                "distance": 5,
                                "threshold": 4,
                                "definition_sha256": phash_hash,
                            },
                            "embedding": {
                                "nearest_protected_id": "protected-0",
                                "similarity": 0.8,
                                "threshold": 0.9,
                                "model_binding_sha256": model_hash,
                            },
                        },
                    }
                    cell_rows.append(row)
                cell_rows.sort(key=lambda row: (row["selection_digest"], row["id"]))
                for rank, row in enumerate(cell_rows, 1):
                    row["selection_rank"] = rank
                    row["selected_split"] = "train" if rank <= 4 else "validation"
                    row["fold"] = rank - 1 if rank <= 4 else None
                    rows.append(row)

        candidate_path = self.parent / "training_candidates.jsonl"
        candidate_path.write_bytes(b"".join(canonical_json_bytes(row) for row in candidates))
        parent = json.loads((self.parent / "manifest.json").read_text(encoding="utf-8"))
        candidate_binding = _descriptor(candidate_path)
        for artifact in parent["artifacts"]:
            if artifact["path"] == "training_candidates.jsonl":
                artifact.update(
                    bytes=candidate_binding["bytes"], sha256=candidate_binding["sha256"]
                )
        _write_json(self.parent / "manifest.json", parent)
        config["parent_catalog"]["manifest_sha256"] = _sha256(  # type: ignore[index]
            self.parent / "manifest.json"
        )
        return config, rows

    def _write_child(self) -> Path:
        staging = self.selected_root / "staging"
        staging.mkdir()
        config_path = staging / "selection-config.json"
        payload_path = staging / "selected-payload.jsonl"
        _write_json(config_path, self.config)
        payload_path.write_bytes(b"".join(canonical_json_bytes(row) for row in self.rows))
        parent_manifest = self.parent / "manifest.json"
        parent = json.loads(parent_manifest.read_text(encoding="utf-8"))
        core = {
            "complete": True,
            "training_ready": True,
            "development_build": False,
            "git": {
                "commit": COMMIT,
                "dirty": False,
                "pushed": True,
                "upstream_commit": COMMIT,
                "upstream_ref": "refs/remotes/origin/fixture",
            },
            "config_repo_path": "eval-pipeline/configs/selected_view_v1.json",
            "parent_catalog": {
                "path": str(parent_manifest.absolute()),
                "release_id": self.parent.name,
                "manifest_sha256": _sha256(parent_manifest),
                "artifacts": copy.deepcopy(parent["artifacts"]),
            },
            "config": _descriptor(config_path, schema=SELECTED_CONFIG_SCHEMA),
            "selected_payload": _descriptor(
                payload_path,
                schema=SELECTED_ROW_SCHEMA,
                rows=96,
                splits={"train": 64, "validation": 32},
            ),
        }
        release_id = selected_release_id(core)
        release = self.selected_root / release_id
        staging.rename(release)
        for key in ("config", "selected_payload"):
            core[key]["path"] = str(release / Path(core[key]["path"]).name)
        # The release ID binds relative artifact names, not staging paths.
        normalized_core = copy.deepcopy(core)
        normalized_core["config"]["path"] = "selection-config.json"
        normalized_core["selected_payload"]["path"] = "selected-payload.jsonl"
        corrected_id = selected_release_id(normalized_core)
        if corrected_id != release.name:
            corrected = self.selected_root / corrected_id
            release.rename(corrected)
            release = corrected
        normalized_core["parent_catalog"]["path"] = str(parent_manifest.absolute())
        _write_json(
            release / "manifest.json",
            {"schema": SELECTED_VIEW_SCHEMA, "release_id": release.name, **normalized_core},
        )
        return release

    def validate(self) -> dict[str, object]:
        parent = json.loads((self.parent / "manifest.json").read_text(encoding="utf-8"))
        with (
            mock.patch.object(
                selected_module, "_validate_candidate_parent", return_value=parent
            ),
            mock.patch.object(selected_module, "_validate_selected_git"),
        ):
            return validate_selected_view_release(
                self.release, repository_root=self.root, require_formal=True
            )

    def reseal_child(self) -> None:
        """Refresh child descriptors and its content-addressed directory name."""
        manifest_path = self.release / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config_path = self.release / "selection-config.json"
        payload_path = self.release / "selected-payload.jsonl"
        config_descriptor = _descriptor(config_path, schema=SELECTED_CONFIG_SCHEMA)
        config_descriptor["path"] = config_path.name
        payload_descriptor = _descriptor(
            payload_path,
            schema=SELECTED_ROW_SCHEMA,
            rows=96,
            splits={"train": 64, "validation": 32},
        )
        payload_descriptor["path"] = payload_path.name
        manifest["config"] = config_descriptor
        manifest["selected_payload"] = payload_descriptor
        core = {
            key: value
            for key, value in manifest.items()
            if key not in {"schema", "release_id"}
        }
        release_id = selected_release_id(core)
        if self.release.name != release_id:
            replacement = self.selected_root / release_id
            self.release.rename(replacement)
            self.release = replacement
            manifest_path = self.release / "manifest.json"
        manifest["release_id"] = release_id
        _write_json(manifest_path, manifest)

    def rebind_parent(self) -> None:
        """Refresh the config and child bindings after a parent mutation."""
        parent_path = self.parent / "manifest.json"
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        parent_hash = _sha256(parent_path)
        config_path = self.release / "selection-config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["parent_catalog"]["manifest_sha256"] = parent_hash
        _write_json(config_path, config)
        manifest_path = self.release / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["parent_catalog"]["manifest_sha256"] = parent_hash
        manifest["parent_catalog"]["artifacts"] = copy.deepcopy(parent["artifacts"])
        _write_json(manifest_path, manifest)
        self.reseal_child()


def test_complete_child_selected_view_passes(tmp_path: Path) -> None:
    fixture = SelectedReleaseFixture(tmp_path)
    manifest = fixture.validate()
    assert manifest["training_ready"] is True
    assert manifest["selected_payload"]["rows"] == 96


def test_candidate_parent_must_remain_non_training_ready(tmp_path: Path) -> None:
    fixture = SelectedReleaseFixture(tmp_path)
    parent_path = fixture.parent / "manifest.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent["training_ready"] = True
    _write_json(parent_path, parent)
    fixture.rebind_parent()
    with pytest.raises(ValueError, match="candidate parent must remain non-training-ready"):
        fixture.validate()


def test_child_must_bind_exactly_all_fifteen_parent_artifacts(tmp_path: Path) -> None:
    fixture = SelectedReleaseFixture(tmp_path)
    manifest_path = fixture.release / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["parent_catalog"]["artifacts"].pop()
    _write_json(manifest_path, manifest)
    fixture.reseal_child()
    with pytest.raises(ValueError, match="15 parent artifacts"):
        fixture.validate()


def test_child_parent_binding_must_point_to_manifest_file(tmp_path: Path) -> None:
    fixture = SelectedReleaseFixture(tmp_path)
    manifest_path = fixture.release / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["parent_catalog"]["path"] = str(
        (fixture.parent / "training_candidates.jsonl").absolute()
    )
    _write_json(manifest_path, manifest)
    fixture.reseal_child()

    with pytest.raises(ValueError, match="must name manifest.json"):
        fixture.validate()


def test_missing_calibration_artifact_fails_closed(tmp_path: Path) -> None:
    fixture = SelectedReleaseFixture(tmp_path)
    calibration = Path(fixture.config["semantic_text"]["calibration"]["path"])
    calibration.unlink()
    with pytest.raises(ValueError, match="calibration"):
        fixture.validate()


def test_calibration_source_hash_is_recomputed_from_bound_file(tmp_path: Path) -> None:
    fixture = SelectedReleaseFixture(tmp_path)
    calibration_path = Path(fixture.config["semantic_text"]["calibration"]["path"])
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["source"]["sha256"] = "d" * 64
    _write_json(calibration_path, calibration)

    # Refresh the calibration file binding in the frozen config so validation
    # reaches the source-file hash check rather than stopping at the outer
    # artifact descriptor.
    config_path = fixture.release / "selection-config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["semantic_text"]["calibration"] = _binding(calibration_path)
    _write_json(config_path, config)
    fixture.reseal_child()

    with pytest.raises(ValueError, match="calibration source content differs"):
        fixture.validate()


def test_calibration_source_binding_cannot_use_prefixed_top_level_fields(
    tmp_path: Path,
) -> None:
    fixture = SelectedReleaseFixture(tmp_path)
    calibration_path = Path(fixture.config["semantic_text"]["calibration"]["path"])
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    source = calibration.pop("source")
    calibration["source_path"] = source["path"]
    calibration["source_bytes"] = source["bytes"]
    calibration["source_sha256"] = source["sha256"]
    _write_json(calibration_path, calibration)

    config_path = fixture.release / "selection-config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["semantic_text"]["calibration"] = _binding(calibration_path)
    _write_json(config_path, config)
    fixture.reseal_child()

    with pytest.raises(ValueError, match="calibration schema is incomplete"):
        fixture.validate()


def test_floating_model_revision_is_rejected(tmp_path: Path) -> None:
    fixture = SelectedReleaseFixture(tmp_path)
    config_path = fixture.release / "selection-config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["classifier"]["model"]["revision"] = "main"
    _write_json(config_path, config)
    fixture.reseal_child()
    with pytest.raises(ValueError, match="revision"):
        fixture.validate()


def test_selected_row_requires_semantic_and_image_gate_evidence(tmp_path: Path) -> None:
    fixture = SelectedReleaseFixture(tmp_path)
    payload = fixture.release / "selected-payload.jsonl"
    rows = [json.loads(line) for line in payload.read_text(encoding="utf-8").splitlines()]
    rows[0].pop("semantic_text_checks")
    payload.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    fixture.reseal_child()
    with pytest.raises(ValueError, match="selected row fields"):
        fixture.validate()


def test_decoded_pixel_hash_is_recomputed_from_the_image(tmp_path: Path) -> None:
    fixture = SelectedReleaseFixture(tmp_path)
    payload = fixture.release / "selected-payload.jsonl"
    rows = [json.loads(line) for line in payload.read_text(encoding="utf-8").splitlines()]
    rows[0]["decoded_pixel_sha256"] = "f" * 64
    payload.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    fixture.reseal_child()
    with pytest.raises(ValueError, match="decoded image evidence differs"):
        fixture.validate()


def test_classifier_evidence_is_bound_to_the_assigned_stratum(tmp_path: Path) -> None:
    fixture = SelectedReleaseFixture(tmp_path)
    payload = fixture.release / "selected-payload.jsonl"
    rows = [json.loads(line) for line in payload.read_text(encoding="utf-8").splitlines()]
    rows[0]["classifier_check"]["stratum"] = "urban"
    payload.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    fixture.reseal_child()
    with pytest.raises(ValueError, match="classifier gate"):
        fixture.validate()


def test_every_source_stratum_cell_requires_four_train_two_validation(tmp_path: Path) -> None:
    fixture = SelectedReleaseFixture(tmp_path)
    payload = fixture.release / "selected-payload.jsonl"
    rows = [json.loads(line) for line in payload.read_text(encoding="utf-8").splitlines()]
    rows[0]["selected_split"] = "validation"
    rows[0]["fold"] = None
    payload.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    fixture.reseal_child()
    with pytest.raises(ValueError, match="source x stratum quota"):
        fixture.validate()


def test_every_source_stratum_cell_requires_all_four_folds(tmp_path: Path) -> None:
    fixture = SelectedReleaseFixture(tmp_path)
    payload = fixture.release / "selected-payload.jsonl"
    rows = [json.loads(line) for line in payload.read_text(encoding="utf-8").splitlines()]
    rows[0]["fold"] = 1
    payload.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    fixture.reseal_child()
    with pytest.raises(ValueError, match="one train row to each fold"):
        fixture.validate()


def test_selection_rank_must_follow_frozen_digest_order(tmp_path: Path) -> None:
    fixture = SelectedReleaseFixture(tmp_path)
    payload = fixture.release / "selected-payload.jsonl"
    rows = [json.loads(line) for line in payload.read_text(encoding="utf-8").splitlines()]
    rows[0]["selection_rank"], rows[1]["selection_rank"] = (
        rows[1]["selection_rank"],
        rows[0]["selection_rank"],
    )
    payload.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    fixture.reseal_child()
    with pytest.raises(ValueError, match="frozen SHA-256 digest order"):
        fixture.validate()


def test_two_sdxl_tokenizers_must_bind_distinct_files(tmp_path: Path) -> None:
    fixture = SelectedReleaseFixture(tmp_path)
    config_path = fixture.release / "selection-config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["tokenizers"][1]["model"] = copy.deepcopy(
        config["tokenizers"][0]["model"]
    )
    _write_json(config_path, config)
    fixture.reseal_child()
    with pytest.raises(ValueError, match="distinct frozen file manifests"):
        fixture.validate()


@pytest.mark.parametrize("value", [None, "unknown_tokenizer"])
def test_clip_tokenizer_binding_must_be_declared(tmp_path: Path, value) -> None:
    fixture = SelectedReleaseFixture(tmp_path)
    config_path = fixture.release / "selection-config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if value is None:
        config.pop("clip_tokenizer_id")
    else:
        config["clip_tokenizer_id"] = value
    _write_json(config_path, config)
    fixture.reseal_child()
    with pytest.raises(ValueError, match="CLIP tokenizer binding|config fields are incomplete"):
        fixture.validate()


def test_classifier_ties_must_fail_closed(tmp_path: Path) -> None:
    fixture = SelectedReleaseFixture(tmp_path)
    config_path = fixture.release / "selection-config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["classifier"]["tie_rule"] = "first_class"
    _write_json(config_path, config)
    fixture.reseal_child()
    with pytest.raises(ValueError, match="classifier ties must reject"):
        fixture.validate()


def test_release_id_binds_child_manifest_core() -> None:
    first = {"training_ready": True, "selected_payload": {"sha256": "1" * 64}}
    second = {"training_ready": True, "selected_payload": {"sha256": "2" * 64}}
    assert selected_release_id(first) != selected_release_id(second)
