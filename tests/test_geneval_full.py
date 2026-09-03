from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval-pipeline"))

import geneval_full as geneval  # noqa: E402


HASH = "a" * 64
CHECKPOINT_HASH = "b" * 64
CONTRACT_HASH = "c" * 64
SEEDS = (2026090301, 2026090302, 2026090303, 2026090304)
SEED_COHORT = {
    "schema": geneval.SEED_COHORT_SCHEMA,
    "id": "geneval_shared_v1",
    "seeds": list(SEEDS),
    "sha256": geneval.seed_cohort_sha256("geneval_shared_v1", SEEDS),
}


@pytest.fixture(scope="module")
def full_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("geneval-full")
    run_dir = root / "run"
    run_dir.mkdir()
    prompt_manifest = geneval.load_prompt_manifest()
    image_dir = run_dir / "images"
    image_dir.mkdir()
    records: list[dict[str, object]] = []
    # The validator only needs the PNG signature, so tiny deterministic files
    # keep this contract test fast while still exercising every cell.
    pngs = [
        b"\x89PNG\r\n\x1a\nfixture-sample-0",
        b"\x89PNG\r\n\x1a\nfixture-sample-1",
        b"\x89PNG\r\n\x1a\nfixture-sample-2",
        b"\x89PNG\r\n\x1a\nfixture-sample-3",
    ]
    for prompt_index in range(geneval.EXPECTED_PROMPT_COUNT):
        prompt_spec = prompt_manifest["prompts"][prompt_index]
        for sample_index, seed in enumerate(SEEDS):
            image_path = image_dir / f"{prompt_index:05d}_{sample_index:04d}.png"
            image_path.write_bytes(pngs[sample_index])
            records.append(
                {
                    "prompt_index": prompt_index,
                    "sample_index": sample_index,
                    "seed": seed,
                    "prompt": prompt_spec["prompt"],
                    "tag": prompt_spec["tag"],
                    "image_path": str(image_path.relative_to(run_dir)),
                }
            )
    rows = geneval.build_input_manifest(
        records,
        prompt_manifest,
        run_dir=run_dir,
        checkpoint_id="checkpoint-c0",
        checkpoint_sha256=CHECKPOINT_HASH,
        method="opd",
        run_contract_sha256=CONTRACT_HASH,
        sample_seeds=SEEDS,
        seed_cohort=SEED_COHORT,
    )
    input_path = run_dir / "geneval" / "input_manifest.jsonl"
    input_hash = geneval.write_input_manifest(input_path, rows)
    layout_dir = run_dir / "geneval" / "layout"
    layout = geneval.materialize_official_layout(
        rows,
        prompt_manifest,
        run_dir=run_dir,
        layout_dir=layout_dir,
        input_manifest_sha256=input_hash,
        expected_seed_cohort=SEED_COHORT,
    )
    raw_rows: list[dict[str, object]] = []
    for row in rows:
        prompt_index = int(row["prompt_index"])
        sample_index = int(row["sample_index"])
        metadata = prompt_manifest["prompts"][prompt_index]["metadata"]
        raw_rows.append(
            {
                "filename": str(
                    layout_dir
                    / f"{prompt_index:05d}"
                    / "samples"
                    / f"{sample_index:04d}.png"
                ),
                "tag": row["tag"],
                "prompt": row["prompt"],
                "correct": (prompt_index + sample_index) % 3 != 0,
                "reason": "",
                "metadata": json.dumps(metadata, sort_keys=True),
                "details": "{}",
            }
        )
    evaluator = {
        "schema": geneval.EVALUATOR_SCHEMA,
        "script_path": "/opt/geneval/evaluate_images.py",
        "script_sha256": HASH,
        "python_path": "/opt/geneval/bin/python",
        "python_sha256": HASH,
        "model_path": "/opt/geneval/models",
        "model_tree_sha256": HASH,
        "gpu_id": 0,
    }
    return {
        "root": root,
        "run_dir": run_dir,
        "manifest": prompt_manifest,
        "records": records,
        "rows": rows,
        "input_hash": input_hash,
        "layout_dir": layout_dir,
        "layout": layout,
        "raw_rows": raw_rows,
        "evaluator": evaluator,
    }


def test_official_manifest_has_frozen_full_contract() -> None:
    manifest = geneval.load_prompt_manifest()
    assert manifest["source_sha256"] == geneval.DEFAULT_METADATA_SHA256
    assert manifest["source_rows"] == 2212
    assert manifest["prompt_count"] == 553
    assert manifest["samples_per_prompt"] == 4
    assert manifest["tag_counts"] == geneval.EXPECTED_TAG_COUNTS


@pytest.mark.parametrize("field", ("metadata_path", "metadata_sha256"))
def test_config_cannot_rebind_official_metadata(field: str) -> None:
    config, _config_hash = geneval._load_config(geneval.DEFAULT_CONFIG_PATH)
    altered = copy.deepcopy(config)
    altered["benchmark"][field] = (
        "/tmp/lookalike-geneval.jsonl" if field == "metadata_path" else "a" * 64
    )
    with pytest.raises(ValueError, match="frozen official metadata"):
        geneval._validate_config_contract(altered)


def test_config_requires_a_self_consistent_shared_seed_cohort() -> None:
    config, _config_hash = geneval._load_config(geneval.DEFAULT_CONFIG_PATH)
    altered = copy.deepcopy(config)
    altered["benchmark"]["seed_cohort"]["sha256"] = "a" * 64
    with pytest.raises(ValueError, match="seed_cohort.sha256"):
        geneval._validate_config_contract(altered)


def test_cli_metadata_cannot_bypass_official_source(tmp_path: Path) -> None:
    alternate = tmp_path / "same-shape-metadata.jsonl"
    alternate.write_text("{}\n", encoding="utf-8")
    args = argparse.Namespace(config=geneval.DEFAULT_CONFIG_PATH, metadata=alternate)
    with pytest.raises(ValueError, match="cannot replace the frozen official"):
        geneval._load_cli_contract(args)


def test_metadata_shuffle_and_missing_rows_are_rejected(tmp_path: Path) -> None:
    source = geneval.DEFAULT_METADATA_PATH.read_text(encoding="utf-8").splitlines()
    shuffled = source[:]
    shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
    shuffled_path = tmp_path / "shuffled.jsonl"
    shuffled_path.write_text("\n".join(shuffled) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source row order"):
        geneval.load_prompt_manifest(shuffled_path, expected_sha256=None)

    missing_path = tmp_path / "missing.jsonl"
    missing_path.write_text("\n".join(source[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="2212 rows"):
        geneval.load_prompt_manifest(missing_path, expected_sha256=None)


def test_input_manifest_rejects_incomplete_and_mixed_identity(full_run: dict[str, object]) -> None:
    run_dir = full_run["run_dir"]
    manifest = full_run["manifest"]
    records = full_run["records"]
    assert isinstance(run_dir, Path)
    assert isinstance(records, list)
    with pytest.raises(ValueError, match="incomplete"):
        geneval.build_input_manifest(
            records[:-1],
            manifest,
            run_dir=run_dir,
            checkpoint_id="checkpoint-c0",
            checkpoint_sha256=CHECKPOINT_HASH,
            method="opd",
            run_contract_sha256=CONTRACT_HASH,
            sample_seeds=SEEDS,
        )
    mixed = copy.deepcopy(records)
    mixed[0]["checkpoint_id"] = "checkpoint-other"
    with pytest.raises(ValueError, match="mixed checkpoints"):
        geneval.build_input_manifest(
            mixed,
            manifest,
            run_dir=run_dir,
            checkpoint_id="checkpoint-c0",
            checkpoint_sha256=CHECKPOINT_HASH,
            method="opd",
            run_contract_sha256=CONTRACT_HASH,
            sample_seeds=SEEDS,
        )


def test_raw_results_require_complete_cells_and_matching_layout(full_run: dict[str, object]) -> None:
    rows = full_run["rows"]
    manifest = full_run["manifest"]
    layout_dir = full_run["layout_dir"]
    assert isinstance(rows, list)
    assert isinstance(manifest, dict)
    assert isinstance(layout_dir, Path)
    raw_rows = full_run["raw_rows"]
    assert isinstance(raw_rows, list)
    kwargs = {
        "input_rows": rows,
        "prompt_manifest": manifest,
        "layout_dir": layout_dir,
        "evaluator": full_run["evaluator"],
        "run_dir": full_run["run_dir"],
    }
    with pytest.raises(ValueError, match="incomplete"):
        geneval.normalize_results(raw_rows[:-1], **kwargs)
    with pytest.raises(ValueError, match="duplicate"):
        geneval.normalize_results([*raw_rows, raw_rows[0]], **kwargs)
    outside = copy.deepcopy(raw_rows)
    outside[0]["filename"] = "/tmp/00000/samples/0000.png"
    with pytest.raises(ValueError, match="unknown image"):
        geneval.normalize_results(outside, **kwargs)


def test_aggregate_result_is_prompt_clustered_and_hash_sealed(full_run: dict[str, object]) -> None:
    run_dir = full_run["run_dir"]
    scores_path = run_dir / "geneval" / "scores.jsonl"
    summary = geneval.aggregate_raw_results(
        full_run["raw_rows"],
        full_run["rows"],
        full_run["manifest"],
        run_dir=run_dir,
        layout_dir=full_run["layout_dir"],
        input_manifest_sha256=full_run["input_hash"],
        raw_results_sha256=hashlib.sha256(b"raw-evaluator-output").hexdigest(),
        config_sha256=HASH,
        evaluator=full_run["evaluator"],
        scores_path=scores_path,
        bootstrap_seed=7,
        bootstrap_resamples=20,
        expected_seed_cohort=SEED_COHORT,
    )
    assert summary["counts"]["images"] == 2212
    assert summary["bootstrap"]["unit"] == "prompt"
    assert summary["scores_sha256"] == geneval.sha256_file(scores_path)
    assert geneval.validate_summary(summary, require_sealed=True)["summary_sha256"] == summary["summary_sha256"]

    tampered = copy.deepcopy(summary)
    tampered["counts"]["images"] = 1
    with pytest.raises(ValueError, match="counts"):
        geneval.validate_summary(tampered, require_sealed=True)

    tampered_cohort = copy.deepcopy(summary)
    tampered_cohort["seed_cohort"] = {
        "schema": geneval.SEED_COHORT_SCHEMA,
        "id": "geneval_shared_v2",
        "seeds": [11, 29, 101, 303],
        "sha256": geneval.seed_cohort_sha256("geneval_shared_v2", [11, 29, 101, 303]),
    }
    tampered_core = {
        key: value for key, value in tampered_cohort.items() if key != "summary_sha256"
    }
    tampered_cohort["summary_sha256"] = geneval.sha256_bytes(
        geneval.canonical_json(tampered_core)
    )
    with pytest.raises(ValueError, match="not registered"):
        geneval.validate_summary(tampered_cohort, require_sealed=True)


def test_seed_values_must_use_the_reviewed_shared_cohort_registration() -> None:
    config, _config_hash = geneval._load_config(geneval.DEFAULT_CONFIG_PATH)
    alternate_seeds = [11, 29, 101, 303]
    alternate = copy.deepcopy(config)
    alternate["benchmark"]["sample_seeds"] = alternate_seeds
    alternate["benchmark"]["seed_cohort"] = {
        "schema": geneval.SEED_COHORT_SCHEMA,
        "id": "geneval_shared_v2",
        "seeds": alternate_seeds,
        "sha256": geneval.seed_cohort_sha256("geneval_shared_v2", alternate_seeds),
    }
    with pytest.raises(ValueError, match="not registered"):
        geneval._validate_config_contract(alternate)


def test_new_cohort_requires_a_new_registered_id_and_revision() -> None:
    config, _config_hash = geneval._load_config(geneval.DEFAULT_CONFIG_PATH)
    alternate = copy.deepcopy(config)
    alternate["benchmark"]["seed_cohort"]["id"] = "geneval_shared_v2"
    alternate["benchmark"]["sample_seeds"] = [11, 29, 101, 303]
    alternate["benchmark"]["seed_cohort"]["seeds"] = [11, 29, 101, 303]
    alternate["benchmark"]["seed_cohort"]["sha256"] = geneval.seed_cohort_sha256(
        "geneval_shared_v2", [11, 29, 101, 303]
    )
    with pytest.raises(ValueError, match="not registered"):
        geneval._validate_config_contract(alternate)


def test_input_manifest_cannot_switch_or_drop_shared_seed_cohort(full_run: dict[str, object]) -> None:
    rows = copy.deepcopy(full_run["rows"])
    manifest = full_run["manifest"]
    run_dir = full_run["run_dir"]
    assert isinstance(rows, list)
    assert isinstance(manifest, dict)
    assert isinstance(run_dir, Path)

    switched = copy.deepcopy(rows)
    alternate_seeds = (11, 29, 101, 303)
    alternate_cohort = {
        "schema": geneval.SEED_COHORT_SCHEMA,
        "id": "geneval_shared_v2",
        "seeds": list(alternate_seeds),
        "sha256": geneval.seed_cohort_sha256("geneval_shared_v2", alternate_seeds),
    }
    for row in switched:
        row["seed"] = alternate_seeds[row["sample_index"]]
        row["seed_cohort_id"] = alternate_cohort["id"]
        row["seed_cohort_sha256"] = alternate_cohort["sha256"]
    with pytest.raises(ValueError, match="seed cohort"):
        geneval._validate_input_rows(
            switched,
            manifest,
            run_dir=run_dir,
            expected_seed_cohort=SEED_COHORT,
        )
    assert geneval._validate_input_rows(
        switched,
        manifest,
        run_dir=run_dir,
        expected_seed_cohort=alternate_cohort,
    )

    missing = copy.deepcopy(rows)
    missing[0].pop("seed_cohort_id")
    missing[0].pop("seed_cohort_sha256")
    with pytest.raises(ValueError, match="seed cohort"):
        geneval._validate_input_rows(
            missing,
            manifest,
            run_dir=run_dir,
            expected_seed_cohort=SEED_COHORT,
        )


def test_formal_evaluator_registry_rejects_path_replacement(monkeypatch: pytest.MonkeyPatch) -> None:
    registered = {
        "python_path": "/trusted/python",
        "script_path": "/trusted/evaluate.py",
        "model_path": "/trusted/model",
    }
    monkeypatch.setattr(geneval, "_registered_evaluator", lambda _id="official_v1": dict(registered))
    args = argparse.Namespace(
        evaluator_id="official_v1",
        evaluator_python=Path("/attacker/python"),
        evaluator_script=Path("/trusted/evaluate.py"),
        model_path=Path("/trusted/model"),
        model_config=None,
    )
    with pytest.raises(ValueError, match="python_path is not the registered"):
        geneval._evaluator_from_args(args)


def test_formal_evaluator_registry_rejects_model_config_override(monkeypatch: pytest.MonkeyPatch) -> None:
    registered = {
        "python_path": "/trusted/python",
        "script_path": "/trusted/evaluate.py",
        "model_path": "/trusted/model",
    }
    monkeypatch.setattr(geneval, "_registered_evaluator", lambda _id="official_v1": dict(registered))
    args = argparse.Namespace(
        evaluator_id="official_v1",
        evaluator_python=Path("/trusted/python"),
        evaluator_script=Path("/trusted/evaluate.py"),
        model_path=Path("/trusted/model"),
        model_config=Path("/attacker/config.yaml"),
    )
    with pytest.raises(ValueError, match="model-config is not supported"):
        geneval._evaluator_from_args(args)
