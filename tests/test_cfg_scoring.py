import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import yaml
from PIL import Image


ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_PIPELINE = ROOT / "eval-pipeline"
if str(EVAL_PIPELINE) not in sys.path:
    sys.path.insert(0, str(EVAL_PIPELINE))
spec = importlib.util.spec_from_file_location(
    "cfg_scoring_test", EVAL_PIPELINE / "score.py"
)
score = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(score)


class CFGScoringTest(unittest.TestCase):
    def scoring_yaml(self):
        return {
            "schema": score.CFG_ACTION_SCHEMA,
            "scoring": {
                "metrics": list(score.CFG_SCORING_METRICS),
                "strict": True,
                "params": dict(score.CFG_SCORING_PARAMS),
                "required_score_keys": list(score.CFG_REQUIRED_SCORE_KEYS),
            },
        }

    def write_actions(self, root, payload=None):
        path = pathlib.Path(root) / "actions.yaml"
        path.write_text(
            yaml.safe_dump(payload or self.scoring_yaml(), sort_keys=False)
        )
        return path

    @staticmethod
    def file_sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def run_config(self, actions_path):
        return {
            "cfg_baseline_registered": True,
            "actions_yaml": str(actions_path),
            "actions_sha256": self.file_sha256(actions_path),
        }

    def test_contract_binds_frozen_recipe_and_actions_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            actions = self.write_actions(tmp)
            run_config = self.run_config(actions)
            contract = score.cfg_scoring_contract(
                run_config,
                score.CFG_SCORING_METRICS,
                score.CFG_SCORING_PARAMS,
                True,
            )
            self.assertEqual(contract["metrics"], list(score.CFG_SCORING_METRICS))
            self.assertEqual(contract["params"], score.CFG_SCORING_PARAMS)
            self.assertTrue(contract["strict"])
            self.assertEqual(contract["actions_sha256"], run_config["actions_sha256"])
            self.assertEqual(
                contract["required_score_keys"],
                list(score.CFG_REQUIRED_SCORE_KEYS),
            )

            with self.assertRaisesRegex(RuntimeError, "requires metrics"):
                score.cfg_scoring_contract(
                    run_config,
                    tuple(reversed(score.CFG_SCORING_METRICS)),
                    score.CFG_SCORING_PARAMS,
                    True,
                )
            with self.assertRaisesRegex(RuntimeError, "requires --strict"):
                score.cfg_scoring_contract(
                    run_config,
                    score.CFG_SCORING_METRICS,
                    score.CFG_SCORING_PARAMS,
                    False,
                )
            with self.assertRaisesRegex(RuntimeError, "config params"):
                score.cfg_scoring_contract(
                    run_config,
                    score.CFG_SCORING_METRICS,
                    {**score.CFG_SCORING_PARAMS, "patch_crops": 4},
                    True,
                )
            with self.assertRaisesRegex(RuntimeError, "config params"):
                score.cfg_scoring_contract(
                    run_config,
                    score.CFG_SCORING_METRICS,
                    {**score.CFG_SCORING_PARAMS, "patch_crops": 5.0},
                    True,
                )

            actions.write_text(actions.read_text() + "# changed after generation\n")
            with self.assertRaisesRegex(RuntimeError, "changed before scoring"):
                score.cfg_scoring_contract(
                    run_config,
                    score.CFG_SCORING_METRICS,
                    score.CFG_SCORING_PARAMS,
                    True,
                )

    def test_self_consistent_yaml_rewrites_do_not_redefine_v1(self):
        mutations = (
            (
                lambda value: value["scoring"].update(
                    {"metrics": ["pixel", "hps", "clip", "iqa"]}
                ),
                "YAML metrics",
            ),
            (
                lambda value: value["scoring"]["params"].update(
                    {"clipscore_w": 3.0}
                ),
                "YAML params",
            ),
            (
                lambda value: value["scoring"]["params"].update(
                    {"patch_crops": 5.0}
                ),
                "YAML params",
            ),
            (
                lambda value: value["scoring"].update(
                    {
                        "required_score_keys": list(
                            reversed(score.CFG_REQUIRED_SCORE_KEYS)
                        )
                    }
                ),
                "required score keys",
            ),
            (
                lambda value: value["scoring"].update({"unreviewed": True}),
                "fields differ",
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for mutate, message in mutations:
                with self.subTest(message=message):
                    payload = self.scoring_yaml()
                    mutate(payload)
                    actions = self.write_actions(tmp, payload)
                    run_config = self.run_config(actions)
                    configured_metrics = payload["scoring"]["metrics"]
                    configured_params = payload["scoring"]["params"]
                    with self.assertRaisesRegex(RuntimeError, message):
                        score.cfg_scoring_contract(
                            run_config,
                            configured_metrics,
                            configured_params,
                            True,
                        )

    def make_run(self, root):
        root = pathlib.Path(root)
        run_dir = root / "run"
        image_dir = run_dir / "images"
        image_dir.mkdir(parents=True)
        actions = self.write_actions(root)
        scoring_config = root / "scoring.yaml"
        scoring_config.write_text(
            yaml.safe_dump(
                {
                    "metrics": list(score.CFG_SCORING_METRICS),
                    "params": dict(score.CFG_SCORING_PARAMS),
                },
                sort_keys=False,
            )
        )
        image_path = image_dir / "p0_seed0_acfg_7p5.png"
        Image.new("RGB", (4, 4), color=(20, 30, 40)).save(image_path)
        image_hash = self.file_sha256(image_path)
        run_hash = "1" * 64
        row = {
            "id": "p0_seed0_acfg_7p5",
            "prompt_index": 0,
            "prompt": "a prompt",
            "bucket": "test",
            "seed": 0,
            "action_id": "cfg_7p5",
            "action_type": "none",
            "action_sha256": "2" * 64,
            "image_path": "images/p0_seed0_acfg_7p5.png",
            "image_sha256": image_hash,
            "run_contract_sha256": run_hash,
            "provenance_schema": score.PROVENANCE_SCHEMA,
        }
        (run_dir / "manifest.jsonl").write_text(json.dumps(row) + "\n")
        config = self.run_config(actions)
        config.update(
            {
                "trajectory_registered": False,
                "actions": [{"id": "cfg_7p5"}],
                "seeds": [0],
                "run_contract_sha256": run_hash,
            }
        )
        (run_dir / "config.json").write_text(json.dumps(config))
        contract = score.cfg_scoring_contract(
            config,
            score.CFG_SCORING_METRICS,
            score.CFG_SCORING_PARAMS,
            True,
        )
        return run_dir, scoring_config, row, contract

    @staticmethod
    def scorer_registry(calls, *, topiq=float("0.9")):
        outputs = {
            "pixel": {
                "colorfulness": 0.1,
                "laplacian_sharpness": 0.2,
                "mean_saturation": 0.3,
                "clipped_fraction": 0.0,
                "contrast_std": 0.4,
            },
            "clip": {"clip_cosine": 0.5, "clipscore": 1.25},
            "hps": {"hpsv2": 0.6},
            "iqa": {"topiq_nr": topiq},
        }
        registry = {}
        for metric_name, metric_outputs in outputs.items():
            output_keys = tuple((key, "higher") for key in metric_outputs)

            class DummyScorer:
                OUTPUT_KEYS = output_keys

                @classmethod
                def weights_status(cls, **params):
                    return True, ""

                def __init__(self, device="cpu", **params):
                    pass

                def score_image(self, image, prompt, _name=metric_name, _values=metric_outputs):
                    calls[_name] += 1
                    return dict(_values)

            registry[metric_name] = DummyScorer
        return registry, outputs

    def invoke_score(self, run_dir, scoring_config, registry):
        argv = [
            "score.py",
            "--run_dir",
            str(run_dir),
            "--config",
            str(scoring_config),
            "--device",
            "cpu",
            "--strict",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.dict(score.REGISTRY, registry, clear=True),
            mock.patch.object(
                score, "validate_run_contract", return_value="1" * 64
            ),
            mock.patch.object(score, "validate_sidecar"),
        ):
            score.main()

    @staticmethod
    def score_row(manifest_row, contract, values):
        row = {
            key: manifest_row[key]
            for key in (
                "id",
                "prompt_index",
                "seed",
                "action_id",
                "action_type",
                "action_sha256",
                "image_path",
                "image_sha256",
                "run_contract_sha256",
                "provenance_schema",
            )
        }
        for output in values.values():
            row.update(output)
        row["scoring_contract"] = contract
        row["scoring_contract_sha256"] = score.json_sha256(contract)
        return row

    def test_stale_scoring_contract_recomputes_all_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, config_path, manifest_row, contract = self.make_run(tmp)
            calls = {name: 0 for name in score.CFG_SCORING_METRICS}
            registry, values = self.scorer_registry(calls)
            existing = self.score_row(manifest_row, contract, values)
            existing["scoring_contract"] = {"schema": "stale"}
            existing["scoring_contract_sha256"] = score.json_sha256(
                existing["scoring_contract"]
            )
            (run_dir / "scores.jsonl").write_text(json.dumps(existing) + "\n")

            self.invoke_score(run_dir, config_path, registry)

            self.assertEqual(calls, {name: 1 for name in score.CFG_SCORING_METRICS})
            result = json.loads((run_dir / "scores.jsonl").read_text())
            self.assertEqual(result["scoring_contract"], contract)
            self.assertEqual(
                result["scoring_contract_sha256"], score.json_sha256(contract)
            )

    def test_registered_cfg_flag_requires_provenance_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, config_path, manifest_row, _ = self.make_run(tmp)
            manifest_row.pop("provenance_schema")
            (run_dir / "manifest.jsonl").write_text(
                json.dumps(manifest_row) + "\n"
            )
            calls = {name: 0 for name in score.CFG_SCORING_METRICS}
            registry, _ = self.scorer_registry(calls)

            with self.assertRaisesRegex(RuntimeError, "missing S7 provenance schema"):
                self.invoke_score(run_dir, config_path, registry)
            self.assertEqual(calls, {name: 0 for name in score.CFG_SCORING_METRICS})

    def test_stale_image_provenance_recomputes_all_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, config_path, manifest_row, contract = self.make_run(tmp)
            calls = {name: 0 for name in score.CFG_SCORING_METRICS}
            registry, values = self.scorer_registry(calls)
            existing = self.score_row(manifest_row, contract, values)
            existing["image_sha256"] = "0" * 64
            (run_dir / "scores.jsonl").write_text(json.dumps(existing) + "\n")

            self.invoke_score(run_dir, config_path, registry)

            self.assertEqual(calls, {name: 1 for name in score.CFG_SCORING_METRICS})
            result = json.loads((run_dir / "scores.jsonl").read_text())
            self.assertEqual(result["image_sha256"], manifest_row["image_sha256"])

    def test_nonfinite_existing_metric_is_recomputed_selectively(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, config_path, manifest_row, contract = self.make_run(tmp)
            calls = {name: 0 for name in score.CFG_SCORING_METRICS}
            registry, values = self.scorer_registry(calls)
            existing = self.score_row(manifest_row, contract, values)
            existing["topiq_nr"] = float("nan")
            (run_dir / "scores.jsonl").write_text(json.dumps(existing) + "\n")

            self.invoke_score(run_dir, config_path, registry)

            self.assertEqual(calls, {"pixel": 0, "clip": 0, "hps": 0, "iqa": 1})
            result = json.loads((run_dir / "scores.jsonl").read_text())
            self.assertEqual(result["topiq_nr"], 0.9)

    def test_new_nonfinite_metric_fails_strictly_without_serializing_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, config_path, _, _ = self.make_run(tmp)
            calls = {name: 0 for name in score.CFG_SCORING_METRICS}
            registry, _ = self.scorer_registry(calls, topiq=float("nan"))

            with self.assertRaisesRegex(RuntimeError, "iqa failed"):
                self.invoke_score(run_dir, config_path, registry)

            payload = (run_dir / "scores.jsonl").read_text()
            self.assertNotIn("NaN", payload)
            self.assertNotIn("topiq_nr", json.loads(payload))

    def test_output_lock_and_atomic_writer_protect_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            with score.scoring_output_lock(tmp):
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    with score.scoring_output_lock(tmp):
                        pass

            destination = pathlib.Path(tmp) / "scores.jsonl"
            destination.write_text("old\n")
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                with score.atomic_text_writer(destination) as handle:
                    handle.write("partial\n")
                    raise RuntimeError("interrupted")
            self.assertEqual(destination.read_text(), "old\n")
            self.assertEqual(list(pathlib.Path(tmp).glob(".scores.jsonl.*.tmp")), [])
            with score.atomic_text_writer(destination) as handle:
                handle.write("new\n")
            self.assertEqual(destination.read_text(), "new\n")


if __name__ == "__main__":
    unittest.main()
