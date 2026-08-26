import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import types
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
scorer_provenance = sys.modules["scorer_provenance"]


class CFGScoringTest(unittest.TestCase):
    def structural_gate_config(self, run_dir):
        config = {
            "structural_control_registered": True,
            "action_schema": "scheduler_native_structural_controls_actions_v1",
            "structural_control_registration_schema": (
                "scheduler_native_structural_controls_v1"
            ),
            "split_role": "development",
            "scorer_provenance_binding_required": True,
            "out_dir": str(pathlib.Path(run_dir).resolve()),
            "run_contract": {
                "action_schema": "scheduler_native_structural_controls_actions_v1"
            },
        }
        (pathlib.Path(run_dir) / "config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        return config

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
                PROVENANCE_PACKAGES = ()

                @classmethod
                def weights_status(cls, **params):
                    return True, ""

                def __init__(self, device="cpu", **params):
                    self.device = device

                def provenance_metadata(self):
                    return {
                        "models": [],
                        "checkpoint_files": [],
                        "preprocessing": {"fixture": "identity"},
                        "parameters": {},
                        "supporting_sources": [],
                    }

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
    def scorer_provenance(registry):
        active = [
            (name, registry[name](device="cpu", **score.CFG_SCORING_PARAMS))
            for name in score.CFG_SCORING_METRICS
        ]
        return score.build_scorer_provenance(
            active,
            params=score.CFG_SCORING_PARAMS,
            device="cpu",
            runner_path=EVAL_PIPELINE / "score.py",
            base_path=EVAL_PIPELINE / "scorers" / "base.py",
            source_root=EVAL_PIPELINE,
        )

    @staticmethod
    def score_row(manifest_row, contract, values, scorer_provenance):
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
        row["scorer_provenance"] = scorer_provenance[0]
        row["scorer_provenance_sha256"] = scorer_provenance[1]
        return row

    def test_stale_scoring_contract_recomputes_all_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, config_path, manifest_row, contract = self.make_run(tmp)
            calls = {name: 0 for name in score.CFG_SCORING_METRICS}
            registry, values = self.scorer_registry(calls)
            existing = self.score_row(
                manifest_row, contract, values, self.scorer_provenance(registry)
            )
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
            existing = self.score_row(
                manifest_row, contract, values, self.scorer_provenance(registry)
            )
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
            existing = self.score_row(
                manifest_row, contract, values, self.scorer_provenance(registry)
            )
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

    def test_generic_strict_recomputes_nonfinite_and_provenance_drift(self):
        for stale_kind in ("nonfinite", "plugin_source"):
            with self.subTest(stale_kind=stale_kind), tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                run_dir = root / "run"
                image_dir = run_dir / "images"
                image_dir.mkdir(parents=True)
                image_path = image_dir / "generic.png"
                Image.new("RGB", (4, 4), color=(10, 20, 30)).save(image_path)
                manifest_row = {
                    "id": "generic",
                    "prompt": "a generic prompt",
                    "image_path": "images/generic.png",
                }
                (run_dir / "manifest.jsonl").write_text(
                    json.dumps(manifest_row) + "\n"
                )
                config_path = root / "score.yaml"
                config_path.write_text(
                    yaml.safe_dump(
                        {
                            "metrics": ["iqa"],
                            "params": {},
                            "scorer_provenance": {
                                "required_schema": score.SCORER_PROVENANCE_SCHEMA
                            },
                        }
                    )
                )
                calls = {name: 0 for name in score.CFG_SCORING_METRICS}
                registry, values = self.scorer_registry(calls)
                active = [("iqa", registry["iqa"](device="cpu"))]
                contract, digest = score.build_scorer_provenance(
                    active,
                    params={},
                    device="cpu",
                    runner_path=EVAL_PIPELINE / "score.py",
                    base_path=EVAL_PIPELINE / "scorers" / "base.py",
                    source_root=EVAL_PIPELINE,
                )
                existing = {
                    "id": "generic",
                    "image_path": "images/generic.png",
                    "topiq_nr": values["iqa"]["topiq_nr"],
                    "scorer_provenance": contract,
                    "scorer_provenance_sha256": digest,
                }
                if stale_kind == "nonfinite":
                    existing["topiq_nr"] = float("nan")
                else:
                    contract["scorers"][0]["plugin_source"]["sha256"] = "0" * 64
                    existing["scorer_provenance_sha256"] = score.json_sha256(contract)
                (run_dir / "scores.jsonl").write_text(json.dumps(existing) + "\n")

                argv = [
                    "score.py",
                    "--run_dir",
                    str(run_dir),
                    "--config",
                    str(config_path),
                    "--device",
                    "cpu",
                    "--strict",
                ]
                with (
                    mock.patch.object(sys, "argv", argv),
                    mock.patch.dict(score.REGISTRY, {"iqa": registry["iqa"]}, clear=True),
                ):
                    score.main()

                result_text = (run_dir / "scores.jsonl").read_text()
                result = json.loads(result_text)
                self.assertNotIn("NaN", result_text)
                self.assertEqual(result["topiq_nr"], 0.9)
                self.assertEqual(calls["iqa"], 1)
                self.assertEqual(
                    result["scorer_provenance"]["schema"],
                    score.SCORER_PROVENANCE_SCHEMA,
                )

    def test_structural_scorer_gate_rejects_unknown_process_context(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            run_dir = pathlib.Path(temporary) / "run"
            run_dir.mkdir()
            config = self.structural_gate_config(run_dir)
            with mock.patch.object(
                scorer_provenance,
                "_STRUCTURAL_FORMAL_RUN_PATH",
                str(run_dir),
            ), mock.patch.object(
                sys, "argv", ["-c"]
            ), self.assertRaisesRegex(ValueError, "process context"):
                score.registered_scorer_provenance_contract(config)

    def test_structural_scorer_gate_uses_live_child_auth_once(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            run_dir = pathlib.Path(temporary) / "run"
            run_dir.mkdir()
            config = self.structural_gate_config(run_dir)
            child_auth = mock.Mock()
            static_attempt = mock.Mock()
            static_success = mock.Mock()
            fake_audit = types.SimpleNamespace(
                STRUCTURAL_CONTROL_ANALYSIS_AMENDMENT_PATH="candidate.yaml",
                STRUCTURAL_CONTROL_PRE_SCORE_SEAL_NAME="seal.json",
                require_scoring_child_authorization=child_auth,
                require_scoring_attempt_marker=static_attempt,
                require_scoring_success_receipt=static_success,
            )
            with mock.patch.object(
                scorer_provenance,
                "_STRUCTURAL_FORMAL_RUN_PATH",
                str(run_dir),
            ), mock.patch.object(
                sys,
                "argv",
                [str(EVAL_PIPELINE / "score.py"), "--fixed-fixture"],
            ), mock.patch.dict(
                sys.modules, {"audit_structural_control_run": fake_audit}
            ):
                self.assertEqual(
                    score.registered_scorer_provenance_contract(config),
                    (None, None),
                )
            child_auth.assert_called_once_with(
                run_dir.resolve(),
                analysis_amendment_path=ROOT / "candidate.yaml",
                pre_score_seal_path=run_dir.resolve() / "seal.json",
            )
            static_attempt.assert_not_called()
            static_success.assert_not_called()

    def test_structural_scorer_gate_allows_frozen_shared_config_then_authenticates_run(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            run_dir = pathlib.Path(temporary) / "run"
            run_dir.mkdir()
            run_config = self.structural_gate_config(run_dir)
            with (EVAL_PIPELINE / "configs" / "eval_common.yaml").open(
                encoding="utf-8"
            ) as handle:
                shared_config = yaml.safe_load(handle)
            child_auth = mock.Mock()
            fake_audit = types.SimpleNamespace(
                STRUCTURAL_CONTROL_ANALYSIS_AMENDMENT_PATH="candidate.yaml",
                STRUCTURAL_CONTROL_PRE_SCORE_SEAL_NAME="seal.json",
                require_scoring_child_authorization=child_auth,
                require_scoring_attempt_marker=mock.Mock(),
                require_scoring_success_receipt=mock.Mock(),
            )
            argv = [
                str(EVAL_PIPELINE / "score.py"),
                "--run_dir",
                str(run_dir),
            ]
            with mock.patch.object(
                scorer_provenance,
                "_STRUCTURAL_FORMAL_RUN_PATH",
                str(run_dir),
            ), mock.patch.object(sys, "argv", argv), mock.patch.dict(
                sys.modules, {"audit_structural_control_run": fake_audit}
            ):
                self.assertEqual(
                    score.registered_scorer_provenance_contract(shared_config),
                    (None, None),
                )
                child_auth.assert_not_called()
                self.assertEqual(
                    score.registered_scorer_provenance_contract(run_config),
                    (None, None),
                )
            child_auth.assert_called_once()

    def test_structural_scorer_gate_rejects_missing_signatures_and_signal_stripping(self):
        attacks = (
            "missing_registration_flag",
            "false_registration_flag",
            "missing_action_schema",
            "missing_registration_schema",
            "missing_run_contract_schema",
            "missing_out_dir",
            "missing_split_role",
            "missing_provenance_binding",
            "all_signals_removed",
        )
        for attack in attacks:
            with self.subTest(attack=attack), tempfile.TemporaryDirectory(
                dir=ROOT
            ) as temporary:
                run_dir = pathlib.Path(temporary) / "run"
                run_dir.mkdir()
                config = self.structural_gate_config(run_dir)
                if attack == "missing_registration_flag":
                    config.pop("structural_control_registered")
                elif attack == "false_registration_flag":
                    config["structural_control_registered"] = False
                elif attack == "missing_action_schema":
                    config.pop("action_schema")
                elif attack == "missing_registration_schema":
                    config.pop("structural_control_registration_schema")
                elif attack == "missing_run_contract_schema":
                    config["run_contract"] = {}
                elif attack == "missing_out_dir":
                    config.pop("out_dir")
                elif attack == "missing_split_role":
                    config.pop("split_role")
                elif attack == "missing_provenance_binding":
                    config.pop("scorer_provenance_binding_required")
                else:
                    for key in (
                        "structural_control_registered",
                        "action_schema",
                        "structural_control_registration_schema",
                        "out_dir",
                        "split_role",
                        "scorer_provenance_binding_required",
                    ):
                        config.pop(key)
                    config["run_contract"] = {}
                (run_dir / "config.json").write_text(
                    json.dumps(config), encoding="utf-8"
                )
                argv = [
                    str(EVAL_PIPELINE / "score.py"),
                    "--run_dir",
                    str(run_dir),
                ]
                with mock.patch.object(
                    scorer_provenance,
                    "_STRUCTURAL_FORMAL_RUN_PATH",
                    str(run_dir),
                ), mock.patch.object(sys, "argv", argv), self.assertRaisesRegex(
                    ValueError, "formal structural"
                ):
                    score.registered_scorer_provenance_contract(config)

    def test_structural_scorer_gate_rejects_stripped_run_after_shared_config(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            run_dir = pathlib.Path(temporary) / "run"
            run_dir.mkdir()
            stripped_run_config = {
                "split_role": "development",
                "scorer_provenance_binding_required": True,
                "run_contract": {},
            }
            (run_dir / "config.json").write_text(
                json.dumps(stripped_run_config), encoding="utf-8"
            )
            with (EVAL_PIPELINE / "configs" / "eval_common.yaml").open(
                encoding="utf-8"
            ) as handle:
                shared_config = yaml.safe_load(handle)
            argv = [
                str(EVAL_PIPELINE / "score.py"),
                "--run_dir",
                str(run_dir),
            ]
            with mock.patch.object(
                scorer_provenance,
                "_STRUCTURAL_FORMAL_RUN_PATH",
                str(run_dir),
            ), mock.patch.object(sys, "argv", argv):
                self.assertEqual(
                    score.registered_scorer_provenance_contract(shared_config),
                    (None, None),
                )
                with self.assertRaisesRegex(ValueError, "formal structural"):
                    score.registered_scorer_provenance_contract(stripped_run_config)

    def test_structural_scorer_gate_rejects_shared_config_as_run_config(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            run_dir = pathlib.Path(temporary) / "run"
            run_dir.mkdir()
            with (EVAL_PIPELINE / "configs" / "eval_common.yaml").open(
                encoding="utf-8"
            ) as handle:
                shared_config = yaml.safe_load(handle)
            (run_dir / "config.json").write_text(
                json.dumps(shared_config), encoding="utf-8"
            )
            argv = [
                str(EVAL_PIPELINE / "score.py"),
                "--run_dir",
                str(run_dir),
            ]
            with mock.patch.object(
                scorer_provenance,
                "_STRUCTURAL_FORMAL_RUN_PATH",
                str(run_dir),
            ), mock.patch.object(sys, "argv", argv), self.assertRaisesRegex(
                ValueError, "formal structural"
            ):
                score.registered_scorer_provenance_contract(shared_config)

    def test_structural_scorer_gate_requires_static_receipts_in_audit_contexts(self):
        for process_name in (
            "audit_structural_control_run.py",
            "evaluate_structural_control_run.py",
        ):
            with self.subTest(process_name=process_name), tempfile.TemporaryDirectory(
                dir=ROOT
            ) as temporary:
                run_dir = pathlib.Path(temporary) / "run"
                run_dir.mkdir()
                config = self.structural_gate_config(run_dir)
                child_auth = mock.Mock()
                static_attempt = mock.Mock()
                static_success = mock.Mock()
                fake_audit = types.SimpleNamespace(
                    STRUCTURAL_CONTROL_ANALYSIS_AMENDMENT_PATH="candidate.yaml",
                    STRUCTURAL_CONTROL_PRE_SCORE_SEAL_NAME="seal.json",
                    require_scoring_child_authorization=child_auth,
                    require_scoring_attempt_marker=static_attempt,
                    require_scoring_success_receipt=static_success,
                )
                with mock.patch.object(
                    scorer_provenance,
                    "_STRUCTURAL_FORMAL_RUN_PATH",
                    str(run_dir),
                ), mock.patch.object(
                    sys, "argv", [str(EVAL_PIPELINE / process_name)]
                ), mock.patch.dict(
                    sys.modules, {"audit_structural_control_run": fake_audit}
                ):
                    score.registered_scorer_provenance_contract(config)
                child_auth.assert_not_called()
                static_attempt.assert_called_once()
                static_success.assert_called_once()

    def test_structural_scorer_gate_allows_frozen_actions_then_authenticates_audit_run(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            run_dir = pathlib.Path(temporary) / "run"
            run_dir.mkdir()
            run_config = self.structural_gate_config(run_dir)
            with (
                EVAL_PIPELINE
                / "configs"
                / "scheduler_native_structural_controls_development_authorized_v1.yaml"
            ).open(encoding="utf-8") as handle:
                actions_config = yaml.safe_load(handle)
            static_attempt = mock.Mock()
            static_success = mock.Mock()
            fake_audit = types.SimpleNamespace(
                STRUCTURAL_CONTROL_ANALYSIS_AMENDMENT_PATH="candidate.yaml",
                STRUCTURAL_CONTROL_PRE_SCORE_SEAL_NAME="seal.json",
                require_scoring_child_authorization=mock.Mock(),
                require_scoring_attempt_marker=static_attempt,
                require_scoring_success_receipt=static_success,
            )
            with mock.patch.object(
                scorer_provenance,
                "_STRUCTURAL_FORMAL_RUN_PATH",
                str(run_dir),
            ), mock.patch.object(
                sys,
                "argv",
                [str(EVAL_PIPELINE / "audit_structural_control_run.py")],
            ), mock.patch.dict(
                sys.modules, {"audit_structural_control_run": fake_audit}
            ):
                self.assertEqual(
                    score.registered_scorer_provenance_contract(actions_config),
                    (None, actions_config["scoring"]["registered_scorer_provenance_sha256"]),
                )
                static_attempt.assert_not_called()
                static_success.assert_not_called()
                score.registered_scorer_provenance_contract(run_config)
            static_attempt.assert_called_once()
            static_success.assert_called_once()

    def test_structural_scorer_gate_redundant_signals_activate_unknown_context(self):
        signals = (
            {"schema": "scheduler_native_structural_controls_actions_v1"},
            {"registration_schema": "scheduler_native_structural_controls_v1"},
        )
        for registration in signals:
            with self.subTest(registration=registration), mock.patch.object(
                sys, "argv", ["-c"]
            ), self.assertRaisesRegex(ValueError, "process context"):
                score.registered_scorer_provenance_contract(registration)

    def test_native_renderer_development_provenance_flags_do_not_activate_gate(self):
        registration = {
            "native_renderer_registered": True,
            "split_role": "development",
            "scorer_provenance_binding_required": True,
        }
        with mock.patch.object(sys, "argv", ["-c"]):
            self.assertEqual(
                score.registered_scorer_provenance_contract(registration),
                (None, None),
            )

    def test_structural_scorer_gate_rejects_copied_run_and_config_drift(self):
        for attack in ("copied_run", "config_drift"):
            with self.subTest(attack=attack), tempfile.TemporaryDirectory(
                dir=ROOT
            ) as temporary:
                root = pathlib.Path(temporary)
                canonical_run = root / "canonical"
                selected_run = root / ("copy" if attack == "copied_run" else "canonical")
                canonical_run.mkdir()
                if selected_run != canonical_run:
                    selected_run.mkdir()
                config = self.structural_gate_config(selected_run)
                if attack == "config_drift":
                    config = {**config, "post_load_edit": True}
                with mock.patch.object(
                    scorer_provenance,
                    "_STRUCTURAL_FORMAL_RUN_PATH",
                    str(canonical_run),
                ), mock.patch.object(
                    sys, "argv", [str(EVAL_PIPELINE / "score.py")]
                ), self.assertRaisesRegex(
                    ValueError, "canonical non-symlink|changed after loading"
                ):
                    score.registered_scorer_provenance_contract(config)

    def test_nonstructural_scorer_contract_does_not_activate_gate(self):
        child_auth = mock.Mock()
        fake_audit = types.SimpleNamespace(
            STRUCTURAL_CONTROL_ANALYSIS_AMENDMENT_PATH="candidate.yaml",
            STRUCTURAL_CONTROL_PRE_SCORE_SEAL_NAME="seal.json",
            require_scoring_child_authorization=child_auth,
            require_scoring_attempt_marker=mock.Mock(),
            require_scoring_success_receipt=mock.Mock(),
        )
        with mock.patch.object(
            sys, "argv", [str(EVAL_PIPELINE / "score.py")]
        ), mock.patch.dict(
            sys.modules, {"audit_structural_control_run": fake_audit}
        ):
            self.assertEqual(
                score.registered_scorer_provenance_contract(
                    {"cfg_baseline_registered": True}
                ),
                (None, None),
            )
        child_auth.assert_not_called()
        fake_audit.require_scoring_attempt_marker.assert_not_called()
        fake_audit.require_scoring_success_receipt.assert_not_called()

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
