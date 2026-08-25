import copy
import hashlib
import importlib.util
import json
import math
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_PIPELINE = ROOT / "eval-pipeline"
if str(EVAL_PIPELINE) not in sys.path:
    sys.path.insert(0, str(EVAL_PIPELINE))
spec = importlib.util.spec_from_file_location(
    "native_headroom_evaluator_test",
    EVAL_PIPELINE / "evaluate_scheduler_native_fixed_headroom_development.py",
)
evaluator = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = evaluator
spec.loader.exec_module(evaluator)


class SchedulerNativeFixedHeadroomEvaluatorTest(unittest.TestCase):
    PRIMARY = (
        "spectral_low_native",
        "spectral_mid_native",
        "spectral_high_native",
        "laplacian_native",
    )
    ABLATIONS = ("semantic_native_ablation", "freeu_native_ablation")
    SEEDS = (11, 29)
    PROMPTS = tuple(range(33))

    def test_repository_authorization_binds_frozen_evaluator_and_runtime(self):
        actions_path = (
            EVAL_PIPELINE
            / "configs"
            / "scheduler_native_fixed_headroom_actions_v1.yaml"
        ).resolve()
        authorization_path = (
            EVAL_PIPELINE
            / "configs"
            / "scheduler_native_fixed_headroom_evaluation_v1.yaml"
        )
        actions = yaml.safe_load(actions_path.read_text())
        source_manifest = actions["source_manifest"]
        hashes = {
            "actions": self.sha256(actions_path),
            "source_template": self.sha256(ROOT / actions["registration_source"]["path"]),
            "prompts": self.sha256(ROOT / source_manifest["prompts"]),
            "prompt_manifest": self.sha256(ROOT / source_manifest["path"]),
        }
        evaluator._validate_evaluation_authorization(
            yaml.safe_load(authorization_path.read_text()),
            actions_file=actions_path,
            hashes=hashes,
            actions_config=actions,
        )

    @staticmethod
    def sha256(path):
        return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

    def write_actions(self, root):
        root = pathlib.Path(root)
        prompts = root / "prompts.csv"
        pd.DataFrame(
            [
                {
                    "index": index,
                    "TEXT": f"registered prompt {index}",
                    "source_challenge": f"challenge_{index // 3:02d}",
                    "split": "development",
                }
                for index in self.PROMPTS
            ]
        ).to_csv(prompts, index=False)
        prompt_manifest = root / "prompt_manifest.json"
        prompt_manifest.write_text(json.dumps({"schema": "fixture"}) + "\n")
        source_template = root / "source_template.yaml"
        source_template.write_text("schema: fixture_template\n")

        actions = [
            {
                "id": "no_op",
                "type": "none",
                "role": "nominal_scheduler_baseline",
                "selection_eligible": False,
            },
            {
                "id": "lazy_zero_identity",
                "type": "latent_renderer_fixed",
                "role": "implementation_identity_control",
                "selection_eligible": False,
            },
        ]
        actions.extend(
            {
                "id": action,
                "type": "latent_renderer_fixed",
                "role": "non_attention_primary",
                "selection_eligible": True,
            }
            for action in self.PRIMARY
        )
        actions.extend(
            (
                {
                    "id": self.ABLATIONS[0],
                    "type": "latent_renderer_fixed",
                    "role": "attention_mechanism_ablation",
                    "selection_eligible": False,
                },
                {
                    "id": self.ABLATIONS[1],
                    "type": "latent_renderer_fixed",
                    "role": "decoder_feature_mechanism_ablation",
                    "selection_eligible": False,
                },
            )
        )
        payload = {
            "schema": evaluator.NATIVE_SCHEMA,
            "status": "authorized_development",
            "authorization": {
                "reviewer": "fixture",
                "method_selection": False,
            },
            "registration_source": {
                "path": str(source_template),
                "sha256": self.sha256(source_template),
            },
            "scoring": {
                "required_schema": "repldm_scorer_provenance_v1",
                "registered_scorer_provenance_sha256": "a" * 64,
            },
            "source_manifest": {
                "path": str(prompt_manifest),
                "sha256": self.sha256(prompt_manifest),
                "prompts": str(prompts),
                "prompts_sha256": self.sha256(prompts),
                "expected_prompt_count": len(self.PROMPTS),
            },
            "split_role": "development",
            "split_seeds": {"development": list(self.SEEDS)},
            "actions": actions,
            "analysis": {
                "baseline_action": "no_op",
                "identity_control": "lazy_zero_identity",
                "primary_metric": "topiq_nr",
                "minimum_practical_mean_delta": 0.005,
                "primary_holm_family": {
                    "alpha": 0.05,
                    "actions": list(self.PRIMARY),
                },
                "mechanism_ablation_family": {
                    "confirmatory_selection": False,
                    "holm_alpha": 0.05,
                    "actions": list(self.ABLATIONS),
                },
                "inference": {
                    "crossed_prompt_seed_bootstrap": 10000,
                    "confidence_level": 0.95,
                    "prompt_level_sign_flips": 100000,
                    "random_seed": 2026,
                },
                "noninferiority_guards": {
                    "hpsv2_ci_lower_min_delta": -0.005,
                    "clip_cosine_ci_lower_min_delta": -0.005,
                    "clipped_fraction_ci_upper_max_delta": 0.001,
                    "mean_saturation_ci_upper_max_delta": 0.005,
                    "contrast_geometric_ratio_interval": [0.95, 1.05],
                },
                "pass_rule": list(evaluator.EXPECTED_PASS_RULE),
                "null_rule": evaluator.EXPECTED_NULL_RULE,
                "ablation_rule": evaluator.EXPECTED_ABLATION_RULE,
            },
        }
        path = root / "actions.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False))
        return path, prompts, payload

    def frame(self, payload, *, primary_topiq_delta=0.01):
        rows = []
        baseline = {
            "topiq_nr": 0.50,
            "hpsv2": 0.30,
            "clip_cosine": 0.40,
            "clipped_fraction": 0.01,
            "mean_saturation": 0.50,
            "contrast_std": 0.20,
        }
        for prompt in self.PROMPTS:
            for seed in self.SEEDS:
                for action in payload["actions"]:
                    action_id = action["id"]
                    values = dict(baseline)
                    if action_id in self.PRIMARY:
                        values["topiq_nr"] += primary_topiq_delta
                    elif action_id in self.ABLATIONS:
                        values["topiq_nr"] += 0.10
                    rows.append(
                        {
                            "id": f"p{prompt}_seed{seed}_a{action_id}",
                            "prompt_index": prompt,
                            "seed": seed,
                            "action_id": action_id,
                            "device": "cuda:7",
                            "source_challenge": f"challenge_{prompt // 3:02d}",
                            **values,
                        }
                    )
        return pd.DataFrame(rows)

    def write_evaluation_authorization(self, root, actions_path, payload):
        root = pathlib.Path(root)
        source_manifest = payload["source_manifest"]
        authorization = {
            "schema": evaluator.AUTHORIZATION_SCHEMA,
            "status": "authorized_quantitative_screen",
            "scope": "development_only_quantitative_screen",
            "authorization": {
                "reviewers": ["plan_progress_audit", "literature_survey"],
                "result_access_before_freeze": False,
                "method_selection": False,
                "validation": False,
                "distillation": False,
                "rl": False,
            },
            "bindings": {
                "evaluator_path": str(pathlib.Path(evaluator.__file__).resolve()),
                "evaluator_sha256": self.sha256(evaluator.__file__),
                "auditor_path": str(evaluator.AUDITOR_PATH.resolve()),
                "auditor_sha256": self.sha256(evaluator.AUDITOR_PATH),
                "actions_path": str(actions_path.resolve()),
                "actions_sha256": self.sha256(actions_path),
                "source_template_sha256": payload["registration_source"]["sha256"],
                "prompts_sha256": source_manifest["prompts_sha256"],
                "prompt_manifest_sha256": source_manifest["sha256"],
                "scorer_provenance_sha256": payload["scoring"][
                    "registered_scorer_provenance_sha256"
                ],
                "evaluation_schema": evaluator.EVALUATION_SCHEMA,
            },
            "runtime": evaluator._runtime_versions(),
            "interpretation": copy.deepcopy(evaluator.EXPECTED_INTERPRETATION),
        }
        path = root / "evaluation_authorization.yaml"
        path.write_text(yaml.safe_dump(authorization, sort_keys=False))
        return path

    def write_verified_run(self, root, *, primary_topiq_delta=0.01):
        root = pathlib.Path(root)
        actions_path, prompts_path, payload = self.write_actions(root)
        evaluation_authorization = self.write_evaluation_authorization(
            root, actions_path, payload
        )
        run_dir = root / "run"
        run_dir.mkdir()
        frame = self.frame(payload, primary_topiq_delta=primary_topiq_delta)
        manifest_columns = ["id", "prompt_index", "seed", "action_id", "device"]
        manifest = frame[manifest_columns].to_dict("records")
        scores = frame[["id", *evaluator.METRICS]].to_dict("records")
        manifest_path = run_dir / "manifest.jsonl"
        scores_path = run_dir / "scores.jsonl"
        manifest_path.write_text("".join(json.dumps(row) + "\n" for row in manifest))
        scores_path.write_text("".join(json.dumps(row) + "\n" for row in scores))

        config = {
            "native_renderer_registered": True,
            "action_schema": evaluator.NATIVE_SCHEMA,
            "split_role": "development",
            "seeds": list(self.SEEDS),
            "actions_yaml": str(actions_path.resolve()),
            "actions_sha256": self.sha256(actions_path),
            "native_renderer_executable_actions_sha256": self.sha256(actions_path),
            "prompts_csv": str(prompts_path.resolve()),
            "prompts_sha256": self.sha256(prompts_path),
            "native_renderer_source_template_sha256": payload[
                "registration_source"
            ]["sha256"],
            "scorer_provenance_binding_required": True,
            "scoring": copy.deepcopy(payload["scoring"]),
        }
        config_path = run_dir / "config.json"
        config_path.write_text(json.dumps(config))
        audit = {
            "passed": True,
            "split_role": "development",
            "records": len(manifest),
            "prompts": len(self.PROMPTS),
            "seeds": list(self.SEEDS),
            "actions": [action["id"] for action in payload["actions"]],
            "required_score_keys": list(evaluator.METRICS),
            "scorer_provenance_sha256": payload["scoring"][
                "registered_scorer_provenance_sha256"
            ],
            "registered_identity_pair": ["no_op", "lazy_zero_identity"],
            "identity_pair_png_hashes_equal": True,
            "provenance": {
                "config_sha256": self.sha256(config_path),
                "manifest_sha256": self.sha256(manifest_path),
                "scores_sha256": self.sha256(scores_path),
                "prompts_sha256": self.sha256(prompts_path),
                "source_actions_sha256": self.sha256(actions_path),
                "source_template_sha256": payload["registration_source"][
                    "sha256"
                ],
            },
        }
        audit_path = run_dir / "run_audit.json"
        audit_path.write_text(json.dumps(audit))
        return (
            run_dir,
            actions_path,
            audit_path,
            evaluation_authorization,
            payload,
            frame,
        )

    def test_protocol_is_exactly_four_primary_actions_and_forbids_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, payload = self.write_actions(temporary)
            protocol = evaluator.load_protocol(payload)
            self.assertEqual(protocol.baseline, "no_op")
            self.assertEqual(protocol.primary_actions, self.PRIMARY)
            self.assertEqual(protocol.randomizations, 100000)

            selected = copy.deepcopy(payload)
            selected["authorization"]["method_selection"] = True
            with self.assertRaisesRegex(ValueError, "forbids method selection"):
                evaluator.load_protocol(selected)

            widened = copy.deepcopy(payload)
            widened["analysis"]["primary_holm_family"]["actions"].append(
                self.ABLATIONS[0]
            )
            with self.assertRaisesRegex(ValueError, "exactly the four"):
                evaluator.load_protocol(widened)

            retuned = copy.deepcopy(payload)
            retuned["analysis"]["inference"]["prompt_level_sign_flips"] = 99999
            with self.assertRaisesRegex(ValueError, "counts/seed"):
                evaluator.load_protocol(retuned)

    def test_quantitative_trigger_requires_independent_review_without_winner(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, payload = self.write_actions(temporary)
            protocol = evaluator.load_protocol(payload)
            result = evaluator.evaluate_frame(
                self.frame(payload),
                protocol,
                expected_prompt_indices=self.PROMPTS,
                expected_seeds=self.SEEDS,
            )
        self.assertEqual(result["decision"], "independent_review_required")
        self.assertTrue(result["qualitative_review_required"])
        self.assertFalse(result["method_selection_performed"])
        self.assertTrue(result["screen_only"])
        self.assertFalse(result["qualitative_review_authorized"])
        self.assertFalse(result["validation_authorized"])
        self.assertFalse(result["distillation_authorized"])
        self.assertFalse(result["rl_authorized"])
        self.assertNotIn("selected_action", result)
        self.assertFalse(result["claims"]["effect_at_least_threshold_established"])
        self.assertFalse(result["claims"]["population_generalization_established"])
        self.assertFalse(
            result["claims"]["guard_familywise_controlled_across_actions"]
        )
        primary = [row for row in result["rows"] if row["family"] == "primary_headroom"]
        ablations = [row for row in result["rows"] if row["family"] == "mechanism_ablation"]
        self.assertEqual(len(primary), 4)
        self.assertTrue(all(row["passes_headroom_gate"] for row in primary))
        self.assertTrue(
            all(len(row["topiq_mean_delta_by_challenge"]) == 11 for row in primary)
        )
        self.assertEqual(len(ablations), 2)
        self.assertTrue(all(row["inferential_use"] == "descriptive_only" for row in ablations))
        self.assertTrue(all(row["topiq_p_holm"] < 0.05 for row in ablations))
        self.assertTrue(all(row["passes_headroom_gate"] is None for row in ablations))

    def test_null_route_when_no_primary_meets_point_estimate_screen_threshold(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, payload = self.write_actions(temporary)
            protocol = evaluator.load_protocol(payload)
            result = evaluator.evaluate_frame(
                self.frame(payload, primary_topiq_delta=0.004),
                protocol,
                expected_prompt_indices=self.PROMPTS,
                expected_seeds=self.SEEDS,
            )
        self.assertEqual(result["decision"], "null_route")
        self.assertFalse(result["qualitative_review_required"])
        primary = [row for row in result["rows"] if row["family"] == "primary_headroom"]
        self.assertTrue(
            all(not row["topiq_point_estimate_screen_pass"] for row in primary)
        )

    def test_every_registered_gate_is_necessary_and_boundaries_are_inclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, payload = self.write_actions(temporary)
            protocol = evaluator.load_protocol(payload)
        passing = {
            "topiq_nr_mean_delta": 0.005,
            "topiq_nr_ci_low": 1e-12,
            "topiq_p_holm": 0.049,
            "hpsv2_ci_low": -0.005,
            "clip_cosine_ci_low": -0.005,
            "clipped_fraction_ci_high": 0.001,
            "mean_saturation_ci_high": 0.005,
            "contrast_ratio_ci_low": 0.95,
            "contrast_ratio_ci_high": 1.05,
        }
        evaluator._apply_primary_gate(passing, protocol)
        self.assertTrue(passing["passes_headroom_gate"])
        failures = {
            "topiq_point_estimate_screen_pass": (
                "topiq_nr_mean_delta",
                0.0049,
            ),
            "topiq_ci_pass": ("topiq_nr_ci_low", 0.0),
            "topiq_holm_pass": ("topiq_p_holm", 0.05),
            "hpsv2_noninferiority_pass": ("hpsv2_ci_low", -0.0051),
            "clip_cosine_noninferiority_pass": ("clip_cosine_ci_low", -0.0051),
            "clipped_fraction_guard_pass": ("clipped_fraction_ci_high", 0.0011),
            "mean_saturation_guard_pass": ("mean_saturation_ci_high", 0.0051),
            "contrast_guard_pass": ("contrast_ratio_ci_low", 0.9499),
        }
        for expected_check, (field, value) in failures.items():
            with self.subTest(check=expected_check):
                row = {key: value_ for key, value_ in passing.items() if not key.endswith("pass") and key != "passes_headroom_gate"}
                row[field] = value
                evaluator._apply_primary_gate(row, protocol)
                self.assertFalse(row[expected_check])
                self.assertFalse(row["passes_headroom_gate"])
        upper = {key: value for key, value in passing.items() if not key.endswith("pass") and key != "passes_headroom_gate"}
        upper["contrast_ratio_ci_high"] = 1.0501
        evaluator._apply_primary_gate(upper, protocol)
        self.assertFalse(upper["contrast_guard_pass"])

    def test_contrast_is_cellwise_log_ratio_and_requires_positive_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, payload = self.write_actions(temporary)
            protocol = evaluator.load_protocol(payload)
            frame = self.frame(payload)
            action = self.PRIMARY[0]
            mask = frame["action_id"] == action
            ratios = [0.96 if prompt % 2 == 0 else 1.04 for prompt in frame.loc[mask, "prompt_index"]]
            frame.loc[mask, "contrast_std"] *= ratios
            result = evaluator.evaluate_frame(
                frame,
                protocol,
                expected_prompt_indices=self.PROMPTS,
                expected_seeds=self.SEEDS,
            )
            row = next(value for value in result["rows"] if value["action"] == action)
            expected = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
            self.assertAlmostEqual(row["contrast_geometric_mean_ratio"], expected)

            frame.loc[mask, "contrast_std"] = 0.0
            with self.assertRaisesRegex(ValueError, "strictly positive"):
                evaluator.evaluate_frame(
                    frame,
                    protocol,
                    expected_prompt_indices=self.PROMPTS,
                    expected_seeds=self.SEEDS,
                )

    def test_passing_audit_and_all_input_hashes_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, actions, audit, authorization, _, _ = self.write_verified_run(
                temporary
            )
            verified = evaluator.load_verified_inputs(
                run_dir, actions, audit, authorization
            )
            self.assertEqual(verified.protocol.primary_actions, self.PRIMARY)
            audit_payload = json.loads(audit.read_text())

            audit_payload["passed"] = False
            audit.write_text(json.dumps(audit_payload))
            with self.assertRaisesRegex(ValueError, "passing development run audit"):
                evaluator.load_verified_inputs(run_dir, actions, audit, authorization)

            audit_payload["passed"] = True
            audit_payload["provenance"]["scores_sha256"] = "f" * 64
            audit.write_text(json.dumps(audit_payload))
            with self.assertRaisesRegex(ValueError, "scores_sha256"):
                evaluator.load_verified_inputs(run_dir, actions, audit, authorization)

    def test_evaluation_authorization_binds_code_runtime_and_screen_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, actions, audit, authorization, _, _ = self.write_verified_run(
                temporary
            )
            frozen = yaml.safe_load(authorization.read_text())

            tampered = copy.deepcopy(frozen)
            tampered["bindings"]["evaluator_sha256"] = "f" * 64
            authorization.write_text(yaml.safe_dump(tampered, sort_keys=False))
            with self.assertRaisesRegex(ValueError, "evaluator_sha256"):
                evaluator.load_verified_inputs(run_dir, actions, audit, authorization)

            tampered = copy.deepcopy(frozen)
            tampered["runtime"]["numpy"] = "0.0.0"
            authorization.write_text(yaml.safe_dump(tampered, sort_keys=False))
            with self.assertRaisesRegex(ValueError, "runtime versions"):
                evaluator.load_verified_inputs(run_dir, actions, audit, authorization)

            tampered = copy.deepcopy(frozen)
            tampered["interpretation"]["practical_threshold_inference"] = True
            authorization.write_text(yaml.safe_dump(tampered, sort_keys=False))
            with self.assertRaisesRegex(ValueError, "screen-only freeze"):
                evaluator.load_verified_inputs(run_dir, actions, audit, authorization)

    def test_cli_writes_one_shot_json_and_csv_without_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, actions, audit, authorization, _, _ = self.write_verified_run(
                temporary
            )
            argv = [
                "evaluate_scheduler_native_fixed_headroom_development.py",
                "--run-dir",
                str(run_dir),
                "--actions",
                str(actions),
                "--audit",
                str(audit),
                "--evaluation-authorization",
                str(authorization),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                evaluator, "_require_committed_inputs", return_value="fixture-commit"
            ):
                evaluator.main()
            result_path = run_dir / evaluator.OUTPUT_JSON
            csv_path = run_dir / evaluator.OUTPUT_CSV
            result = json.loads(result_path.read_text())
            self.assertEqual(result["decision"], "independent_review_required")
            self.assertNotIn("selected_action", result)
            self.assertEqual(len(pd.read_csv(csv_path)), 6)
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                evaluator, "_require_committed_inputs", return_value="fixture-commit"
            ):
                with self.assertRaisesRegex(ValueError, "one-shot"):
                    evaluator.main()


if __name__ == "__main__":
    unittest.main()
