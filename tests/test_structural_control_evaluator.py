import copy
import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_PIPELINE = ROOT / "eval-pipeline"
REGISTRATION = (
    EVAL_PIPELINE
    / "configs"
    / "scheduler_native_structural_controls_development_registration_v1.yaml"
)
PROMPTS = (
    EVAL_PIPELINE / "prompts" / "scheduler_native_fixed_headroom_development.csv"
)
if str(EVAL_PIPELINE) not in sys.path:
    sys.path.insert(0, str(EVAL_PIPELINE))
spec = importlib.util.spec_from_file_location(
    "structural_control_evaluator_test",
    EVAL_PIPELINE / "evaluate_structural_control_run.py",
)
evaluator = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = evaluator
spec.loader.exec_module(evaluator)


class StructuralControlEvaluatorTest(unittest.TestCase):
    PROMPT_INDICES = tuple(range(33))
    SEEDS = (1932556753, 1065503757, 201635682)

    @staticmethod
    def sha256(path):
        return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

    def authorized_payload(self):
        payload = yaml.safe_load(REGISTRATION.read_text(encoding="utf-8"))
        registration_hash = self.sha256(REGISTRATION)
        payload["schema"] = evaluator.EXECUTABLE_SCHEMA
        payload["status"] = "authorized_development"
        payload.pop("blocking_conditions", None)
        payload["authorization"] = {
            "reviewer": "fixture_reviewer",
            "reviewed_commit": "a" * 40,
            "source_template": str(REGISTRATION.relative_to(ROOT)),
            "source_template_sha256": registration_hash,
            "scope": evaluator.AUTH_SCOPE,
            "gpu_generation": True,
            "scoring": True,
            "method_selection": False,
            "result_access_before_freeze": False,
        }
        payload["registration_source"] = {
            "path": str(REGISTRATION.relative_to(ROOT)),
            "sha256": registration_hash,
        }
        payload["implementation_source"] = {
            "reviewed_commit": "a" * 40,
            "files": {"fixture.py": "b" * 64},
        }
        return payload

    def frame(self, payload=None):
        payload = payload or self.authorized_payload()
        offsets = {
            "no_op_cfg7p5": 0.00,
            "cfg_only_5": 0.00,
            "conference_tfsa": 0.01,
            "freeu_diffusers_historical": 0.01,
            "freeu_diffusers_paper_parameters": 0.02,
            "freeu_paper_adaptive": 0.03,
            "pladis_operator_port": 0.01,
            "gag_eq13_reimplementation": 0.01,
        }
        baseline = {
            "topiq_nr": 0.50,
            "hpsv2": 0.30,
            "clip_cosine": 0.40,
            "clipped_fraction": 0.01,
            "mean_saturation": 0.50,
            "colorfulness": 30.0,
            "laplacian_sharpness": 100.0,
            "contrast_std": 0.20,
        }
        rows = []
        for prompt in self.PROMPT_INDICES:
            for seed in self.SEEDS:
                for action_index, action in enumerate(payload["actions"]):
                    action_id = action["id"]
                    values = dict(baseline)
                    values["topiq_nr"] += offsets[action_id]
                    rows.append(
                        {
                            "id": f"p{prompt}_seed{seed}_a{action_id}",
                            "prompt_index": prompt,
                            "seed": seed,
                            "action_id": action_id,
                            "device": "cuda:7",
                            "source_challenge": f"challenge_{prompt // 3:02d}",
                            "inference_seconds": 2.0 + action_index * 0.1,
                            "peak_gpu_memory_bytes": 1_000_000 + action_index * 1000,
                            "num_inference_steps": 50,
                            "guidance_scale": action["cfg_scale"],
                            "extra_unet_calls": 0,
                            "unet_calls_per_step": [1] * 50,
                            **values,
                        }
                    )
        return pd.DataFrame(rows)

    @staticmethod
    def mean_ci(values, **_kwargs):
        mean = float(values.mean())
        return mean, mean

    def test_protocol_freezes_three_holm_families_and_rejects_selection(self):
        payload = self.authorized_payload()
        protocol = evaluator.load_protocol(payload)
        self.assertEqual(protocol.action_order, evaluator.EXPECTED_ACTIONS)
        family_counts = {
            family: sum(contrast.family == family for contrast in protocol.contrasts)
            for family in evaluator.EXPECTED_FAMILY_SPECS
        }
        self.assertEqual(
            family_counts,
            {
                "freeu_vs_no_op": 3,
                "freeu_mechanism_contrasts": 2,
                "attention_vs_cfg5": 2,
            },
        )
        self.assertEqual(protocol.bootstrap, 10000)
        self.assertEqual(protocol.randomizations, 100000)

        blocked = yaml.safe_load(REGISTRATION.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ValueError, "authorized executable schema"):
            evaluator.load_protocol(blocked)

        selected = copy.deepcopy(payload)
        selected["authorization"]["method_selection"] = True
        with self.assertRaisesRegex(ValueError, "forbids method selection"):
            evaluator.load_protocol(selected)

        smoke_drift = copy.deepcopy(payload)
        smoke_drift["engineering_smoke"]["quality_scoring"] = True
        with self.assertRaisesRegex(ValueError, "engineering smoke"):
            evaluator.load_protocol(smoke_drift)

        wrong_unit = copy.deepcopy(payload)
        wrong_unit["analysis"]["inference"]["generalization_unit"] = "prompt_seed_cell"
        with self.assertRaisesRegex(ValueError, "prompt_after_seed_mean"):
            evaluator.load_protocol(wrong_unit)

        family_drift = copy.deepcopy(payload)
        family_drift["analysis"]["multiplicity_families"]["attention_vs_cfg5"][
            "reference"
        ] = "no_op_cfg7p5"
        with self.assertRaisesRegex(ValueError, "Holm families"):
            evaluator.load_protocol(family_drift)

    def test_crossed_bootstrap_preserves_registered_prompt_seed_shape(self):
        index = pd.MultiIndex.from_product(
            [self.PROMPT_INDICES, self.SEEDS], names=["prompt_index", "seed"]
        )
        values = pd.Series(np.full(len(index), 0.0125), index=index)
        low, high = evaluator._crossed_bootstrap_ci(
            values, n_boot=10000, confidence_level=0.95, seed=2026
        )
        self.assertAlmostEqual(low, 0.0125)
        self.assertAlmostEqual(high, 0.0125)

        incomplete = values.drop(index[0])
        with self.assertRaisesRegex(ValueError, "complete prompt x seed"):
            evaluator._crossed_bootstrap_ci(
                incomplete, n_boot=10000, confidence_level=0.95, seed=2026
            )

    def test_complete_report_has_no_selection_and_all_registered_diagnostics(self):
        payload = self.authorized_payload()
        protocol = evaluator.load_protocol(payload)
        pvalues = [0.0001] * 8
        with mock.patch.object(
            evaluator, "_crossed_bootstrap_ci", side_effect=self.mean_ci
        ), mock.patch.object(
            evaluator, "prompt_sign_flip_pvalue", side_effect=pvalues
        ):
            result = evaluator.evaluate_frame(
                self.frame(payload),
                protocol,
                expected_prompt_indices=self.PROMPT_INDICES,
                expected_seeds=self.SEEDS,
            )

        self.assertEqual(result["scope"], "development_evidence_reporting_only")
        self.assertFalse(result["method_selection_authorized"])
        self.assertFalse(result["method_selection_performed"])
        self.assertFalse(result["rl_authorized"])
        self.assertFalse(result["publication_superiority_established"])
        self.assertNotIn("selected_action", result)
        self.assertEqual(result["task_accounting"]["expected_tasks"], 792)
        self.assertEqual(result["task_accounting"]["observed_manifest_score_pairs"], 792)
        self.assertEqual(len(result["action_summaries"]), 8)
        self.assertTrue(
            all(summary["observed_tasks"] == 99 for summary in result["action_summaries"])
        )
        self.assertTrue(
            all(
                summary["branch_equivalent_nfe_per_task"] == 100
                for summary in result["action_summaries"]
            )
        )
        self.assertEqual(len(result["contrasts"]), 8)
        inferential = [
            row
            for row in result["contrasts"]
            if row["inferential_use"] == "development_screen_only"
        ]
        self.assertEqual(len(inferential), 7)
        self.assertTrue(all(row["all_quality_guards_pass"] for row in inferential))
        self.assertTrue(
            all(row["passes_registered_development_screen"] for row in inferential)
        )
        for row in result["contrasts"]:
            self.assertEqual(row["sign_flip_unit"], "prompt_after_seed_mean")
            self.assertEqual(len(row["per_seed_mean_delta"]), 3)
            self.assertEqual(len(row["leave_one_seed_out_mean_delta"]), 3)
            self.assertEqual(len(row["leave_one_prompt_out_mean_delta"]), 33)
            self.assertEqual(len(row["challenge_mean_delta"]), 11)
            self.assertIn("prompt_median_delta", row)
            self.assertIn("prompt_win_probability", row)
            self.assertEqual(len(row["quality_guards"]), 5)
            self.assertFalse(row["effect_at_least_point_threshold_established"])

    def test_holm_is_applied_within_each_registered_family_only(self):
        payload = self.authorized_payload()
        protocol = evaluator.load_protocol(payload)
        raw = [0.01, 0.03, 0.04, 0.01, 0.04, 0.03, 0.04, 0.50]
        with mock.patch.object(
            evaluator, "_crossed_bootstrap_ci", side_effect=self.mean_ci
        ), mock.patch.object(
            evaluator, "prompt_sign_flip_pvalue", side_effect=raw
        ):
            result = evaluator.evaluate_frame(
                self.frame(payload),
                protocol,
                expected_prompt_indices=self.PROMPT_INDICES,
                expected_seeds=self.SEEDS,
            )
        by_family = {
            family: [
                row["topiq_p_holm"]
                for row in result["contrasts"]
                if row["family"] == family
            ]
            for family in evaluator.EXPECTED_FAMILY_SPECS
        }
        np.testing.assert_allclose(by_family["freeu_vs_no_op"], [0.03, 0.06, 0.06])
        np.testing.assert_allclose(
            by_family["freeu_mechanism_contrasts"], [0.02, 0.04]
        )
        np.testing.assert_allclose(by_family["attention_vs_cfg5"], [0.06, 0.06])
        descriptive = next(
            row
            for row in result["contrasts"]
            if row["family"] == "descriptive_contrasts"
        )
        self.assertIsNone(descriptive["topiq_p_holm"])
        self.assertIsNone(descriptive["passes_registered_development_screen"])

    def test_every_quality_guard_is_reported_and_jointly_required(self):
        payload = self.authorized_payload()
        protocol = evaluator.load_protocol(payload)
        frame = self.frame(payload)
        historical = frame["action_id"] == "freeu_diffusers_historical"
        frame.loc[historical, "hpsv2"] -= 0.01
        with mock.patch.object(
            evaluator, "_crossed_bootstrap_ci", side_effect=self.mean_ci
        ), mock.patch.object(
            evaluator, "prompt_sign_flip_pvalue", return_value=0.0001
        ):
            result = evaluator.evaluate_frame(
                frame,
                protocol,
                expected_prompt_indices=self.PROMPT_INDICES,
                expected_seeds=self.SEEDS,
            )
        row = next(
            value
            for value in result["contrasts"]
            if value["contrast_id"]
            == "freeu_diffusers_historical_vs_no_op_cfg7p5"
        )
        self.assertFalse(row["hpsv2_noninferiority_pass"])
        self.assertFalse(row["all_quality_guards_pass"])
        self.assertFalse(row["passes_registered_development_screen"])
        self.assertEqual(
            set(row["quality_guards"]),
            {
                "hpsv2_noninferiority_pass",
                "clip_cosine_noninferiority_pass",
                "clipped_fraction_guard_pass",
                "mean_saturation_guard_pass",
                "contrast_guard_pass",
            },
        )

    def test_missing_nonfinite_or_nfe_drift_fails_closed(self):
        payload = self.authorized_payload()
        protocol = evaluator.load_protocol(payload)
        cases = []
        missing = self.frame(payload).iloc[:-1].copy()
        cases.append((missing, "registered prompt x seed x action grid"))
        nonfinite = self.frame(payload)
        nonfinite.loc[0, "topiq_nr"] = np.nan
        cases.append((nonfinite, "non-finite"))
        nfe_drift = self.frame(payload)
        nfe_drift.at[0, "unet_calls_per_step"] = [1] * 49 + [2]
        cases.append((nfe_drift, "one U-Net call"))
        cross_device = self.frame(payload)
        cross_device.loc[0, "device"] = "cuda:6"
        cases.append((cross_device, "span multiple devices"))

        for frame, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    evaluator.evaluate_frame(
                        frame,
                        protocol,
                        expected_prompt_indices=self.PROMPT_INDICES,
                        expected_seeds=self.SEEDS,
                    )

    def write_verified_fixture(self, root):
        root = pathlib.Path(root)
        payload = self.authorized_payload()
        actions_path = root / "actions.yaml"
        actions_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        frame = self.frame(payload)
        run_dir = root / "run"
        run_dir.mkdir()
        manifest_columns = [
            "id",
            "prompt_index",
            "seed",
            "action_id",
            "device",
            *evaluator.RUNTIME_COLUMNS,
        ]
        manifest_path = run_dir / "manifest.jsonl"
        manifest_path.write_text(
            "".join(
                json.dumps(row) + "\n"
                for row in frame[manifest_columns].to_dict("records")
            ),
            encoding="utf-8",
        )
        scores_path = run_dir / "scores.jsonl"
        scores_path.write_text(
            "".join(
                json.dumps(row) + "\n"
                for row in frame[["id", *evaluator.REQUIRED_SCORE_METRICS]].to_dict(
                    "records"
                )
            ),
            encoding="utf-8",
        )
        scorer_hash = payload["scoring"]["registered_scorer_provenance_sha256"]
        config = {
            "structural_control_registered": True,
            "action_schema": evaluator.EXECUTABLE_SCHEMA,
            "structural_control_registration_schema": evaluator.REGISTRATION_SCHEMA,
            "git_commit": "c" * 40,
            "split_role": "development",
            "seeds": list(self.SEEDS),
            "actions_yaml": str(actions_path.resolve()),
            "actions_sha256": self.sha256(actions_path),
            "structural_control_executable_actions_sha256": self.sha256(actions_path),
            "prompts_csv": str(PROMPTS.resolve()),
            "prompts_sha256": self.sha256(PROMPTS),
            "structural_control_source_template_sha256": self.sha256(REGISTRATION),
            "structural_control_source_template": str(REGISTRATION.relative_to(ROOT)),
            "structural_control_authorization": copy.deepcopy(payload["authorization"]),
            "structural_control_implementation_source": copy.deepcopy(
                payload["implementation_source"]
            ),
            "structural_control_analysis_implementation": copy.deepcopy(
                payload["analysis_implementation"]
            ),
            "structural_control_failure_policy": payload["failure_policy"],
            "scorer_provenance_binding_required": True,
            "scoring": copy.deepcopy(payload["scoring"]),
        }
        config_path = run_dir / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        audit = {
            "passed": True,
            "audit_schema": evaluator.AUDIT_SCHEMA,
            "split_role": "development",
            "records": 792,
            "prompts": 33,
            "blocks": 99,
            "expected_task_count": 792,
            "seeds": list(self.SEEDS),
            "actions": list(evaluator.EXPECTED_ACTIONS),
            "devices": ["cuda:7"],
            "required_score_keys": list(evaluator.REQUIRED_SCORE_METRICS),
            "scorer_provenance_schema": evaluator.SCORER_SCHEMA,
            "scorer_provenance_sha256": scorer_hash,
            "all_action_png_hashes_distinct_within_block": False,
            "structural_control_contract_passed": True,
            "image_decode_verified": True,
            "quality_results_inspected": False,
            "runtime_activation_ledgers_verified": True,
            "duplicate_action_pngs_are_failure": False,
            "isolated_duplicate_action_pngs_are_failure": False,
            "full_action_collapse_is_failure": True,
            "duplicate_action_png_policy": "reject_action_pair_equal_in_all_99_blocks",
            "duplicate_action_png_pair_counts": [],
            "fully_collapsed_action_pairs": [],
            "matched_unet_calls": "50x1",
            "scheduler": "EulerDiscreteScheduler",
            "scheduler_schedule_sha256": payload["scheduler_runtime"][
                "schedule_sha256"
            ],
            "generation_commit": "c" * 40,
            "executable_actions_sha256": self.sha256(actions_path),
            "registration_sha256": self.sha256(REGISTRATION),
            "analysis_implementation": copy.deepcopy(
                payload["analysis_implementation"]
            ),
            "analysis_implementation_sha256": evaluator.structural_audit.json_sha256(
                payload["analysis_implementation"]
            ),
            "device_identities": {
                "cuda:7": {
                    "logical_device_index": 7,
                    "physical_device_index": 7,
                    "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
                    "pci_bus_id": "00000000:01:00.0",
                    "cuda_visible_devices": None,
                }
            },
            "warnings": [],
            "provenance": {
                "config_sha256": self.sha256(config_path),
                "manifest_sha256": self.sha256(manifest_path),
                "scores_sha256": self.sha256(scores_path),
                "prompts_sha256": self.sha256(PROMPTS),
                "source_actions_sha256": self.sha256(actions_path),
                "source_template_sha256": self.sha256(REGISTRATION),
                "audit_script_sha256": self.sha256(evaluator.AUDITOR_PATH),
                "input_snapshot_stable": True,
            },
        }
        audit_path = run_dir / "run_audit.json"
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        return run_dir, actions_path, audit_path

    def test_verified_loader_requires_passing_hash_bound_792_task_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, actions_path, audit_path = self.write_verified_fixture(temporary)
            original_audit = json.loads(audit_path.read_text(encoding="utf-8"))
            with mock.patch.object(
                evaluator.structural_audit, "audit_run", return_value=original_audit
            ):
                verified = evaluator.load_verified_inputs(
                    run_dir, actions_path, audit_path
                )
            self.assertEqual(len(verified.frame), 792)
            self.assertEqual(verified.protocol.action_order, evaluator.EXPECTED_ACTIONS)

            generic = copy.deepcopy(original_audit)
            generic.pop("audit_schema")
            audit_path.write_text(json.dumps(generic), encoding="utf-8")
            with mock.patch.object(
                evaluator.structural_audit, "audit_run", return_value=generic
            ), self.assertRaisesRegex(ValueError, "dedicated structural-control audit"):
                evaluator.load_verified_inputs(run_dir, actions_path, audit_path)

            forged = copy.deepcopy(original_audit)
            forged["provenance"]["scores_sha256"] = "f" * 64
            audit_path.write_text(json.dumps(forged), encoding="utf-8")
            with mock.patch.object(
                evaluator.structural_audit, "audit_run", return_value=original_audit
            ), self.assertRaisesRegex(ValueError, "in-lock recomputation"):
                evaluator.load_verified_inputs(run_dir, actions_path, audit_path)

    def test_formal_loader_rejects_engineering_smoke_run_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, actions_path, audit_path = self.write_verified_fixture(temporary)
            config_path = run_dir / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config.update(
                {
                    "split_role": "engineering_smoke",
                    "engineering_only": True,
                    "formal_matrix_evidence": False,
                    "quality_claim_allowed": False,
                    "method_selection_allowed": False,
                }
            )
            config_path.write_text(json.dumps(config), encoding="utf-8")
            audit_report = json.loads(audit_path.read_text(encoding="utf-8"))
            audit_report["provenance"]["config_sha256"] = self.sha256(config_path)
            audit_path.write_text(json.dumps(audit_report), encoding="utf-8")
            with mock.patch.object(
                evaluator.structural_audit, "audit_run", return_value=audit_report
            ), self.assertRaisesRegex(ValueError, "engineering-smoke evidence-scope"):
                evaluator.load_verified_inputs(run_dir, actions_path, audit_path)

    def test_cli_is_one_shot_and_never_writes_a_selected_action(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, actions_path, audit_path = self.write_verified_fixture(temporary)
            argv = [
                "evaluate_structural_control_run.py",
                "--run-dir",
                str(run_dir),
                "--actions",
                str(actions_path),
                "--audit",
                str(audit_path),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                evaluator, "_require_committed_inputs", return_value="fixture-commit"
            ), mock.patch.object(
                evaluator.structural_audit,
                "audit_run",
                return_value=json.loads(audit_path.read_text(encoding="utf-8")),
            ), mock.patch.object(
                evaluator, "_crossed_bootstrap_ci", side_effect=self.mean_ci
            ), mock.patch.object(
                evaluator, "prompt_sign_flip_pvalue", return_value=0.0001
            ):
                evaluator.main()
            report = json.loads((run_dir / evaluator.OUTPUT_JSON).read_text())
            contrasts = pd.read_csv(run_dir / evaluator.OUTPUT_CSV)
            self.assertNotIn("selected_action", report)
            self.assertFalse(report["method_selection_performed"])
            self.assertFalse(report["rl_authorized"])
            self.assertEqual(len(contrasts), 8)

            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "one-shot"):
                    evaluator.main()


if __name__ == "__main__":
    unittest.main()
