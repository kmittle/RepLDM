import copy
import hashlib
import importlib.util
import json
import os
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
        self.assertFalse(result["publication_claim_authorized"])
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

    def write_verified_fixture(self, root, *, evaluation_attempt=True):
        root = pathlib.Path(root)
        payload = self.authorized_payload()
        actions_path = root / "actions.yaml"
        actions_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        frame = self.frame(payload)
        run_dir = root / "run"
        run_dir.mkdir()
        audit_path = run_dir / "run_audit.json"
        audit_attempt_path = evaluator.structural_audit.create_audit_attempt_marker(
            run_dir, audit_path
        )
        if evaluation_attempt:
            evaluator.create_evaluation_attempt_marker(run_dir)
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
        amendment_path = root / "analysis_amendment.json"
        amendment_path.write_text(json.dumps({"fixture": "amendment"}), encoding="utf-8")
        seal_path = run_dir / evaluator.structural_audit.STRUCTURAL_CONTROL_PRE_SCORE_SEAL_NAME
        seal_path.write_text(json.dumps({"fixture": "seal"}), encoding="utf-8")
        effective_analysis = {
            "schema": evaluator.ANALYSIS_IMPLEMENTATION_SCHEMA,
            "files": {
                relative_path: self.sha256(ROOT / relative_path)
                for relative_path in evaluator.EXPECTED_ANALYSIS_PATHS
            },
        }
        audit = {
            "passed": True,
            "audit_schema": evaluator.AUDIT_SCHEMA,
            "auditor_scope": "formal_development_only",
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
            "structural_control_contract_passed": True,
            "image_decode_verified": True,
            "quality_results_inspected": False,
            "runtime_activation_ledgers_verified": True,
            "duplicate_action_pngs_are_failure": False,
            "isolated_duplicate_action_pngs_are_failure": False,
            "full_action_collapse_is_failure": True,
            "duplicate_action_png_policy": (
                "reject_any_action_pair_equal_in_all_registered_blocks"
            ),
            "full_action_collapse_check_passed": True,
            "outcome_details_disclosed": False,
            "matched_unet_calls": "50x1",
            "scheduler": "EulerDiscreteScheduler",
            "scheduler_schedule_sha256": payload["scheduler_runtime"][
                "schedule_sha256"
            ],
            "generation_commit": "c" * 40,
            "executable_actions_sha256": self.sha256(actions_path),
            "registration_sha256": self.sha256(REGISTRATION),
            "analysis_amendment_sha256": self.sha256(amendment_path),
            "pre_score_seal_sha256": self.sha256(seal_path),
            "audit_attempt_sha256": self.sha256(audit_attempt_path),
            "analysis_implementation": effective_analysis,
            "analysis_implementation_sha256": evaluator.structural_audit.json_sha256(
                effective_analysis
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
                "analysis_amendment_sha256": self.sha256(amendment_path),
                "pre_score_seal_sha256": self.sha256(seal_path),
                "audit_attempt_sha256": self.sha256(audit_attempt_path),
                "audit_script_sha256": self.sha256(evaluator.AUDITOR_PATH),
                "input_snapshot_stable": True,
            },
        }
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        return run_dir, actions_path, audit_path, amendment_path, seal_path

    def test_verified_loader_requires_passing_hash_bound_792_task_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            (
                run_dir,
                actions_path,
                audit_path,
                amendment_path,
                seal_path,
            ) = self.write_verified_fixture(temporary)
            original_audit = json.loads(audit_path.read_text(encoding="utf-8"))
            with mock.patch.object(
                evaluator.structural_audit, "audit_run", return_value=original_audit
            ), mock.patch.object(
                evaluator.structural_audit,
                "validate_analysis_amendment",
                return_value={},
                create=True,
            ), mock.patch.object(
                evaluator.structural_audit,
                "validate_pre_score_seal",
                return_value={},
                create=True,
            ):
                verified = evaluator.load_verified_inputs(
                    run_dir,
                    actions_path,
                    audit_path,
                    amendment_path,
                    seal_path,
                )
            self.assertEqual(len(verified.frame), 792)
            self.assertEqual(verified.protocol.action_order, evaluator.EXPECTED_ACTIONS)

            generic = copy.deepcopy(original_audit)
            generic.pop("audit_schema")
            audit_path.write_text(json.dumps(generic), encoding="utf-8")
            with mock.patch.object(
                evaluator.structural_audit, "audit_run", return_value=generic
            ), mock.patch.object(
                evaluator.structural_audit,
                "validate_analysis_amendment",
                return_value={},
                create=True,
            ), mock.patch.object(
                evaluator.structural_audit,
                "validate_pre_score_seal",
                return_value={},
                create=True,
            ), self.assertRaisesRegex(ValueError, "dedicated structural-control audit"):
                evaluator.load_verified_inputs(
                    run_dir,
                    actions_path,
                    audit_path,
                    amendment_path,
                    seal_path,
                )

            marker_forged = copy.deepcopy(original_audit)
            marker_forged["provenance"]["audit_attempt_sha256"] = "f" * 64
            audit_path.write_text(json.dumps(marker_forged), encoding="utf-8")
            with mock.patch.object(
                evaluator.structural_audit, "audit_run", return_value=marker_forged
            ), mock.patch.object(
                evaluator.structural_audit,
                "validate_analysis_amendment",
                return_value={},
                create=True,
            ), mock.patch.object(
                evaluator.structural_audit,
                "validate_pre_score_seal",
                return_value={},
                create=True,
            ), self.assertRaisesRegex(ValueError, "hash mismatch for audit_attempt_sha256"):
                evaluator.load_verified_inputs(
                    run_dir,
                    actions_path,
                    audit_path,
                    amendment_path,
                    seal_path,
                )

            forged = copy.deepcopy(original_audit)
            forged["provenance"]["scores_sha256"] = "f" * 64
            audit_path.write_text(json.dumps(forged), encoding="utf-8")
            with mock.patch.object(
                evaluator.structural_audit, "audit_run", return_value=original_audit
            ), mock.patch.object(
                evaluator.structural_audit,
                "validate_analysis_amendment",
                return_value={},
                create=True,
            ), mock.patch.object(
                evaluator.structural_audit,
                "validate_pre_score_seal",
                return_value={},
                create=True,
            ), self.assertRaisesRegex(ValueError, "in-lock recomputation"):
                evaluator.load_verified_inputs(
                    run_dir,
                    actions_path,
                    audit_path,
                    amendment_path,
                    seal_path,
                )

    def test_verified_loader_rejects_missing_or_tampered_attempt_markers_before_outcomes(self):
        cases = (
            ("audit_attempt", "missing"),
            ("audit_attempt", "tampered"),
            ("evaluation_attempt", "missing"),
            ("evaluation_attempt", "tampered"),
        )
        for marker_key, mutation in cases:
            with self.subTest(
                marker_key=marker_key, mutation=mutation
            ), tempfile.TemporaryDirectory() as temporary:
                (
                    run_dir,
                    actions_path,
                    audit_path,
                    amendment_path,
                    seal_path,
                ) = self.write_verified_fixture(temporary)
                marker_path = run_dir / (
                    evaluator.structural_audit.STRUCTURAL_CONTROL_AUDIT_ATTEMPT_NAME
                    if marker_key == "audit_attempt"
                    else evaluator.ATTEMPT_MARKER
                )
                if mutation == "missing":
                    marker_path.unlink()
                else:
                    marker_path.write_bytes(b"tampered marker\n")

                original_read_bytes = pathlib.Path.read_bytes
                outcome_paths = {
                    (run_dir / "scores.jsonl").resolve(),
                    audit_path.resolve(),
                }

                def reject_outcome_read(path):
                    if path.resolve() in outcome_paths:
                        raise AssertionError("outcome bytes were read before marker rejection")
                    return original_read_bytes(path)

                with mock.patch.object(
                    pathlib.Path, "read_bytes", reject_outcome_read
                ), mock.patch.object(
                    evaluator.structural_audit, "audit_run"
                ) as audit_run, mock.patch.object(
                    evaluator.structural_audit,
                    "validate_analysis_amendment",
                    return_value={},
                ), mock.patch.object(
                    evaluator.structural_audit,
                    "validate_pre_score_seal",
                    return_value={},
                ), self.assertRaisesRegex(ValueError, "attempt marker is (missing|invalid)"):
                    evaluator.load_verified_inputs(
                        run_dir,
                        actions_path,
                        audit_path,
                        amendment_path,
                        seal_path,
                    )
                audit_run.assert_not_called()

    def test_formal_loader_rejects_engineering_smoke_run_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            (
                run_dir,
                actions_path,
                audit_path,
                amendment_path,
                seal_path,
            ) = self.write_verified_fixture(temporary)
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
            ), mock.patch.object(
                evaluator.structural_audit,
                "validate_analysis_amendment",
                return_value={},
                create=True,
            ), mock.patch.object(
                evaluator.structural_audit,
                "validate_pre_score_seal",
                return_value={},
                create=True,
            ), self.assertRaisesRegex(ValueError, "engineering-smoke evidence-scope"):
                evaluator.load_verified_inputs(
                    run_dir,
                    actions_path,
                    audit_path,
                    amendment_path,
                    seal_path,
                )

    def test_formal_loader_rejects_symlinked_amendment_or_seal(self):
        with tempfile.TemporaryDirectory() as temporary:
            (
                run_dir,
                actions_path,
                audit_path,
                amendment_path,
                seal_path,
            ) = self.write_verified_fixture(temporary)
            for label, target in (
                ("amendment", amendment_path),
                ("seal", seal_path),
            ):
                alias = pathlib.Path(temporary) / f"{label}_alias"
                alias.symlink_to(target)
                with self.subTest(label=label), self.assertRaisesRegex(
                    ValueError, "cannot be symlinks"
                ):
                    evaluator.load_verified_inputs(
                        run_dir,
                        actions_path,
                        audit_path,
                        alias if label == "amendment" else amendment_path,
                        alias if label == "seal" else seal_path,
                    )

    def bundle_payloads(self, input_hashes=None):
        input_hashes = input_hashes or {
            key: f"{index + 1:064x}"
            for index, key in enumerate(evaluator.INPUT_PROVENANCE_KEYS)
        }
        report = {
            "schema": evaluator.EVALUATION_SCHEMA,
            "evaluator_version": evaluator.EVALUATOR_VERSION,
            "scope": evaluator.EVALUATION_SCOPE,
            "screen_only": True,
            "method_selection_authorized": False,
            "method_selection_performed": False,
            "validation_authorized": False,
            "rl_authorized": False,
            "publication_claim_authorized": False,
            "publication_superiority_established": False,
            "claims": dict(evaluator.EXPECTED_EVALUATION_CLAIMS),
            "action_summaries": [{"action_id": "a", "observed_tasks": 99}],
            "provenance": {"fixture": "canonical"},
            "input_provenance_sha256": input_hashes,
            "contrasts": [
                {"contrast_id": "a_vs_b", "topiq_nr_mean_delta": 0.01},
                {"contrast_id": "c_vs_d", "topiq_nr_mean_delta": -0.02},
            ],
        }
        csv_payload = evaluator._csv_payload(report["contrasts"], input_hashes)
        report["output_bundle"] = {
            "schema": evaluator.OUTPUT_BUNDLE_SCHEMA,
            "scope": evaluator.EVALUATION_SCOPE,
            "csv": {
                "filename": evaluator.OUTPUT_CSV,
                "sha256": evaluator.sha256_bytes(csv_payload),
                "row_count": len(report["contrasts"]),
                "artifact_schema": evaluator.CONTRAST_ARTIFACT_SCHEMA,
                "scope": evaluator.EVALUATION_SCOPE,
            },
        }
        json_payload = (
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        return input_hashes, json_payload, csv_payload

    @staticmethod
    def evaluation_payload_object(input_hashes, json_payload, csv_payload):
        return evaluator.EvaluationPayloads(
            result=json.loads(json_payload),
            json_payload=json_payload,
            csv_payload=csv_payload,
            input_hashes=input_hashes,
        )

    def write_bundle_input_files(self, root):
        root = pathlib.Path(root)
        names = {
            "run_config": "config.json",
            "manifest": "manifest.jsonl",
            "scores": "scores.jsonl",
            "actions": "actions.yaml",
            "audit": "run_audit.json",
            "audit_attempt": (
                evaluator.structural_audit.STRUCTURAL_CONTROL_AUDIT_ATTEMPT_NAME
            ),
            "evaluation_attempt": evaluator.ATTEMPT_MARKER,
            "registration": "registration.yaml",
            "prompts": "prompts.csv",
            "prompt_manifest": "prompt_manifest.yaml",
            "analysis_amendment": "analysis_amendment.yaml",
            "pre_score_seal": (
                evaluator.structural_audit.STRUCTURAL_CONTROL_PRE_SCORE_SEAL_NAME
            ),
        }
        paths = {key: root / name for key, name in names.items()}
        for key, path in paths.items():
            if key == "audit_attempt":
                payload = evaluator.structural_audit._audit_attempt_marker_payload()
            elif key == "evaluation_attempt":
                payload = evaluator._attempt_marker_payload()
            else:
                payload = f"current input: {key}\n".encode("ascii")
            path.write_bytes(payload)
        hashes = {key: self.sha256(path) for key, path in paths.items()}
        return paths, hashes

    def test_bundle_verifier_rejects_missing_tampered_swapped_or_drifted_csv(self):
        cases = (
            "missing",
            "tampered",
            "swapped",
            "hash",
            "row",
            "schema",
            "scope",
            "envelope",
            "provenance",
            "publication",
            "population",
            "selected",
            "extra_authorization",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                input_hashes, json_payload, csv_payload = self.bundle_payloads()
                bundle = evaluator.publish_evaluation_bundle(
                    root,
                    json_payload,
                    csv_payload,
                    expected_input_hashes=input_hashes,
                )
                json_path = bundle / evaluator.OUTPUT_JSON
                csv_path = bundle / evaluator.OUTPUT_CSV
                if case == "missing":
                    csv_path.unlink()
                elif case == "tampered":
                    csv_path.write_bytes(csv_path.read_bytes() + b"tampered\n")
                elif case == "swapped":
                    report = json.loads(json_path.read_text(encoding="utf-8"))
                    foreign_csv = evaluator._csv_payload(
                        [
                            {"contrast_id": "foreign_1", "topiq_nr_mean_delta": 1.0},
                            {"contrast_id": "foreign_2", "topiq_nr_mean_delta": 2.0},
                        ],
                        input_hashes,
                    )
                    csv_path.write_bytes(foreign_csv)
                    report["output_bundle"]["csv"]["sha256"] = (
                        evaluator.sha256_bytes(foreign_csv)
                    )
                    json_path.write_text(
                        json.dumps(report, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                else:
                    report = json.loads(json_path.read_text(encoding="utf-8"))
                    if case == "hash":
                        report["output_bundle"]["csv"]["sha256"] = "f" * 64
                    elif case == "row":
                        report["output_bundle"]["csv"]["row_count"] += 1
                    elif case == "schema":
                        report["output_bundle"]["csv"]["artifact_schema"] = (
                            "structural_control_contrast_artifact_v1"
                        )
                    elif case == "scope":
                        report["output_bundle"]["scope"] = "validation"
                    elif case == "envelope":
                        changed = csv_path.read_bytes().replace(
                            b",True,False,", b",False,False,", 1
                        )
                        csv_path.write_bytes(changed)
                        report["output_bundle"]["csv"]["sha256"] = (
                            evaluator.sha256_bytes(changed)
                        )
                    elif case == "provenance":
                        report["input_provenance_sha256"]["scores"] = "e" * 64
                    elif case == "publication":
                        report["publication_claim_authorized"] = True
                    elif case == "population":
                        report["claims"]["population_generalization_established"] = True
                    elif case == "selected":
                        report["selected_action"] = "a"
                    elif case == "extra_authorization":
                        report["distillation_authorized"] = False
                    json_path.write_text(
                        json.dumps(report, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                with self.assertRaises(ValueError):
                    evaluator.verify_evaluation_bundle(
                        bundle,
                        expected_input_hashes=input_hashes,
                        strict=False,
                    )

    def test_strict_bundle_verifier_rehashes_current_inputs_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            current_paths, current_hashes = self.write_bundle_input_files(root)
            self.assertEqual(set(current_paths), set(evaluator.INPUT_PROVENANCE_KEYS))
            self.assertEqual(len(current_paths), 12)
            _, json_payload, csv_payload = self.bundle_payloads(current_hashes)
            bundle = evaluator.publish_evaluation_bundle(
                root,
                json_payload,
                csv_payload,
                expected_input_hashes=current_hashes,
            )
            replayed = self.evaluation_payload_object(
                current_hashes, json_payload, csv_payload
            )
            with mock.patch.object(
                evaluator, "replay_evaluation_payloads", return_value=replayed
            ) as replay:
                evaluator.verify_evaluation_bundle(
                    bundle, current_input_paths=current_paths
                )
            replay.assert_called_once()
            with self.assertRaisesRegex(ValueError, "requires current input paths"):
                evaluator.verify_evaluation_bundle(bundle)

            current_paths["scores"].write_bytes(b"drifted scores\n")
            evaluator.verify_evaluation_bundle(
                bundle,
                expected_input_hashes=current_hashes,
                strict=False,
            )
            with self.assertRaisesRegex(ValueError, "differs from current inputs"):
                evaluator.verify_evaluation_bundle(
                    bundle, current_input_paths=current_paths
                )

    def test_strict_verifier_rejects_coordinated_semantic_rewrites(self):
        for attack in ("contrast", "action_summary", "provenance"):
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                current_paths, current_hashes = self.write_bundle_input_files(root)
                _, json_payload, csv_payload = self.bundle_payloads(current_hashes)
                canonical = self.evaluation_payload_object(
                    current_hashes, json_payload, csv_payload
                )
                bundle = evaluator.publish_evaluation_bundle(
                    root,
                    json_payload,
                    csv_payload,
                    expected_input_hashes=current_hashes,
                )
                report_path = bundle / evaluator.OUTPUT_JSON
                csv_path = bundle / evaluator.OUTPUT_CSV
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if attack == "contrast":
                    report["contrasts"][0]["topiq_nr_mean_delta"] = 99.0
                elif attack == "action_summary":
                    report["action_summaries"][0]["observed_tasks"] = 1
                else:
                    report["provenance"]["fixture"] = "rewritten"
                forged_csv = evaluator._csv_payload(
                    report["contrasts"], current_hashes
                )
                report["output_bundle"]["csv"]["sha256"] = (
                    evaluator.sha256_bytes(forged_csv)
                )
                csv_path.write_bytes(forged_csv)
                report_path.write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

                evaluator.verify_evaluation_bundle(
                    bundle,
                    expected_input_hashes=current_hashes,
                    strict=False,
                )
                with mock.patch.object(
                    evaluator, "replay_evaluation_payloads", return_value=canonical
                ), self.assertRaisesRegex(ValueError, "semantic replay"):
                    evaluator.verify_evaluation_bundle(
                        bundle, current_input_paths=current_paths
                    )

    def test_strict_verifier_rejects_bundle_replacement_during_semantic_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            current_paths, current_hashes = self.write_bundle_input_files(root)
            _, json_payload, csv_payload = self.bundle_payloads(current_hashes)
            canonical = self.evaluation_payload_object(
                current_hashes, json_payload, csv_payload
            )
            bundle = evaluator.publish_evaluation_bundle(
                root,
                json_payload,
                csv_payload,
                expected_input_hashes=current_hashes,
            )
            replacement = root / "replacement_bundle"
            replacement.mkdir()
            forged = json.loads(json_payload)
            forged["provenance"]["fixture"] = "replaced-during-replay"
            (replacement / evaluator.OUTPUT_JSON).write_text(
                json.dumps(forged, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (replacement / evaluator.OUTPUT_CSV).write_bytes(csv_payload)

            def replace_bundle(_paths):
                bundle.rename(root / "original_bundle")
                replacement.rename(bundle)
                return canonical

            with mock.patch.object(
                evaluator,
                "replay_evaluation_payloads",
                side_effect=replace_bundle,
            ), self.assertRaisesRegex(ValueError, "changed during semantic replay"):
                evaluator.verify_evaluation_bundle(
                    bundle, current_input_paths=current_paths
                )

    def test_strict_verifier_rejects_attempt_marker_drift_during_replay(self):
        for marker_key in ("audit_attempt", "evaluation_attempt"):
            with self.subTest(marker_key=marker_key), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                current_paths, current_hashes = self.write_bundle_input_files(root)
                _, json_payload, csv_payload = self.bundle_payloads(current_hashes)
                canonical = self.evaluation_payload_object(
                    current_hashes, json_payload, csv_payload
                )
                bundle = evaluator.publish_evaluation_bundle(
                    root,
                    json_payload,
                    csv_payload,
                    expected_input_hashes=current_hashes,
                )

                def drift_marker(_paths):
                    current_paths[marker_key].write_bytes(b"drifted marker\n")
                    return canonical

                with mock.patch.object(
                    evaluator,
                    "replay_evaluation_payloads",
                    side_effect=drift_marker,
                ), self.assertRaisesRegex(ValueError, "attempt marker is invalid"):
                    evaluator.verify_evaluation_bundle(
                        bundle, current_input_paths=current_paths
                    )

    def test_semantic_replay_uses_formal_loader_and_shared_builder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            current_paths, current_hashes = self.write_bundle_input_files(root)
            _, json_payload, csv_payload = self.bundle_payloads(current_hashes)
            expected = self.evaluation_payload_object(
                current_hashes, json_payload, csv_payload
            )
            inputs = mock.sentinel.verified_inputs
            with mock.patch.object(
                evaluator, "load_verified_inputs", return_value=inputs
            ) as loader, mock.patch.object(
                evaluator,
                "_committed_evaluator_version",
                return_value="fixture-commit",
            ) as committed, mock.patch.object(
                evaluator, "build_evaluation_payloads", return_value=expected
            ) as builder:
                observed = evaluator.replay_evaluation_payloads(current_paths)
            self.assertIs(observed, expected)
            loader.assert_called_once_with(
                root.resolve(),
                current_paths["actions"].resolve(),
                current_paths["audit"].resolve(),
                current_paths["analysis_amendment"].resolve(),
                current_paths["pre_score_seal"].resolve(),
            )
            committed.assert_called_once_with(inputs)
            builder.assert_called_once_with(
                inputs, evaluator_git_commit="fixture-commit"
            )

    def test_strict_bundle_verifier_rejects_symlinks_and_noncanonical_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            current_paths, current_hashes = self.write_bundle_input_files(root)
            _, json_payload, csv_payload = self.bundle_payloads(current_hashes)
            bundle = evaluator.publish_evaluation_bundle(
                root,
                json_payload,
                csv_payload,
                expected_input_hashes=current_hashes,
            )
            bundle_alias = root / "bundle_alias"
            bundle_alias.symlink_to(bundle, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "missing or not a directory"):
                evaluator.verify_evaluation_bundle(
                    bundle_alias, current_input_paths=current_paths
                )

            noncanonical_audit = root / "other_audit.json"
            noncanonical_audit.write_bytes(current_paths["audit"].read_bytes())
            wrong_paths = dict(current_paths)
            wrong_paths["audit"] = noncanonical_audit
            with self.assertRaisesRegex(ValueError, "canonical audit path"):
                evaluator.verify_evaluation_bundle(
                    bundle, current_input_paths=wrong_paths
                )

            scores_target = root / "scores_target.jsonl"
            current_paths["scores"].replace(scores_target)
            current_paths["scores"].symlink_to(scores_target)
            with self.assertRaisesRegex(ValueError, "symlink input: scores"):
                evaluator.verify_evaluation_bundle(
                    bundle, current_input_paths=current_paths
                )

    def test_attempt_marker_uses_exclusive_nofollow_and_is_one_shot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            original_open = evaluator.os.open
            with mock.patch.object(evaluator.os, "open", wraps=original_open) as opened:
                marker = evaluator.create_evaluation_attempt_marker(root)
            first_flags = opened.call_args_list[0].args[1]
            self.assertTrue(first_flags & evaluator.os.O_EXCL)
            self.assertTrue(first_flags & evaluator.os.O_NOFOLLOW)
            self.assertEqual(marker.read_bytes(), evaluator._attempt_marker_payload())
            with self.assertRaisesRegex(ValueError, "one-shot"):
                evaluator.create_evaluation_attempt_marker(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            target = root / "target"
            target.write_text("unchanged", encoding="utf-8")
            (root / evaluator.ATTEMPT_MARKER).symlink_to(target)
            with self.assertRaisesRegex(ValueError, "one-shot"):
                evaluator.create_evaluation_attempt_marker(root)
            self.assertEqual(target.read_text(encoding="utf-8"), "unchanged")

        for existing in (
            evaluator.OUTPUT_BUNDLE_DIR,
            evaluator.OUTPUT_JSON,
            evaluator.OUTPUT_CSV,
        ):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                path = root / existing
                if existing == evaluator.OUTPUT_BUNDLE_DIR:
                    path.mkdir()
                else:
                    path.write_text("legacy", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "one-shot"):
                    evaluator.create_evaluation_attempt_marker(root)
                self.assertFalse((root / evaluator.ATTEMPT_MARKER).exists())

    def test_strict_verification_requires_both_attempt_markers_without_creating_them(self):
        for marker_key in ("audit_attempt", "evaluation_attempt"):
            with self.subTest(marker_key=marker_key), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                current_paths, current_hashes = self.write_bundle_input_files(root)
                _, json_payload, csv_payload = self.bundle_payloads(current_hashes)
                bundle = evaluator.publish_evaluation_bundle(
                    root,
                    json_payload,
                    csv_payload,
                    expected_input_hashes=current_hashes,
                )
                current_paths[marker_key].unlink()
                with self.assertRaisesRegex(
                    ValueError, "(attempt marker is missing|input is missing: .*attempt)"
                ):
                    evaluator.verify_evaluation_bundle(
                        bundle, current_input_paths=current_paths
                    )
                self.assertFalse(current_paths[marker_key].exists())

    def test_attempt_marker_failures_are_persistent_and_forbid_retry(self):
        for phase in ("file_fsync", "parent_fsync"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                if phase == "file_fsync":
                    patcher = mock.patch.object(
                        evaluator.os,
                        "fsync",
                        side_effect=OSError("injected marker file fsync failure"),
                    )
                else:
                    patcher = mock.patch.object(
                        evaluator,
                        "_fsync_directory",
                        side_effect=OSError("injected marker parent fsync failure"),
                    )
                with patcher, self.assertRaises(OSError):
                    evaluator.create_evaluation_attempt_marker(root)
                self.assertTrue(os.path.lexists(root / evaluator.ATTEMPT_MARKER))
                with self.assertRaisesRegex(ValueError, "one-shot"):
                    evaluator.create_evaluation_attempt_marker(root)

    def test_cli_consumes_attempt_before_loading_any_outcome_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            (
                run_dir,
                actions_path,
                audit_path,
                amendment_path,
                seal_path,
            ) = self.write_verified_fixture(temporary, evaluation_attempt=False)
            argv = [
                "evaluate_structural_control_run.py",
                "--run-dir",
                str(run_dir),
                "--actions",
                str(actions_path),
                "--audit",
                str(audit_path),
                "--analysis-amendment",
                str(amendment_path),
                "--pre-score-seal",
                str(seal_path),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                evaluator,
                "load_verified_inputs",
                side_effect=RuntimeError("injected pre-read stop"),
            ) as loader, self.assertRaisesRegex(RuntimeError, "pre-read stop"):
                evaluator.main()
            loader.assert_called_once()
            marker = run_dir / evaluator.ATTEMPT_MARKER
            self.assertEqual(marker.read_bytes(), evaluator._attempt_marker_payload())
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                evaluator, "load_verified_inputs"
            ) as second_loader, self.assertRaisesRegex(ValueError, "one-shot"):
                evaluator.main()
            second_loader.assert_not_called()

    def test_bundle_publish_faults_leave_no_partial_canonical_directory(self):
        for phase in ("write", "file_fsync", "directory_fsync", "rename"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                input_hashes, json_payload, csv_payload = self.bundle_payloads()
                if phase == "write":
                    patcher = mock.patch.object(
                        evaluator,
                        "_write_fsynced_file",
                        side_effect=OSError("injected write failure"),
                    )
                elif phase == "file_fsync":
                    patcher = mock.patch.object(
                        evaluator.os,
                        "fsync",
                        side_effect=OSError("injected file fsync failure"),
                    )
                elif phase == "directory_fsync":
                    patcher = mock.patch.object(
                        evaluator,
                        "_fsync_directory",
                        side_effect=OSError("injected directory fsync failure"),
                    )
                else:
                    patcher = mock.patch.object(
                        evaluator.os,
                        "replace",
                        side_effect=OSError("injected rename failure"),
                    )
                with patcher, self.assertRaises(OSError):
                    evaluator.publish_evaluation_bundle(
                        root,
                        json_payload,
                        csv_payload,
                        expected_input_hashes=input_hashes,
                    )
                self.assertFalse((root / evaluator.OUTPUT_BUNDLE_DIR).exists())
                self.assertEqual(
                    list(root.glob(f".{evaluator.OUTPUT_BUNDLE_DIR}.*")), []
                )
                bundle = evaluator.publish_evaluation_bundle(
                    root,
                    json_payload,
                    csv_payload,
                    expected_input_hashes=input_hashes,
                )
                evaluator.verify_evaluation_bundle(
                    bundle,
                    expected_input_hashes=input_hashes,
                    strict=False,
                )

    def test_parent_fsync_failure_leaves_complete_one_shot_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            input_hashes, json_payload, csv_payload = self.bundle_payloads()
            with mock.patch.object(
                evaluator,
                "_fsync_directory",
                side_effect=[None, OSError("injected parent fsync failure")],
            ), self.assertRaises(OSError):
                evaluator.publish_evaluation_bundle(
                    root,
                    json_payload,
                    csv_payload,
                    expected_input_hashes=input_hashes,
                )
            bundle = root / evaluator.OUTPUT_BUNDLE_DIR
            evaluator.verify_evaluation_bundle(
                bundle,
                expected_input_hashes=input_hashes,
                strict=False,
            )
            with self.assertRaisesRegex(ValueError, "one-shot"):
                evaluator.publish_evaluation_bundle(
                    root,
                    json_payload,
                    csv_payload,
                    expected_input_hashes=input_hashes,
                )

    def test_cli_is_one_shot_and_never_writes_a_selected_action(self):
        with tempfile.TemporaryDirectory() as temporary:
            (
                run_dir,
                actions_path,
                audit_path,
                amendment_path,
                seal_path,
            ) = self.write_verified_fixture(temporary, evaluation_attempt=False)
            argv = [
                "evaluate_structural_control_run.py",
                "--run-dir",
                str(run_dir),
                "--actions",
                str(actions_path),
                "--audit",
                str(audit_path),
                "--analysis-amendment",
                str(amendment_path),
                "--pre-score-seal",
                str(seal_path),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                evaluator, "_require_committed_inputs", return_value="fixture-commit"
            ), mock.patch.object(
                evaluator.structural_audit,
                "audit_run",
                return_value=json.loads(audit_path.read_text(encoding="utf-8")),
            ), mock.patch.object(
                evaluator.structural_audit,
                "validate_analysis_amendment",
                return_value={},
                create=True,
            ), mock.patch.object(
                evaluator.structural_audit,
                "validate_pre_score_seal",
                return_value={},
                create=True,
            ), mock.patch.object(
                evaluator, "_crossed_bootstrap_ci", side_effect=self.mean_ci
            ), mock.patch.object(
                evaluator, "prompt_sign_flip_pvalue", return_value=0.0001
            ):
                evaluator.main()
            bundle = run_dir / evaluator.OUTPUT_BUNDLE_DIR
            current_paths = evaluator.resolve_evaluation_input_paths(
                run_dir,
                actions_path,
                audit_path,
                amendment_path,
                seal_path,
            )
            canonical_json = (bundle / evaluator.OUTPUT_JSON).read_bytes()
            canonical_csv = (bundle / evaluator.OUTPUT_CSV).read_bytes()
            canonical_report = json.loads(canonical_json)
            canonical_payloads = self.evaluation_payload_object(
                canonical_report["input_provenance_sha256"],
                canonical_json,
                canonical_csv,
            )
            with mock.patch.object(
                evaluator,
                "replay_evaluation_payloads",
                return_value=canonical_payloads,
            ) as replay:
                report = evaluator.verify_evaluation_bundle(
                    bundle, current_input_paths=current_paths
                )
                with mock.patch.object(sys, "argv", [*argv, "--verify-bundle"]):
                    evaluator.main()
            self.assertEqual(replay.call_count, 2)
            contrasts = pd.read_csv(bundle / evaluator.OUTPUT_CSV)
            self.assertFalse((run_dir / evaluator.OUTPUT_JSON).exists())
            self.assertFalse((run_dir / evaluator.OUTPUT_CSV).exists())
            self.assertNotIn("selected_action", report)
            self.assertFalse(report["method_selection_performed"])
            self.assertFalse(report["rl_authorized"])
            self.assertEqual(
                report["output_bundle"]["csv"]["sha256"],
                evaluator.sha256_file(bundle / evaluator.OUTPUT_CSV),
            )
            self.assertEqual(len(contrasts), 8)
            self.assertEqual(
                set(contrasts["artifact_schema"]),
                {evaluator.CONTRAST_ARTIFACT_SCHEMA},
            )
            self.assertEqual(
                set(canonical_report["input_provenance_sha256"]),
                set(evaluator.INPUT_PROVENANCE_KEYS),
            )
            for marker_key in ("audit_attempt", "evaluation_attempt"):
                self.assertEqual(
                    set(contrasts[f"input_{marker_key}_sha256"]),
                    {self.sha256(current_paths[marker_key])},
                )
            self.assertTrue(contrasts["screen_only"].all())
            for field in (
                "method_selection_authorized",
                "method_selection_performed",
                "validation_authorized",
                "rl_authorized",
                "publication_claim_authorized",
                "publication_superiority_established",
                "global_multiplicity_across_families_controlled",
            ):
                self.assertTrue((~contrasts[field]).all(), field)

            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "one-shot"):
                    evaluator.main()


if __name__ == "__main__":
    unittest.main()
