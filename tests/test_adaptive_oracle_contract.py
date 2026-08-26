import copy
import csv
import hashlib
import importlib.util
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_PIPELINE = ROOT / "eval-pipeline"
PROMPT_DIR = EVAL_PIPELINE / "prompts"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract = load_module(
    "adaptive_oracle_contract_test",
    EVAL_PIPELINE / "adaptive_oracle_contract.py",
)


def digest(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class AdaptiveOracleContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (PROMPT_DIR / "adaptive_oracle_engineering.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            cls.prompt_rows = list(csv.DictReader(handle))
        cls.manifest = json.loads(
            (PROMPT_DIR / "adaptive_oracle_prompt_manifest_v1.json").read_text(
                encoding="utf-8"
            )
        )
        cls.design = contract.build_engineering_design(
            cls.prompt_rows, cls.manifest
        )

    def task_for(self, family):
        for task in self.design["tasks"][: contract.TASKS_PER_PROMPT]:
            action = task["action"]
            if action["family"] == family and not action["physical_no_op"]:
                return task
            if family == "P0" and action["physical_no_op"]:
                return task
        self.fail(f"no task for family {family}")

    def make_sidecar(self, task):
        action = task["action"]
        prompt = task["prompt"]
        physical_no_op = action["physical_no_op"]
        initial_latent_hash = digest(
            f"{prompt['prompt_row_id']}:initial-latent"
        )
        latent_before_hash = initial_latent_hash
        steps = []
        for step_index in range(contract.NUM_INFERENCE_STEPS):
            latent_after_hash = digest(
                f"{task['task_id']}:latent-after:{step_index}"
            )
            feature_hook = None
            if action["feature_hook_required"]:
                feature_hook = {
                    "module_path": contract.FEATURE_BLOCK,
                    "input_name": contract.FEATURE_INPUT_NAME,
                    "hook_calls": 1,
                    "consume_calls": 1,
                    "expected_cfg_shape": list(contract.FEATURE_CFG_SHAPE),
                    "captured_conditional_shape": list(
                        contract.FEATURE_CONDITIONAL_SHAPE
                    ),
                    "cfg_row_order": contract.CFG_ROW_ORDER,
                    "detached": True,
                }
            random_counter = None
            if action["random_counter_required"]:
                random_counter = {
                    "schema": contract.RANDOM_EDGE_COUNTER_SCHEMA,
                    "experiment_id": contract.EXPERIMENT_ID,
                    "split_role": contract.SPLIT_ROLE,
                    "prompt_row_id": prompt["prompt_row_id"],
                    "seed": prompt["seed"],
                    "step_index": step_index,
                    "orbit_name": action["orbit_name"],
                    "undirected_edge_count": contract.ORBIT_EDGE_COUNTS[
                        action["orbit_name"]
                    ],
                    "counter_set_sha256": (
                        contract.random_edge_counter_set_sha256(
                            experiment_id=contract.EXPERIMENT_ID,
                            split_role=contract.SPLIT_ROLE,
                            prompt_row_id=prompt["prompt_row_id"],
                            seed=prompt["seed"],
                            step_index=step_index,
                            orbit_name=action["orbit_name"],
                            grid_size=contract.GRID_SIZE,
                        )
                    ),
                }
            steps.append(
                {
                    "step_index": step_index,
                    "affinity_source": action["affinity_source"],
                    "orbit_name": action["orbit_name"],
                    "sign": action["sign"],
                    "unet_calls": 1,
                    "scheduler_calls": 1,
                    "extra_unet_calls": 0,
                    "extra_scheduler_calls": 0,
                    "backbone_backward_calls": 0,
                    "intermediate_decode_calls": 0,
                    "feature_hook": feature_hook,
                    "random_edge_counter": random_counter,
                    "basis_sha256": (
                        None
                        if physical_no_op
                        else digest(
                            (
                                f"{prompt['prompt_row_id']}:{action['family']}:"
                                f"{action['orbit_name']}:basis:0"
                            )
                            if step_index == 0 and action["family"] in {"P", "R"}
                            else f"{task['task_id']}:basis:{step_index}"
                        )
                    ),
                    "latent_before_sha256": latent_before_hash,
                    "latent_after_sha256": latent_after_hash,
                    "applied_update_ratio": (
                        0.0
                        if physical_no_op
                        else contract.ACTIVE_TARGET_UPDATE_RATIO
                    ),
                    "target_ratio_error": 0.0,
                    "channel_mean_abs_error": 0.0 if physical_no_op else 5e-5,
                    "channel_variance_relative_error": (
                        0.0 if physical_no_op else 5e-4
                    ),
                    "channel_covariance_drift": (
                        0.0 if physical_no_op else 5e-3
                    ),
                    "pred_original_sample_relative_l2_error": (
                        0.0 if physical_no_op else 5e-4
                    ),
                    "expected_prev_sample_relative_l2_error": (
                        0.0 if physical_no_op else 5e-4
                    ),
                    "native_round_trip_max_abs_error": (
                        0.0 if physical_no_op else 8e-4
                    ),
                    "solver_target_update_ratio": (
                        0.0
                        if physical_no_op
                        else contract.ACTIVE_TARGET_UPDATE_RATIO
                    ),
                    "mapped_ratio_solver_evaluations": (
                        0 if physical_no_op else 1
                    ),
                    "finite": True,
                    "cap_hit": False,
                }
            )
            latent_before_hash = latent_after_hash
        provenance = task["provenance"]
        return {
            "schema": contract.SIDECAR_SCHEMA,
            "experiment_id": contract.EXPERIMENT_ID,
            "task_id": task["task_id"],
            "prompt": copy.deepcopy(prompt),
            "action": copy.deepcopy(action),
            "trajectory": {
                "trajectory_id": f"trajectory:{task['task_id']}",
                "initial_latent_sha256": initial_latent_hash,
                "final_latent_sha256": latent_before_hash,
                "png_sha256": digest(f"{task['task_id']}:png"),
                "physical_no_op": physical_no_op,
                "scheduler_class": contract.SCHEDULER_CLASS,
                "scheduler_churn": contract.SCHEDULER_CHURN,
                "scheduler_prediction_type": contract.SCHEDULER_PREDICTION_TYPE,
                "scheduler_config_sha256_v2": contract.SCHEDULER_CONFIG_SHA256_V2,
                "scheduler_schedule_sha256": contract.SCHEDULER_SCHEDULE_SHA256,
                "scheduler_timesteps": list(contract.SCHEDULER_TIMESTEPS),
                "scheduler_sigmas": list(contract.SCHEDULER_SIGMAS),
                "scheduler_construction_init_noise_sigma": (
                    contract.SCHEDULER_CONSTRUCTION_INIT_NOISE_SIGMA
                ),
                "scheduler_effective_init_noise_sigma": (
                    contract.SCHEDULER_EFFECTIVE_INIT_NOISE_SIGMA
                ),
                "model_id": contract.MODEL_ID,
                "model_revision": contract.MODEL_REVISION,
                "model_snapshot_manifest_sha256": (
                    contract.MODEL_SNAPSHOT_MANIFEST_SHA256
                ),
                "num_inference_steps": contract.NUM_INFERENCE_STEPS,
                "width": contract.WIDTH,
                "height": contract.HEIGHT,
                "cfg_scale": contract.CFG_SCALE,
                "grid_size": contract.GRID_SIZE,
            },
            "call_totals": {
                "unet_calls": contract.NUM_INFERENCE_STEPS,
                "scheduler_calls": contract.NUM_INFERENCE_STEPS,
                "extra_unet_calls": 0,
                "extra_scheduler_calls": 0,
                "backbone_backward_calls": 0,
                "intermediate_decode_calls": 0,
                "final_decode_calls": 1,
                "attention_probability_reads": 0,
                "qk_reads": 0,
            },
            "step_ledger": steps,
            "evidence_scope": {
                "schema": contract.EVIDENCE_SCOPE_SCHEMA,
                "split_role": contract.SPLIT_ROLE,
                "generation_only": True,
                "scoring_authorized": False,
                "quality_outcomes_present": False,
                "prompt_csv_sha256": provenance["prompt_csv_sha256"],
                "prompt_manifest_canonical_sha256": provenance[
                    "prompt_manifest_canonical_sha256"
                ],
                "prompt_rows_sha256": provenance["prompt_rows_sha256"],
                "contract_sha256": contract.CONTRACT_SHA256,
                "task_sha256": task["task_sha256"],
            },
        }

    def first_prompt_block(self):
        tasks = self.design["tasks"][: contract.TASKS_PER_PROMPT]
        return tasks, [self.make_sidecar(task) for task in tasks]

    def test_exact_165_task_design_and_six_orbit_cycle(self):
        design = self.design
        self.assertEqual(design["task_count"], 165)
        self.assertEqual(design["tasks_per_prompt"], 15)
        self.assertEqual(len({task["task_id"] for task in design["tasks"]}), 165)
        for task in design["tasks"]:
            unhashed = {key: value for key, value in task.items() if key != "task_sha256"}
            self.assertEqual(task["task_sha256"], contract.canonical_sha256(unhashed))

        expected_primary = [
            "p0",
            "p_axis_r1_pos",
            "p_axis_r1_neg",
            "p_diagonal_r1_pos",
            "p_diagonal_r1_neg",
            "p_axis_r2_pos",
            "p_axis_r2_neg",
            "r_axis_r1_pos",
            "r_axis_r1_neg",
            "r_diagonal_r1_pos",
            "r_diagonal_r1_neg",
            "r_axis_r2_pos",
            "r_axis_r2_neg",
        ]
        self.assertEqual(
            [row["action_id"] for row in contract.canonical_primary_action_bank()],
            expected_primary,
        )
        observed_cycle = []
        for prompt_index in range(contract.PROMPT_COUNT):
            block = design["tasks"][prompt_index * 15 : (prompt_index + 1) * 15]
            controls = block[-2:]
            self.assertEqual([row["action"]["family"] for row in controls], ["U", "X"])
            self.assertEqual(
                controls[1]["action"]["matched_feature_action_id"],
                controls[0]["action"]["matched_feature_action_id"],
            )
            observed_cycle.append(
                (controls[0]["action"]["orbit_name"], controls[0]["action"]["sign"])
            )
        self.assertEqual(
            observed_cycle,
            list(contract.SIGNED_ORBIT_CYCLE)
            + list(contract.SIGNED_ORBIT_CYCLE[:5]),
        )

    def test_design_hashes_are_reproducible_and_tampering_is_rejected(self):
        rebuilt = contract.build_engineering_design(self.prompt_rows, self.manifest)
        self.assertEqual(rebuilt, self.design)
        self.assertEqual(self.design["contract_hashes"], contract.canonical_hashes())
        self.assertEqual(
            self.design["design_sha256"],
            contract.canonical_sha256(
                {
                    key: value
                    for key, value in self.design.items()
                    if key != "design_sha256"
                }
            ),
        )
        tampered = copy.deepcopy(self.design)
        tampered["tasks"][1]["action"]["sign"] = -1
        with self.assertRaisesRegex(ValueError, "canonical contract"):
            contract.validate_engineering_design(
                tampered, self.prompt_rows, self.manifest
            )
        with self.assertRaisesRegex(ValueError, "finite canonical JSON"):
            contract.canonical_sha256({"nonfinite": float("nan")})
        equal_but_noncanonical = copy.deepcopy(self.design)
        equal_but_noncanonical["task_count"] = 165.0
        with self.assertRaisesRegex(ValueError, "canonical contract"):
            contract.validate_engineering_design(
                equal_but_noncanonical, self.prompt_rows, self.manifest
            )

    def test_exported_contract_mappings_are_immutable(self):
        mappings_and_keys = (
            (contract.ORBIT_OFFSETS, "axis-r1"),
            (contract.ORBIT_EDGE_COUNTS, "axis-r1"),
            (contract.EXECUTION_CONTRACT, "width"),
        )
        hashes_before = contract.canonical_hashes()
        for mapping, key in mappings_and_keys:
            with self.subTest(key=key):
                with self.assertRaises(TypeError):
                    mapping[key] = "tampered"
        self.assertEqual(contract.canonical_hashes(), hashes_before)
        self.assertEqual(
            contract.build_engineering_design(self.prompt_rows, self.manifest),
            self.design,
        )
        original_feature_block = contract.FEATURE_BLOCK
        try:
            contract.FEATURE_BLOCK = "forged.feature.block"
            with self.assertRaisesRegex(RuntimeError, "module constants drifted"):
                contract.build_engineering_design(self.prompt_rows, self.manifest)
        finally:
            contract.FEATURE_BLOCK = original_feature_block

    def test_prompt_count_global_seed_and_retirement_fail_closed(self):
        cases = []
        short_rows = copy.deepcopy(self.prompt_rows[:-1])
        cases.append((short_rows, copy.deepcopy(self.manifest), "11 rows"))
        mixed_seed_rows = copy.deepcopy(self.prompt_rows)
        mixed_seed_rows[-1]["seed"] = str(int(mixed_seed_rows[-1]["seed"]) + 1)
        cases.append((mixed_seed_rows, copy.deepcopy(self.manifest), "global seed"))
        not_retired = copy.deepcopy(self.manifest)
        not_retired["seed_registration"]["engineering_seed"]["retired_on_use"] = False
        cases.append((copy.deepcopy(self.prompt_rows), not_retired, "retired_on_use"))
        wrong_status = copy.deepcopy(self.manifest)
        wrong_status["seed_registration"]["engineering_seed"]["status"] = "available"
        cases.append(
            (copy.deepcopy(self.prompt_rows), wrong_status, "reserved_retired_on_use")
        )
        for rows, manifest, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    contract.build_engineering_design(rows, manifest)

    def test_seed_selection_namespace_counter_digest_and_value_fail_closed(self):
        cases = []
        wrong_namespace = copy.deepcopy(self.manifest)
        wrong_namespace["seed_registration"]["namespace"] = "forged-seed-namespace"
        cases.append((wrong_namespace, "namespace differs"))

        noncanonical_counter = copy.deepcopy(self.manifest)
        noncanonical_counter["seed_registration"]["engineering_seed"]["counter"] = "0"
        cases.append((noncanonical_counter, "canonical integer"))

        wrong_digest = copy.deepcopy(self.manifest)
        wrong_digest["seed_registration"]["engineering_seed"]["selection_digest"] = "0" * 64
        cases.append((wrong_digest, "selection digest is inconsistent"))

        wrong_value = copy.deepcopy(self.manifest)
        registered = wrong_value["seed_registration"]["engineering_seed"]
        registered["seed"] += 1
        changed_rows = copy.deepcopy(self.prompt_rows)
        for row in changed_rows:
            row["seed"] = str(registered["seed"])
        wrong_value["engineering"]["csv_sha256"] = hashlib.sha256(
            contract._prompt_csv_bytes(changed_rows)
        ).hexdigest()
        cases.append((wrong_value, "deterministic selection digest", changed_rows))

        for case in cases:
            manifest, message, *rows = case
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    contract.validate_prompt_assets(
                        rows[0] if rows else self.prompt_rows, manifest
                    )

    def test_prompt_rows_are_bound_to_the_registered_csv_bytes(self):
        changed_case = copy.deepcopy(self.prompt_rows)
        changed_case[0]["TEXT"] = changed_case[0]["TEXT"].upper()
        self.assertEqual(
            contract.normalize_prompt(changed_case[0]["TEXT"]),
            contract.normalize_prompt(self.prompt_rows[0]["TEXT"]),
        )
        with self.assertRaisesRegex(ValueError, "registered CSV SHA-256"):
            contract.build_engineering_design(changed_case, self.manifest)

    def test_prompt_manifest_binds_the_frozen_exclusion_inventory(self):
        prompt_assets = contract.validate_prompt_assets(
            self.prompt_rows, self.manifest
        )
        self.assertEqual(
            prompt_assets["exclusion_inventory_sha256"],
            self.manifest["exclusion_inventory"]["sha256"],
        )
        cases = {
            "schema": "forged_inventory_schema",
            "path": "eval-pipeline/prompts/forged.json",
            "sha256": "not-a-sha256",
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                manifest = copy.deepcopy(self.manifest)
                manifest["exclusion_inventory"][field] = value
                with self.assertRaises(ValueError):
                    contract.build_engineering_design(self.prompt_rows, manifest)

        changed_hash = copy.deepcopy(self.manifest)
        changed_hash["exclusion_inventory"]["sha256"] = "0" * 64
        changed_design = contract.build_engineering_design(
            self.prompt_rows, changed_hash
        )
        self.assertNotEqual(
            changed_design["design_sha256"], self.design["design_sha256"]
        )
        self.assertNotEqual(
            changed_design["tasks"][0]["task_sha256"],
            self.design["tasks"][0]["task_sha256"],
        )

    def test_sidecar_rejects_a_tampered_expected_task(self):
        task = copy.deepcopy(self.task_for("P"))
        task["prompt"]["source_category"] = "forged-category"
        record = self.make_sidecar(task)
        with self.assertRaisesRegex(ValueError, "task SHA-256 is inconsistent"):
            contract.validate_sidecar(record, task)

        forged_action = copy.deepcopy(self.task_for("P"))
        forged_action["action"]["family"] = "FORGED"
        forged_action["task_sha256"] = contract.canonical_sha256(
            {
                key: value
                for key, value in forged_action.items()
                if key != "task_sha256"
            }
        )
        forged_record = self.make_sidecar(forged_action)
        with self.assertRaisesRegex(ValueError, "registered task action"):
            contract.validate_sidecar(forged_record, forged_action)

    def test_p0_p_r_u_and_x_sidecars_are_accepted(self):
        for family in ("P0", "P", "R", "U", "X"):
            with self.subTest(family=family):
                task = self.task_for(family)
                summary = contract.validate_sidecar(self.make_sidecar(task), task)
                self.assertEqual(summary["task_id"], task["task_id"])
                self.assertEqual(summary["physical_no_op"], family == "P0")

    def test_feature_hook_and_non_feature_source_rules_fail_closed(self):
        feature_task = self.task_for("P")
        feature_record = self.make_sidecar(feature_task)
        feature_record["step_ledger"][0]["feature_hook"]["module_path"] = "down_blocks.0"
        with self.assertRaisesRegex(ValueError, "feature hook"):
            contract.validate_sidecar(feature_record, feature_task)

        for family in ("R", "U", "X"):
            task = self.task_for(family)
            record = self.make_sidecar(task)
            record["step_ledger"][0]["feature_hook"] = {
                "unexpected": "feature read"
            }
            with self.subTest(family=family):
                with self.assertRaisesRegex(ValueError, "must not contain a feature hook"):
                    contract.validate_sidecar(record, task)

    def test_random_counter_identity_and_uniqueness_fail_closed(self):
        task = self.task_for("R")
        identity_record = self.make_sidecar(task)
        identity_record["step_ledger"][0]["random_edge_counter"][
            "prompt_row_id"
        ] = "other-prompt"
        with self.assertRaisesRegex(ValueError, "random counter field"):
            contract.validate_sidecar(identity_record, task)

        duplicate_record = self.make_sidecar(task)
        duplicate_record["step_ledger"][1]["random_edge_counter"][
            "counter_set_sha256"
        ] = duplicate_record["step_ledger"][0]["random_edge_counter"][
            "counter_set_sha256"
        ]
        with self.assertRaisesRegex(ValueError, "registered D4 counter graph"):
            contract.validate_sidecar(duplicate_record, task)

        feature_task = self.task_for("P")
        feature_record = self.make_sidecar(feature_task)
        feature_record["step_ledger"][0]["random_edge_counter"] = {}
        with self.assertRaisesRegex(ValueError, "must not contain a random counter"):
            contract.validate_sidecar(feature_record, feature_task)

    def test_call_totals_and_step_ledger_fail_closed(self):
        task = self.task_for("U")
        mutations = (
            ("total calls", lambda row: row["call_totals"].__setitem__("unet_calls", 51)),
            ("ledger length", lambda row: row["step_ledger"].pop()),
            ("step order", lambda row: row["step_ledger"][0].__setitem__("step_index", 1)),
            ("extra call", lambda row: row["step_ledger"][0].__setitem__("extra_unet_calls", 1)),
            ("backward", lambda row: row["step_ledger"][0].__setitem__("backbone_backward_calls", 1)),
            ("decode", lambda row: row["step_ledger"][0].__setitem__("intermediate_decode_calls", 1)),
            ("nonfinite flag", lambda row: row["step_ledger"][0].__setitem__("finite", False)),
            ("cap flag", lambda row: row["step_ledger"][0].__setitem__("cap_hit", True)),
        )
        for name, mutate in mutations:
            record = self.make_sidecar(task)
            mutate(record)
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    contract.validate_sidecar(record, task)

    def test_actual_scheduler_arrays_are_hash_bound_and_exact(self):
        task = self.task_for("U")
        altered_timestep = self.make_sidecar(task)
        altered_timestep["trajectory"]["scheduler_timesteps"][0] -= 1.0
        with self.assertRaisesRegex(ValueError, "schedule hash|timesteps"):
            contract.validate_sidecar(altered_timestep, task)

        altered_sigma = self.make_sidecar(task)
        altered_sigma["trajectory"]["scheduler_sigmas"][-1] = 1e-6
        altered_sigma["trajectory"]["scheduler_schedule_sha256"] = (
            contract.canonical_sha256(
                {
                    "timesteps": altered_sigma["trajectory"][
                        "scheduler_timesteps"
                    ],
                    "sigmas": altered_sigma["trajectory"]["scheduler_sigmas"],
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "scheduler_schedule|sigmas|field"):
            contract.validate_sidecar(altered_sigma, task)

    def test_equal_but_noncanonical_json_types_are_rejected(self):
        task = self.task_for("U")
        mutations = (
            (
                "float total",
                lambda row: row["call_totals"].__setitem__("unet_calls", 50.0),
            ),
            (
                "boolean churn",
                lambda row: row["trajectory"].__setitem__(
                    "scheduler_churn", False
                ),
            ),
            (
                "float step count",
                lambda row: row["trajectory"].__setitem__(
                    "num_inference_steps", 50.0
                ),
            ),
            (
                "boolean step call",
                lambda row: row["step_ledger"][0].__setitem__("unet_calls", True),
            ),
        )
        for name, mutate in mutations:
            record = self.make_sidecar(task)
            mutate(record)
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    contract.validate_sidecar(record, task)

        float_action = self.make_sidecar(task)
        float_action["action"]["sign"] = float(float_action["action"]["sign"])
        for step in float_action["step_ledger"]:
            step["sign"] = float(step["sign"])
        with self.assertRaisesRegex(ValueError, "sidecar action differs"):
            contract.validate_sidecar(float_action, task)

    def test_ratio_moment_covariance_and_round_trip_bounds_fail_closed(self):
        task = self.task_for("X")
        threshold_mutations = (
            ("channel_mean_abs_error", contract.CHANNEL_MEAN_ABS_ERROR_MAX + 1e-12),
            (
                "channel_variance_relative_error",
                contract.CHANNEL_VARIANCE_RELATIVE_ERROR_MAX + 1e-12,
            ),
            (
                "channel_covariance_drift",
                contract.CHANNEL_COVARIANCE_DRIFT_MAX + 1e-12,
            ),
            (
                "pred_original_sample_relative_l2_error",
                contract.PRED_ORIGINAL_SAMPLE_RELATIVE_L2_MAX + 1e-12,
            ),
            (
                "expected_prev_sample_relative_l2_error",
                contract.EXPECTED_PREV_SAMPLE_RELATIVE_L2_MAX + 1e-12,
            ),
        )
        for field, value in threshold_mutations:
            record = self.make_sidecar(task)
            record["step_ledger"][0][field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    contract.validate_sidecar(record, task)

        inconsistent = self.make_sidecar(task)
        inconsistent["step_ledger"][0]["target_ratio_error"] = 1e-6
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            contract.validate_sidecar(inconsistent, task)
        missed = self.make_sidecar(task)
        missed["step_ledger"][0]["applied_update_ratio"] = 0.021
        missed["step_ledger"][0]["target_ratio_error"] = 0.001
        with self.assertRaisesRegex(ValueError, "missed the target"):
            contract.validate_sidecar(missed, task)

    def test_p0_diagnostics_and_active_basis_hashes_fail_closed(self):
        p0_task = self.task_for("P0")
        p0_record = self.make_sidecar(p0_task)
        p0_record["step_ledger"][0]["applied_update_ratio"] = 1e-12
        with self.assertRaisesRegex(ValueError, "exact zero"):
            contract.validate_sidecar(p0_record, p0_task)

        active_task = self.task_for("U")
        missing = self.make_sidecar(active_task)
        missing["step_ledger"][0]["basis_sha256"] = None
        with self.assertRaisesRegex(ValueError, "basis SHA-256"):
            contract.validate_sidecar(missing, active_task)
        duplicate = self.make_sidecar(active_task)
        duplicate["step_ledger"][1]["basis_sha256"] = duplicate[
            "step_ledger"
        ][0]["basis_sha256"]
        with self.assertRaisesRegex(ValueError, "basis hashes"):
            contract.validate_sidecar(duplicate, active_task)

    def test_native_max_abs_is_unbounded_diagnostic_but_relative_errors_are_gated(self):
        self.assertEqual(
            contract.EXECUTION_CONTRACT["native_round_trip_max_abs_policy"],
            "finite_diagnostic_only_not_a_pass_fail_bound",
        )
        task = self.task_for("X")
        record = self.make_sidecar(task)
        record["step_ledger"][0]["native_round_trip_max_abs_error"] = 1e12
        contract.validate_sidecar(record, task)

        negative = self.make_sidecar(task)
        negative["step_ledger"][0]["native_round_trip_max_abs_error"] = -1e-12
        with self.assertRaisesRegex(ValueError, "finite and at least"):
            contract.validate_sidecar(negative, task)

        bounded_relative = self.make_sidecar(task)
        bounded_relative["step_ledger"][0][
            "expected_prev_sample_relative_l2_error"
        ] = contract.EXPECTED_PREV_SAMPLE_RELATIVE_L2_MAX + 1e-12
        with self.assertRaisesRegex(ValueError, "expected prev_sample round-trip bound"):
            contract.validate_sidecar(bounded_relative, task)

        p0_task = self.task_for("P0")
        p0_record = self.make_sidecar(p0_task)
        p0_record["step_ledger"][0]["native_round_trip_max_abs_error"] = 1.0
        with self.assertRaisesRegex(ValueError, "exact zero"):
            contract.validate_sidecar(p0_record, p0_task)

    def test_evidence_scope_rejects_scoring_and_quality_outcomes(self):
        task = self.task_for("U")
        for field in ("scoring_authorized", "quality_outcomes_present"):
            record = self.make_sidecar(task)
            record["evidence_scope"][field] = True
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "evidence scope"):
                    contract.validate_sidecar(record, task)
        record = self.make_sidecar(task)
        record["scores"] = {"reward": 1.0}
        with self.assertRaisesRegex(ValueError, "fields differ"):
            contract.validate_sidecar(record, task)

    def test_cache_and_later_state_keys_are_rejected_recursively(self):
        task = self.task_for("P")
        for forbidden_key in ("cached_feature", "later-step-state"):
            record = self.make_sidecar(task)
            record["step_ledger"][0]["feature_hook"][forbidden_key] = "forbidden"
            with self.subTest(forbidden_key=forbidden_key):
                with self.assertRaisesRegex(ValueError, "forbidden later-state/cache key"):
                    contract.validate_sidecar(record, task)

    def test_complete_prompt_block_is_accepted(self):
        tasks, records = self.first_prompt_block()
        evidence = contract.validate_prompt_block(records, tasks)
        self.assertEqual(evidence["schema"], contract.BLOCK_EVIDENCE_SCHEMA)
        self.assertEqual(evidence["record_count"], contract.TASKS_PER_PROMPT)
        self.assertRegex(
            evidence["antithetic_step0_basis_pairs_sha256"], r"^[0-9a-f]{64}$"
        )
        unhashed = {
            key: value
            for key, value in evidence.items()
            if key != "block_evidence_sha256"
        }
        self.assertEqual(
            evidence["block_evidence_sha256"], contract.canonical_sha256(unhashed)
        )

    def test_random_positive_and_negative_actions_share_counter_sets(self):
        tasks, records = self.first_prompt_block()
        contract.validate_prompt_block(records, tasks)
        random_by_orbit = {}
        for record in records:
            action = record["action"]
            if action["family"] == "R":
                random_by_orbit.setdefault(action["orbit_name"], {})[
                    action["sign"]
                ] = [
                    step["random_edge_counter"]["counter_set_sha256"]
                    for step in record["step_ledger"]
                ]
        self.assertEqual(set(random_by_orbit), set(contract.ORBIT_OFFSETS))
        for signed_sets in random_by_orbit.values():
            self.assertEqual(signed_sets[-1], signed_sets[1])

        negative = next(
            record
            for record in records
            if record["action"]["family"] == "R"
            and record["action"]["orbit_name"] == "axis-r1"
            and record["action"]["sign"] == -1
        )
        negative["step_ledger"][0]["random_edge_counter"][
            "counter_set_sha256"
        ] = digest("sign-dependent-counter-set")
        with self.assertRaisesRegex(
            ValueError, "registered D4 counter graph|reuse the same counter sets"
        ):
            contract.validate_prompt_block(records, tasks)

    def test_antithetic_positive_and_negative_actions_share_step0_basis(self):
        tasks, records = self.first_prompt_block()
        evidence = contract.validate_prompt_block(records, tasks)
        self.assertRegex(
            evidence["antithetic_step0_basis_pairs_sha256"], r"^[0-9a-f]{64}$"
        )

        pairs = {}
        for record in records:
            action = record["action"]
            if action["family"] not in {"P", "R"} or action["physical_no_op"]:
                continue
            pairs.setdefault((action["family"], action["orbit_name"]), {})[
                action["sign"]
            ] = record["step_ledger"][0]["basis_sha256"]
        self.assertEqual(len(pairs), 6)
        for signed_bases in pairs.values():
            self.assertEqual(signed_bases[-1], signed_bases[1])

        negative = next(
            record
            for record in records
            if record["action"]["family"] == "P"
            and record["action"]["orbit_name"] == "axis-r1"
            and record["action"]["sign"] == -1
        )
        negative["step_ledger"][0]["basis_sha256"] = digest(
            "sign-dependent-step0-basis"
        )
        with self.assertRaisesRegex(ValueError, "share the same step-0 basis"):
            contract.validate_prompt_block(records, tasks)

    def test_prompt_block_count_and_task_identity_fail_closed(self):
        tasks, records = self.first_prompt_block()
        with self.assertRaisesRegex(ValueError, "exactly 15"):
            contract.validate_prompt_block(records[:-1], tasks)
        duplicate = copy.deepcopy(records)
        duplicate[-1]["task_id"] = duplicate[0]["task_id"]
        with self.assertRaisesRegex(ValueError, "duplicate task id"):
            contract.validate_prompt_block(duplicate, tasks)

    def test_prompt_block_trajectory_isolation_fail_closed(self):
        tasks, records = self.first_prompt_block()
        wrong_initial = copy.deepcopy(records)
        wrong_initial[1]["trajectory"]["initial_latent_sha256"] = digest(
            "wrong-initial"
        )
        wrong_initial[1]["step_ledger"][0]["latent_before_sha256"] = digest(
            "wrong-initial"
        )
        duplicate_trajectory = copy.deepcopy(records)
        duplicate_trajectory[1]["trajectory"]["trajectory_id"] = duplicate_trajectory[
            0
        ]["trajectory"]["trajectory_id"]
        duplicate_active_final = copy.deepcopy(records)
        duplicate_active_final[2]["trajectory"][
            "final_latent_sha256"
        ] = duplicate_active_final[1]["trajectory"]["final_latent_sha256"]
        duplicate_active_final[2]["step_ledger"][-1][
            "latent_after_sha256"
        ] = duplicate_active_final[1]["trajectory"]["final_latent_sha256"]
        active_equals_p0 = copy.deepcopy(records)
        active_equals_p0[1]["trajectory"][
            "final_latent_sha256"
        ] = active_equals_p0[0]["trajectory"]["final_latent_sha256"]
        active_equals_p0[1]["step_ledger"][-1][
            "latent_after_sha256"
        ] = active_equals_p0[0]["trajectory"]["final_latent_sha256"]
        duplicate_active_png = copy.deepcopy(records)
        duplicate_active_png[2]["trajectory"]["png_sha256"] = duplicate_active_png[
            1
        ]["trajectory"]["png_sha256"]
        cases = (
            (wrong_initial, "share one initial latent"),
            (duplicate_trajectory, "trajectory ids are not unique"),
            (duplicate_active_final, "final latent hashes are not unique"),
            (active_equals_p0, "equals physical P0"),
            (duplicate_active_png, "PNG hashes are not unique"),
        )
        for mutated_records, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    contract.validate_prompt_block(mutated_records, tasks)

    def test_step_latent_hash_chain_fails_closed(self):
        task = self.task_for("P")
        record = self.make_sidecar(task)
        record["step_ledger"][17]["latent_before_sha256"] = digest(
            "reused-later-state"
        )
        with self.assertRaisesRegex(ValueError, "breaks the trajectory chain"):
            contract.validate_sidecar(record, task)

        record = self.make_sidecar(task)
        record["step_ledger"][17]["latent_after_sha256"] = record[
            "step_ledger"
        ][17]["latent_before_sha256"]
        with self.assertRaisesRegex(ValueError, "exact no-op"):
            contract.validate_sidecar(record, task)

        record = self.make_sidecar(task)
        record["trajectory"]["final_latent_sha256"] = digest("detached-final")
        with self.assertRaisesRegex(ValueError, "differs from the last step"):
            contract.validate_sidecar(record, task)

    def test_prompt_block_tasks_must_share_the_full_prompt_identity(self):
        tasks, records = self.first_prompt_block()
        tasks = copy.deepcopy(tasks)
        records = copy.deepcopy(records)
        tasks[1]["prompt"]["seed"] += 1
        tasks[1]["task_sha256"] = contract.canonical_sha256(
            {
                key: value
                for key, value in tasks[1].items()
                if key != "task_sha256"
            }
        )
        records[1]["prompt"] = copy.deepcopy(tasks[1]["prompt"])
        records[1]["evidence_scope"]["task_sha256"] = tasks[1]["task_sha256"]
        with self.assertRaisesRegex(ValueError, "share one prompt identity"):
            contract.validate_prompt_block(records, tasks)


if __name__ == "__main__":
    unittest.main()
