import copy
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_PIPELINE = ROOT / "eval-pipeline"
if str(EVAL_PIPELINE) not in sys.path:
    sys.path.insert(0, str(EVAL_PIPELINE))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load_module(
    "structural_control_audit_test", EVAL_PIPELINE / "audit_structural_control_run.py"
)
generate = sys.modules["generate"]


class StructuralControlAuditTest(unittest.TestCase):
    REVIEWED_COMMIT = "a" * 40
    AUTHORIZATION_COMMIT = "b" * 40

    def setUp(self):
        self.original_formal_run_path = audit.STRUCTURAL_CONTROL_FORMAL_RUN_PATH
        self.original_analysis_amendment_path = (
            audit.STRUCTURAL_CONTROL_ANALYSIS_AMENDMENT_PATH
        )
        self.real_git_bytes = audit._git_bytes
        self.timesteps = [float(value) for value in range(50, 0, -1)]
        self.sigmas = [float(value) / 10.0 for value in range(51, 0, -1)]
        schedule_hash = audit.json_sha256(
            {"timesteps": self.timesteps, "sigmas": self.sigmas}
        )
        self.scheduler_config = {"beta_start": 0.00085, "prediction_type": "epsilon"}
        self.scheduler_runtime = {
            "config_sha256_v2": generate.scheduler_config_payload_sha256(
                self.scheduler_config
            ),
            "num_inference_steps": 50,
            "schedule_sha256": schedule_hash,
            "construction_init_noise_sigma": 5.1,
            "effective_init_noise_sigma": 5.0,
        }
        self.sampling = {
            "model": "stabilityai/stable-diffusion-xl-base-1.0",
            "model_revision": "b" * 40,
            "pipeline": "RepLDMSDXLPipeline",
            "resolution": 1024,
            "num_inference_steps": 50,
            "default_cfg_scale": 7.5,
            "negative_prompt": generate.DEFAULT_NEG,
            "power_calibrate": 0,
            "guidance_rescale": 0.0,
            "stage2": False,
            "torch_dtype": "float16",
            "variant": "fp16",
            "local_files_only": True,
        }
        self.determinism = {
            "deterministic_algorithms": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": True,
        }
        self.actions = [
            {"id": "no_op", "type": "none", "cfg_scale": 7.5},
            {"id": "cfg5", "type": "none", "cfg_scale": 5.0},
            {
                "id": "tfsa",
                "type": "legacy",
                "cfg_scale": 7.5,
                "scale": 0.004,
                "delay_steps": 3,
                "decay": ["cosine", 0.0, 3],
            },
            {
                "id": "freeu_a",
                "type": "freeu",
                "cfg_scale": 7.5,
                "freeu_schedule": {
                    "knots": [
                        {"position": 0.0, "parameters": [0.6, 0.4, 1.1, 1.2]},
                        {"position": 1.0, "parameters": [0.6, 0.4, 1.1, 1.2]},
                    ]
                },
                "implementation": "diffusers_constant_v1",
                "implementation_diffusers_version": "0.32.1",
                "expected_operator_calls_per_step": 9,
                "expected_resolution_idx_call_counts_per_step": [3, 3, 3],
                "expected_hidden_channel_call_counts_per_step": {"1280": 4, "640": 3, "320": 2},
                "expected_resolution_channel_call_counts_per_step": {"0:1280": 3, "1:1280": 1, "1:640": 2, "2:640": 1, "2:320": 2},
                "expected_operator_effect_call_counts_per_step": {"b1_s1": 3, "b2_s2": 3, "no_op": 3},
            },
            {
                "id": "freeu_b",
                "type": "freeu",
                "cfg_scale": 7.5,
                "freeu_schedule": {
                    "knots": [
                        {"position": 0.0, "parameters": [0.9, 0.2, 1.3, 1.4]},
                        {"position": 1.0, "parameters": [0.9, 0.2, 1.3, 1.4]},
                    ]
                },
                "implementation": "diffusers_constant_v1",
                "implementation_diffusers_version": "0.32.1",
                "expected_operator_calls_per_step": 9,
                "expected_resolution_idx_call_counts_per_step": [3, 3, 3],
                "expected_hidden_channel_call_counts_per_step": {"1280": 4, "640": 3, "320": 2},
                "expected_resolution_channel_call_counts_per_step": {"0:1280": 3, "1:1280": 1, "1:640": 2, "2:640": 1, "2:320": 2},
                "expected_operator_effect_call_counts_per_step": {"b1_s1": 3, "b2_s2": 3, "no_op": 3},
            },
            {
                "id": "freeu_c",
                "type": "freeu",
                "cfg_scale": 7.5,
                "freeu_schedule": {
                    "knots": [
                        {"position": 0.0, "parameters": [0.9, 0.2, 1.3, 1.4]},
                        {"position": 1.0, "parameters": [0.9, 0.2, 1.3, 1.4]},
                    ]
                },
                "implementation": "paper_adaptive_3676d36",
                "implementation_diffusers_version": "0.32.1",
                "source_commit": "c" * 40,
                "expected_operator_calls_per_step": 9,
                "expected_resolution_idx_call_counts_per_step": [3, 3, 3],
                "expected_hidden_channel_call_counts_per_step": {"1280": 4, "640": 3, "320": 2},
                "expected_resolution_channel_call_counts_per_step": {"0:1280": 3, "1:1280": 1, "1:640": 2, "2:640": 1, "2:320": 2},
                "expected_operator_effect_call_counts_per_step": {"b1_s1": 4, "b2_s2": 3, "no_op": 2},
            },
            {
                "id": "pladis",
                "type": "attention_baseline",
                "cfg_scale": 5.0,
                "implementation": "pladis_operator_port_248b9d1",
                "source_commit": "d" * 40,
                "expected_processor_group_counts": {"up": 36, "down": 24},
                "expected_processor_count": 60,
                "expected_processor_names_sha256": "e" * 64,
            },
            {
                "id": "gag",
                "type": "attention_baseline",
                "cfg_scale": 5.0,
                "implementation": "gag_eq13_reimplementation_2603.02531v2",
                "paper_id": "2603.02531v2",
                "expected_processor_group_counts": {"up": 36, "down": 24},
                "expected_processor_count": 60,
                "expected_processor_names_sha256": "e" * 64,
            },
        ]
        self.source = {
            "design": {"expected_task_count": len(self.actions)},
            "sampling": self.sampling,
            "scheduler_runtime": self.scheduler_runtime,
        }
        self.config = {
            "structural_control_registered": True,
            "devices": ["cuda:7"],
            "resolution": 1024,
            "power_calibrate": 0,
            "frequency_band_cutoffs": [0.08, 0.25],
            "stage_name": "stage1_1024",
            "stage2_enabled": False,
            "models_to_cpu": False,
            "multi_encoder": False,
            "multi_decoder": False,
            "num_resample_timesteps": 50,
            "init_rates": [0.8],
            "stage2_noise_source": None,
            "model_name": self.sampling["model"],
            "model_revision": self.sampling["model_revision"],
            "registered_sampling": self.sampling,
            "scheduler_runtime": self.scheduler_runtime,
            "runtime_provenance": {
                "generation_environment_hardware": {
                    "gpu": "NVIDIA GeForce RTX 3090",
                    "compute_capability": "8.6",
                },
                "generation_environment_determinism": self.determinism,
            },
        }
        self.records = [self.record(action) for action in self.actions]

    def tearDown(self):
        audit.STRUCTURAL_CONTROL_FORMAL_RUN_PATH = self.original_formal_run_path
        audit.STRUCTURAL_CONTROL_ANALYSIS_AMENDMENT_PATH = (
            self.original_analysis_amendment_path
        )

    def record(self, action):
        record = {
            "id": action["id"],
            "action_id": action["id"],
            "action": action,
            "registered_sampling": self.sampling,
            "scheduler_runtime": self.scheduler_runtime,
            "num_inference_steps": 50,
            "unet_calls_per_step": [1] * 50,
            "extra_unet_calls": 0,
            "guidance_scale": action["cfg_scale"],
            "guidance_rescale": 0.0,
            "height": 1024,
            "width": 1024,
            "power_calibrate": 0,
            "frequency_band_cutoffs": [0.08, 0.25],
            "stage": "stage1_1024",
            "stage2_enabled": False,
            "models_to_cpu": False,
            "multi_encoder": False,
            "multi_decoder": False,
            "num_resample_timesteps": 50,
            "init_rates": [0.8],
            "stage2_noise_source": None,
            "model_name": self.sampling["model"],
            "model_revision": self.sampling["model_revision"],
            "inference_seconds": 1.0,
            "peak_gpu_memory_bytes": 1024,
            "scheduler_name": "EulerDiscreteScheduler",
            "base_scheduler_name": "EulerDiscreteScheduler",
            "scheduler_config_sha256_v2": self.scheduler_runtime[
                "config_sha256_v2"
            ],
            "active_scheduler_config_sha256_v2": self.scheduler_runtime[
                "config_sha256_v2"
            ],
            "scheduler_schedule_sha256": self.scheduler_runtime["schedule_sha256"],
            "scheduler_timesteps": self.timesteps,
            "scheduler_sigmas": self.sigmas,
            "scheduler_construction_init_noise_sigma": 5.1,
            "scheduler_effective_init_noise_sigma": 5.0,
            "scheduler_init_noise_sigma": 5.1,
            "scheduler_order": 1,
            "scheduler_kwargs": {},
            "scheduler_config": self.scheduler_config,
            "active_scheduler_config": self.scheduler_config,
            "device": "cuda:7",
            "worker_device_provenance": {
                "gpu": "NVIDIA GeForce RTX 3090",
                "compute_capability": "8.6",
                "total_memory_bytes": 1024,
                "requested_device": "cuda:7",
                "logical_device_index": 7,
                "physical_device_index": 7,
                "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
                "pci_bus_id": "00000000:01:00.0",
                "cuda_visible_devices": None,
            },
            "worker_determinism_provenance": self.determinism,
            **self.config["runtime_provenance"],
            "model_load_provenance": {
                "torch_dtype": "float16",
                "variant": "fp16",
                "local_files_only": True,
                "revision": "b" * 40,
            },
            "freeu_schedule": None,
            "freeu_runtime": [],
            "freeu_operator_runtime": None,
            "freeu_preserve_moments": False,
            "freeu_implementation": None,
            "freeu_source_commit": None,
            "freeu_implementation_diffusers_version": None,
            "attention_baseline_implementation": None,
            "attention_baseline_source_commit": None,
            "attention_baseline_paper_id": None,
            "attention_baseline_topology": None,
            "attn_guidance_scale": 0.0,
            "attn_guidance_density": "all",
            "attn_guidance_decay": None,
            "attention_guidance_runtime": [],
        }
        if action["type"] == "legacy":
            record.update(
                {
                    "attn_guidance_scale": 0.004,
                    "attn_guidance_density": [1] * 47 + [0] * 3,
                    "attn_guidance_decay": ["cosine", 0.0, 3],
                    "attention_guidance_runtime": [
                        {
                            "step_index": step,
                            "t_index": 49 - step,
                            "active": step >= 3,
                            "applied_scale": (
                                0.0
                                if step < 3
                                else 0.004
                                * (
                                    (math.cos(math.pi * (step - 3) / 46.0) + 1.0)
                                    / 2.0
                                )
                                ** 3
                            ),
                        }
                        for step in range(50)
                    ],
                }
            )
        if action["type"] == "freeu":
            schedule = generate.freeu_runtime(action)
            record.update(
                {
                    "freeu_schedule": action["freeu_schedule"],
                    "freeu_implementation": action["implementation"],
                    "freeu_source_commit": action.get("source_commit"),
                    "freeu_implementation_diffusers_version": action[
                        "implementation_diffusers_version"
                    ],
                    "freeu_runtime": [
                        {
                            "step_index": step,
                            "parameters": list(
                                schedule.at(step / 49.0).as_tuple()
                            ),
                        }
                        for step in range(50)
                    ],
                    "freeu_operator_runtime": {
                        "implementation": action["implementation"],
                        "operator_calls_total": 450,
                        "resolution_idx_call_counts": {
                            "0": 150,
                            "1": 150,
                            "2": 150,
                        },
                        "hidden_channel_call_counts": {
                            "1280": 200,
                            "640": 150,
                            "320": 100,
                        },
                        "resolution_channel_call_counts": {
                            "0:1280": 150,
                            "1:1280": 50,
                            "1:640": 100,
                            "2:640": 50,
                            "2:320": 100,
                        },
                        "operator_effect_call_counts": {
                            effect: count * 50
                            for effect, count in action[
                                "expected_operator_effect_call_counts_per_step"
                            ].items()
                        },
                    },
                }
            )
        if action["type"] == "attention_baseline":
            record.update(
                {
                    "attention_baseline_implementation": action["implementation"],
                    "attention_baseline_source_commit": action.get("source_commit"),
                    "attention_baseline_paper_id": action.get("paper_id"),
                    "attention_baseline_topology": {
                        "group_counts": action["expected_processor_group_counts"],
                        "processor_count": 60,
                        "processor_names_sha256": "e" * 64,
                        "processors_called": 60,
                        "processor_calls_total": 3000,
                        "processor_call_count_min": 50,
                        "processor_call_count_max": 50,
                    },
                }
            )
        return record

    def write_analysis_amendment(self, root):
        path = pathlib.Path(root) / "analysis_amendment.yaml"
        base_files = audit._base_analysis_files(
            audit.STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT
        )
        replacements = {}
        for relative_path in audit.STRUCTURAL_CONTROL_ANALYSIS_REPLACEMENTS:
            replacements[relative_path] = {
                "base_sha256": base_files[relative_path],
                "amended_sha256": hashlib.sha256(
                    (ROOT / relative_path).read_bytes()
                ).hexdigest(),
            }
        payload = {
            "schema": audit.STRUCTURAL_CONTROL_ANALYSIS_AMENDMENT_SCHEMA,
            "status": audit.STRUCTURAL_CONTROL_AMENDMENT_STATUS,
            "reviewer": "independent-analysis-reviewer",
            "reviewed_commit": self.REVIEWED_COMMIT,
            "base_commit": audit.STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT,
            "result_access_before_authorization": False,
            "scoring_started_before_authorization": False,
            "unchanged_invariants": {
                key: True for key in audit.STRUCTURAL_CONTROL_UNCHANGED_INVARIANTS
            },
            "authorizations": {
                key: False for key in audit.STRUCTURAL_CONTROL_UNAUTHORIZED_USES
            },
            "replacements": replacements,
        }
        path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
        return path, payload

    def blocked_amendment_bytes(self, authorized_bytes):
        replacements = (
            (b"status: authorized_pre_score", b"status: blocked_pending_independent_review"),
            (
                b"reviewer: independent-analysis-reviewer",
                b"reviewer: null",
            ),
            (
                f"reviewed_commit: {self.REVIEWED_COMMIT}".encode("ascii"),
                b"reviewed_commit: null",
            ),
        )
        candidate_bytes = authorized_bytes
        for authorized_line, blocked_line in replacements:
            self.assertEqual(candidate_bytes.count(authorized_line), 1)
            candidate_bytes = candidate_bytes.replace(
                authorized_line, blocked_line, 1
            )
        return candidate_bytes

    @contextmanager
    def authorized_git_state(
        self,
        amendment_path,
        *,
        head_commit=None,
        committed_amendment_bytes=None,
        reviewed_overrides=None,
        head_overrides=None,
        ancestry_returncode=0,
    ):
        amendment_path = pathlib.Path(amendment_path).resolve()
        amendment_relative = amendment_path.relative_to(ROOT.resolve()).as_posix()
        if committed_amendment_bytes is None:
            committed_amendment_bytes = amendment_path.read_bytes()
        reviewed_overrides = dict(reviewed_overrides or {})
        head_overrides = dict(head_overrides or {})
        head_commit = head_commit or self.AUTHORIZATION_COMMIT
        candidate_bytes = self.blocked_amendment_bytes(amendment_path.read_bytes())

        def git_bytes(commit, relative_path):
            if commit == audit.STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT:
                return self.real_git_bytes(commit, relative_path)
            if commit == self.REVIEWED_COMMIT:
                if relative_path == amendment_relative:
                    result = reviewed_overrides.get(relative_path, candidate_bytes)
                else:
                    result = reviewed_overrides.get(
                        relative_path, (ROOT / relative_path).read_bytes()
                    )
                if isinstance(result, BaseException):
                    raise result
                return result
            if commit == head_commit:
                if relative_path == amendment_relative:
                    return committed_amendment_bytes
                return head_overrides.get(
                    relative_path, (ROOT / relative_path).read_bytes()
                )
            raise AssertionError(f"unexpected git blob request: {commit}:{relative_path}")

        def require_ancestor(ancestor, descendant, *, label, strict):
            if strict and ancestor == descendant:
                raise ValueError(
                    f"{label} must use a strict two-commit authorization flow"
                )
            if ancestry_returncode != 0:
                raise ValueError(f"{label} ancestry verification failed")

        with mock.patch.object(
            audit, "_current_head_commit", return_value=head_commit
        ), mock.patch.object(
            audit, "_git_bytes", side_effect=git_bytes
        ), mock.patch.object(
            audit, "_require_git_ancestor", side_effect=require_ancestor
        ), mock.patch.object(
            audit, "_validate_analysis_authorization_topology"
        ):
            yield

    def write_pre_score_inputs(self, root):
        root = pathlib.Path(root)
        run_dir = root / "run"
        run_dir.mkdir()
        audit.STRUCTURAL_CONTROL_FORMAL_RUN_PATH = str(run_dir)
        (run_dir / "config.json").write_text(
            json.dumps(
                {"git_commit": audit.STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT}
            ),
            encoding="utf-8",
        )
        (run_dir / "manifest.jsonl").write_text(
            "".join(
                json.dumps({"id": f"fixture-{index}"}) + "\n"
                for index in range(audit.STRUCTURAL_CONTROL_EXPECTED_TASKS)
            ),
            encoding="utf-8",
        )
        amendment_path, _ = self.write_analysis_amendment(root)
        audit.STRUCTURAL_CONTROL_ANALYSIS_AMENDMENT_PATH = str(amendment_path)
        return run_dir, amendment_path

    def write_scoring_launcher_inputs(self, root):
        run_dir, amendment_path = self.write_pre_score_inputs(root)
        seal_path = run_dir / audit.STRUCTURAL_CONTROL_PRE_SCORE_SEAL_NAME
        seal_path.write_bytes(b"sealed fixture\n")
        return run_dir, amendment_path, seal_path

    @contextmanager
    def mocked_scoring_validation(self, run_dir, amendment_path, seal_path):
        head_commit = "c" * 40
        with mock.patch.object(
            audit,
            "_validate_scoring_inputs",
            return_value=(
                amendment_path.resolve(),
                seal_path.resolve(),
                head_commit,
            ),
        ), mock.patch.object(
            audit, "_validate_scoring_sources", return_value=head_commit
        ):
            yield head_commit

    def test_complete_structural_record_contract_passes(self):
        report = audit.audit_structural_records(
            self.config, self.source, self.records, self.actions
        )
        self.assertTrue(report["structural_control_contract_passed"])
        self.assertEqual(report["matched_unet_calls"], "50x1")
        self.assertFalse(report["quality_results_inspected"])

    def test_png_collapse_validator_is_result_blind_and_relabel_invariant(self):
        records = []
        for prompt_index in (0, 1):
            for action_id, image_hash in (
                ("left", "a" * 64),
                ("right", "a" * 64 if prompt_index == 0 else "b" * 64),
            ):
                records.append(
                    {
                        "id": f"p{prompt_index}_seed7_a{action_id}",
                        "prompt_index": prompt_index,
                        "seed": 7,
                        "action_id": action_id,
                        "image_sha256": image_hash,
                    }
                )
        passed = audit.validate_action_png_collapse(
            records, ["left", "right"], expected_blocks=2
        )
        self.assertIs(passed, True)

        relabeled = copy.deepcopy(records)
        labels = {"left": "method_z", "right": "method_a"}
        for record in relabeled:
            record["action_id"] = labels[record["action_id"]]
        self.assertIs(
            audit.validate_action_png_collapse(
                relabeled, ["method_a", "method_z"], expected_blocks=2
            ),
            True,
        )

        records[-1]["image_sha256"] = "a" * 64
        with self.assertRaises(ValueError) as collapse:
            audit.validate_action_png_collapse(
                records, ["left", "right"], expected_blocks=2
            )
        self.assertEqual(
            str(collapse.exception), "structural outcome collapse validation failed"
        )
        self.assertNotIn("left", str(collapse.exception))
        self.assertNotIn("right", str(collapse.exception))
        self.assertNotIn("2", str(collapse.exception))

        action_ids = ["left", "right"]
        for record in records:
            record["execution_rank"] = audit.expected_execution_ranks(
                record["prompt_index"], record["seed"], action_ids
            )[record["action_id"]]
        audit.validate_execution_ranks(records, action_ids)
        records[0]["execution_rank"], records[1]["execution_rank"] = (
            records[1]["execution_rank"],
            records[0]["execution_rank"],
        )
        with self.assertRaisesRegex(ValueError, "action-order-v1"):
            audit.validate_execution_ranks(records, action_ids)

    def test_analysis_amendment_is_strict_and_tamper_evident(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            amendment_path, payload = self.write_analysis_amendment(temporary)
            with self.authorized_git_state(amendment_path):
                validated = audit.validate_analysis_amendment(
                    amendment_path,
                    base_commit=audit.STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT,
                )
            self.assertEqual(validated, payload)
            source = yaml.safe_load(
                (ROOT / audit.STRUCTURAL_CONTROL_ACTIONS_PATH).read_text(
                    encoding="utf-8"
                )
            )
            registration = yaml.safe_load(
                (
                    ROOT / generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE
                ).read_text(encoding="utf-8")
            )
            effective = audit._validate_analysis_implementation(
                source,
                registration,
                audit.STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT,
                replacements=validated["replacements"],
            )
            for relative_path in audit.STRUCTURAL_CONTROL_ANALYSIS_REPLACEMENTS:
                self.assertEqual(
                    effective["files"][relative_path],
                    payload["replacements"][relative_path]["amended_sha256"],
                )

            tampered = copy.deepcopy(payload)
            replacement_path = next(iter(tampered["replacements"]))
            tampered["replacements"][replacement_path]["amended_sha256"] = "f" * 64
            amendment_path.write_text(
                yaml.safe_dump(tampered, sort_keys=True), encoding="utf-8"
            )
            with self.authorized_git_state(amendment_path), self.assertRaisesRegex(
                ValueError, "reviewed analysis replacement blob differs"
            ):
                audit.validate_analysis_amendment(
                    amendment_path,
                    base_commit=audit.STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT,
                )

            tampered = copy.deepcopy(payload)
            tampered["result_access_before_authorization"] = True
            amendment_path.write_text(
                yaml.safe_dump(tampered, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "not result-blind"):
                audit.validate_analysis_amendment(
                    amendment_path,
                    base_commit=audit.STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT,
                )
            with self.assertRaisesRegex(ValueError, "not readable YAML"):
                audit.validate_analysis_amendment(
                    pathlib.Path(temporary) / "absent.yaml",
                    base_commit=audit.STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT,
                )

    def test_analysis_authorization_requires_two_commits_and_frozen_scope(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            amendment_path, payload = self.write_analysis_amendment(temporary)

            scalar_cases = (
                ("status", "blocked_candidate", "authorized_pre_score"),
                ("reviewer", "", "safe ASCII identity"),
                ("reviewer", "reviewer name", "safe ASCII identity"),
                ("reviewer", "审稿人", "safe ASCII identity"),
                ("reviewer", "r" * 129, "safe ASCII identity"),
                ("reviewed_commit", "abc", "full lowercase 40-hex"),
                ("result_access_before_authorization", True, "not result-blind"),
                (
                    "scoring_started_before_authorization",
                    True,
                    "authorized before scoring",
                ),
            )
            for field, value, message in scalar_cases:
                with self.subTest(field=field):
                    changed = copy.deepcopy(payload)
                    changed[field] = value
                    amendment_path.write_text(
                        yaml.safe_dump(changed, sort_keys=True), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        audit.validate_analysis_amendment(
                            amendment_path,
                            base_commit=audit.STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT,
                        )

            for invariant in audit.STRUCTURAL_CONTROL_UNCHANGED_INVARIANTS:
                with self.subTest(invariant=invariant):
                    changed = copy.deepcopy(payload)
                    changed["unchanged_invariants"][invariant] = False
                    amendment_path.write_text(
                        yaml.safe_dump(changed, sort_keys=True), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(ValueError, "frozen study invariant"):
                        audit.validate_analysis_amendment(
                            amendment_path,
                            base_commit=audit.STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT,
                        )

            for authorization in audit.STRUCTURAL_CONTROL_UNAUTHORIZED_USES:
                with self.subTest(authorization=authorization):
                    changed = copy.deepcopy(payload)
                    changed["authorizations"][authorization] = True
                    amendment_path.write_text(
                        yaml.safe_dump(changed, sort_keys=True), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(ValueError, "unauthorized downstream use"):
                        audit.validate_analysis_amendment(
                            amendment_path,
                            base_commit=audit.STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT,
                        )

            amendment_path.write_text(
                yaml.safe_dump(payload, sort_keys=True), encoding="utf-8"
            )
            with self.authorized_git_state(
                amendment_path, ancestry_returncode=1
            ), self.assertRaisesRegex(ValueError, "ancestry verification failed"):
                audit.validate_analysis_amendment(
                    amendment_path,
                    base_commit=audit.STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT,
                )

            amendment_relative = amendment_path.resolve().relative_to(
                ROOT.resolve()
            ).as_posix()
            with self.authorized_git_state(
                amendment_path,
                reviewed_overrides={
                    amendment_relative: ValueError("candidate does not exist")
                },
            ), self.assertRaisesRegex(ValueError, "blocked amendment candidate is missing"):
                audit.validate_analysis_amendment(
                    amendment_path,
                    base_commit=audit.STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT,
                )

            authorized_bytes = amendment_path.read_bytes()
            candidate_bytes = self.blocked_amendment_bytes(authorized_bytes)
            for label, malformed_candidate in (
                (
                    "missing",
                    candidate_bytes.replace(b"reviewer: null", b"reviewer: ~", 1),
                ),
                ("duplicate", candidate_bytes + b"reviewer: null\n"),
            ):
                with self.subTest(candidate_authorization_line=label):
                    with self.authorized_git_state(
                        amendment_path,
                        reviewed_overrides={
                            amendment_relative: malformed_candidate
                        },
                    ), self.assertRaisesRegex(ValueError, "exactly once"):
                        audit.validate_analysis_amendment(
                            amendment_path,
                            base_commit=(
                                audit.STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT
                            ),
                        )

            for label, drifted_authorization in (
                ("comment", b"# unauthorized drift\n" + authorized_bytes),
                (
                    "whitespace",
                    authorized_bytes.replace(b"schema: ", b"schema:  ", 1),
                ),
            ):
                with self.subTest(authorization_drift=label):
                    amendment_path.write_bytes(drifted_authorization)
                    with self.authorized_git_state(
                        amendment_path,
                        committed_amendment_bytes=drifted_authorization,
                        reviewed_overrides={amendment_relative: candidate_bytes},
                    ), self.assertRaisesRegex(ValueError, "outside the three permitted"):
                        audit.validate_analysis_amendment(
                            amendment_path,
                            base_commit=(
                                audit.STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT
                            ),
                        )
            amendment_path.write_bytes(authorized_bytes)

            with self.authorized_git_state(
                amendment_path,
                reviewed_overrides={
                    amendment_relative: amendment_path.read_bytes()
                },
            ), self.assertRaisesRegex(ValueError, "not blocked and unauthored"):
                audit.validate_analysis_amendment(
                    amendment_path,
                    base_commit=audit.STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT,
                )

            drifted_candidate = copy.deepcopy(payload)
            drifted_candidate.update(
                {
                    "status": audit.STRUCTURAL_CONTROL_AMENDMENT_CANDIDATE_STATUS,
                    "reviewer": None,
                    "reviewed_commit": None,
                }
            )
            drifted_candidate["unchanged_invariants"]["metrics"] = False
            with self.authorized_git_state(
                amendment_path,
                reviewed_overrides={
                    amendment_relative: yaml.safe_dump(
                        drifted_candidate, sort_keys=True
                    ).encode("utf-8")
                },
            ), self.assertRaisesRegex(ValueError, "candidate content differs"):
                audit.validate_analysis_amendment(
                    amendment_path,
                    base_commit=audit.STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT,
                )

            replacement_path = next(iter(audit.STRUCTURAL_CONTROL_ANALYSIS_REPLACEMENTS))
            with self.authorized_git_state(
                amendment_path,
                reviewed_overrides={replacement_path: b"unreviewed replacement"},
            ), self.assertRaisesRegex(ValueError, "reviewed analysis replacement blob"):
                audit.validate_analysis_amendment(
                    amendment_path,
                    base_commit=audit.STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT,
                )

            with self.authorized_git_state(
                amendment_path, committed_amendment_bytes=b"stale authorization"
            ), self.assertRaisesRegex(ValueError, "amendment bytes differ from current HEAD"):
                audit.validate_analysis_amendment(
                    amendment_path,
                    base_commit=audit.STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT,
                )

            unchanged_path = next(
                path
                for path in generate.STRUCTURAL_CONTROL_ANALYSIS_PATHS
                if path not in audit.STRUCTURAL_CONTROL_ANALYSIS_REPLACEMENTS
            )
            with self.authorized_git_state(
                amendment_path,
                head_overrides={unchanged_path: b"uncommitted effective analysis"},
            ), self.assertRaisesRegex(
                ValueError, "effective analysis implementation differs from current HEAD"
            ):
                audit.validate_analysis_amendment(
                    amendment_path,
                    base_commit=audit.STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT,
                )

    def test_git_ancestry_check_requires_a_strict_review_predecessor(self):
        with mock.patch.object(
            audit.subprocess, "run", return_value=mock.Mock(returncode=0)
        ) as run:
            audit._require_git_ancestor(
                self.REVIEWED_COMMIT,
                self.AUTHORIZATION_COMMIT,
                label="fixture review",
                strict=True,
            )
            run.assert_called_once()
        with mock.patch.object(audit.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "strict two-commit"):
                audit._require_git_ancestor(
                    self.REVIEWED_COMMIT,
                    self.REVIEWED_COMMIT,
                    label="fixture review",
                    strict=True,
                )
            run.assert_not_called()
        with mock.patch.object(
            audit.subprocess, "run", return_value=mock.Mock(returncode=1)
        ), self.assertRaisesRegex(ValueError, "ancestry verification failed"):
            audit._require_git_ancestor(
                self.REVIEWED_COMMIT,
                self.AUTHORIZATION_COMMIT,
                label="fixture review",
                strict=True,
            )

    def test_analysis_authorization_topology_requires_one_direct_file_change(self):
        amendment_relative = "eval-pipeline/configs/amendment.yaml"
        parent_row = (
            f"{self.AUTHORIZATION_COMMIT} {self.REVIEWED_COMMIT}\n".encode("ascii")
        )

        def modification(
            path=amendment_relative,
            *,
            old_mode="100644",
            new_mode="100644",
            status="M",
            old_hash="a" * 40,
            new_hash="b" * 40,
            extra_paths=(),
        ):
            metadata = (
                f":{old_mode} {new_mode} {old_hash} {new_hash} {status}"
            ).encode("ascii")
            return b"\0".join(
                (metadata, path.encode("utf-8"), *(p.encode("utf-8") for p in extra_paths))
            ) + b"\0"

        with mock.patch.object(
            audit, "_git_command_bytes", side_effect=[parent_row, modification()]
        ) as git_command:
            audit._validate_analysis_authorization_topology(
                self.REVIEWED_COMMIT,
                self.AUTHORIZATION_COMMIT,
                amendment_relative,
            )
        self.assertEqual(git_command.call_count, 2)

        for label, invalid_parent in (
            (
                "intermediate",
                f"{self.AUTHORIZATION_COMMIT} {'3' * 40}\n".encode("ascii"),
            ),
            (
                "merge",
                (
                    f"{self.AUTHORIZATION_COMMIT} {self.REVIEWED_COMMIT} "
                    f"{'4' * 40}\n"
                ).encode("ascii"),
            ),
        ):
            with self.subTest(parent_attack=label), mock.patch.object(
                audit, "_git_command_bytes", return_value=invalid_parent
            ), self.assertRaisesRegex(ValueError, "sole parent"):
                audit._validate_analysis_authorization_topology(
                    self.REVIEWED_COMMIT,
                    self.AUTHORIZATION_COMMIT,
                    amendment_relative,
                )

        ordinary = modification()
        attacks = (
            ("extra", ordinary + modification("extra.txt")),
            ("add", modification(old_mode="000000", status="A", old_hash="0" * 40)),
            ("delete", modification(new_mode="000000", status="D", new_hash="0" * 40)),
            (
                "rename",
                modification(
                    "old.yaml",
                    status="R100",
                    extra_paths=(amendment_relative,),
                ),
            ),
            (
                "copy",
                modification(
                    "source.yaml",
                    status="C100",
                    extra_paths=(amendment_relative,),
                ),
            ),
            ("mode", modification(new_mode="100755")),
            ("wrong_path", modification("other.yaml")),
            ("same_blob", modification(new_hash="a" * 40)),
        )
        for label, raw_diff in attacks:
            with self.subTest(diff_attack=label), mock.patch.object(
                audit,
                "_git_command_bytes",
                side_effect=[parent_row, raw_diff],
            ), self.assertRaisesRegex(ValueError, "authorization commit"):
                audit._validate_analysis_authorization_topology(
                    self.REVIEWED_COMMIT,
                    self.AUTHORIZATION_COMMIT,
                    amendment_relative,
                )

        with mock.patch.object(
            audit,
            "_git_command_bytes",
            side_effect=ValueError("parent verification failed"),
        ), self.assertRaisesRegex(ValueError, "verification failed"):
            audit._validate_analysis_authorization_topology(
                self.REVIEWED_COMMIT,
                self.AUTHORIZATION_COMMIT,
                amendment_relative,
            )

    @mock.patch.object(audit, "validate_analysis_amendment", return_value={})
    def test_pre_score_seal_is_one_shot_and_hash_bound(self, _validate_amendment):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            run_dir, amendment_path = self.write_pre_score_inputs(temporary)
            with mock.patch.object(
                audit, "_fsync_directory", wraps=audit._fsync_directory
            ) as fsync_directory:
                seal = audit.create_pre_score_seal(
                    run_dir,
                    ROOT / audit.STRUCTURAL_CONTROL_ACTIONS_PATH,
                    ROOT / generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE,
                    amendment_path,
                )
            fsync_directory.assert_called_once_with(run_dir.resolve())
            seal_path = run_dir / audit.STRUCTURAL_CONTROL_PRE_SCORE_SEAL_NAME
            self.assertEqual(
                seal["schema"], audit.STRUCTURAL_CONTROL_PRE_SCORE_SEAL_SCHEMA
            )
            self.assertTrue(seal["outcome_artifacts_absent_at_creation"])
            self.assertEqual(
                set(seal["bindings"]),
                {"config", "manifest", "actions", "registration", "analysis_amendment"},
            )
            audit.validate_pre_score_seal(
                seal_path,
                analysis_amendment_path=amendment_path,
                base_commit=audit.STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT,
            )
            with self.assertRaisesRegex(ValueError, "one-shot"):
                audit.create_pre_score_seal(
                    run_dir,
                    ROOT / audit.STRUCTURAL_CONTROL_ACTIONS_PATH,
                    ROOT / generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE,
                    amendment_path,
                )

            with (run_dir / "manifest.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"id": "tampered"}) + "\n")
            with self.assertRaisesRegex(ValueError, "manifest bytes differ"):
                audit.validate_pre_score_seal(
                    seal_path,
                    analysis_amendment_path=amendment_path,
                    base_commit=audit.STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT,
                )

        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            run_dir, amendment_path = self.write_pre_score_inputs(temporary)
            seal_path = run_dir / audit.STRUCTURAL_CONTROL_PRE_SCORE_SEAL_NAME
            seal_path.symlink_to(run_dir / "missing-seal-target.json")
            with self.assertRaisesRegex(ValueError, "one-shot"):
                audit.create_pre_score_seal(
                    run_dir,
                    ROOT / audit.STRUCTURAL_CONTROL_ACTIONS_PATH,
                    ROOT / generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE,
                    amendment_path,
                )

    @mock.patch.object(audit, "validate_analysis_amendment", return_value={})
    def test_pre_score_seal_rejects_noncanonical_amendment_before_publication(
        self, _validate_amendment
    ):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            run_dir, amendment_path = self.write_pre_score_inputs(temporary)
            alternate_amendment = pathlib.Path(temporary) / "alternate-amendment.yaml"
            alternate_amendment.write_bytes(amendment_path.read_bytes())
            seal_path = run_dir / audit.STRUCTURAL_CONTROL_PRE_SCORE_SEAL_NAME
            with self.assertRaisesRegex(ValueError, "canonical analysis amendment"):
                audit.create_pre_score_seal(
                    run_dir,
                    ROOT / audit.STRUCTURAL_CONTROL_ACTIONS_PATH,
                    ROOT / generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE,
                    alternate_amendment,
                )
            self.assertFalse(os.path.lexists(seal_path))

    @mock.patch.object(audit, "validate_analysis_amendment", return_value={})
    def test_pre_score_seal_rejects_outcomes_and_active_generation_lock(
        self, _validate_amendment
    ):
        forbidden_names = (
            "scores.jsonl",
            audit.STRUCTURAL_CONTROL_SCORING_ATTEMPT_NAME,
            audit.STRUCTURAL_CONTROL_SCORING_SUCCESS_NAME,
            audit.STRUCTURAL_CONTROL_AUDIT_OUTPUT_NAME,
            audit.STRUCTURAL_CONTROL_AUDIT_ATTEMPT_NAME,
            audit.STRUCTURAL_CONTROL_EVALUATION_BUNDLE,
            audit.STRUCTURAL_CONTROL_EVALUATION_ATTEMPT_NAME,
            *audit.STRUCTURAL_CONTROL_LEGACY_EVALUATION_OUTPUTS,
            f".{audit.STRUCTURAL_CONTROL_EVALUATION_BUNDLE}.staging",
            ".scores.jsonl.crash.tmp",
            ".structural_control_scoring_attempt.json.crash.tmp",
            ".structural_control_scoring_success.json.crash.tmp",
            ".run_audit.json.crash.tmp",
            ".structural_control_evaluation.json.crash.tmp",
            ".structural_control_contrasts.csv.crash.tmp",
            ".structural_control_pre_score_seal.json.crash.tmp",
        )
        for forbidden_name in forbidden_names:
            with self.subTest(forbidden_name=forbidden_name), tempfile.TemporaryDirectory(
                dir=ROOT
            ) as temporary:
                run_dir, amendment_path = self.write_pre_score_inputs(temporary)
                forbidden_path = run_dir / forbidden_name
                if forbidden_name in {
                    audit.STRUCTURAL_CONTROL_EVALUATION_BUNDLE,
                    f".{audit.STRUCTURAL_CONTROL_EVALUATION_BUNDLE}.staging",
                }:
                    forbidden_path.mkdir()
                else:
                    forbidden_path.write_bytes(b"partial outcome bytes")
                with self.assertRaisesRegex(ValueError, "artifacts absent"):
                    audit.create_pre_score_seal(
                        run_dir,
                        ROOT / audit.STRUCTURAL_CONTROL_ACTIONS_PATH,
                        ROOT / generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE,
                        amendment_path,
                    )

        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            run_dir, amendment_path = self.write_pre_score_inputs(temporary)
            with (run_dir / ".generate.lock").open("a+") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "generation or scoring"):
                    audit.create_pre_score_seal(
                        run_dir,
                        ROOT / audit.STRUCTURAL_CONTROL_ACTIONS_PATH,
                        ROOT / generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE,
                        amendment_path,
                    )
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            run_dir, amendment_path = self.write_pre_score_inputs(temporary)
            for allowed_name in (
                ".generate.lock",
                ".structural_control_audit.lock",
                ".structural_control_evaluation.lock",
                ".unrelated.tmp",
            ):
                (run_dir / allowed_name).write_bytes(b"not an outcome artifact")
            audit.create_pre_score_seal(
                run_dir,
                ROOT / audit.STRUCTURAL_CONTROL_ACTIONS_PATH,
                ROOT / generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE,
                amendment_path,
            )

    @mock.patch.object(audit, "validate_analysis_amendment", return_value={})
    def test_create_seal_cli_publishes_only_the_canonical_seal(
        self, _validate_amendment
    ):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            run_dir, amendment_path = self.write_pre_score_inputs(temporary)
            argv = [
                "audit_structural_control_run.py",
                "--run_dir",
                str(run_dir),
                "--actions",
                str(ROOT / audit.STRUCTURAL_CONTROL_ACTIONS_PATH),
                "--registration",
                str(ROOT / generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE),
                "--analysis-amendment",
                str(amendment_path),
                "--create-seal",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
                audit.main()
            self.assertTrue(
                (run_dir / audit.STRUCTURAL_CONTROL_PRE_SCORE_SEAL_NAME).is_file()
            )
            self.assertFalse((run_dir / "scores.jsonl").exists())
            self.assertFalse(
                (run_dir / audit.STRUCTURAL_CONTROL_AUDIT_OUTPUT_NAME).exists()
            )
            self.assertFalse(
                (run_dir / audit.STRUCTURAL_CONTROL_AUDIT_ATTEMPT_NAME).exists()
            )
            self.assertFalse(
                (run_dir / audit.STRUCTURAL_CONTROL_SCORING_ATTEMPT_NAME).exists()
            )
            self.assertFalse(
                (run_dir / audit.STRUCTURAL_CONTROL_SCORING_SUCCESS_NAME).exists()
            )

    def test_sealed_scoring_rejects_invalid_inputs_before_marker_or_child(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            run_dir, amendment_path, seal_path = self.write_scoring_launcher_inputs(
                temporary
            )
            with mock.patch.object(
                audit,
                "_validate_scoring_inputs",
                side_effect=ValueError("invalid sealed inputs"),
            ), mock.patch.object(audit.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(ValueError, "invalid sealed inputs"):
                    audit.launch_sealed_scoring(
                        run_dir,
                        ROOT / audit.STRUCTURAL_CONTROL_ACTIONS_PATH,
                        ROOT / generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE,
                        amendment_path,
                        seal_path,
                    )
            popen.assert_not_called()
            self.assertFalse(
                os.path.lexists(
                    run_dir / audit.STRUCTURAL_CONTROL_SCORING_ATTEMPT_NAME
                )
            )

        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            run_dir, amendment_path, seal_path = self.write_scoring_launcher_inputs(
                temporary
            )
            audit.STRUCTURAL_CONTROL_FORMAL_RUN_PATH = self.original_formal_run_path
            with mock.patch.object(audit.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(ValueError, "canonical non-symlink"):
                    audit.launch_sealed_scoring(
                        run_dir,
                        ROOT / audit.STRUCTURAL_CONTROL_ACTIONS_PATH,
                        ROOT / generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE,
                        amendment_path,
                        seal_path,
                    )
            popen.assert_not_called()

    def test_sealed_scoring_rejects_existing_scores_temps_and_marker_nodes(self):
        artifact_states = ("scores", "temp", "marker_symlink", "marker_fifo")
        for state in artifact_states:
            with self.subTest(state=state), tempfile.TemporaryDirectory(
                dir=ROOT
            ) as temporary:
                run_dir, amendment_path, seal_path = self.write_scoring_launcher_inputs(
                    temporary
                )
                marker_path = (
                    run_dir / audit.STRUCTURAL_CONTROL_SCORING_ATTEMPT_NAME
                )
                if state == "scores":
                    (run_dir / "scores.jsonl").write_bytes(b"partial score\n")
                elif state == "temp":
                    (run_dir / ".scores.jsonl.any punctuation!.tmp").write_bytes(
                        b"partial temp\n"
                    )
                elif state == "marker_symlink":
                    marker_path.symlink_to(run_dir / "missing-target")
                else:
                    os.mkfifo(marker_path)
                with self.mocked_scoring_validation(
                    run_dir, amendment_path, seal_path
                ), mock.patch.object(audit.subprocess, "Popen") as popen:
                    with self.assertRaisesRegex(
                        ValueError, "one-shot|temporary debris"
                    ):
                        audit.launch_sealed_scoring(
                            run_dir,
                            ROOT / audit.STRUCTURAL_CONTROL_ACTIONS_PATH,
                            ROOT / generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE,
                            amendment_path,
                            seal_path,
                        )
                popen.assert_not_called()

    def test_sealed_scoring_rejects_audit_and_evaluation_debris_before_attempt(self):
        debris_names = (
            audit.STRUCTURAL_CONTROL_AUDIT_OUTPUT_NAME,
            audit.STRUCTURAL_CONTROL_AUDIT_ATTEMPT_NAME,
            audit.STRUCTURAL_CONTROL_EVALUATION_ATTEMPT_NAME,
            audit.STRUCTURAL_CONTROL_EVALUATION_BUNDLE,
            *audit.STRUCTURAL_CONTROL_LEGACY_EVALUATION_OUTPUTS,
            ".run_audit.json.interrupted output.tmp",
            f".{audit.STRUCTURAL_CONTROL_EVALUATION_BUNDLE}.staging output",
            ".structural_control_evaluation.json.interrupted output.tmp",
            ".structural_control_contrasts.csv.interrupted output.tmp",
            ".structural_control_pre_score_seal.json.interrupted output.tmp",
        )
        directory_names = {
            audit.STRUCTURAL_CONTROL_EVALUATION_BUNDLE,
            f".{audit.STRUCTURAL_CONTROL_EVALUATION_BUNDLE}.staging output",
        }
        for debris_name in debris_names:
            with self.subTest(debris_name=debris_name), tempfile.TemporaryDirectory(
                dir=ROOT
            ) as temporary:
                run_dir, amendment_path, seal_path = self.write_scoring_launcher_inputs(
                    temporary
                )
                debris_path = run_dir / debris_name
                if debris_name in directory_names:
                    debris_path.mkdir()
                else:
                    debris_path.write_bytes(b"pre-existing post-seal debris")
                marker_path = (
                    run_dir / audit.STRUCTURAL_CONTROL_SCORING_ATTEMPT_NAME
                )
                with self.mocked_scoring_validation(
                    run_dir, amendment_path, seal_path
                ), mock.patch.object(audit.subprocess, "Popen") as popen:
                    with self.assertRaisesRegex(
                        ValueError, "score, audit, and evaluation artifacts absent"
                    ):
                        audit.launch_sealed_scoring(
                            run_dir,
                            ROOT / audit.STRUCTURAL_CONTROL_ACTIONS_PATH,
                            ROOT / generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE,
                            amendment_path,
                            seal_path,
                        )
                self.assertFalse(os.path.lexists(marker_path))
                popen.assert_not_called()

    def test_sealed_scoring_launch_is_durable_fixed_and_releases_lock(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            run_dir, amendment_path, seal_path = self.write_scoring_launcher_inputs(
                temporary
            )
            marker_path = run_dir / audit.STRUCTURAL_CONTROL_SCORING_ATTEMPT_NAME

            class Child:
                pid = 4321

                @staticmethod
                def wait():
                    return 0

            def popen_child(argv, **kwargs):
                marker = json.loads(marker_path.read_bytes())
                self.assertEqual(argv, audit._canonical_scoring_argv(run_dir))
                self.assertFalse(kwargs["shell"])
                self.assertEqual(kwargs["cwd"], str(ROOT.resolve()))
                capability = kwargs["env"][audit.STRUCTURAL_CONTROL_SCORING_CAPABILITY_ENV]
                self.assertEqual(
                    hashlib.sha256(capability.encode("ascii")).hexdigest(),
                    marker["launcher"]["capability_sha256"],
                )
                for key, value in audit.STRUCTURAL_CONTROL_SCORING_ENVIRONMENT.items():
                    self.assertEqual(kwargs["env"][key], value)
                with (run_dir / ".generate.lock").open("a+") as handle:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                (run_dir / "scores.jsonl").write_bytes(b"{}\n")
                return Child()

            with self.mocked_scoring_validation(
                run_dir, amendment_path, seal_path
            ), mock.patch.object(
                audit, "_validate_completed_scoring",
                return_value=audit.STRUCTURAL_CONTROL_SCORER_PROVENANCE_SHA256,
            ), mock.patch.object(
                audit.subprocess, "Popen", side_effect=popen_child
            ) as popen, mock.patch.object(
                audit.os, "open", wraps=audit.os.open
            ) as open_file, mock.patch.object(
                audit.os, "fsync", wraps=audit.os.fsync
            ) as fsync, mock.patch.object(
                audit, "_fsync_directory", wraps=audit._fsync_directory
            ) as fsync_directory:
                receipt_path = audit.launch_sealed_scoring(
                    run_dir,
                    ROOT / audit.STRUCTURAL_CONTROL_ACTIONS_PATH,
                    ROOT / generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE,
                    amendment_path,
                    seal_path,
                )
            popen.assert_called_once()
            self.assertEqual(
                receipt_path,
                run_dir / audit.STRUCTURAL_CONTROL_SCORING_SUCCESS_NAME,
            )
            for output_path in (marker_path, receipt_path):
                exclusive_calls = [
                    call
                    for call in open_file.call_args_list
                    if call.args and pathlib.Path(call.args[0]) == output_path
                    and len(call.args) > 1
                    and call.args[1] & audit.os.O_WRONLY
                ]
                self.assertEqual(len(exclusive_calls), 1)
                self.assertTrue(exclusive_calls[0].args[1] & audit.os.O_EXCL)
                self.assertTrue(exclusive_calls[0].args[1] & audit.os.O_NOFOLLOW)
            self.assertGreaterEqual(fsync.call_count, 4)
            self.assertGreaterEqual(fsync_directory.call_count, 2)
            receipt = json.loads(receipt_path.read_bytes())
            self.assertEqual(receipt["child"], {"pid": 4321, "returncode": 0})
            self.assertEqual(
                receipt["bindings"]["scoring_attempt"]["sha256"],
                hashlib.sha256(marker_path.read_bytes()).hexdigest(),
            )

    def test_sealed_scoring_child_failure_consumes_attempt_and_blocks_retry(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            run_dir, amendment_path, seal_path = self.write_scoring_launcher_inputs(
                temporary
            )
            child = mock.Mock(pid=7654)
            child.wait.return_value = 9
            with self.mocked_scoring_validation(
                run_dir, amendment_path, seal_path
            ), mock.patch.object(audit.subprocess, "Popen", return_value=child) as popen:
                with self.assertRaises(subprocess.CalledProcessError):
                    audit.launch_sealed_scoring(
                        run_dir,
                        ROOT / audit.STRUCTURAL_CONTROL_ACTIONS_PATH,
                        ROOT / generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE,
                        amendment_path,
                        seal_path,
                    )
                self.assertTrue(
                    (run_dir / audit.STRUCTURAL_CONTROL_SCORING_ATTEMPT_NAME).is_file()
                )
                self.assertFalse(
                    (run_dir / audit.STRUCTURAL_CONTROL_SCORING_SUCCESS_NAME).exists()
                )
                with self.assertRaisesRegex(ValueError, "one-shot"):
                    audit.launch_sealed_scoring(
                        run_dir,
                        ROOT / audit.STRUCTURAL_CONTROL_ACTIONS_PATH,
                        ROOT / generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE,
                        amendment_path,
                        seal_path,
                    )
            self.assertEqual(popen.call_count, 1)

    def test_scoring_attempt_fsync_failures_leave_terminal_nodes(self):
        for failure_point in ("file", "directory"):
            with self.subTest(failure_point=failure_point), tempfile.TemporaryDirectory(
                dir=ROOT
            ) as temporary:
                marker_path = pathlib.Path(temporary) / "attempt.json"
                if failure_point == "file":
                    failure = mock.patch.object(
                        audit.os, "fsync", side_effect=OSError("file fsync failed")
                    )
                else:
                    failure = mock.patch.object(
                        audit,
                        "_fsync_directory",
                        side_effect=OSError("directory fsync failed"),
                    )
                with failure, self.assertRaisesRegex(OSError, "fsync failed"):
                    audit._exclusive_durable_write(
                        marker_path, b"attempt\n", label="fixture marker"
                    )
                self.assertTrue(os.path.lexists(marker_path))
                with self.assertRaisesRegex(ValueError, "already exists"):
                    audit._exclusive_durable_write(
                        marker_path, b"retry\n", label="fixture marker"
                    )

    def test_completed_scoring_accepts_structural_rows_without_cfg_contract(self):
        scorer_outputs = {
            "imagereward": (
                "imagereward",
                "patch_ir_mean",
                "patch_ir_std",
                "patch_ir_n",
            ),
            "pixel": (
                "colorfulness",
                "laplacian_sharpness",
                "clipped_fraction",
                "mean_saturation",
                "contrast_std",
            ),
            "clip": ("clip_cosine", "clipscore"),
            "hps": ("hpsv2",),
            "aesthetic": ("aesthetic",),
            "iqa": ("topiq_nr",),
        }
        provenance = {
            "metrics": list(audit.STRUCTURAL_CONTROL_SCORE_METRICS),
            "scorers": [
                {
                    "name": name,
                    "output_keys": [
                        {"name": key, "direction": "higher"} for key in keys
                    ],
                }
                for name, keys in scorer_outputs.items()
            ],
        }
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            run_dir = pathlib.Path(temporary) / "run"
            run_dir.mkdir()
            manifest = [
                {"id": f"fixture-{index}"}
                for index in range(audit.STRUCTURAL_CONTROL_EXPECTED_TASKS)
            ]
            scores = [
                {
                    "id": row["id"],
                    "scorer_provenance": provenance,
                    **{
                        key: 0.5
                        for key in audit.STRUCTURAL_CONTROL_SCORE_OUTPUT_KEYS
                    },
                }
                for row in manifest
            ]
            (run_dir / "manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in manifest),
                encoding="utf-8",
            )
            scores_path = run_dir / "scores.jsonl"

            def write_scores(rows):
                scores_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )

            patches = (
                mock.patch.object(audit, "_load_json", return_value={}),
                mock.patch.object(generate, "load_actions", return_value=([], [])),
                mock.patch.object(
                    audit,
                    "_validate_registered_artifact_bindings",
                    return_value=audit.STRUCTURAL_CONTROL_SCORER_PROVENANCE_SHA256,
                ),
            )
            write_scores(scores)
            with patches[0], patches[1], patches[2]:
                self.assertEqual(
                    audit._validate_completed_scoring(run_dir),
                    audit.STRUCTURAL_CONTROL_SCORER_PROVENANCE_SHA256,
                )

            for attack in ("missing", "nan"):
                attacked = copy.deepcopy(scores)
                if attack == "missing":
                    attacked[0].pop("topiq_nr")
                else:
                    attacked[0]["topiq_nr"] = float("nan")
                write_scores(attacked)
                with patches[0], patches[1], patches[2], self.assertRaisesRegex(
                    ValueError, "missing or non-finite"
                ):
                    audit._validate_completed_scoring(run_dir)

    def test_scoring_marker_read_rejects_symlink_fifo_and_directory_without_blocking(self):
        for node_type in ("symlink", "fifo", "directory"):
            with self.subTest(node_type=node_type), tempfile.TemporaryDirectory(
                dir=ROOT
            ) as temporary:
                run_dir, amendment_path, seal_path = self.write_scoring_launcher_inputs(
                    temporary
                )
                marker_path = run_dir / audit.STRUCTURAL_CONTROL_SCORING_ATTEMPT_NAME
                if node_type == "symlink":
                    marker_path.symlink_to(run_dir / "missing-target")
                elif node_type == "fifo":
                    os.mkfifo(marker_path)
                else:
                    marker_path.mkdir()
                with mock.patch.object(
                    audit, "_validate_scoring_inputs"
                ) as validate_inputs, self.assertRaisesRegex(
                    ValueError, "missing or unsafe|not regular"
                ):
                    audit.require_scoring_attempt_marker(
                        run_dir,
                        analysis_amendment_path=amendment_path,
                        pre_score_seal_path=seal_path,
                    )
                validate_inputs.assert_not_called()

    def test_scoring_static_artifacts_reject_zero_capability_and_forged_scorer_hash(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            run_dir, amendment_path, seal_path = self.write_scoring_launcher_inputs(
                temporary
            )
            marker_path = run_dir / audit.STRUCTURAL_CONTROL_SCORING_ATTEMPT_NAME
            attempt = json.loads(
                audit._scoring_attempt_marker_payload(
                    run_dir,
                    amendment_path,
                    seal_path,
                    analysis_commit="c" * 40,
                )
            )
            attempt["launcher"]["capability_sha256"] = "0" * 64
            marker_path.write_bytes(
                (json.dumps(attempt, indent=2, sort_keys=True) + "\n").encode("utf-8")
            )
            with mock.patch.object(
                audit, "_validate_scoring_inputs"
            ) as validate_inputs, self.assertRaisesRegex(ValueError, "attempt marker"):
                audit.require_scoring_attempt_marker(
                    run_dir,
                    analysis_amendment_path=amendment_path,
                    pre_score_seal_path=seal_path,
                )
            validate_inputs.assert_not_called()

            marker_path.write_bytes(
                audit._scoring_attempt_marker_payload(
                    run_dir,
                    amendment_path,
                    seal_path,
                    analysis_commit="c" * 40,
                )
            )
            (run_dir / "scores.jsonl").write_bytes(b"{}\n")
            receipt = json.loads(
                audit._scoring_success_receipt_payload(
                    run_dir,
                    amendment_path,
                    seal_path,
                    analysis_commit="c" * 40,
                )
            )
            receipt["scorer_provenance_sha256"] = "f" * 64
            (run_dir / audit.STRUCTURAL_CONTROL_SCORING_SUCCESS_NAME).write_bytes(
                (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
            )
            with self.mocked_scoring_validation(
                run_dir, amendment_path, seal_path
            ), self.assertRaisesRegex(ValueError, "scorer hash differs"):
                audit.require_scoring_success_receipt(
                    run_dir,
                    analysis_amendment_path=amendment_path,
                    pre_score_seal_path=seal_path,
                )

    def test_scoring_child_authorization_binds_parent_token_argv_and_cwd(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            run_dir, amendment_path, seal_path = self.write_scoring_launcher_inputs(
                temporary
            )
            capability = "one-use-test-capability"
            parent_pid = 4242
            parent_ticks = 8675309
            boot_id = "12345678-1234-1234-1234-123456789abc"
            head_commit = "c" * 40
            marker_path = run_dir / audit.STRUCTURAL_CONTROL_SCORING_ATTEMPT_NAME
            marker_path.write_bytes(
                audit._scoring_attempt_marker_payload(
                    run_dir,
                    amendment_path,
                    seal_path,
                    analysis_commit=head_commit,
                    launcher_pid=parent_pid,
                    launcher_start_ticks=parent_ticks,
                    boot_id=boot_id,
                    capability_sha256=hashlib.sha256(
                        capability.encode("ascii")
                    ).hexdigest(),
                )
            )
            child_argv = audit._canonical_scoring_argv(run_dir)
            child_environment = {
                **audit.STRUCTURAL_CONTROL_SCORING_ENVIRONMENT,
                audit.STRUCTURAL_CONTROL_SCORING_CAPABILITY_ENV: capability,
            }
            with self.mocked_scoring_validation(
                run_dir, amendment_path, seal_path
            ), mock.patch.object(
                audit.os, "getppid", return_value=parent_pid
            ), mock.patch.object(
                audit, "_linux_process_start_ticks", return_value=parent_ticks
            ), mock.patch.object(
                audit, "_linux_boot_id", return_value=boot_id
            ), mock.patch.object(
                sys, "argv", child_argv[1:]
            ), mock.patch.dict(
                audit.os.environ, child_environment, clear=True
            ):
                self.assertEqual(
                    audit.require_scoring_child_authorization(
                        run_dir,
                        analysis_amendment_path=amendment_path,
                        pre_score_seal_path=seal_path,
                    ),
                    marker_path,
                )
                self.assertNotIn(
                    audit.STRUCTURAL_CONTROL_SCORING_CAPABILITY_ENV,
                    audit.os.environ,
                )

            for attack in ("parent", "token", "argv", "cwd"):
                with self.subTest(attack=attack), self.mocked_scoring_validation(
                    run_dir, amendment_path, seal_path
                ), mock.patch.object(
                    audit.os,
                    "getppid",
                    return_value=parent_pid + (1 if attack == "parent" else 0),
                ), mock.patch.object(
                    audit, "_linux_process_start_ticks", return_value=parent_ticks
                ), mock.patch.object(
                    audit, "_linux_boot_id", return_value=boot_id
                ), mock.patch.object(
                    sys,
                    "argv",
                    child_argv[1:] + (["--metrics", "pixel"] if attack == "argv" else []),
                ), mock.patch.dict(
                    audit.os.environ,
                    {
                        **child_environment,
                        audit.STRUCTURAL_CONTROL_SCORING_CAPABILITY_ENV: (
                            "wrong" if attack == "token" else capability
                        ),
                    },
                    clear=True,
                ), mock.patch.object(
                    audit.Path,
                    "cwd",
                    return_value=(ROOT.parent if attack == "cwd" else ROOT),
                ), self.assertRaises(ValueError):
                    audit.require_scoring_child_authorization(
                        run_dir,
                        analysis_amendment_path=amendment_path,
                        pre_score_seal_path=seal_path,
                    )

    def test_audit_attempt_marker_is_durable_and_one_shot(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            run_dir = pathlib.Path(temporary) / "run"
            run_dir.mkdir()
            audit.STRUCTURAL_CONTROL_FORMAL_RUN_PATH = str(run_dir)
            output_path = run_dir / audit.STRUCTURAL_CONTROL_AUDIT_OUTPUT_NAME
            with mock.patch.object(
                audit.os, "open", wraps=audit.os.open
            ) as open_file, mock.patch.object(
                audit.os, "fsync", wraps=audit.os.fsync
            ) as fsync, mock.patch.object(
                audit, "_fsync_directory", wraps=audit._fsync_directory
            ) as fsync_directory:
                marker_path = audit.create_audit_attempt_marker(
                    run_dir, output_path
                )
            marker_open = open_file.call_args_list[0]
            self.assertTrue(marker_open.args[1] & audit.os.O_EXCL)
            self.assertTrue(marker_open.args[1] & audit.os.O_NOFOLLOW)
            self.assertGreaterEqual(fsync.call_count, 2)
            fsync_directory.assert_called_once_with(run_dir.resolve())
            self.assertEqual(
                marker_path.read_bytes(), audit._audit_attempt_marker_payload()
            )
            self.assertEqual(
                audit.require_audit_attempt_marker(run_dir), marker_path
            )
            with self.assertRaisesRegex(ValueError, "one-shot"):
                audit.create_audit_attempt_marker(run_dir, output_path)

        for existing_name in (audit.STRUCTURAL_CONTROL_AUDIT_OUTPUT_NAME,):
            with self.subTest(existing_output=existing_name), tempfile.TemporaryDirectory(
                dir=ROOT
            ) as temporary:
                run_dir = pathlib.Path(temporary) / "run"
                run_dir.mkdir()
                (run_dir / existing_name).write_bytes(b"existing output")
                audit.STRUCTURAL_CONTROL_FORMAL_RUN_PATH = str(run_dir)
                selected_output = run_dir / audit.STRUCTURAL_CONTROL_AUDIT_OUTPUT_NAME
                with self.assertRaisesRegex(ValueError, "one-shot"):
                    audit.create_audit_attempt_marker(run_dir, selected_output)

    def test_audit_attempt_marker_rejects_symlink_and_creation_race(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            run_dir = pathlib.Path(temporary) / "run"
            run_dir.mkdir()
            audit.STRUCTURAL_CONTROL_FORMAL_RUN_PATH = str(run_dir)
            marker_path = run_dir / audit.STRUCTURAL_CONTROL_AUDIT_ATTEMPT_NAME
            marker_path.symlink_to(run_dir / "missing-marker-target")
            with self.assertRaisesRegex(ValueError, "one-shot"):
                audit.create_audit_attempt_marker(
                    run_dir, run_dir / audit.STRUCTURAL_CONTROL_AUDIT_OUTPUT_NAME
                )
            self.assertTrue(marker_path.is_symlink())

        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            run_dir = pathlib.Path(temporary) / "run"
            run_dir.mkdir()
            audit.STRUCTURAL_CONTROL_FORMAL_RUN_PATH = str(run_dir)
            with mock.patch.object(
                audit.os.path, "lexists", return_value=False
            ), mock.patch.object(
                audit.os, "open", side_effect=FileExistsError("racing marker")
            ), self.assertRaisesRegex(ValueError, "one-shot"):
                audit.create_audit_attempt_marker(
                    run_dir, run_dir / audit.STRUCTURAL_CONTROL_AUDIT_OUTPUT_NAME
                )

    def test_audit_attempt_marker_fsync_failures_remain_terminal(self):
        for failure_point in ("file", "directory"):
            with self.subTest(failure_point=failure_point), tempfile.TemporaryDirectory(
                dir=ROOT
            ) as temporary:
                run_dir = pathlib.Path(temporary) / "run"
                run_dir.mkdir()
                audit.STRUCTURAL_CONTROL_FORMAL_RUN_PATH = str(run_dir)
                output_path = run_dir / audit.STRUCTURAL_CONTROL_AUDIT_OUTPUT_NAME
                if failure_point == "file":
                    failure_patch = mock.patch.object(
                        audit.os, "fsync", side_effect=OSError("file fsync failed")
                    )
                else:
                    failure_patch = mock.patch.object(
                        audit,
                        "_fsync_directory",
                        side_effect=OSError("directory fsync failed"),
                    )
                with failure_patch, self.assertRaisesRegex(OSError, "fsync failed"):
                    audit.create_audit_attempt_marker(run_dir, output_path)
                marker_path = (
                    run_dir / audit.STRUCTURAL_CONTROL_AUDIT_ATTEMPT_NAME
                )
                self.assertTrue(os.path.lexists(marker_path))
                self.assertFalse(output_path.exists())
                with self.assertRaisesRegex(ValueError, "one-shot"):
                    audit.create_audit_attempt_marker(run_dir, output_path)

    def test_audit_cli_rejects_noncanonical_output_and_run_alias_before_attempt(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = pathlib.Path(temporary)
            run_dir, amendment_path, seal_path = self.write_scoring_launcher_inputs(
                temporary
            )
            marker_path = run_dir / audit.STRUCTURAL_CONTROL_AUDIT_ATTEMPT_NAME
            canonical_inputs = {
                "prompts": ROOT / generate.STRUCTURAL_CONTROL_PROMPTS,
                "actions": ROOT / audit.STRUCTURAL_CONTROL_ACTIONS_PATH,
                "registration": (
                    ROOT / generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE
                ),
                "analysis-amendment": amendment_path,
                "pre-score-seal": seal_path,
            }
            base_argv = [
                "audit_structural_control_run.py",
                "--run_dir",
                str(run_dir),
                "--prompts",
                str(canonical_inputs["prompts"]),
                "--actions",
                str(canonical_inputs["actions"]),
                "--registration",
                str(canonical_inputs["registration"]),
                "--analysis-amendment",
                str(canonical_inputs["analysis-amendment"]),
                "--pre-score-seal",
                str(canonical_inputs["pre-score-seal"]),
            ]

            def replace_argument(argv, option, value):
                attacked = list(argv)
                attacked[attacked.index(option) + 1] = str(value)
                return attacked

            run_copy = root / "run-copy"
            run_copy.mkdir()
            (root / "run-alias").symlink_to(run_dir, target_is_directory=True)
            output_alias = root / "output-alias.json"
            output_alias.symlink_to(run_dir / audit.STRUCTURAL_CONTROL_AUDIT_OUTPUT_NAME)
            attacks = [
                ("run_wrong", replace_argument(base_argv, "--run_dir", run_copy)),
                (
                    "run_alias",
                    replace_argument(base_argv, "--run_dir", root / "run-alias"),
                ),
                (
                    "output_wrong",
                    [*base_argv, "--output", str(root / "custom.json")],
                ),
                (
                    "output_alias",
                    [*base_argv, "--output", str(output_alias)],
                ),
            ]
            for option, canonical_path in canonical_inputs.items():
                wrong_path = root / f"wrong-{option}"
                wrong_path.write_bytes(b"wrong path fixture")
                alias_path = root / f"alias-{option}"
                alias_path.symlink_to(canonical_path)
                attacks.extend(
                    (
                        (
                            f"{option}_wrong",
                            replace_argument(base_argv, f"--{option}", wrong_path),
                        ),
                        (
                            f"{option}_alias",
                            replace_argument(base_argv, f"--{option}", alias_path),
                        ),
                    )
                )
            for attack, argv in attacks:
                with self.subTest(attack=attack), mock.patch.object(
                    sys, "argv", argv
                ), mock.patch.object(audit, "audit_lock") as audit_lock, mock.patch.object(
                    audit, "audit_run"
                ) as audit_run, mock.patch.object(
                    pathlib.Path,
                    "read_bytes",
                    side_effect=AssertionError("path preflight must not read bytes"),
                ), self.assertRaisesRegex(
                    ValueError, "canonical|missing|unsafe"
                ):
                    audit.main()
                self.assertFalse(os.path.lexists(marker_path))
                audit_lock.assert_not_called()
                audit_run.assert_not_called()

    def test_require_audit_attempt_marker_rejects_missing_tampered_and_nonregular(self):
        for marker_state in ("missing", "tampered", "symlink", "directory"):
            with self.subTest(marker_state=marker_state), tempfile.TemporaryDirectory(
                dir=ROOT
            ) as temporary:
                run_dir = pathlib.Path(temporary) / "run"
                run_dir.mkdir()
                marker_path = (
                    run_dir / audit.STRUCTURAL_CONTROL_AUDIT_ATTEMPT_NAME
                )
                expected_message = "missing"
                if marker_state == "tampered":
                    marker_path.write_bytes(b"tampered marker")
                    expected_message = "invalid"
                elif marker_state == "symlink":
                    marker_path.symlink_to(run_dir / "missing-target")
                elif marker_state == "directory":
                    marker_path.mkdir()
                    expected_message = "not regular"
                with self.assertRaisesRegex(ValueError, expected_message):
                    audit.require_audit_attempt_marker(run_dir)
                if marker_state == "missing":
                    self.assertFalse(os.path.lexists(marker_path))

    def test_audit_run_requires_marker_before_reading_outcomes(self):
        for marker_state in ("missing", "tampered"):
            with self.subTest(marker_state=marker_state), tempfile.TemporaryDirectory(
                dir=ROOT
            ) as temporary:
                run_dir = pathlib.Path(temporary) / "run"
                run_dir.mkdir()
                audit.STRUCTURAL_CONTROL_FORMAL_RUN_PATH = str(run_dir)
                if marker_state == "tampered":
                    (
                        run_dir / audit.STRUCTURAL_CONTROL_AUDIT_ATTEMPT_NAME
                    ).write_bytes(b"tampered marker")
                with mock.patch.object(
                    audit,
                    "_load_jsonl",
                    side_effect=AssertionError("outcomes must not be read"),
                ), self.assertRaisesRegex(ValueError, "attempt marker"):
                    audit.audit_run(
                        run_dir,
                        "missing-prompts.csv",
                        "missing-actions.yaml",
                        analysis_amendment_path="missing-amendment.yaml",
                        pre_score_seal_path="missing-seal.json",
                    )

    def test_audit_rejects_scoring_bypass_before_parsing_outcomes(self):
        for state in (
            "missing_attempt",
            "tampered_attempt",
            "missing_success",
            "tampered_success",
        ):
            with self.subTest(state=state), tempfile.TemporaryDirectory(
                dir=ROOT
            ) as temporary:
                run_dir = pathlib.Path(temporary) / "run"
                run_dir.mkdir()
                audit.STRUCTURAL_CONTROL_FORMAL_RUN_PATH = str(run_dir)
                amendment_path = pathlib.Path(temporary) / "amendment.yaml"
                amendment_path.write_bytes(b"amendment\n")
                seal_path = run_dir / audit.STRUCTURAL_CONTROL_PRE_SCORE_SEAL_NAME
                seal_path.write_bytes(b"seal\n")
                audit.create_audit_attempt_marker(
                    run_dir, run_dir / audit.STRUCTURAL_CONTROL_AUDIT_OUTPUT_NAME
                )
                scoring_marker_path = (
                    run_dir / audit.STRUCTURAL_CONTROL_SCORING_ATTEMPT_NAME
                )
                if state != "missing_attempt":
                    scoring_marker_path.write_bytes(
                        b"tampered marker"
                        if state == "tampered_attempt"
                        else audit._scoring_attempt_marker_payload(
                            run_dir,
                            amendment_path,
                            seal_path,
                            analysis_commit="c" * 40,
                        )
                    )
                success_path = run_dir / audit.STRUCTURAL_CONTROL_SCORING_SUCCESS_NAME
                if state == "tampered_success":
                    success_path.write_bytes(b"tampered receipt")
                with self.mocked_scoring_validation(
                    run_dir, amendment_path, seal_path
                ), mock.patch.object(
                    audit,
                    "_load_jsonl",
                    side_effect=AssertionError("outcomes must not be parsed"),
                ), self.assertRaises(ValueError):
                    audit.audit_run(
                        run_dir,
                        "missing-prompts.csv",
                        "missing-actions.yaml",
                        analysis_amendment_path=amendment_path,
                        pre_score_seal_path=seal_path,
                    )

    def test_formal_audit_consumes_attempt_before_read_and_preserves_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            run_dir, amendment_path, seal_path = self.write_scoring_launcher_inputs(
                temporary
            )
            marker_path = run_dir / audit.STRUCTURAL_CONTROL_AUDIT_ATTEMPT_NAME
            output_path = run_dir / audit.STRUCTURAL_CONTROL_AUDIT_OUTPUT_NAME
            argv = [
                "audit_structural_control_run.py",
                "--run_dir",
                str(run_dir),
                "--prompts",
                str(ROOT / generate.STRUCTURAL_CONTROL_PROMPTS),
                "--actions",
                str(ROOT / audit.STRUCTURAL_CONTROL_ACTIONS_PATH),
                "--registration",
                str(ROOT / generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE),
                "--analysis-amendment",
                str(amendment_path),
                "--pre-score-seal",
                str(seal_path),
            ]

            def fail_after_marker(*_args, **_kwargs):
                self.assertEqual(
                    marker_path.read_bytes(), audit._audit_attempt_marker_payload()
                )
                raise RuntimeError("audit failed after reading began")

            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                audit, "audit_run", side_effect=fail_after_marker
            ), self.assertRaisesRegex(RuntimeError, "audit failed"):
                audit.main()
            self.assertTrue(marker_path.is_file())
            self.assertFalse(output_path.exists())

            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                audit, "audit_run"
            ) as audit_run, self.assertRaisesRegex(ValueError, "one-shot"):
                audit.main()
            audit_run.assert_not_called()

    def test_v2_audit_removes_all_outcome_detail_fields(self):
        action_ids = ["left", "right"]
        normalized_actions = [
            {"id": action_id, "type": "none", "cfg_scale": 7.5}
            for action_id in action_ids
        ]
        manifest = []
        for prompt_index in range(99):
            for action_id in action_ids:
                manifest.append(
                    {
                        "id": f"p{prompt_index}_seed7_a{action_id}",
                        "prompt_index": prompt_index,
                        "seed": 7,
                        "action_id": action_id,
                        "image_sha256": (
                            "a" * 64
                            if prompt_index == 0
                            else hashlib.sha256(
                                f"{prompt_index}:{action_id}".encode("utf-8")
                            ).hexdigest()
                        ),
                    }
                )
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            run_dir = root / "run"
            run_dir.mkdir()
            audit.STRUCTURAL_CONTROL_FORMAL_RUN_PATH = str(run_dir)
            (run_dir / "config.json").write_text(
                json.dumps({"git_commit": audit.STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT}),
                encoding="utf-8",
            )
            (run_dir / "manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in manifest),
                encoding="utf-8",
            )
            (run_dir / "scores.jsonl").write_text("{}\n", encoding="utf-8")
            prompts_path = root / "prompts.csv"
            prompts_path.write_text("index,TEXT\n0,fixture\n", encoding="utf-8")
            source_path = root / "actions.yaml"
            source_path.write_text(
                yaml.safe_dump(
                    {
                        "sampling": {"model": "fixture-model"},
                        "scoring": {"required_schema": "fixture"},
                    }
                ),
                encoding="utf-8",
            )
            registration_path = root / "registration.yaml"
            registration_path.write_text(
                yaml.safe_dump({"schema": "structural_control_registration_v1"}),
                encoding="utf-8",
            )
            amendment_path = root / "amendment.yaml"
            amendment_path.write_text("schema: fixture\n", encoding="utf-8")
            audit.STRUCTURAL_CONTROL_ANALYSIS_AMENDMENT_PATH = str(amendment_path)
            seal_path = run_dir / audit.STRUCTURAL_CONTROL_PRE_SCORE_SEAL_NAME
            seal_path.write_text("{}\n", encoding="utf-8")
            scoring_marker_path = (
                run_dir / audit.STRUCTURAL_CONTROL_SCORING_ATTEMPT_NAME
            )
            scoring_marker_path.write_bytes(
                audit._scoring_attempt_marker_payload(
                    run_dir,
                    amendment_path,
                    seal_path,
                )
            )
            scoring_success_path = (
                run_dir / audit.STRUCTURAL_CONTROL_SCORING_SUCCESS_NAME
            )
            scoring_success_path.write_bytes(
                audit._scoring_success_receipt_payload(
                    run_dir,
                    amendment_path,
                    seal_path,
                )
            )
            marker_path = audit.create_audit_attempt_marker(
                run_dir, run_dir / audit.STRUCTURAL_CONTROL_AUDIT_OUTPUT_NAME
            )
            base_report = {
                "passed": True,
                "split_role": "development",
                "all_action_png_hashes_distinct_within_block": False,
                "allowed_identity_pairs": [["left", "right"]],
                "observed_identity_pairs": [["left", "right"]],
                "registered_identity_pair": ["left", "right"],
                "identity_pair_png_hashes_equal": True,
            }
            amendment = {
                "replacements": {
                    relative_path: {"amended_sha256": "a" * 64}
                    for relative_path in audit.STRUCTURAL_CONTROL_ANALYSIS_REPLACEMENTS
                }
            }
            patches = (
                mock.patch.object(generate, "validate_structural_control_authorization"),
                mock.patch.object(
                    generate,
                    "load_actions",
                    return_value=(normalized_actions, [0.08, 0.25]),
                ),
                mock.patch.object(generate, "validate_structural_control_design"),
                mock.patch.object(audit, "_validate_config_contract"),
                mock.patch.object(
                    audit,
                    "_validate_generation_commit",
                    return_value=audit.STRUCTURAL_CONTROL_AMENDMENT_BASE_COMMIT,
                ),
                mock.patch.object(
                    audit, "validate_analysis_amendment", return_value=amendment
                ),
                mock.patch.object(audit, "validate_pre_score_seal"),
                mock.patch.object(
                    audit, "_validate_analysis_implementation", return_value={}
                ),
                mock.patch.object(audit, "_validate_environment_contract"),
                mock.patch.object(audit.base_audit, "audit_run", return_value=base_report),
                mock.patch.object(audit, "validate_manifest_sidecars"),
                mock.patch.object(audit, "validate_execution_ranks"),
                mock.patch.object(
                    audit,
                    "_validate_registered_artifact_bindings",
                    return_value="a" * 64,
                ),
                mock.patch.object(
                    audit,
                    "audit_structural_records",
                    return_value={"structural_control_contract_passed": True},
                ),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[
                5
            ], patches[6], patches[7], patches[8], patches[9], patches[10], patches[
                11
            ], patches[12], patches[13]:
                report = audit.audit_run(
                    run_dir,
                    prompts_path,
                    source_path,
                    registration_actions_path=registration_path,
                    analysis_amendment_path=amendment_path,
                    pre_score_seal_path=seal_path,
                )
            self.assertEqual(
                report["audit_schema"],
                "scheduler_native_structural_control_audit_v2",
            )
            self.assertEqual(
                report["auditor_scope"], audit.STRUCTURAL_CONTROL_AUDITOR_SCOPE
            )
            self.assertFalse(report["outcome_details_disclosed"])
            self.assertTrue(report["full_action_collapse_check_passed"])
            marker_hash = hashlib.sha256(marker_path.read_bytes()).hexdigest()
            scoring_marker_hash = hashlib.sha256(
                scoring_marker_path.read_bytes()
            ).hexdigest()
            scoring_success_hash = hashlib.sha256(
                scoring_success_path.read_bytes()
            ).hexdigest()
            self.assertEqual(
                report["scoring_attempt_sha256"], scoring_marker_hash
            )
            self.assertEqual(
                report["scoring_success_sha256"], scoring_success_hash
            )
            self.assertEqual(report["audit_attempt_sha256"], marker_hash)
            self.assertEqual(
                report["provenance"]["audit_attempt_sha256"], marker_hash
            )
            self.assertEqual(
                report["provenance"]["scoring_attempt_sha256"],
                scoring_marker_hash,
            )
            self.assertEqual(
                report["provenance"]["scoring_success_sha256"],
                scoring_success_hash,
            )
            for leaked_field in (
                "duplicate_action_png_pair_counts",
                "fully_collapsed_action_pairs",
                "allowed_identity_pairs",
                "observed_identity_pairs",
                "registered_identity_pair",
                "identity_pair_png_hashes_equal",
                "all_action_png_hashes_distinct_within_block",
            ):
                self.assertNotIn(leaked_field, report)

    def test_v2_auditor_rejects_engineering_smoke_before_reading_inputs(self):
        with mock.patch.object(
            audit, "_load_json", side_effect=AssertionError("must not read inputs")
        ):
            with self.assertRaisesRegex(ValueError, "v2 is formal-only"):
                audit.audit_engineering_smoke(
                    "missing-run",
                    "missing-prompts.csv",
                    "missing-actions.yaml",
                )

        argv = [
            "audit_structural_control_run.py",
            "--run_dir",
            "missing-run",
            "--prompts",
            "missing-prompts.csv",
            "--actions",
            "missing-actions.yaml",
            "--engineering_smoke",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            audit, "audit_lock", side_effect=AssertionError("must not lock run")
        ), self.assertRaises(SystemExit) as exit_context:
            audit.main()
        self.assertEqual(exit_context.exception.code, 2)

    def test_resume_and_audit_reject_missing_worker_or_nfe_provenance(self):
        for field in ("worker_determinism_provenance", "model_load_provenance"):
            with self.subTest(field=field):
                tampered = copy.deepcopy(self.records)
                tampered[0].pop(field)
                with self.assertRaises(ValueError):
                    audit.audit_structural_records(
                        self.config, self.source, tampered, self.actions
                    )
                with self.assertRaises(ValueError):
                    generate.validate_structural_control_sidecar(
                        tampered[0], self.config, expected_task={"action": self.actions[0]}
                    )

        tampered = copy.deepcopy(self.records)
        tampered[0]["unet_calls_per_step"][-1] = 2
        with self.assertRaisesRegex(ValueError, "50x1"):
            audit.audit_structural_records(
                self.config, self.source, tampered, self.actions
            )
        with self.assertRaisesRegex(ValueError, "50x1"):
            generate.validate_structural_control_sidecar(
                tampered[0], self.config, expected_task={"action": self.actions[0]}
            )

        smoke_config = copy.deepcopy(self.config)
        smoke_config.update(
            {
                "split_role": "engineering_smoke",
                **generate.STRUCTURAL_CONTROL_SMOKE_EVIDENCE_SCOPE,
            }
        )
        smoke_record = copy.deepcopy(self.records[0])
        smoke_record.update(generate.STRUCTURAL_CONTROL_SMOKE_EVIDENCE_SCOPE)
        generate.validate_structural_control_sidecar(
            smoke_record, smoke_config, expected_task={"action": self.actions[0]}
        )
        smoke_record.pop("quality_claim_allowed")
        with self.assertRaisesRegex(ValueError, "evidence scope"):
            generate.validate_structural_control_sidecar(
                smoke_record,
                smoke_config,
                expected_task={"action": self.actions[0]},
            )

    def test_rejects_freeu_and_attention_topology_drift(self):
        tampered = copy.deepcopy(self.records)
        tampered[3]["freeu_implementation"] = "paper_adaptive_3676d36"
        with self.assertRaisesRegex(ValueError, "FreeU"):
            audit.audit_structural_records(
                self.config, self.source, tampered, self.actions
            )

        tampered = copy.deepcopy(self.records)
        tampered[3]["freeu_operator_runtime"]["operator_calls_total"] = 449
        with self.assertRaisesRegex(ValueError, "operator calls"):
            audit.audit_structural_records(
                self.config, self.source, tampered, self.actions
            )

        tampered = copy.deepcopy(self.records)
        tampered[6]["attention_baseline_topology"]["processor_count"] = 59
        with self.assertRaisesRegex(ValueError, "attention_baseline_topology"):
            audit.audit_structural_records(
                self.config, self.source, tampered, self.actions
            )

    def test_rejects_scheduler_payload_and_tfsa_activation_drift(self):
        tampered = copy.deepcopy(self.records)
        tampered[0]["scheduler_config"]["beta_start"] = 0.1
        with self.assertRaisesRegex(ValueError, "scheduler_config_sha256_v2"):
            audit.audit_structural_records(
                self.config, self.source, tampered, self.actions
            )

        tampered = copy.deepcopy(self.records)
        tampered[2]["attention_guidance_runtime"][3]["active"] = False
        with self.assertRaisesRegex(ValueError, "TFSA activation"):
            audit.audit_structural_records(
                self.config, self.source, tampered, self.actions
            )

    def test_environment_contract_requires_full_exact_runtime(self):
        relative = "eval-pipeline/configs/generation_environment_diff_attn_20260825.yaml"
        lock_path = ROOT / relative
        lock_bytes = lock_path.read_bytes()
        lock = yaml.safe_load(lock_bytes)
        source = {
            "environment_lock": {
                "path": relative,
                "sha256": hashlib.sha256(lock_bytes).hexdigest(),
            }
        }
        runtime = {
            "python_version": str(lock["runtime"]["python"]),
            "torch_version": str(lock["packages"]["torch"]),
            "diffusers_version": str(lock["packages"]["diffusers"]),
            "cuda_runtime_version": str(lock["runtime"]["cuda"]),
            "cudnn_version": lock["runtime"]["cudnn"],
            "generation_environment_lock_id": lock["lock_id"],
            "generation_environment_lock_path": relative,
            "generation_environment_lock_sha256": source["environment_lock"][
                "sha256"
            ],
            "generation_environment_packages": lock["packages"],
            "generation_environment_platform": lock["platform"],
            "generation_environment_hardware": lock["reference_hardware"],
            "generation_environment_determinism": lock["determinism"],
        }
        with mock.patch.object(audit, "_git_bytes", return_value=lock_bytes):
            audit._validate_environment_contract(
                {"runtime_provenance": runtime}, source, "a" * 40
            )
            tampered = copy.deepcopy(runtime)
            tampered["generation_environment_packages"]["diffusers"] = "0.33.0"
            with self.assertRaisesRegex(ValueError, "full environment lock"):
                audit._validate_environment_contract(
                    {"runtime_provenance": tampered}, source, "a" * 40
                )

    def test_run_config_sampling_fields_are_bound_to_registration(self):
        registration_path = (
            ROOT
            / "eval-pipeline/configs/"
            "scheduler_native_structural_controls_development_registration_v1.yaml"
        )
        source = yaml.safe_load(registration_path.read_text())
        registration_hash = hashlib.sha256(registration_path.read_bytes()).hexdigest()
        source.update(
            {
                "schema": generate.STRUCTURAL_CONTROL_SCHEMA,
                "status": "authorized_development",
                "authorization": {
                    "source_template": generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE,
                    "source_template_sha256": registration_hash,
                },
                "registration_source": {
                    "path": generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE,
                    "sha256": registration_hash,
                },
                "implementation_source": {"reviewed_commit": "a" * 40, "files": {}},
            }
        )
        sampling = source["sampling"]
        runtime = {
            "generation_environment_lock_path": source["environment_lock"]["path"],
            "generation_environment_lock_sha256": source["environment_lock"]["sha256"],
            "generation_environment_hardware": {},
            "generation_environment_determinism": {},
            "diffusers_version": "0.32.1",
        }
        config = {
            "structural_control_registered": True,
            "native_renderer_registered": False,
            "action_schema": generate.STRUCTURAL_CONTROL_SCHEMA,
            "structural_control_registration_schema": source["registration_schema"],
            "structural_control_authorization": source["authorization"],
            "structural_control_source_template": source["authorization"][
                "source_template"
            ],
            "structural_control_source_template_sha256": registration_hash,
            "structural_control_implementation_source": source[
                "implementation_source"
            ],
            "structural_control_analysis_implementation": source[
                "analysis_implementation"
            ],
            "structural_control_executable_actions_sha256": None,
            "structural_control_failure_policy": source["failure_policy"],
            "registered_sampling": sampling,
            "scheduler_runtime": source["scheduler_runtime"],
            "scoring": source["scoring"],
            "actions": self.actions,
            "scorer_provenance_binding_required": True,
            "model_name": sampling["model"],
            "model_revision": sampling["model_revision"],
            "resolution": 1024,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "guidance_rescale": 0.0,
            "negative_prompt": sampling["negative_prompt"],
            "power_calibrate": 0,
            "low_vram": False,
            "stage2_enabled": False,
            "stage_name": "stage1_1024",
            "models_to_cpu": False,
            "multi_encoder": False,
            "multi_decoder": False,
            "num_resample_timesteps": 50,
            "init_rates": [0.8],
            "stage2_noise_source": None,
            "frequency_band_cutoffs": [0.08, 0.25],
            "trajectory_registered": False,
            "scheduler_baseline_registered": False,
            "cfg_baseline_registered": False,
            "runtime_provenance": runtime,
        }
        with tempfile.TemporaryDirectory() as temporary:
            source_path = pathlib.Path(temporary) / "actions.yaml"
            source_path.write_text(yaml.safe_dump(source, sort_keys=False))
            config["structural_control_executable_actions_sha256"] = hashlib.sha256(
                source_path.read_bytes()
            ).hexdigest()
            audit._validate_config_contract(
                config,
                source,
                self.actions,
                source_path,
                registration_path,
            )
            config["resolution"] = 512
            with self.assertRaisesRegex(ValueError, "resolution"):
                audit._validate_config_contract(
                    config,
                    source,
                    self.actions,
                    source_path,
                    registration_path,
                )

if __name__ == "__main__":
    unittest.main()
