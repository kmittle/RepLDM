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

import yaml
from PIL import Image


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
    def setUp(self):
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

    def test_complete_structural_record_contract_passes(self):
        report = audit.audit_structural_records(
            self.config, self.source, self.records, self.actions
        )
        self.assertTrue(report["structural_control_contract_passed"])
        self.assertEqual(report["matched_unet_calls"], "50x1")
        self.assertFalse(report["quality_results_inspected"])

    def test_duplicate_png_policy_reports_isolated_and_rejects_full_collapse(self):
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
        report = audit.audit_action_png_pair_counts(
            records, ["left", "right"], expected_blocks=2
        )
        self.assertEqual(
            report,
            [
                {
                    "actions": ["left", "right"],
                    "matching_blocks": 1,
                    "total_blocks": 2,
                }
            ],
        )

        records[-1]["image_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "identical PNGs in all 2 blocks"):
            audit.audit_action_png_pair_counts(
                records, ["left", "right"], expected_blocks=2
            )

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

    def test_engineering_smoke_audits_scope_and_distinct_artifact_grid(self):
        seed = generate.STRUCTURAL_CONTROL_SPLIT_SEEDS["engineering_smoke"][0]
        evidence_scope = generate.STRUCTURAL_CONTROL_SMOKE_EVIDENCE_SCOPE
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            run_dir = root / "run"
            image_dir = run_dir / "images"
            image_dir.mkdir(parents=True)
            prompts_path = root / "smoke.csv"
            prompts_path.write_text(
                "index,TEXT,source_challenge,split\n"
                + "".join(
                    f'{index},"prompt {index}",challenge_{index:02d},engineering_smoke\n'
                    for index in range(11)
                ),
                encoding="utf-8",
            )
            source_path = root / "actions.yaml"
            source_path.write_text(yaml.safe_dump(self.source), encoding="utf-8")
            registration_path = root / "registration.yaml"
            registration_path.write_text(
                yaml.safe_dump({"schema": "structural_control_registration_v1"}),
                encoding="utf-8",
            )
            config = copy.deepcopy(self.config)
            config.update(
                {
                    "split_role": "engineering_smoke",
                    "seeds": [seed],
                    **evidence_scope,
                }
            )
            (run_dir / "config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )

            records = []
            for prompt_index in range(11):
                expected_ranks = audit.expected_execution_ranks(
                    prompt_index, seed, [action["id"] for action in self.actions]
                )
                for action_index, action in enumerate(self.actions):
                    record = copy.deepcopy(self.record(action))
                    record_id = (
                        f"p{prompt_index}_seed{seed}_a{action['id']}"
                    )
                    relative_image = f"images/{record_id}.png"
                    image_path = run_dir / relative_image
                    Image.new(
                        "RGB",
                        (1024, 1024),
                        color=(action_index * 29, action_index * 17, action_index * 11),
                    ).save(image_path)
                    record.update(
                        {
                            "id": record_id,
                            "prompt_index": prompt_index,
                            "prompt": f"prompt {prompt_index}",
                            "seed": seed,
                            "action_id": action["id"],
                            "action_type": action["type"],
                            "action": action,
                            "execution_rank": expected_ranks[action["id"]],
                            "image_path": relative_image,
                            "image_sha256": hashlib.sha256(
                                image_path.read_bytes()
                            ).hexdigest(),
                            "run_contract_sha256": "contract",
                            **evidence_scope,
                        }
                    )
                    records.append(record)
                    (image_dir / f"{record_id}.json").write_text(
                        json.dumps(record), encoding="utf-8"
                    )
            (run_dir / "manifest.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            patches = (
                mock.patch.object(
                    generate, "validate_structural_control_authorization"
                ),
                mock.patch.object(
                    generate,
                    "load_actions",
                    return_value=(self.actions, [0.08, 0.25]),
                ),
                mock.patch.object(generate, "validate_structural_control_design"),
                mock.patch.object(audit, "_validate_config_contract"),
                mock.patch.object(
                    audit, "_validate_generation_commit", return_value="a" * 40
                ),
                mock.patch.object(
                    audit,
                    "_validate_analysis_implementation",
                    return_value={"schema": "fixture", "files": {}},
                ),
                mock.patch.object(audit, "_validate_environment_contract"),
                mock.patch.object(audit, "validate_run_contract", return_value="contract"),
                mock.patch.object(audit, "validate_sidecar"),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[
                5
            ], patches[6], patches[7], patches[8]:
                report = audit.audit_engineering_smoke(
                    run_dir,
                    prompts_path,
                    source_path,
                    registration_actions_path=registration_path,
                )
            self.assertTrue(report["passed"])
            self.assertEqual(report["records"], 88)
            self.assertTrue(report["all_actions_distinct_within_every_block"])
            self.assertEqual(
                {key: report[key] for key in evidence_scope}, evidence_scope
            )
            first_sidecar = image_dir / f"{records[0]['id']}.json"
            first_sidecar.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from manifest row"):
                audit.validate_manifest_sidecars(run_dir, records)

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
