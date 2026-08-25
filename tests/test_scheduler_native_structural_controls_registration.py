import copy
import hashlib
import importlib.util
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_PIPELINE = ROOT / "eval-pipeline"
CONFIG = (
    ROOT
    / "eval-pipeline/configs/"
    "scheduler_native_structural_controls_development_registration_v1.yaml"
)
PROMPTS = (
    ROOT
    / "eval-pipeline/prompts/scheduler_native_fixed_headroom_development.csv"
)
SMOKE_PROMPTS = (
    ROOT
    / "eval-pipeline/prompts/scheduler_native_fixed_headroom_smoke.csv"
)
if str(EVAL_PIPELINE) not in sys.path:
    sys.path.insert(0, str(EVAL_PIPELINE))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate = load_module(
    "scheduler_native_structural_controls_generate_test",
    EVAL_PIPELINE / "generate.py",
)


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SchedulerNativeStructuralControlsRegistrationTest(unittest.TestCase):
    def setUp(self):
        self.registration = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    def executable_design(self):
        config = copy.deepcopy(self.registration)
        config["schema"] = generate.STRUCTURAL_CONTROL_SCHEMA
        config["status"] = "authorized_development"
        config.pop("blocking_conditions", None)
        return config

    def write_yaml(self, directory, config, name="actions.yaml"):
        path = pathlib.Path(directory) / name
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return path

    def validate_design(
        self,
        path,
        *,
        prompts=PROMPTS,
        seeds=(1932556753, 1065503757, 201635682),
        split_role="development",
    ):
        actions, _ = generate.load_actions(str(path), 50)
        generate.validate_structural_control_design(
            str(path),
            prompts_path=str(prompts),
            actions=actions,
            seeds=list(seeds),
            model_name="stabilityai/stable-diffusion-xl-base-1.0",
            resolution=1024,
            num_inference_steps=50,
            guidance_scale=7.5,
            negative_prompt=generate.DEFAULT_NEG,
            power_calibrate=0,
            stage2_enabled=False,
            split_role=split_role,
        )
        return actions

    def test_registration_is_non_executable_and_freezes_complete_design(self):
        self.assertEqual(
            sha256_file(CONFIG),
            "dc8f0333d89e5815f560a0de75fa17d99869ef1da251bed95dac075b901024b6",
        )
        self.assertEqual(
            self.registration["schema"], "structural_control_registration_v1"
        )
        self.assertFalse(self.registration["authorization"]["gpu_generation"])
        self.assertEqual(
            self.registration["environment_lock"]["sha256"],
            "8f7b38ccb770880537f5080b1d3b4eb426a294458ea644ec8a4ef6b61f771da4",
        )
        self.assertEqual(
            set(self.registration["sampling"]),
            generate.STRUCTURAL_CONTROL_SAMPLING_KEYS,
        )
        self.assertEqual(
            set(self.registration["analysis_implementation"]["files"]),
            set(generate.STRUCTURAL_CONTROL_ANALYSIS_PATHS),
        )
        self.assertEqual(
            self.registration["execution_order"], generate.CFG_BASELINE_EXECUTION_ORDER
        )
        self.assertEqual(
            self.registration["engineering_smoke"],
            generate.STRUCTURAL_CONTROL_ENGINEERING_SMOKE,
        )
        self.assertEqual(
            self.registration["failure_policy"],
            "shared_abort_after_first_task_error",
        )
        self.assertEqual(
            [action["id"] for action in self.registration["actions"]],
            list(generate.STRUCTURAL_CONTROL_ACTION_IDS),
        )
        for action in self.registration["actions"]:
            if action["type"] == "freeu":
                self.assertEqual(action["expected_operator_calls_per_step"], 9)
                self.assertEqual(
                    action["expected_resolution_idx_call_counts_per_step"],
                    [3, 3, 3],
                )
                self.assertEqual(
                    action["expected_hidden_channel_call_counts_per_step"],
                    {"1280": 4, "640": 3, "320": 2},
                )
                self.assertEqual(
                    action["expected_resolution_channel_call_counts_per_step"],
                    {
                        "0:1280": 3,
                        "1:1280": 1,
                        "1:640": 2,
                        "2:640": 1,
                        "2:320": 2,
                    },
                )
        with self.assertRaisesRegex(ValueError, "registration manifests"):
            generate.load_actions(str(CONFIG), 50)

    def test_executable_design_validates_and_rejects_sampling_order_or_cfg_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.executable_design()
            path = self.write_yaml(directory, source)
            actions = self.validate_design(path)
            self.assertEqual(len(actions), 8)
            smoke_actions = self.validate_design(
                path,
                prompts=SMOKE_PROMPTS,
                seeds=(1798464083,),
                split_role="engineering_smoke",
            )
            self.assertEqual(smoke_actions, actions)
            self.assertEqual(
                generate.worker_resume_contract_sha256(
                    {
                        "structural_control_registered": True,
                        "run_contract_sha256": "contract",
                    }
                ),
                "contract",
            )
            registered_cfg = {
                "structural_control_registered": True,
                "num_inference_steps": 50,
            }
            self.assertTrue(generate.strict_registered_run(registered_cfg))
            self.assertTrue(generate.scheduler_isolation_required(registered_cfg))
            generate.validate_structural_control_worker_result(
                registered_cfg, [object()], [1] * 50
            )
            with self.assertRaisesRegex(RuntimeError, "one U-Net call"):
                generate.validate_structural_control_worker_result(
                    registered_cfg, [object()], [1] * 49 + [2]
                )
            with self.assertRaisesRegex(RuntimeError, "one image"):
                generate.validate_structural_control_worker_result(
                    registered_cfg, [object(), object()], [1] * 50
                )

            base_contract_config = {
                "action_schema": generate.STRUCTURAL_CONTROL_SCHEMA,
                "model_name": source["sampling"]["model"],
                "resolution": 1024,
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
                "negative_prompt": generate.DEFAULT_NEG,
                "power_calibrate": 0,
                "stage_name": "stage1_1024",
                "stage2_enabled": False,
                "models_to_cpu": False,
                "multi_encoder": False,
                "multi_decoder": False,
                "num_resample_timesteps": 50,
                "init_rates": [0.8],
                "frequency_band_cutoffs": [0.08, 0.25],
                "git_commit": "a" * 40,
                "runtime_provenance": {},
            }
            for role, role_seeds in (
                ("engineering_smoke", [1798464083]),
                ("development", [1932556753, 1065503757, 201635682]),
            ):
                with self.subTest(contract_role=role):
                    config = {**base_contract_config, "split_role": role}
                    config.update(generate.structural_control_evidence_scope(role))
                    contract = generate.generation_contract(
                        config,
                        actions=actions,
                        seeds=role_seeds,
                        prompts_sha256="b" * 64,
                        actions_sha256="c" * 64,
                    )
                    config.update(
                        {
                            "actions": actions,
                            "seeds": role_seeds,
                            "prompts_sha256": "b" * 64,
                            "actions_sha256": "c" * 64,
                            "run_contract": contract,
                            "run_contract_sha256": generate.json_sha256(contract),
                        }
                    )
                    self.assertEqual(
                        generate.validate_run_contract(config),
                        config["run_contract_sha256"],
                    )

            cases = (
                ("negative_prompt", "different"),
                ("power_calibrate", 1),
                ("torch_dtype", "bfloat16"),
                ("attention_mask_policy", "supported"),
                ("low_vram", True),
            )
            for key, value in cases:
                with self.subTest(key=key):
                    tampered = copy.deepcopy(source)
                    tampered["sampling"][key] = value
                    tampered_path = self.write_yaml(directory, tampered, f"{key}.yaml")
                    with self.assertRaisesRegex(ValueError, f"sampling.{key}"):
                        self.validate_design(tampered_path)

            tampered = copy.deepcopy(source)
            tampered["execution_order"]["input"] = "action_id"
            tampered_path = self.write_yaml(directory, tampered, "order.yaml")
            with self.assertRaisesRegex(ValueError, "execution_order"):
                self.validate_design(tampered_path)

            tampered = copy.deepcopy(source)
            tampered["actions"][6]["cfg_scale"] = 7.5
            tampered_path = self.write_yaml(directory, tampered, "cfg.yaml")
            with self.assertRaisesRegex(ValueError, "cfg_scale"):
                self.validate_design(tampered_path)

            tampered = copy.deepcopy(source)
            tampered["actions"][3]["expected_operator_calls_per_step"] = 8
            tampered_path = self.write_yaml(directory, tampered, "freeu_calls.yaml")
            with self.assertRaisesRegex(ValueError, "FreeU call topology"):
                self.validate_design(tampered_path)

    def test_authorization_binds_reviewed_template_source_blobs_and_clean_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = pathlib.Path(directory) / "repo"
            repo.mkdir()
            reviewed_paths = set(generate.STRUCTURAL_CONTROL_IMPLEMENTATION_PATHS) | set(
                generate.STRUCTURAL_CONTROL_ANALYSIS_PATHS
            )
            for relative_path in reviewed_paths:
                destination = repo / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative_path, destination)
            template = repo / generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE
            template.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(CONFIG, template)
            template_config = yaml.safe_load(template.read_text(encoding="utf-8"))
            template_config["analysis_implementation"]["files"] = {
                relative_path: sha256_file(repo / relative_path)
                for relative_path in generate.STRUCTURAL_CONTROL_ANALYSIS_PATHS
            }
            template.write_text(
                yaml.safe_dump(template_config, sort_keys=False), encoding="utf-8"
            )
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repo), "config", "user.name", "Test"],
                check=True,
            )
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "commit", "-qm", "reviewed implementation"],
                check=True,
            )
            reviewed_commit = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip()
            template_hash = sha256_file(template)
            source_hashes = {
                relative_path: sha256_file(repo / relative_path)
                for relative_path in generate.STRUCTURAL_CONTROL_IMPLEMENTATION_PATHS
            }

            executable = copy.deepcopy(template_config)
            executable["schema"] = generate.STRUCTURAL_CONTROL_SCHEMA
            executable["status"] = "authorized_development"
            executable.pop("blocking_conditions", None)
            executable["authorization"] = {
                "reviewer": "independent_contract_audit",
                "reviewed_commit": reviewed_commit,
                "source_template": generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE,
                "source_template_sha256": template_hash,
                "scope": generate.STRUCTURAL_CONTROL_AUTH_SCOPE,
                "gpu_generation": True,
                "scoring": True,
                "method_selection": False,
                "result_access_before_freeze": False,
            }
            executable["registration_source"] = {
                "path": generate.STRUCTURAL_CONTROL_AUTH_SOURCE_TEMPLATE,
                "sha256": template_hash,
            }
            executable["implementation_source"] = {
                "reviewed_commit": reviewed_commit,
                "files": source_hashes,
            }
            executable_path = repo / "structural_control_actions.yaml"
            executable_path.write_text(
                yaml.safe_dump(executable, sort_keys=False), encoding="utf-8"
            )
            subprocess.run(
                ["git", "-C", str(repo), "add", "structural_control_actions.yaml"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repo), "commit", "-qm", "authorize executable"],
                check=True,
            )
            generate.validate_structural_control_authorization(
                str(executable_path), repository_root=str(repo)
            )

            outside_path = pathlib.Path(directory) / "outside_actions.yaml"
            outside_path.write_bytes(executable_path.read_bytes())
            with self.assertRaisesRegex(ValueError, "inside the reviewed repository"):
                generate.validate_structural_control_authorization(
                    str(outside_path),
                    repository_root=str(repo),
                    require_clean=False,
                )

            untracked_path = repo / "untracked_actions.yaml"
            untracked_path.write_bytes(executable_path.read_bytes())
            with self.assertRaisesRegex(ValueError, "tracked at repository HEAD"):
                generate.validate_structural_control_authorization(
                    str(untracked_path),
                    repository_root=str(repo),
                    require_clean=False,
                )
            untracked_path.unlink()

            committed_executable = executable_path.read_bytes()
            executable_path.write_bytes(committed_executable + b"\n")
            with self.assertRaisesRegex(ValueError, "bytes differ from repository HEAD"):
                generate.validate_structural_control_authorization(
                    str(executable_path),
                    repository_root=str(repo),
                    require_clean=False,
                )
            executable_path.write_bytes(committed_executable)

            dirty_path = repo / "unreviewed.txt"
            dirty_path.write_text("dirty", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "clean repository worktree"):
                generate.validate_structural_control_authorization(
                    str(executable_path), repository_root=str(repo)
                )
            dirty_path.unlink()

            changed_source = repo / generate.STRUCTURAL_CONTROL_IMPLEMENTATION_PATHS[0]
            original_source = changed_source.read_bytes()
            changed_source.write_text(
                changed_source.read_text(encoding="utf-8") + "\n# drift\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "current implementation source hash"):
                generate.validate_structural_control_authorization(
                    str(executable_path),
                    repository_root=str(repo),
                    require_clean=False,
                )
            changed_source.write_bytes(original_source)

            changed_analysis = repo / "eval-pipeline/evaluate_structural_control_run.py"
            changed_analysis.write_text(
                changed_analysis.read_text(encoding="utf-8") + "\n# drift\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "current analysis source hash"):
                generate.validate_structural_control_authorization(
                    str(executable_path),
                    repository_root=str(repo),
                    require_clean=False,
                )


if __name__ == "__main__":
    unittest.main()
