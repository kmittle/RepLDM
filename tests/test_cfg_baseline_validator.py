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


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


generate = load_module("generate_cfg_validator_test", "eval-pipeline/generate.py")
provenance = load_module("cfg_validator_provenance_test", "eval-pipeline/s7_provenance.py")
validator = load_module(
    "cfg_baseline_validator_test", "eval-pipeline/validate_cfg_baseline_run.py"
)


class CFGBaselineValidatorTest(unittest.TestCase):
    def validate_fixture(self, run_dir, actions, prompts, kind):
        with mock.patch.object(validator, "validate_cfg_authorization"), mock.patch.object(
            validator, "validate_frozen_registration"
        ):
            return validator.validate(str(run_dir), str(actions), str(prompts), kind)

    def make_fixture(self, root, *, duplicate_images=False):
        root = pathlib.Path(root)
        run_dir = root / "run"
        image_dir = run_dir / "images"
        image_dir.mkdir(parents=True)
        prompts_path = root / "prompts.csv"
        prompts_path.write_text("index,TEXT,bucket\n0,a prompt,test\n")
        actions_path = root / "actions.yaml"
        sampling = {
            "model": "test/model",
            "model_revision": "test-revision",
            "base_scheduler": "EulerDiscreteScheduler",
            "resolution": 4,
            "num_inference_steps": 2,
            "default_cfg_scale": 7.5,
            "guidance_rescale": 0.0,
            "negative_prompt": generate.DEFAULT_NEG,
            "power_calibrate": 0,
            "stage2": False,
            "extra_unet_calls": 0,
            "initialization": "scheduler_native_init_sigma",
            "cfg_source": "action.cfg_scale",
            "cfg_pipeline_argument": "guidance_scale",
        }
        source_actions = [
            {"id": action_id, "type": "none", "cfg_scale": scale}
            for action_id, scale in zip(validator.CFG_ACTION_IDS, validator.CFG_SCALES)
        ]
        actions_config = {
            "schema": "cfg_baselines_v1",
            "status": "authorized",
            "authorization": {
                "reviewer": "fixture",
                "reviewed_commit": "fixture",
            },
            "source_manifest": {
                "prompts": str(prompts_path),
                "prompts_sha256": provenance.sha256_file(prompts_path),
            },
            "split_seeds": {"development": [0, 1]},
            "design": {
                "expected_prompt_count": 1,
                "expected_action_count": 5,
                "expected_seed_count": 2,
                "expected_block_count": 2,
                "expected_task_count": 10,
                "action_ids": list(validator.CFG_ACTION_IDS),
                "cfg_scales": list(validator.CFG_SCALES),
                "baseline_action_id": validator.BASELINE_ACTION_ID,
                "execution_order": "deterministic_sha256_v1",
                "pairing": "prompt_seed_same_device",
                "action_sha256": {},
            },
            "sampling": sampling,
            "scoring": validator.CFG_SCORING,
            "execution_order": validator.EXECUTION_ORDER,
            "actions": source_actions,
        }
        actions_path.write_text(yaml.safe_dump(actions_config, sort_keys=False))
        actions, _ = generate.load_actions(str(actions_path), 2)
        actions_config["design"]["action_sha256"] = {
            action["id"]: provenance.action_sha256(action) for action in actions
        }
        actions_path.write_text(yaml.safe_dump(actions_config, sort_keys=False))

        config = {
            "action_schema": "cfg_baselines_v1",
            "actions_sha256": provenance.sha256_file(actions_path),
            "actions": actions,
            "prompts_sha256": provenance.sha256_file(prompts_path),
            "seeds": [0, 1],
            "model_name": sampling["model"],
            "resolution": sampling["resolution"],
            "num_inference_steps": sampling["num_inference_steps"],
            "guidance_scale": sampling["default_cfg_scale"],
            "negative_prompt": sampling["negative_prompt"],
            "power_calibrate": sampling["power_calibrate"],
            "stage_name": "stage1_1024",
            "stage2_enabled": False,
            "models_to_cpu": False,
            "multi_encoder": False,
            "multi_decoder": False,
            "num_resample_timesteps": 50,
            "init_rates": [0.8],
            "frequency_band_cutoffs": [0.08, 0.25],
            "split_role": "development",
            "git_commit": "f" * 40,
            "runtime_provenance": {"python_version": "fixture"},
            "cfg_baseline_registered": True,
            "scheduler_baseline_registered": False,
            "trajectory_registered": False,
            "registered_sampling": sampling,
        }
        config["run_contract"] = generate.generation_contract(
            config,
            actions=actions,
            seeds=config["seeds"],
            prompts_sha256=config["prompts_sha256"],
            actions_sha256=config["actions_sha256"],
        )
        config["run_contract_sha256"] = provenance.json_sha256(
            config["run_contract"]
        )
        (run_dir / "config.json").write_text(json.dumps(config))

        rows = []
        scores = []
        timesteps = [1.0, 0.0]
        sigmas = [2.0, 1.0, 0.0]
        schedule_hash = provenance.json_sha256(
            {"timesteps": timesteps, "sigmas": sigmas}
        )
        scheduler_config = {
            "_class_name": "EulerDiscreteScheduler",
            "num_train_timesteps": 1000,
        }
        scheduler_config_hash = validator.scheduler_config_sha256_v2(
            scheduler_config
        )
        actions_config["scheduler_runtime"] = {
            "config_sha256_v2": scheduler_config_hash,
            "num_inference_steps": 2,
            "schedule_sha256": schedule_hash,
            "construction_init_noise_sigma": 2.0,
            "effective_init_noise_sigma": 2.0,
        }
        actions_path.write_text(yaml.safe_dump(actions_config, sort_keys=False))
        config["actions_sha256"] = provenance.sha256_file(actions_path)
        config["actions_yaml"] = str(actions_path)
        config["prompts_csv"] = str(prompts_path)
        config["model_revision"] = sampling["model_revision"]
        config["guidance_rescale"] = sampling["guidance_rescale"]
        config["scheduler_runtime"] = actions_config["scheduler_runtime"]
        config["run_contract"] = generate.generation_contract(
            config,
            actions=actions,
            seeds=config["seeds"],
            prompts_sha256=config["prompts_sha256"],
            actions_sha256=config["actions_sha256"],
        )
        config["run_contract_sha256"] = provenance.json_sha256(
            config["run_contract"]
        )
        (run_dir / "config.json").write_text(json.dumps(config))
        scoring_contract = {
            "schema": validator.SCORING_SCHEMA,
            "action_schema": "cfg_baselines_v1",
            "metrics": list(validator.CFG_SCORING["metrics"]),
            "strict": True,
            "params": dict(validator.CFG_SCORING["params"]),
            "required_score_keys": list(
                validator.CFG_SCORING["required_score_keys"]
            ),
            "actions_sha256": config["actions_sha256"],
        }
        scoring_contract_sha256 = provenance.json_sha256(scoring_contract)
        for seed in config["seeds"]:
            ranks = validator.expected_execution_ranks(
                0, seed, validator.CFG_ACTION_IDS
            )
            for action_index, action in enumerate(actions):
                action_id = action["id"]
                row_id = f"p0_seed{seed}_a{action_id}"
                image_path = image_dir / f"{row_id}.png"
                color_rank = 0 if duplicate_images else action_index
                Image.new(
                    "RGB",
                    (4, 4),
                    color=(20 + color_rank * 20, 30 + seed * 10, 40),
                ).save(image_path)
                row = {
                    "id": row_id,
                    "prompt_index": 0,
                    "prompt": "a prompt",
                    "bucket": "test",
                    "seed": seed,
                    "action_id": action_id,
                    "action_type": "none",
                    "action": action,
                    "action_sha256": provenance.action_sha256(action),
                    "image_path": f"images/{row_id}.png",
                    "image_sha256": provenance.image_sha256(image_path),
                    "width": 4,
                    "height": 4,
                    "run_contract_sha256": config["run_contract_sha256"],
                    "provenance_schema": provenance.PROVENANCE_SCHEMA,
                    "registered_sampling": sampling,
                    "scheduler_runtime": actions_config["scheduler_runtime"],
                    "model_name": sampling["model"],
                    "model_revision": sampling["model_revision"],
                    "guidance_rescale": sampling["guidance_rescale"],
                    "stage2_enabled": False,
                    "power_calibrate": 0,
                    "base_scheduler_name": sampling["base_scheduler"],
                    "scheduler_name": sampling["base_scheduler"],
                    "scheduler_reference": False,
                    "scheduler_kwargs": {},
                    "scheduler_solver_order": None,
                    "scheduler_order": 1,
                    "scheduler_init_noise_sigma": 2.0,
                    "scheduler_construction_init_noise_sigma": 2.0,
                    "scheduler_effective_init_noise_sigma": 2.0,
                    "scheduler_timesteps": timesteps,
                    "scheduler_sigmas": sigmas,
                    "scheduler_schedule_sha256": schedule_hash,
                    "scheduler_config_sha256_v2": scheduler_config_hash,
                    "active_scheduler_config_sha256_v2": scheduler_config_hash,
                    "scheduler_config": scheduler_config,
                    "active_scheduler_config": scheduler_config,
                    "num_inference_steps": 2,
                    "guidance_scale": float(action["cfg_scale"]),
                    "unet_calls_per_step": [1, 1],
                    "extra_unet_calls": 0,
                    "error": None,
                    "device": "cuda:0",
                    "execution_rank": ranks[action_id],
                }
                rows.append(row)
                score = {
                    key: row[key]
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
                    )
                }
                score["provenance_schema"] = provenance.PROVENANCE_SCHEMA
                score.update({key: 0.5 for key in validator.SCORE_KEYS})
                score["scoring_contract"] = scoring_contract
                score["scoring_contract_sha256"] = scoring_contract_sha256
                scores.append(score)
        (run_dir / "manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        (run_dir / "scores.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in scores)
        )
        return run_dir, actions_path, prompts_path

    def test_clean_fixture_passes_manifest_and_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, actions, prompts = self.make_fixture(tmp)
            report = self.validate_fixture(run_dir, actions, prompts, "scores")
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["row_count"], 10)

    def test_duplicate_images_within_cfg_block_are_audited(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, actions, prompts = self.make_fixture(
                tmp, duplicate_images=True
            )
            report = self.validate_fixture(run_dir, actions, prompts, "scores")
            self.assertEqual(report["duplicate_image_group_count"], 2)

    def test_duplicate_image_scores_must_be_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, actions, prompts = self.make_fixture(
                tmp, duplicate_images=True
            )
            path = pathlib.Path(run_dir) / "scores.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[1]["topiq_nr"] = 0.6
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "inconsistent deterministic score"):
                self.validate_fixture(run_dir, actions, prompts, "scores")

    def test_scoring_contract_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, actions, prompts = self.make_fixture(tmp)
            path = pathlib.Path(run_dir) / "scores.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["scoring_contract"]["metrics"] = ["pixel"]
            rows[0]["scoring_contract_sha256"] = provenance.json_sha256(
                rows[0]["scoring_contract"]
            )
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "scoring contract drifted"):
                self.validate_fixture(run_dir, actions, prompts, "scores")

    def test_fractional_design_key_and_noncanonical_id_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, actions, prompts = self.make_fixture(tmp)
            path = pathlib.Path(run_dir) / "manifest.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["prompt_index"] = 0.5
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "prompt_index must be an integer"):
                self.validate_fixture(run_dir, actions, prompts, "manifest")

    def test_self_consistent_sampling_replacement_is_rejected(self):
        config_path = EVAL_PIPELINE / "configs" / "cfg_baselines_v1.yaml"
        prompts_path = EVAL_PIPELINE / "prompts" / "s5_development.csv"
        config = yaml.safe_load(config_path.read_text())
        config["sampling"]["model"] = "replacement/model"
        with self.assertRaisesRegex(ValueError, "sampling differs from frozen v1"):
            validator.validate_frozen_registration(
                config, str(prompts_path), str(ROOT)
            )

    def test_effective_cfg_scale_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, actions, prompts = self.make_fixture(tmp)
            path = pathlib.Path(run_dir) / "manifest.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["guidance_scale"] = 7.5
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "effective CFG scale"):
                self.validate_fixture(run_dir, actions, prompts, "manifest")

    def test_execution_order_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, actions, prompts = self.make_fixture(tmp)
            path = pathlib.Path(run_dir) / "manifest.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["execution_rank"] = 99
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "execution ranks"):
                self.validate_fixture(run_dir, actions, prompts, "manifest")

    def test_nonfinite_score_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, actions, prompts = self.make_fixture(tmp)
            path = pathlib.Path(run_dir) / "scores.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["topiq_nr"] = float("nan")
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "must be finite"):
                self.validate_fixture(run_dir, actions, prompts, "scores")

    def test_scheduler_payload_tampering_is_rejected_even_with_rehashed_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, actions, prompts = self.make_fixture(tmp)
            path = pathlib.Path(run_dir) / "manifest.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["active_scheduler_config"]["num_train_timesteps"] = 999
            rows[0]["active_scheduler_config_sha256_v2"] = (
                validator.scheduler_config_sha256_v2(
                    rows[0]["active_scheduler_config"]
                )
            )
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "Euler config differs from frozen v1"):
                self.validate_fixture(run_dir, actions, prompts, "manifest")


if __name__ == "__main__":
    unittest.main()
