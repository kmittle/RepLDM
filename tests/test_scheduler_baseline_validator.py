import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest

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


generate = load_module("generate_scheduler_validator_test", "eval-pipeline/generate.py")
provenance = load_module("s7_scheduler_validator_test", "eval-pipeline/s7_provenance.py")
validator = load_module(
    "scheduler_baseline_validator_test",
    "eval-pipeline/validate_scheduler_baseline_run.py",
)


class SchedulerBaselineValidatorTest(unittest.TestCase):
    def make_fixture(self, root):
        root = pathlib.Path(root)
        run_dir = root / "run"
        image_dir = run_dir / "images"
        image_dir.mkdir(parents=True)
        prompts_path = root / "prompts.csv"
        prompts_path.write_text("index,TEXT,bucket\n0,a prompt,test\n")
        actions_path = root / "actions.yaml"
        sampling = {
            "model": "test/model",
            "base_scheduler": "EulerDiscreteScheduler",
            "resolution": 4,
            "num_inference_steps": 2,
            "cfg_scale": 7.5,
            "stage2": False,
            "extra_unet_calls": 0,
            "initialization": "scheduler_native_init_sigma",
        }
        source_actions = [
            {
                "id": "no_correction",
                "type": "none",
                "selection_eligible": False,
            },
            {
                "id": "euler_ancestral_reference",
                "type": "scheduler_baseline",
                "scheduler_class": "EulerAncestralDiscreteScheduler",
                "scheduler_kwargs": {},
                "selection_eligible": False,
            },
        ]
        actions_config = {
            "schema": "scheduler_baselines_v1",
            "status": "authorized",
            "authorization": {
                "reviewer": "fixture",
                "reviewed_commit": "fixture",
            },
            "split_seeds": {"development": [0, 1]},
            "design": {
                "expected_prompt_count": 1,
                "expected_action_count": 2,
                "expected_seed_count": 2,
                "expected_task_count": 4,
                "action_ids": [action["id"] for action in source_actions],
                "action_sha256": {},
            },
            "sampling": sampling,
            "actions": source_actions,
        }
        actions_path.write_text(yaml.safe_dump(actions_config, sort_keys=False))
        actions, _ = generate.load_actions(str(actions_path), 2)
        actions_config["design"]["action_sha256"] = {
            action["id"]: provenance.action_sha256(action) for action in actions
        }
        actions_path.write_text(yaml.safe_dump(actions_config, sort_keys=False))

        config = {
            "action_schema": "scheduler_baselines_v1",
            "actions_sha256": provenance.sha256_file(actions_path),
            "actions": actions,
            "prompts_sha256": provenance.sha256_file(prompts_path),
            "seeds": [0, 1],
            "model_name": sampling["model"],
            "resolution": sampling["resolution"],
            "num_inference_steps": sampling["num_inference_steps"],
            "guidance_scale": sampling["cfg_scale"],
            "negative_prompt": "bad",
            "power_calibrate": 0,
            "stage_name": "stage1_1024",
            "stage2_enabled": False,
            "models_to_cpu": False,
            "multi_encoder": False,
            "multi_decoder": False,
            "num_resample_timesteps": 50,
            "init_rates": [0.8],
            "frequency_band_cutoffs": [0.08, 0.25],
            "split_role": "development",
            "git_commit": "fixture",
            "runtime_provenance": {"python_version": "fixture"},
            "scheduler_baseline_registered": True,
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
        for seed in config["seeds"]:
            for rank, action in enumerate(actions):
                action_id = action["id"]
                row_id = f"p0_seed{seed}_a{action_id}"
                image_path = image_dir / f"{row_id}.png"
                Image.new(
                    "RGB",
                    (4, 4),
                    color=(20 + seed * 10, 30 + rank * 20, 40),
                ).save(image_path)
                reference = action["type"] == "scheduler_baseline"
                timesteps = [1.0, 0.0]
                sigmas = [2.0, 1.0, 0.0]
                schedule_hash = provenance.json_sha256(
                    {"timesteps": timesteps, "sigmas": sigmas}
                )
                row = {
                    "id": row_id,
                    "prompt_index": 0,
                    "prompt": "a prompt",
                    "bucket": "test",
                    "seed": seed,
                    "action_id": action_id,
                    "action_type": action["type"],
                    "action": action,
                    "action_sha256": provenance.action_sha256(action),
                    "image_path": f"images/{row_id}.png",
                    "image_sha256": provenance.image_sha256(image_path),
                    "width": 4,
                    "height": 4,
                    "run_contract_sha256": config["run_contract_sha256"],
                    "provenance_schema": provenance.PROVENANCE_SCHEMA,
                    "registered_sampling": sampling,
                    "model_name": sampling["model"],
                    "base_scheduler_name": sampling["base_scheduler"],
                    "num_inference_steps": 2,
                    "unet_calls_per_step": [1, 1],
                    "extra_unet_calls": 0,
                    "error": None,
                    "scheduler_reference": reference,
                    "scheduler_name": (
                        action.get("scheduler_class")
                        if reference
                        else sampling["base_scheduler"]
                    ),
                    "scheduler_kwargs": action.get("scheduler_kwargs", {}),
                    "scheduler_solver_order": None,
                    "scheduler_order": 1,
                    "scheduler_init_noise_sigma": 2.0,
                    "scheduler_construction_init_noise_sigma": 2.0,
                    "scheduler_effective_init_noise_sigma": 2.0,
                    "scheduler_timesteps": timesteps,
                    "scheduler_sigmas": sigmas,
                    "scheduler_schedule_sha256": schedule_hash,
                    "scheduler_config_sha256_v2": "base-hash",
                    "active_scheduler_config_sha256_v2": (
                        "reference-hash" if reference else "base-hash"
                    ),
                    "device": "cuda:0",
                    "execution_rank": rank,
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
                score.update({key: 0.5 for key in validator.SCORE_KEYS})
                score["patch_ir_n"] = 5
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
            report = validator.validate(
                str(run_dir), str(actions), str(prompts), "scores"
            )
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["row_count"], 4)
            self.assertEqual(
                report["schedule_summary"]["no_correction"]["count"], 2
            )

    def test_schedule_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, actions, prompts = self.make_fixture(tmp)
            manifest_path = run_dir / "manifest.jsonl"
            rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
            rows[0]["scheduler_sigmas"][0] = 3.0
            manifest_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "schedule hash"):
                validator.validate(
                    str(run_dir), str(actions), str(prompts), "manifest"
                )

    def test_action_dependent_sigma_history_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, actions, prompts = self.make_fixture(tmp)
            manifest_path = run_dir / "manifest.jsonl"
            rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
            target = next(
                row
                for row in rows
                if row["action_id"] == "no_correction" and row["seed"] == 1
            )
            target["scheduler_effective_init_noise_sigma"] = 1.5
            manifest_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "distinct"):
                validator.validate(
                    str(run_dir), str(actions), str(prompts), "manifest"
                )


if __name__ == "__main__":
    unittest.main()
