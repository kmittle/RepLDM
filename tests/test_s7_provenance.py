import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

import pandas as pd
import yaml
from PIL import Image


ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_PIPELINE = ROOT / "eval-pipeline"
if str(EVAL_PIPELINE) not in sys.path:
    sys.path.insert(0, str(EVAL_PIPELINE))


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


provenance = load_module("s7_provenance_test", "eval-pipeline/s7_provenance.py")
generate = load_module("generate_s7_provenance_test", "eval-pipeline/generate.py")
freeze = load_module(
    "freeze_trajectory_correction_s7_test",
    "eval-pipeline/freeze_trajectory_correction_validation.py",
)


class S7ProvenanceTest(unittest.TestCase):
    @staticmethod
    def _valid_sidecar(root):
        root = pathlib.Path(root)
        image_dir = root / "images"
        image_dir.mkdir()
        image_path = image_dir / "p0_seed7_atest.png"
        Image.new("RGB", (4, 4), color=(20, 40, 60)).save(image_path)
        action = {
            "id": "test",
            "type": "trajectory_correction",
            "mix": 0.5,
            "noise_mode": "sqrt",
            "max_correction_ratio": None,
        }
        record = {
            "id": "p0_seed7_atest",
            "prompt_index": 0,
            "prompt": "a prompt",
            "seed": 7,
            "action_id": "test",
            "action_type": "trajectory_correction",
            "action": action,
            "action_sha256": provenance.action_sha256(action),
            "image_path": "images/p0_seed7_atest.png",
            "image_sha256": provenance.image_sha256(image_path),
            "run_contract_sha256": "a" * 64,
        }
        return image_path, record

    def test_tampered_png_and_sidecar_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            image_path, record = self._valid_sidecar(root)
            provenance.validate_sidecar(
                record, root, expected_contract_sha256="a" * 64
            )

            image_path.write_bytes(image_path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "image hash"):
                provenance.validate_sidecar(
                    record, root, expected_contract_sha256="a" * 64
                )

            # Restore the image, then alter the normalized action without its
            # digest. This exercises the sidecar integrity check independently.
            Image.new("RGB", (4, 4), color=(20, 40, 60)).save(image_path)
            record["action"]["mix"] = 0.75
            with self.assertRaisesRegex(ValueError, "action hash"):
                provenance.validate_sidecar(
                    record, root, expected_contract_sha256="a" * 64
                )

    def test_legacy_manifest_consolidation_keeps_partial_resume_semantics(self):
        prompts = pd.DataFrame([{"index": 0, "TEXT": "a prompt"}])
        actions = generate.scale_actions([0.0, 0.004])
        tasks = generate.build_tasks(prompts, [7], actions)
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            image_dir = root / "images"
            image_dir.mkdir()
            task = tasks[0]
            (image_dir / f"{task['id']}.png").touch()
            (image_dir / f"{task['id']}.json").write_text(
                json.dumps({"id": task["id"], "device": "cuda:0"})
            )
            count = generate.consolidate_manifest(
                str(root),
                {item["id"] for item in tasks},
                expected_tasks=tasks,
                run_contract_sha256=None,
                strict=False,
            )
            self.assertEqual(count, 1)
            self.assertEqual(len((root / "manifest.jsonl").read_text().splitlines()), 1)

    def test_missing_prompt_or_action_cells_are_rejected(self):
        complete = [
            {
                "id": f"p{prompt}_s{seed}_a{action}",
                "prompt_index": prompt,
                "seed": seed,
                "action_id": action,
            }
            for prompt in (0, 1)
            for seed in (7, 9)
            for action in ("a", "b")
        ]
        missing_action = [row for row in complete if row["action_id"] != "b" or row["prompt_index"] != 1]
        with self.assertRaisesRegex(ValueError, "incomplete design"):
            provenance.validate_design_rows(
                missing_action,
                expected_action_ids=("a", "b"),
                expected_seeds=(7, 9),
                expected_prompt_indices=(0, 1),
            )

        missing_prompt = [row for row in complete if row["prompt_index"] != 1]
        with self.assertRaisesRegex(ValueError, "observed prompt indices"):
            provenance.validate_design_rows(
                missing_prompt,
                expected_action_ids=("a", "b"),
                expected_seeds=(7, 9),
                expected_prompt_indices=(0, 1),
            )

    def test_stale_score_image_hash_is_rejected(self):
        manifest = [
            {
                "id": "p0_s7_atest",
                "image_sha256": "1" * 64,
                "run_contract_sha256": "2" * 64,
            }
        ]
        scores = [
            {
                "id": "p0_s7_atest",
                "image_sha256": "0" * 64,
                "run_contract_sha256": "2" * 64,
            }
        ]
        with self.assertRaisesRegex(ValueError, "image hash"):
            provenance.validate_scores_against_manifest(manifest, scores)

        scores[0]["image_sha256"] = manifest[0]["image_sha256"]
        scores[0]["run_contract_sha256"] = "3" * 64
        with self.assertRaisesRegex(ValueError, "run contract"):
            provenance.validate_scores_against_manifest(manifest, scores)

    def test_registered_model_and_scheduler_mismatches_are_rejected(self):
        actions = ROOT / "eval-pipeline/configs/trajectory_correction_development.yaml"
        prompts = ROOT / "eval-pipeline/prompts/trajectory_correction_heldout_v1.csv"
        kwargs = {
            "prompts_path": str(prompts),
            "resolution": 1024,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "stage2_enabled": False,
            "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
            "scheduler_name": "EulerDiscreteScheduler",
            "extra_unet_calls": 0,
        }
        generate.validate_registered_trajectory_design(str(actions), **kwargs)

        wrong_model = dict(kwargs, model_name="some/other-model")
        with self.assertRaisesRegex(ValueError, "model is registered"):
            generate.validate_registered_trajectory_design(str(actions), **wrong_model)

        wrong_scheduler = dict(kwargs, scheduler_name="EulerAncestralDiscreteScheduler")
        with self.assertRaisesRegex(ValueError, "scheduler is registered"):
            generate.validate_registered_trajectory_design(str(actions), **wrong_scheduler)

    @staticmethod
    def _freeze_fixture(root, selected_action, *, selected_eligible=True, selected_type="trajectory_correction"):
        root = pathlib.Path(root)
        root.mkdir(parents=True, exist_ok=True)
        template = root / "template.yaml"
        source = root / "development.yaml"
        selected_spec = {
            "id": selected_action,
            "type": selected_type,
            "selection_eligible": selected_eligible,
        }
        if selected_type == "trajectory_correction":
            selected_spec.update({"mix": 0.5, "noise_mode": "sqrt"})
        elif selected_type == "scheduler_baseline":
            selected_spec["scheduler_class"] = "EulerAncestralDiscreteScheduler"
        actions = [
            {"id": "no_correction", "type": "none"},
            selected_spec,
        ]
        template.write_text(
            yaml.safe_dump(
                {
                    "schema": "trajectory_correction_validation_v1",
                    "selected_action": None,
                    "actions": actions,
                },
                sort_keys=False,
            )
        )
        source.write_text(
            yaml.safe_dump(
                {"schema": "trajectory_correction_actions_v1", "actions": actions},
                sort_keys=False,
            )
        )
        run_dir = root / "run"
        run_dir.mkdir()
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "actions_sha256": freeze.sha256_file(str(source)),
                    "run_contract_sha256": "b" * 64,
                }
            )
        )
        (run_dir / "manifest.jsonl").write_text("")
        (run_dir / "scores.jsonl").write_text("")
        selection = run_dir / "selection.json"
        selection.write_text(
            json.dumps(
                {
                    "selected_action": selected_action,
                    "rows": [
                        {
                            "action": selected_action,
                            "passes_gate": True,
                            "selection_eligible": selected_eligible,
                        }
                    ],
                    "provenance": {
                        "actions_sha256": freeze.sha256_file(str(source)),
                        "run_dir": str(run_dir),
                        "config_sha256": freeze.sha256_file(str(run_dir / "config.json")),
                        "run_contract_sha256": "b" * 64,
                        "manifest_sha256": freeze.sha256_file(str(run_dir / "manifest.jsonl")),
                        "scores_sha256": freeze.sha256_file(str(run_dir / "scores.jsonl")),
                        "selector_version": "test",
                        "selector_script_sha256": "c" * 64,
                        "selector_git_commit": "fixture",
                    },
                }
            )
        )
        return selection, template, source

    def test_freeze_rejects_reference_and_noneligible_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            selection, template, source = self._freeze_fixture(
                pathlib.Path(tmp) / "reference",
                "reference",
                selected_type="scheduler_baseline",
            )
            with self.assertRaisesRegex(ValueError, "only trajectory_correction"):
                freeze.freeze(
                    str(selection), str(template), str(pathlib.Path(tmp) / "reference.yaml"), str(source)
                )

        with tempfile.TemporaryDirectory() as tmp:
            selection, template, source = self._freeze_fixture(
                pathlib.Path(tmp) / "ineligible",
                "ineligible",
                selected_eligible=False,
            )
            with self.assertRaisesRegex(ValueError, "not selection_eligible"):
                freeze.freeze(
                    str(selection), str(template), str(pathlib.Path(tmp) / "ineligible.yaml"), str(source)
                )

    def _run_queue_dry_run(self, root, selection):
        queue_dir = pathlib.Path(root) / "queue"
        env = os.environ.copy()
        env.update(
            {
                "S7_QUEUE_DIR": str(queue_dir),
                "S7_DEV_RUN_DIR": str(pathlib.Path(root) / "dev"),
                "S7_VAL_RUN_DIR": str(pathlib.Path(root) / "val"),
                "S7_VAL_ACTIONS": str(pathlib.Path(root) / "validation.yaml"),
                "S7_GEN_PYTHON": sys.executable,
                "S7_EVAL_PYTHON": sys.executable,
                "S7_MIN_FREE_MIB": "1",
                "S7_TEST_FREE_MIB": "1",
                "S7_TEST_GPU": "3",
                "S7_POLL_SECONDS": "1",
                "S7_DRY_RUN_SELECTION": selection,
            }
        )
        return subprocess.run(
            ["bash", str(ROOT / "eval-pipeline/run_trajectory_correction_queue.sh"), "--dry-run"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        ), queue_dir

    def test_queue_dry_run_pass_and_null_branches(self):
        if shutil.which("jq") is None or shutil.which("flock") is None:
            self.skipTest("queue shell test requires jq and flock")
        with tempfile.TemporaryDirectory() as tmp:
            result, queue_dir = self._run_queue_dry_run(tmp, "ancestral_mix_050")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            state = json.loads((queue_dir / "state.json").read_text())
            self.assertEqual(state["stage"], "validation_queue")
            self.assertEqual(state["status"], "awaiting_review")
            self.assertTrue(state["terminal"])

        with tempfile.TemporaryDirectory() as tmp:
            result, queue_dir = self._run_queue_dry_run(tmp, "no_correction")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            state = json.loads((queue_dir / "state.json").read_text())
            self.assertEqual(state["stage"], "null_route")
            self.assertEqual(state["status"], "null_route")
            self.assertTrue(state["terminal"])
            route = json.loads((queue_dir / "null_route.json").read_text())
            self.assertEqual(route["reason"], "selector_no_correction")


if __name__ == "__main__":
    unittest.main()
