import json
import importlib.util
import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare_actions = load_module("compare_actions", "eval-pipeline/compare_actions.py")
generate = load_module("generate", "eval-pipeline/generate.py")


class EvalPipelineTest(unittest.TestCase):
    def test_frequency_action_config_and_task_metadata(self):
        actions, cutoffs = generate.load_actions(
            ROOT / "eval-pipeline/configs/frequency_action_pilot.yaml", 50
        )
        self.assertEqual(cutoffs, [0.08, 0.25])
        self.assertEqual(actions[1]["delay_steps"], 3)
        self.assertEqual(len(actions[-1]["band_scales"]), 3)
        prompts = pd.DataFrame([{"index": 7, "TEXT": "test prompt", "bucket": "test"}])
        tasks = generate.build_tasks(prompts, [42], actions[:2])
        self.assertEqual(tasks[0]["id"], "p7_seed42_ano_ag")
        self.assertEqual(tasks[1]["action_id"], "conference_expert")
        groups = generate.group_tasks_by_pair(tasks)
        self.assertEqual(len(groups), 1)
        self.assertEqual([task["action_id"] for task in groups[0]], ["no_ag", "conference_expert"])

    def test_invalid_action_config_is_rejected(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as handle:
            handle.write("actions:\n  - id: bad/action\n    type: none\n")
            handle.flush()
            with self.assertRaises(ValueError):
                generate.load_actions(handle.name, 50)

    def test_missing_model_cache_is_rejected_before_workers_start(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            with self.assertRaises(FileNotFoundError):
                generate.validate_model_cache("missing/model", cache_dir)

    def test_resume_assignment_uses_full_design_order(self):
        prompts = pd.DataFrame(
            [
                {"index": 0, "TEXT": "first"},
                {"index": 1, "TEXT": "second"},
                {"index": 2, "TEXT": "third"},
            ]
        )
        actions = generate.scale_actions([0.0, 0.004])
        tasks = generate.build_tasks(prompts, [0], actions)
        with tempfile.TemporaryDirectory() as img_dir:
            for task in tasks[:2]:
                pathlib.Path(img_dir, task["id"] + ".png").touch()
                pathlib.Path(img_dir, task["id"] + ".json").write_text(
                    json.dumps({"device": "cuda:1"})
                )

            assigned = generate.assign_tasks_to_devices(
                tasks, ["cuda:1", "cuda:2"], img_dir
            )

        self.assertEqual(
            [task["prompt_index"] for task in assigned["cuda:2"]], [1, 1]
        )
        self.assertEqual(
            [task["prompt_index"] for task in assigned["cuda:1"]], [2, 2]
        )

    def test_resume_assignment_preserves_recorded_device(self):
        prompts = pd.DataFrame([{"index": 0, "TEXT": "test"}])
        tasks = generate.build_tasks(prompts, [0], generate.scale_actions([0.0, 0.004]))
        with tempfile.TemporaryDirectory() as img_dir:
            first = tasks[0]
            pathlib.Path(img_dir, first["id"] + ".png").touch()
            pathlib.Path(img_dir, first["id"] + ".json").write_text(
                json.dumps({"device": "cuda:2"})
            )
            assigned = generate.assign_tasks_to_devices(
                tasks, ["cuda:1", "cuda:2"], img_dir
            )

        self.assertNotIn("cuda:1", assigned)
        self.assertEqual([task["id"] for task in assigned["cuda:2"]], [tasks[1]["id"]])

    def test_resume_assignment_rejects_existing_cross_device_block(self):
        prompts = pd.DataFrame([{"index": 0, "TEXT": "test"}])
        tasks = generate.build_tasks(prompts, [0], generate.scale_actions([0.0, 0.004]))
        with tempfile.TemporaryDirectory() as img_dir:
            for task, device in zip(tasks, ["cuda:1", "cuda:2"]):
                pathlib.Path(img_dir, task["id"] + ".json").write_text(
                    json.dumps({"device": device})
                )
            with self.assertRaises(ValueError):
                generate.assign_tasks_to_devices(tasks, ["cuda:1", "cuda:2"], img_dir)

    def test_crossed_bootstrap_constant_effect(self):
        index = pd.MultiIndex.from_product(
            [[0, 1, 2], [10, 20]], names=["prompt_index", "seed"]
        )
        delta = pd.Series(np.ones(len(index)) * 0.25, index=index)
        low, high = compare_actions.crossed_bootstrap_ci(delta, n_boot=100, seed=1)
        self.assertAlmostEqual(low, 0.25)
        self.assertAlmostEqual(high, 0.25)
        self.assertLess(compare_actions.prompt_sign_flip_pvalue(delta, n_random=1000), 0.3)

    def test_holm_adjustment_is_monotone_in_rank(self):
        adjusted = compare_actions.holm_adjust([0.01, 0.04, 0.03])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])

    def test_cross_device_pairing_is_rejected(self):
        frame = pd.DataFrame(
            [
                {"prompt_index": 0, "seed": 0, "device": "cuda:1"},
                {"prompt_index": 0, "seed": 0, "device": "cuda:2"},
            ]
        )
        with self.assertRaises(ValueError):
            compare_actions.validate_pairing(frame)

    def test_missing_device_metadata_is_rejected(self):
        frame = pd.DataFrame([{"prompt_index": 0, "seed": 0}])
        with self.assertRaises(ValueError):
            compare_actions.validate_pairing(frame)

    def test_action_execution_order_is_deterministic(self):
        prompts = pd.DataFrame([{"index": 3, "TEXT": "test"}])
        actions = generate.scale_actions([0.0, 0.001, 0.002, 0.004])
        first_tasks = generate.build_tasks(prompts, [42], actions)
        second_tasks = generate.build_tasks(prompts, [42], actions)
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = generate.assign_tasks_to_devices(first_tasks, ["cuda:1"], first_dir)
            second = generate.assign_tasks_to_devices(second_tasks, ["cuda:1"], second_dir)

        first_ids = [task["id"] for task in first["cuda:1"]]
        second_ids = [task["id"] for task in second["cuda:1"]]
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(
            [task["execution_rank"] for task in first["cuda:1"]], list(range(4))
        )
        self.assertNotEqual(first_ids, [task["id"] for task in first_tasks])


if __name__ == "__main__":
    unittest.main()
