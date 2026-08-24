import json
import importlib.util
import pathlib
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval-pipeline"))


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare_actions = load_module("compare_actions", "eval-pipeline/compare_actions.py")
generate = load_module("generate", "eval-pipeline/generate.py")
from InferencePipelines.RepLDM.pipeline_repldm_sdxl import (  # noqa: E402
    _sample_resample_noise,
)
analyze_adaptivity = load_module(
    "analyze_adaptivity", "eval-pipeline/analyze_adaptivity.py"
)
select_fixed_action = load_module(
    "select_fixed_action", "eval-pipeline/select_fixed_action.py"
)


class EvalPipelineTest(unittest.TestCase):
    def test_fixed_action_selector_rejects_final_seed_leakage(self):
        prompts = pd.DataFrame(
            [{"index": 0, "TEXT": "train prompt", "split": "train"}]
        )
        frame = pd.DataFrame(
            [{"prompt_index": 0, "prompt": "train prompt", "seed": 42}]
        )
        with self.assertRaisesRegex(ValueError, "final-test seeds"):
            select_fixed_action.validate_train_design(prompts, frame, [0, 42, 123])

    def test_fixed_action_selector_rejects_non_train_prompt_file(self):
        prompts = pd.DataFrame(
            [{"index": 0, "TEXT": "validation prompt", "split": "validation"}]
        )
        frame = pd.DataFrame(
            [{"prompt_index": 0, "prompt": "validation prompt", "seed": 7}]
        )
        with self.assertRaisesRegex(ValueError, "split=train"):
            select_fixed_action.validate_train_design(prompts, frame, [0, 42, 123])

    def test_fixed_action_selector_applies_proxy_and_guard_rule(self):
        rows = []
        for prompt_index in (0, 1):
            for seed in (0, 42):
                device = "cuda:1" if prompt_index == 0 else "cuda:2"
                for action, hps, clip, clipped, saturation in (
                    ("no_ag", 0.0, 0.0, 0.0, 0.0),
                    ("eligible", 0.2, 0.0, 0.0, 0.0),
                    ("clip_bad", 0.5, -0.01, 0.0, 0.0),
                ):
                    rows.append(
                        {
                            "prompt_index": prompt_index,
                            "seed": seed,
                            "action_id": action,
                            "device": device,
                            "hpsv2": hps,
                            "clip_cosine": clip,
                            "clipped_fraction": clipped,
                            "mean_saturation": saturation,
                            "action": {"max_update_ratio": 0.05},
                            "latent_renderer_diagnostics": {
                                "update_ratio": [0.01],
                                "mean_error": [0.0],
                                "variance_error": [0.0],
                            },
                        }
                    )
        result = select_fixed_action.select_fixed_action(
            pd.DataFrame(rows), action_order=["no_ag", "eligible", "clip_bad"], bootstrap=100
        )
        self.assertEqual(result["selected_action"], "eligible")
        table = {row["action"]: row for row in result["rows"]}
        self.assertTrue(table["eligible"]["eligible"])
        self.assertFalse(table["clip_bad"]["eligible"])

    def test_fixed_action_selector_falls_back_to_no_ag(self):
        rows = []
        for seed in (0, 42):
            for action in ("no_ag", "unsafe"):
                rows.append(
                    {
                        "prompt_index": 0,
                        "seed": seed,
                        "action_id": action,
                        "device": "cuda:1",
                        "hpsv2": 0.0,
                        "clip_cosine": -0.01 if action == "unsafe" else 0.0,
                        "clipped_fraction": 0.0,
                        "mean_saturation": 0.0,
                        "action": {"max_update_ratio": 0.05},
                        "latent_renderer_diagnostics": {
                            "update_ratio": [0.01],
                            "mean_error": [0.0],
                            "variance_error": [0.0],
                        },
                    }
                )
        result = select_fixed_action.select_fixed_action(
            pd.DataFrame(rows), action_order=["no_ag", "unsafe"], bootstrap=100
        )
        self.assertEqual(result["selected_action"], "no_ag")

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

    def test_latent_renderer_fixed_action_config(self):
        with open(ROOT / "eval-pipeline/configs/latent_renderer_fixed_lr1.yaml") as handle:
            import yaml

            raw_config = yaml.safe_load(handle)
        self.assertEqual(raw_config["split_seeds"]["train_search"], [7, 19, 73])
        self.assertEqual(raw_config["split_seeds"]["test_final"], [0, 42, 123])
        actions, cutoffs = generate.load_actions(
            ROOT / "eval-pipeline/configs/latent_renderer_fixed_lr1.yaml", 50
        )
        self.assertEqual(cutoffs, [0.08, 0.25])
        self.assertEqual(len(actions), 10)
        fixed = next(action for action in actions if action["id"] == "semantic_pos")
        self.assertEqual(fixed["type"], "latent_renderer_fixed")
        self.assertEqual(fixed["coefficients"], [0.08, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(fixed["latent_renderer_provider"]["semantic_topk"], 16)
        prompts = pd.DataFrame([{"index": 3, "TEXT": "test prompt"}])
        tasks = generate.build_tasks(prompts, [0], actions[:2])
        self.assertEqual(tasks[1]["action_type"], "latent_renderer_fixed")

    def test_moment_tangent_config_and_runtime_wiring(self):
        actions, _ = generate.load_actions(
            ROOT / "eval-pipeline/configs/moment_tangent_smoke.yaml", 50
        )
        self.assertEqual(len(actions), 16)
        by_id = {action["id"]: action for action in actions}
        action = by_id["moment_tangent_rescaled_0.004"]
        self.assertEqual(action["residual_mode"], "moment_tangent_rescaled")

        controller, scale, density, decay = generate.guidance_runtime(action, 50)
        self.assertEqual(scale, 0.0)
        self.assertEqual(density, "all")
        self.assertIsNone(decay)
        self.assertEqual(controller(None).scale, 0.004)
        self.assertEqual(
            controller(None).residual_mode, "moment_tangent_rescaled"
        )

        raw_controller, raw_scale, _, _ = generate.guidance_runtime(
            by_id["raw_0.004"], 50
        )
        self.assertIsNone(raw_controller)
        self.assertEqual(raw_scale, 0.004)

        development_actions, _ = generate.load_actions(
            ROOT / "eval-pipeline/configs/moment_tangent_development.yaml", 50
        )
        self.assertEqual(len(development_actions), 10)

        cone_actions, _ = generate.load_actions(
            ROOT / "eval-pipeline/configs/trajectory_cone_smoke.yaml", 50
        )
        self.assertEqual(len(cone_actions), 11)
        cone = {action["id"]: action for action in cone_actions}[
            "trajectory_cone_0.002"
        ]
        cone_controller, cone_scale, _, _ = generate.guidance_runtime(cone, 50)
        self.assertEqual(cone_scale, 0.0)
        self.assertEqual(
            cone_controller(None).residual_mode, "trajectory_cone_tangent"
        )
        cone_development, _ = generate.load_actions(
            ROOT / "eval-pipeline/configs/trajectory_cone_development.yaml", 50
        )
        self.assertEqual(len(cone_development), 9)

        stage2_smoke, _ = generate.load_actions(
            ROOT / "eval-pipeline/configs/stage2_engineering_smoke.yaml", 50
        )
        self.assertEqual(len(stage2_smoke), 3)
        self.assertEqual(
            [action["type"] for action in stage2_smoke[:2]], ["none", "none"]
        )
        stage2_pilot, _ = generate.load_actions(
            ROOT / "eval-pipeline/configs/stage2_transfer_pilot.yaml", 50
        )
        self.assertEqual(len(stage2_pilot), 5)

    def test_stage2_requires_explicit_high_resolution_opt_in(self):
        stage1 = generate.generation_stage_settings(False, 1024, False)
        self.assertEqual(stage1["stage_name"], "stage1_1024")
        self.assertFalse(stage1["models_to_cpu"])

        stage2 = generate.generation_stage_settings(True, 2048, False)
        self.assertEqual(stage2["stage_name"], "stage2_2048")
        self.assertTrue(stage2["models_to_cpu"])
        self.assertTrue(stage2["multi_encoder"])
        self.assertFalse(stage2["multi_decoder"])
        self.assertEqual(stage2["init_rates"], [0.8])
        self.assertEqual(stage2["stage2_noise_source"], "task_generator")

        with self.assertRaisesRegex(ValueError, "explicit --stage2"):
            generate.generation_stage_settings(False, 2048, False)
        with self.assertRaisesRegex(ValueError, "greater than 1024"):
            generate.generation_stage_settings(True, 1024, False)
        with self.assertRaisesRegex(ValueError, "multiple of 8"):
            generate.generation_stage_settings(True, 2050, False)

    def test_stage2_resample_noise_uses_the_task_generator(self):
        latents = torch.empty(2, 4, 8, 8)
        first_generator = torch.Generator("cpu").manual_seed(123)
        second_generator = torch.Generator("cpu").manual_seed(123)
        torch.manual_seed(1)
        first = _sample_resample_noise(latents, first_generator)
        torch.manual_seed(999)
        second = _sample_resample_noise(latents, second_generator)
        self.assertTrue(torch.equal(first, second))

    def test_invalid_action_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "registration manifests"):
            generate.load_actions(
                ROOT / "eval-pipeline/configs/latent_renderer_mechanism_audit.yaml", 50
            )

        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as handle:
            handle.write("actions:\n  - id: bad/action\n    type: none\n")
            handle.flush()
            with self.assertRaises(ValueError):
                generate.load_actions(handle.name, 50)

        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as handle:
            handle.write(
                "actions:\n"
                "  - id: invalid_geometry\n"
                "    type: legacy\n"
                "    scale: 0.004\n"
                "    residual_mode: moment_tangent\n"
            )
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

    def test_seed_cv_reports_per_prompt_headroom(self):
        rows = []
        for prompt_index in [0, 1]:
            for seed in [0, 1, 2]:
                values = {
                    "no_ag": 0.0,
                    "action_a": 1.0 if prompt_index == 0 else -1.0,
                    "action_b": -1.0 if prompt_index == 0 else 1.0,
                }
                for action, score in values.items():
                    rows.append(
                        {
                            "prompt_index": prompt_index,
                            "seed": seed,
                            "device": "cuda:0",
                            "action_id": action,
                            "topiq_nr": score,
                        }
                    )
        result, selections = analyze_adaptivity.seed_cv_headroom(
            pd.DataFrame(rows),
            baseline="no_ag",
            selection_metric="topiq_nr",
            metrics=["topiq_nr"],
            n_boot=100,
            n_random=100,
        )

        means = result.set_index("comparison")["mean_delta"]
        self.assertAlmostEqual(means["global_static_vs_baseline"], 0.0)
        self.assertAlmostEqual(means["per_prompt_vs_baseline"], 1.0)
        self.assertAlmostEqual(means["per_prompt_vs_global"], 1.0)
        self.assertEqual(
            set(selections["per_prompt_action"]), {"action_a", "action_b"}
        )

    def test_seed_cv_does_not_use_held_out_seed_for_selection(self):
        rows = []
        scores = {
            "no_ag": [0.0, 0.0, 0.0],
            "action_a": [2.0, 2.0, -10.0],
            "action_b": [-1.0, -1.0, 10.0],
        }
        for prompt_index in [0, 1]:
            for seed in [0, 1, 2]:
                for action, values in scores.items():
                    rows.append(
                        {
                            "prompt_index": prompt_index,
                            "seed": seed,
                            "device": "cuda:0",
                            "action_id": action,
                            "topiq_nr": values[seed],
                        }
                    )
        _, selections = analyze_adaptivity.seed_cv_headroom(
            pd.DataFrame(rows),
            baseline="no_ag",
            selection_metric="topiq_nr",
            metrics=["topiq_nr"],
            n_boot=100,
            n_random=100,
        )

        held_out_two = selections[selections["held_out_seed"] == 2]
        self.assertEqual(set(held_out_two["per_prompt_action"]), {"action_a"})
        self.assertEqual(set(held_out_two["global_action"]), {"action_a"})

    def test_seed_cv_rejects_an_incomplete_action_block(self):
        frame = pd.DataFrame(
            [
                {
                    "prompt_index": prompt_index,
                    "seed": seed,
                    "device": "cuda:0",
                    "action_id": action,
                    "topiq_nr": 0.0,
                }
                for prompt_index in [0, 1]
                for seed in [0, 1]
                for action in ["no_ag", "candidate"]
                if not (prompt_index == 1 and seed == 1 and action == "candidate")
            ]
        )
        with self.assertRaisesRegex(ValueError, "complete prompt x seed x action"):
            analyze_adaptivity.seed_cv_headroom(
                frame,
                baseline="no_ag",
                selection_metric="topiq_nr",
                metrics=["topiq_nr"],
                n_boot=10,
                n_random=10,
            )


if __name__ == "__main__":
    unittest.main()
