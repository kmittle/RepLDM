import copy
import csv
import json
import importlib.util
import math
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval-pipeline"))


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare_actions = load_module("compare_actions", "eval-pipeline/compare_actions.py")
generate = load_module("generate", "eval-pipeline/generate.py")
score = load_module("score", "eval-pipeline/score.py")
from InferencePipelines.RepLDM.pipeline_repldm_sdxl import (  # noqa: E402
    _sample_resample_noise,
    _validate_trajectory_correction_generator,
)
from InferencePipelines.cfg_batch import (  # noqa: E402
    expand_cfg_latents,
    expand_cfg_time_ids,
    split_cfg_noise_pred,
)
analyze_adaptivity = load_module(
    "analyze_adaptivity", "eval-pipeline/analyze_adaptivity.py"
)
select_fixed_action = load_module(
    "select_fixed_action", "eval-pipeline/select_fixed_action.py"
)
freeze_validation = load_module(
    "freeze_latent_renderer_validation",
    "eval-pipeline/freeze_latent_renderer_validation.py",
)
audit_renderer_run = load_module(
    "audit_latent_renderer_run", "eval-pipeline/audit_latent_renderer_run.py"
)
scorer_provenance = load_module(
    "scorer_provenance_test", "eval-pipeline/scorer_provenance.py"
)
evaluate_renderer_validation = load_module(
    "evaluate_latent_renderer_validation",
    "eval-pipeline/evaluate_latent_renderer_validation.py",
)
blind_montage = load_module(
    "make_latent_renderer_blind_montage",
    "eval-pipeline/make_latent_renderer_blind_montage.py",
)
finalize_renderer_validation = load_module(
    "finalize_latent_renderer_validation",
    "eval-pipeline/finalize_latent_renderer_validation.py",
)
select_trajectory_correction = load_module(
    "select_trajectory_correction", "eval-pipeline/select_trajectory_correction.py"
)
freeze_trajectory_correction = load_module(
    "freeze_trajectory_correction", "eval-pipeline/freeze_trajectory_correction_validation.py"
)


class EvalPipelineTest(unittest.TestCase):
    def test_score_device_normalizes_numeric_gpu_indices(self):
        self.assertEqual(score.resolve_device("7", cuda_available=True), "cuda:7")
        self.assertEqual(
            score.resolve_device("cuda:3", cuda_available=True), "cuda:3"
        )
        self.assertEqual(score.resolve_device("cpu", cuda_available=True), "cpu")
        self.assertEqual(score.resolve_device("7", cuda_available=False), "cpu")

    def test_cfg_latent_expansion_matches_concatenated_embedding_order(self):
        latents = torch.tensor([[10.0], [20.0]])
        expanded = expand_cfg_latents(latents, enabled=True)
        torch.testing.assert_close(expanded, torch.tensor([[10.0], [20.0], [10.0], [20.0]]))
        negative, positive = expanded.chunk(2)
        torch.testing.assert_close(negative, latents)
        torch.testing.assert_close(positive, latents)
        negative, positive = split_cfg_noise_pred(
            torch.tensor([[100.0], [200.0], [300.0], [400.0]])
        )
        torch.testing.assert_close(negative, torch.tensor([[100.0], [200.0]]))
        torch.testing.assert_close(positive, torch.tensor([[300.0], [400.0]]))
        self.assertIs(expand_cfg_latents(latents, enabled=False), latents)

    def test_cfg_time_ids_expand_in_branch_block_order(self):
        branch_ids = torch.tensor([[10.0, 11.0], [20.0, 21.0]])
        expanded = expand_cfg_time_ids(branch_ids, batch_size=3, enabled=True)
        torch.testing.assert_close(
            expanded,
            torch.tensor(
                [
                    [10.0, 11.0],
                    [10.0, 11.0],
                    [10.0, 11.0],
                    [20.0, 21.0],
                    [20.0, 21.0],
                    [20.0, 21.0],
                ]
            ),
        )

        no_cfg = expand_cfg_time_ids(branch_ids[:1], batch_size=3, enabled=False)
        torch.testing.assert_close(no_cfg, branch_ids[:1].repeat(3, 1))

        with self.assertRaises(ValueError):
            expand_cfg_time_ids(branch_ids[:1], batch_size=2, enabled=True)

    def test_trajectory_correction_requires_single_generator(self):
        correction = generate.trajectory_correction_runtime(
            {"type": "trajectory_correction", "mix": 0.0, "noise_mode": "sqrt"}
        )
        _validate_trajectory_correction_generator(None, correction)
        _validate_trajectory_correction_generator(torch.Generator(), correction)
        with self.assertRaisesRegex(TypeError, "single torch.Generator"):
            _validate_trajectory_correction_generator(
                [torch.Generator()], correction
            )

    def test_generation_runtime_provenance_is_json_safe(self):
        provenance = generate.runtime_provenance()
        self.assertEqual(
            set(provenance),
            {
                "python_version",
                "torch_version",
                "diffusers_version",
                "cuda_runtime_version",
            },
        )
        json.dumps(provenance)
        self.assertTrue(provenance["python_version"])
        self.assertTrue(provenance["torch_version"])
        self.assertTrue(provenance["diffusers_version"])

    def test_freeu_action_is_normalized_and_reentrant(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "freeu.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "actions": [
                            {
                                "id": "freeu_dynamic",
                                "type": "freeu",
                                "knots": [
                                    {"position": 0.0, "parameters": [1, 1, 1, 1]},
                                    {"position": 1.0, "parameters": [0.6, 0.4, 1.1, 1.2]},
                                ],
                                "preserve_moments": True,
                            }
                        ]
                    }
                )
            )
            actions, _ = generate.load_actions(str(path), 4)
            self.assertEqual(actions[0]["type"], "freeu")
            self.assertTrue(actions[0]["freeu_preserve_moments"])
            schedule = generate.freeu_runtime(actions[0])
            self.assertEqual(schedule.at(1.0).as_tuple(), (0.6, 0.4, 1.1, 1.2))

    def test_trajectory_correction_action_is_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "trajectory.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "actions": [
                            {
                                "id": "ancestral_half",
                                "type": "trajectory_correction",
                                "mix": 0.5,
                                "noise_mode": "sqrt",
                                "max_correction_ratio": 0.2,
                            }
                        ]
                    }
                )
            )
            actions, _ = generate.load_actions(str(path), 4)
            self.assertEqual(actions[0]["type"], "trajectory_correction")
            correction = generate.trajectory_correction_runtime(actions[0])
            self.assertEqual(correction.to_record()["mix"], 0.5)
            self.assertEqual(correction.to_record()["max_correction_ratio"], 0.2)

    def test_trajectory_correction_registration_rejects_sampling_drift(self):
        config = ROOT / "eval-pipeline/configs/trajectory_correction_development.yaml"
        prompts = ROOT / "eval-pipeline/prompts/trajectory_correction_heldout_v1.csv"
        generate.validate_registered_trajectory_design(
            str(config),
            prompts_path=str(prompts),
            resolution=1024,
            num_inference_steps=50,
            guidance_scale=7.5,
            stage2_enabled=False,
        )
        with self.assertRaisesRegex(ValueError, "sampling is registered"):
            generate.validate_registered_trajectory_design(
                str(config),
                prompts_path=str(prompts),
                resolution=1024,
                num_inference_steps=30,
                guidance_scale=7.5,
                stage2_enabled=False,
            )

    def test_trajectory_validation_requires_and_filters_frozen_selection(self):
        config_path = ROOT / "eval-pipeline/configs/trajectory_correction_validation_template.yaml"
        with self.assertRaisesRegex(ValueError, "selected_action"):
            generate.load_actions(str(config_path), 50)
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "validation.yaml"
            source = yaml.safe_load(config_path.read_text())
            source["selected_action"] = "ancestral_mix_050"
            path.write_text(yaml.safe_dump(source, sort_keys=False))
            actions, _ = generate.load_actions(str(path), 50)
            self.assertEqual(
                [action["id"] for action in actions],
                ["no_correction", "euler_ancestral_reference", "ancestral_mix_050"],
            )

    def test_scheduler_reference_is_normalized_and_retained_for_validation(self):
        config_path = ROOT / "eval-pipeline/configs/trajectory_correction_validation_template.yaml"
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "validation.yaml"
            source = yaml.safe_load(config_path.read_text())
            source["selected_action"] = "ancestral_mix_050"
            path.write_text(yaml.safe_dump(source, sort_keys=False))
            actions, _ = generate.load_actions(str(path), 50)
            by_id = {action["id"]: action for action in actions}
            self.assertIn("euler_ancestral_reference", by_id)
            self.assertEqual(
                by_id["euler_ancestral_reference"]["type"], "scheduler_baseline"
            )
            self.assertFalse(by_id["euler_ancestral_reference"]["selection_eligible"])

    def test_trajectory_selector_applies_primary_and_pixel_gates(self):
        rows = []
        for prompt_index in range(8):
            for seed in (0, 42):
                for action_id, topiq_delta, clip_delta in (
                    ("no_correction", 0.0, 0.0),
                    ("euler_ancestral_reference", 0.004, 0.0),
                    ("ancestral_mix_050", 0.01, 0.0),
                    ("ancestral_mix_075", 0.01, 0.002),
                ):
                    rows.append(
                        {
                            "prompt_index": prompt_index,
                            "seed": seed,
                            "action_id": action_id,
                            "device": "cuda:1",
                            "topiq_nr": 0.5 + topiq_delta,
                            "hpsv2": 0.3,
                            "clip_cosine": 0.3 + clip_delta,
                            "clipped_fraction": 0.003 if action_id == "ancestral_mix_075" else 0.001,
                            "mean_saturation": 0.2,
                            "trajectory_correction_diagnostics": (
                                None
                                if action_id in {"no_correction", "euler_ancestral_reference"}
                                else [
                                    {
                                        "step_index": 0,
                                        "sigma_from": 1.0,
                                        "sigma_to": 0.5,
                                        "sigma_up": 0.25,
                                        "raw_correction_norm_ratio": 0.1,
                                        "applied_correction_norm_ratio": 0.1,
                                    }
                                ]
                            ),
                        }
                    )
        frame = pd.DataFrame(rows)
        result = select_trajectory_correction.select(
            frame,
            action_order=[
                "no_correction",
                "euler_ancestral_reference",
                "ancestral_mix_050",
                "ancestral_mix_075",
            ],
            selection_eligible=["ancestral_mix_050", "ancestral_mix_075"],
            bootstrap=500,
            seed=3,
        )
        self.assertEqual(result["selected_action"], "ancestral_mix_050")
        rows_by_action = {row["action"]: row for row in result["rows"]}
        self.assertIn("euler_ancestral_reference", result["reference_actions"])
        self.assertFalse(rows_by_action["euler_ancestral_reference"]["passes_gate"])
        self.assertTrue(rows_by_action["ancestral_mix_050"]["passes_gate"])
        self.assertFalse(rows_by_action["ancestral_mix_075"]["passes_gate"])

    def test_trajectory_validation_freezer_requires_passing_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            template = root / "template.yaml"
            template.write_text(
                yaml.safe_dump(
                    {
                        "schema": "trajectory_correction_validation_v1",
                        "selected_action": None,
                        "actions": [
                            {"id": "no_correction", "type": "none"},
                            {
                                "id": "ancestral_mix_050",
                                "type": "trajectory_correction",
                                "mix": 0.5,
                                "noise_mode": "sqrt",
                            },
                        ],
                    }
                )
            )
            source = root / "development.yaml"
            source.write_text(
                yaml.safe_dump(
                    {
                        "schema": "trajectory_correction_actions_v1",
                        "actions": [
                            {"id": "no_correction", "type": "none"},
                            {
                                "id": "ancestral_mix_050",
                                "type": "trajectory_correction",
                                "mix": 0.5,
                                "noise_mode": "sqrt",
                            },
                        ],
                    }
                )
            )
            run_dir = root / "development"
            run_dir.mkdir()
            (run_dir / "manifest.jsonl").write_text("")
            (run_dir / "scores.jsonl").write_text("")
            run_contract = "a" * 64
            run_config = {
                "actions_sha256": freeze_trajectory_correction.sha256_file(str(source)),
                "run_contract_sha256": run_contract,
            }
            (run_dir / "config.json").write_text(json.dumps(run_config))
            selection = run_dir / "selection.json"
            selection_provenance = {
                "actions_sha256": freeze_trajectory_correction.sha256_file(str(source)),
                "run_dir": str(run_dir),
                "config_sha256": freeze_trajectory_correction.sha256_file(str(run_dir / "config.json")),
                "run_contract_sha256": run_contract,
                "manifest_sha256": freeze_trajectory_correction.sha256_file(str(run_dir / "manifest.jsonl")),
                "scores_sha256": freeze_trajectory_correction.sha256_file(str(run_dir / "scores.jsonl")),
                "selector_version": "test",
                "selector_script_sha256": "b" * 64,
                "selector_git_commit": "test-commit",
            }
            selection.write_text(
                json.dumps(
                    {
                        "selected_action": "ancestral_mix_050",
                        "gate": {"primary_metric": "topiq_nr"},
                        "rows": [{"action": "ancestral_mix_050", "passes_gate": True, "selection_eligible": True}],
                        "provenance": selection_provenance,
                    }
                )
            )
            output = root / "frozen.yaml"
            frozen = freeze_trajectory_correction.freeze(
                str(selection), str(template), str(output), str(source)
            )
            self.assertEqual(frozen["selected_action"], "ancestral_mix_050")
            self.assertTrue(output.exists())

    @staticmethod
    def _registered_validation_inputs():
        with open(ROOT / "eval-pipeline/configs/latent_renderer_fixed_lr1.yaml") as handle:
            source = yaml.safe_load(handle)
        with open(
            ROOT / "eval-pipeline/configs/latent_renderer_validation_template.yaml"
        ) as handle:
            template = yaml.safe_load(handle)
        action_ids = [action["id"] for action in source["actions"]]
        candidates = [
            action["id"]
            for action in source["actions"]
            if action["type"] == "latent_renderer_fixed"
        ]
        requirements = template["train_selection_requirements"]
        selection = {
            "selected_action": "spectral_low_pos",
            "baseline": requirements["baseline"],
            "candidate_actions": candidates,
            "selection_metric": requirements["selection_metric"],
            "topiq_used_for_selection": requirements["topiq_used_for_selection"],
            "require_positive_hpsv2_ci": requirements[
                "require_positive_hpsv2_ci"
            ],
            "constraints": requirements["constraints"],
            "bootstrap": requirements["bootstrap"],
            "seed": requirements["seed"],
            "registration": {
                "source_schema": source["schema"],
                "source_experiment_id": source["experiment_id"],
                "split_role": requirements["split_role"],
                "seeds": source["split_seeds"][requirements["split_role"]],
                "action_ids": action_ids,
                "candidate_actions": candidates,
            },
        }
        return source, template, selection

    @staticmethod
    def _native_renderer_ledger(steps=3):
        if steps <= 0:
            raise ValueError("test ledger requires a positive step count")
        sigma_pairs = [
            (float(steps - step_index), float(steps - step_index - 1))
            for step_index in range(steps)
        ]
        provider = {
            "semantic_entropy": [0.5],
            "basis_rms": [0.25],
        }
        records = []
        for step_index, (sigma_from, sigma_to) in enumerate(sigma_pairs):
            records.append(
                {
                    "step_index": step_index,
                    "scheduler_step_index": step_index,
                    "timestep": float(steps - step_index),
                    "sigma_from": sigma_from,
                    "sigma_to": sigma_to,
                    "prediction_type": "epsilon",
                    "raw_update_norm": [0.2],
                    "bounded_update_norm": [0.1],
                    "scheduler_update_norm": [1.0],
                    "update_ratio": [0.1],
                    "mean_error": [0.0],
                    "variance_error": [0.0],
                    "clean_update_gain": [1.0 - sigma_to / sigma_from],
                    "applied_update_norm": [0.01],
                    "applied_update_ratio": [0.01],
                    "provider_diagnostics": copy.deepcopy(provider),
                }
            )
        return records

    @staticmethod
    def _write_renderer_audit_fixture(
        root, *, duplicate_images=False, native=False
    ):
        root = pathlib.Path(root)
        run_dir = root / "run"
        image_dir = run_dir / "images"
        image_dir.mkdir(parents=True)
        provider = {
            "feature_block": "up_blocks.0",
            "semantic_layer": "test.attn1",
            "semantic_mode": "reciprocal_semantic",
            "semantic_topk": 16,
            "permutation_seed": 1729,
            "prompt_dim": 0,
            "state_dim": 0,
        }
        if native:
            provider.update(
                {
                    "scheduler_mapping": "euler_clean_endpoint",
                    "basis_normalization": "match_rms",
                }
            )
        source = {
            "schema": "latent_renderer_actions_v1",
            "experiment_id": "fixture",
            "split_seeds": {"train_search": [7]},
            "latent_renderer_provider": provider,
            "frequency_band_cutoffs": [0.08, 0.25],
            "actions": [
                {"id": "no_ag", "type": "none"},
                {
                    "id": "semantic_pos",
                    "type": "latent_renderer_fixed",
                    "coefficients": [0.08, 0.0, 0.0, 0.0, 0.0, 0.0],
                },
            ],
        }
        source_path = root / "actions.yaml"
        source_path.write_text(yaml.safe_dump(source, sort_keys=False))
        prompts_path = root / "prompts.csv"
        pd.DataFrame(
            [{"index": 0, "TEXT": "a test prompt", "split": "train"}]
        ).to_csv(prompts_path, index=False)
        configured_actions = [
            {"id": "no_ag", "type": "none"},
            {
                "id": "semantic_pos",
                "type": "latent_renderer_fixed",
                "coefficients": [0.08, 0.0, 0.0, 0.0, 0.0, 0.0],
                "latent_renderer_provider": provider,
                "max_update_ratio": 0.05,
            },
        ]
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "resolution": 8,
                    "num_inference_steps": 3,
                    "split_role": "train_search",
                    "seeds": [7],
                    "actions": configured_actions,
                    "frequency_band_cutoffs": [0.08, 0.25],
                }
            )
        )
        manifest = []
        scores = []
        for rank, action in enumerate(configured_actions):
            action_id = action["id"]
            row_id = f"p0_seed7_a{action_id}"
            color = (10, 20, 30) if rank == 0 or duplicate_images else (30, 20, 10)
            image_path = image_dir / f"{row_id}.png"
            Image.new("RGB", (8, 8), color=color).save(image_path)
            diagnostics = None
            provider_diagnostics = None
            step_diagnostics = None
            if action["type"] == "latent_renderer_fixed":
                provider_diagnostics = {
                    "semantic_entropy": [0.5],
                    "basis_rms": [0.25],
                }
                if native:
                    step_diagnostics = EvalPipelineTest._native_renderer_ledger()
                    diagnostics = {
                        key: copy.deepcopy(value)
                        for key, value in step_diagnostics[-1].items()
                        if key
                        not in {
                            "step_index",
                            "scheduler_step_index",
                            "timestep",
                            "sigma_from",
                            "sigma_to",
                            "prediction_type",
                            "provider_diagnostics",
                        }
                    }
                else:
                    diagnostics = {
                        "update_ratio": [0.01],
                        "mean_error": [0.0],
                        "variance_error": [0.0],
                    }
            manifest.append(
                {
                    "id": row_id,
                    "prompt_index": 0,
                    "prompt": "a test prompt",
                    "seed": 7,
                    "action_id": action_id,
                    "action_type": action["type"],
                    "action": action,
                    "execution_rank": rank,
                    "image_path": f"images/{row_id}.png",
                    "device": "cuda:1",
                    "latent_renderer_diagnostics": diagnostics,
                    "latent_renderer_provider_diagnostics": provider_diagnostics,
                    "latent_renderer_scheduler_mapping": (
                        "euler_clean_endpoint"
                        if action["type"] == "latent_renderer_fixed" and native
                        else None
                    ),
                    "latent_renderer_step_diagnostics": step_diagnostics,
                    "num_inference_steps": 3,
                    "scheduler_name": "EulerDiscreteScheduler",
                    "unet_calls_per_step": [1, 1, 1],
                    "extra_unet_calls": 0,
                }
            )
            score = {"id": row_id}
            score.update({key: 0.1 for key in audit_renderer_run.DEFAULT_SCORE_KEYS})
            if native:
                checkpoint_files = []
                provenance = {
                    "schema": scorer_provenance.SCORER_PROVENANCE_SCHEMA,
                    "metrics": ["fixture"],
                    "config_params": {},
                    "runtime": {
                        "device": "cpu",
                        "python_implementation": "CPython",
                        "python_version": "3.11.0",
                        "torch_version": "2.5.1",
                        "cuda_runtime_version": None,
                        "cudnn_version": None,
                        "cuda_device": None,
                        "runner_package_versions": {},
                    },
                    "shared_preprocessing": {},
                    "implementation": {
                        "score_runner": {"sha256": "1" * 64},
                        "scorer_base": {"sha256": "2" * 64},
                    },
                    "scorers": [
                        {
                            "name": "fixture",
                            "class": "tests.FixtureScorer",
                            "plugin_source": {"sha256": "0" * 64},
                            "supporting_sources": [],
                            "package_versions": {},
                            "models": [],
                            "checkpoints": {
                                "schema": scorer_provenance.CHECKPOINT_MANIFEST_SCHEMA,
                                "files": checkpoint_files,
                                "files_sha256": scorer_provenance.json_sha256(
                                    checkpoint_files
                                ),
                            },
                            "preprocessing": {},
                            "parameters": {},
                            "output_keys": [],
                        }
                    ],
                }
                score["scorer_provenance"] = provenance
                score["scorer_provenance_sha256"] = (
                    scorer_provenance.json_sha256(provenance)
                )
            scores.append(score)
        (run_dir / "manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in manifest)
        )
        (run_dir / "scores.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in scores)
        )
        return run_dir, prompts_path, source_path

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

    def test_fixed_action_selector_requires_hps_interval_above_baseline(self):
        rows = []
        for prompt_index, candidate_hps in ((0, 0.10), (1, -0.09)):
            for action, hps in (("no_ag", 0.0), ("uncertain", candidate_hps)):
                rows.append(
                    {
                        "prompt_index": prompt_index,
                        "seed": 7,
                        "action_id": action,
                        "device": "cuda:1",
                        "hpsv2": hps,
                        "clip_cosine": 0.0,
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
            pd.DataFrame(rows),
            action_order=["no_ag", "uncertain"],
            bootstrap=1000,
        )
        self.assertEqual(result["selected_action"], "no_ag")
        self.assertTrue(result["require_positive_hpsv2_ci"])

    def test_registered_train_run_requires_exact_seeds_actions_and_coefficients(self):
        source, _, _ = self._registered_validation_inputs()
        run_actions = []
        for registered in source["actions"]:
            generated = copy.deepcopy(registered)
            if generated["type"] == "latent_renderer_fixed":
                generated["latent_renderer_provider"] = copy.deepcopy(
                    source["latent_renderer_provider"]
                )
            run_actions.append(generated)
        run_config = {
            "seeds": source["split_seeds"]["train_search"],
            "actions": run_actions,
            "frequency_band_cutoffs": source["frequency_band_cutoffs"],
        }
        frame = pd.DataFrame(
            [
                {"seed": seed, "action_id": action["id"]}
                for seed in source["split_seeds"]["train_search"]
                for action in source["actions"]
            ]
        )
        registration = select_fixed_action.validate_registered_train_run(
            frame, run_config, source
        )
        self.assertEqual(registration["split_role"], "train_search")
        self.assertEqual(
            registration["action_ids"], [action["id"] for action in source["actions"]]
        )

        wrong_seeds = copy.deepcopy(run_config)
        wrong_seeds["seeds"] = [8, 20, 74]
        with self.assertRaisesRegex(ValueError, "exactly the registered"):
            select_fixed_action.validate_registered_train_run(frame, wrong_seeds, source)

        wrong_coefficient = copy.deepcopy(run_config)
        wrong_coefficient["actions"][1]["coefficients"][0] = 0.09
        with self.assertRaisesRegex(ValueError, "coefficients differ"):
            select_fixed_action.validate_registered_train_run(
                frame, wrong_coefficient, source
            )

    def test_validation_freezer_matches_random_control_norm(self):
        source, template, selection = self._registered_validation_inputs()
        frozen = freeze_validation.freeze_validation_config(
            selection, source, template
        )
        self.assertEqual(
            [action["id"] for action in frozen["actions"]],
            ["no_ag", "spectral_low_pos", "conference_expert", "matched_random"],
        )
        selected_norm = math.sqrt(
            sum(value * value for value in frozen["actions"][1]["coefficients"])
        )
        random_norm = math.sqrt(
            sum(value * value for value in frozen["actions"][3]["coefficients"])
        )
        self.assertAlmostEqual(selected_norm, random_norm)
        self.assertEqual(
            frozen["validation_requirements"]["seeds"], [11, 29, 101]
        )
        self.assertEqual(
            frozen["split_seeds"], {"validation_confirmation": [11, 29, 101]}
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            frozen_path = pathlib.Path(temp_dir) / "validation.yaml"
            frozen_path.write_text(yaml.safe_dump(frozen, sort_keys=False))
            actions, _ = generate.load_actions(frozen_path, 50)
            self.assertEqual(len(actions), 4)
            generate.validate_split_seed_role(
                frozen_path, "validation_confirmation", [11, 29, 101]
            )
            with self.assertRaisesRegex(ValueError, "unknown --split_role"):
                generate.validate_split_seed_role(
                    frozen_path, "test_final", [0, 42, 123]
                )

    def test_validation_freezer_rejects_selection_protocol_drift(self):
        source, template, selection = self._registered_validation_inputs()
        mutations = (
            ("TOPIQ leakage", lambda item: item.update(topiq_used_for_selection=True)),
            ("bootstrap drift", lambda item: item.update(bootstrap=999)),
            (
                "seed drift",
                lambda item: item["registration"].update(seeds=[8, 20, 74]),
            ),
            (
                "candidate omission",
                lambda item: item.update(candidate_actions=item["candidate_actions"][:-1]),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                changed = copy.deepcopy(selection)
                mutate(changed)
                with self.assertRaises(ValueError):
                    freeze_validation.freeze_validation_config(
                        changed, source, template
                    )

    def test_validation_gate_requires_controls_and_qualitative_review(self):
        source, template, selection = self._registered_validation_inputs()
        frozen = freeze_validation.freeze_validation_config(
            selection, source, template
        )
        frozen["validation_requirements"]["bootstrap"] = 100
        frozen["validation_requirements"]["randomizations"] = 5000
        rows = []
        selected = frozen["selected_action"]
        for prompt_index in range(12):
            for seed in (11, 29, 101):
                for action in ("no_ag", selected, "conference_expert", "matched_random"):
                    topiq = 0.0
                    if action in ("conference_expert", "matched_random"):
                        topiq = 0.005
                    if action == selected:
                        topiq = 0.02
                    rows.append(
                        {
                            "prompt_index": prompt_index,
                            "seed": seed,
                            "action_id": action,
                            "device": "cuda:1",
                            "topiq_nr": topiq,
                            "clip_cosine": 0.3,
                            "hpsv2": 0.25,
                            "clipped_fraction": 0.0,
                            "mean_saturation": 0.2,
                        }
                    )
        frame = pd.DataFrame(rows)
        result = evaluate_renderer_validation.evaluate_validation(frame, frozen)
        self.assertTrue(result["statistical_pass"])
        self.assertFalse(result["validation_pass"])
        self.assertEqual(result["decision"], "qualitative_review_required")
        self.assertEqual(
            {row["comparator"] for row in result["primary"]},
            {"no_ag", "conference_expert", "matched_random"},
        )

        failed = frame.copy()
        failed.loc[failed["action_id"] == selected, "topiq_nr"] = 0.001
        failed_result = evaluate_renderer_validation.evaluate_validation(failed, frozen)
        self.assertFalse(failed_result["statistical_pass"])
        self.assertEqual(failed_result["decision"], "close_lr1")

    def test_blind_montage_is_deterministic_and_hides_action_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            run_dir, prompts_path, _ = self._write_renderer_audit_fixture(root)
            frozen_path = root / "frozen.yaml"
            frozen_path.write_text(
                yaml.safe_dump(
                    {
                        "selected_action": "semantic_pos",
                        "validation_requirements": {
                            "qualitative_montage": {
                                "prompt_indices": [0],
                                "seed": 7,
                                "actions": ["no_ag", "selected_action"],
                                "blinding_seed": 123,
                            }
                        },
                    },
                    sort_keys=False,
                )
            )
            first = root / "blind_a"
            second = root / "blind_b"
            result_a = blind_montage.build_blind_package(
                run_dir, prompts_path, frozen_path, first
            )
            result_b = blind_montage.build_blind_package(
                run_dir, prompts_path, frozen_path, second
            )
            self.assertEqual(result_a["pairs"], 1)
            self.assertEqual(
                (first / "pair_p0_s7.png").read_bytes(),
                (second / "pair_p0_s7.png").read_bytes(),
            )
            self.assertEqual(result_a["private_file"], "review_key.json")
            self.assertNotIn("semantic_pos", (first / "review_prompts.csv").read_text())
            self.assertTrue((first / "review_form_template.csv").is_file())
            key = json.loads((first / "review_key.json").read_text())
            self.assertIn(key["pairs"][0]["left_action"], {"no_ag", "semantic_pos"})

    def test_finalizer_requires_blinded_review_before_test_authorization(self):
        source, template, selection = self._registered_validation_inputs()
        frozen = freeze_validation.freeze_validation_config(
            selection, source, template
        )
        frozen["validation_requirements"]["qualitative_montage"].update(
            {
                "prompt_indices": [0, 1],
                "minimum_reviewers": 2,
                "minimum_overall_preference_rate": 0.55,
                "overall_wilson_ci_low": 0.5,
                "minimum_positive_dimensions": 2,
                "dimension_minimum_win_rate": 0.5,
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            frozen_path = root / "frozen.yaml"
            gate_path = root / "gate.json"
            key_path = root / "key.json"
            frozen_path.write_text(yaml.safe_dump(frozen, sort_keys=False))
            gate_path.write_text(json.dumps({"statistical_pass": True}))
            key = {
                "selected_action": frozen["selected_action"],
                "pairs": [
                    {
                        "pair_id": "pair_p0_s11",
                        "left_action": frozen["selected_action"],
                        "right_action": "no_ag",
                    },
                    {
                        "pair_id": "pair_p1_s11",
                        "left_action": frozen["selected_action"],
                        "right_action": "no_ag",
                    },
                ],
            }
            key_path.write_text(json.dumps(key))
            form_paths = []
            header = [
                "reviewer_id",
                "pair_id",
                "overall",
                "structure",
                "text",
                "counting",
                "position",
                "detail",
            ]
            for reviewer in ("r1", "r2"):
                form_path = root / f"{reviewer}.csv"
                with form_path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=header)
                    writer.writeheader()
                    for pair_id in ("pair_p0_s11", "pair_p1_s11"):
                        writer.writerow(
                            {
                                "reviewer_id": reviewer,
                                "pair_id": pair_id,
                                "overall": "A",
                                "structure": "A",
                                "text": "A",
                                "counting": "A",
                                "position": "A",
                                "detail": "A",
                            }
                        )
                form_paths.append(form_path)
            summary = finalize_renderer_validation.summarize_reviews(
                key,
                form_paths,
                frozen["validation_requirements"]["qualitative_montage"],
            )
            self.assertTrue(summary["passed"])
            final_config, authorization = finalize_renderer_validation.finalize(
                frozen,
                {"statistical_pass": True},
                key,
                summary,
                frozen_path=str(frozen_path),
                gate_path=str(gate_path),
                key_path=str(key_path),
            )
            final_path = root / "final.yaml"
            final_path.write_text(yaml.safe_dump(final_config, sort_keys=False))
            authorization["source_actions_sha256"] = generate.sha256_file(final_path)
            auth_path = root / "authorization.json"
            auth_path.write_text(json.dumps(authorization))
            generate.validate_split_seed_role(
                final_path, "test_final", [0, 42, 123]
            )
            generate.validate_final_test_authorization(
                auth_path, final_path, [0, 42, 123]
            )

    def test_latent_renderer_run_audit_accepts_complete_safe_design(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir, prompts_path, source_path = self._write_renderer_audit_fixture(
                temp_dir
            )
            report = audit_renderer_run.audit_run(
                run_dir,
                prompts_path,
                source_path,
                split_role="train_search",
            )
        self.assertTrue(report["passed"])
        self.assertEqual(report["records"], 2)
        self.assertEqual(report["blocks"], 1)
        self.assertAlmostEqual(report["max_update_ratio"], 0.01)

    def test_latent_renderer_run_audit_rejects_duplicate_action_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir, prompts_path, source_path = self._write_renderer_audit_fixture(
                temp_dir, duplicate_images=True
            )
            with self.assertRaisesRegex(ValueError, "identical PNG hashes"):
                audit_renderer_run.audit_run(
                    run_dir,
                    prompts_path,
                    source_path,
                    split_role="train_search",
                )

    def test_native_renderer_run_audit_accepts_complete_step_ledger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir, prompts_path, source_path = self._write_renderer_audit_fixture(
                temp_dir, native=True
            )
            report = audit_renderer_run.audit_run(
                run_dir,
                prompts_path,
                source_path,
                split_role="train_search",
            )
        self.assertTrue(report["passed"])
        self.assertAlmostEqual(report["max_update_ratio"], 0.01)

    def test_native_renderer_run_audit_rejects_incomplete_ledger_and_nfe(self):
        for mutation, message in (
            (
                lambda row: row["latent_renderer_step_diagnostics"].pop(),
                "must contain 3 steps",
            ),
            (
                lambda row: row.__setitem__("unet_calls_per_step", [1, 2, 1]),
                "one-UNet-call",
            ),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temp_dir:
                run_dir, prompts_path, source_path = self._write_renderer_audit_fixture(
                    temp_dir, native=True
                )
                manifest_path = run_dir / "manifest.jsonl"
                rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
                renderer_row = next(
                    row for row in rows if row["action_type"] == "latent_renderer_fixed"
                )
                mutation(renderer_row)
                manifest_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows)
                )
                with self.assertRaisesRegex(ValueError, message):
                    audit_renderer_run.audit_run(
                        run_dir,
                        prompts_path,
                        source_path,
                        split_role="train_search",
                    )

    def test_native_renderer_run_audit_rejects_mapping_and_normalization_drift(self):
        for target, value, message in (
            ("config", "legacy_l2_to_rms", "basis_normalization"),
            ("sidecar", "legacy_unit", "runtime scheduler mapping"),
        ):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temp_dir:
                run_dir, prompts_path, source_path = self._write_renderer_audit_fixture(
                    temp_dir, native=True
                )
                if target == "config":
                    config_path = run_dir / "config.json"
                    config = json.loads(config_path.read_text())
                    renderer_action = next(
                        action
                        for action in config["actions"]
                        if action["type"] == "latent_renderer_fixed"
                    )
                    renderer_action["latent_renderer_provider"][
                        "basis_normalization"
                    ] = value
                    config_path.write_text(json.dumps(config))
                else:
                    manifest_path = run_dir / "manifest.jsonl"
                    rows = [
                        json.loads(line)
                        for line in manifest_path.read_text().splitlines()
                    ]
                    renderer_row = next(
                        row
                        for row in rows
                        if row["action_type"] == "latent_renderer_fixed"
                    )
                    renderer_row["latent_renderer_scheduler_mapping"] = value
                    manifest_path.write_text(
                        "".join(json.dumps(row) + "\n" for row in rows)
                    )
                with self.assertRaisesRegex(ValueError, message):
                    audit_renderer_run.audit_run(
                        run_dir,
                        prompts_path,
                        source_path,
                        split_role="train_search",
                    )

    def test_native_renderer_run_audit_requires_uniform_scorer_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir, prompts_path, source_path = self._write_renderer_audit_fixture(
                temp_dir, native=True
            )
            scores_path = run_dir / "scores.jsonl"
            rows = [json.loads(line) for line in scores_path.read_text().splitlines()]
            rows[-1]["scorer_provenance_sha256"] = "f" * 64
            scores_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            with self.assertRaisesRegex(ValueError, "hash drifted"):
                audit_renderer_run.audit_run(
                    run_dir,
                    prompts_path,
                    source_path,
                    split_role="train_search",
                )

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
            raw_config = yaml.safe_load(handle)
        self.assertEqual(raw_config["split_seeds"]["train_search"], [7, 19, 73])
        self.assertEqual(raw_config["split_seeds"]["test_final"], [0, 42, 123])
        actions, cutoffs = generate.load_actions(
            ROOT / "eval-pipeline/configs/latent_renderer_fixed_lr1.yaml", 50
        )
        self.assertEqual(cutoffs, [0.08, 0.25])
        self.assertEqual(len(actions), 10)
        fixed = next(action for action in actions if action["id"] == "semantic_pos")
        self.assertFalse(generate.has_scheduler_native_renderer(actions))
        self.assertEqual(fixed["type"], "latent_renderer_fixed")
        self.assertEqual(fixed["coefficients"], [0.08, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(fixed["latent_renderer_provider"]["semantic_topk"], 16)
        self.assertEqual(
            fixed["latent_renderer_provider"]["scheduler_mapping"], "legacy_unit"
        )
        self.assertEqual(
            fixed["latent_renderer_provider"]["basis_normalization"],
            "legacy_l2_to_rms",
        )
        prompts = pd.DataFrame([{"index": 3, "TEXT": "test prompt"}])
        tasks = generate.build_tasks(prompts, [0], actions[:2])
        self.assertEqual(tasks[1]["action_type"], "latent_renderer_fixed")

    def test_native_renderer_config_normalization_and_runtime_propagation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "native.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "latent_renderer_provider": {},
                        "actions": [
                            {
                                "id": "native",
                                "type": "latent_renderer_fixed",
                                "coefficients": [0.01, 0, 0, 0, 0, 0],
                                "provider": {
                                    "scheduler_mapping": "euler_clean_endpoint",
                                    "basis_normalization": "match_rms",
                                },
                            }
                        ],
                    }
                )
            )
            actions, _ = generate.load_actions(str(path), 3)
        action = actions[0]
        provider_config = action["latent_renderer_provider"]
        self.assertEqual(provider_config["scheduler_mapping"], "euler_clean_endpoint")
        self.assertEqual(provider_config["basis_normalization"], "match_rms")
        self.assertTrue(generate.has_scheduler_native_renderer(actions))

        fake_provider = mock.sentinel.provider
        fake_pipe = mock.Mock(unet=mock.sentinel.unet)
        with mock.patch.object(
            generate,
            "StructuralUNetBasisProvider",
            return_value=fake_provider,
        ):
            renderer, provider = generate.latent_renderer_runtime(
                action, fake_pipe, "cpu", 7.5
            )
        self.assertEqual(renderer.config.basis_normalization, "match_rms")
        kwargs = generate.latent_renderer_pipeline_kwargs(
            action, renderer, provider
        )
        self.assertIs(kwargs["latent_renderer"], renderer)
        self.assertIs(kwargs["latent_renderer_basis_provider"], fake_provider)
        self.assertEqual(
            kwargs["latent_renderer_scheduler_mapping"], "euler_clean_endpoint"
        )
        euler = type("EulerDiscreteScheduler", (), {})()
        self.assertEqual(
            generate.validate_latent_renderer_scheduler(action, euler),
            "euler_clean_endpoint",
        )
        with self.assertRaisesRegex(RuntimeError, "exactly EulerDiscreteScheduler"):
            generate.validate_latent_renderer_scheduler(action, object())

    def test_native_renderer_no_op_skips_step_diagnostics(self):
        registered = {"native_renderer_registered": True}
        self.assertFalse(
            generate.native_renderer_step_diagnostics_required(
                registered,
                {"id": "no_op", "type": "none"},
            )
        )
        self.assertTrue(
            generate.native_renderer_step_diagnostics_required(
                registered,
                {
                    "id": "lazy_zero_identity",
                    "type": "latent_renderer_fixed",
                    "latent_renderer_provider": {
                        "scheduler_mapping": "euler_clean_endpoint"
                    },
                },
            )
        )

    def test_native_provider_action_level_aliases_override_shared_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "native_alias.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "required_provider": {
                            "implementation": "LazyLatentStructureBasisProvider",
                            "provenance_id": "shared-default",
                            "scheduler_mapping": "euler_clean_endpoint",
                            "basis_normalization": "match_rms",
                        },
                        "actions": [
                            {
                                "id": "spectral",
                                "type": "latent_renderer_fixed",
                                "provider_provenance_id": "action-specific",
                                "requested_bases": ["reciprocal_semantic_transport"],
                                "required_hook_names": [
                                    "up_blocks.0.attentions.0.transformer_blocks.0.attn1_qk"
                                ],
                                "coefficients": [0.01, 0, 0, 0, 0, 0],
                            }
                        ],
                    }
                )
            )
            actions, _ = generate.load_actions(str(path), 3)
        provider = actions[0]["latent_renderer_provider"]
        self.assertEqual(provider["implementation"], generate.LAZY_LATENT_STRUCTURE_PROVIDER_ID)
        self.assertEqual(provider["provider_id"], "action-specific")
        self.assertEqual(provider["provider_provenance_id"], "action-specific")
        self.assertEqual(provider["requested_bases"], ["semantic"])

    def test_native_provider_nested_alias_overrides_shared_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "native_nested_alias.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "required_provider": {
                            "provenance_id": "shared-default",
                            "scheduler_mapping": "euler_clean_endpoint",
                            "basis_normalization": "match_rms",
                        },
                        "actions": [
                            {
                                "id": "spectral",
                                "type": "latent_renderer_fixed",
                                "provider": {
                                    "provider_provenance_id": "nested-specific",
                                    "requested_bases": ["spectral_low"],
                                    "required_hook_names": [],
                                },
                                "coefficients": [0.0, 0.01, 0, 0, 0, 0],
                            }
                        ],
                    }
                )
            )
            actions, _ = generate.load_actions(str(path), 3)
        provider = actions[0]["latent_renderer_provider"]
        self.assertEqual(provider["provider_id"], "nested-specific")
        self.assertEqual(provider["provider_provenance_id"], "nested-specific")

    def test_native_provider_rejects_null_or_conflicting_provenance(self):
        cases = (
            {
                "provider_provenance_id": None,
            },
            {
                "provider": {
                    "provider_id": "nested-a",
                    "provider_provenance_id": "nested-b",
                },
            },
        )
        for overrides in cases:
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as temp_dir:
                path = pathlib.Path(temp_dir) / "invalid_provider.yaml"
                action = {
                    "id": "invalid",
                    "type": "latent_renderer_fixed",
                    "coefficients": [0.01, 0, 0, 0, 0, 0],
                }
                action.update(overrides)
                path.write_text(
                    yaml.safe_dump(
                        {
                            "required_provider": {
                                "scheduler_mapping": "euler_clean_endpoint",
                                "basis_normalization": "match_rms",
                            },
                            "actions": [action],
                        }
                    )
                )
                with self.assertRaisesRegex(ValueError, "provenance"):
                    generate.load_actions(str(path), 3)

    def test_native_provider_rejects_structural_subset_before_runtime(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "structural_subset.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "required_provider": {
                            "implementation": "StructuralUNetBasisProvider",
                            "scheduler_mapping": "euler_clean_endpoint",
                            "basis_normalization": "match_rms",
                        },
                        "actions": [
                            {
                                "id": "subset",
                                "type": "latent_renderer_fixed",
                                "requested_bases": ["spectral_low"],
                                "required_hook_names": [],
                                "coefficients": [0.0, 0.01, 0, 0, 0, 0],
                            }
                        ],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "structural_unet_basis_v1"):
                generate.load_actions(str(path), 3)

    def test_renderer_config_rejects_unknown_mapping_and_normalization(self):
        for field, value, message in (
            ("scheduler_mapping", "dpm_post_step", "scheduler_mapping"),
            ("basis_normalization", "unit", "basis_normalization"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp_dir:
                path = pathlib.Path(temp_dir) / "invalid.yaml"
                path.write_text(
                    yaml.safe_dump(
                        {
                            "latent_renderer_provider": {field: value},
                            "actions": [
                                {
                                    "id": "invalid",
                                    "type": "latent_renderer_fixed",
                                    "coefficients": [0.01, 0, 0, 0, 0, 0],
                                }
                            ],
                        }
                    )
                )
                with self.assertRaisesRegex(ValueError, message):
                    generate.load_actions(str(path), 3)

    def test_native_renderer_sidecar_contract_is_json_safe_and_complete(self):
        ledger = self._native_renderer_ledger(steps=50)
        last_diagnostics = {
            key: copy.deepcopy(value)
            for key, value in ledger[-1].items()
            if key
            not in {
                "step_index",
                "scheduler_step_index",
                "timestep",
                "sigma_from",
                "sigma_to",
                "prediction_type",
                "provider_diagnostics",
            }
        }
        pipe = mock.Mock()
        pipe._last_latent_renderer_scheduler_mapping = "euler_clean_endpoint"
        pipe._last_latent_renderer_diagnostics = last_diagnostics
        pipe._last_latent_renderer_provider_diagnostics = copy.deepcopy(
            ledger[-1]["provider_diagnostics"]
        )
        pipe._last_latent_renderer_step_diagnostics = ledger
        action = {
            "type": "latent_renderer_fixed",
            "max_update_ratio": 0.05,
            "latent_renderer_provider": {
                "scheduler_mapping": "euler_clean_endpoint"
            },
        }
        fields = generate.latent_renderer_sidecar_fields(
            action, pipe, [1] * 50, 50
        )
        self.assertEqual(fields["latent_renderer_scheduler_mapping"], "euler_clean_endpoint")
        self.assertEqual(len(fields["latent_renderer_step_diagnostics"]), 50)
        json.dumps(fields, allow_nan=False)

        with self.assertRaisesRegex(RuntimeError, "exactly one U-Net call"):
            generate.latent_renderer_sidecar_fields(
                action, pipe, [1] * 49 + [2], 50
            )
        invalid_ledger = copy.deepcopy(ledger)
        invalid_ledger[0]["applied_update_ratio"] = [0.051]
        pipe._last_latent_renderer_step_diagnostics = invalid_ledger
        with self.assertRaisesRegex(RuntimeError, "applied trust bound"):
            generate.latent_renderer_sidecar_fields(
                action, pipe, [1] * 50, 50
            )
        pipe._last_latent_renderer_step_diagnostics = ledger
        stale_pipe = mock.Mock()
        stale_pipe._last_latent_renderer_diagnostics = None
        stale_pipe._last_latent_renderer_provider_diagnostics = None
        stale_pipe._last_latent_renderer_scheduler_mapping = None
        stale_pipe._last_latent_renderer_step_diagnostics = [{}]
        with self.assertRaisesRegex(RuntimeError, "stale renderer diagnostics"):
            generate.latent_renderer_sidecar_fields(
                {"type": "none"}, stale_pipe, [1, 1, 1], 3
            )

    def test_latent_renderer_split_seeds_are_enforced(self):
        path = ROOT / "eval-pipeline/configs/latent_renderer_fixed_lr1.yaml"
        generate.validate_split_seed_role(path, "train_search", [7, 19, 73])
        with self.assertRaisesRegex(ValueError, "pass --split_role"):
            generate.validate_split_seed_role(path, None, [7, 19, 73])
        with self.assertRaisesRegex(ValueError, "do not match"):
            generate.validate_split_seed_role(path, "train_search", [0, 42, 123])
        with self.assertRaisesRegex(ValueError, "unknown --split_role"):
            generate.validate_split_seed_role(path, "unknown", [7, 19, 73])

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

    @staticmethod
    def _complete_comparison_frame():
        return pd.DataFrame(
            [
                {
                    "id": f"p{prompt}_s{seed}_a{action}",
                    "prompt_index": prompt,
                    "seed": seed,
                    "action_id": action,
                    "device": "cuda:0",
                    "quality": float(prompt + seed + (action == "candidate")),
                }
                for prompt in (0, 1)
                for seed in (7, 9)
                for action in ("baseline", "candidate")
            ]
        )

    def test_compare_requires_complete_registered_grid(self):
        result = compare_actions.compare(
            self._complete_comparison_frame(),
            "baseline",
            ["quality"],
            n_boot=20,
            n_random=20,
            expected_prompt_indices=[0, 1],
            expected_seeds=[7, 9],
            expected_actions=["baseline", "candidate"],
        )
        self.assertEqual(result.loc[0, "n_prompts"], 2)
        self.assertEqual(result.loc[0, "n_seeds"], 2)
        self.assertEqual(result.loc[0, "n_pairs"], 4)
        self.assertEqual(result.loc[0, "mean_delta"], 1.0)

    def test_compare_rejects_duplicate_missing_extra_and_nonfinite_cells(self):
        complete = self._complete_comparison_frame()
        cases = {
            "duplicate": pd.concat([complete, complete.iloc[[0]]], ignore_index=True),
            "missing": complete[complete["prompt_index"] != 1],
            "extra": pd.concat(
                [
                    complete,
                    pd.DataFrame(
                        [
                            {
                                "id": "extra",
                                "prompt_index": 2,
                                "seed": 7,
                                "action_id": "baseline",
                                "device": "cuda:0",
                                "quality": 0.0,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            ),
            "nonfinite": complete.assign(
                quality=lambda frame: frame["quality"].mask(frame.index == 0, np.nan)
            ),
        }
        for label, frame in cases.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                compare_actions.compare(
                    frame,
                    "baseline",
                    ["quality"],
                    n_boot=20,
                    n_random=20,
                    expected_prompt_indices=[0, 1],
                    expected_seeds=[7, 9],
                    expected_actions=["baseline", "candidate"],
                )

    def test_score_ids_must_exactly_match_manifest_before_merge(self):
        manifest = pd.DataFrame(
            [{"id": "a", "prompt_index": 0}, {"id": "b", "prompt_index": 1}]
        )
        complete = pd.DataFrame([{"id": "a", "score": 1.0}, {"id": "b", "score": 2.0}])
        merged = compare_actions.merge_manifest_scores(manifest, complete)
        self.assertEqual(list(merged["id"]), ["a", "b"])

        invalid = {
            "missing": complete.iloc[[0]],
            "extra": pd.concat(
                [complete, pd.DataFrame([{"id": "c", "score": 3.0}])],
                ignore_index=True,
            ),
            "duplicate": pd.concat([complete, complete.iloc[[0]]], ignore_index=True),
        }
        for label, scores in invalid.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                compare_actions.merge_manifest_scores(manifest, scores)

    def test_expected_grid_is_loaded_from_run_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            prompts = root / "prompts.csv"
            prompts.write_text("index,TEXT\n0,first\n1,second\n")
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "prompts_csv": str(prompts),
                        "seeds": [7, 9],
                        "actions": [{"id": "baseline"}, {"id": "candidate"}],
                    }
                )
            )
            grid = compare_actions.load_expected_grid(str(root))
        self.assertEqual(grid, ([0, 1], [7, 9], ["baseline", "candidate"]))

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
