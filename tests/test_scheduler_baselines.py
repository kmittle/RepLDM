import importlib.util
import inspect
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

import torch
import yaml
from diffusers import EulerDiscreteScheduler


ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_PIPELINE = ROOT / "eval-pipeline"
sys.path.insert(0, str(EVAL_PIPELINE))
spec = importlib.util.spec_from_file_location(
    "generate_scheduler_test", EVAL_PIPELINE / "generate.py"
)
generate = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(generate)


class SchedulerBaselineTest(unittest.TestCase):
    def make_base(self):
        return EulerDiscreteScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            prediction_type="epsilon",
            timestep_spacing="linspace",
            steps_offset=1,
        )

    def test_allowlist_normalizes_registered_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "actions.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "actions": [
                            {
                                "id": "dpm",
                                "type": "scheduler_baseline",
                                "scheduler_class": "DPMSolverMultistepScheduler",
                            },
                            {
                                "id": "uni",
                                "type": "scheduler_baseline",
                                "scheduler_class": "UniPCMultistepScheduler",
                            },
                            {"id": "ancestral", "type": "scheduler_baseline"},
                        ]
                    }
                )
            )
            actions, _ = generate.load_actions(str(path), 50)
        by_id = {action["id"]: action for action in actions}
        self.assertEqual(by_id["dpm"]["scheduler_kwargs"]["solver_order"], 2)
        self.assertEqual(
            by_id["dpm"]["scheduler_kwargs"]["algorithm_type"], "dpmsolver++"
        )
        self.assertEqual(by_id["uni"]["scheduler_kwargs"]["solver_type"], "bh2")
        self.assertEqual(by_id["ancestral"]["scheduler_kwargs"], {})
        self.assertFalse(by_id["dpm"]["selection_eligible"])

    def test_frozen_reference_config_has_four_actions_and_registered_seeds(self):
        config_path = ROOT / "eval-pipeline/configs/scheduler_baselines_v1.yaml"
        actions, _ = generate.load_actions(str(config_path), 50)
        self.assertEqual(
            [action["id"] for action in actions],
            [
                "no_correction",
                "euler_ancestral_reference",
                "dpmpp_2m_reference",
                "unipc_2_reference",
            ],
        )
        generate.validate_split_seed_role(str(config_path), "development", [0, 42])
        with self.assertRaisesRegex(ValueError, "do not match"):
            generate.validate_split_seed_role(str(config_path), "development", [0, 1])

    def test_unreviewed_reference_config_cannot_start_generation(self):
        config_path = ROOT / "eval-pipeline/configs/scheduler_baselines_v1.yaml"
        with self.assertRaisesRegex(ValueError, "not authorized"):
            generate.validate_scheduler_baseline_authorization(str(config_path))
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "authorized.yaml"
            config = yaml.safe_load(config_path.read_text())
            config["status"] = "authorized"
            config["authorization"] = {
                "reviewer": "reviewer-id",
                "reviewed_commit": "deadbeef",
            }
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            generate.validate_scheduler_baseline_authorization(str(path))

    def test_frozen_design_binds_sampling_provenance_and_action_kwargs(self):
        cases = (
            (
                "scheduler_baselines_v1.yaml",
                "trajectory_correction_heldout_v1.csv",
                "development",
                [0, 42],
            ),
            (
                "scheduler_baselines_validation_v1.yaml",
                "trajectory_correction_validation_v1.csv",
                "validation_confirmation",
                [11, 29, 101],
            ),
        )
        for config_name, prompt_name, role, seeds in cases:
            config_path = ROOT / "eval-pipeline/configs" / config_name
            prompt_path = ROOT / "eval-pipeline/prompts" / prompt_name
            actions, _ = generate.load_actions(str(config_path), 50)
            generate.validate_scheduler_baseline_design(
                str(config_path),
                prompts_path=str(prompt_path),
                actions=actions,
                seeds=seeds,
                model_name="stabilityai/stable-diffusion-xl-base-1.0",
                resolution=1024,
                num_inference_steps=50,
                guidance_scale=7.5,
                stage2_enabled=False,
            )
            generate.validate_split_seed_role(str(config_path), role, seeds)

    def test_allowlist_rejects_unknown_or_stochastic_kwargs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "actions.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "actions": [
                            {
                                "id": "bad",
                                "type": "scheduler_baseline",
                                "scheduler_class": "DPMSolverMultistepScheduler",
                                "scheduler_kwargs": {
                                    "algorithm_type": "sde-dpmsolver++"
                                },
                            }
                        ]
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "deterministic"):
                generate.load_actions(str(path), 50)
            path.write_text(
                yaml.safe_dump(
                    {
                        "actions": [
                            {
                                "id": "bad_type",
                                "type": "scheduler_baseline",
                                "scheduler_class": "DPMSolverMultistepScheduler",
                                "scheduler_kwargs": {"solver_order": "2"},
                            }
                        ]
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "integer 2"):
                generate.load_actions(str(path), 50)

    def test_all_registered_schedulers_execute_50_cpu_steps(self):
        base = self.make_base()
        for name in generate.SCHEDULER_BASELINE_CLASSES:
            action = {
                "type": "scheduler_baseline",
                "scheduler_class": name,
                "scheduler_kwargs": dict(
                    generate.SCHEDULER_BASELINE_DEFAULT_KWARGS[name]
                ),
            }
            scheduler = generate.scheduler_baseline_runtime(action, base)
            scheduler.set_timesteps(50, device="cpu")
            sample = torch.randn(
                (1, 4, 8, 8), generator=torch.Generator().manual_seed(3)
            ) * scheduler.init_noise_sigma
            calls = 0
            for timestep in scheduler.timesteps:
                model_input = scheduler.scale_model_input(sample, timestep)
                kwargs = {}
                if "generator" in inspect.signature(scheduler.step).parameters:
                    kwargs["generator"] = torch.Generator().manual_seed(3)
                sample = scheduler.step(
                    torch.zeros_like(model_input),
                    timestep,
                    sample,
                    return_dict=False,
                    **kwargs,
                )[0]
                calls += 1
            self.assertEqual(calls, 50, name)
            self.assertEqual(len(scheduler.timesteps), 50, name)
            self.assertTrue(torch.isfinite(sample).all(), name)

    def test_registered_solver_order_is_read_from_scheduler_config(self):
        base = self.make_base()
        for name in ("DPMSolverMultistepScheduler", "UniPCMultistepScheduler"):
            action = {
                "type": "scheduler_baseline",
                "scheduler_class": name,
                "scheduler_kwargs": dict(
                    generate.SCHEDULER_BASELINE_DEFAULT_KWARGS[name]
                ),
            }
            scheduler = generate.scheduler_baseline_runtime(action, base)
            self.assertEqual(scheduler.config.get("solver_order"), 2, name)

    def test_registered_euler_baseline_records_native_scheduler_provenance(self):
        record = generate.scheduler_provenance_record(
            {"id": "no_correction", "type": "none"},
            include=True,
            base_config_sha256_v2="base-hash",
            active_config_sha256_v2="active-hash",
            order=1,
            solver_order=None,
            init_noise_sigma=14.5,
        )
        self.assertEqual(record["scheduler_kwargs"], {})
        self.assertEqual(record["scheduler_order"], 1)
        self.assertIsNone(record["scheduler_solver_order"])
        self.assertEqual(record["scheduler_init_noise_sigma"], 14.5)
        self.assertEqual(
            record["active_scheduler_config_sha256_v2"], "active-hash"
        )
        self.assertEqual(
            generate.scheduler_provenance_record(
                {},
                include=False,
                base_config_sha256_v2="base-hash",
                active_config_sha256_v2="active-hash",
                order=1,
                solver_order=None,
                init_noise_sigma=1.0,
            ),
            {},
        )

    def test_v2_hash_is_stable_when_default_metadata_order_changes(self):
        class FakeScheduler:
            def __init__(self, values):
                self.config = {
                    "num_train_timesteps": 1000,
                    "_use_default_values": values,
                }

        first = generate.scheduler_config_sha256_v2(FakeScheduler(["b", "a"]))
        second = generate.scheduler_config_sha256_v2(FakeScheduler(["a", "b"]))
        self.assertEqual(first, second)
        legacy_first = generate.scheduler_config_sha256(FakeScheduler(["b", "a"]))
        legacy_second = generate.scheduler_config_sha256(FakeScheduler(["a", "b"]))
        self.assertNotEqual(legacy_first, legacy_second)

    def test_v2_hash_matches_across_python_hash_seeds(self):
        code = (
            "import sys; sys.path.insert(0, 'eval-pipeline'); "
            "import generate; from diffusers import EulerDiscreteScheduler; "
            "s=EulerDiscreteScheduler(num_train_timesteps=1000); "
            "print(generate.scheduler_config_sha256_v2(s))"
        )
        values = []
        for hash_seed in ("0", "1"):
            env = {**os.environ, "PYTHONHASHSEED": hash_seed}
            values.append(
                subprocess.check_output(
                    [sys.executable, "-c", code],
                    cwd=ROOT,
                    env=env,
                    text=True,
                ).strip()
            )
        self.assertEqual(values[0], values[1])


if __name__ == "__main__":
    unittest.main()
