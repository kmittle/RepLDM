import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import multiprocessing
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd
import yaml
from diffusers import EulerDiscreteScheduler
from PIL import Image


ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_PIPELINE = ROOT / "eval-pipeline"
if str(EVAL_PIPELINE) not in sys.path:
    sys.path.insert(0, str(EVAL_PIPELINE))
spec = importlib.util.spec_from_file_location(
    "generate_cfg_baselines_test", EVAL_PIPELINE / "generate.py"
)
generate = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(generate)


def attempt_generation_lock(path, result_queue):
    try:
        with generate.generation_output_lock(path):
            result_queue.put("acquired")
    except Exception as exc:
        result_queue.put(f"{type(exc).__name__}: {exc}")


class CFGBaselineTest(unittest.TestCase):
    config_path = ROOT / "eval-pipeline/configs/cfg_baselines_v1.yaml"
    prompts_path = ROOT / "eval-pipeline/prompts/s5_development.csv"

    @staticmethod
    def write_config(root: str, config: dict) -> pathlib.Path:
        path = pathlib.Path(root) / "actions.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        return path

    def load_frozen(self):
        config = yaml.safe_load(self.config_path.read_text())
        actions, _ = generate.load_actions(str(self.config_path), 50)
        return config, actions

    def make_authorized_config(self, root):
        root = pathlib.Path(root)
        template = root / generate.CFG_BASELINE_AUTH_SOURCE_TEMPLATE
        template.parent.mkdir(parents=True)
        template.write_bytes(self.config_path.read_bytes())
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(
            ["git", "-C", str(root), "config", "user.email", "test@example.com"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(root), "config", "user.name", "Test Reviewer"],
            check=True,
        )
        subprocess.run(["git", "-C", str(root), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(root), "commit", "-q", "-m", "freeze template"],
            check=True,
        )
        reviewed_commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        config = yaml.safe_load(template.read_text())
        config["status"] = "authorized"
        config["authorization"] = {
            "reviewer": "reviewer-id",
            "reviewed_commit": reviewed_commit,
            "source_template": generate.CFG_BASELINE_AUTH_SOURCE_TEMPLATE,
            "source_template_sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
            "scope": generate.CFG_BASELINE_AUTH_SCOPE,
        }
        return self.write_config(str(root), config), config

    @staticmethod
    def frozen_euler_scheduler():
        return EulerDiscreteScheduler.from_config(
            {
                "_class_name": "EulerDiscreteScheduler",
                "_diffusers_version": "0.19.0.dev0",
                "_use_default_values": [
                    "final_sigmas_type",
                    "rescale_betas_zero_snr",
                    "sigma_max",
                    "sigma_min",
                    "timestep_type",
                    "use_beta_sigmas",
                    "use_exponential_sigmas",
                ],
                "beta_end": 0.012,
                "beta_schedule": "scaled_linear",
                "beta_start": 0.00085,
                "clip_sample": False,
                "final_sigmas_type": "zero",
                "interpolation_type": "linear",
                "num_train_timesteps": 1000,
                "prediction_type": "epsilon",
                "rescale_betas_zero_snr": False,
                "sample_max_value": 1.0,
                "set_alpha_to_one": False,
                "sigma_max": None,
                "sigma_min": None,
                "skip_prk_steps": True,
                "steps_offset": 1,
                "timestep_spacing": "leading",
                "timestep_type": "discrete",
                "trained_betas": None,
                "use_beta_sigmas": False,
                "use_exponential_sigmas": False,
                "use_karras_sigmas": False,
            }
        )

    def validate_frozen_design(self, path, actions, seeds=(0, 42, 123)):
        generate.validate_cfg_baseline_design(
            str(path),
            prompts_path=str(self.prompts_path),
            actions=actions,
            seeds=list(seeds),
            model_name=generate.DEFAULT_MODEL,
            resolution=1024,
            num_inference_steps=50,
            guidance_scale=7.5,
            negative_prompt=generate.DEFAULT_NEG,
            power_calibrate=0,
            stage2_enabled=False,
        )

    def test_frozen_actions_normalize_to_registered_ids_scales_and_hashes(self):
        config, actions = self.load_frozen()
        self.assertEqual(
            [action["id"] for action in actions],
            list(generate.CFG_BASELINE_ACTION_SCALES),
        )
        self.assertEqual(
            [action["cfg_scale"] for action in actions],
            list(generate.CFG_BASELINE_ACTION_SCALES.values()),
        )
        for action in actions:
            self.assertEqual(action["type"], "none")
            self.assertEqual(action["scale"], 0.0)
            self.assertEqual(
                generate.action_sha256(action),
                config["design"]["action_sha256"][action["id"]],
            )

    def test_cfg_action_schema_rejects_implicit_or_intervening_actions(self):
        cases = (
            (
                {"id": "cfg_2p5", "type": "none"},
                "explicit cfg_scale",
            ),
            (
                {"id": "cfg_2p5", "type": "scalar", "cfg_scale": 2.5},
                "must use type 'none'",
            ),
            (
                {
                    "id": "cfg_2p5",
                    "type": "none",
                    "cfg_scale": 2.5,
                    "delay_steps": 1,
                },
                "unsupported fields",
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for action, message in cases:
                with self.subTest(action=action):
                    path = self.write_config(
                        tmp,
                        {"schema": generate.CFG_BASELINE_SCHEMA, "actions": [action]},
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        generate.load_actions(str(path), 50)

    def test_authorization_is_explicit_and_review_bound(self):
        config, _ = self.load_frozen()
        with self.assertRaisesRegex(ValueError, "not authorized"):
            generate.validate_cfg_baseline_authorization(str(self.config_path))
        with tempfile.TemporaryDirectory() as tmp:
            config["status"] = "authorized"
            config["authorization"] = {
                "reviewer": "reviewer-id",
                "reviewed_commit": "deadbeef",
                "source_template": generate.CFG_BASELINE_AUTH_SOURCE_TEMPLATE,
                "source_template_sha256": "0" * 64,
                "scope": generate.CFG_BASELINE_AUTH_SCOPE,
            }
            path = self.write_config(tmp, config)
            with self.assertRaisesRegex(ValueError, "full 40-hex"):
                generate.validate_cfg_baseline_authorization(str(path))
            config["authorization"]["reviewer"] = ""
            path = self.write_config(tmp, config)
            with self.assertRaisesRegex(ValueError, "authorization.reviewer"):
                generate.validate_cfg_baseline_authorization(str(path))

    def test_authorization_accepts_an_unchanged_ancestor_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self.make_authorized_config(tmp)
            generate.validate_cfg_baseline_authorization(
                str(path), repository_root=tmp
            )

    def test_authorization_rejects_hash_body_and_ancestry_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, config = self.make_authorized_config(tmp)
            changed = copy.deepcopy(config)
            changed["authorization"]["source_template_sha256"] = "0" * 64
            changed_path = self.write_config(tmp, changed)
            with self.assertRaisesRegex(ValueError, "differs from reviewed Git bytes"):
                generate.validate_cfg_baseline_authorization(
                    str(changed_path), repository_root=tmp
                )

            changed = copy.deepcopy(config)
            changed["selection"]["seed"] += 1
            changed_path = self.write_config(tmp, changed)
            with self.assertRaisesRegex(ValueError, "differs from the reviewed"):
                generate.validate_cfg_baseline_authorization(
                    str(changed_path), repository_root=tmp
                )

            tree = subprocess.check_output(
                ["git", "-C", tmp, "rev-parse", "HEAD^{tree}"], text=True
            ).strip()
            unrelated = subprocess.check_output(
                ["git", "-C", tmp, "commit-tree", tree],
                input="unrelated history\n",
                text=True,
            ).strip()
            changed = copy.deepcopy(config)
            changed["authorization"]["reviewed_commit"] = unrelated
            changed_path = self.write_config(tmp, changed)
            with self.assertRaisesRegex(ValueError, "not an ancestor"):
                generate.validate_cfg_baseline_authorization(
                    str(changed_path), repository_root=tmp
                )

    def test_authorization_rejects_unregistered_claim_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, config = self.make_authorized_config(tmp)
            config["authorization"]["test_final_approved"] = True
            path = self.write_config(tmp, config)
            with self.assertRaisesRegex(ValueError, "authorization fields differ"):
                generate.validate_cfg_baseline_authorization(
                    str(path), repository_root=tmp
                )

    def test_frozen_design_binds_prompts_seeds_counts_sampling_and_order(self):
        _, actions = self.load_frozen()
        self.validate_frozen_design(self.config_path, actions)
        generate.validate_split_seed_role(
            str(self.config_path), "development", [0, 42, 123]
        )
        with self.assertRaisesRegex(ValueError, "do not match"):
            generate.validate_split_seed_role(
                str(self.config_path), "development", [0, 42, 124]
            )

    def test_design_drift_is_rejected_before_generation(self):
        config, actions = self.load_frozen()
        mutations = (
            (
                lambda value: value["source_manifest"].update(
                    {"prompts_sha256": "0" * 64}
                ),
                "prompt SHA-256",
            ),
            (
                lambda value: value["source_manifest"].update(
                    {"prompts": str(self.prompts_path)}
                ),
                "prompt registration",
            ),
            (
                lambda value: value["design"]["action_sha256"].update(
                    {"cfg_2p5": "0" * 64}
                ),
                "action hash",
            ),
            (
                lambda value: value["design"].update(
                    {"expected_task_count": 179}
                ),
                "task_count differs",
            ),
            (
                lambda value: value["sampling"].update({"guidance_rescale": 0.1}),
                "sampling.guidance_rescale differs",
            ),
            (
                lambda value: value["execution_order"].update(
                    {"sort": "yaml_order"}
                ),
                "execution_order differs",
            ),
            (
                lambda value: value["design"].update({"baseline_action_id": "cfg_5p0"}),
                "baseline_action_id",
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for mutate, message in mutations:
                with self.subTest(message=message):
                    changed = copy.deepcopy(config)
                    mutate(changed)
                    path = self.write_config(tmp, changed)
                    with self.assertRaisesRegex(ValueError, message):
                        self.validate_frozen_design(path, actions)

    def test_v1_seeds_cannot_be_redefined_by_a_rewritten_yaml(self):
        config, actions = self.load_frozen()
        config["split_seeds"] = {"development": [0, 42, 124]}
        config["design"]["expected_seed_count"] = 3
        config["design"]["expected_block_count"] = 36
        config["design"]["expected_task_count"] = 180
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_config(tmp, config)
            with self.assertRaisesRegex(ValueError, "split_seeds must be exactly"):
                self.validate_frozen_design(path, actions, seeds=(0, 42, 124))

    def test_each_prompt_seed_block_gets_deterministic_action_ranks(self):
        _, actions = self.load_frozen()
        prompts = pd.DataFrame(
            [
                {"index": 0, "TEXT": "first"},
                {"index": 1, "TEXT": "second"},
            ]
        )
        tasks = generate.build_tasks(prompts, [0, 42], actions)
        with tempfile.TemporaryDirectory() as tmp:
            assignments = generate.assign_tasks_to_devices(
                tasks, ["cuda:0", "cuda:1"], tmp, run_contract_sha256="contract"
            )
        by_pair = {}
        for device, assigned in assignments.items():
            for task in assigned:
                by_pair.setdefault((task["prompt_index"], task["seed"]), []).append(
                    (device, task)
                )
        self.assertEqual(len(by_pair), 4)
        for (prompt_index, seed), rows in by_pair.items():
            self.assertEqual(len({device for device, _ in rows}), 1)
            ordered_ids = sorted(
                generate.CFG_BASELINE_ACTION_SCALES,
                key=lambda action_id: hashlib.sha256(
                    (
                        f"action-order-v1:{prompt_index}:{seed}:{action_id}"
                    ).encode("utf-8")
                ).digest(),
            )
            observed = {
                task["action_id"]: task["execution_rank"] for _, task in rows
            }
            self.assertEqual(
                observed,
                {action_id: rank for rank, action_id in enumerate(ordered_ids)},
            )

    def test_native_euler_clones_are_fresh_and_schedule_is_fully_recorded(self):
        base = EulerDiscreteScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            prediction_type="epsilon",
            timestep_spacing="leading",
            steps_offset=1,
        )
        first = generate.native_scheduler_runtime(base)
        second = generate.native_scheduler_runtime(base)
        self.assertIsNot(first, base)
        self.assertIsNot(second, base)
        self.assertIsNot(first, second)
        self.assertEqual(
            generate.scheduler_config_payload(first),
            generate.scheduler_config_payload(second),
        )
        ledger = generate.prepare_scheduler_schedule_provenance(first, 50, "cpu")
        self.assertEqual(len(ledger["scheduler_timesteps"]), 50)
        self.assertEqual(len(ledger["scheduler_sigmas"]), 51)
        self.assertEqual(
            ledger["scheduler_schedule_sha256"],
            generate.json_sha256(
                {
                    "timesteps": ledger["scheduler_timesteps"],
                    "sigmas": ledger["scheduler_sigmas"],
                }
            ),
        )
        self.assertEqual(
            generate.scheduler_config_sha256_v2(first),
            generate.scheduler_config_sha256_v2(second),
        )
        self.assertIn("prediction_type", generate.scheduler_config_payload(first))

    def test_frozen_euler_runtime_rejects_config_schedule_and_sigma_drift(self):
        scheduler = self.frozen_euler_scheduler()
        config_hash = generate.scheduler_config_sha256_v2(scheduler)
        self.assertEqual(
            config_hash,
            generate.CFG_BASELINE_SCHEDULER_RUNTIME["config_sha256_v2"],
        )
        ledger = generate.prepare_scheduler_schedule_provenance(scheduler, 50, "cpu")
        generate.validate_cfg_scheduler_runtime(
            copy.deepcopy(generate.CFG_BASELINE_SCHEDULER_RUNTIME),
            config_sha256_v2=config_hash,
            schedule_provenance=ledger,
        )
        with self.assertRaisesRegex(RuntimeError, "Euler config"):
            generate.validate_cfg_scheduler_runtime(
                generate.CFG_BASELINE_SCHEDULER_RUNTIME,
                config_sha256_v2="0" * 64,
            )
        changed = copy.deepcopy(ledger)
        changed["scheduler_effective_init_noise_sigma"] += 1.0
        with self.assertRaisesRegex(RuntimeError, "Euler schedule"):
            generate.validate_cfg_scheduler_runtime(
                generate.CFG_BASELINE_SCHEDULER_RUNTIME,
                config_sha256_v2=config_hash,
                schedule_provenance=changed,
            )

    def test_worker_uses_revision_zero_rescale_and_frozen_scheduler_ledger(self):
        class FakePipeline:
            def __init__(self, scheduler):
                self.scheduler = scheduler
                self.unet = mock.Mock()
                self.call_kwargs = None

            def to(self, _device):
                return self

            def set_progress_bar_config(self, **_kwargs):
                return None

            def __call__(self, _prompt, **kwargs):
                self.call_kwargs = kwargs
                self._last_unet_calls_per_step = [1] * 50
                return [Image.new("RGB", (8, 8), "blue")]

        config, actions = self.load_frozen()
        task = generate.build_tasks(
            pd.DataFrame([{"index": 0, "TEXT": "test prompt"}]),
            [0],
            [actions[0]],
        )[0]
        task["execution_rank"] = 0
        with tempfile.TemporaryDirectory() as tmp:
            (pathlib.Path(tmp) / "images").mkdir()
            task_queue = generate.queue.Queue()
            task_queue.put(task)
            error_queue = generate.queue.Queue()
            fake_pipe = FakePipeline(self.frozen_euler_scheduler())
            cfg = {
                "model_name": generate.DEFAULT_MODEL,
                "model_revision": generate.CFG_BASELINE_MODEL_REVISION,
                "cache_dir": "unused",
                "registered_sampling": copy.deepcopy(config["sampling"]),
                "scheduler_runtime": copy.deepcopy(config["scheduler_runtime"]),
                "guidance_rescale": 0.0,
                "trajectory_registered": False,
                "scheduler_baseline_registered": False,
                "cfg_baseline_registered": True,
                "out_dir": tmp,
                "git_commit": "a" * 40,
                "run_contract_sha256": "b" * 64,
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
                "negative_prompt": generate.DEFAULT_NEG,
                "resolution": 8,
                "power_calibrate": 0,
                "frequency_band_cutoffs": [0.08, 0.25],
                "stage_name": "stage1_1024",
                "stage2_enabled": False,
                "models_to_cpu": False,
                "multi_encoder": False,
                "multi_decoder": False,
                "num_resample_timesteps": 50,
                "init_rates": [0.8],
                "stage2_noise_source": "not_applicable",
                "runtime_provenance": {},
            }
            original_prepare = generate.prepare_scheduler_schedule_provenance
            original_generator = generate.torch.Generator
            with mock.patch.object(
                generate.RepLDMSDXLPipeline,
                "from_pretrained",
                return_value=fake_pipe,
            ) as from_pretrained, mock.patch.object(
                generate.torch.cuda, "set_device"
            ), mock.patch.object(
                generate.torch.cuda, "is_available", return_value=False
            ), mock.patch.object(
                generate.torch,
                "Generator",
                side_effect=lambda _device: original_generator(),
            ), mock.patch.object(
                generate,
                "prepare_scheduler_schedule_provenance",
                side_effect=lambda scheduler, steps, _device: original_prepare(
                    scheduler, steps, "cpu"
                ),
            ):
                generate.worker_process(cfg, "cuda:0", task_queue, error_queue)

            self.assertTrue(error_queue.empty())
            self.assertEqual(
                from_pretrained.call_args.kwargs["revision"],
                generate.CFG_BASELINE_MODEL_REVISION,
            )
            self.assertEqual(fake_pipe.call_kwargs["guidance_scale"], 2.5)
            self.assertEqual(fake_pipe.call_kwargs["guidance_rescale"], 0.0)
            sidecar = yaml.safe_load(
                (pathlib.Path(tmp) / "images" / f"{task['id']}.json").read_text()
            )
            self.assertEqual(
                sidecar["model_revision"], generate.CFG_BASELINE_MODEL_REVISION
            )
            self.assertEqual(sidecar["guidance_rescale"], 0.0)
            self.assertEqual(
                sidecar["scheduler_runtime"],
                generate.CFG_BASELINE_SCHEDULER_RUNTIME,
            )

    def test_registered_nfe_requires_exactly_one_unet_call_each_step(self):
        generate.validate_one_unet_call_per_step(
            [1] * 50, 50, run_label="CFG baselines"
        )
        for calls in ([1] * 49, [1] * 49 + [2], [1] * 49 + [True]):
            with self.subTest(calls=calls[-1:]):
                with self.assertRaisesRegex(RuntimeError, "exactly one U-Net"):
                    generate.validate_one_unet_call_per_step(
                        calls, 50, run_label="CFG baselines"
                    )

    def test_worker_does_not_skip_a_stale_registered_sidecar(self):
        task = {"id": "p0_seed0_acfg_2p5", "execution_rank": 1}
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = pathlib.Path(tmp)
            (image_dir / f"{task['id']}.png").write_bytes(b"stale")
            (image_dir / f"{task['id']}.json").write_text(
                '{"execution_rank": 2}'
            )
            cfg = {
                "cfg_baseline_registered": True,
                "scheduler_baseline_registered": False,
                "trajectory_registered": False,
                "run_contract_sha256": "contract",
            }
            contract = generate.worker_resume_contract_sha256(cfg)
            self.assertEqual(contract, "contract")
            with mock.patch.object(generate, "validate_sidecar"):
                self.assertFalse(
                    generate.task_is_complete(
                        task,
                        str(image_dir),
                        run_contract_sha256=contract,
                    )
                )
                self.assertTrue(
                    generate.task_is_complete(
                        task,
                        str(image_dir),
                        run_contract_sha256=None,
                    )
                )

    def test_truncated_registered_sidecar_is_repaired_but_legacy_is_rejected(self):
        task = {"id": "p0_seed0_acfg_2p5", "prompt_index": 0, "seed": 0}
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = pathlib.Path(tmp)
            (image_dir / f"{task['id']}.png").write_bytes(b"partial")
            (image_dir / f"{task['id']}.json").write_text('{"device":')
            self.assertFalse(
                generate.task_is_complete(
                    task, tmp, run_contract_sha256="contract"
                )
            )
            self.assertIsNone(
                generate.recorded_group_device(
                    [task], tmp, run_contract_sha256="contract"
                )
            )
            with self.assertRaisesRegex(ValueError, "cannot read existing sidecar"):
                generate.recorded_group_device([task], tmp)

    def test_generation_lock_rejects_contention_and_reentry(self):
        context = multiprocessing.get_context("fork")
        with tempfile.TemporaryDirectory() as tmp:
            with generate.generation_output_lock(tmp):
                with self.assertRaisesRegex(RuntimeError, "already locked"):
                    with generate.generation_output_lock(tmp):
                        pass
                result_queue = context.Queue()
                process = context.Process(
                    target=attempt_generation_lock, args=(tmp, result_queue)
                )
                process.start()
                process.join(10)
                self.assertFalse(process.is_alive())
                self.assertEqual(process.exitcode, 0)
                self.assertRegex(result_queue.get(timeout=1), "already locked")
            with generate.generation_output_lock(tmp):
                pass

    def test_generation_lock_releases_after_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "boom"):
                with generate.generation_output_lock(tmp):
                    raise RuntimeError("boom")
            with generate.generation_output_lock(tmp):
                pass

    def test_actions_are_loaded_and_validated_after_output_lock_acquisition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            prompts = root / "prompts.csv"
            prompts.write_text("index,TEXT\n0,test prompt\n")
            actions = root / "actions.yaml"
            actions.write_text(
                yaml.safe_dump({"actions": [{"id": "plain", "type": "none"}]})
            )
            replacement = {
                "schema": generate.CFG_BASELINE_SCHEMA,
                "status": "cpu_audited_not_authorized",
                "actions": [
                    {"id": "cfg_2p5", "type": "none", "cfg_scale": 2.5}
                ],
            }

            @contextmanager
            def replace_on_lock(_out_dir):
                actions.write_text(yaml.safe_dump(replacement))
                yield str(root / ".generate.lock")

            argv = [
                "generate.py",
                "--prompts",
                str(prompts),
                "--out_dir",
                str(root / "run"),
                "--actions",
                str(actions),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                generate, "generation_output_lock", side_effect=replace_on_lock
            ), mock.patch.object(generate, "run_generation_locked") as run:
                with self.assertRaises(SystemExit):
                    generate.main()
            run.assert_not_called()

    def test_atomic_writers_publish_complete_files_and_clean_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            json_path = root / "record.json"
            generate.atomic_write_json(str(json_path), {"status": "complete"})
            self.assertEqual(yaml.safe_load(json_path.read_text()), {"status": "complete"})

            json_path.write_text("old")
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                with generate.atomic_text_writer(str(json_path)) as handle:
                    handle.write("new")
                    raise RuntimeError("interrupted")
            self.assertEqual(json_path.read_text(), "old")

            png_path = root / "image.png"
            generate.atomic_save_png(Image.new("RGB", (2, 2), "red"), str(png_path))
            with Image.open(png_path) as image:
                self.assertEqual(image.size, (2, 2))
            self.assertEqual(list(root.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
