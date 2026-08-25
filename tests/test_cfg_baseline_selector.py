import importlib.util
import pathlib
import sys
import tempfile
import unittest
import fcntl

import pandas as pd
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_PIPELINE = ROOT / "eval-pipeline"
if str(EVAL_PIPELINE) not in sys.path:
    sys.path.insert(0, str(EVAL_PIPELINE))
spec = importlib.util.spec_from_file_location(
    "cfg_baseline_selector_test", EVAL_PIPELINE / "select_cfg_baseline.py"
)
selector = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(selector)


class CFGBaselineSelectorTest(unittest.TestCase):
    def make_frame(self, overrides=None):
        overrides = overrides or {}
        rows = []
        base = {
            "topiq_nr": 0.70,
            "hpsv2": 0.30,
            "clip_cosine": 0.35,
            "clipped_fraction": 0.01,
            "mean_saturation": 0.30,
            "contrast_std": 50.0,
        }
        for prompt_index in range(12):
            for seed in (0, 42, 123):
                for action in selector.CFG_ACTION_IDS:
                    values = dict(base)
                    values.update(overrides.get(action, {}))
                    rows.append(
                        {
                            "prompt_index": prompt_index,
                            "seed": seed,
                            "action_id": action,
                            "device": f"cuda:{prompt_index % 2}",
                            **values,
                        }
                    )
        return pd.DataFrame(rows)

    def test_gate_and_practical_tie_break_are_frozen(self):
        frame = self.make_frame(
            {
                "cfg_2p5": {"topiq_nr": 0.72, "hpsv2": 0.28},
                "cfg_5p0": {"topiq_nr": 0.706},
                "cfg_10p0": {"topiq_nr": 0.710},
                "cfg_15p0": {"topiq_nr": 0.73, "contrast_std": 60.0},
            }
        )
        result = selector.select(
            frame,
            bootstrap=200,
            randomizations=10000,
        )
        self.assertEqual(result["selected_action"], "cfg_5p0")
        self.assertEqual(result["decision"], "selected_nondefault_scale")
        table = {row["action"]: row for row in result["rows"]}
        self.assertFalse(table["cfg_2p5"]["passes_gate"])
        self.assertTrue(table["cfg_5p0"]["passes_gate"])
        self.assertTrue(table["cfg_10p0"]["passes_gate"])
        self.assertFalse(table["cfg_15p0"]["passes_gate"])

    def test_no_passing_candidate_returns_registered_default(self):
        frame = self.make_frame(
            {
                action: {"topiq_nr": 0.704}
                for action in selector.CFG_ACTION_IDS
                if action != selector.BASELINE_ACTION_ID
            }
        )
        result = selector.select(frame, bootstrap=100, randomizations=1000)
        self.assertEqual(result["selected_action"], selector.BASELINE_ACTION_ID)
        self.assertEqual(result["selected_cfg_scale"], 7.5)
        self.assertEqual(result["decision"], "null_route")

    def test_duplicate_cell_is_rejected(self):
        frame = self.make_frame()
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            selector.select(frame, bootstrap=10, randomizations=10)

    def test_nonpositive_contrast_is_rejected(self):
        frame = self.make_frame()
        frame.loc[0, "contrast_std"] = 0.0
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            selector.select(frame, bootstrap=10, randomizations=10)

    def test_action_grid_drift_is_rejected(self):
        frame = self.make_frame()
        with self.assertRaisesRegex(ValueError, "frozen ordered action grid"):
            selector.select(
                frame,
                action_order=list(reversed(selector.CFG_ACTION_IDS)),
                bootstrap=10,
                randomizations=10,
            )

    def test_repository_config_matches_the_frozen_selector_rule(self):
        path = EVAL_PIPELINE / "configs" / "cfg_baselines_v1.yaml"
        config = yaml.safe_load(path.read_text())
        selector._require_frozen_rule(config)
        config["selection"]["minimum_mean_delta"] = 0.0
        with self.assertRaisesRegex(ValueError, "selection rule differs"):
            selector._require_frozen_rule(config)

    def test_selection_lock_rejects_a_concurrent_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            with selector._selection_lock(tmp):
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    with selector._selection_lock(tmp):
                        pass

    def test_selection_lock_rejects_generation_or_scoring_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = pathlib.Path(tmp) / ".generate.lock"
            with lock_path.open("a+") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "generation, scoring"):
                    with selector._selection_lock(tmp):
                        pass

    def test_interrupted_half_result_is_rebuilt_but_complete_result_is_one_shot(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = pathlib.Path(tmp) / "cfg_baseline_selection.json"
            csv_path = pathlib.Path(tmp) / "cfg_baseline_selection.csv"
            output_path.write_text("partial")
            selector._prepare_selection_outputs(str(output_path), str(csv_path))
            self.assertFalse(output_path.exists())
            self.assertFalse(csv_path.exists())

            output_path.write_text("complete")
            csv_path.write_text("complete")
            with self.assertRaisesRegex(ValueError, "intentionally one-shot"):
                selector._prepare_selection_outputs(str(output_path), str(csv_path))
            self.assertTrue(output_path.exists())
            self.assertTrue(csv_path.exists())

    def test_verified_snapshot_rejects_post_validation_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "scores.jsonl"
            original = b'{"id":"row"}\n'
            path.write_bytes(original)
            digest = selector._sha256_bytes(original)
            self.assertEqual(
                selector._read_verified_bytes(str(path), digest, "scores.jsonl"),
                original,
            )
            path.write_bytes(b'{"id":"changed"}\n')
            with self.assertRaisesRegex(RuntimeError, "changed after"):
                selector._read_verified_bytes(str(path), digest, "scores.jsonl")
            with self.assertRaisesRegex(RuntimeError, "changed during"):
                selector._require_unchanged_snapshot(
                    original, path.read_bytes(), "scores.jsonl"
                )

    def test_atomic_publication_replaces_destination_without_fixed_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = pathlib.Path(tmp) / "selection.json"
            target.write_bytes(b"old")
            selector._atomic_write_bytes(str(target), b"new")
            self.assertEqual(target.read_bytes(), b"new")
            self.assertEqual(list(pathlib.Path(tmp).glob(".selection.json.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
