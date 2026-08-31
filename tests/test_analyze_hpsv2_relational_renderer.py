from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVAL_PIPELINE = ROOT / "eval-pipeline"
sys.path.insert(0, str(EVAL_PIPELINE))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


analyzer = load_module(
    "hpsv2_relational_analyzer_test",
    EVAL_PIPELINE / "analyze_hpsv2_relational_renderer.py",
)


class HPSv2AnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = analyzer.load_contract()
        cls.metrics = list(
            cls.contract["config"]["scoring"]["required_outputs"]
        )

    def make_frame(self) -> pd.DataFrame:
        offsets = {
            "no_ag": 0.0,
            "feature_axis_r1_pos": 0.010,
            "uniform_axis_r1_pos": 0.003,
            "random_axis_r1_pos": 0.001,
        }
        prompt_labels = [f"prompt {index}" for index in range(3200)]
        for indices in ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10)):
            for index in indices[1:]:
                prompt_labels[index] = prompt_labels[indices[0]]
        for pair_start in range(11, 51, 2):
            prompt_labels[pair_start + 1] = prompt_labels[pair_start]

        rows = []
        for prompt_index in range(analyzer.EXPECTED_PROMPT_COUNT):
            style_number = prompt_index // 800
            style = analyzer.EXPECTED_STYLES[style_number]
            style_index = prompt_index % 800
            for action_id, _source, _role in analyzer.EXPECTED_SETTINGS:
                values = {
                    "hpsv2": 0.20 + offsets[action_id],
                    "topiq_nr": 0.50,
                    "imagereward": 0.10,
                    "patch_ir_mean": 0.10,
                    "patch_ir_std": 0.01,
                    "patch_ir_n": 5.0,
                    "clip_cosine": 0.30,
                    "clipscore": 0.75,
                    "aesthetic": 5.0,
                    "colorfulness": 20.0,
                    "laplacian_sharpness": 30.0,
                    "clipped_fraction": 0.0,
                    "mean_saturation": 0.4,
                    "contrast_std": 0.2,
                }
                rows.append(
                    {
                        "id": f"p{prompt_index:04d}_{action_id}",
                        "prompt": prompt_labels[prompt_index],
                        "prompt_index": prompt_index,
                        "benchmark_style": style,
                        "style_index": style_index,
                        "seed": analyzer.EXPECTED_BASE_SEED + prompt_index,
                        "action_id": action_id,
                        **values,
                    }
                )
        return pd.DataFrame(rows)

    def test_complete_style_group_and_paired_outputs(self) -> None:
        frame = self.make_frame()
        analyzer.validate_analysis_grid(frame, self.metrics)
        styles = analyzer.build_style_means(frame, self.metrics)
        groups = analyzer.build_group_means(frame, self.metrics)
        summary = analyzer.build_group_summary(groups)
        self.assertEqual(len(styles), 20)
        self.assertEqual(len(groups), 160)
        self.assertEqual(len(summary), 16)
        self.assertTrue((groups["sample_count"] == 80).all())
        self.assertTrue(
            (summary["hpsv2_official_group_std"].abs() < 1e-12).all()
        )

        def mean_interval(values, _cluster_labels, **_kwargs):
            mean = float(pd.Series(values).mean())
            return mean - 1e-4, mean + 1e-4

        with mock.patch.object(
            analyzer, "cluster_bootstrap_mean_interval", side_effect=mean_interval
        ):
            comparisons = analyzer.build_paired_comparisons(
                frame,
                self.metrics,
                self.contract["config"]["analysis"],
            )
        self.assertEqual(len(comparisons), 3 * (len(self.metrics) + 1))
        checks, passed = analyzer.apply_decision(
            comparisons, self.contract["config"]["analysis"]
        )
        self.assertTrue(passed)
        self.assertTrue(all(item["passed"] for item in checks))

    def test_analysis_rejects_an_incomplete_official_matrix(self) -> None:
        frame = self.make_frame().iloc[:-1]
        with self.assertRaisesRegex(ValueError, "12,800"):
            analyzer.validate_analysis_grid(frame, self.metrics)

    def test_frozen_cluster_bootstrap_handles_constant_effect(self) -> None:
        frame = self.make_frame()
        prompt_rows = frame[frame["action_id"] == "no_ag"].sort_values(
            "prompt_index"
        )
        low, high = analyzer.cluster_bootstrap_mean_interval(
            [0.01] * analyzer.EXPECTED_PROMPT_COUNT,
            prompt_rows["prompt"].tolist(),
            samples=10000,
            confidence=0.95,
            seed=analyzer.EXPECTED_BASE_SEED,
        )
        self.assertAlmostEqual(low, 0.01)
        self.assertAlmostEqual(high, 0.01)


class QueueAnalysisContractTest(unittest.TestCase):
    def test_analysis_binds_the_registered_scorer_hash(self) -> None:
        contract = analyzer.load_contract()
        scoring_config, _, registered_contract, registered_hash = (
            analyzer._load_frozen_scoring_config(contract)
        )
        self.assertIsNone(registered_contract)
        self.assertEqual(
            registered_hash,
            "c8b2adf8f4f7d2aa7812f6a0c5e8f8cf33d709bed4b769c8bc3e47c8e16743b2",
        )
        with mock.patch.object(
            analyzer,
            "validate_hardened_score_rows",
            return_value=registered_hash,
        ) as validator:
            self.assertEqual(
                analyzer.validate_registered_scorer_rows([{"id": "fixture"}], contract),
                registered_hash,
            )
        validator.assert_called_once_with(
            [{"id": "fixture"}],
            required_schema=scoring_config["scorer_provenance"]["required_schema"],
            expected_sha256=registered_hash,
            expected_contract=None,
        )

    def test_queue_runs_analysis_only_after_strict_scoring(self) -> None:
        source = (
            EVAL_PIPELINE / "run_hpsv2_relational_renderer_queue.sh"
        ).read_text(encoding="utf-8")
        scoring = source.index("--require-scorer-provenance")
        analysis = source.index('"$EVAL_PYTHON" "$ANALYZER"')
        self.assertLess(scoring, analysis)
        self.assertIn("score_hpsv2_relational_renderer.py", source)
        self.assertIn("paired_analysis", source)
        self.assertIn("HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1", source)


if __name__ == "__main__":
    unittest.main()
