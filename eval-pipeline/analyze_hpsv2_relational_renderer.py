"""Analyze the complete frozen HPSv2 relational-renderer experiment."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from generate_hpsv2_relational_renderer import (
    EXPECTED_BASE_SEED,
    EXPECTED_PROMPT_COUNT,
    EXPECTED_SETTINGS,
    EXPECTED_STYLES,
    EXPECTED_TASK_COUNT,
    EXPECTED_UNIQUE_PROMPT_COUNT,
    _atomic_write_bytes,
    load_contract,
    sha256_file,
    validate_complete_run,
)
from score_hpsv2_relational_renderer import (
    SCORING_SUCCESS_NAME,
    scoring_success_receipt_bytes,
    validate_exclusive_cuda_execution_contract,
    validate_scoring_success_receipt,
)
from scorer_provenance import validate_hardened_score_rows
from s7_provenance import validate_scores_against_manifest


ANALYSIS_SCHEMA = "hpsv2_relational_renderer_analysis_v1"
ROOT = Path(__file__).resolve().parents[1]
STYLE_MEANS_NAME = "hpsv2_style_means.csv"
GROUP_MEANS_NAME = "hpsv2_80_prompt_groups.csv"
GROUP_SUMMARY_NAME = "hpsv2_group_summary.csv"
COMPARISONS_NAME = "paired_comparisons.csv"
DECISION_NAME = "analysis.json"
REPORT_NAME = "analysis_summary.md"
CONTRASTS = (
    ("feature_axis_r1_pos", "no_ag"),
    ("feature_axis_r1_pos", "uniform_axis_r1_pos"),
    ("feature_axis_r1_pos", "random_axis_r1_pos"),
)


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing or unsafe: {path}")
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{label} line {line_number} is not a mapping")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    return rows


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing or unsafe: {path}")
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a mapping")
    return value, payload


def validate_scoring_publication(
    run_dir: Path,
    manifest: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    scorer_hash: str,
) -> str:
    """Recompute the scoring receipt from the current frozen artifacts."""
    if not scores:
        raise ValueError("scoring receipt validation requires score rows")
    scoring_contract = contract["config"]["scoring"]
    scoring_config_path = ROOT / str(scoring_contract["config"])
    try:
        scoring_config_payload = scoring_config_path.read_bytes()
        scoring_config = yaml.safe_load(scoring_config_payload) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("cannot read the frozen scoring config") from exc
    if not isinstance(scoring_config, dict):
        raise ValueError("frozen scoring config must contain a mapping")
    run_config, run_config_payload = _read_json(
        run_dir / "config.json", "recorded run config"
    )
    receipt, receipt_payload = _read_json(
        run_dir / SCORING_SUCCESS_NAME, "scoring success receipt"
    )
    scores_path = run_dir / "scores.jsonl"
    expected = validate_scoring_success_receipt(
        receipt,
        scores_sha256=sha256_file(scores_path),
        score_rows=scores,
        manifest_sha256=sha256_file(run_dir / "manifest.jsonl"),
        manifest_rows=manifest,
        run_config_sha256=hashlib.sha256(run_config_payload).hexdigest(),
        run_contract_sha256=run_config.get("run_contract_sha256"),
        scoring_config_sha256=hashlib.sha256(scoring_config_payload).hexdigest(),
        scoring_config=scoring_config,
        metric_names=list(scoring_config.get("metrics", [])),
        params=dict(scoring_config.get("params", {})),
        strict=True,
        required_scorer_provenance_schema=(
            scoring_config.get("scorer_provenance", {}).get("required_schema")
        ),
        scorer_provenance=scores[0].get("scorer_provenance"),
        scorer_provenance_sha256=scorer_hash,
        cuda_execution_provenance=scores[0].get(
            "scoring_execution_provenance"
        ),
        cuda_execution_provenance_sha256=scores[0].get(
            "scoring_execution_provenance_sha256"
        ),
    )
    if receipt_payload != scoring_success_receipt_bytes(expected):
        raise ValueError("scoring success receipt is not canonical")
    return sha256_file(run_dir / SCORING_SUCCESS_NAME)


def _finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


@contextmanager
def _run_lock(run_dir: Path):
    lock_path = run_dir / ".generate.lock"
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("generation or scoring still owns the run lock") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_verified_frame(
    run_dir: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], str]:
    """Audit artifacts, scores, hashes, and the exact one-to-one score join."""
    manifest = validate_complete_run(run_dir)
    scores = _read_jsonl(run_dir / "scores.jsonl", "strict scores")
    if len(manifest) != EXPECTED_TASK_COUNT or len(scores) != EXPECTED_TASK_COUNT:
        raise ValueError("analysis requires exactly 12,800 manifest and score rows")
    validate_scores_against_manifest(manifest, scores)
    scorer_hash = validate_hardened_score_rows(scores)

    contract = load_contract()
    metrics = list(contract["config"]["scoring"]["required_outputs"])
    scorer_contract = scores[0].get("scorer_provenance")
    expected_scorers = list(contract["config"]["scoring"]["metrics"])
    expected_params = {"patch_crops": 5, "clip_model": "ViT-B/32", "clipscore_w": 2.5}
    if (
        not isinstance(scorer_contract, Mapping)
        or scorer_contract.get("metrics") != expected_scorers
        or scorer_contract.get("config_params") != expected_params
    ):
        raise ValueError("strict scorer provenance differs from the frozen scoring recipe")
    scorer_rows = scorer_contract.get("scorers")
    if not isinstance(scorer_rows, list):
        raise ValueError("strict scorer provenance lacks scorer records")
    observed_outputs = []
    for scorer in scorer_rows:
        if not isinstance(scorer, Mapping) or not isinstance(
            scorer.get("output_keys"), list
        ):
            raise ValueError("strict scorer provenance output contract is invalid")
        observed_outputs.extend(
            output.get("name")
            for output in scorer["output_keys"]
            if isinstance(output, Mapping)
        )
    if len(observed_outputs) != len(set(observed_outputs)) or set(observed_outputs) != set(
        metrics
    ):
        raise ValueError("strict scorer outputs differ from the frozen required metrics")
    score_by_id: dict[str, dict[str, Any]] = {}
    for score in scores:
        validate_exclusive_cuda_execution_contract(
            score.get("scoring_execution_provenance"),
            score.get("scoring_execution_provenance_sha256"),
            expected_device="cuda:2",
        )
        score_id = score.get("id")
        if not isinstance(score_id, str) or not score_id or score_id in score_by_id:
            raise ValueError("scores contain an empty or duplicate id")
        missing = [metric for metric in metrics if metric not in score]
        if missing:
            raise ValueError(f"score {score_id} lacks required metrics: {missing}")
        if any(not _finite(score[metric]) for metric in metrics):
            raise ValueError(f"score {score_id} contains a non-finite required metric")
        score_by_id[score_id] = score

    validate_scoring_publication(
        run_dir, manifest, scores, contract, scorer_hash
    )

    rows = []
    for record in manifest:
        score = score_by_id.get(str(record["id"]))
        if score is None:
            raise ValueError(f"manifest row {record['id']} has no score")
        rows.append({**dict(record), **{metric: float(score[metric]) for metric in metrics}})
    frame = pd.DataFrame(rows)
    validate_analysis_grid(frame, metrics)
    return frame, manifest, scores, scorer_hash


def validate_analysis_grid(frame: pd.DataFrame, metrics: Sequence[str]) -> None:
    required = {
        "id",
        "prompt",
        "prompt_index",
        "benchmark_style",
        "style_index",
        "seed",
        "action_id",
        *metrics,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"analysis frame lacks columns: {missing}")
    if len(frame) != EXPECTED_TASK_COUNT or frame["id"].duplicated().any():
        raise ValueError("analysis frame must contain 12,800 unique task rows")
    observed_seeds = frame[["prompt_index", "seed"]].drop_duplicates()
    if len(observed_seeds) != EXPECTED_PROMPT_COUNT or any(
        int(row.seed) != EXPECTED_BASE_SEED + int(row.prompt_index)
        for row in observed_seeds.itertuples(index=False)
    ):
        raise ValueError("analysis frame differs from the frozen per-row seed policy")
    expected_actions = {row[0] for row in EXPECTED_SETTINGS}
    expected_cells = {
        (prompt_index, action_id)
        for prompt_index in range(EXPECTED_PROMPT_COUNT)
        for action_id in expected_actions
    }
    observed_cells = set(
        frame[["prompt_index", "action_id"]].itertuples(index=False, name=None)
    )
    if observed_cells != expected_cells:
        raise ValueError("analysis frame differs from the complete prompt/setting product")
    prompt_counts = frame.groupby("prompt_index")["prompt"].nunique()
    if not (prompt_counts == 1).all():
        raise ValueError("paired settings disagree on prompt text")
    prompts = (
        frame[frame["action_id"] == EXPECTED_SETTINGS[0][0]]
        .sort_values("prompt_index")["prompt"]
        .astype(str)
    )
    exact_counts = prompts.value_counts()
    if (
        len(exact_counts) != EXPECTED_UNIQUE_PROMPT_COUNT
        or int((exact_counts > 1).sum()) != 23
        or int(exact_counts.max()) != 4
    ):
        raise ValueError("analysis frame exact-prompt clusters differ from HPSv2")
    for style_index, style in enumerate(EXPECTED_STYLES):
        selected = frame[frame["benchmark_style"] == style]
        if len(selected) != 800 * len(expected_actions):
            raise ValueError(f"style {style} does not contain 800 rows per setting")
        expected_prompt_indices = set(range(style_index * 800, (style_index + 1) * 800))
        if set(selected["prompt_index"]) != expected_prompt_indices:
            raise ValueError(f"style {style} prompt indices differ from the official rows")
        by_prompt = selected.groupby("prompt_index")["style_index"].nunique()
        if not (by_prompt == 1).all() or set(selected["style_index"]) != set(range(800)):
            raise ValueError(f"style {style} indices differ from 0..799")
    numeric = frame[list(metrics)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("analysis frame contains non-finite metric values")


def cluster_bootstrap_mean_interval(
    values: Sequence[float],
    cluster_labels: Sequence[str],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    labels = np.asarray(cluster_labels, dtype=object)
    if (
        array.ndim != 1
        or labels.ndim != 1
        or len(array) != EXPECTED_PROMPT_COUNT
        or len(labels) != EXPECTED_PROMPT_COUNT
    ):
        raise ValueError("cluster bootstrap requires 3,200 paired values and labels")
    if not np.isfinite(array).all() or samples != 10000 or confidence != 0.95:
        raise ValueError("cluster bootstrap inputs differ from the frozen protocol")
    codes, unique_labels = pd.factorize(labels, sort=False)
    if len(unique_labels) != EXPECTED_UNIQUE_PROMPT_COUNT:
        raise ValueError("cluster bootstrap requires exactly 3,172 prompt clusters")
    cluster_sums = np.bincount(codes, weights=array)
    cluster_counts = np.bincount(codes).astype(np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    batch_size = 250
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        indices = rng.integers(
            0,
            len(unique_labels),
            size=(stop - start, len(unique_labels)),
        )
        means[start:stop] = cluster_sums[indices].sum(axis=1) / cluster_counts[
            indices
        ].sum(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _complete_pivots(
    frame: pd.DataFrame, metrics: Sequence[str]
) -> dict[str, pd.DataFrame]:
    action_order = [row[0] for row in EXPECTED_SETTINGS]
    pivots = {}
    for metric in metrics:
        pivot = frame.pivot(index="prompt_index", columns="action_id", values=metric)
        pivot = pivot.reindex(index=range(EXPECTED_PROMPT_COUNT), columns=action_order)
        if pivot.isna().any().any():
            raise ValueError(f"metric {metric} is incomplete on the paired grid")
        pivots[metric] = pivot
    if (pivots["contrast_std"] <= 0.0).any().any():
        raise ValueError("contrast_std must be positive for geometric-ratio inference")
    return pivots


def build_style_means(frame: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    rows = []
    for action_id, _source, _role in EXPECTED_SETTINGS:
        for style in EXPECTED_STYLES:
            selected = frame[
                (frame["action_id"] == action_id)
                & (frame["benchmark_style"] == style)
            ]
            if len(selected) != 800:
                raise ValueError("style mean does not contain exactly 800 prompt rows")
            row = {
                "action_id": action_id,
                "benchmark_style": style,
                "sample_count": 800,
                **{f"{metric}_mean": float(selected[metric].mean()) for metric in metrics},
            }
            row["hpsv2_official_mean"] = 100.0 * row["hpsv2_mean"]
            rows.append(row)
        selected = frame[frame["action_id"] == action_id]
        if len(selected) != EXPECTED_PROMPT_COUNT:
            raise ValueError("overall HPSv2 mean requires exactly 3,200 prompt rows")
        row = {
            "action_id": action_id,
            "benchmark_style": "Average",
            "sample_count": EXPECTED_PROMPT_COUNT,
            **{f"{metric}_mean": float(selected[metric].mean()) for metric in metrics},
        }
        row["hpsv2_official_mean"] = 100.0 * row["hpsv2_mean"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_group_means(frame: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    work = frame.copy()
    work["official_group_index"] = work["style_index"].astype(int) // 80
    rows = []
    for action_id, _source, _role in EXPECTED_SETTINGS:
        for style in EXPECTED_STYLES:
            for group_index in range(10):
                selected = work[
                    (work["action_id"] == action_id)
                    & (work["benchmark_style"] == style)
                    & (work["official_group_index"] == group_index)
                ]
                if len(selected) != 80:
                    raise ValueError("official HPSv2 group must contain exactly 80 rows")
                row = {
                    "action_id": action_id,
                    "benchmark_style": style,
                    "official_group_index": group_index,
                    "style_index_start": group_index * 80,
                    "style_index_end": group_index * 80 + 79,
                    "sample_count": 80,
                    **{
                        f"{metric}_mean": float(selected[metric].mean())
                        for metric in metrics
                    },
                }
                row["hpsv2_official_mean"] = 100.0 * row["hpsv2_mean"]
                rows.append(row)
    return pd.DataFrame(rows)


def build_group_summary(groups: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for action_id, _source, _role in EXPECTED_SETTINGS:
        for style in EXPECTED_STYLES:
            values = groups[
                (groups["action_id"] == action_id)
                & (groups["benchmark_style"] == style)
            ]["hpsv2_official_mean"].to_numpy(float)
            if len(values) != 10:
                raise ValueError("group summary requires ten official groups per style")
            rows.append(
                {
                    "action_id": action_id,
                    "benchmark_style": style,
                    "group_count": 10,
                    "group_size": 80,
                    "hpsv2_official_group_mean": float(values.mean()),
                    "hpsv2_official_group_std": float(values.std(ddof=0)),
                    "hpsv2_official_group_min": float(values.min()),
                    "hpsv2_official_group_max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def build_paired_comparisons(
    frame: pd.DataFrame,
    metrics: Sequence[str],
    analysis: Mapping[str, Any],
) -> pd.DataFrame:
    pivots = _complete_pivots(frame, metrics)
    cluster_labels = (
        frame[frame["action_id"] == EXPECTED_SETTINGS[0][0]]
        .set_index("prompt_index")
        .reindex(range(EXPECTED_PROMPT_COUNT))["prompt"]
        .astype(str)
        .to_numpy()
    )
    if len(set(cluster_labels)) != int(analysis["bootstrap_cluster_count"]):
        raise ValueError("analysis bootstrap cluster count differs from the prompt grid")
    if analysis.get("bootstrap_unit") != "exact_prompt_text_cluster":
        raise ValueError("analysis bootstrap unit differs from the frozen protocol")
    if analysis.get("official_duplicate_rows_in_point_estimates") is not True:
        raise ValueError("analysis must retain official duplicate rows in point estimates")
    samples = int(analysis["paired_bootstrap_samples"])
    confidence = float(analysis["confidence_level"])
    base_seed = int(analysis["bootstrap_seed"])
    rows = []
    for contrast_index, (candidate, baseline) in enumerate(CONTRASTS):
        for metric_index, metric in enumerate(metrics):
            values = (
                pivots[metric][candidate] - pivots[metric][baseline]
            ).to_numpy(float)
            low, high = cluster_bootstrap_mean_interval(
                values,
                cluster_labels,
                samples=samples,
                confidence=confidence,
                seed=base_seed + contrast_index * 1000 + metric_index,
            )
            row = {
                "candidate": candidate,
                "baseline": baseline,
                "metric": metric,
                "sample_count": EXPECTED_PROMPT_COUNT,
                "bootstrap_cluster_count": EXPECTED_UNIQUE_PROMPT_COUNT,
                "mean_delta": float(values.mean()),
                "ci95_low": low,
                "ci95_high": high,
            }
            if metric == "hpsv2":
                row.update(
                    {
                        "official_mean_delta": 100.0 * row["mean_delta"],
                        "official_ci95_low": 100.0 * low,
                        "official_ci95_high": 100.0 * high,
                    }
                )
            rows.append(row)

        log_ratio = np.log(
            pivots["contrast_std"][candidate].to_numpy(float)
            / pivots["contrast_std"][baseline].to_numpy(float)
        )
        low, high = cluster_bootstrap_mean_interval(
            log_ratio,
            cluster_labels,
            samples=samples,
            confidence=confidence,
            seed=base_seed + contrast_index * 1000 + len(metrics),
        )
        rows.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "metric": "contrast_std_geometric_ratio",
                "sample_count": EXPECTED_PROMPT_COUNT,
                "bootstrap_cluster_count": EXPECTED_UNIQUE_PROMPT_COUNT,
                "mean_delta": float(np.exp(log_ratio.mean())),
                "ci95_low": float(np.exp(low)),
                "ci95_high": float(np.exp(high)),
            }
        )
    return pd.DataFrame(rows)


def apply_decision(
    comparisons: pd.DataFrame, analysis: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], bool]:
    def row(metric: str, baseline: str) -> pd.Series:
        selected = comparisons[
            (comparisons["candidate"] == "feature_axis_r1_pos")
            & (comparisons["baseline"] == baseline)
            & (comparisons["metric"] == metric)
        ]
        if len(selected) != 1:
            raise ValueError(f"missing unique decision row for {metric} vs {baseline}")
        return selected.iloc[0]

    guards = analysis["guards"]
    checks = []
    for baseline in ("no_ag", "uniform_axis_r1_pos", "random_axis_r1_pos"):
        observed = float(row("hpsv2", baseline)["ci95_low"])
        checks.append(
            {
                "id": f"hpsv2_ci_lower_above_{baseline}",
                "passed": observed > 0.0,
                "observed": observed,
                "rule": "> 0.0",
            }
        )
    for metric, key in (
        ("clip_cosine", "clip_cosine_ci_lower_min_delta"),
        ("topiq_nr", "topiq_nr_ci_lower_min_delta"),
    ):
        observed = float(row(metric, "no_ag")["ci95_low"])
        threshold = float(guards[key])
        checks.append(
            {
                "id": f"{metric}_noninferiority",
                "passed": observed >= threshold,
                "observed": observed,
                "rule": f">= {threshold}",
            }
        )
    for metric, key in (
        ("clipped_fraction", "clipped_fraction_ci_upper_max_delta"),
        ("mean_saturation", "mean_saturation_ci_upper_max_delta"),
    ):
        observed = float(row(metric, "no_ag")["ci95_high"])
        threshold = float(guards[key])
        checks.append(
            {
                "id": f"{metric}_upper_guard",
                "passed": observed <= threshold,
                "observed": observed,
                "rule": f"<= {threshold}",
            }
        )
    contrast = row("contrast_std_geometric_ratio", "no_ag")
    lower, upper = [float(value) for value in guards["contrast_geometric_ratio_interval"]]
    observed_interval = [float(contrast["ci95_low"]), float(contrast["ci95_high"])]
    checks.append(
        {
            "id": "contrast_geometric_ratio_interval",
            "passed": observed_interval[0] >= lower and observed_interval[1] <= upper,
            "observed": observed_interval,
            "rule": f"CI within [{lower}, {upper}]",
        }
    )
    return checks, all(bool(item["passed"]) for item in checks)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    return buffer.getvalue().encode("utf-8")


def _markdown_report(
    style_means: pd.DataFrame,
    comparisons: pd.DataFrame,
    checks: Sequence[Mapping[str, Any]],
    passed: bool,
) -> str:
    feature = style_means[style_means["action_id"] == "feature_axis_r1_pos"]
    baseline = style_means[style_means["action_id"] == "no_ag"]
    style_rows = []
    for style in EXPECTED_STYLES:
        feature_value = float(
            feature[feature["benchmark_style"] == style]["hpsv2_official_mean"].iloc[0]
        )
        baseline_value = float(
            baseline[baseline["benchmark_style"] == style]["hpsv2_official_mean"].iloc[0]
        )
        style_rows.append(
            f"| {style} | {baseline_value:.4f} | {feature_value:.4f} | "
            f"{feature_value - baseline_value:+.4f} |"
        )
    feature_average = float(
        feature[feature["benchmark_style"] == "Average"]["hpsv2_official_mean"].iloc[0]
    )
    baseline_average = float(
        baseline[baseline["benchmark_style"] == "Average"]["hpsv2_official_mean"].iloc[0]
    )
    style_rows.append(
        f"| Average | {baseline_average:.4f} | {feature_average:.4f} | "
        f"{feature_average - baseline_average:+.4f} |"
    )
    primary = comparisons[
        (comparisons["candidate"] == "feature_axis_r1_pos")
        & (comparisons["baseline"] == "no_ag")
        & (comparisons["metric"] == "hpsv2")
    ].iloc[0]
    conclusion = (
        "通过：可以进入 prompt-disjoint 的 OPD / latent renderer 下一阶段。"
        if passed
        else "未通过：这个固定方向不能作为 OPD、DPO 或 RL 的 teacher/监督目标。"
    )
    guard_lines = [
        f"- [{'x' if item['passed'] else ' '}] `{item['id']}`: "
        f"observed={item['observed']}, rule={item['rule']}"
        for item in checks
    ]
    return "\n".join(
        [
            "# HPSv2 全量实验结果",
            "",
            "## 为什么做这个实验",
            "",
            "比较 U-Net feature relation、普通局部平滑和随机方向，判断当前固定 latent "
            "修正方向是否真的包含可用于后续训练的结构信息。",
            "",
            "## 完整性",
            "",
            "四个实验设置都覆盖 HPSv2 官方 3,200 行，每个设置 3,200 张，共 12,800 张。"
            "所有结论按同一官方行、同 seed 配对计算；官方均值保留全部 3,200 行，"
            "置信区间按 3,172 个不同 prompt 文本成组重采样。",
            "",
            "## HPSv2 官方风格分数",
            "",
            "| 风格 | no_ag | feature | 差值 |",
            "|---|---:|---:|---:|",
            *style_rows,
            "",
            "feature 相对 no_ag 的全量 HPSv2 原始余弦差值为 "
            f"`{float(primary['mean_delta']):+.6f}`，95% CI "
            f"`[{float(primary['ci95_low']):+.6f}, {float(primary['ci95_high']):+.6f}]`。",
            "",
            "## 门槛检查",
            "",
            *guard_lines,
            "",
            "## 结论",
            "",
            conclusion,
            "",
            "详细数据见同目录 CSV 和 `analysis.json`。",
            "",
        ]
    )


def analyze(run_dir: Path) -> dict[str, Any]:
    contract = load_contract()
    run_dir = run_dir.resolve()
    if run_dir != contract["output_dir"]:
        raise ValueError("analyzer only accepts the frozen HPSv2 full-v1 run directory")
    with _run_lock(run_dir):
        frame, manifest, _scores, scorer_hash = load_verified_frame(run_dir)
        metrics = list(contract["config"]["scoring"]["required_outputs"])
        analysis = contract["config"]["analysis"]
        style_means = build_style_means(frame, metrics)
        groups = build_group_means(frame, metrics)
        group_summary = build_group_summary(groups)
        comparisons = build_paired_comparisons(frame, metrics, analysis)
        checks, passed = apply_decision(comparisons, analysis)

        outputs = {
            STYLE_MEANS_NAME: _csv_bytes(style_means),
            GROUP_MEANS_NAME: _csv_bytes(groups),
            GROUP_SUMMARY_NAME: _csv_bytes(group_summary),
            COMPARISONS_NAME: _csv_bytes(comparisons),
        }
        for name, payload in outputs.items():
            _atomic_write_bytes(run_dir / name, payload)
        report = _markdown_report(style_means, comparisons, checks, passed)
        _atomic_write_bytes(run_dir / REPORT_NAME, report.encode("utf-8"))

        result = {
            "schema": ANALYSIS_SCHEMA,
            "experiment_id": contract["config"]["experiment_id"],
            "status": "complete",
            "sample_count_per_setting": EXPECTED_PROMPT_COUNT,
            "setting_count": len(EXPECTED_SETTINGS),
            "total_sample_count": EXPECTED_TASK_COUNT,
            "statistical_unit": "official_hpsv2_prompt_row",
            "bootstrap_unit": "exact_prompt_text_cluster",
            "bootstrap_cluster_count": EXPECTED_UNIQUE_PROMPT_COUNT,
            "bootstrap_samples": int(analysis["paired_bootstrap_samples"]),
            "confidence_level": float(analysis["confidence_level"]),
            "scorer_provenance_sha256": scorer_hash,
            "inputs": {
                "manifest_sha256": sha256_file(run_dir / "manifest.jsonl"),
                "scores_sha256": sha256_file(run_dir / "scores.jsonl"),
                "scoring_success_sha256": sha256_file(
                    run_dir / SCORING_SUCCESS_NAME
                ),
                "config_sha256": contract["config_sha256"],
            },
            "artifacts": {
                name: sha256_file(run_dir / name)
                for name in (*outputs, REPORT_NAME)
            },
            "checks": checks,
            "passed": passed,
            "decision": (
                "advance_fixed_direction_to_prompt_disjoint_opd_design"
                if passed
                else "do_not_use_fixed_direction_as_rl_opd_dpo_teacher"
            ),
        }
        _atomic_write_bytes(
            run_dir / DECISION_NAME,
            (json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode(
                "ascii"
            ),
        )
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the frozen complete HPSv2 relational-renderer run",
        allow_abbrev=False,
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = analyze(args.run_dir)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
