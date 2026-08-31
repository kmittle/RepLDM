"""Build figures for results/08-30/results-08-30.md from frozen artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

COLORS = {
    "blue": "#2F6F9F",
    "green": "#248A5A",
    "orange": "#D47A24",
    "red": "#C34A46",
    "gray": "#6B737C",
    "light_gray": "#D7DDE2",
    "ink": "#20262D",
}

EVIDENCE_SCALE = (
    ("S7 scheduler correction", 22, "11 prompts x 2 seeds"),
    ("S0-S5 early pilots", 36, "12 prompts x 3 seeds"),
    ("S6 FreeU", 48, "24 prompts x 2 seeds"),
    ("Native headroom screen", 99, "33 prompts x 3 seeds"),
    ("Structural controls", 99, "33 prompts x 3 seeds"),
    ("LR-1 fixed renderer", 144, "48 prompts x 3 seeds"),
    ("Formal 1024px target", 3000, "1,000 prompts x 3 seeds"),
    ("Formal 2048px target", 6000, "2,000 prompts x 3 seeds"),
)

OUTPUT_NAMES = (
    "01_evidence_scale.png",
    "02_topiq_overview.png",
    "03_guidance_progression.jpg",
    "04_stage2_examples.jpg",
    "05_renderer_wiring.jpg",
    "06_freeu_tradeoff.png",
    "07_freeu_examples.jpg",
    "08_scheduler_gain.png",
)

SOURCES: set[Path] = set()


def register(path: Path) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    SOURCES.add(path)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    path = register(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_run_provenance(run_dir: Path) -> tuple[dict, pd.DataFrame]:
    config = load_json(run_dir / "config.json")
    configured_path = Path(config["prompts_csv"])
    if "eval-pipeline" in configured_path.parts:
        start = configured_path.parts.index("eval-pipeline")
        prompts_path = ROOT.joinpath(*configured_path.parts[start:])
    else:
        prompts_path = configured_path if configured_path.is_absolute() else ROOT / configured_path
    prompts_path = register(prompts_path)
    expected_hash = config.get("prompts_sha256")
    if expected_hash is not None and sha256(prompts_path) != expected_hash:
        raise ValueError(f"Prompt CSV hash mismatch: {prompts_path}")
    prompts = pd.read_csv(prompts_path)
    required = {"index", "TEXT"}
    if not required.issubset(prompts.columns):
        raise ValueError(f"Prompt CSV lacks {sorted(required)}: {prompts_path}")
    return config, prompts


def registered_prompt(prompts: pd.DataFrame, prompt_index: int, source: str) -> str:
    prompt_rows = prompts[prompts["index"] == prompt_index]
    if len(prompt_rows) != 1:
        raise ValueError(f"Expected one registered prompt {prompt_index} in {source}")
    return str(prompt_rows.iloc[0]["TEXT"])


def validated_generated_image(
    run_dir: Path,
    config: dict,
    prompts: pd.DataFrame,
    prompt_index: int,
    seed: int,
    action_id: str,
) -> Path:
    image_path = run_dir / "images" / f"p{prompt_index}_seed{seed}_a{action_id}.png"
    image_path = register(image_path)
    metadata = load_json(image_path.with_suffix(".json"))

    expected_prompt = registered_prompt(prompts, prompt_index, config["prompts_csv"])
    configured_actions = {item["id"] for item in config["actions"]}
    if action_id not in configured_actions or seed not in config["seeds"]:
        raise ValueError(f"Image setting is absent from run config: {image_path}")

    resolution = int(config["resolution"])
    expected = {
        "id": image_path.stem,
        "prompt_index": prompt_index,
        "prompt": expected_prompt,
        "seed": seed,
        "action_id": action_id,
        "image_path": f"images/{image_path.name}",
        "height": resolution,
        "width": resolution,
        "model_name": config["model_name"],
        "git_commit": config["git_commit"],
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Image sidecar mismatch for {image_path}: {mismatches}")
    with Image.open(image_path) as image:
        if image.size != (resolution, resolution):
            raise ValueError(f"Image dimensions disagree with sidecar: {image_path}")
    return image_path


def configure_plot_style() -> None:
    missing = [str(path) for path in (FONT_REGULAR, FONT_BOLD) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required Noto CJK fonts are missing: "
            + ", ".join(missing)
            + ". Install the fonts-noto-cjk package before regenerating figures."
        )
    font_manager.fontManager.addfont(FONT_REGULAR)
    family = font_manager.FontProperties(fname=FONT_REGULAR).get_name()
    plt.rcParams["font.family"] = family
    plt.rcParams.update(
        {
            "axes.edgecolor": COLORS["light_gray"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "grid.color": "#E6EAED",
            "grid.linewidth": 0.8,
            "text.color": COLORS["ink"],
            "xtick.color": "#4B535B",
            "ytick.color": "#4B535B",
            "axes.unicode_minus": False,
        }
    )


def comparison_row(
    relative_csv: str, metric: str, action: str, baseline: str
) -> tuple[float, float, float]:
    path = register(ROOT / relative_csv)
    frame = pd.read_csv(path)
    row = frame[
        (frame["metric"] == metric)
        & (frame["action"] == action)
        & (frame["baseline"] == baseline)
    ]
    if len(row) != 1:
        raise ValueError(f"Expected one row for {path}: {metric}/{baseline}/{action}")
    item = row.iloc[0]
    return float(item.mean_delta), float(item.ci95_low), float(item.ci95_high)


def native_headroom_row(action: str) -> tuple[float, float, float]:
    path = register(
        ROOT
        / "outputs/archived_results/latent_renderer/scheduler_native_fixed_headroom_development_v2/"
        "scheduler_native_fixed_headroom_evaluation.csv"
    )
    frame = pd.read_csv(path)
    row = frame[frame["action"] == action]
    if len(row) != 1:
        raise ValueError(f"Expected one row for {action}")
    item = row.iloc[0]
    return (
        float(item.topiq_nr_mean_delta),
        float(item.topiq_nr_ci_low),
        float(item.topiq_nr_ci_high),
    )


def save_plot(fig: plt.Figure, name: str) -> None:
    path = FIGURES / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_evidence_scale() -> None:
    labels = [label for label, _, _ in EVIDENCE_SCALE]
    values = np.array([value for _, value, _ in EVIDENCE_SCALE])
    colors = [COLORS["gray"]] * 6 + [COLORS["green"], COLORS["green"]]

    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.62)
    ax.set_xscale("log")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("每个实验设置的样本数（prompt 数 x seed 数，对数坐标）")
    ax.set_title("每个实验设置的评估样本数")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    for yi, value in zip(y, values):
        ax.text(value * 1.08, yi, f"{value:,}", va="center", fontsize=10)
    ax.text(
        0.01,
        -0.15,
        "灰色：pilot/development；绿色：主流正式目标。设置数不计入单个方法的样本数。",
        transform=ax.transAxes,
        fontsize=9.5,
        color=COLORS["gray"],
    )
    save_plot(fig, "01_evidence_scale.png")


def make_topiq_overview() -> None:
    rows = [
        (
            "统一 scalar 0.004（合规复现）",
            comparison_row(
                "outputs/archived_results/exp_spectral_headroom/pilot_12prompt_3seed_v1/action_comparisons.csv",
                "topiq_nr",
                "scalar_0.004",
                "no_ag",
            ),
            "negative",
        ),
        (
            "S1 只增强中频",
            comparison_row(
                "outputs/archived_results/exp_spectral_headroom/pilot_12prompt_3seed_v1/action_comparisons.csv",
                "topiq_nr",
                "mid_only_0.004",
                "no_ag",
            ),
            "uncertain",
        ),
        (
            "S2 固定均值/方差",
            comparison_row(
                "outputs/archived_results/exp_moment_tangent/development_12prompt_3seed_v1/action_comparisons.csv",
                "topiq_nr",
                "moment_tangent_0.002",
                "no_ag",
            ),
            "uncertain",
        ),
        (
            "S3 删除反向分量",
            comparison_row(
                "outputs/archived_results/exp_trajectory_cone/development_12prompt_3seed_v1/action_comparisons.csv",
                "topiq_nr",
                "trajectory_cone_0.002",
                "no_ag",
            ),
            "uncertain",
        ),
        (
            "S4 2048px trajectory cone",
            comparison_row(
                "outputs/archived_results/exp_stage2_transfer/pilot_12prompt_3seed_v1/action_comparisons.csv",
                "topiq_nr",
                "trajectory_cone_0.002",
                "no_ag",
            ),
            "uncertain",
        ),
        (
            "S5 semantic transport",
            comparison_row(
                "outputs/archived_results/exp_s5/development_12prompt_3seed_v2/action_comparisons.csv",
                "topiq_nr",
                "reciprocal_semantic_0.01",
                "no_ag",
            ),
            "uncertain",
        ),
        (
            "LR-1 fixed renderer",
            comparison_row(
                "outputs/archived_results/latent_renderer/lr1_fixed_train_searchseeds_v2/action_comparisons.csv",
                "topiq_nr",
                "spectral_mid_pos",
                "no_ag",
            ),
            "uncertain",
        ),
        ("修正坐标后的中频", native_headroom_row("spectral_mid_native"), "guard_fail"),
    ]

    fig = plt.figure(figsize=(11.5, 7.7))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.2, 6.0], hspace=0.42)
    top = fig.add_subplot(grid[0])
    bottom = fig.add_subplot(grid[1])
    fig.suptitle("代表性实验设置的 TOPIQ-NR 配对差值与 95% CI", fontsize=16, fontweight="bold")

    label, (mean, low, high), _ = rows[0]
    top.errorbar(
        mean,
        0,
        xerr=[[mean - low], [high - mean]],
        fmt="o",
        color=COLORS["red"],
        capsize=4,
        markersize=7,
    )
    top.axvline(0, color=COLORS["ink"], linewidth=1)
    top.axvline(0.005, color=COLORS["green"], linestyle="--", linewidth=1.5)
    top.set_xlim(-0.10, 0.015)
    top.set_ylim(-0.6, 0.6)
    top.set_yticks([0], [label])
    top.grid(axis="x")
    top.annotate(f"{mean:+.4f}", (mean, 0), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9)

    later = rows[1:]
    y = np.arange(len(later))
    status_colors = {
        "uncertain": COLORS["gray"],
        "guard_fail": COLORS["orange"],
        "negative": COLORS["red"],
    }
    for yi, (item_label, (item_mean, item_low, item_high), status) in zip(y, later):
        color = status_colors[status]
        bottom.errorbar(
            item_mean,
            yi,
            xerr=[[item_mean - item_low], [item_high - item_mean]],
            fmt="o",
            color=color,
            capsize=4,
            markersize=7,
        )
        bottom.text(0.0097, yi, f"{item_mean:+.4f}", ha="right", va="center", fontsize=9)
    bottom.axvline(0, color=COLORS["ink"], linewidth=1, label="与基线相同")
    bottom.axvline(
        0.005,
        color=COLORS["green"],
        linestyle="--",
        linewidth=1.5,
        label="预设 +0.005 门槛",
    )
    bottom.set_xlim(-0.0105, 0.0105)
    bottom.set_yticks(y, [item[0] for item in later])
    bottom.invert_yaxis()
    bottom.set_xlabel("相对同一运行内 no-op/no-AG 的 TOPIQ-NR 差值")
    bottom.grid(axis="x")
    bottom.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=False,
    )
    bottom.text(
        0.0,
        -0.28,
        "橙色点：CI 为正，但效应量或保护指标失败；其余灰色点的 CI 跨 0。不同运行不可直接排名。",
        transform=bottom.transAxes,
        fontsize=9.5,
        color=COLORS["gray"],
    )
    save_plot(fig, "02_topiq_overview.png")


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(str(path), size=size)


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: str = "#20262D",
) -> None:
    left, top, right, bottom = box
    bounds = draw.multiline_textbbox((0, 0), text, font=font, align="center", spacing=4)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.multiline_text(
        ((left + right - width) / 2, (top + bottom - height) / 2),
        text,
        font=font,
        fill=fill,
        align="center",
        spacing=4,
    )


def make_grid(
    title: str,
    row_labels: list[str],
    column_labels: list[str],
    paths: list[list[Path]],
    output_name: str,
    tile: int = 320,
) -> None:
    if len(paths) != len(row_labels) or any(len(row) != len(column_labels) for row in paths):
        raise ValueError("Image grid shape does not match labels")
    left_margin, top_margin, header, gap, bottom = 112, 74, 56, 10, 22
    width = left_margin + len(column_labels) * tile + (len(column_labels) - 1) * gap
    height = top_margin + header + len(row_labels) * tile + (len(row_labels) - 1) * gap + bottom
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    centered_text(draw, (0, 8, width, top_margin - 4), title, load_font(28, bold=True))

    for col, label in enumerate(column_labels):
        x = left_margin + col * (tile + gap)
        centered_text(draw, (x, top_margin, x + tile, top_margin + header), label, load_font(18, bold=True))

    for row, label in enumerate(row_labels):
        y = top_margin + header + row * (tile + gap)
        centered_text(draw, (0, y, left_margin - 8, y + tile), label, load_font(18, bold=True))
        for col, path in enumerate(paths[row]):
            path = register(path)
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image = ImageOps.fit(image, (tile, tile), method=Image.Resampling.LANCZOS)
            x = left_margin + col * (tile + gap)
            canvas.paste(image, (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline="#C9CFD4", width=2)

    output = FIGURES / output_name
    canvas.save(output, quality=90, subsampling=0, optimize=True)


def make_guidance_examples() -> None:
    spectral = ROOT / "outputs/archived_results/exp_spectral_headroom/pilot_12prompt_3seed_v1"
    moment = ROOT / "outputs/archived_results/exp_moment_tangent/development_12prompt_3seed_v1"
    cone = ROOT / "outputs/archived_results/exp_trajectory_cone/development_12prompt_3seed_v1"
    spectral_config, spectral_prompts = load_run_provenance(spectral)
    moment_config, moment_prompts = load_run_provenance(moment)
    cone_config, cone_prompts = load_run_provenance(cone)
    paths = []
    for prompt in (0, 2):
        paths.append(
            [
                validated_generated_image(
                    spectral, spectral_config, spectral_prompts, prompt, 0, "no_ag"
                ),
                validated_generated_image(
                    spectral, spectral_config, spectral_prompts, prompt, 0, "scalar_0.004"
                ),
                validated_generated_image(
                    spectral, spectral_config, spectral_prompts, prompt, 0, "mid_only_0.004"
                ),
                validated_generated_image(
                    moment, moment_config, moment_prompts, prompt, 0, "moment_tangent_0.002"
                ),
                validated_generated_image(
                    cone, cone_config, cone_prompts, prompt, 0, "trajectory_cone_0.002"
                ),
            ]
        )
    make_grid(
        "固定配对样本：约束越多，变化越小，但没有稳定结构修复",
        ["prompt 0\nseed 0", "prompt 2\nseed 0"],
        ["no-AG", "统一 scalar", "只增强中频", "固定均值/方差", "删除反向分量"],
        paths,
        "03_guidance_progression.jpg",
    )


def make_stage2_examples() -> None:
    run = ROOT / "outputs/archived_results/exp_stage2_transfer/pilot_12prompt_3seed_v1"
    config, prompts = load_run_provenance(run)
    actions = [
        ("no-AG", "no_ag"),
        ("会议设置", "conference_expert"),
        ("raw 0.001", "raw_0.001"),
        ("固定均值/方差", "moment_tangent_0.002"),
        ("删除反向分量", "trajectory_cone_0.002"),
    ]
    paths = [
        [
            validated_generated_image(run, config, prompts, prompt, 0, action)
            for _, action in actions
        ]
        for prompt in (0, 2)
    ]
    make_grid(
        "2048px Stage-2：高分辨率阶段没有反转前面的排序",
        ["prompt 0\nseed 0", "prompt 2\nseed 0"],
        [label for label, _ in actions],
        paths,
        "04_stage2_examples.jpg",
    )


def difference_image(reference: Image.Image, other: Image.Image, factor: float = 4.0) -> Image.Image:
    ref = np.asarray(reference.convert("RGB"), dtype=np.float32)
    alt = np.asarray(other.convert("RGB"), dtype=np.float32)
    difference = np.mean(np.abs(ref - alt), axis=2)
    amplified = np.clip(difference * factor / 255.0, 0.0, 1.0)
    colored = plt.get_cmap("magma")(amplified, bytes=True)[..., :3]
    return Image.fromarray(colored, mode="RGB")


def make_renderer_wiring() -> None:
    run = ROOT / "outputs/archived_results/latent_renderer/wiring_smoke_50_fee18b3"
    report = load_json(run / "report.json")
    if report.get("schema") != "latent_renderer_wiring_smoke_v1" or report.get("seed") != 123:
        raise ValueError("Unexpected LR-0 wiring report provenance")
    base = register(run / "no_renderer.png")
    zero = register(run / "zero_renderer.png")
    probe = register(run / "probe_renderer.png")
    with Image.open(base) as base_image, Image.open(zero) as zero_image, Image.open(probe) as probe_image:
        base_rgb = base_image.convert("RGB")
        zero_rgb = zero_image.convert("RGB")
        probe_rgb = probe_image.convert("RGB")
        zero_diff = difference_image(base_rgb, zero_rgb)
        probe_diff = difference_image(base_rgb, probe_rgb)
        zero_array = np.asarray(zero_rgb, dtype=np.int16) - np.asarray(base_rgb, dtype=np.int16)
        probe_array = np.asarray(probe_rgb, dtype=np.int16) - np.asarray(base_rgb, dtype=np.int16)
        zero_max = int(np.abs(zero_array).max())
        probe_mean = float(np.abs(probe_array).mean())
        pixel_hashes = {
            "no_renderer": hashlib.sha256(base_rgb.tobytes()).hexdigest(),
            "zero_renderer": hashlib.sha256(zero_rgb.tobytes()).hexdigest(),
            "probe_renderer": hashlib.sha256(probe_rgb.tobytes()).hexdigest(),
        }
        if pixel_hashes != report.get("hashes"):
            raise ValueError("LR-0 image pixels disagree with the wiring report")
        if zero_max != 0 or not report.get("zero_matches_no_renderer"):
            raise ValueError("LR-0 zero-renderer identity check failed")
        if probe_mean == 0 or not report.get("probe_differs_from_no_renderer"):
            raise ValueError("LR-0 nonzero probe did not change the image")

        temp_zero = FIGURES / ".renderer_zero_diff.png"
        temp_probe = FIGURES / ".renderer_probe_diff.png"
        zero_diff.save(temp_zero)
        probe_diff.save(temp_probe)
        try:
            make_grid(
                f"LR-0 接线验证：zero max diff = {zero_max}；probe mean abs diff = {probe_mean:.2f}",
                ["同一 prompt\n同一 seed"],
                ["无 renderer", "zero renderer", "固定非零 probe", "|no-zero| x4", "|no-probe| x4"],
                [[base, zero, probe, temp_zero, temp_probe]],
                "05_renderer_wiring.jpg",
                tile=300,
            )
        finally:
            temp_zero.unlink(missing_ok=True)
            temp_probe.unlink(missing_ok=True)
            SOURCES.discard(temp_zero.resolve())
            SOURCES.discard(temp_probe.resolve())


def paired_score_deltas(run: str, actions: Iterable[str], baseline: str) -> dict[str, dict[str, float]]:
    run_dir = ROOT / run
    config, prompts = load_run_provenance(run_dir)
    path = register(run_dir / "scores.jsonl")
    frame = pd.read_json(path, lines=True)
    index = ["prompt_index", "seed"]
    identity = ["action_id", *index]
    duplicates = frame[frame.duplicated(identity, keep=False)]
    if not duplicates.empty:
        raise ValueError(f"Duplicate scored pairs in {path}: {duplicates[identity].to_dict('records')}")

    selected = [baseline, *actions]
    configured_actions = {item["id"] for item in config["actions"]}
    if not set(selected).issubset(configured_actions):
        raise ValueError(f"Scored setting is absent from run config: {path}")
    expected_index = pd.MultiIndex.from_product(
        [prompts["index"].astype(int).tolist(), [int(seed) for seed in config["seeds"]]],
        names=index,
    ).sort_values()

    rows_by_action = {}
    for action in selected:
        rows = frame[frame.action_id == action].set_index(index).sort_index()
        if not rows.index.equals(expected_index):
            raise ValueError(f"Incomplete prompt/seed matrix for {action}: {path}")
        rows_by_action[action] = rows
    metrics = ["topiq_nr", "clipped_fraction", "mean_saturation"]
    selected_scores = frame[frame.action_id.isin(selected)][metrics].to_numpy(dtype=float)
    if not np.isfinite(selected_scores).all():
        raise ValueError(f"Non-finite paired score in {path}")

    baseline_rows = rows_by_action[baseline]
    result: dict[str, dict[str, float]] = {}
    for action in actions:
        action_rows = rows_by_action[action]
        result[action] = {
            metric: float((action_rows[metric] - baseline_rows[metric]).mean())
            for metric in metrics
        }
    return result


def make_freeu_tradeoff() -> None:
    actions = ["freeu_backbone_only", "freeu_low_window", "mp_backbone_only", "mp_low_window"]
    data = paired_score_deltas(
        "outputs/archived_results/freeu_moment_followup_v1", actions, "no_freeu"
    )
    labels = {
        "freeu_backbone_only": "普通 FreeU\nbackbone",
        "freeu_low_window": "普通 FreeU\nlow-window",
        "mp_backbone_only": "保持 moment\nbackbone",
        "mp_low_window": "保持 moment\nlow-window",
    }
    colors = {
        "freeu_backbone_only": COLORS["orange"],
        "freeu_low_window": "#E5A04B",
        "mp_backbone_only": COLORS["blue"],
        "mp_low_window": "#54A6C4",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
    for ax, metric, threshold, x_label in [
        (axes[0], "clipped_fraction", 0.001, "clipping 差值"),
        (axes[1], "mean_saturation", 0.005, "saturation 差值"),
    ]:
        xs = [data[action][metric] for action in actions]
        ys = [data[action]["topiq_nr"] for action in actions]
        left = min(xs) - 0.001
        right = max(xs) + (0.002 if metric == "clipped_fraction" else 0.004)
        ax.axhspan(0.005, 0.012, xmin=0, xmax=max(0, min(1, (threshold - left) / (right - left))), color="#E5F3EA")
        ax.axhline(0.005, color=COLORS["green"], linestyle="--", linewidth=1.2)
        ax.axvline(threshold, color=COLORS["red"], linestyle="--", linewidth=1.2)
        ax.axhline(0, color=COLORS["light_gray"], linewidth=1)
        ax.axvline(0, color=COLORS["light_gray"], linewidth=1)
        for action, x, y in zip(actions, xs, ys):
            ax.scatter(x, y, s=85, color=colors[action], edgecolor="white", linewidth=1.2, zorder=3)
        ax.set_xlim(left, right)
        ax.set_ylim(-0.001, 0.0105)
        ax.set_xlabel(x_label)
        ax.grid()
        ax.set_axisbelow(True)
    axes[0].set_ylabel("TOPIQ-NR 差值")
    axes[0].set_title("TOPIQ 与 clipping")
    axes[1].set_title("TOPIQ 与 saturation")
    fig.suptitle("FreeU：表面质量分数与像素副作用不能同时通过门槛", fontsize=15, fontweight="bold")
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=colors[action],
            markeredgecolor="white",
            markersize=9,
            label=labels[action].replace("\n", " "),
        )
        for action in actions
    ]
    axes[0].legend(
        handles=legend_handles,
        loc="lower right",
        ncol=1,
        frameon=True,
        framealpha=0.92,
        fontsize=8.5,
    )
    fig.text(
        0.5,
        -0.045,
        "左上浅绿色区域才是目标：TOPIQ ≥ +0.005，同时保护指标不越线。",
        ha="center",
        fontsize=9.5,
        color=COLORS["gray"],
    )
    save_plot(fig, "06_freeu_tradeoff.png")


def make_freeu_examples() -> None:
    run = ROOT / "outputs/archived_results/freeu_moment_followup_v1"
    config, prompts = load_run_provenance(run)
    first_indices = [int(value) for value in prompts["index"].head(3)]
    if first_indices != [0, 1, 2]:
        raise ValueError(f"Unexpected first three registered prompts: {first_indices}")
    actions = [
        ("no FreeU", "no_freeu"),
        ("普通 FreeU", "freeu_backbone_only"),
        ("保持 feature moment", "mp_backbone_only"),
    ]
    paths = [
        [
            validated_generated_image(run, config, prompts, prompt, 7, action)
            for _, action in actions
        ]
        for prompt in (0, 1, 2)
    ]
    make_grid(
        "FreeU 固定样本：增强颜色/对比度后分数上涨；保持 moment 后变化减弱",
        ["prompt 0\nseed 7", "prompt 1\nseed 7", "prompt 2\nseed 7"],
        [label for label, _ in actions],
        paths,
        "07_freeu_examples.jpg",
        tile=360,
    )


def make_scheduler_gain() -> None:
    run = ROOT / "outputs/archived_results/latent_renderer/scheduler_native_fixed_headroom_development_v2"
    config, prompts = load_run_provenance(run)
    path = run / "images/p0_seed1932556753_aspectral_mid_native.json"
    data = load_json(path)
    prompt_index, seed, action_id = 0, 1932556753, "spectral_mid_native"
    expected = {
        "id": path.stem,
        "prompt_index": prompt_index,
        "prompt": registered_prompt(prompts, prompt_index, config["prompts_csv"]),
        "seed": seed,
        "action_id": action_id,
        "height": int(config["resolution"]),
        "width": int(config["resolution"]),
        "num_inference_steps": int(config["num_inference_steps"]),
        "model_name": config["model_name"],
        "model_revision": config["model_revision"],
        "scheduler_name": config["registered_sampling"]["scheduler"],
        "git_commit": config["git_commit"],
    }
    configured_actions = {item["id"] for item in config["actions"]}
    mismatches = {
        key: (data.get(key), value) for key, value in expected.items() if data.get(key) != value
    }
    if action_id not in configured_actions or seed not in config["seeds"] or mismatches:
        raise ValueError(f"Scheduler diagnostic provenance mismatch: {mismatches}")

    diagnostics = data["latent_renderer_step_diagnostics"]
    step_count = int(config["num_inference_steps"])
    if len(diagnostics) != step_count or [step["step_index"] for step in diagnostics] != list(
        range(step_count)
    ):
        raise ValueError("Scheduler diagnostic steps are incomplete or out of order")
    gains = np.asarray(
        [step["clean_update_gain"] for step in diagnostics], dtype=np.float64
    )
    if gains.shape != (step_count, 1) or not np.isfinite(gains).all() or np.any(gains <= 0):
        raise ValueError("Scheduler clean-update gains must be finite positive scalars")
    gains = gains[:, 0]
    sigma_from = np.asarray([step["sigma_from"] for step in diagnostics], dtype=np.float64)
    sigma_to = np.asarray([step["sigma_to"] for step in diagnostics], dtype=np.float64)
    if not np.isfinite(sigma_from).all() or not np.isfinite(sigma_to).all() or np.any(sigma_from <= 0):
        raise ValueError("Scheduler diagnostics contain invalid sigma values")
    expected_gains = 1.0 - sigma_to / sigma_from
    if not np.allclose(gains, expected_gains, rtol=1e-6, atol=1e-8):
        raise ValueError("Scheduler gains disagree with 1 - sigma_to / sigma_from")
    amplification = 1.0 / gains
    steps = np.arange(len(amplification))
    median = float(np.median(amplification))
    maximum = float(np.max(amplification))
    max_step = int(np.argmax(amplification))

    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    ax.plot(steps, amplification, color=COLORS["red"], linewidth=2.2)
    ax.fill_between(steps, 1, amplification, color="#F5D8D5", alpha=0.8)
    ax.axhline(1, color=COLORS["ink"], linewidth=1, label="正确作用量")
    ax.axhline(median, color=COLORS["orange"], linestyle="--", linewidth=1.4, label=f"中位放大 {median:.2f}x")
    ax.scatter([0, max_step, 49], amplification[[0, max_step, 49]], color=COLORS["red"], zorder=3)
    ax.annotate(f"首步 {amplification[0]:.2f}x", (0, amplification[0]), xytext=(12, 12), textcoords="offset points")
    ax.annotate(f"最大 {maximum:.2f}x", (max_step, maximum), xytext=(-70, 12), textcoords="offset points")
    ax.annotate("最后一步 1.00x", (49, amplification[-1]), xytext=(-92, 18), textcoords="offset points")
    ax.set_xlim(0, 49)
    ax.set_ylim(0, 17.3)
    ax.set_xlabel("去噪 step")
    ax.set_ylabel("旧单位增益 / 正确 Euler 增益")
    ax.set_title("Scheduler 坐标审计：旧 S5/LR-1 注入在大多数步骤被放大")
    ax.grid()
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left")
    save_plot(fig, "08_scheduler_gain.png")


def write_manifest() -> None:
    expected = {FIGURES / name for name in OUTPUT_NAMES}
    actual = {
        path for path in FIGURES.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    }
    if actual != expected:
        missing = sorted(path.name for path in expected - actual)
        unexpected = sorted(path.name for path in actual - expected)
        raise ValueError(f"Figure set mismatch; missing={missing}, unexpected={unexpected}")
    outputs = [FIGURES / name for name in OUTPUT_NAMES]
    manifest = {
        "schema": "repldm_result_visualizations_v1",
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "embedded_data": {
            "01_evidence_scale.png": [
                {
                    "label": label,
                    "samples_per_setting": value,
                    "basis": basis,
                }
                for label, value, basis in EVIDENCE_SCALE
            ]
        },
        "rendering_dependencies": {
            "fonts": {
                path.name: {"path": str(path), "sha256": sha256(path)}
                for path in (FONT_REGULAR, FONT_BOLD)
            }
        },
        "inputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in sorted(SOURCES)
        },
        "outputs": {
            path.name: {
                "sha256": sha256(path),
                "width": Image.open(path).width,
                "height": Image.open(path).height,
            }
            for path in outputs
        },
    }
    (FIGURES / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    make_evidence_scale()
    make_topiq_overview()
    make_guidance_examples()
    make_stage2_examples()
    make_renderer_wiring()
    make_freeu_tradeoff()
    make_freeu_examples()
    make_scheduler_gain()
    write_manifest()
    print(f"Wrote {len(OUTPUT_NAMES)} figures and manifest to {FIGURES}")


if __name__ == "__main__":
    main()
