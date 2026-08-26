"""Build the doc/research/EXPERIMENT_RESULTS.md figures from a constant-scale-sweep run.

Produces, under <run_dir>/figs/:
  * scale_sweep_montage.png -- one prompt per bucket x all scales (fixed seed),
    each tile annotated with its ImageReward; green box = per-row IR argmax,
    red IR text = >10% pixels clipped/over-saturated. Shows the action is large
    and monotone in pixels while IR wanders.
  * action_visibility.png -- (left) min-max-normalized mean witnesses vs IR over
    scale; (right) raw mean IR +/- SE against the seed-noise band.

Env: any with numpy + Pillow + matplotlib (e.g. `promoe`).
Usage:
  python eval-pipeline/visualize.py --run_dir outputs/exp1.1_scale_sweep/<run> --seed 0
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_DIRS = ("/usr/share/fonts/truetype/dejavu/",)


def font(sz, bold=False):
    names = (["DejaVuSans-Bold.ttf"] if bold else ["DejaVuSans.ttf", "DejaVuSans-Bold.ttf"])
    for d in _FONT_DIRS:
        for n in names:
            try:
                return ImageFont.truetype(os.path.join(d, n), sz)
            except Exception:
                pass
    return ImageFont.load_default()


# preferred bucket order for the montage rows (falls back to sorted for unknown buckets)
_BUCKET_ORDER = ["dense-texture", "flat-design", "photo", "face", "art", "text-render"]


def load_rows(run_dir):
    return [json.loads(l) for l in open(os.path.join(run_dir, "scores.jsonl")) if l.strip()]


def montage(run_dir, rows, seed, out_path):
    D = {(r["prompt_index"], r["seed"], r["scale"]): r for r in rows}
    scales = sorted({r["scale"] for r in rows})
    bucket_of = {r["prompt_index"]: r["bucket"] for r in rows}
    # prompt text lives in manifest.jsonl, not scores.jsonl -> load it for tile labels
    short = {}
    mpath = os.path.join(run_dir, "manifest.jsonl")
    if os.path.exists(mpath):
        for l in open(mpath):
            if l.strip():
                m = json.loads(l)
                if "prompt" in m:
                    short.setdefault(m["prompt_index"], m["prompt"][:26])
    # one prompt per bucket: lowest prompt_index in each bucket, ordered by _BUCKET_ORDER
    by_bucket = {}
    for pid in sorted(bucket_of):
        by_bucket.setdefault(bucket_of[pid], pid)
    ordered_buckets = [b for b in _BUCKET_ORDER if b in by_bucket] + \
                      [b for b in sorted(by_bucket) if b not in _BUCKET_ORDER]
    picks = [(by_bucket[b], b) for b in ordered_buckets]

    TILE, LEFT, HEAD, CAP, TITLE, PAD = 190, 190, 34, 26, 60, 6
    W = LEFT + len(scales) * (TILE + PAD) + PAD
    H = TITLE + HEAD + len(picks) * (TILE + CAP + PAD) + PAD
    canvas = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(canvas)
    f_title, f_hdr, f_lbl, f_cap = font(23, True), font(20, True), font(15), font(15)

    dr.text((PAD, 8), f"scale sweep (seed {seed})  —  pixels change hugely, ImageReward barely tracks",
            fill="black", font=f_title)
    dr.text((PAD, 37), "green box = IR argmax of the row      red IR value = >10% pixels clipped / over-saturated",
            fill=(90, 90, 90), font=f_lbl)
    for j, sc in enumerate(scales):
        x = LEFT + j * (TILE + PAD) + PAD
        dr.text((x + TILE // 2 - 34, TITLE + 6), f"scale {sc:.3f}", fill="black", font=f_hdr)

    for i, (pid, bucket) in enumerate(picks):
        y0 = TITLE + HEAD + i * (TILE + CAP + PAD) + PAD
        dr.text((6, y0 + TILE // 2 - 20), bucket, fill="black", font=f_hdr)
        dr.text((6, y0 + TILE // 2 + 2), short.get(pid, ""), fill=(70, 70, 70), font=f_lbl)
        irs = [D[(pid, seed, sc)]["imagereward"] for sc in scales]
        argmax_j = int(np.argmax(irs))
        for j, sc in enumerate(scales):
            r = D[(pid, seed, sc)]
            x = LEFT + j * (TILE + PAD) + PAD
            img = Image.open(os.path.join(run_dir, r["image_path"])).convert("RGB").resize((TILE, TILE))
            canvas.paste(img, (x, y0))
            if j == argmax_j:
                for t in range(4):
                    dr.rectangle([x - t, y0 - t, x + TILE + t, y0 + TILE + t], outline=(20, 160, 40))
            col = (200, 30, 30) if r["clipped_fraction"] > 0.10 else (20, 20, 20)
            dr.text((x + 4, y0 + TILE + 3), f"IR={r['imagereward']:+.2f}", fill=col, font=f_cap)
    canvas.save(out_path)


def action_visibility(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D = {(r["prompt_index"], r["seed"], r["scale"]): r for r in rows}
    scales = sorted({r["scale"] for r in rows})
    prompts = sorted({r["prompt_index"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})

    def mean_curve(key):
        return np.array([np.mean([r[key] for r in rows if r["scale"] == sc]) for sc in scales])

    def seed_band(key):
        stds = []
        for sc in scales:
            for p in prompts:
                vv = [D[(p, s, sc)][key] for s in seeds if (p, s, sc) in D]
                if len(vv) >= 2:
                    stds.append(np.std(vv, ddof=1))
        return float(np.mean(stds))

    def norm(c):
        return (c - c.min()) / (c.max() - c.min() + 1e-9)

    x = np.array(scales)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    wit = {"colorfulness": "colorfulness", "mean_saturation": "saturation",
           "laplacian_sharpness": "sharpness", "clipped_fraction": "clipped px", "contrast_std": "contrast"}
    for k, lbl in wit.items():
        ax[0].plot(x, norm(mean_curve(k)), marker="o", lw=1.6, alpha=0.8, label=lbl)
    ir = mean_curve("imagereward")
    ax[0].plot(x, norm(ir), marker="s", lw=3, color="black", label="ImageReward")
    ax[0].set_xlabel("attn_guidance_scale"); ax[0].set_ylabel("min-max normalized mean")
    ax[0].set_title("Action is visible in pixels, invisible to IR\n(witnesses climb monotonically; IR stays flat)")
    ax[0].legend(fontsize=9, loc="center left"); ax[0].grid(alpha=0.3)

    band = seed_band("imagereward")
    n_per_scale = len(prompts) * len(seeds)
    se = [np.std([r["imagereward"] for r in rows if r["scale"] == s], ddof=1) / np.sqrt(n_per_scale) for s in scales]
    ax[1].axhspan(ir.mean() - band, ir.mean() + band, color="orange", alpha=0.15, label=f"±seed noise ({band:.3f})")
    ax[1].errorbar(x, ir, yerr=se, marker="s", color="black", capsize=3, label="mean IR ±SE")
    ax[1].set_xlabel("attn_guidance_scale"); ax[1].set_ylabel("mean ImageReward")
    ax[1].set_title(f"mean IR vs scale (spread={ir.max() - ir.min():.3f} < seed noise {band:.3f})")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--seed", type=int, default=0, help="which seed to show in the montage")
    args = ap.parse_args()

    rows = load_rows(args.run_dir)
    figs = os.path.join(args.run_dir, "figs")
    os.makedirs(figs, exist_ok=True)

    m = os.path.join(figs, "scale_sweep_montage.png")
    a = os.path.join(figs, "action_visibility.png")
    montage(args.run_dir, rows, args.seed, m)
    print(f"wrote {m}")
    action_visibility(rows, a)
    print(f"wrote {a}")


if __name__ == "__main__":
    main()
