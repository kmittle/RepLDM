"""Join manifest + scores and run the Exp-1.1 analysis (env: `repldm` or any
env with pandas/numpy/matplotlib).

Produces eval_results.csv and, for the constant-scale sweep, the sharpened
go/no-go diagnostics from EXPERIMENT_PLAN §13.4:
  * mean ImageReward vs scale with a seed-variance band (the original Exp-1.1 plot);
  * the *interior-optimum* test: per prompt, is IR(scale) non-monotone with an
    argmax scale > 0 in the swept interior? Monotone-increasing for (nearly) all
    prompts  =>  "tune the clamp", not "learn guidance"  => weak motivation.
  * heterogeneity of the per-prompt argmax scale (content-adaptivity evidence);
  * global-IR vs patch-IR argmax (does the detail-sensitive reward peak where the
    224px global reward is monotone? §13.2);
  * corr(IR, colorfulness/sharpness) -- is an IR gain just a saturation push?

Usage:
  conda run -n repldm python eval-pipeline/aggregate.py --run_dir outputs/exp1.1_scale_sweep/pilot
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def per_prompt_argmax(curve: pd.Series):
    """curve: index=scale, value=mean IR. Returns (argmax_scale, is_interior, is_monotone_increasing)."""
    scales = list(curve.index)
    vals = curve.values
    amax = scales[int(np.argmax(vals))]
    is_interior = (amax != scales[0]) and (amax != scales[-1])
    is_mono_inc = bool(np.all(np.diff(vals) >= -1e-9))
    return amax, is_interior, is_mono_inc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--reward", default="imagereward", help="which reward column to analyze (imagereward|patch_ir_mean)")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    manifest = pd.DataFrame(load_jsonl(os.path.join(args.run_dir, "manifest.jsonl")))
    scores = pd.DataFrame(load_jsonl(os.path.join(args.run_dir, "scores.jsonl")))
    df = manifest.merge(scores[[c for c in scores.columns if c not in ("prompt_index", "bucket", "seed", "scale", "image_path")]],
                        on="id", how="inner")
    out_csv = os.path.join(args.run_dir, "eval_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"joined {len(df)} rows -> {out_csv}\n")

    rcol = args.reward if args.reward in df.columns else "imagereward"

    # --- overall: mean reward vs scale with seed band ---
    g = df.groupby("scale")[rcol]
    overall = pd.DataFrame({"mean": g.mean(), "std": g.std(), "n": g.count()})
    print(f"=== {rcol} vs scale (mean ± std across all prompts*seeds) ===")
    print(overall.to_string(float_format=lambda x: f"{x:.4f}"))
    seed_band = df.groupby(["prompt_index", "scale"])[rcol].mean().groupby("scale").std().mean()
    spread = overall["mean"].max() - overall["mean"].min()
    print(f"\nmean-curve spread across scale: {spread:.4f} | typical per-prompt seed std: {seed_band:.4f}")
    print(f"  -> {'SIGNAL: spread exceeds seed noise' if spread > seed_band else 'NO SIGNAL: spread within seed noise (RISK-1)'}\n")

    # --- per-prompt interior-optimum test (sharpened Exp-1.1 gate, §13.4) ---
    rows = []
    for pid, sub in df.groupby("prompt_index"):
        curve = sub.groupby("scale")[rcol].mean()
        amax, interior, mono = per_prompt_argmax(curve)
        rows.append({"prompt_index": pid, "bucket": sub["bucket"].iloc[0],
                     "argmax_scale": amax, "interior": interior, "monotone_increasing": mono})
    pp = pd.DataFrame(rows)
    n = len(pp)
    print(f"=== per-prompt argmax of {rcol} (interior-optimum gate) ===")
    print(pp.to_string(index=False))
    print(f"\ninterior optimum: {pp['interior'].sum()}/{n} prompts | "
          f"monotone-increasing: {pp['monotone_increasing'].sum()}/{n} prompts")
    print(f"argmax-scale heterogeneity: {pp['argmax_scale'].nunique()} distinct values across {n} prompts "
          f"(values: {sorted(pp['argmax_scale'].unique())})")
    if pp["monotone_increasing"].mean() > 0.8:
        print("  -> WARNING: reward ~monotone in scale for most prompts => 'tune the clamp', weak adaptivity motivation.")
    elif pp["interior"].mean() >= 0.5:
        print("  -> interior optima are common => an adaptive optimum exists; heterogeneity supports content-adaptivity.")

    # --- PAIRED analysis + seed-CV oracle gap (the decision-relevant numbers) ---
    # Generation uses the SAME seed across scales for a fixed (prompt,seed), so the
    # initial noise is identical and scale is the only difference => paired design.
    # The unpaired std above is dominated by prompt difficulty; the paired delta
    # IR(scale)-IR(0) is the correct RISK-1 test.
    piv = df.pivot_table(index=["prompt_index", "seed"], columns="scale", values=rcol)
    sc = sorted(piv.columns)
    base = sc[0]
    print(f"\n=== PAIRED delta {rcol}(scale) - {rcol}({base}) over {len(piv)} (prompt,seed) pairs ===")
    for s in sc[1:]:
        d = (piv[s] - piv[base]).dropna()
        se = d.std() / np.sqrt(len(d))
        sig = "*" if abs(d.mean()) > 2 * se else " "
        print(f"  scale {s:.4f}: mean Δ = {d.mean():+.4f} ± {se:.4f} (2se){sig}")
    print("  (* = paired Δ exceeds 2 standard errors => guidance moves IR beyond paired noise)")

    seeds = sorted(df["seed"].unique())
    if len(seeds) >= 2:
        gs, ga = [], []  # held-out gains: best-global-static, per-prompt-adaptive
        for ho in seeds:
            tr = piv[piv.index.get_level_values("seed") != ho]
            te = piv[piv.index.get_level_values("seed") == ho]
            best_global = tr.mean(axis=0).idxmax()                 # pick on train seeds
            gs.append(float((te[best_global] - te[base]).mean()))  # eval on held-out seed
            best_pp = tr.groupby(level="prompt_index").mean().idxmax(axis=1)  # per-prompt best scale
            ga.append(float(np.mean([row[best_pp.get(pid, base)] - row[base]
                                     for (pid, _sd), row in te.iterrows()])))
        gs, ga = np.array(gs), np.array(ga)
        print(f"\n=== seed-CV oracle gap (leave-one-seed-out, held-out gain over no-guidance) ===")
        print(f"  best GLOBAL static scale  : Δ{rcol} = {gs.mean():+.4f} ± {gs.std():.4f}")
        print(f"  per-PROMPT adaptive scale : Δ{rcol} = {ga.mean():+.4f} ± {ga.std():.4f}")
        print(f"  ORACLE GAP (adaptive - static) = {(ga - gs).mean():+.4f} ± {(ga - gs).std():.4f}")
        print("  -> gap is the content-adaptivity prize on held-out seeds; ~0 within noise => weak motivation (pilot: 3 seeds, noisy).")

    # --- patch-IR vs global-IR (detail-sensitivity, §13.2) ---
    if "patch_ir_mean" in df.columns and rcol == "imagereward":
        gp = df.groupby("scale")["patch_ir_mean"].mean()
        gi = df.groupby("scale")["imagereward"].mean()
        print(f"\n=== global-IR vs patch-IR argmax over scale ===")
        print(f"global-IR argmax scale: {gi.idxmax():.4f} (monotone_inc={bool(np.all(np.diff(gi.values)>=-1e-9))})")
        print(f"patch-IR  argmax scale: {gp.idxmax():.4f} (monotone_inc={bool(np.all(np.diff(gp.values)>=-1e-9))})")

    # --- reward-hacking witnesses ---
    # The valid signal is WITHIN-prompt: as scale rises on the SAME prompt, does the
    # reward co-move with saturation/colorfulness? Raw correlation is confounded by
    # large between-prompt baseline differences, so we report per-prompt-demeaned.
    def within_corr(a, b):
        da = df.groupby("prompt_index")[a].transform(lambda s: s - s.mean())
        db = df.groupby("prompt_index")[b].transform(lambda s: s - s.mean())
        if da.std() < 1e-12 or db.std() < 1e-12:
            return float("nan")
        return float(da.corr(db))

    print("\n=== reward-hacking check: corr(reward, witness) ===")
    print("    within = per-prompt-demeaned (valid signal); raw = confounded by between-prompt differences")
    for w in ("colorfulness", "laplacian_sharpness", "mean_saturation", "clipped_fraction", "patch_ir_mean"):
        if w in df.columns:
            print(f"  {rcol} vs {w:20s} within={within_corr(rcol, w):+.3f}  raw={df[rcol].corr(df[w]):+.3f}")

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(13, 5))
            ax[0].errorbar(overall.index, overall["mean"], yerr=overall["std"], marker="o", capsize=3, label="all prompts")
            for pid, sub in df.groupby("prompt_index"):
                c = sub.groupby("scale")[rcol].mean()
                ax[0].plot(c.index, c.values, alpha=0.3, lw=1)
            ax[0].set_xlabel("attn_guidance_scale"); ax[0].set_ylabel(rcol)
            ax[0].set_title(f"{rcol} vs scale (band=seed std; thin=per-prompt)"); ax[0].legend()
            ax[1].hist(pp["argmax_scale"].astype(str), bins=len(pp["argmax_scale"].unique()))
            ax[1].set_xlabel("per-prompt argmax scale"); ax[1].set_ylabel("# prompts")
            ax[1].set_title("argmax-scale heterogeneity")
            fig.tight_layout()
            fig_path = os.path.join(args.run_dir, "analysis.png")
            fig.savefig(fig_path, dpi=120)
            print(f"\nfigure -> {fig_path}")
        except Exception as e:
            print(f"[warn] plotting failed: {e}")


if __name__ == "__main__":
    main()
