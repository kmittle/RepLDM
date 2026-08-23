# RepLDM eval-pipeline (quantitative measurement instrument)

Decoupled two-stage harness that feeds all Phase-1 experiments (constant-scale
sweep / CMA-ES static baseline / per-prompt oracle gap) and, later, the DRaFT
reward. Generation and scoring live in **different conda envs** and communicate
through PNGs + a JSON manifest on disk, so scoring is re-runnable and extensible
without regenerating images.

```
generate.py  [env repldm]   prompt x seed x scale  ->  images/*.png + images/*.json
                                                         -> manifest.jsonl
score.py     [env sana_cby]  manifest + PNGs         ->  scores.jsonl
aggregate.py [env repldm]    manifest + scores       ->  eval_results.csv + analysis
```

Why decoupled: ImageReward / HPSv2 / CLIP live only in `sana_cby` (py3.11);
the SDXL pipeline needs `repldm` (py3.9, diffusers 0.21.4). See the project-env memory.

## 1. Generate (env `repldm`)

Stage-1-only (1024², so Stage 2 is skipped), constant per-step guidance scale.
`scale=0` is the exact no-guidance baseline.

```bash
conda run -n repldm python eval-pipeline/generate.py \
  --devices 6,7 \
  --prompts eval-pipeline/prompts/eval_v1.csv \
  --out_dir outputs/exp1.1_scale_sweep/pilot \
  --scales 0,0.001,0.002,0.003,0.005 \
  --seeds 0,42,123 \
  --low_vram          # offload to CPU between phases; use when GPUs are busy
```
- One worker per `--device`, shared task queue, resume-safe (existing PNGs skipped).
- `--guidance_scale` (CFG, default 7.5) and `--power_calibrate` (0) are held fixed
  across the sweep — they are confounds, keep them constant.
- Pick currently-free GPUs (`nvidia-smi`); the box is shared/saturated.

## 2. Score (env `sana_cby`) — config-driven, call the env python directly

One-time weight pre-stage (downloads CLIP/HPSv2/aesthetic to **shared caches**, ~7GB;
decoupled from Sana — same upstream packages, weights in `~/.cache/{clip,hpsv2,aesthetic}`):
```bash
/home/bycao/miniforge3/envs/sana_cby/bin/python eval-pipeline/prestage_weights.py
```

```bash
/home/bycao/miniforge3/envs/sana_cby/bin/python eval-pipeline/score.py \
  --run_dir outputs/exp1.1_scale_sweep/pilot --device cuda:0
```
`configs/eval_common.yaml` lists which metrics run; each is a self-contained module under
`scorers/` (DECOUPLED — upstream pip packages, **never Sana imports/copies**). Weights are
validated up front, so a missing metric is skipped (not a crash). Resume is **additive**:
re-running adds new metric columns to existing rows without redoing old ones. `scores.jsonl`
columns:
- `imagereward` — full-image IR (224 downsample → color/layout, not detail).
- `patch_ir_mean/std/n` — IR over native-res 224 crops (detail-sensitive; §13.2). **[RepLDM-unique]**
- `clip_cosine`, `clipscore` — CLIP-Score, canonical `2.5·max(cos,0)` @ ViT-B/32 + raw cosine — alignment witness.
- `hpsv2` — HPSv2 v2.1 (ViT-H-14) human preference.
- `aesthetic` — LAION aesthetic (CLIP ViT-L/14 + MLP), ~[1,10].
- `colorfulness, laplacian_sharpness, mean_saturation, clipped_fraction, contrast_std`
  — weightless decorrelated reward-hacking witnesses (§13.5). **[RepLDM-unique]**
> §13.5 caveat: clip/hpsv2/aesthetic are all CLIP-family@224 → mutually correlated, **NOT**
> independent detail witnesses; pair them with patch-IR + pixel sharpness.
Add `geneval`/`dpg`/`fid` later by dropping a module in `scorers/` + listing it in the config.

## 3. Analyze (env `repldm`)

```bash
conda run -n repldm python eval-pipeline/aggregate.py --run_dir outputs/exp1.1_scale_sweep/pilot
```
Prints the sharpened Exp-1.1 go/no-go (§13.4): mean-IR-vs-scale spread vs seed
noise; per-prompt **interior-optimum** test (monotone-in-scale ⇒ "tune the clamp",
not "learn guidance"); argmax-scale heterogeneity (content-adaptivity); global-IR
vs patch-IR argmax; corr(IR, colorfulness/sharpness). Saves `eval_results.csv` +
`analysis.png`.

## 4. Visualize (env with numpy + Pillow + matplotlib)

```bash
python eval-pipeline/visualize.py --run_dir outputs/exp1.1_scale_sweep/pilot --seed 0
```
Saves `figs/scale_sweep_montage.png` (one prompt/bucket × all scales, IR-annotated,
green box = per-row IR argmax, red IR = >10% clipped) and `figs/action_visibility.png`
(normalized witnesses-vs-IR + raw IR vs seed-noise band). These are the figures embedded
in `EXPERIMENT_RESULTS.md`.

## Layout
```
configs/eval_common.yaml   which metrics + params + fixed sampling recipe
scorers/                   one self-contained Scorer per metric (register via @register_metric)
  base.py                  REGISTRY + Scorer base + weights_status()
  imagereward_scorer.py    global-IR + patch-IR        pixel_scorer.py   weightless witnesses
  clip_scorer.py           CLIP-Score                  hps_scorer.py     HPSv2
  aesthetic_scorer.py      LAION aesthetic
generate.py  score.py  aggregate.py  metrics.py  prestage_weights.py  prompts/
```

## Notes / not-yet
- `geneval` / `dpg` / `fid` are pluggable but not yet implemented (heavy: detection/VQA
  models + benchmark prompt sets + large samples; lower priority for a guidance-scale study).
- The differentiable reward path for DRaFT is `ImageReward.score_gard` (this harness
  uses the non-diff `score()` — fine for measurement).
- `prompts/eval_v1.csv` is a 12-prompt × 6-bucket starter set; expand to ~50 for full Exp-1.1.
```
