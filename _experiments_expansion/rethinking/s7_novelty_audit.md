# S7 新颖性审计：调度器与频谱主张

**截止日期：** 2026-08-25。本文是文献与实验协议说明，不是实验结果。底层
arXiv API 审计于 2026-08-24 获取，并记录在
`SCHEDULER_SPECTRAL_NOVELTY_AUDIT_2026.md`。本文不授权任何 GPU 实验。

## 动机

S7 检验冻结 diffusion sampler 周围的有界校正。容易想到的主张是
“scheduler-consistent ancestral correction”或“spectral trajectory
improvement”，但这些说法范围过大。近期工作已经把 sampler、随机性、轨迹几何
和 latent spectrum 当作可设计对象。因此，本项目最多只能在这里检验一个窄的
因果问题：在固定的 SDXL/U-Net、scheduler 和计算预算下，所注册的校正是否产生
独立于现有控制的改进。文献中的机制不等于本项目已经验证的机制。

## 最接近的工作与新颖性边界

- **CPS (2509.05952)、Precise (2605.23522) 和 LC-GRPO
  (2608.05600)：** 已讨论 finite-step ODE/SDE mismatch、
  coefficient-preserving noise，以及 ODE Euler 加 Langevin correction。
  Ancestral noise 或随机插值不能称为新的 RL sampler。
- **SlerpFlow (2607.21326) 和 SGPS (2512.23232)：** 已报告 spherical
  velocity repair、缓存和 low-NFE trajectory-gradient correction。仅增加 trust
  cap 或一步 clean-latent update，不足以构成新颖性。
- **SPA (2607.22091)、SpectralDiT (2606.18765)、Frequency-Forcing
  (2604.20902)、SEGA (2605.22668) 和 SPAE (2608.01306)：** 已覆盖 FFT
  prior、timestep-conditioned low/high residual、wavelet coarse-to-fine
  forcing、frequency-aware attention 和 spectral latent adaptation。某个
  spectral band 或 schedule 本身不能成为贡献。
- **CFG distortion analysis (2602.00716)：** variance shrinkage 和
  negative-guidance window 已是已知控制；任何 CFG 或 energy 改动都必须同时
  提供 diversity 和 variance 证据。

## S7 设计约束

1. 必须分别运行并报告四种条件：原生
   `EulerAncestralDiscreteScheduler`；令 `sigma_up=0` 的 deterministic
   Euler；注册的 `mix` 插值；以及一种 RL rollout sampler（CPS、Precise 或
   LC-GRPO-style）。只有 `mix=1,sqrt` 是注册的精确 ancestral endpoint。
   如果没有 SDE 推导，中间 `mix` 只能作为 ablation。
2. 必须冻结并记录 installed `diffusers` version、scheduler config hash、
   timestep list、`sigma_up/down`、prediction type、CFG、initial-noise seed 和
   实际 U-Net call count。当前 provenance mismatch 也必须记录：运行环境使用
   `diffusers==0.32.1`，而 package declaration 要求 `~=0.21.4`。同时报告
   noise norm、latent moment、wall time、peak memory 和 reward variance。
   nominal step 数量相同不代表计算量相同。
3. 如果要提出 scheduler 相关主张，方法必须在 held-out prompt 上同时击败原生
   ancestral 和 deterministic Euler，而且不能输给 matched-NFE 的
   DPM-Solver++ 或 UniPC。否则只能把结果称为 sampler choice，不能称为新的
   correction。

## FRLA 约束

SARA、iREPA、sREPA、SGA 和 SSVAE 已覆盖 autocorrelation、cosine
self-similarity、Gram structure 和 local correlation。因此，FRLA 最多只能
检验一个固定、rank-compatible descriptor 与 scheduler-coordinate
clean-latent injection 的**交互作用**。development matrix 必须包含 no-op、
pointwise descriptor、full-Gram descriptor、shuffled token、每步 RMS 与 trust
cap 匹配的 random direction、Euclidean injection、MOG/DiffRGD geometry、SPA
或固定 DCT/wavelet control，以及 matched-compute extra-U-Net reference。

使用预注册的 difference-in-differences interaction，而不是单个分数增益：

```text
I = (FRLA - descriptor-only)
    - (scheduler-injection-only - no-op)
```

计算开销必须包括 feature hook、backward/FFT、reward call、wall time 和 memory。
只有 spectral、VIV 或 latent-moment 指标变好，不能证明机制成立；还必须有独立的
TOPIQ/HPS/CLIP、OCR/count/layout/detail、diversity 和 pixel-safety 证据。

## 停止规则

S7 必须先通过注册的 fixed-action gate：TOPIQ-NR delta `>= +0.005`、crossed
prompt/seed 95% interval 的下界高于零、经过 Holm correction 的 prompt
sign-flip significance、HPSv2/CLIP non-inferiority、有限数值输出、moment 和
trust-cap bounds，以及 clipping/saturation guards。

如果方法只击败 deterministic Euler，就把它归类为 ancestral/sampler effect。
如果 FRLA 没有正的 interaction，或任一独立证据未通过，就报告仍然成立的最简单
baseline，并停止 S8。在看到 held-out scores 后，不得再添加 frequency band、
scheduler blend、negative-guidance window、reward weight、distillation stage 或
RL controller。
