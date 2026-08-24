# FreeU 结构干预实验

## 为什么做这个实验

前面已经试过 TFSA、semantic transport 和固定 latent renderer，但它们在没有参与调参的数据上都没有表现出可靠的提升空间。因此，当时没有理由继续训练 RL controller。

这次实验换了一条路：不再直接修改 latent，而是利用扩散模型本来就有的 U-Net 结构来调整特征。这个思路来自论文版方法更宽泛的原则，也就是“模型权重保持冻结，只在模型原有结构里做干预”。

这里使用的 FreeU（[Si et al., 2023](https://arxiv.org/abs/2309.11497)）会分别调整 U-Net 主干特征和 skip feature 的权重。大白话说，主干负责逐步整理整体内容，skip connection 会把较早阶段的细节直接送到后面；FreeU 改的是这两路信息各占多大分量。FreeU 是已有的成熟基线，不是本项目的新方法，也不在这里宣称创新。

另外两篇 latent representation 论文决定了这次实验必须设置哪些保护措施：

- *Improving the Diffusability of Autoencoders*（[2502.14831](https://arxiv.org/abs/2502.14831)）认为，latent 中高频信息过多，会打乱从大结构到小细节的正常去噪过程。
- *EQ-VAE*（[2502.09509](https://arxiv.org/abs/2502.09509)）通过 transformation-equivariance regularization，让 latent 表示变得更简单、更规整。

但这两项工作改的是 autoencoder 的训练方式。它们不能证明：在 SDXL 推理过程中直接、无差别地过滤 latent，就一定合理或有效。

## 具体改了什么

`AttentionGuidance/freeu.py` 新增了经过检查的分段线性 FreeU schedule。它支持在同一个 pipeline 中连续运行不同动作，不会让上一个动作留下的 U-Net 状态影响下一个动作。

SDXL pipeline 只在 Stage 1 使用这些 schedule。每组配对动作之间，都会重置 U-Net 中可变的状态。评估程序会记录实际使用的完整 schedule，同时支持两种动作：一种是全程使用固定值，另一种是在若干时间点设值、在点与点之间线性变化。

后续对照实验仍使用同样的 FreeU 特征变换，但额外要求变换前后每个通道的 feature mean 和 RMS 保持一致。这里可以把 mean 和 RMS 简单理解成特征的整体中心和整体强度。这样做是为了分清：指标变化到底来自“结构信息被重新分配”，还是只来自整体幅度、对比度发生变化。

还试过一个更直接的办法：scheduler 每走一步，都把当前 latent 的统计量强行拉回上一步。但这个方案在 two-prompt smoke 中直接失败，所有经过投影的图片都变成黑图或无效图，所以没有进入评分。

## 实验设置与结果

`outputs/freeu_conservative_search_v1` 包含 8 个动作、24 个 prompts 和 2 个 seeds，共有 384/384 条完整记录。`outputs/freeu_moment_followup_v1` 使用同样的实验规模，成对比较普通 FreeU 和保持 feature moment 的版本；该次运行对应 commit `29dcf0f`。

两次实验都使用 30 个去噪 steps，在 Stage 1 生成 1024px 图像，并执行严格评分。TOPIQ-NR 是主指标；HPSv2 和 CLIP 用来观察人类偏好与文本对齐；clipping、saturation、contrast、colorfulness 和 sharpness 用作保护指标或辅助诊断。

统计分析使用 prompt/seed 交叉 bootstrap CI、按 prompt 的 sign-flip test 和 Holm correction。简单来说，这些检验是在确认提升是否能跨 prompt 和 seed 稳定出现，而不是刚好碰上几个有利样本。

| 动作 | Δ TOPIQ-NR [95% CI] | Δ HPSv2 | Δ clipped | Δ saturation |
|---|---:|---:|---:|---:|
| 普通 FreeU，只调整 backbone | +0.008447 [+0.003323,+0.013826] | +0.005473 | +0.006656 | +0.018090 |
| 普通 FreeU，只作用于 early/low window | +0.004131 [+0.001053,+0.007595] | +0.001488 | +0.001583 | +0.005257 |
| 保持 moment，只调整 backbone | +0.000609 [-0.001100,+0.002540] | +0.000506 | -0.000074 | -0.000469 |
| 保持 moment，只作用于 low window | +0.001445 [-0.000894,+0.003758] | -0.000346 | -0.000042 | -0.000032 |

最重要的结果是：普通 FreeU 的画质分数看起来提高了，但这个提升和明显增加的 contrast、colorfulness、clipping、saturation 分不开。也就是说，它可能主要通过把画面变得更强烈、更鲜艳来提高指标，而不是真的改善了结构。

当实验强制保持 feature moment 时，上述副作用基本消失，但原来的画质提升也一起消失了。没有任何动作能在满足 clipping 增量不超过 `+0.001`、saturation 增量不超过 `+0.005` 的同时，达到预先登记的 TOPIQ `+0.005` 门槛。保持 moment 的版本也没有稳定提高 CLIP 或 ImageReward。

## 最终决定与下一步

这是一次开发阶段的负面结果：候选动作已经在开发数据上被淘汰。它不是在独立 validation split 或 test split 上得到的最终验证结论。

FreeU 仍然保留，作用有两个：一是作为较强的结构干预基线；二是提醒后续实验注意指标投机，也就是“分数提高了，但提高的只是颜色、对比度等表面属性”。不过，这一整类动作到此关闭，不再继续调整 FreeU scale 或作用 window，也不再基于它做 distillation 或 RL。

下一个假设必须从别的、并且与 scheduler 一致的来源取得更新，例如 equivariance residual 或 low-frequency consistency residual。它首先要在一套新登记、与旧 prompts 完全不重合的数据上，用固定动作胜过 `no_freeu`，同时通过所有保护指标。

如果这样的固定动作仍然失败，期刊扩展应该如实报告负面结果，而不是再加一个 RL controller。

复现实验的命令和 scorer 定义见 `eval-pipeline/README.md`。生成图片和评分结果仍放在不会被 Git 跟踪的 `outputs/` 下。
