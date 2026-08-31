# FreeU 结构干预实验

## 为什么做这次实验

前面已经试过 TFSA、semantic transport（语义传输）和固定的 latent renderer（潜空间渲染器）。不过，在没有拿来调参数的那批数据上，它们都没有稳定变好。因此，当时没有理由继续训练 RL controller（强化学习控制器）。

这次换一个思路：不直接改 latent（潜变量），而是利用扩散模型现成的 U-Net 结构来调节特征。遵循的原则很简单：模型权重保持不变，只在模型原有的结构里做小范围干预。

这里用的是 FreeU（[Si et al., 2023](https://arxiv.org/abs/2309.11497)）。它分别调节 U-Net 主干特征和 skip feature（跳连特征）的权重。用大白话说，主干一路负责把整体画面整理出来；skip connection（跳跃连接）则把前面阶段抓到的细节直接传到后面。FreeU 做的事情，就是改变这两路信息各自占多大比重。FreeU 是已有的成熟基线，不是本项目提出的新方法，这里也不把它当成创新。

另外两篇关于 latent 表示的论文，帮助我们决定要加哪些保护检查：

- *Improving the Diffusability of Autoencoders*（[2502.14831](https://arxiv.org/abs/2502.14831)）指出，latent 里如果高频信息太多，去噪时就可能顾不上先把大结构做好、再补小细节。
- *EQ-VAE*（[2502.09509](https://arxiv.org/abs/2502.09509)）用 transformation-equivariance regularization（变换等变正则），让 latent 表示更简单、更规整。

但这两篇论文改的是 autoencoder（自动编码器）的训练方法。它们并不能说明：在 SDXL 推理时直接、不加区别地过滤 latent，就一定合理或有效。

## 实际改了什么

我们在 `AttentionGuidance/freeu.py` 里加入了经过检查的分段线性 FreeU schedule（参数随时间变化的表）。同一套 pipeline（生成流程）可以连续测试不同动作，而且每次都会清掉上一个动作留下的 U-Net 状态，不会互相干扰。

SDXL pipeline 只在 Stage 1 使用这些 schedule。每一对要比较的动作之间，都会重置 U-Net 里会变化的状态。评估程序会把实际用过的完整 schedule 记录下来，并支持两种写法：一种是从头到尾用同一个固定值；另一种是在几个时间点指定数值，时间点之间自动做线性变化。

后面的对照实验仍然使用同样的 FreeU 特征变换，但加了一条限制：变换前后，每个通道的 feature mean（特征均值）和 RMS（均方根）必须保持一样。可以把它们粗略理解为特征的“中心位置”和“整体强度”。这样才能看清分数变化究竟是因为结构信息重新分配，还是只是整体幅度、对比度变了。

我们还试过一个更直接的办法：scheduler（调度器）每走一步，就把当前 latent 的统计量硬拉回上一步的数值。这个办法在只用两个 prompt（提示词）的 smoke 测试（小规模冒烟测试）里就失败了，所有做过投影的图片都变成黑图或无效图，所以没有继续评分。

## 实验设置和结果

`outputs/freeu_conservative_search_v1` 包含 8 个动作、24 个 prompts（提示词）和 2 个 seeds（随机种子），一共 384/384 条完整记录。`outputs/freeu_moment_followup_v1` 规模相同，成对比较普通 FreeU 和保持 feature moment（特征统计量）的版本；这次运行对应 commit `29dcf0f`。

两次实验都用 30 个去噪 steps（步），在 Stage 1 生成 1024px 图片，然后进行严格评分。TOPIQ-NR 是主要指标；HPSv2 和 CLIP 用来观察人类偏好和文字匹配程度；clipping（像素截断）、saturation（饱和度）、contrast（对比度）、colorfulness（色彩丰富度）、sharpness（锐度）则用来防止取巧，或帮助定位问题。

统计时用了 prompt/seed 交叉 bootstrap CI（交叉自助法置信区间）、按 prompt 做的 sign-flip test（符号翻转检验），以及 Holm correction（Holm 多重比较校正）。说白了，就是检查所谓的提升能不能在不同 prompt 和 seed 上都稳定出现，而不是碰巧有几条样本占了便宜。

| 动作 | Δ TOPIQ-NR [95% CI] | Δ HPSv2 | Δ clipped | Δ saturation |
|---|---:|---:|---:|---:|
| 普通 FreeU，只调主干（backbone） | +0.008447 [+0.003323,+0.013826] | +0.005473 | +0.006656 | +0.018090 |
| 普通 FreeU，只在 early/low window（前期/低频时间段）作用 | +0.004131 [+0.001053,+0.007595] | +0.001488 | +0.001583 | +0.005257 |
| 保持 moment，只调主干（backbone） | +0.000609 [-0.001100,+0.002540] | +0.000506 | -0.000074 | -0.000469 |
| 保持 moment，只在 low window（低频时间段）作用 | +0.001445 [-0.000894,+0.003758] | -0.000346 | -0.000042 | -0.000032 |

最关键的结果是：普通 FreeU 的画质分数看起来变高了，但同时对比度、色彩丰富度、像素截断和饱和度也明显上升。两者分不开，所以更可能是画面被调得更强烈、更鲜艳，指标因此变高，并不代表画面结构真的变好了。

强制保持 feature moment 后，这些副作用基本没了，但原先的画质提升也没了。在 clipping 增量不超过 `+0.001`、saturation 增量不超过 `+0.005` 的前提下，没有任何动作达到事先登记的 TOPIQ `+0.005` 门槛。保持 moment 的版本也没有稳定提高 CLIP 或 ImageReward。

## 最终决定和下一步

这是开发阶段得到的负面结果：这些候选动作已经在开发数据上被淘汰。它不是在独立的 validation split（验证集）或 test split（测试集）上得出的最终验证结论。

FreeU 仍然保留，主要有两个用途：第一，作为一个较强的结构干预基线；第二，提醒后面的实验防范“指标投机”，也就是分数涨了，但涨的只是颜色、对比度等表面效果。不过，FreeU 这一整类动作到这里就停止了：不再继续调 FreeU scale（强度）或作用 window（时间段），也不再用它做 distillation（蒸馏）或 RL。

下一个假设必须从别的、而且和 scheduler（调度器）逻辑一致的地方获得更新，例如 equivariance residual（等变残差）或 low-frequency consistency residual（低频一致性残差）。第一关是在一套重新登记、和旧 prompts（提示词）完全不重复的数据上，用固定动作超过 `no_freeu`，并且同时通过所有保护指标。

如果连这样的固定动作也失败，期刊扩展就应该如实报告这个负面结果，而不是再加一个 RL controller。

复现实验的命令和 scorer（评分器）定义见 `eval-pipeline/README.md`。生成的图片和评分结果仍放在不会被 Git 跟踪的 `outputs/` 目录中。
