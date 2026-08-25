# FRLA 关系型潜变量算子（仅方案）

## 状态与动机

这是一个预注册的后续假设，不是实验结果。只有 S7 scheduler-correction gate
得到结论后，才允许运行该方案。此前 TFSA、fixed renderer 和 FreeU 搜索没有提供
满足保护条件的 headroom。因此，本方案更换 update 的来源，而不是继续搜索 scale
或直接加入 RL controller。

方案名称是 **Frozen Relational Latent Alignment (FRLA)**。iREPA
([2512.10794](https://arxiv.org/abs/2512.10794))、SARA
([2503.08253](https://arxiv.org/abs/2503.08253))、sREPA
([2605.16949](https://arxiv.org/abs/2605.16949)) 和 *Diffusing in the Right
Space* ([2606.03578](https://arxiv.org/abs/2606.03578)) 提示 spatial
self-similarity 和 relational structure 可能比单一 frequency statistic 更稳定。
这只是文献启发，不表示 FRLA 已在 RepLDM 或 SDXL Euler 上得到验证。FRLA 与
RepLDM 只有思想上的软联系：两者都重新编排冻结模型的原生 latent trajectory，
但 FRLA 不是 TFSA。

LaRender ([2508.07647](https://arxiv.org/abs/2508.07647)) 和 SATeCo
([2403.17000](https://arxiv.org/abs/2403.17000)) 构成额外的新颖性边界：前者使用
object-conditioned rendering，后者使用 feature adapter；FRLA 没有 object mask、
prompt branch 或 second denoiser。DUNE
([2607.09753](https://arxiv.org/abs/2607.09753))、DIAMOND
([2602.00883](https://arxiv.org/abs/2602.00883))、InfSplign
([2512.17851](https://arxiv.org/abs/2512.17851)) 和 PMG/LGDM
([2506.00327](https://arxiv.org/abs/2506.00327)) 已覆盖 frozen-U-Net feature
intervention 或 clean-latent gradient correction。SSVAE
([2512.05394](https://arxiv.org/abs/2512.05394)) 也研究 local correlation
regularization。这些工作是直接 baseline，也是新颖性限制：FRLA 不能把 gradient、
tangent projection、local relation loss 或 frozen manipulation 本身称为新贡献。

Covariance 在本方案中只是 deterministic metric。由于已有工作报告
covariance-aware noise 会损害 VAE latent diffusion，本方案不使用这种噪声。
SGA ([2605.20808](https://arxiv.org/abs/2605.20808)) 和 SARA
([2503.08253](https://arxiv.org/abs/2503.08253)) 还排除了“首个 relational 或
Gram alignment”的主张。如果实验通过，最多只能讨论固定、rank-compatible
local descriptor 与其有界 scheduler-consistent update 的组合；当前文档不证明
该组合有效或新颖。

## 固定算子

在普通 conditional U-Net forward 中，捕获一个 decoder feature map 并执行
detach。用确定性方法把它降到四个 channel，再把该 feature 和
`pred_original_sample` 都 resize 到 `16 x 16` grid。对五个固定 lag
`(1,0),(0,1),(1,1),(2,0),(0,2)` 计算 local cosine autocorrelation，然后只对
latent 做一步 gradient update，以减小两者的 squared discrepancy。

Channel reduction、lag boundary handling、step size `eta` 和 trust-cap ratio
必须在评分前固定到 YAML 中。降维使用记录过 seed 的固定 projection，不使用 learned
reduction。由于 pipeline 全局处于 `no_grad`，只能在 cloned、detached clean latent
和 relation loss 周围使用 `torch.enable_grad()`；不得通过 U-Net
backpropagate。

在 scheduler step 后注入有界 clean-latent residual，然后施加
scheduler-update trust cap 和 shrinkage channel-covariance retraction。这是
covariance control，不能据此声称 DiffRGD 的 isotropic sphere 实际是
anisotropic。Full token Gram 版本只能作为 ablation：否则，四 channel latent
和更宽的 feature map 存在 rank mismatch。必须记录 backward time、FLOPs、peak
memory、residual norm、descriptor error 和 latent-prior drift。

该 gradient step 是实现对照，不是新的 optimizer：DiffRGD、MPGD 和 DPS 已覆盖
test-time latent-gradient action。FRLA 只检验这条 U-Net-native relational
direction 是否存在固定的 quality headroom。如果存在，后续可能研究一个预测或蒸馏
该方向的有界小型 renderer；当前文档不授权该模型。

## 证伪门槛

在新的 prompt-disjoint split 上，固定相同的 seed、CFG、scheduler、NFE 和
initial noise。必须比较 no-op、conference TFSA、SPA、FreeU、FRLA、shuffled
feature token、feature-gradient/no-gradient ablation、pointwise feature
matching、relation-preserving tangent、isotropic shell、Mahalanobis shell、
每步 RMS 与 trust cap 匹配的 dummy latent-gradient control、matched-compute
extra-U-Net reference，以及 same-wall-time no-op/FFT control。

通过条件为：TOPIQ-NR delta `>= +0.005`；crossed-bootstrap CI 下界高于零；
经过 Holm correction 的 prompt sign-flip `p < 0.05`；HPSv2/CLIP
non-inferiority；并通过 clipping/saturation guards (`+0.001`/`+0.005`)。还必须
报告 OCR/count/spatial accuracy、LPIPS/diversity、latent moment、DCT/SEC、
runtime 和 qualitative crop。

如果 FRLA 无法通过这个 fixed gate，就关闭 representation route，不允许 parameter
search、distillation 或 RL。如果通过，必须先在 development data 上冻结一个
action，之后才能进入 validation 或 learned controller。DiffRGD
([2606.28417](https://arxiv.org/abs/2606.28417)) 和 LatSearch
([2603.14526](https://arxiv.org/abs/2603.14526)) 仍是计算与新颖性对照，不能作为
FRLA 的新颖性主张。VIV 只作为从 Flow Matching 得到、用于 SDXL Euler 的探索性
diagnostic；SEC/LNC/LDS/CDS/SRSS 是主要的 scheduler-agnostic structural
diagnostics。
