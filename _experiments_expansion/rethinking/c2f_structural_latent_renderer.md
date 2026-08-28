# Coarse-to-Fine Structural Latent Renderer

## 当前状态和边界

这是一个独立候选的预注册前设计，不是已完成实验，也不授权 GPU 训练、生成、评分或
RL。它不能用来挽救旧 LR-1、scheduler-native frequency screen 或当前 adaptive
oracle 的结果。当前 adaptive oracle 的 165 条工程轨迹仍应先按原授权完成和盲审计。

## 为什么提出这个方案

前面的固定 TFSA、频带、FreeU、sampler correction 和 latent basis 大多没有稳定
通过质量与像素保护指标。继续只搜索一个固定方向，可能遗漏真正依赖状态的修正。
会议版的底层思想仍有价值：读取冻结生成模型已经产生的结构信号，并在 latent 轨迹
上做小幅渲染。本方案把这个思想实现为一个小于 1M 参数的网络，而不要求它继续使用
attention map。

Improving the Diffusability of Autoencoders 和 EQ-VAE 说明 latent 的频谱与空间
等变性会影响生成；FreeU 说明 U-Net 天然层级结构可用于推理增强。Diffusion
Controller、Noise Level Correction、SteeringDiffusion 和 Latent Reward Registers
已经覆盖冻结主干、side network、feature reader 和 latent reward。DiffusionOPSD、
RVM、Self-OPD 也覆盖了 reward target、native regression 和无教师逐步蒸馏。
因此，网络小、冻结主干、coarse-to-fine 或采用 RL 都不能单独作为创新点。

## 可证伪假设

主假设是：普通 SDXL forward 中的 detached `up_blocks.0` feature，能够帮助一个
小网络预测比 `pred_original_sample` 更好的 scheduler 下一状态；该增益必须超过
只看当前 latent、prompt/timestep-only、打乱 feature 和同参数自由残差网络。

如果 feature-aware head 在 held-out 图像上的去噪误差不优于这些对照，或生成时只
提高训练 proxy 而不提高结构与视觉质量，这个方案失败。不能用更大的网络、换 reward
或追加 feature layer 来事后救场。

## 待冻结架构草案

在 1024 分辨率下，只从一次普通 U-Net 调用读取 conditional
`up_blocks.0` 输入 `[B,1280,32,32]`。feature、预测 clean latent、native scheduler
update、pooled prompt 和 timestep 全部 detach，梯度只进入 renderer。

primary 网络使用 `1280 -> 32` 的 `1x1` feature 投影、64 个 hidden channels 和
32/64/128 三个分辨率分支。每个尺度把 resized clean latent 与 native update 共 8 个
channel 拼到 feature 上，再运行两个 D4 对称 depthwise `3x3` + `1x1` residual
blocks；尺度之间只用 bilinear upsampling。pooled prompt 先投影到 32 维，再和 16 维
timestep embedding 一起对每个尺度做全局 FiLM；不使用坐标编码。这个草案约 0.15M
参数，实际实现必须逐项计算并断言小于 1M。

32/64/128 分支分别映射到 low/mid/high；每个输出先 bilinear resize 到 128 latent grid，
再经过对应的互斥 FFT mask。低频从第一个去噪步就开放，中频在去噪进度 0.25 后平滑
开放，高频在 0.55 后平滑开放。这里的进度固定为
`p_i=i/(N-1)`，`i=0` 是首个高噪声 reverse step；频率半径来自 square latent 上的
完整 `fft2/fftfreq`。low 为 `r<=0.08`，mid 为 `0.08<r<=0.25`，high 为
`r>0.25`。令 `smoothstep(u)=3u^2-2u^3` 且先把 `u` clip 到 `[0,1]`，mid gate
为 `smoothstep((p-0.25)/0.20)`，high gate 为
`smoothstep((p-0.55)/0.20)`。这些边界和宽度不在质量结果后调整。所有输出层零初始化。

为了防止后续 row-space 或 Gram 操作把关闭频带重新注入，约束按频带分别完成。令
`M_b` 是三个共轭对称、互斥且完备的 FFT 投影，clean latent 的均值为 `mu`，中心化
矩阵为 `Q`，renderer raw output 中心化后为 `R`。每次 filtering 后都展平为空间列的
`4 x HW` 矩阵。对每个 band `b`：

```text
Q_b = M_b(Q)
R_b = g_b(p) * M_b(R)
P_b(Z) = Z - Z Q_b^T (Q_b Q_b^T)^-1 Q_b
V0_b = P_b(R_b)
```

关闭 band 令 `V_b=0` 并直接保留 `Q_b`。开放 band 还要去掉“只把 CFG scale 调小”的
一阶捷径。令 `C_b=M_b(C-mean(C))`，其中 `C` 是 guided clean prediction 减
conditional clean prediction，再令 `Ct_b=P_b(C_b)`。投影在 float32 累积下计算；
`tau_C` 和 condition-number 上限在 registration 中冻结。若 `||Ct_b||^2<=tau_C`，
登记 CFG-degenerate 并令 `V_b=V0_b`；否则计算：

```text
V_b = V0_b - <V0_b,Ct_b>_F / ||Ct_b||_F^2 * Ct_b
```

上式中的逆只表示线性方程，代码必须使用 Cholesky/triangular solve，不能形成显式 inverse。
非退化 band 必须满足 `|<V_b,Ct_b>|/(||V_b|| ||Ct_b||)<=1e-6`。令
`A_b=Q_b+V_b`，对每个开放 band 分别做：

```text
Q_b Q_b^T = Lq_b Lq_b^T
A_b A_b^T = La_b La_b^T
Y_b = Lq_b * triangular_solve(La_b, A_b)
D_b = Y_b - Q_b
D = sum_b D_b
Y = mu + Q + D
```

正交频带使 cross-band inner product 为零；逐 band Gram 保持因此推出最终 full Gram
保持，关闭 band 的 `D_b` 精确为零。实现必须用 Parseval、closed-band leakage 和最终
`Y_centered Y_centered^T` 三种独立检查验证这个推导。retraction 是非线性的，所以还要
检查最终 `D` 的 CFG leakage；非退化样本要求 global cosine `<=1e-3`。

cap 使用共同的确定性 backtracking：依次尝试 `V_b,k=2^-k V_b, k=0,...,12`，只接受
第一个同时满足 closed-band leakage `<=1e-6`、CFG leakage、per-band/full-Gram tolerance
和 `||x_next_Y-x_next_base||/||x_next_base-x_i||<=0.05` 的候选。试探阶段用下面与
Euler 等价的 endpoint 公式，不额外调用 scheduler；接受后才调用一次真实
`scheduler.step`。算法只缩小 residual，不会放大。单个 band 的 Gram 条件失败就保持
该 band 原生不变并记账；全部 band 不可用、12 次都失败或出现非有限值时 fail closed。

零初始化不能切断训练图。推理时 raw output 逐位为零就直接走 native prediction；训练
时令 `S=sum_b(V_b)`，并只在 exact-zero 分支使用：

```text
epsilon_train = epsilon_g - (S - stopgrad(S)) / sigma_i
```

forward 中括号逐位为零，backward 对每个保留在图中的 `V_b` 返回
`-grad_output/sigma_i`，梯度继续经过 gate、FFT、row-space 和 CFG projection。因为
full-Gram retraction 在 `V_b=0` 的 Jacobian 就是这个可行切空间方向，它既保持 bitwise
forward parity，也给零初始化输出层非零梯度。`C2F-0` 必须检查 bitwise scheduler parity、
初始参数梯度非零和 finite-difference Jacobian parity，不能只检查前向。逐 band
retraction 是已知矩阵几何工具，不作为新颖性主张。

## 第一阶段监督

第一阶段不用 ImageReward、HPS 或 TOPIQ。对冻结且去重的真实图像-caption 训练集，
primary 始终使用部署坐标：epsilon-prediction、CFG `w=7.5`、guidance rescale `0`、
冻结的负面提示和 conditional `up_blocks.0` feature。一次 CFG batch 同时给出
`epsilon_u` 与 `epsilon_c`，然后固定
`epsilon_g=epsilon_u+w(epsilon_c-epsilon_u)`。不允许在训练后切换到 conditional-only
prediction。scheduler 固定为 deterministic `EulerDiscreteScheduler`，调用参数显式为
`s_churn=0.0, s_tmin=0.0, s_tmax=inf, s_noise=1.0`；必须核对实际
`sigma_hat==sigma_i` 且没有 stochastic draw。任一非零 churn 或隐藏 stochasticity 都
fail closed，下面公式不能外推到 Euler-Ancestral 或 multistep scheduler。

对每条记录，从真实 `set_timesteps` 后的 schedule 读取相邻
`sigma_i,sigma_(i+1)`，并复用同一个 counter-keyed noise `epsilon`。未缩放的 scheduler
state 与监督目标固定为：

```text
x_i          = x0 + sigma_i * epsilon
x_next_gt    = x0 + sigma_(i+1) * epsilon
x0_hat_g     = x_i - sigma_i * epsilon_g
x_next_base  = x_i + (sigma_(i+1)-sigma_i) * epsilon_g
```

`scale_model_input(x_i,t_i)` 只生成 U-Net 输入，不能覆盖未缩放的 `x_i`。最后一个
transition 使用 schedule 追加的真实 `sigma_(i+1)=0`，所以 `x_next_gt=x0`。renderer
给出 `Y` 后，重建 `epsilon_Y=(x_i-Y)/sigma_i`，并只调用一次真实
`scheduler.step(epsilon_Y,t_i,x_i)`；训练 loss 读取它的实际 `prev_sample`，不用手写
post-step nudge。

执行步骤是：

1. 用冻结 SDXL VAE 编码 clean latent，并从实际 `set_timesteps` 后的 Euler sigma
   ledger 分层采样 step。VAE posterior 的 mode/sample 规则和随机 counter 必须提前
   冻结。
2. 用 scheduler 的 `add_noise` 构造 `x_i`，并对所有非末步核对同一 API 构造的
   `x_next_gt`；公式、API 与 actual sigma ledger 必须逐张量一致。
3. 用上述固定 CFG batch 运行一次冻结 U-Net，得到 guided base prediction 和
   conditional detached feature。
4. 训练 renderer 缩小真实下一状态与 base Euler 下一状态之间的误差，同时保留完整
   channel Gram、5% trust cap 和 D4 等变性。

主要训练损失是归一化的 scheduler-next-state Huber loss。低/中/高频误差、D4 drift、
更新比例和 Gram drift 全部单独报告，不能靠加权总 loss 隐藏失败。train、validation、
test 的图像 ID、caption、hash 和增强轨迹必须先冻结且互不重合。还要在不知道方法结果
时，把训练 caption 与所有历史、C2F-2、replay 和 C2F-3 prompt 做 exact hash 及近重复
排除；图像使用 source ID、内容 hash 和 perceptual duplicate 检查。文本/图像阈值、
embedding/checkpoint 与 tie policy 必须写进 registration，不能看结果后调整。

## 实验阶梯和停止规则

`C2F-0` 只做 CPU 数学与接线检查：zero forward parity、非零初始梯度与 finite-difference
Jacobian parity、逐频带与 full Gram、D4、关闭频带 leakage、频带开放顺序、真实 Euler
target/API parity、`sigma_hat==sigma_i`、零 churn/零 stochastic draw、CFG nuisance
projection、末步 `sigma=0`、一次 U-Net/一次 scheduler call、参数与显存账本。它不产生
质量证据。

`C2F-1` 是真实图像 denoising feasibility pilot，不使用生成质量 reward。样本量由不看
action identity 或方法均值的重复编码/噪声方差 pilot 按 90% power、familywise alpha
`0.05` 和 1% 最小效应决定；registration 必须冻结至少三个训练 seed、held-out image
ID、每个 image 的 counter-keyed noise，以及
early/mid/late 三个等数量 timestep stratum。配对统计单位是 image ID；noise 和 step
先在 image 内聚合，训练 seed 与 image 用 crossed hierarchical bootstrap。primary 与
no-feature、latent-only、prompt/timestep-only、feature-shuffled 和参数匹配自由 residual
的 `3 strata x 5 controls` 共 15 个差异构成一个 Holm family。每项都必须达到至少 1%
paired relative next-state-error reduction、Holm-adjusted 95% lower bound `>0`，且每个
训练 seed 的差异同号。否则 C2F-1 失败。

no-feature control 把 feature 恒置零；feature-shuffled control 在 train 和 held-out
evaluation 中都使用按 source/timestep 分层的 counter-keyed derangement，不能只在验证
时制造分布偏移。所有 learned controls 使用相同初始化族、数据、优化步、early-stop
规则和训练 seed 数。checkpoint 只按 validation loss、固定 seed 顺序和预登记 tie rule
选择；C2F-1 test 不参与 checkpoint、超参或模型 seed 选择。

proposal-prior 机制消融还包括 all-bands-always-on、固定 reverse band order、实验 ID
hash 在训练前确定的 random band order，以及把 D4 orbit averaging 换成普通 depthwise `3x3` 的
stored-parameter-matched non-D4 head。另训两个只输出有效 CFG scale 的对照：一个读取
prompt/timestep/latent statistics，另一个还读取和 primary 相同的 detached feature；它们
使用相同 next-state loss、训练数据和优化预算。四个机制消融和两个 CFG 对照在三个
timestep strata 上形成 `3 x 6` 的第二个 Holm family。primary 对每个差异都必须有至少
0.5% paired relative error reduction，
adjusted lower bound `>0`，并通过 applied closed-band leakage gate；否则只能否决
coarse-to-fine proposal/D4/非 CFG 的因果假设，整个 family 关闭，不能把 raw gate 写成
applied hard constraint，也不能挑中一个事后赢家改名。

还要分别报告 teacher-forced 1-step、连续 5-step 和完整 50-step 的误差。5-step 与
50-step 分别按 held-out image 聚合，并要求相对 base 的平均 error reduction `>=1%`、
95% lower bound `>0`；任一失败就定义为 exposure-bias failure。任一 stratum 的
adjusted lower bound `<=0`，或超过 1% held-out transition 的 mapped update ratio
`>=0.049`，分别定义为明显退化和大量 cap saturation，都会立即停止。

`C2F-2` 才是 prompt-disjoint 1024 生成。只运行 C2F-1 预登记规则选出的单个 checkpoint；
不得在 C2F-2 prompt、图像或分数上选训练 seed。正式样本量由只读方差/power simulator
按 90% power、双侧 familywise alpha `0.05` 和下述最小效应冻结。比较 no-op、会议
TFSA、最优合规固定动作、tuned/static/dynamic CFG、APG、
FreeU、SPA/FreSca/FDG、DUNE、latent-only、feature-shuffled、feature-aware dynamic-CFG、
SteeringDiffusion-style FiLM/AdaGN、Noise-Level-Correction-style lookup/corrector、
Diffusion-Controller-style 同参数自由 residual 和额外 U-Net 上界。所有最近邻实现需先
通过源码与坐标审计；无法忠实复现时明确标为 local surrogate，不能静默削弱 baseline。

配对单位是 prompt，seed 在 prompt 内聚合。primary confirmatory comparisons 固定为
no-op、会议 TFSA 和在独立 development split 选出的最强合规 baseline，三者构成 Holm
family。每项要求 TOPIQ-NR paired mean delta `>=+0.005` 且 adjusted 95% lower bound
`>0`；OCR/count/spatial 的预冻结 macro delta `>=+0.02` 且 lower bound `>0`；独立、
action-blind 人评 preference 点估计 `>=55%` 且 95% lower bound `>50%`。HPSv2 与 CLIP
的 lower bound 必须分别 `>=-0.005`。clipped fraction delta `<=+0.001`、mean
saturation delta `<=+0.005`、contrast ratio 位于 `[0.95,1.05]`，diversity 不劣界、
频谱/moment tolerance、CFG leakage、延迟和显存上限都要在 registration 中由重复测量
冻结，并作为 conjunction gates，不允许互相补偿。denoising proxy 改善但任何生成 gate
失败，仍判整个方案失败。

独立 1024 replay 使用新的 prompt/seed namespace，复用同一 checkpoint 和全部阈值，
并重复 C2F-2 confirmatory conjunction；它不能重做 baseline selection。通过后才运行
`C2F-3` transfer。C2F-3 从 C2F-2 起冻结同一个 renderer checkpoint、feature hook、
频带门、cap 和阈值，不允许逐轴重训或调参。renderer 仍只作用在 RepLDM Stage-1；
Stage-2 只测试最终高分辨率输出。ControlNet、第二个具有同一 feature shape 的
SDXL-compatible checkpoint、30 NFE 和 CFG `5.0` 分别与原 50 NFE/CFG `7.5` 构成五个
预登记 transfer settings；shape 或 scheduler 不兼容直接计为失败，不能换 layer。

整个 transfer 阶段只建立一个新的 namespace，而不是给每个 cell 换数据。Stage-2、
30/50 NFE、CFG `5.0/7.5` 和第二 checkpoint 的可比较 cells 共用同一批 prompt、seed、
初始 noise 和 counter-keyed stochastic draws；ControlNet 使用独立 condition manifest，
但同一 condition/prompt/seed 内的 renderer/no-renderer 仍严格配对。共享 T2I factorial
报告逐 prompt paired delta 和 method-by-setting interaction；ControlNet 只报告其 manifest
内部的 paired method effect，不能拿跨 manifest 的边际均值解释 interaction。每个 transfer
axis 都用不读取 method mean 的自身重复/外部方差 pilot，按 90% power、familywise alpha
`0.05` 和预登记最小效应冻结 prompt/condition 数、生成 seed 数及聚合层级；不能直接
沿用 C2F-2 的样本量。所有 setting 同时报告 matched-NFE、wall time 与显存。

五个 TOPIQ method comparison 构成一个 Holm family；每项相对 matched no-renderer 都
要求 mean delta
`>=+0.003`、adjusted lower bound `>0`，结构 macro delta `>=+0.01`、lower bound `>0`，
并通过 C2F-2 的非劣与像素 guards。只有五项全部通过才称为 `C2F-3 pass`。任一失败时
可以把监督 renderer 的结论收窄到已通过的设定，但不得进入 reward distillation 或 RL。

只有 `C2F-2`、新的独立 replay 和 `C2F-3` transfer 都通过，才比较
frozen-teacher distillation、OPSD-style bounded target、Self-OPD-style branch
distillation 和 anchored native prediction regression。还要实现 LeSAMP-style
prompt/timestep sampling-policy control，并匹配 renderer 参数、reward queries、rollout
和 GPU-hour 预算；没有可忠实复现的实现时，只能使用通过源码审计且明确标注的 surrogate，
或把 RL 对比结论标为 no-go。任何 RL authorization 之前，
还要在不与最终 test/replay 重合的 development prompts 上冻结 bounded branch/action
bank，用 nested OOF selector 估计相对“监督 C2F checkpoint 与最佳蒸馏模型中较强者”的
可实现 residual headroom。只有 OOF TOPIQ delta `>=+0.005`、结构 macro delta
`>=+0.02`、两者 95% lower bound `>0` 且全部 C2F-2 guards 通过，才允许签发 RL
authorization；同时冻结 reward query、rollout、GPU-hour、停止规则和最终 test 的一次性
预算。最终 RL 必须在新的未见 prompts 上同时显著胜过监督 C2F checkpoint、最佳蒸馏
模型和 LeSAMP-style policy control，重复结构、人评和安全 conjunction，并计入全部
reward queries 与 GPU 小时；否则论文方法应停在更简单的监督或蒸馏 renderer。

## 严格反思和下一步

真实图像去噪目标可能只降低重建误差，却不能改善开放文本生成；完整 moment 保持也
可能排除真正需要的校正；32x32 feature 可能没有提供超出 clean latent 的信息。
这些都是生死问题，不是可通过写作消除的限制。自动 OCR 和 detector 还必须先做
action-blind 可靠性校准，正式结构结论需要人工复核。

下一步先等待 adaptive oracle engineering audit。通过后再为这个新 family 提交独立、
不可执行的 registration，冻结数据来源、容量、频带门、loss、样本量和停止规则；
随后实现 `C2F-0`，完成 warning-as-error CPU audit。`$check 1`、push 和远端提交核对
都是必要条件，但仍不能授权训练；还需要独立审查人签发绑定 reviewed commit、CPU
audit、环境、模型和输出目录的一次性 executable authorization。当前报告必须强制
加入 git，不能因为 `_experiments_expansion/` 的 ignore 规则而只留在本地。
