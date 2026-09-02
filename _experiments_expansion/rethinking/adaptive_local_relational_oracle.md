# Adaptive Local-Relational Oracle

## 为什么提出这个方案

S0-S5、旧 LR-1、FreeU、trajectory correction 和 scheduler-native frequency
screen 都没有找到同时改善质量并通过保护指标的固定动作。继续调 TFSA scale、频带、
FreeU 强度或 scheduler mix，只是在已经失败的动作空间中重复搜索。因此本组实验更换
更新方向的来源：不读取 attention map，而是读取冻结 SDXL U-Net 在普通 forward 中
已经产生的局部 feature relation。

iREPA、SARA、sREPA、SGA 和 SPARE 表明，token 之间的关系可能比单点 feature
alignment 更能描述结构。Improving the Diffusability of Autoencoders 和 EQ-VAE
说明 latent 的频谱与空间等变性会影响生成难度。这些工作只提供动机；它们没有证明
冻结 SDXL 上的推理期局部传输有效。双边滤波、图 Laplacian、relation matching 和
小型 frozen-backbone controller 也都有先例，所以本项目不能把任何单个组件写成
新颖贡献。

## 当前假设和因果对照

在每个 Euler 步，只捕获 conditional branch 进入 `up_blocks.0` 的 feature。把它和
scheduler 预测的 clean latent 都缩到 16x16，在距离 1 的水平/垂直、距离 1 的对角、
距离 2 的水平/垂直三个 D4 orbit 上计算局部 cosine affinity。局部加权传输给出三个
候选方向。方向随后投影到完整 channel row space 的正交补，并用 full-Gram geodesic
保持四个 latent channel 的均值和完整协方差。

干预不是直接加到 `prev_sample`。代码先把目标 clean endpoint 转回 Euler 的原生
prediction，再调用一次真实 `scheduler.step`。实际输出相对原生输出的更新比例目标为
`0.02`，容差 `5e-4`，硬上限 `0.05`。

主要动作是 no-op 加三个 orbit 的正负方向。三类对照用于回答不同问题：

- `R` 用 D4 对称、norm-matched 的随机 edge affinity，检查任何方向是否都能得分；
- `U` 使用 uniform local graph，检查收益是否只来自普通局部平滑；
- `X` 用 predicted-clean latent 自身的 affinity，检查 U-Net feature 是否真的提供额外信息。

后续 selector 还必须比较 structural descriptor、prompt-only descriptor、随机 injection
和 context-free 选择器。只有 descriptor 与 relational injection 的交互项为正，并同时
胜过 `R/U/X`，才能把收益归因于冻结 U-Net 的局部结构。

## 当前工程结果

截至 2026-08-26，尚未运行 GPU 生成、图像评分、renderer 训练或 RL。已完成的只是
CPU 数学与机制检查。full-Gram 修复后，真实 50-step Euler 测试中的最大 covariance
drift 为约 `1.46e-7`，远低于冻结的 1% 上限；float32 和 float16 输入都通过。数学与
机制、注册和审计统一测试为 `237/237`，其余 legacy 测试为 `333/333`；二者均在 CPU-only、离线和 warning-as-error 条件下
通过。prompt 资产可以从固定 PartiPrompts 源逐字节重建；不可执行 registration 已绑定
prompt、设计、模型快照、Euler schedule 和环境锁 hash。完整 PNG container、真实
scheduler ledger、真实 pre/post-step latent hash、warning/stderr 证据、物理 GPU
身份和运行中源码复核也已接入测试。冻结的 exclusion inventory 把历史 metadata 中
逐文件的 prompt、source row 和 seed
投影固定为排序且无重复的列表，并单独登记禁止读取的评分/质量路径；因此 prompt 资产
可以在没有 `outputs/` 的私有快照中逐字节重建。真实 `cuda:1` 的无生成环境检查通过，
但一次性 CPU audit、独立 executable authorization 和 GPU smoke 尚未执行，因此状态
仍是 `blocked_registration_only`。

计划中的 engineering smoke 是 11 prompts x 15 trajectories，共 165 张 1024 图像。
它禁止评分和看图，只检查真实 SDXL 上的 hook、scheduler、hash chain、moment、调用
次数、输出完整性和可复现性。即使通过，也不能作为效果证据。

## 严格反思

这个方案的最大风险不是代码能否运行，而是局部 graph smoothing 可能只改变纹理，
并不修复 OCR、数量和空间关系。固定 `0.02` 也可能比真实有效范围更强。工程 smoke
不能回答这些问题，正式 search 必须使用新 prompts/seeds、结构答案标签、随机动作、
uniform/self-affinity controls 和结果盲 OOF selector。TOPIQ 的小幅上涨若伴随 clipping、
saturation、高频能量或多样性恶化，仍然判为失败。

DiffusionOPSD 和 RVM 进一步说明，reward-to-clean-target distillation、signed native-
coordinate regression 和 reference anchoring 已不是空白。如果本方案以后通过 replay，
首个学习实验必须比较 frozen search teacher、OPSD-style bounded target distillation 和
anchored native-prediction regression，并分开报告 target 本身的收益与小 renderer 实际
实现的收益。RL 算法本身不能作为贡献。

## 下一步

先完成 warning-as-error CPU gate、提交不可执行 registration、生成一次性 CPU audit，
再由独立审查签发仅允许 generation 的 executable authorization。之后运行 165 轨迹
engineering smoke 和结果盲审计。若通过，再实现 power simulator、正式 search、OOF
freezer、OCR/count/spatial scorer 和一次性 futility evaluator。

正式 search 若没有同时出现 TOPIQ `>= +0.005`、结构 macro 增益、正 interaction、
`P-U/P-X` 增益及全部保护指标，就关闭这个 family，不用 reward tuning、蒸馏或 RL
挽救。只有 search 和独立 replay 都通过，才进入 search-then-distill；RL 还必须证明
相对蒸馏模型存在新的 held-out headroom。
