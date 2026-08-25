# 与调度器一致的轨迹校正

## 动机

会议版本的方法在 U-Net 给出预测后修改冻结扩散模型的生成轨迹。此前对
attention、semantic transport、固定 latent basis 和 FreeU 的搜索，没有在
held-out 数据上得到同时满足保护条件的改进，因此当时没有依据训练 RL 策略。
本实验检验一种与会议版本较弱但明确的联系：保持模型和初始噪声不变，直接用
调度器所定义的解析 ancestral transition 校正 Euler transition。

这一方案受到 LC-GRPO
([2608.05600](https://arxiv.org/abs/2608.05600)) 的 predictor-corrector、
CreFlow ([2605.14274](https://arxiv.org/abs/2605.14274)) 的校正方向与
credit assignment，以及 diffusion policy 的 path-space 视角
([2608.14430](https://arxiv.org/abs/2608.14430)) 启发。这些文献只限定本实验的
新颖性边界，不表示该机制已在本项目中得到验证。本实现不能声称提出了新的 RL
估计器。当前只问一个窄问题：在考虑 renderer 或 RL 之前，有界的
Euler-to-ancestral 固定动作是否有效。只有 Euler 和 Euler-Ancestral 两个端点
与原生调度器严格一致；中间 `mix` 只是插值对照，除非另有 SDE 推导，否则不能
称为新的 SDE。

## 方法

对 epsilon-prediction Euler step，设噪声水平为 `sigma_from` 和
`sigma_to`：

```text
sigma_up^2   = sigma_to^2 (sigma_from^2 - sigma_to^2) / sigma_from^2
sigma_down^2 = sigma_to^2 - sigma_up^2
derivative   = (x_t - x_0_hat) / sigma_from
ancestral    = x_t + derivative (sigma_down - sigma_from)
raw          = mix (ancestral - euler_prev) + sigma_up sqrt(mix) epsilon
```

可选的 trust cap 对每个样本限制
`||raw|| / ||euler_prev - x_t||`。`mix=0` 直接返回调度器输出，不进行额外
算术，也不消耗随机数；不使用 cap 时，`mix=1` 在数值上等于
Euler-Ancestral transition。该 hook 只用于 Stage 1，独立运行，并记录每一步
correction norm 和 update norm。

端点实现保留仓库固定版本 `diffusers~=0.21.4` 的原生低精度算术，也保留当前
评估环境 `0.32.1` 的 upcast 路径；严格的 fp16/bf16 parity tests 覆盖这两条
路径。配对校正运行要求每个 task 只传一个 `torch.Generator`，不接受 generator
列表。

## 开发结果（已完成，探索性）

四个 prompt 的 probe 显式使用匹配的 Euler scheduler，并确认 `mix=0` 生成的
PNG 字节完全一致。与 Euler 相比，`mix=0.25`、`0.50`、`0.75` 和 `1.00` 的
平均 TOPIQ-NR 变化分别为 `+0.005616`、`+0.013206`、`+0.021842` 和
`+0.024682`。这只能算探索性结果：单个 prompt 的 TOPIQ 变化范围是
`-0.026565` 到 `+0.065755`，较大的 `mix` 还会在部分 prompt 上增加 clipping
和 saturation。四个 prompt 不足以支持统计推断，也不能据此决定是否训练
renderer 或 RL。

新注册的 development manifest 包含 11 个与先前数据不重合的 prompt、两个
seed、一个原生 Euler-Ancestral reference，以及 `mix=0.25/0.50/0.75`。YAML
和 metadata 记录了源 TSV hash，以及已核验的 PartiPrompts commit
`5a657978134374ce28973948331b319adef164bd`。这是 development gate，不是
journal test set。

## 审稿门槛

在提出 state-conditioned renderer 之前，必须先在 development 数据上选择
固定动作，再到新冻结、规模更大且 prompt-disjoint 的 validation split 上击败
`no_correction`。主指标仍是 TOPIQ-NR，并使用 crossed prompt/seed bootstrap
和 prompt sign-flip tests。HPSv2/CLIP 是相关的代理见证，不能互相充当独立
确认；clipping、saturation、
contrast、colorfulness 和 sharpness 是保护指标。

原生 Euler-Ancestral 是必需的 sampler control；发表前还必须在相同 NFE 下
比较 DPM-Solver++ 和 UniPC。分析必须使用 Holm correction 和预注册的最小效应
阈值。如果校正方案无法通过上述保护条件，就关闭这条路线，不训练 RL。只有固定
动作通过后，才允许搜索并蒸馏小型 state-conditioned renderer；只有 RL 在同一
held-out 协议下同时击败固定搜索和 distilled controller，才允许继续采用 RL。
