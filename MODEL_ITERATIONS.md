# 模型迭代记录

这份文档记录每一轮方案是怎么提出、怎么验证、最后为什么保留或放弃的。为了避免“看完结果再改说法”，每轮实验前定好的假设和门槛，与实验后得到的结果分开写。生成文件仍放在不会提交到 Git 的 `outputs/` 下，每次运行都记录对应的 Git commit。

## 先看这些术语

- `no-AG`：完全不加 Attention Guidance 的基线。后文说“比 no-AG 好或差”，就是和原模型直接比较。
- latent：扩散模型内部处理的特征，不是最终图片。这里可以简单理解成“还没解码成图片的中间表示”。
- TFSA residual：做完 TFSA 后与原 latent 的差，即“TFSA 想把 latent 往哪里推、推多少”。
- scheduler：负责执行正常去噪步骤的模块。它给出的更新方向，可以理解成原模型本来准备怎么走。
- smoke：很小的冒烟测试，只用来检查代码是否正常、参数是否会造成明显坏图，不能证明方法有效。
- development gate：正式开发实验必须跨过的门槛。没过门槛，就不能继续拿这个方案做 RL 或反复调参。
- CI：置信区间。区间跨过零，表示目前的数据还不能确定提升是真的。
- Holm correction：同时比较很多方案时使用的统计校正，避免因为“试得多”而碰巧得到显著结果。
- guard：防止指标投机的保护指标。例如主指标提高了，但过曝、饱和度或裁切明显变坏，仍算失败。
- prompt-disjoint：新实验使用与旧实验完全不重合的提示词，防止在已经看过的题目上越调越好。
- NFE：UNet 前向计算次数，用来比较不同方法的计算量是否公平。
- CFG：文本条件引导强度。
- `x_0`：scheduler 估计的干净图像 latent，也就是它认为去噪结束后会得到的结果。
- `top-k`：每个位置只保留最强的若干连接。

## 各轮结论一览

| 编号 | 方案 | 状态 | 一句话结论 |
|---|---|---|---|
| S0 | 给 TFSA residual 乘固定系数 | 已否定 | 系数变大会让画面更容易过饱和，并降低 TOPIQ-NR；系数很小时又几乎等于 no-AG。 |
| S1 | 分别调低频、中频和高频 residual 的强度 | 已否定 | 最好的动作 `mid_only_0.004` 也不优于 no-AG；按 seed 做交叉验证后，也看不到值得自适应选择的空间。 |
| S2 | 固定均值和方差的 Attention Guidance（MTAG） | 已否定 | 它确实消除了均值和方差漂移，但没有提高画质或偏好指标。 |
| S3 | 删除“逆着正常去噪方向走”的更新分量（TCMG） | 已否定 | 这个方向约束确实生效了，但没有提高画质、偏好指标，也没有带来可用的自适应空间。 |
| S4 | 在最终 2048² 高分辨率结果上复查前面方案（Stage-2 audit） | 已否定 | 第二阶段（Stage 2）没有改变此前的负面排序；没有任何固定动作胜过 no-AG，因此仍不能开始 RL。 |
| S5 | 与 scheduler 一致的双向语义传输（SCRST） | 已登记，待验证 | 不再从 noisy latent 的 TFSA 取更新方向，改用 conditional UNet self-attention 和 scheduler 返回的 predicted-clean latent。这里登记的是待检验方案，不是成功结论。 |

S0-S4 的完整证据见 `EXPERIMENT_RESULTS.md`。这些轮次已经失败，不能把它们的动作集合拿去训练 RL。

## S2：先把每个通道的均值和方差固定住

### 为什么做

论文版 TFSA 更新同时做了两件事：一是重新安排空间位置上的信息，二是改变 latent 每个通道的均值和方差。后者会直接改变画面的裁切、对比度和饱和度。S2 想回答一个很具体的问题：如果不准 TFSA 改均值和方差，只保留它对空间结构的推动，结果会不会变好？

### 具体怎么改

对每个样本、每个通道，定义：

```text
d = TFSA(z) - z
x = z - mean_spatial(z)
v = center_spatial(d) - <center_spatial(d), x> x / ||x||^2
theta = scale ||v|| / ||x||
z' = mean_spatial(z) + cos(theta)x + sin(theta)||x||v/||v||.
```

大白话解释如下：`d` 是原始 TFSA 想施加的改变量；`x` 是去掉空间均值后的 latent；`v` 再从 `d` 中删掉会改变整体尺度的那一部分；`theta` 决定实际走多远；最后得到的 `z'` 把原来的均值加回来。几何上，这相当于让 `x` 在一个保持长度不变的球面上转动，所以每个通道的均值和方差只会有数值舍入级别的误差。

`moment_tangent_rescaled` 会先把 `||v||` 调回原始 residual 的大小，再执行同样的转动。这样可以分清：结果变化究竟来自“方向约束”，还是仅仅因为更新被缩小了。

已有的 GAG、NAG、ASAG 主要修改 attention 的外推、归一化或 negative attention 分支。MTAG 改的是 RepLDM 在 scheduler 之后额外加到 latent 上的更新，并强制它保持均值和方差。LeSAMP、AdaGen 这类工作与未来的策略学习有关，但不等于这个算子本身已有新意；“可以自适应控制”本身也不是这里的创新点。

### 冒烟测试怎么定

- 先提交代码，再在同一张 GPU 上用 seed `0` 生成 `smoke.csv`。成对比较的所有动作必须共用硬件。
- 比较 `raw`、`mean_centered`、`moment_tangent` 和能量匹配的 moment tangent，scale 使用 `0.001`、`0.002`、`0.004`；只有 tangent 类模式额外试 `0.008`。同时放入 no-AG 和论文版 expert。
- 冒烟测试只检查实现和危险参数，不用来宣称有效。如果出现非有限值、非零动作却完全没改图、任一提示词的 TOPIQ 下降超过 `0.05`，或 clipped fraction 上升超过 `0.01`，就淘汰该动作。
- 如果某个 tangent 系列还有可用范围，就保留一段连续的 scale 区间，进入 12-prompt、3-seed 的开发实验。不能从冒烟结果里单独挑一个看起来最好的 scale。

### 正式开发实验的通过条件

主指标是 TOPIQ-NR。每个已登记动作还必须报告 HPSv2、ImageReward、patch-IR、CLIP alignment、aesthetic score、clipped fraction、saturation、contrast、colorfulness 和 sharpness。统计上使用同设备配对、prompt/seed 交叉 bootstrap 区间、按 prompt 的 sign-flip test，以及 Holm correction。

候选方案必须同时满足下面所有条件，才可以进入 RL：

1. 相对 no-AG 的平均 TOPIQ-NR 至少提高 `+0.005`，95% CI 不跨零，而且同一指标内经过 Holm 校正的检验值低于 `0.05`。
2. 在相同生成预算下，还要胜过论文版 expert 和最好的 raw scalar。
3. HPSv2 和 CLIP 不能明显变差；clipped fraction 和 saturation 相对 no-AG 的增量分别不能超过 `0.001` 和 `0.005`。
4. 固定版式的对比图必须显示可信的结构或细节改善，不能只是全局颜色或对比度变化。

通过这道门槛，只代表可以换一套不重合的提示词做确认实验，并测试高分辨率第二阶段（Stage-2）迁移；还不能直接宣称形成了期刊结论。未通过就结束 S2，下一步必须换一个新的算子假设，不能继续调 reward 或堆更复杂的 controller。

### 冒烟测试结果

运行目录：`outputs/exp_moment_tangent/smoke_2prompt_1seed_v1`，由 commit `b9f3fbc` 在 GPU 1 上生成。预期的 32 组 PNG/JSON 全部有效，都是 1024² RGB 图像；动作元数据完整，而且 PNG hash 各不相同。no-AG、论文版 expert，以及 raw scale `0.001/0.002/0.004`，都在同一张 GPU 上精确复现了历史 PNG SHA-256 hash。

下面只是两个提示词的平均值，只能用来排查问题，不能证明哪个动作有效：

| Action | Δ TOPIQ | Δ HPSv2 | Δ clipped | Δ saturation |
|---|---:|---:|---:|---:|
| raw 0.001 | +0.001310 | +0.000366 | +0.001281 | +0.008493 |
| mean-centered 0.001 | -0.000746 | +0.001953 | -0.002757 | +0.005103 |
| tangent 0.002 | -0.007128 | +0.005005 | -0.004189 | -0.000084 |
| tangent-rescaled 0.002 | -0.006053 | +0.001465 | +0.003750 | +0.003801 |

固定对比图里没有稳定的结构赢家。moment tangent 确实压住了 raw 更新带来的对比度和饱和度变化，但 energy-matched 版本在墙面物体上产生了扁平设计一样的伪影。按照预先登记的灾难性淘汰门槛，删掉 tangent `0.008`、tangent-rescaled `0.004/0.008` 和 mean-centered `0.004`。

正式开发范围已经固定在 `eval-pipeline/configs/moment_tangent_development.yaml`：mean-centered 为 `0.001–0.002`，tangent 为 `0.001–0.004`，tangent-rescaled 为 `0.001–0.002`。raw `0.001` 是此前已经确认的最佳 scalar 对照。因为 S2 的选择过程已经参考过这套 12-prompt 数据，所以它只能算开发数据；任何通过的动作仍需用不重合的提示词再次确认。

### 正式开发结果

运行目录：`outputs/exp_moment_tangent/development_12prompt_3seed_v1`，12 prompts × 3 seeds × 10 actions，由 commit `65ba734` 生成。全部 360 条记录完整且配对正确；108 张 no-AG、expert、raw 对照图在相同 GPU 上精确复现了之前的 hash。

没有动作通过预先登记的门槛。TOPIQ 降得最少的是 mean-centered `0.001`：`-0.000946`，95% CI `[-0.005246,+0.003537]`。moment tangent `0.001/0.002/0.004` 分别为 `-0.002316/-0.002268/-0.004424`；energy-matched `0.001` 在同指标 Holm 校正后显著更差（`-0.003763`，CI `[-0.005925,-0.001483]`，adjusted `p=0.025920`）。所有 S2 动作的 HPSv2 变化都在 `±0.001` 内，CLIP 也没有提高。

机制判断本身得到了支持：raw `0.001` 让 saturation 增加 `+0.013449`，而三个未做能量匹配的 tangent 动作只改变 `-0.000862/-0.000997/-0.001767`，clipped fraction 也保持在 `±0.001` 内。也就是说，固定均值和方差确实堵住了 TFSA 通过颜色和对比度改分的路，但剩下的空间更新方向没有可测量的收益。

按 TOPIQ 为每个 prompt 做 seed-CV，相对 no-AG 是 `+0.001791`，CI `[-0.004392,+0.007529]`，同时 clipping 和 saturation 反而增加。按 HPS 选择的每 prompt 动作，比全局 expert 低 `-0.002082`，CI `[-0.003872,-0.000434]`。固定对比图能看到构图变化，但看不到稳定的结构修复。因此 S2 到此结束，不再继续调它的 scale，也不能用它启动 RL 训练。

## S3：删掉逆着正常去噪方向走的部分

### 为什么做

S2 已经说明，均值和方差漂移有害；但即便去掉这条路，TFSA 剩下的空间更新仍没有稳定收益。S3 接着问：是不是 TFSA 更新里有一部分正好和原模型的正常去噪方向相反，因此拖累了结果？

### 具体怎么改

令 `P_T` 表示 S2 的固定均值、固定方差切向投影，`d = TFSA(z)-z`，`g` 表示 scheduler 刚刚为了得到 `z` 而执行的更新。对每个样本、每个 latent 通道：

```text
v = P_T(d),  q = P_T(g)
c = v - min(<v,q>, 0) q / ||q||^2
z' = Exp_z(scale c).
```

这里 `v` 是处理后的 TFSA 方向，`q` 是处理后的正常去噪方向。如果 `<v,q>` 为负，就说明 `v` 有一部分在和 `q` 对着干；公式把这一部分删掉，得到 `c`。因此 `<c,q> >= 0`：最后保留的更新不会逆着正常去噪方向走，同时仍保持每个通道的均值和方差。

所谓“trajectory cone（轨迹锥）”，只是一个几何说法：把允许的更新方向限制在与正常去噪方向夹角不超过直角的那一侧。它不是图像中出现了锥形结构。这个操作不需要新参数、不看未来状态，也不增加 UNet 调用。它的验证问题很明确：如果 S2 失败真是因为存在反向分量，那么相同 scale 下 TCMG 应该胜过普通 moment tangent。energy-matched 版本仍用于区分“方向变了”和“更新只是变小了”。

GAG 的平行/正交分解处理的是 sparse 与 dense cross-attention guidance。TCMG 处理的则是外加的 self-attention latent 更新，并拿 scheduler 实际走过的方向来限制它，同时保持均值和方差。这里不把“做一次方向分解”单独当作通用创新；只有完整算子和后续 learned policy 都能胜过基线，才可能形成期刊主张。

### 冒烟测试怎么定

配置为 `eval-pipeline/configs/trajectory_cone_smoke.yaml`：两个 prompts、seed `0`、GPU 1。动作包括 no-AG、expert、raw `0.001`、plain tangent `0.002`、TCMG `0.001/0.002/0.004/0.008`，以及 energy-matched TCMG `0.001/0.002/0.004`。沿用 S2 的淘汰规则：TOPIQ 变化低于 `-0.05`，或 clipping 增量高于 `+0.01`；非有限值和动作没生效也会淘汰。冒烟测试只确定一段连续 scale 范围。只要至少有一段 TCMG 区间存活，才允许做 12×3 开发实验；正式门槛与 S2 完全相同。

### 冒烟测试结果

运行目录：`outputs/exp_trajectory_cone/smoke_2prompt_1seed_v1`，由 commit `0e00fe6` 在 GPU 1 上生成。22 条记录都完整、有限，而且像素结果不同。八个 no-AG、expert、raw、plain-tangent 对照精确复现了 S2 的 hash，说明 tangent 重构没有引入集成回归。TCMG 与 plain tangent 的 PNG 不同，说明“删掉反向分量”这个操作在真实 scheduler 轨迹上确实生效了。

两个提示词上的 TOPIQ 平均变化全部为负：TCMG `0.001/0.002/0.004` 分别是 `-0.006597/-0.007909/-0.010637`；energy-matched `0.001/0.002` 分别是 `-0.010521/-0.008155`。TCMG `0.002` 的 HPSv2 是 `+0.005859`，但两个提示词的区间没有解释力，而且 ImageReward 是 `-0.065652`。固定对比图显示它会随状态改变构图，却没有稳定修复结构。

TCMG `0.008` 因为一个 prompt 的 clipping 超出 `+0.01` 限制而被删掉；energy-matched `0.004` 也因同样原因删掉。未触发灾难性问题的连续范围是 TCMG `0.001–0.004` 和 energy-matched TCMG `0.001–0.002`。这些范围连同 no-AG、expert、raw `0.001` 和 plain tangent `0.002`，固定在 `eval-pipeline/configs/trajectory_cone_development.yaml`。这次冒烟测试仍不能当作有效性证据。

### 正式开发结果

运行目录：`outputs/exp_trajectory_cone/development_12prompt_3seed_v1`，12 prompts × 3 seeds × 9 actions，由 commit `b0aa343` 生成。全部 324 组 PNG/JSON 都是有效的 1024² RGB 图像；每个 prompt/seed block 都在同一张 GPU 上包含全部九个动作，每张 GPU 各生成 81 条记录。144 个 no-AG、expert、raw、plain-tangent 对照全部精确复现 S2 的 PNG hash。

没有 TCMG 动作通过预先登记的门槛。相对 no-AG 的结果如下：

| Action | Δ TOPIQ-NR [95% CI] | Δ HPSv2 | Δ clipped | Δ saturation |
|---|---:|---:|---:|---:|
| cone 0.001 | -0.002519 [-0.005576,-0.000152] | -0.000553 | -0.000175 | -0.000812 |
| cone 0.002 | -0.001740 [-0.005274,+0.001283] | +0.000085 | +0.000050 | -0.000385 |
| cone 0.004 | -0.004220 [-0.010200,+0.001771] | +0.000665 | +0.000600 | -0.000368 |
| cone-rescaled 0.001 | -0.003178 [-0.005424,-0.000840] | +0.000173 | +0.000085 | -0.000642 |
| cone-rescaled 0.002 | -0.005843 [-0.010720,-0.000922] | +0.000346 | +0.000998 | +0.000022 |

在相同 scale 下，cone `0.002` 的 TOPIQ 只比 plain tangent `0.002` 高 `+0.000528`，CI `[-0.000412,+0.001596]`；HPSv2 只变化 `+0.000031`。所以，半空间投影确实能删掉反向分量，但被删掉的这部分并不能解释 S2 为什么整体无效。

按 TOPIQ 做 seed-CV 时，三个 fold 的全局选择都是 no-AG。每 prompt 选择相对 no-AG 是 `+0.001120`，CI `[-0.004272,+0.006516]`，但 clipping 上升 `+0.001262`，saturation 上升 `+0.012097`。按 HPS 选择的每 prompt 动作，比全局 expert 低 `-0.001604`，CI `[-0.003455,-0.000190]`，而且两个 guard 都明显变坏。

完整的 seed-0 对比图位于 `outputs/exp_trajectory_cone/development_12prompt_3seed_v1/figs/all_actions_seed0.png`。它显示动作会随状态改变构图，但没有稳定修复结构或文字。因此 S3 到此结束，不再做另一轮 scale 搜索，也不能拿来启动 RL。

## S4：只检查高分辨率第二阶段（Stage-2）会不会改变结论

### 为什么做

S0-S3 评估的是第一阶段（Stage-1）生成的 1024² 图像，但 RepLDM 最终交付的是重采样后的高分辨率图像。S4 只检查这两者是否存在偏差：前面表现不好的动作，经过第二阶段（Stage 2）后会不会反而变好。

这不是给失败方案“再调一次”的机会，也不能把已经用过的开发 prompts 当成独立确认数据。Attention Guidance 仍然只作用在第一阶段（Stage-1-only）；S4 只是观察后续高分辨率处理会放大、减弱，还是反转它带来的影响。

### 先过工程正确性检查

批量生成之前，必须在 `prompts/stage2_smoke.csv`、seed `0`、单张 GPU 和 `configs/stage2_engineering_smoke.yaml` 上通过第二阶段正确性检查（Stage-2 correctness）。第二阶段噪声（Stage-2 noise）必须来自当前任务自己的 generator，不能依赖全局 RNG；启用 phase offload 时，正常 2048² 解码必须把 VAE 和 latents 放到实际执行设备上。

两个重复的 no-AG 动作必须得到完全相同的最终 PNG hash，expert 必须不同；所有输出必须是 2048² RGB，metadata 必须记录第二阶段设置（Stage-2 settings），重新运行也必须复现 hash。任何一项失败，都不能开始 pilot。

### Pilot 固定方案和通过条件

固定 pilot 使用 `configs/stage2_transfer_pilot.yaml`：12 prompts × 3 seeds × 5 actions = 180 张最终 2048² 图像。五个动作是 no-AG、论文版 expert、raw `0.001`、plain tangent `0.002` 和 cone `0.002`。它们是根据此前证据固定的机制阶梯，不允许在高分辨率上重新搜索 scale。同一 prompt/seed 下的动作必须共用 GPU 和任务级第二阶段噪声（Stage-2 noise）。

主指标仍是 TOPIQ-NR。只有 TCMG 相对 no-AG 至少提高 `+0.005`、95% CI 不跨零、同指标 Holm 校正后的检验值低于 `0.05`，并且直接配对比较也胜过 expert、raw 和 plain tangent，才允许重新打开这个方法。HPSv2 和 CLIP cosine 的 CI 下界必须高于 `-0.003` 和 `-0.005`；clipped fraction 和 saturation 的平均增量必须低于 `+0.001` 和 `+0.005`。还要报告 ImageReward、patch-IR、aesthetic、contrast、colorfulness、sharpness、所有失败动作和固定对比图。

即使通过，也仍需使用不重合的 prompts 做确认，并进行盲评的高分辨率偏好比较和细节裁剪比较。如果没有动作通过，就关闭 Attention Guidance 在目标 pipeline 第一阶段（Stage-1）上的路线，不能直接跳去设计 learned controller。

### 工程检查结果

`outputs/exp_stage2_transfer/engineering_smoke_v1` 和 `engineering_smoke_repeat_v1` 是两个独立运行，都由 commit `f18b6b1` 在 GPU 1 上生成。三个动作都完成了正常的 2048² encoder/decoder 流程，没有 OOM 或设备不匹配；观察到的 GPU 峰值显存约为 21.2 GB。每张输出都是有效、非空白的 2048² RGB PNG，第二阶段 metadata 完整。

每次运行内部，`no_ag` 与 `no_ag_repeat` 的 SHA-256 相同（`3187f72c...0930a55`），而 `conference_expert` 不同（`a513da66...8efddf`）。换新进程和输出目录后，三个动作的 hash 都能精确复现。因此，任务 generator 确实控制了两个采样阶段，CPU phase offload 后 pipeline 仍可再次运行。第二阶段 metadata（Stage-2 metadata）也完整。工程门槛通过，可以按已经固定的五个动作开始 pilot；动作和阈值都不变。

### Pilot 结果

运行目录：`outputs/exp_stage2_transfer/pilot_12prompt_3seed_v1`，12 prompts × 3 seeds × 5 actions，由 commit `a31cb75` 生成。全部 180 张最终图像都是有效的 2048² RGB PNG；36 个 prompt/seed block 完整且使用同一设备，每张 GPU 生成 45 条记录，每个 block 内的动作 hash 都不同，两个 smoke 对照也精确复现。严格评分为每张图都生成了配置要求的 14 项输出。

没有任何动作在主指标上胜过 no-AG：

| Action | Δ TOPIQ-NR [95% CI] | Δ HPSv2 | Δ CLIP | Δ clipped | Δ saturation |
|---|---:|---:|---:|---:|---:|
| conference expert | -0.020052 [-0.037713,-0.003316] | +0.004754 | -0.004435 | +0.010235 | +0.044811 |
| raw 0.001 | -0.005907 [-0.011608,-0.000872] | +0.001333 | -0.004632 | +0.002392 | +0.014894 |
| plain tangent 0.002 | -0.002836 [-0.006932,+0.001057] | +0.000756 | -0.003126 | -0.000102 | -0.000058 |
| cone 0.002 | -0.002091 [-0.006817,+0.002305] | +0.001133 | -0.003167 | -0.000059 | +0.000653 |

cone `0.002` 只比相同 scale 的 plain tangent 高 `+0.000745`，CI `[-0.001212,+0.002815]`；比 raw 高 `+0.003816`，CI `[-0.001550,+0.009215]`；比 expert 高 `+0.017961`，CI `[+0.002937,+0.033567]`。这些数字只说明 cone “少伤害了一些”，不能说明它有效，因为它仍低于 no-AG，而且没有通过预先登记的 CLIP 非劣门槛。patch-IR 也下降 `-0.017641`，没有局部细节改善的支持信号。

TOPIQ seed-CV 相对 no-AG 的结果是：全局 policy `-0.002148`，每 prompt 选择 `-0.003216`。按 HPS 选择虽然名义上提高 `+0.005022`，CI `[+0.000604,+0.009759]`，但 TOPIQ 下降 `-0.012735`，clipping 上升 `+0.001673`，saturation 上升 `+0.028848`。这说明选择器只是在迎合 HPS，同时破坏主指标和 guard，并不代表存在可用的自适应空间。

固定对比图位于 `outputs/exp_stage2_transfer/pilot_12prompt_3seed_v1/figs/all_actions_seed0.png`。expert 和 raw 带来了更明显的颜色、构图变化，但没有稳定修复结构或文字。因此第二阶段（Stage 2）没有救回第一阶段（Stage-1）的 Attention Guidance。S4 以及“高分辨率可能翻盘”的例外都到此关闭，不再做高分辨率 scale 调参。

## S4 之后：旧 TFSA residual 路线不再继续调参

S4 结束时，还没有登记 S5 的方法主张。此前四类算子加上一次高分辨率复查都失败了，这些证据说明：不能继续围绕同一个四通道 noisy-latent TFSA residual，只改 scale、schedule、频率权重、几何投影或 controller。

下一轮允许研究的假设必须更换“更新方向从哪里来”。一个可以认真检验的候选方案是 scheduler-consistent predicted-clean semantic transport：先拿到 scheduler 预测的干净样本 `x0`；再从冻结 UNet 的 self-attention 特征中提取空间传输关系，而不是对四通道 latent 直接做点积；然后对 `x0` 做固定均值和方差的传输；最后在保持预测噪声分量不变的前提下，重建下一步 latent。

这个方案直接检验：S0-S4 是否因为在噪声很强、语义很弱的 latent 上计算 TFSA 才失败。正式实现前，必须先完整检查相关工作，并精确验证 scheduler 重建是否一致。

任何这类新算子都必须先用一套全新的、不重合的 prompts，并先比较固定动作。对照必须包括 raw-latent TFSA、predicted-clean TFSA、semantic-feature transport、no-AG、论文版 expert，以及相同计算量的 post-step control；如果增加 UNet 调用，还必须加入调用次数相同的 score-correction baseline。仍沿用 `+0.005` TOPIQ 门槛、偏好与对齐非劣要求、像素 guard、高分辨率迁移和固定定性对比图。只有固定动作先赢，才有理由做搜索、蒸馏或 RL；如果再次得到空结果，就关闭 Attention Guidance 作为期刊扩展方向。

## S5：相关工作检查后的决定

`S5_RELATED_WORK.md` 记录了在 24 August 2026 固定下来的全文检查结果。PLADIS 和 GAG 已经覆盖 training-free、single-forward、sparse-versus-dense attention contrast，因此是必须加入的相同 NFE 基线。SAG、PAG、SEG 和 ASAG 属于需要额外 prediction 的上限对照组。

predicted `x_0`、内部特征、attention extrapolation、projection、adaptive schedule 和 RL 这些单独组件都已有相关工作。这里唯一要检验的是完整的 scheduler-consistent reciprocal semantic transport 算子，不宣称任何单个组件本身有新意。

## S5：已登记的机制假设

### 为什么换更新来源

S0-S4 可能失败，是因为四通道 noisy-latent 的点积没有足够强的语义。S5 不再从那里构图，而是读取标准 conditional branch 中这一层的 self-attention：

```text
up_blocks.0.attentions.0.transformer_blocks.0.attn1
```

它只读取结果，不替换 attention processor，也不改变该层输出。在 1024² 下，这一层对应 32x32 网格、20 个 heads 和 1280 个 channels。冻结诊断显示，在 timesteps `801/601/401/201/1`，距离超过四个 tokens 的 nonlocal mass 分别为 `0.8433/0.7424/0.7132/0.6930/0.6163`，但 normalized entropy 仍为 `0.9000/0.8636/0.8414/0.8202/0.7876`。

这意味着注意力覆盖范围很广，但也非常分散。直接用完整 dense attention 做传输，很可能把不同位置平均到一起。因此 S5 只保留“双方都把彼此当作重要邻居”的连接，也就是 reciprocal mutual-neighbor sparsification。

### 图怎么构造

对正常 conditional forward，定义：

```text
A = mean_h softmax(Q_h K_h^T / sqrt(d))
R_ij = sqrt(A_ij A_ji)
W = row_normalize(R restricted to mutual top-16 edges plus the diagonal)
c_t = 1 - mean_i H(A_i) / log(N).
```

大白话解释：`A` 是所有 attention heads 的平均连接强度；`R_ij` 只有在 `i` 看重 `j`、`j` 也看重 `i` 时才会大；`W` 对每个位置只保留最强的双向连接和它自己，再把每行归一化；`c_t` 是置信度，attention 越集中，置信度越高，越分散则更新越弱。

把 scheduler 返回的 `x_0` 按面积缩小到 attention 网格，计算 `d = upsample(W x_0 - x_0)`，再按 latent 通道把 `d` 投影到固定均值、固定方差的切空间。随后用固定角度 `theta * c_t` 在球面上移动，结果叫 `guided_x_0`。

scheduler 侧只允许下面这一种接法：

```text
step_output = scheduler.step(...)
u = guided_x_0 - step_output.pred_original_sample
latents = step_output.prev_sample + u.
```

scheduler 使用 Euler 和 epsilon prediction。S5 不得自己重新计算 `x_0`，也不得自己重写 Euler step，因为 fp16 下这样做的误差最高可到 `0.03125`。`theta=0` 必须精确复现 no-AG。整个算子不做 backpropagation，也不增加 UNet evaluation。

## S5：固定数据与对照

`eval-pipeline/prompts/s5_development.csv` 包含 12 条新 prompts，分别从 PartiPrompts 的 Complex、Fine-grained Detail、Properties & Positioning、Quantity、Writing & Symbols、Perspective 六类挑战中各取两条。`s5_smoke.csv` 另有两个不同 prompts。它们与仓库里此前所有 prompt CSV 都不重合。

来源 commit 是 `5a657978134374ce28973948331b319adef164bd`；在生成前，已经按 evaluation README 记录的 SHA-256 流程选定这些行。看过图片后不能替换 prompts。

必须包含这些对照：no-AG、论文版 expert、raw noisy-latent TFSA、plain predicted-clean TFSA、predicted-clean reciprocal latent affinity、reciprocal semantic affinity，以及把同一个 semantic graph 的行列一起打乱后再应用的版本。这个 permutation 保留 graph 的数值和计算量，但破坏它与空间位置的对应关系。

还要加入 CFG-only 5.0、官方 PLADIS（`alpha=1.5`、scale `2.0`）和官方 GAG（`lambda=10`、`eta=15`、`zeta=0`）。按论文登记方式，PLADIS/GAG 使用 CFG 5.0；所有 RepLDM 对照保持 CFG 7.5。必须报告 NFE、wall time 和 GPU 峰值显存。

## S5：冒烟测试和工程门槛

`eval-pipeline/configs/s5_smoke.yaml` 固定 top-k 为 `16`，使用上面指定的 semantic layer；匹配对照的 angle 为 `0.02`，semantic angle 范围为 `0.005/0.01/0.02/0.04`。在同一张 GPU 上运行两个 prompts、seed `0`。

冒烟测试仍只检查实现和灾难性参数，不能给动作排名，也不能从中挑一个孤立 angle。只有在任一 prompt 出现非有限输出、非零动作没有实际变化、TOPIQ 比 no-AG 低超过 `0.05`，或 clipped fraction 比 no-AG 高超过 `0.01` 时，才从连续区间的极端一侧删参数。

严格冒烟测试已在 24 August 2026 完成 `28/28` 个配对任务。两个 prompts 都生成了有限、非空的 1024² RGB PNG；no-AG 和 zero-angle identity 的 hash 完全相同，独立重复运行也复现了全部 28 个图像 hash 和必需 metadata。没有 semantic angle 触发预先登记的灾难性门槛，因此完整区间已固定到 `eval-pipeline/configs/s5_development.yaml`。这些分数只证明工程实现通过检查，不能用来选择赢家。

随后在当前运行时 commit `cb8eddd` 上重跑了同一个 smoke，目录为 `outputs/exp_s5/engineering_smoke_v2`。它的 28 个 PNG 与此前两轮 smoke 的 hash 完全一致，关键 metadata 也一致；因此早期工作树 provenance 问题不会改变区间判定。

在解释冒烟分数之前，unit test 和真实模型检查还必须证明：确实使用 scheduler 返回的 `x_0`；no-AG 和 zero-angle PNG hash 精确相同；`guided_x_0` 保持各通道均值和方差；reciprocal support 同时满足 mutual 和 row-stochastic；spatial permutation 是确定性的；重复运行能复现 hash；每个 sidecar 都记录 latency、peak memory、normalized affinity entropy、transport confidence，以及 transport 与 scheduler update 的 norm ratio。

## S5：正式开发实验的通过条件

冒烟测试后，先把完整、连续、未触发灾难性问题的 semantic angle 区间和全部匹配对照固定到新 YAML，再开始开发生成。运行已经固定的 12 个 prompts，seeds 为 `0,42,123`；同一 prompt/seed 的所有动作必须放在同一张 GPU 上。

主指标为 TOPIQ-NR。还要报告 HPSv2、ImageReward、patch-IR、CLIP alignment、aesthetic score、clipped fraction、saturation、contrast、colorfulness、sharpness、latency、memory 和每一个失败动作。统计方法为 prompt/seed 交叉 bootstrap 区间、prompt sign-flip test 和同指标 Holm correction。

S5 动作只有同时满足以下条件，才能进入下一步：

1. 相对 no-AG 的 TOPIQ 增益至少为 `+0.005`，95% CI 不跨零，同指标 Holm 校正后的检验值低于 `0.05`。
2. 在相同 NFE 记账方式下，直接胜过论文版 expert、raw TFSA、匹配的 latent-affinity 与 permuted-graph 对照、CFG 5.0、PLADIS 和 GAG。
3. HPSv2 和 CLIP 不能明显变差；clipped fraction 和 saturation 相对 no-AG 的增量分别不能超过 `0.001` 和 `0.005`。
4. 固定的全动作对比图必须显示真实的结构、计数、位置、文字或细节改善，不能只是全局颜色、对比度或锐度变化。

通过只代表可以继续做不重合 prompts 的确认实验和第二阶段（Stage-2）迁移，还不能开始 RL。如果结果仍为空，就关闭 Attention Guidance 作为期刊扩展的方向；之后不再继续搜索 angle、top-k、layer、schedule、reward 或 controller。

## S5：正式开发结果（Null）

正式目录为 `outputs/exp_s5/development_12prompt_3seed_v2`，由 `cb8eddd` 生成并通过 strict scorer。12 个 prompt、3 个 seed、14 个动作共 `504/504` 条记录完整；36 个 prompt/seed block 各自固定在一张 GPU 上。所有图像都是有效的 1024² RGB PNG，所有非零 semantic action 都记录了 50 个 transport steps、entropy、confidence 和 scheduler-update ratio。

TOPIQ-NR 的 crossed-bootstrap 相对 no-AG 结果如下（均未通过 `+0.005`、CI 排除零和 Holm 三项门槛）：

| action | mean delta | 95% CI |
|---|---:|---:|
| reciprocal semantic 0.005 | `+0.000459` | `[-0.000445, +0.001400]` |
| reciprocal semantic 0.01 | `+0.000509` | `[-0.000571, +0.001771]` |
| reciprocal semantic 0.02 | `-0.000233` | `[-0.001787, +0.001227]` |
| reciprocal semantic 0.04 | `-0.000775` | `[-0.002919, +0.001477]` |
| reciprocal semantic permuted 0.02 | `-0.000974` | `[-0.003585, +0.001618]` |

The matched latent, clean-TFSA, raw-TFSA, conference expert, PLADIS and GAG
controls also failed the registered S5 gate. HPSv2 and pixel guards do not
rescue the missing TOPIQ gain; the fixed seed-0 montage at
`outputs/exp_s5/development_12prompt_3seed_v2/figs/s5_structure_focus_seed0.png`
shows no stable
counting, position, text, or detail correction. The semantic and permuted
graphs are statistically indistinguishable at the required scale, so the
development run does not establish a causal spatial-graph benefit.

Per the preregistration, S5 is closed as an Attention Guidance extension: do
not search another angle, top-k, layer, schedule, reward, or controller, and do
not start RL on this failed static family. The separately documented
low-capacity latent-renderer proposal may be registered as a new project, but
it must compete directly with no-AG and cannot be presented as a rescue of S5.
