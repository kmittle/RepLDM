# 实验结果与阶段性结论

> 分支：`rl-version` ｜ 方法与停止规则：[`REVIEWER_PROTOCOL.md`](./REVIEWER_PROTOCOL.md)
> 截止提交：频谱 pilot `63184bc`，幅度 follow-up `c385223`，固定矩 pilot `65ba734`，轨迹锥 pilot `b0aa343`，Stage-2 pilot `a31cb75`。本文只记录当前产物可复核的事实。

## 阅读说明

本文把每一种 Attention Guidance 设置称为一个“动作”。常用名称如下：

- `no_ag`：完全不使用 Attention Guidance，是所有差值的基线。
- `conference expert`：会议论文中使用的人工设定。
- `raw` 或 `scalar`：直接用一个标量缩放原始 TFSA 残差。
- `latent`：扩散模型在去噪过程中使用的中间特征，不是最终像素图像。TFSA 残差是 TFSA 处理前后 latent 的差，代表 Attention Guidance 想额外施加的更新。
- `scheduler`：根据模型预测把当前 latent 推进到下一去噪状态的规则。本文把这一步实际产生的变化视为基础模型的“正常去噪方向”。
- `scale`：动作强度。方向不变时，scale 越大，额外更新越强。
- `headroom`：相对基线仍可获得的改进空间。没有正向 headroom，意味着当前动作没有为搜索、控制器或 RL 提供可学习的收益。
- `guard`：不直接代表图像质量、但用于监测副作用的指标。本文主要关注 clipping 和 saturation；它们上升通常表示过曝、欠曝或过饱和风险增加。
- `pilot`：用于判断方向是否值得继续的初步实验；`follow-up` 是看到初步结果后进行的补充检查，证据等级低于独立确认实验。

表中的 `Δ` 均为“该动作减去比较基线”。对 TOPIQ-NR、HPSv2、ImageReward 和 CLIP 等质量或偏好代理指标，正值更好；对 clipped fraction 和 saturation 等 guard，正值表示副作用增加。`CI` 是置信区间；区间跨过零时，现有数据不能确认差值方向稳定。Holm 校正用于控制同时比较多个动作时的假阳性风险。“非劣界”是事先允许的最大退步幅度；越过它就不能声称该指标“不比基线差”。

“完整配对”表示同一 `(prompt, seed)` 下的所有动作都在同一张 GPU 上生成，只改变待比较的动作。“seed-CV”表示留出一个 seed 做测试，仅用其他 seed 选择动作；它检查动作选择能否跨 seed 重复，但不等于能泛化到新 prompt。`prompt-disjoint` 表示确认集使用与开发阶段完全不重叠的新 prompts。

## 0. 先看结论

**G1（频谱 headroom）、G3（内容自适应 headroom）、S2（固定矩切空间）、S3（轨迹锥）和 S4（2048² 目标域迁移）均失败，因此这些动作空间都不应训练 RL 控制器。** 频谱分配相对统一标量大幅减轻了 TOPIQ-NR 损伤，但没有优于 `no_ag`。把幅度从 `0.004` 降到 `0.001` 只会收敛回 `no_ag`，未出现内部质量最优点。固定矩约束成功消除了饱和度漂移；轨迹锥进一步删除与 scheduler 转移相反的切向分量，但两者都没有质量或偏好收益。Stage 2 没有逆转这一排序，反而放大了 expert/raw 的质量与像素 guard 损伤。

直白地说：当前实验依次尝试了改变频率分配、减小幅度、限制均值与方差漂移、删除与正常去噪方向相反的分量，以及检查高分辨率阶段是否会改变结果。它们最多只能减少某些损伤，没有任何一种稳定胜过“不使用 Attention Guidance”。因此不能把这些动作交给 RL 控制器继续优化。

这只是对当前动作算子和当前实验范围的否定性 pilot 结论，不等于证明“Attention Guidance 在所有任务上都无效”。样本只有 12 prompts、3 seeds 和单一 SDXL backbone；S4 虽然评估了最终 2048² 图像，仍然复用了开发 prompts，也没有进行人类偏好盲评。因此这些数值不能被包装成最终 benchmark 结果。

### LR-1 合规 latent-renderer 搜索（独立 null result）

为检验一个小型、结构条件 latent renderer 是否有足够的静态 headroom，另行
冻结了 48 个 prompt 的 train split、10 个固定六维动作和 seeds `7,19,73`。
正式目录 `outputs/latent_renderer/lr1_fixed_train_searchseeds_v2` 包含
`1440/1440` 完整记录；144 个 `(prompt,seed)` block 均单卡完成，严格审计
通过，最大 trust ratio `0.0112049`，最大绝对 moment errors
`1.19e-7/7.15e-7`，所有评分键 finite。该 run 仅用于 train-side action
selection，不是功效确认集。

选择规则在查看结果前已冻结：只用 HPSv2，要求 paired mean 和 crossed
bootstrap 95% CI 下界都高于 `no_ag` 的零差值，并同时满足 CLIP、clipping、
saturation 和 renderer finite/moment/trust guards。没有候选满足这一门槛，
selector 返回 `no_ag`：

| candidate | Δ HPSv2 [95% CI] | Δ CLIP | Δ clipped | Δ saturation |
|---|---:|---:|---:|---:|
| `semantic_neg` | `+0.000149 [-0.000173,+0.000483]` | `-0.000424` | `+0.000301` | `+0.000269` |
| `spectral_high_pos` | `+0.000088 [-0.000292,+0.000488]` | `-0.000242` | `-0.000076` | `-0.000567` |
| `balanced_pos` | `+0.000132 [-0.000212,+0.000533]` | `-0.000331` | `-0.000087` | `-0.000557` |
| `spectral_mid_pos` | `-0.000011 [-0.000356,+0.000326]` | `-0.000154` | `+0.000221` | `-0.000037` |

An independent descriptive comparison across the registered train metrics found
no TOPIQ-NR interval wholly above zero; its largest exploratory mean was only
`+0.000457` (`spectral_mid_pos`, CI `[-0.000418,+0.001450]`). The action-dependent
pixel changes therefore do not establish perceptual or structural improvement.
Because selection returned `no_ag`, the preregistered protocol correctly emits
no validation configuration, does not use seeds `11,29,101` or `0,42,123`, and
does not authorize distillation or RL. This is a complete negative LR-1 result,
not evidence that a later renderer or representation factorial has been tested.

## 1. 实验规模与数据完整性

| 实验 | 设计 | 证据用途 |
|---|---|---|
| 旧 scalar sweep | 12×3×7 = 252 | **作废为配对实验**；36/36 区块跨 GPU |
| spectral pilot | 12×3×11 = 396 | 预注册 G1 动作筛选 |
| amplitude follow-up | 12×3×10 = 360 | 后验替代解释检查，不是确认集 |
| moment-tangent pilot | 12×3×10 = 360 | S2 冻结开发实验 |
| trajectory-cone pilot | 12×3×9 = 324 | S3 冻结开发实验 |
| Stage-2 transfer pilot | 12×3×5 = 180 | S4 预注册 2048² 目标域诊断 |

频谱 pilot 和 amplitude follow-up 都采用完整配对：同一 `(prompt, seed)` 下的动作固定在同一张 GPU 上。两轮共有 756 条记录；follow-up 与 pilot 重复的 4 个动作形成 144 对 PNG，其 SHA-256 和全部评分逐值相同。S2 的 360 条记录和 S3 的 324 条记录也都满足同卡配对；S3 中的 144 个历史控制结果精确复现了 S2 的 PNG 哈希。这些重复结果证明生成流程可以复现，但不能被当作新增的独立样本。

恒定标量动作 `scalar_0.004` 与三个频带使用相同增益的动作 `[0.004, 0.004, 0.004]` 在 36/36 个配对中像素完全一致。这说明频带实现能在等增益条件下正确退化为原标量实现。另有 21 项单元测试覆盖工程正确性；这些检查只能证明实现按预期工作，不能证明算法有效。

## 2. 预注册频谱实验（Spectral Pilot）

这个实验要回答的问题是：与其给所有频率统一乘同一个标量，是否只增强低频、中频或高频会更好。`no_ag` 的绝对均值为 TOPIQ-NR `0.710145`、HPSv2 `0.296071`、ImageReward `0.850563`、clipped fraction `0.003017`。下表均为相对 `no_ag` 的配对差值；TOPIQ 区间采用同时重采样 prompt 和 seed 的 crossed bootstrap。

| 动作 | Δ TOPIQ-NR [95% CI] | Δ HPSv2 | Δ clipped | Δ saturation |
|---|---:|---:|---:|---:|
| mid only 0.004 | -0.002014 [-0.007778, +0.003404] | +0.001614 | +0.000944 | +0.008766 |
| low only 0.004 | -0.008714 [-0.018697, -0.001165] | -0.000444 | +0.003865 | +0.034034 |
| high only 0.004 | -0.020647 [-0.049706, -0.003405] | +0.003821 | +0.004682 | +0.021142 |
| conference expert | -0.010174 [-0.026698, +0.002482] | +0.003676 | +0.009746 | +0.038096 |
| scalar/equal bands 0.004 | -0.051005 [-0.090117, -0.022882] | +0.004520 | +0.017929 | +0.070444 |

结果很直接：11 个动作中没有一个提高主指标。HPSv2 虽有小幅正差，但置信区间均跨零；ImageReward 也没有确认到稳定收益。`mid_only` 只是损伤最小的动作，不是优于 `no_ag` 的获胜动作，而且它仍明显提高了饱和度。定性拼图 `outputs/exp_spectral_headroom/pilot_12prompt_3seed_v1/figs/core_actions_seed0.png` 显示，较强 guidance 主要增加色彩、对比度和锐化，没有稳定修复结构或文字。由于这项观察没有盲化，只能用作诊断。

## 3. 后验检查：是否只是幅度过强

频谱实验失败后，一个可能的替代解释是：动作方向本身也许有效，只是 `0.004` 的幅度过强。follow-up 专门检查这一解释。所有 TOPIQ 差值仍以 `no_ag` 为基线：

| 幅度 | mid-only Δ TOPIQ | scalar Δ TOPIQ |
|---:|---:|---:|
| 0.001 | -0.000121 | -0.001570 |
| 0.002 | -0.000458 | -0.007160 |
| 0.003 | -0.000804 | -0.023781 |
| 0.004 | -0.002014 | -0.051005 |

`mid_only_0.001` 的 95% CI 为 `[-0.002610, +0.002767]`。随着幅度减小，结果只是逐渐回到 `no_ag`，没有在中间幅度出现正向最优点；相反，scalar 的损伤会随幅度增大而快速恶化。与此同时，mid-only 的 saturation 增量从 `+0.002084` 单调增至 `+0.008766`。因此数据不支持“只要把幅度调小就能获得收益”的解释。由于本轮复用了同一批 prompts，而且是在看到主 pilot 后才触发，它只能用于排除这个替代解释，不能再从中挑选动作并声称获得了确认性 winner。

## 4. 内容自适应是否还有改进空间

即使所有固定动作都没有获胜，仍可能存在另一种解释：不同 prompt 适合不同动作，逐 prompt 选择也许能获得收益。[`analyze_adaptivity.py`](./eval-pipeline/analyze_adaptivity.py) 用留一 seed 交叉验证检查这一点。对每个 held-out seed，脚本只用另外两个 seed 的 TOPIQ-NR 选择一个全局统一动作，以及每个 prompt 各自的动作；`no_ag` 始终包含在候选集中。下表报告留出 seed 上的结果：

| 动作集 | 全局 vs no-AG | 逐 prompt vs no-AG | 逐 prompt vs 全局 |
|---|---:|---:|---:|
| spectral pilot | +0.000000 [0, 0] | -0.003802 [-0.009381, +0.001590] | -0.003802 [-0.009381, +0.001590] |
| amplitude follow-up | -0.000902 [-0.003760, +0.001298] | -0.000160 [-0.006398, +0.005128] | +0.000743 [-0.006168, +0.005500] |

主 pilot 的全局选择在三个 fold 中每次都是 `no_ag`。逐 prompt 策略不仅没有提高 TOPIQ，还使 clipped fraction 增加 `+0.002629 [ +0.000796, +0.005064 ]`、saturation 增加 `+0.020408 [ +0.006689, +0.041321 ]`；两项副作用在全局 Holm 校正后仍显著。follow-up 中逐 prompt 相对全局动作的微小正 headroom 区间跨零，而逐 prompt saturation 仍增加 `+0.015126 [ +0.002757, +0.034040 ]`。HPSv2 在两轮逐 prompt 评估中的差值分别为 `+0.000275` 和 `+0.001370`，区间也都跨零。

这项检验只衡量同一个 prompt 上的动作选择能否跨 seed 保持稳定，尚未测试对 unseen prompt 的泛化。因此它已经是一个偏乐观的上界。连这个上界都没有通过，当前数据就没有为训练 prompt-conditioned controller 提供实证依据。

## 5. 决策门与后续路线

这里的 Gate 是开始下一阶段前必须满足的条件。前一项失败后，依赖它的后续实验不再继续，以避免用更复杂的搜索或 RL 掩盖基础动作本身没有收益的问题。

| Gate | 判定 | 后续动作 |
|---|---|---|
| G0 correctness | 通过 | 保留频带算子、trust cap 和配对工具 |
| G1 spectral headroom | **失败** | 停止当前 spectral/RL 路线 |
| G2 schedule granularity | 不执行 | 被 G1 阻断 |
| G3 adaptivity | **失败** | 不做 search-distill 或 prompt controller |
| G4 RL necessity | **拒绝进入** | 不以调 reward/控制器复杂度绕过失败门 |
| G5 transfer | **失败** | 最终 2048² 图像仍无正向 headroom；关闭目标域例外 |

若继续研究，必须先提出**实质不同的残差算子**，而不是在当前频带增益上叠加 RL。新算子首先要以固定动作在全新的 prompt-disjoint 数据集上同时优于 `no_ag`、conference expert 和等预算 scalar，并且不能恶化 clipping/saturation。只有通过这个基础门槛后，才值得设计约 50 prompts × ≥5 seeds、第二 backbone、Stage-2/ControlNet，以及随机左右顺序的人类配对盲评。

## 6. 复现

```bash
# 预注册动作的配对统计
/home/bycao/miniforge3/envs/repldm_eval/bin/python eval-pipeline/compare_actions.py \
  --run_dir outputs/exp_spectral_headroom/pilot_12prompt_3seed_v1 \
  --baseline no_ag \
  --metrics topiq_nr,hpsv2,imagereward,patch_ir_mean,clip_cosine,aesthetic,clipped_fraction,mean_saturation

# 留一 seed 的自适应上界；对 follow-up 替换 run_dir 即可
/home/bycao/miniforge3/envs/repldm_eval/bin/python eval-pipeline/analyze_adaptivity.py \
  --run_dir outputs/exp_spectral_headroom/pilot_12prompt_3seed_v1 \
  --selection_metric topiq_nr
```

生成与严格离线评分命令见 [`eval-pipeline/README.md`](./eval-pipeline/README.md)。按照仓库规则，`outputs/` 和 checkpoint 不进入 Git。

## 7. 这些结果可以和不可以说明什么

- 旧 252 图 cross-GPU sweep 没有满足同卡配对条件，因此此前基于它报告的配对 CI、显著性和 oracle gap 全部撤回。
- primary pilot、follow-up 和动作选择都参与了方法搜索或决策，不能再次用作最终无偏测试集。
- TOPIQ、HPSv2、ImageReward、CLIP 和 aesthetic 都只是代理指标；像素统计只用于暴露 reward hacking，不能单独证明感知质量提高。
- “CI 跨零”不代表两个方法被证明严格等价；但当前 RL 扩展必须先展示正向收益，而这些结果没有达到该前提，因此足以否决当前 RL 主张。

## 8. S2 保持均值和方差的更新（固定矩切空间，失败）

S2 要检查的是：原始 TFSA 更新会同时改变 latent 的空间结构、通道均值和通道方差；其中均值与方差的漂移可能只是通过颜色、对比度和饱和度影响评分，而不是真正改善结构。为分离这两种作用，本实验把更新限制在“保持每个通道均值和方差不变”的方向上。数学上，这些允许的方向构成固定矩约束面的切空间，因此称为“固定矩切空间”。

表中的动作可以直观理解为：`mean-centered` 只去掉残差的空间均值；`moment tangent` 同时去掉会改变均值和方差的分量；`tangent-rescaled` 再把约束后更新的能量拉回原始残差的水平，用于区分“方向约束有效”还是“更新仅仅变弱”。如果原始方法的主要问题确实来自均值或方差漂移，那么这些动作应在减少饱和副作用的同时保留或提高质量。

`outputs/exp_moment_tangent/development_12prompt_3seed_v1` 包含 12 prompts × 3 seeds × 10 actions = 360 个完整配对记录。四张卡各生成 90 条，36 个 `(prompt, seed)` 区块均未跨卡；108 张历史控制图的 PNG 哈希完全复现。以下是相对 `no_ag` 的预注册主结果：

| 动作 | Δ TOPIQ-NR [95% CI] | Δ HPSv2 | Δ clipped | Δ saturation |
|---|---:|---:|---:|---:|
| mean-centered 0.001 | -0.000946 [-0.005246,+0.003537] | +0.000336 | +0.001048 | +0.006411 |
| moment tangent 0.001 | -0.002316 [-0.005221,-0.000162] | -0.000593 | -0.000283 | -0.000862 |
| moment tangent 0.002 | -0.002268 [-0.005403,+0.000653] | +0.000054 | +0.000031 | -0.000997 |
| tangent-rescaled 0.001 | -0.003763 [-0.005925,-0.001483] | -0.000251 | +0.000017 | -0.000916 |
| raw 0.001 | -0.001570 [-0.006276,+0.003529] | +0.000543 | +0.001399 | +0.013449 |

**质量结果：**没有动作达到 `ΔTOPIQ >= +0.005`，也没有任何正收益的置信区间排除零。`tangent-rescaled_0.001` 在 TOPIQ 指标内经过 Holm 校正后仍显著为负（`p=0.025920`）。因此这些动作没有通过质量门槛。

**机制结果：**固定矩约束把 raw 的饱和度增量压到接近零，说明“保持均值和方差”的几何约束确实生效。但是 HPSv2、CLIP 和定性 montage 都没有稳定改善。这意味着去掉颜色和对比度这条捷径后，TFSA 剩余的无条件空间重排本身仍没有可测的质量 headroom。

**自适应检查与决策：**TOPIQ 留一 seed 的逐 prompt 选择相对 `no_ag` 为 `+0.001791 [-0.004392,+0.007529]`，而且 clipping/saturation 上升。按 HPSv2 选择时，逐 prompt 策略比全局 expert 低 `-0.002082 [-0.003872,-0.000434]`。因此 S2 不进入 RL；下一轮若继续研究，必须改变残差如何根据当前去噪状态确定方向，而不是继续扫描 scale。

## 9. S3 删除与正常去噪方向相反的更新（轨迹锥，失败）

S2 已经消除了均值和方差漂移，但质量仍未提高。S3 进一步检查另一种可能：固定矩后的 TFSA 更新，有些分量可能正好与基础扩散模型本次 scheduler 去噪转移的方向相反，从而抵消模型原本要完成的去噪。

“轨迹锥”做的事情并不复杂。先把 TFSA 更新和 scheduler 已经走出的更新都投影到 S2 的固定矩切空间，再从 TFSA 更新中删掉“朝 scheduler 反方向”的那部分。删完后，允许的更新方向都不会与 scheduler 更新负相关；这些方向在几何上形成一个半空间，也可看作一个锥，因此得名。`cone` 表示应用这项方向约束，`cone-rescaled` 表示约束后再匹配原始残差能量。关键检验是：如果 S2 的损失主要由反向分量造成，那么相同 scale 下的 cone 应明显优于 plain tangent。

`outputs/exp_trajectory_cone/development_12prompt_3seed_v1` 包含 12 prompts × 3 seeds × 9 actions = 324 个完整配对记录。四张卡各生成 81 条，36 个 `(prompt, seed)` 区块均未跨卡；144 张 no-AG、expert、raw 和 plain-tangent 控制图与 S2 的 PNG 哈希逐一相同。全部评分键都存在，并且都是有限值。

| 动作 | Δ TOPIQ-NR [95% CI] | Δ HPSv2 | Δ clipped | Δ saturation |
|---|---:|---:|---:|---:|
| cone 0.001 | -0.002519 [-0.005576,-0.000152] | -0.000553 | -0.000175 | -0.000812 |
| cone 0.002 | -0.001740 [-0.005274,+0.001283] | +0.000085 | +0.000050 | -0.000385 |
| cone 0.004 | -0.004220 [-0.010200,+0.001771] | +0.000665 | +0.000600 | -0.000368 |
| cone-rescaled 0.001 | -0.003178 [-0.005424,-0.000840] | +0.000173 | +0.000085 | -0.000642 |
| cone-rescaled 0.002 | -0.005843 [-0.010720,-0.000922] | +0.000346 | +0.000998 | +0.000022 |

**直接结果：**没有动作达到 `+0.005` 的主门槛。为单独判断轨迹锥约束是否有效，应比较相同 scale 的动作。cone `0.002` 相对 plain tangent `0.002` 的 TOPIQ 增量只有 `+0.000528 [-0.000412,+0.001596]`，HPSv2 差值为 `+0.000031`。也就是说，删除反向分量并没有带来可确认的改善，因此数据不支持“反向分量是 S2 失败主因”的解释。

**自适应检查：**以 TOPIQ 做 seed-CV 时，三个 fold 选出的全局动作都是 `no_ag`。逐 prompt 选择相对 no-AG 为 `+0.001120 [-0.004272,+0.006516]`，但 clipping 增加 `+0.001262`、saturation 增加 `+0.012097`。改用 HPSv2 选择时，逐 prompt 策略反而比全局 expert 低 `-0.001604 [-0.003455,-0.000190]`，而且像素 guard 显著恶化。

**定性观察与决策：**固定拼图 `outputs/exp_trajectory_cone/development_12prompt_3seed_v1/figs/all_actions_seed0.png` 覆盖全部 12 prompt 和全部 9 个动作，没有显示稳定的结构或文字修复。S3 因此按照停止规则关闭，不再继续扫描幅度，也不进入 RL。

S3 失败后，按 `MODEL_ITERATIONS.md` 中预先登记的规则，唯一允许继续做的是 2048² Stage-2 目标域诊断。前面的 S0-S3 都在 1024² 的基础分辨率图像上评估，而论文的核心场景是最终高分辨率图像；这项诊断只检查高分辨率阶段是否会改变前面的负面排序，不代表 S0-S3 重新通过，也不能借此重新调参。

## 10. S4 检查 2048² 高分辨率 Stage 2 是否改变结论（失败）

这项实验不搜索新动作，只把前面已经冻结的代表性动作放进完整的高分辨率流程，观察高分辨率重采样会放大、减弱还是逆转基础分辨率阶段留下的差异。

正式实验前，工程 smoke 修复了两个会破坏配对的问题：Stage-2 noise 改为使用每个任务自己的 generator；当 `models_to_cpu=True` 时，普通 decoder 会显式把 VAE 和 latent 放回执行设备。两个独立 smoke run 的三个动作哈希达到 3/3 精确复现；同一 run 中重复的 no-AG 完全一致，而 expert 与它不同。正常 2048² 路径的峰值显存约为 21.2 GB。

正式 run `outputs/exp_stage2_transfer/pilot_12prompt_3seed_v1` 包含 180 张最终 2048² 图像。36 个区块都是五个动作在同一张卡上的完整配对，四张卡各生成 45 条；全部图像、metadata 和 14 个评分键完整，两个 smoke 控制的哈希再次复现。no-AG 的绝对均值为 TOPIQ `0.440022`、HPSv2 `0.290171`、ImageReward `0.839265`、CLIP cosine `0.345825`、clipped fraction `0.002606`。

| 动作 | Δ TOPIQ-NR [95% CI] | Δ HPSv2 | Δ CLIP | Δ clipped | Δ saturation |
|---|---:|---:|---:|---:|---:|
| conference expert | -0.020052 [-0.037713,-0.003316] | +0.004754 | -0.004435 | +0.010235 | +0.044811 |
| raw 0.001 | -0.005907 [-0.011608,-0.000872] | +0.001333 | -0.004632 | +0.002392 | +0.014894 |
| plain tangent 0.002 | -0.002836 [-0.006932,+0.001057] | +0.000756 | -0.003126 | -0.000102 | -0.000058 |
| cone 0.002 | -0.002091 [-0.006817,+0.002305] | +0.001133 | -0.003167 | -0.000059 | +0.000653 |

cone 相对相同 scale 的 tangent 仅高 `+0.000745 [-0.001212,+0.002815]`；相对 raw 高 `+0.003816 [-0.001550,+0.009215]`。相对 expert 的配对差值为 `+0.017961 [+0.002937,+0.033567]`。这些差值说明 cone 的损伤小于其他 guidance 动作，但不代表它有效：cone 仍低于 no-AG，没有达到 `+0.005`，而且 CLIP CI 下界 `-0.010214` 超出预设的非劣界。Patch-IR 也下降 `-0.017641`，没有确认到局部细节收益。

TOPIQ seed-CV 的全局选择和逐 prompt 选择结果分别为 `-0.002148` 和 `-0.003216`。按 HPSv2 选择的逐 prompt 策略虽然达到 `+0.005022 [+0.000604,+0.009759]`，却同时造成 TOPIQ `-0.012735`、clipping `+0.001673`、saturation `+0.028848`。这属于明确的 proxy/guard 冲突：一个偏好代理指标上升了，但主质量指标和安全监测指标同时恶化，不能视为有效收益。固定 seed-0 拼图 `outputs/exp_stage2_transfer/pilot_12prompt_3seed_v1/figs/all_actions_seed0.png` 也没有显示稳定的结构或文字修复。

因此 S4 失败，高分辨率例外到此关闭：不再扫描 2048² 下的 scale 或 schedule，也不进入 RL。

如果继续提出候选方法，必须更换“更新方向从哪里来”，而不是继续调整现有动作。具体来说，应停止直接在 raw 四通道 noisy latent 上计算 TFSA 残差，改为从 scheduler 预测的干净样本或 UNet 语义特征中构造空间更新，再以符合 scheduler 轨迹的方式还原下一步 latent。新方法必须先在新的 prompt-disjoint 数据上用固定静态动作证明存在 headroom。若这个静态门槛仍然失败，就应放弃把 Attention Guidance 作为期刊扩展的主轴。
