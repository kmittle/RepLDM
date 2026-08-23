# 实验结果与阶段性结论

> 分支：`rl-version` ｜ 方法与停止规则：[`REVIEWER_PROTOCOL.md`](./REVIEWER_PROTOCOL.md)
> 截止提交：频谱 pilot `63184bc`，幅度 follow-up `c385223`，固定矩 pilot `65ba734`，轨迹锥 pilot `b0aa343`。本文只记录当前产物可复核的事实。

## 0. 审稿结论

**G1（频谱 headroom）、G3（内容自适应 headroom）、S2（固定矩切空间）和 S3（轨迹锥）均失败，因此这些动作空间都不应训练 RL 控制器。** 频谱分配相对统一标量大幅减轻了 TOPIQ-NR 损伤，但没有优于 `no_ag`。把幅度从 `0.004` 降到 `0.001` 只会收敛回 `no_ag`，未出现内部质量最优点。固定矩约束成功消除了饱和度漂移；轨迹锥进一步删除与 scheduler 转移相反的切向分量，但两者都没有质量或偏好收益。

这是对当前动作算子和实验域的否定性 pilot 结论，不是“Attention Guidance 在所有任务上无效”的等价性证明。样本只有 12 prompts、3 seeds、SDXL Stage-1 1024²，且没有盲评人类偏好；因此不得把这些数值包装成最终 benchmark 结果。

## 1. 实验与完整性审计

| 实验 | 设计 | 证据用途 |
|---|---|---|
| 旧 scalar sweep | 12×3×7 = 252 | **作废为配对实验**；36/36 区块跨 GPU |
| spectral pilot | 12×3×11 = 396 | 预注册 G1 动作筛选 |
| amplitude follow-up | 12×3×10 = 360 | 后验替代解释检查，不是确认集 |
| moment-tangent pilot | 12×3×10 = 360 | S2 冻结开发实验 |
| trajectory-cone pilot | 12×3×9 = 324 | S3 冻结开发实验 |

频谱 pilot 与 amplitude follow-up 的每个 `(prompt, seed)` 动作区块均固定在同一 GPU；两轮共 756 条记录，follow-up 与 pilot 重复的 4 个动作共 144 对 PNG，其 SHA-256 和全部评分逐值相同。S2 的 360 条和 S3 的 324 条也都完整同卡配对；S3 中 144 个历史控制哈希精确复现 S2。重复控制用于工程复现，不是新增独立证据。

恒定 `scalar_0.004` 与等增益频带 `[0.004, 0.004, 0.004]` 在 36/36 配对中像素完全一致，验证频带实现退化到原标量行为。工程正确性由 21 项单元测试覆盖；它不等同于算法收益。

## 2. 预注册 Spectral Pilot

`no_ag` 的绝对均值为 TOPIQ-NR `0.710145`、HPSv2 `0.296071`、ImageReward `0.850563`、clipped fraction `0.003017`。以下均为相对 `no_ag` 的配对差值；TOPIQ 区间采用 prompt/seed crossed bootstrap。

| 动作 | Δ TOPIQ-NR [95% CI] | Δ HPSv2 | Δ clipped | Δ saturation |
|---|---:|---:|---:|---:|
| mid only 0.004 | -0.002014 [-0.007778, +0.003404] | +0.001614 | +0.000944 | +0.008766 |
| low only 0.004 | -0.008714 [-0.018697, -0.001165] | -0.000444 | +0.003865 | +0.034034 |
| high only 0.004 | -0.020647 [-0.049706, -0.003405] | +0.003821 | +0.004682 | +0.021142 |
| conference expert | -0.010174 [-0.026698, +0.002482] | +0.003676 | +0.009746 | +0.038096 |
| scalar/equal bands 0.004 | -0.051005 [-0.090117, -0.022882] | +0.004520 | +0.017929 | +0.070444 |

11 个动作中没有一个提高主指标。HPSv2 的小幅正差均跨零；ImageReward 也没有确认信号。`mid_only` 是损伤最小而非获胜者，并且饱和度显著上升。定性拼图 `outputs/exp_spectral_headroom/pilot_12prompt_3seed_v1/figs/core_actions_seed0.png` 显示较强 guidance 主要增加色彩、对比和锐化，未稳定修复结构或文字；该观察未盲化，只作诊断。

## 3. 后验幅度检查

follow-up 检查“0.004 仅仅过强”的解释。所有 TOPIQ 差值仍以 `no_ag` 为基线：

| 幅度 | mid-only Δ TOPIQ | scalar Δ TOPIQ |
|---:|---:|---:|
| 0.001 | -0.000121 | -0.001570 |
| 0.002 | -0.000458 | -0.007160 |
| 0.003 | -0.000804 | -0.023781 |
| 0.004 | -0.002014 | -0.051005 |

`mid_only_0.001` 的 95% CI 为 `[-0.002610, +0.002767]`。减小幅度使结果趋近 `no_ag`，没有出现正向内部最优；scalar 的损伤随幅度快速增大。与此同时，mid-only 的 saturation 增量从 `+0.002084` 单调增至 `+0.008766`。由于本轮使用同一批 prompts 且由主 pilot 触发，它只能关闭替代解释，不能用来选择一个再声称为确认性 winner 的动作。

## 4. 自适应 Headroom

[`analyze_adaptivity.py`](./eval-pipeline/analyze_adaptivity.py) 对每个 held-out seed，仅用另外两个 seed 的 TOPIQ-NR 选择一个全局动作和逐 prompt 动作；`no_ag` 始终是候选。下表是严格留出后的结果：

| 动作集 | 全局 vs no-AG | 逐 prompt vs no-AG | 逐 prompt vs 全局 |
|---|---:|---:|---:|
| spectral pilot | +0.000000 [0, 0] | -0.003802 [-0.009381, +0.001590] | -0.003802 [-0.009381, +0.001590] |
| amplitude follow-up | -0.000902 [-0.003760, +0.001298] | -0.000160 [-0.006398, +0.005128] | +0.000743 [-0.006168, +0.005500] |

主 pilot 的全局选择在三个 fold 都是 `no_ag`。逐 prompt 策略还使 clipped fraction 增加 `+0.002629 [ +0.000796, +0.005064 ]`、saturation 增加 `+0.020408 [ +0.006689, +0.041321 ]`，两者全局 Holm 校正后仍显著。follow-up 的微小正 headroom 区间跨零，逐 prompt saturation 仍增加 `+0.015126 [ +0.002757, +0.034040 ]`。HPSv2 在两轮逐 prompt 评估中的差值分别为 `+0.000275` 和 `+0.001370`，区间均跨零。

这个检验只衡量同一 prompt 跨 seed 的选择稳定性，尚未测试 unseen-prompt 泛化。连这一乐观上界都未通过，训练 prompt-conditioned controller 缺乏实证入口。

## 5. Gate 决策与算法路线

| Gate | 判定 | 后续动作 |
|---|---|---|
| G0 correctness | 通过 | 保留频带算子、trust cap 和配对工具 |
| G1 spectral headroom | **失败** | 停止当前 spectral/RL 路线 |
| G2 schedule granularity | 不执行 | 被 G1 阻断 |
| G3 adaptivity | **失败** | 不做 search-distill 或 prompt controller |
| G4 RL necessity | **拒绝进入** | 不以调 reward/控制器复杂度绕过失败门 |
| G5 transfer | 仅作 S4 目标域诊断 | 检查 1024² proxy 是否掩盖最终 2048² 效应；不能绕过 G1-G4 |

若继续研究，必须先提出**实质不同的残差算子**，而不是在当前频带增益上叠加 RL。新算子应先用固定动作在全新的 prompt-disjoint 集上同时优于 `no_ag`、conference expert 和等预算 scalar，并保持 clipping/saturation 非劣。只有该门通过后，才值得设计约 50 prompts × ≥5 seeds、第二 backbone、Stage-2/ControlNet 和随机左右顺序的人类配对盲评。

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

生成与严格离线评分命令见 [`eval-pipeline/README.md`](./eval-pipeline/README.md)。`outputs/` 和 checkpoint 按仓库规则保持不跟踪。

## 7. 报告边界

- 旧 252 图 cross-GPU sweep 的配对 CI、显著性和 oracle gap 全部撤回。
- primary pilot、follow-up 和动作选择都属于搜索/决策数据，不得作为最终无偏测试集复用。
- TOPIQ、HPSv2、ImageReward、CLIP 和 aesthetic 都是代理指标；像素统计只用于发现 reward hacking。
- “CI 跨零”不证明严格等价，但足以否决当前需要正向收益才能成立的 RL 扩展主张。

## 8. S2 固定矩切空间（失败）

`outputs/exp_moment_tangent/development_12prompt_3seed_v1` 包含 12 prompts × 3 seeds × 10 actions = 360 个完整配对记录。四卡各 90 条，36 个区块均未跨卡；108 张历史控制图的 PNG 哈希完全复现。以下为相对 `no_ag` 的预注册主结果：

| 动作 | Δ TOPIQ-NR [95% CI] | Δ HPSv2 | Δ clipped | Δ saturation |
|---|---:|---:|---:|---:|
| mean-centered 0.001 | -0.000946 [-0.005246,+0.003537] | +0.000336 | +0.001048 | +0.006411 |
| moment tangent 0.001 | -0.002316 [-0.005221,-0.000162] | -0.000593 | -0.000283 | -0.000862 |
| moment tangent 0.002 | -0.002268 [-0.005403,+0.000653] | +0.000054 | +0.000031 | -0.000997 |
| tangent-rescaled 0.001 | -0.003763 [-0.005925,-0.001483] | -0.000251 | +0.000017 | -0.000916 |
| raw 0.001 | -0.001570 [-0.006276,+0.003529] | +0.000543 | +0.001399 | +0.013449 |

没有动作达到 `ΔTOPIQ >= +0.005`，更没有置信区间排除零的正收益。`tangent-rescaled_0.001` 经 TOPIQ metric 内 Holm 校正后仍显著为负（`p=0.025920`）。固定矩几何把 raw 的饱和度增量压到接近零，说明机制实现有效；但 HPSv2、CLIP 和定性 montage 均无稳定改善，说明去掉颜色/对比度通道后，TFSA 的无条件空间重排本身没有 headroom。

TOPIQ 留一 seed 的逐 prompt 选择相对 `no_ag` 为 `+0.001791 [-0.004392,+0.007529]`，且 clipping/saturation 上升。按 HPSv2 选择时，逐 prompt 策略比全局 expert 低 `-0.002082 [-0.003872,-0.000434]`。因此 S2 不进入 RL；下一迭代必须改变残差的状态条件方向，而非继续扫 scale。

## 9. S3 轨迹锥（失败）

`outputs/exp_trajectory_cone/development_12prompt_3seed_v1` 包含 12 prompts × 3 seeds × 9 actions = 324 个完整配对记录。四卡各 81 条，36 个区块均未跨卡；144 张 no-AG、expert、raw 和 plain-tangent 控制图与 S2 的 PNG 哈希逐一相同。全部评分键完整且为有限值。

| 动作 | Δ TOPIQ-NR [95% CI] | Δ HPSv2 | Δ clipped | Δ saturation |
|---|---:|---:|---:|---:|
| cone 0.001 | -0.002519 [-0.005576,-0.000152] | -0.000553 | -0.000175 | -0.000812 |
| cone 0.002 | -0.001740 [-0.005274,+0.001283] | +0.000085 | +0.000050 | -0.000385 |
| cone 0.004 | -0.004220 [-0.010200,+0.001771] | +0.000665 | +0.000600 | -0.000368 |
| cone-rescaled 0.001 | -0.003178 [-0.005424,-0.000840] | +0.000173 | +0.000085 | -0.000642 |
| cone-rescaled 0.002 | -0.005843 [-0.010720,-0.000922] | +0.000346 | +0.000998 | +0.000022 |

没有动作达到 `+0.005` 主门槛。matched-scale cone `0.002` 相对 plain tangent `0.002` 只有 `+0.000528 [-0.000412,+0.001596]` TOPIQ 增量，HPSv2 为 `+0.000031`，因此“反向分量导致 S2 损失”的机制解释没有得到支持。

TOPIQ seed-CV 的全局动作在三个 fold 都是 `no_ag`；逐 prompt 相对 no-AG 为 `+0.001120 [-0.004272,+0.006516]`，同时 clipping 增加 `+0.001262`、saturation 增加 `+0.012097`。按 HPSv2 选择时，逐 prompt 策略反而比全局 expert 低 `-0.001604 [-0.003455,-0.000190]`，且像素 guard 显著恶化。固定拼图 `outputs/exp_trajectory_cone/development_12prompt_3seed_v1/figs/all_actions_seed0.png` 覆盖全 12 prompt × 全 9 动作，没有显示稳定结构或文字修复。S3 按停止规则关闭，不再扫幅度，也不进入 RL。

下一步仅允许执行已在 `MODEL_ITERATIONS.md` 注册的 2048² Stage-2 目标域诊断。它用于检验当前 1024² 代理终点是否错配论文的高分辨率主张，不构成对 S0-S3 的确认性复活。
