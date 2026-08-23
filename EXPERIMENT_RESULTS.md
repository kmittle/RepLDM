# 实验结果与阶段性结论

> 分支：`rl-version` ｜ 方法与停止规则：[`REVIEWER_PROTOCOL.md`](./REVIEWER_PROTOCOL.md)
> 本文件只记录可由当前代码和产物复核的事实。探索集上的动作选择不作为最终论文结果。

## 0. 当前结论

旧的 12-prompt × 3-seed × 7-scale 实验不能支持配对因果结论。审计发现，36/36 个 `(prompt, seed)` 区块中的不同 scale 被分配到了 4 块 GPU；因此所谓“同 prompt、同 seed 配对”仍混入了设备数值差异。此前报告的配对置信区间、显著性和 oracle gap 全部撤回，只保留为探索性现象。

当前唯一有效的方向性证据来自 2-prompt × 1-seed 的同设备 smoke test：中频单独引导可能比统一标量更稳，但样本量不足以声称提升。结论仍是：**在新的配对 pilot 通过预注册门槛前，不训练 RL 控制器。**

## 1. 进度与证据等级

| 项目 | 状态 | 可支持的结论 |
|---|---|---|
| 可重入 Attention Guidance、控制器接线 | 已完成，18 项单元测试通过 | 工程正确性 |
| 三频带残差动作、更新能量上限 | 已完成 | 候选动作空间可运行 |
| 标量/等增益频带等价性 | 2 prompts 像素逐值一致 | 向后兼容性 |
| 旧 252 图 scale sweep | **作废为配对实验** | 仅可生成后续假设 |
| 同设备 spectral smoke | 2 prompts × 1 seed | 方向性信号，不可推断 |
| 正式 spectral pilot | 待运行 | Gate-1 头部空间判断 |
| RL / 反事实控制器 | 未开始 | 必须先通过静态基线门 |

## 2. 已作废实验的审计

输出目录为 `outputs/exp1.1_scale_sweep/pilot_12prompt_3seed_7scale/`。sidecar 中记录了每张图的 `device`；按 `(prompt_index, seed)` 聚合后，全部 36 个区块的 `device.nunique()` 都大于 1。共享任务队列按单张图抢占工作，动作并未成组进入同一 GPU。

旧数据曾显示 scale 增大时饱和度、锐度、对比度和裁剪率上升，ImageReward 曲线较平，且 TOPIQ-NR 从 scale 0 的约 `0.710` 降至 scale 0.004 的约 `0.660`。这些趋势可用于提出“统一标量主要放大低频颜色/对比度”的假设，但**不得报告为动作效应**，也不得用于动作选择或显著性检验。

生成器现已修复：完整 `(prompt, seed)` 区块被固定到一块 GPU；续跑优先继承 sidecar 的设备，且已有跨设备区块会被拒绝。`compare_actions.py` 在统计前再次检查该约束。

## 3. 有效 Smoke Test

### 3.1 数值等价性

`outputs/exp_spectral_headroom/equivalence_v3/` 包含两个 prompt。恒定标量 `0.004` 与三频带等增益 `[0.004, 0.004, 0.004]` 在两张输出上均像素逐值一致。这验证了频带分解在退化到统一增益时保留原 fp16 行为。

### 3.2 方向性质量信号

`outputs/exp_spectral_headroom/smoke_paired_v2/` 中，每个 prompt 的 11 个动作位于同一 GPU。两提示均值如下：

| 动作 | TOPIQ-NR | HPSv2 | ImageReward | clipped fraction |
|---|---:|---:|---:|---:|
| no AG | 0.71427 | 0.30115 | 0.72510 | 0.01051 |
| mid only 0.004 | **0.71646** | 0.30200 | **0.61874** | **0.00989** |
| scalar 0.004 | 0.65738 | 0.31360 | 0.75951 | 0.05486 |
| conference expert | 0.69299 | 0.30701 | 0.64592 | 0.04443 |

这只说明频谱动作存在值得复核的候选方向。`mid only` 的 TOPIQ/裁剪方向与 ImageReward 明显冲突，统一标量的偏好分提高又伴随 TOPIQ 和裁剪恶化，任何加权总分都会掩盖这一事实。`n_prompt=2, n_seed=1` 无法估计 prompt/seed 方差，表中粗体用于标出冲突，不是显著性声明。

## 4. 下一轮预注册 Pilot

设计固定为 12 prompts × 3 seeds × 11 actions，共 396 张 1024² Stage-1 图。每个 `(prompt, seed)` 区块在同一 GPU 上完成所有动作，并记录完整动作、代码提交、设备和生成参数。

比较对象包括 no-AG、会议版 expert、恒定标量 0.004、等增益频带、三个单频动作和四个倾斜/保护低频动作。TOPIQ-NR 是 pilot 的主质量指标；HPSv2、ImageReward、patch-IR、CLIPScore 与 aesthetic 是确认/诊断指标；饱和度、裁剪率、对比度和拉普拉斯方差只作防 reward-hacking 见证。

推断使用 prompt/seed crossed bootstrap 95% CI、prompt-level sign-flip test，以及跨动作的 Holm 校正。pilot 只筛选动作：候选必须优于 no-AG、会议版 expert 和最佳统一标量，且偏好/对齐指标不出现一致下降、裁剪率不恶化。即使通过，也必须在 prompt-disjoint 的更大测试集上冻结动作后复验。

## 5. 复现命令

```bash
# 生成：GPU 1-4；每个 prompt/seed 的全部动作固定在同卡
/home/bycao/miniforge3/envs/diff_attn/bin/python eval-pipeline/generate.py \
  --devices 1,2,3,4 \
  --prompts eval-pipeline/prompts/eval_v1.csv \
  --out_dir outputs/exp_spectral_headroom/pilot_12prompt_3seed_v1 \
  --actions eval-pipeline/configs/frequency_action_pilot.yaml \
  --seeds 0,42,123

# 评分：生成完成后运行；正式结果要求全部指标加载成功
CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/bycao/miniforge3/envs/repldm_eval/bin/python eval-pipeline/score.py \
  --run_dir outputs/exp_spectral_headroom/pilot_12prompt_3seed_v1 \
  --device cuda:0 --strict

# 配对统计
/home/bycao/miniforge3/envs/repldm_eval/bin/python eval-pipeline/compare_actions.py \
  --run_dir outputs/exp_spectral_headroom/pilot_12prompt_3seed_v1 \
  --baseline no_ag \
  --metrics topiq_nr,hpsv2,imagereward,patch_ir_mean,clip_cosine,aesthetic,clipped_fraction,mean_saturation
```

## 6. 报告纪律

- pilot、消融集和动作搜索集只用于决策，不作为最终无偏性能估计。
- 不以 Laplacian variance 或 patch-IR 单独宣称“细节更好”；二者容易奖励噪声或局部纹理。
- 不把多个 CLIP 家族指标当成独立重复证据。
- 只报告预注册比较、完整失败案例、绝对分数、配对差值、95% CI 和校正后 p 值。
- 若静态频带或 search-distill 已解释全部收益，RL 贡献应被否决，而不是追加控制器复杂度。
