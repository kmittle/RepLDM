# 固定 CFG 对照实验

## 为什么做这个实验

多个 guidance 和 renderer 动作都没有稳定胜过 `no_op`。一个直接的质疑是：基线 CFG `7.5` 可能没有调好，所谓新方法差异只是 guidance scale 选择不足。为排除这个混杂因素，我们在看结果前冻结一次 CFG development sweep。

Dynamic CFG、Guidance Interval、APG 和 CFG++ 已经研究了动态强度、作用窗口和几何约束。本轮只回答更小的问题：在当前 SDXL、Euler、50 NFE 和 prompts 下，是否有另一个常数 CFG 明显优于 `7.5`。它不是新方法，也不授权动态 controller。

## 实验与结果

`outputs/cfg_baselines/development_v1` 包含 12 prompts、3 seeds 和 5 个 scales，共 `180/180` 个完整样本。manifest 与 strict score audit 验证了完整 grid、同卡配对、每步一次 U-Net、Euler schedule、输入输出哈希和 scorer provenance。

相对 CFG `7.5`：

| CFG | TOPIQ-NR 平均差值 | 判断 |
|---:|---:|---|
| 2.5 | -0.056278 | 明显退化 |
| 5.0 | -0.015090 | 明显退化 |
| 10.0 | +0.003251 | CI 跨零，未达 +0.005 |
| 15.0 | +0.004252 | CI 跨零，未达 +0.005 |

高 CFG 还违反 clipping、saturation 或 contrast guards。后两个动作的 Holm `p=0.702893`。冻结 selector 返回 `cfg_7p5` 和 `null_route`。

## 结论与反思

当前证据支持把后续 CFG 固定为 `7.5`，不能声称 10 或 15 有可靠收益。这次实验只否定本次登记的常数网格，不否定所有动态 CFG 方法。若未来方法声称来自结构 renderer，仍应在确认阶段加入强动态 guidance controls，避免把 CFG 效果误归因于 renderer。

本次 development 已使用 `0,42,123`，所以这些 seeds 永久视为已暴露，不能再称作新方法的 final seeds。
