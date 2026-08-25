# Scheduler-Native 固定结构 Headroom

## 当前问题

S5 和 LR-1 的旧注入没有按 Euler scheduler 的实际坐标缩放。当前实验重新回答一个更干净的问题：在冻结 SDXL、一次 U-Net forward 和 50-step Euler 轨迹下，把一个固定结构方向映射到 clean endpoint，是否能可靠优于 `no_op`。

这仍不是 RL，也不是 learned renderer。它只是训练前的静态闸门。与会议版的软联系是：两者都利用冻结生成过程中暴露的结构，并修改 latent 轨迹；当前 primary 刻意不用 attention hook，用来检验这种原则是否可以脱离原 TFSA 实现。

## 固定设计

development 使用 33 个新 prompts、3 个新 seeds 和 8 个动作，共 792 cells。四个 primary 分别只请求 low/mid/high spectral 或 Laplacian basis；semantic 和 FreeU 只作机制 ablation。所有动作固定系数 `+0.02`、CFG `7.5`、1024px、50 NFE、zero churn，并要求每步一次 U-Net、`match_rms`、moment preservation 和 `0.05` trust cap。

`lazy_zero_identity` 必须与 `no_op` 的 PNG bytes 完全相同。component smoke v3 已验证 zero parity、lazy basis 构造、空 hook、Euler gain、moment error 和非零 probe 接线；它只跑 4 steps 和两个非零动作，不能作为质量证据。

正式筛选条件在评分前固定：TOPIQ-NR 平均差值至少 `+0.005`，crossed-bootstrap CI 下界大于零，四动作 prompt sign-flip 经 Holm 后小于 `0.05`；HPSv2/CLIP 非劣；clipping、saturation 和 contrast 全部通过。这里的 `+0.005` 只是点估计筛选线，并没有检验“真实提升至少为 `+0.005`”。Holm 只覆盖四个 TOPIQ 对零检验，其他 guard 的区间也没有在四个动作间做 simultaneous correction。因此，本轮只能叫 quantitative screen，不能把 trigger 写成已经证明 practical headroom。

结果盲 evaluator 已单独绑定脚本、auditor、actions、prompts、scorer contract 和 Python/NumPy/pandas/PyYAML 版本。授权中的 `method_selection=false`。如果任一 primary 触发，输出只能是 `independent_review_required`，而且在新的盲评规则冻结前不得查看图片或挑 winner；如果没有触发，输出 `null_route`。两种结果都不能直接启动 validation 或 RL。

## 严格解释边界

当前四个 primary 都是对 clean latent 单个 raw band 加固定正系数。它们不是 FreSca/SPA 那种“当前频谱到目标频谱”的 residual，也没有覆盖负向高频抑制、时变 schedule 或 learned descriptor。因此，如果四个动作失败，只能关闭这次登记的 `+0.02` 固定动作族及其蒸馏/RL 路径，不能宣称所有频谱或 latent renderer 假设都被否定。

反过来，若任一动作触发，也只说明注册的 33 prompts、3 seeds 上出现了值得复核的 proxy signal。下一阶段必须在没有读取图像前先冻结盲评 rubric 和多动作处理规则，再使用由独立方保管的新 split。确认实验还必须加入 SPA/FreSca/FDG、APG/Guidance Interval、FreeU、会议版 TFSA、matched-RMS random direction 和 matched-compute controls，并对带入动作及 guards 使用 simultaneous one-sided inference。没有这些直接基线、外部 prompts、更多 seeds、高分辨率迁移和盲评，结果不足以支持 TPAMI 主张。

## 状态

正式 `development_v2` 已从 clean commit `57a8479` 在物理 GPU 7 启动。首个完整 block 已通过 zero parity 和 50-step ledger fail-fast。此处暂不写任何质量结果；必须等待 792/792 生成、严格 scorer、完整 audit 和结果盲 evaluator 全部完成后再更新。
