# 历史实验归档

本目录保存 2026-08-30 及以前的实验输出。它们没有丢失，但不再作为当前正式评测的输入。
这些实验大多只使用 11--48 条提示词，样本数、seed 或配对方式也不完全一致，因此只能帮助回顾思路和排查代码，不能放入论文主表，也不能用来证明方法优于基线。

## 已观察到的现象

- 旧 scalar sweep 的实验设置被分到不同 GPU，严格配对失效，原有排名作废。
- 低频、中频、高频和统一 scalar 大多降低 TOPIQ-NR。减小强度只会逐渐回到 `no_ag`，没有找到稳定的中间最优点。
- 固定均值/方差、轨迹锥和 scheduler 方向约束减少了颜色与饱和度副作用，但质量提升也随之消失。
- 2048px Stage-2 没有反转 1024px 的负结果。
- self-attention semantic transport、固定 latent renderer 和 prompt 自适应选择都没有通过预设门槛，不能作为 RL 的可靠监督信号。
- FreeU 和部分 scheduler 设置曾提高 TOPIQ-NR 点估计，但同时明显增加 clipping、饱和度或对比度；这更像指标偏好，而不是可靠的结构改善。
- 旧 S5/LR-1 使用了错误的 Euler 注入增益。除最后一步外，旧实现通常把 clean-endpoint 修正放大约 9--16 倍，因此这些图不能监督新的 scheduler-native renderer。
- 修正 scheduler 坐标后，中频设置得到过小幅正值，但低于预设效果门槛并违反 clipping 保护条件。结构 baseline 校准也没有得到可发表的胜者。
- LR-0 只证明 latent renderer 接线正确：零 renderer 与基线逐像素一致，非零 probe 能改变输出；它没有证明画质收益。

## 后续使用规则

1. 这里只保留工程回归、失败分析和复现实验所需的文件。
2. 不从这些结果重新挑选 scale、layer、seed、prompt 或实验设置。
3. 新的正式实验必须使用完整 benchmark。HPSv2 必须覆盖官方四种风格各 800 条，共 3,200 条提示词。
4. 基线和候选方法必须共享 prompt、seed、模型、scheduler、CFG、NFE 和后处理，并以 prompt 为统计单位做配对分析。
5. 任何新结论都写入新的输出目录和独立报告，不修改本归档中的历史文件。

更完整的逐实验说明、数值和图示见 `results/08-30/results-08-30.md`。当前正式协议见 `doc/protocols/MAINSTREAM_EVALUATION_PROTOCOL.md`。
