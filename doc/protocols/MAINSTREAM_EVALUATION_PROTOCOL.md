# 主流评测协议

本协议用于 RepLDM 期刊扩展的正式方法比较。它遵循 RepLDM 会议版论文
（[arXiv:2410.06055](https://arxiv.org/abs/2410.06055)）的样本规模，并把
随机性和配对关系写清楚。

## 正式规模

| 目标分辨率 | 每次运行的 prompt 数 | 预注册 seed 重复数 | 每个设置的评估样本数 |
|---|---:|---:|---:|
| 1024x1024 / Stage-1 方法门槛 | 1,000 | 3 | 3,000 |
| 2048x2048 | 2,000 | 3 | 6,000 |
| 大于 2048x2048 | 1,000 | 3 | 3,000 |

这里的 3 次重复对应每个 prompt 的 3 个预先固定的随机 seed。每个 seed 运行
整套 prompt 清单一次，而不是看完结果后为某个 prompt 挑选有利 seed。若资源
允许增加到 5 个 seed，必须在生成前登记，并让所有方法使用完全相同的 seed 清单。

## 固定推理条件

默认沿用会议版的 `SDXL`、`EulerDiscreteScheduler`、50 个 denoising steps 和
CFG `7.5`。VAE、upscaler、精度、prompt 顺序、分辨率和所有方法的后处理也必须
固定；Stage-2 若使用额外噪声，噪声必须由同一个 `(prompt, seed)` 派生。会议版的
FID、IS、裁剪版 FID/IS 和 CLIP 作为主表基础指标，HPSv2、TOPIQ-NR 及结构/像素
保护指标作为补充见证。

## HPSv2 对照

HPSv2 官方 benchmark 有 4 个风格、每种 800 个 prompt，共 3,200 个 prompt；
每次运行每个 prompt 生成一张图。它的 10 组标准差来自每种风格的 800 个
prompt 分成 10 组，每组 80 个，不是 10 个 seed。我们可用这套 3,200 prompt
作为辅助结果，并在需要稳健性时对整套清单做同样的 3-seed 重复。

## 配对和统计

正式比较必须冻结 prompt manifest、seed 清单、模型权重、VAE、scheduler、CFG、
NFE、分辨率和 scorer。所有方法在同一 `(prompt, seed)` 上生成，动作顺序可随机
但必须记录。先对同一 prompt 的 seed 求均值，再以 prompt 为统计单位报告配对
差值、交叉 bootstrap 95% CI 和预注册检验；不能把 seed 当成独立 prompt。

三次重复必须使用同一份、预先冻结的 prompt manifest；不能每次重新随机抽取
prompt，否则不同方法之间没有严格的配对关系。

“每个设置的评估样本数”是 `prompt 数 x seed 数`；“总生成图数”还要再乘以
实验设置数。旧的 `12 prompt x 3 seed` 运行只作探索性 pilot，不得放入正式主表
或作为跨方法排名依据。
