# 文生图全量评测与 Latent Renderer 训练调研

## 先说结论

当前最稳妥的主线不是直接用黑盒 RL 调一个标量，而是训练一个远小于 SDXL 的
scheduler-native latent renderer。先让 student 在冻结 SDXL 上生成自己的轨迹，
再由 teacher 在 student 实际访问的 latent 上给逐步目标，这才是 on-policy
distillation（OPD）。公开偏好数据没有初始噪声和中间 latent，只能训练 reward
model 或提供 prompt，不能直接冒充 renderer 的逐步监督。

## 完整 Benchmark

| Benchmark | 完整协议 | 主要检查内容 | 项目用途 |
|---|---:|---|---|
| [HPSv2](https://github.com/tgxs002/HPSv2) | 3,200 prompts x 1 图 = 3,200 图/设置 | 人类偏好、四种风格 | 每个候选必跑 |
| [GenEval](https://github.com/djghosh13/geneval) | 553 prompts x 4 图 = 2,212 图/设置 | 物体、数量、颜色、位置 | 每个候选必跑 |
| [DPG-Bench](https://github.com/TencentQQGYLab/ELLA) | 1,065 prompts x 4 图 = 4,260 图/设置 | 长文本与复杂语义 | 通过首轮后追加 |
| [T2I-CompBench++](https://github.com/Karine-Huang/T2I-CompBench) | 2,400 prompts x 10 图 = 24,000 图/设置 | 属性绑定、空间、数量、复杂组合 | 最终强候选 |
| [ConceptMix](https://github.com/princetonvisualai/conceptmix) | 使用官方完整清单和采样数 | 多概念组合 | 最终强候选 |
| [Gecko](https://github.com/google-deepmind/gecko_benchmark_t2i) / [GenAI-Bench](https://github.com/linzhiqiu/t2v_metrics) | 使用各自官方完整协议 | 细粒度对齐和人工可解释问题 | 最终强候选 |

筛选只能减少**实验设置数**，不能随机抽 benchmark 子集。第一道门固定为完整
HPSv2 + GenEval，即每个实验设置 5,412 张；通过后再增加 DPG-Bench。DrawBench
和 PartiPrompts 的原始结论依赖多次采样和人工比较，不适合作为日常自动门槛。
最终分布质量可以增加 MJHQ-30K 和固定索引的 COCO FID-30K。

## 偏好与训练数据

| 数据 | 规模与问题 | 建议用途 |
|---|---|---|
| [HPDv3](https://huggingface.co/datasets/MizzenAI/HPDv3) | 约 1.08M 图文对、1.17M 偏好对；官方 split 有 prompt/图像重叠 | reward 预训练，先重划 prompt/image-disjoint split |
| [HPDv3++](https://huggingface.co/datasets/Junjun2333/HPDv3-PlusPlus) | aesthetic 100,463 对、text-following 90,908 对；官方 train/test 泄漏明显 | 当前优先数据，严格去重后使用 |
| [HPDv2](https://huggingface.co/datasets/ymhao/HPDv2) | 798K choices、约 430K 图、107K prompts | 常用基线；若用于训练，HPSv2 不能是唯一主评测 |
| [Pick-a-Pic](https://github.com/yuvalkirstain/PickScore) | v1 约 584K 对，v2 超过 1M；真实用户但噪声较多 | reward 补充，先做 NSFW、重复和低质量清洗 |
| [ImageRewardDB](https://huggingface.co/datasets/THUDM/ImageRewardDB) | 约 62.6K 图，含 overall/alignment/fidelity | 规模适中的 reward 冷启动 |
| [RichHF-18K](https://github.com/google-research-datasets/richhf-18k) | 质量分、局部缺陷热图、token 标注；图像需另行解析 | 局部结构 reward 和错误定位 |

训练 prompt 必须排除 HPSv2、GenEval、DPG-Bench、T2I-CompBench++、
DrawBench、ConceptMix、Gecko、GenAI-Bench、MJHQ 和冻结 COCO caption。仅比较
字符串不够：还要做规范化文本、近重复文本和图像哈希/感知哈希去重，并保存
排除清单和哈希。

## 方法边界

- **OPD**：student 先 rollout，teacher 再标注 student 实际访问的 `x_t`。
  [DiffusionOPD](https://arxiv.org/abs/2605.15055) 和
  [Flow-OPD](https://github.com/CostaliyA/Flow-OPD) 是最接近当前问题的训练范式。
- **LPO**：直接在 noisy latent 上建偏好目标，官方实现支持 SDXL；应作为核心
  preference baseline，见 [LPO](https://github.com/Kwai-Kolors/LPO)。
- **DRaFT-K / AlignProp / ReFL**：reward 可微时，只对小 renderer 回传，是比
  黑盒 RL 方差更低的强基线。
- **DDPO / DPOK**：能使用黑盒 reward，但 rollout 多、方差高。只在蒸馏仍有
  明确提升空间时使用。
- **Diffusion-DPO**：针对完整 diffusion likelihood，不能原样套到确定性 residual
  renderer 上。
- [Diffusion Controller](https://arxiv.org/abs/2603.06981) 已覆盖 frozen backbone
  加轻量 latent side network，是新颖性比较中必须正面讨论的工作。

OPD、on-policy self-distillation 和 reward velocity matching 不是同一个方法，
报告中不能混用简称。

## 冻结的首轮训练方案

1. 先用完整 HPSv2 验证固定 feature relation 是否同时胜过 `no_ag`、普通平滑和
   匹配随机方向，并通过 CLIP、TOPIQ、clipping、saturation 和 contrast 门槛。
2. 若通过，使用去除所有 benchmark prompt/图像后的 HPDv3++ train prompts，
   由冻结 SDXL 生成同 prompt、同初始噪声的 trajectory。
3. 同一 trajectory 构造 no-op、会议版结构 teacher 和有界 reward-gradient teacher；
   teacher 不允许多调用 backbone 来获得不公平算力。
4. 训练 sub-1M basis renderer。输入当前 latent、timestep、文本条件和冻结 U-Net
   结构特征，只输出有界 scheduler-native residual；以 base teacher 作 anchor。
5. 对比 supervised off-policy、OPD、LPO、DRaFT-K 和黑盒 RL。参数量、NFE、
   U-Net 调用数、训练数据和 reward 预算必须分别报告。
6. 每个固定 checkpoint 跑完整 HPSv2 + GenEval；训练 reward 只能作诊断，不能
   同时作为唯一主评测指标。

## 严苛审稿检查

- 增益是否超过 `no_ag`、会议版设置、普通平滑、随机方向和等算力 controller？
- student 是否真的使用结构特征？必须做 feature shuffle、teacher removal 和
  relation/random basis 消融。
- 收益是否来自饱和度、锐度或 reward hacking？必须同时报告像素保护指标和盲评。
- 是否只记住训练 prompt 或 reward model？必须做 prompt/image-disjoint 测试、
  未参与训练的指标和跨模型验证。
- 1024px 的收益能否转移到会议版高分辨率 Stage 2？最终必须报告 2K/4K、
  ControlNet 和至少一个额外 backbone。

## 与会议版的联系

会议版用 U-Net 内部关系对 latent 做 training-free 结构修正；期刊版把同一底层思想
推广为“冻结 diffusion backbone、学习小型 latent renderer”。结构联系来自修正对象、
关系特征和 scheduler 坐标，而不要求继续沿用手工 attention-guidance 标量。FreeU
说明 U-Net 自身结构可以增强生成；[频谱正则 latent](https://arxiv.org/pdf/2502.14831)
和 [VAE 不变性正则](https://arxiv.org/pdf/2502.09509) 进一步说明 latent 的低频结构与
鲁棒性值得直接建模。这一联系可以支持论文叙事，但不能替代效果、机制和完整验证。
