# OPD、DPO、RL Latent Renderer 文献审查（2026-09-03）

## 核验范围

我在 2026-09-03 通过 arXiv 的论文页面、PDF 和论文附带的代码链接核对了下面的出处。表中写的是论文实际做了什么，不代表 RepLDM 已复现其结果；论文中的数字也不能直接搬到 SDXL 或本项目。

## 已核验的出处和差异

| 工作 | 已核验出处 | 核心做法 | 对本项目的直接含义 |
|---|---|---|---|
| DDPO | [arXiv:2305.13301](https://arxiv.org/abs/2305.13301v4) | 把去噪链写成 MDP，用终局 reward 做 denoising policy gradient。 | 是黑盒 RL 基线；它更新扩散模型策略，不是冻结 U-Net 上的微小 renderer。 |
| DPOK | [arXiv:2305.16381](https://arxiv.org/abs/2305.16381v3) | 在线 policy gradient，并把相对预训练模型的 KL 当作稳定项。 | 若做 RL，必须有 reference/KL 对照；不能把“有 KL”写成新意。 |
| GRPO / Flow-GRPO | [DeepSeekMath](https://arxiv.org/abs/2402.03300v3)、[Flow-GRPO](https://arxiv.org/abs/2505.05470v5) | GRPO 用同一 prompt 的一组样本计算相对优势；Flow-GRPO 将确定性 ODE 改写成可探索的 SDE，并减少训练去噪步。 | 组优势、SDE 探索和少步训练已有先例；renderer 的连续 action 概率、查询数和实际 NFE 要单独记账。 |
| DRaFT | [arXiv:2309.17400](https://arxiv.org/abs/2309.17400v2), ICLR 2024 | 对可微 reward 穿过完整采样链反传；DRaFT-K 只反传最后 K 步，DRaFT-LV 在 K=1 时用多个噪声降低方差。 | 这是可微 reward 上的强基线。它主要给 LoRA/扩散模型反传，不能说此前没有冻结小 renderer 的先例。 |
| Diffusion-DPO | [arXiv:2311.12908](https://arxiv.org/abs/2311.12908v1) | 用离线 winner/loser 图像对和扩散 ELBO 构造 DPO 损失，并保留 reference 分布约束。 | 它不是 on-policy，也不提供中间 latent target；在本项目中只能作为离线偏好基线，且要明确 pair 的来源。 |
| LPO | [arXiv:2502.01051](https://arxiv.org/abs/2502.01051v5), NeurIPS 2025 | 训练能看 noisy latent 和 timestep 的 Latent Reward Model (LRM)，再在 noisy latent 上做 step-level preference optimization；论文还试了 step-wise GRPO。 | “在 latent 上做偏好学习”已有直接先例。我们的 renderer 若使用偏好，差异必须来自冻结 backbone、受限 action frame 和小参数量，而不是“latent”三个字。 |
| DiffusionOPD | [arXiv:2605.15055](https://arxiv.org/abs/2605.15055v1) | 先为不同任务训练 teacher，再让 student 按自己的 rollout 访问状态；同协方差高斯有闭式逐步 KL，确定性 ODE 退化为 transition matching。 | OPD 的 on-policy 状态和闭式 native-coordinate 目标已被占用。我们不能声称首个 diffusion OPD；需说明 teacher 是否外部、是否只训练 renderer。 |
| Flow-OPD | [arXiv:2605.08063](https://arxiv.org/abs/2605.08063v5) | 在 SD3.5-M 的 flow matching 中先训练单任务 GRPO teacher，再用 on-policy sampling、task routing、dense trajectory supervision 合并为 student，并加入 manifold anchor。 | 这是 OPD 在 flow model 上的近邻，不是 SDXL Euler 的复现；“teacher 合并/trajectory supervision”不能单独作为创新。 |
| DiffusionOPSD | [arXiv:2608.24646](https://arxiv.org/abs/2608.24646v1) | 冻结 behavior policy 采样 query/anchor，reward gradient 构造有界正负 clean-output target，detach 后有限步拟合，再用 EMA 刷新 behavior。 | reward-to-target、正负 target、finite fitting 与 EMA 都是近期先例；若采用，必须分开报告 target gain 和拟合后 gain。该文是 technical report，结果尚未在本项目复现。 |
| RVM | [arXiv:2608.23664](https://arxiv.org/abs/2608.23664v1) | 直接在 native velocity 上做正负 reward 加权回归，可选 reference/anchor；论文把 RAM、DiffusionNFT 视为特例。 | “native velocity 训练、signed reward、anchor”已被近期工作覆盖；SDXL epsilon/Euler 需重新推导，不能直接移植 flow 公式。 |
| BranchGRPO | [arXiv:2509.06040](https://arxiv.org/abs/2509.06040v5) | 在中间步分支，共享 prefix；把叶子 reward 融合成 depth-wise advantage，并剪枝反传。 | 共享前缀和稠密 credit assignment 不是新估计器。只有在相同 U-Net/reward 查询预算下才可作为效率实验。 |
| Diffusion Controller (DiffCon) | [arXiv:2603.06981](https://arxiv.org/abs/2603.06981v1) | 用 LS-MDP 推导“冻结 pretrained score + 轻量 side network”控制形式，支持 SFT、reward-weighted loss 和 PPO；实验在 SD1.4。 | 这是与“冻结 backbone + latent side network + RL”最直接的重叠。不能宣称 first frozen controller；必须加入 DiffCon 风格自由 residual、参数量和强度冻结的对照。 |
| AE 结构工作 | [Improving the Diffusability of Autoencoders](https://arxiv.org/abs/2502.14831v3)、[EQ-VAE](https://arxiv.org/abs/2502.09509v3)（均 ICML 2025） | 前者发现 AE latent 高频过强，用 decoder 的 scale-equivariance 正则并微调 AE；后者让 scale/rotation 在 latent 中保持 equivariance，降低 latent 复杂度，也需微调 AE。 | 它们支持测量频谱和变换一致性的动机，但都改变了 AE/训练分布；不能把 frozen SDXL 推理时的低通或投影直接称为其方法复现。 |

补充：LaRender（[arXiv:2508.07647](https://arxiv.org/abs/2508.07647v1)）已经使用“latent rendering”这一名称，但它解决的是有对象遮挡图的训练-free 控制，不是通用质量提升。因此论文中不能写“首次提出 latent renderer”，最多说明本项目采用了不同的通用质量/结构控制定义。

## 可防守的创新边界

下面这些单独都不能作为创新：OPD、DPO、GRPO、reward gradient、KL/reference、shared-prefix branching、zero-init adapter、频谱分解、关系图、timestep gate、低维 action、冻结 backbone、latent renderer 这个术语，或“参数少于 U-Net”。

目前唯一可能防守的主张是一个**组合且可证伪**的命题：在冻结且固定版本的 SDXL CFG forward 上，一个小于 1M 参数、零初始化、严格 no-op 的 renderer 读取普通 forward 已产生的结构描述，输出 scheduler-native 的有界 clean-latent residual；在相同 prompt、seed、NFE、U-Net 调用、reward 查询和训练步数下，它同时胜过自由 residual、uniform/random relation、predicted-clean relation、DiffCon 风格 side network，以及固定 teacher。论文还必须证明收益来自“结构描述 × scheduler 坐标”的交互，而不是 reward hacking 或更大的动作幅度。

## 方案设计结论

1. 先固定一个 renderer/action frame 和外部 teacher，做普通 supervised/off-policy 与 DiffusionOPD 式 on-policy transition matching；若 teacher 目标本身不提升，不能靠 RL 挽救。
2. DPO 应使用同一 renderer 在同一初始噪声下产生的成对 action/结果，并与 Diffusion-DPO 的离线图像对分开命名；LPO 的 LRM 只能作为已知 step-level latent reward 基线。
3. RL 先做带 reference/KL 的 DDPO/DPOK 或 GRPO 对照；BranchGRPO 只有在共享 prefix 确实降低等预算成本时才加入。所有方法都要报告完整 HPSv2（3,200）和 GenEval（2,212），训练 reward 不能替代独立评测。

## 尚未核验的内容

本文没有声称复现任何论文数字，也没有核验每个作者仓库的长期可用性、许可证和训练数据授权。正式实验前仍需锁定论文版本、代码 commit、依赖和数据许可证；尤其是 DiffusionOPSD、BranchGRPO、DiffCon 等 2026/近期预印本，投稿前应再次检查版本和审稿状态。
