# OPD、DPO 与 RL 文献更新

**检索日期：2026-09-04。** 本次用 arXiv 摘要、版本号和项目页做了补充核对；没有复现论文代码，也不把论文报告的数字当作 RepLDM 结果。现有 `related_work_opd_dpo_rl_audit_2026-09-03.md` 仍是主表。

## 直接重叠

| 工作 | 主要内容 | 对 RepLDM 的含义 |
|---|---|---|
| [ToPO](https://arxiv.org/abs/2609.03688) | token 条件的空间/时间偏好路由，面向 attention latent diffusion | 与结构条件 renderer-DPO 很接近；必须讨论或做等预算对照。 |
| [STEP-OPD](https://arxiv.org/abs/2608.04887) | 在 teacher 输出之外约束中间表示的变化 | 仅做 output transition matching 不能声称覆盖这条路线。 |
| [DiffusionOPSD](https://arxiv.org/abs/2608.24646) | reward 梯度产生正负 clean target，再有限步拟合和 EMA | 与 F0/T_OPSD 同构；应称为借鉴的 teacher-construction baseline。 |
| [Latent Reward Registers](https://arxiv.org/abs/2608.03929)、[SURE](https://arxiv.org/abs/2608.06125) | 从 noisy latent 估计稠密 reward，并用可靠性加权 | 说明 latent reward/credit assignment 已有先例，不能单独作为创新。 |
| [Self-OPD](https://arxiv.org/abs/2608.26872) | 用学生自己的随机分支替代外部 teacher | 可作 teacher-free 边界；本项目仍固定外部 teacher。 |

## 方法与工程边界

[REST](https://arxiv.org/abs/2608.09226) 把 RL 的 scored trajectory 直接用于蒸馏；[Linear-DPO](https://arxiv.org/abs/2605.21123)、[Diffusion LAIR](https://arxiv.org/abs/2605.26491) 和 [DrPO](https://arxiv.org/abs/2606.02521) 分别覆盖线性、listwise 和 one-step 偏好目标。[Manifold Drift](https://arxiv.org/abs/2608.20011) 提醒偏好更新会离开基础模型流形，支持保留 reference、动作 cap 和像素保护指标。[LeanGRPO](https://arxiv.org/abs/2609.03528) 主要减少 rollout 重算，属于等目标的效率方案，不是新的 renderer 目标。[HPSv3++](https://arxiv.org/abs/2606.14657) 提供新的偏好数据和 reward-model 训练方向，但其数据授权、版本和校准尚未在本项目登记。

## 实验处理

现阶段这些新增工作只进入相关工作、风险矩阵和待复现清单，不进入已授权结果表。正式主对照仍为 `no-op`、`conference_settings`、参数匹配的 free residual、随机/相位匹配结构 frame、固定 T_OPSD、search-distill、renderer-DPO，以及带 reference/KL 的 RL；所有方法共用冻结 SDXL、Euler-NFE、prompt、同一有序 seed cohort 和 reward 查询预算。GenEval 的完整清单仍要求每个 prompt 四个样本；这里的“同一 seed”是复用同一组四个 seed，不是降为单 seed。只有在代码、权重、许可证和 equal-compute 协议均核验后，才考虑把 ToPO、STEP-OPD 或 Self-OPD 加入结果基线。
