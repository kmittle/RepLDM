# LR-1 固定 Latent Renderer 实验

## 为什么做这个实验

S5 失败后，我们把 Attention Guidance 重新解释为更一般的 latent rendering：冻结扩散模型，用一个远小于 SDXL 的模块在每个 scheduler step 后产生受限更新。LR-1 先不训练网络，只测试六个可解释 basis 的固定系数。如果固定动作都没有可靠 headroom，直接做蒸馏或 RL 只会放大选择噪声。

六个 basis 包括 semantic transport、低中高频、FreeU 风格 backbone-minus-skip 和 Laplacian。renderer 限制更新幅度，并尽量保持通道均值与方差。LR-1 使用的是后来确认有单位增益问题的 legacy post-step 注入，因此本报告只描述该旧算子的结果。

## 合规实验与结果

第一次 1440-image 运行误用了保留给 final 的 seeds `0,42,123`，所以只保留工程价值，不能用于选择。修正后的目录 `outputs/latent_renderer/lr1_fixed_train_searchseeds_v2` 使用 48 prompts、seeds `7,19,73` 和 10 actions，共 `1440/1440` 条记录。结果盲审计通过；最大 update ratio 为 `0.0112049`，mean/variance 最大绝对误差为 `1.19e-7/7.15e-7`。

冻结 selector 只看 train HPSv2，并要求 crossed-bootstrap CI 下界为正，同时通过 CLIP、clipping、saturation 和数值 guards。所有动作的 HPSv2 CI 都跨零：

| 动作 | HPSv2 差值 [95% CI] | CLIP 差值 |
|---|---:|---:|
| `semantic_neg` | +0.000149 [-0.000173,+0.000483] | -0.000424 |
| `spectral_high_pos` | +0.000088 [-0.000292,+0.000488] | -0.000242 |
| `balanced_pos` | +0.000132 [-0.000212,+0.000533] | -0.000331 |
| `spectral_mid_pos` | -0.000011 [-0.000356,+0.000326] | -0.000154 |

selector 返回 `no_ag`。描述性 TOPIQ 比较的最好均值也只有 `+0.000457`，CI 为 `[-0.000418,+0.001450]`。

## 结论与反思

LR-1 没有显示足够训练的静态信号，因此没有生成 validation YAML，也没有进入 distillation 或 RL。旧 clean-`x0` 路径后来被证明是单位增益 post-step nudge，而无 churn Euler 的正确 clean-endpoint gain 是 `1-sigma_next/sigma`；旧路径会放大约 `9.08x` 到 `15.77x`。所以这些结果既不能监督新 renderer，也不能证明 scheduler-native basis 无效。

科学结论只限于：legacy 注入下的这组固定 basis 失败。若研究新 scheduler-native 算子，必须重新登记方向、数据、seeds、scorer 和逐步 ledger，并先通过新的固定动作门槛。
