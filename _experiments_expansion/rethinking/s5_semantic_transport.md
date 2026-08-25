# S5 双向语义传输实验

## 为什么做这个实验

S0-S4 一直从四通道 noisy latent 计算 TFSA 残差。结果表明，继续调 scale、频段或投影几何只会接近 `no_ag`，或者增加饱和与 clipping。S5 因此换了更新来源：读取冻结 SDXL 一次正常 forward 中的 self-attention，用互为近邻的连接构造空间图，再搬运 scheduler 预测的 clean latent。

这个设计与会议版的联系是“用模型内部结构重写冻结生成轨迹”，但不是原 TFSA 的简单放大。PLADIS、GAG、SAG/PAG/SEG 等工作已经覆盖 attention contrast 或额外 prediction，因此 S5 不能把 attention guidance 本身当作新意，也必须直接比较这些基线。

## 实验与结果

正式目录 `outputs/exp_s5/development_12prompt_3seed_v2` 包含 12 prompts、3 seeds、14 actions，共 `504/504` 个同卡配对样本。所有非零 semantic 动作都有完整的 50 步 transport、entropy、confidence 和 update-ratio 记录。

相对 `no_ag`，reciprocal semantic 的 TOPIQ-NR 结果为：

| scale | 平均差值 | 95% crossed-bootstrap CI |
|---:|---:|---:|
| 0.005 | +0.000459 | [-0.000445, +0.001400] |
| 0.010 | +0.000509 | [-0.000571, +0.001771] |
| 0.020 | -0.000233 | [-0.001787, +0.001227] |
| 0.040 | -0.000775 | [-0.002919, +0.001477] |

最好均值只有 `+0.000509`，远低于预先登记的 `+0.005`，区间也跨过零。permuted graph、matched latent、clean/raw TFSA、会议版 expert、PLADIS 和 GAG 同样没有通过。固定拼图没有显示稳定的计数、位置、文字或细节修复。

## 结论与反思

结果不支持“互为近邻 attention 图能稳定改善无条件画质”。semantic graph 与 permuted control 的差异也太小，无法建立图结构的因果作用。S5 因此按停止规则关闭，不再搜索 layer、top-k、angle、schedule 或 controller，也不能拿这些输出训练 RL。

这并不证明所有内部结构都无效。它只否定了这套已登记的 reciprocal transport 算子。后续假设必须使用新的更新来源、新 prompts 和新的静态门槛；不能用写作把 S5 的空结果包装成方法成功。
