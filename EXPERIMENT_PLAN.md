# 实验计划：动态可学习的 Attention Guidance Scale

> **状态修订（2026-08-24）：** 下面的 DRaFT/RL 计划是历史设计，当前不满足进入条件。
> S5、LR-1 和 S6（FreeU 结构 baseline 及 feature-moment 对照）均未产生通过 guards
> 的固定动作，因此不得按本文旧计划启动训练。下一轮必须先登记新的
> scheduler-consistent static operator，并在 prompt-disjoint 数据上通过静态 headroom
> gate；详见 `MODEL_ITERATIONS.md`、`EXPERIMENT_RESULTS.md` 和
> `_experiments_expansion/rethinking/freeu_structural.md`。

> 分支：`rl-version` ｜ 定位：RepLDM (NeurIPS 2025) 的**期刊扩展**工作
> 目标：把手调的 guidance 超参 `(scale, density, decay)` 替换成一个**动态、可学习、(content/prompt)自适应**的控制器，去除人工消融、并提升质量与通用性。
> 说明：本文件是活文档（living doc），随实验推进更新。行号引用基于改动前的代码，仅作定位用，以符号名为准。

---

## 0. 背景与核心问题

RepLDM 会议版 = **training-free** 高分辨率生成 = Attention Guidance（手调 scale）+ 两阶段 pipeline。

Attention Guidance 的全部作用是每个去噪步后对 latent 做一次微调：

```
z_new = z + scale · (TFSA(z) − z)            # AttentionGuidance/attention_guidance.py，vanilla_attn_guidance + __call__
TFSA(z) = reshape⁻¹( softmax( f(z)·f(z)ᵀ / λ ) · f(z) )   # 通道维自注意力，softmax/matmul，全可微
```

其中 `scale ~ [0.001, 0.005]`，每步一个标量，来自**预计算的手调调度**（`guidance_scale` × `guidance_scale_decay` × `guidance_density` 门控）。

**痛点（用户）**：通过大量消融实验手调这个关键参数，不优雅、不 principled。
**扩展假设（待验证）**：最优 scale 是 **step 依赖**且 **content/prompt 依赖**的，因此一个动态自适应控制器能优于任何固定标量/调度。

---

## 1. 核心判断与方法选型（已定）

| 决策 | 选择 | 理由 |
|---|---|---|
| 训练范式 | **可微分 reward 回传（DRaFT-K）**，非 RL | 动作 `a_t` **线性**进入 latent（`z_{t+1}=z_t+a_t·δ_t`，`δ_t=TFSA(z_t)−z_t` 与 `a_t` 无关），且 reward（ImageReward）可微 ⇒ `∂R/∂a_t` 闭式、低方差，比黑盒策略梯度省 1–2 个数量级算力。 |
| 论文贡献定位 | **动态自适应控制器本身**是贡献，DRaFT-K 只是训练载体 | 用成熟可信的训练算法，reviewer 不质疑；贡献聚焦在“自适应 guidance”。 |
| 主基线 | **CMA-ES 学到的最优静态调度** | “自适应 > 最优静态”是核心实证主张，必须打败它，而不是只打败手调。 |
| RL | 作为**对照基线**写进消融 | 证明方法选型合理；非主线。 |
| 训练阶段 | **仅 Stage-1 / 1024²** | guidance 只在 Stage 1 起作用（Stage 2 不调用，见 §4）；base-res 解码廉价、可回传。 |

**为什么不是 RL（关键风险，见 §9 RISK-1/2）**：动作是 ~1e-3 的小标量，叠在整步 SDXL 去噪上，其对终局 ImageReward 的影响被种子方差与 reward 方差淹没；手调已在平坦最优附近 ⇒ 策略梯度又小又噪；终局稀疏 reward 摊到 50 步信用分配几乎不可能。

---

## 2. 论文定位：化解 “training-free” 张力

会议版卖点是 training-free。引入“训练”的控制器需诚实定位：

- **Base 扩散模型保持冻结、推理时 training-free**；只额外学习一个 **≪1M 参数的轻量外部控制器**，学一次、即插即用。
- **残差热启动**：控制器输出零初始化 ⇒ 训练第 0 步严格等于会议版手调调度 ⇒ 本方法是会议版的**严格泛化**，不是推翻。
- 叙事：把手调的 `(scale, density, decay)` 三元组替换成“**学一次的动态自适应调度器**”，主打 **自动化 + 自适应 + 通用性**。

---

## 3. 论文叙事结构（相对会议版的 delta）

```
Motivation / 分析章 → 最优 scale 是 step + content 依赖的；手调标量次优
    └─ 直接由 Phase 1 “卡关实验”产出分析图（敏感度曲线 / 逐步 leave-one-out / 跨 prompt 最优 scale 异质性）
Method → 轻量动态控制器 + DRaFT-K + reward-hacking 防护 + 残差热启动
Generality → 去除手调负担；跨分辨率 / ControlNet / 其他 backbone 泛化
```

---

## 4. 代码集成点（grounding）

**`AttentionGuidance/attention_guidance.py`**
- 唯一需要替换的量：每步标量 `scale`（`vanilla_attn_guidance` 的 `z + scale·(TFSA−z)`）。
- `__call__(t_index, latents, alpha_t, scale=None)`：`t_index` 是**反向步索引**（T=50 → 49..0）。
- `scale == 0` 精确复现无 guidance 的 latent ⇒ 控制器输出 0 即“OFF”，**自动吸收 `guidance_density`**；输出任意幅度 ⇒ 吸收 `guidance_scale_decay`。

**`InferencePipelines/RepLDM/pipeline_repldm_sdxl.py`**
- Stage-1 实例化 AttnGuidance：约 L1090–1097；调用点：约 L1152 `attn_guidance(num_timesteps-1-i, latents, alphas_cumprod_sample[i])`，在 Stage-1 去噪环（约 L1107–1159）内。
- Stage-1 base-res 解码：约 L1188 `vae.decode(latents / scaling_factor)`；**SDXL VAE 在 fp16 溢出**，pipeline 用 `upcast_vae()`（约 L713 / L1172–1191）。reward 解码必须复刻，或用 `madebyollin/sdxl-vae-fp16-fix`（仓库已引用）。
- **Stage 2 不用 guidance**：调用被注释（约 L1337）；Stage 2 从 `init_rate=0.8` 重启（约 L1296），只跑最后 ~20% 步、作用在 VAE 重编码 + anchor 归一化后的全新 latent 分布上。⇒ 控制器**只训练/作用于 Stage 1**，reward 在 Stage-1 1024² 上算。

---

## 5. Phase 0 —— 地基重构（✅ 已完成并验证）

`AttnGuidance` 解耦“外部传入 per-step scale”，**完全向后兼容**：
- `__call__` 增加可选 `scale` 参数；`scale=None` 时逐字节等价于原训练-free 路径（仍全程 `no_grad`）。
- 移除 `filter` / `vanilla_attn_guidance` / `__call__` 上的 `@torch.no_grad()`，改由调用方上下文控制；原路径用 `with torch.no_grad()` 包裹，行为不变。
- `scale` 为可微张量时，梯度可回流到 `scale`（及在 `latents` 需要梯度时回流到 attention nudge）。

验证（env `diff_attn`，已通过）：
1. 原路径 `requires_grad=False`（不变）；2. 可学习路径梯度回流到 `scale`；3. `scale=0` ≡ 恒等；4. 新路径在同一 scalar 下与原调度路径逐元素相等。

---

## 6. Phase 1 —— Motivation / 卡关实验（先做；既去风险又是论文图）

三个实验，**便宜、无训练**，用数据决定是否值得上学习式控制器，并直接产出分析章配图。

**Exp-1.1 常数 scale 敏感度扫描（生死关）**
- 固定 prompt + 固定种子，常数 `attn_guidance_scale ∈ {0, 0.001, 0.002, 0.003, 0.005}`，~50 prompts × 3 seeds。
- 画 ImageReward 均值 ± 种子方差带。
- **判据**：若跨 scale 的均值波动落在种子方差带内 ⇒ reward 看不见动作 ⇒ 任何学习方法都无效，停止/重构问题。成本 ~750 次 Stage-1 生成。

**Exp-1.2 CMA-ES 静态调度基线（设定必须打败的标杆）**
- 对现有 ~5 个超参（`scale_max, decay_type, min_scale, factor, density 门控`）做 CMA-ES，直接最大化固定评测集的均值 ImageReward（黑盒、population、天然抗噪）。
- 产出“最优静态调度”的 ImageReward。**学习式控制器必须显著打败此数，而非手调基线。**

**Exp-1.3 逐步 leave-one-out（验证 per-step 结构是否存在）**
- 单独关掉某一步 guidance（扫步索引），测 ImageReward delta vs 全开。
- 若无任何单步可辨（delta < 种子噪声）⇒ per-step 网络缺乏可利用结构 ⇒ 退化为 2–3 段粗调度。
- 产出“每步重要性”曲线（论文图）。

**（加分）Exp-1.4 最优 scale 的内容异质性**
- 对不同 prompt 桶（密集纹理 / 极简平面 / 人脸 / 风景）各自找最优常数/调度，比较其差异。若桶间最优显著不同 ⇒ content-adaptive 有依据（motivation 的核心证据）。

---

## 7. Phase 2 —— 控制器 + DRaFT-K 训练

### 7.1 控制器架构（method-agnostic：回传/RL 共用同一前向）

输出：每步标量 `a_t`，**残差热启动**形式
```
a_t = clamp( base_schedule[t] + a_max · tanh(head(feat_t)),  0,  a_max )
# head 零初始化 ⇒ 训练初始 ≈ 手调调度；a_max ≈ 1.5–2× 手调上限
```

条件输入 `feat_t`（消融阶梯，见 7.2）：
- **(C) 廉价 latent 统计量**：通道均值/方差、`||z||`、`alpha_t`、SNR、attn 熵（`vanilla_attn_guidance` 里现成的 `attn`）、FFT 频带能量比（`filter` 里现成）。≈零成本，`__call__` 内可算。
- **(D) prompt embedding**：`pooled_prompt_embeds`（低秩投影到 ~16 维）。
- **(full) 全 latent 卷积编码器**：硬下采样 + 全局池化 → 全局摘要（~0.1–0.3M 参数）。作为上限消融。

**建议起点：`D+C`（prompt + 廉价 latent 统计）**——对 “content-adaptive guidance” 叙事最有说服力且最便宜；full-latent conv 当上限对照。
策略分布（若走 RL 对照）：squashed Gaussian + 精确 Jacobian log-prob；否则回传用确定性输出。

### 7.2 消融阶梯（每级须超过下一级、差距 > 种子噪声）

```
0. 手调调度（待超越）
1. CMA-ES 静态调度（Exp-1.2）            ← 主基线
2. 自由 50-向量 / 仅 t 查表（等价于 1 的上界）
3. MLP(t)：只测“跨步平滑”
4. + 廉价 latent 统计（C 级）
5. + prompt embedding（D 级）
6. D+C
7. 全 latent 卷积编码器 / 全 latent + prompt
```
**判别实验（证明“真自适应”而非伪装成调度）**：
- *latent-shuffle*：喂别的 prompt/seed 的 latent，reward 不变 ⇒ latent 输入是装饰。
- *统计残差*：把 full 的输出回归到 C 统计量，`R²→1` ⇒ 没学到统计量以外的东西。
- *分桶分层*：full 应**仅在**有内容信号的桶里打败 C。

### 7.3 DRaFT-K 训练 harness（Stage-1 / 1024²）

- 仅对**最后 K=1–4 步**保留计算图；前面步 `no_grad` + 在窗口边界 `.detach()`。
- **UNet 永远冻结、`no_grad`**；`unet.enable_gradient_checkpointing()`（训练期）；用 fp16-fix VAE 保持解码在图内、避免 fp32 显存爆。
- 显存预算：DRaFT-K(1–4) + 梯度检查点 + base-res ⇒ 峰值 16–24GB，单卡可训。
- DRaFT-LV（随机截断点）降方差 + 抗 hacking。
- Loss = `−R_total`（见 §8）。

---

## 8. Reward 设计与 anti-hacking（必做）

ImageReward 系统性偏好高饱和/高锐度 —— 正是 `scale·(TFSA−z)` 移动的轴 ⇒ 代理梯度会把 scale 往上顶。防护组合：

```
R_total = z(IR) + Σ guards
          − β·‖a‖₁                      # 幅度/稀疏惩罚，顺带学出 density
          − η·‖a − a_expert‖²            # 锚定手调调度的信任域（早期强、退火）
          − λ·Var_models(z_R)            # 多 reward 集成的“分歧惩罚”
          − μ·saturation_excess          # 解码 RGB 上的直接饱和度/裁剪像素惩罚
```
- **硬 clamp `a_max`**：结构上禁止越界（最强、最便宜的护栏）。
- per-prompt 标准化 ImageReward（原始分只有相对意义，~[-3,3]）。
- **（可选）potential-based x0 预测 shaping**：用 `noise_pred + alphas_cumprod_sample[i]` 形成 `x0_hat`，对稀疏子集步解码打分，按 `alpha_t` 加权；仅作 shaping，终局 reward 为主。
- **早停看留出指标**：训练监控未被优化的 HPSv2 / aesthetic / CLIP / colorfulness，在**留出指标见顶处**早停（而非 IR 见顶处）。

---

## 9. 风险登记（来自对抗性评审）

| ID | 严重度 | 风险 | 最便宜的确认/否决实验 |
|---|---|---|---|
| RISK-1 | 高 | 动作 SNR 太低，reward 看不见 1e-3 的扰动 | **Exp-1.1** 敏感度扫描 |
| RISK-2 | 高 | 终局稀疏 reward 摊 50 步，信用分配无解 | **Exp-1.3** 逐步 leave-one-out |
| RISK-3 | 高 | RL 全 rollout 成本 > 它要替代的消融 | 估单次 rollout 墙钟 × rollout 预算 vs 历史消融成本（→ 选 DRaFT-K 缓解） |
| RISK-4 | 高 | 零迁移到 Stage 2 / 高分辨率 / ControlNet / 其他 backbone | **Phase 3** 迁移实验；先确立手调标量的跨设定鲁棒性作为 bar |
| RISK-5 | 高(定位) | “training-free” 框架被破坏；贡献被读成“只是学了个调度” | §2 定位 + §3 叙事 + 残差热启动 |
| RISK-6 | 高(论证) | 过度工程：静态学习调度/CMA-ES 可能拿 95% 收益 | **Exp-1.2** CMA-ES 设为主基线 |
| RISK-7 | 中 | 一次性消耗的 iterator（`guidance_step_scale`/`filter_range`）非可重入 | Phase 0 已解耦 scale；filter 路径用到时再处理 |
| RISK-8 | 中 | reward hacking / 分布坍缩 | §8 防护 + 留出指标早停 |

---

## 10. 评估协议

- **生成预算**：搜索/消融阶段只跑 Stage-1（~1024²）；最终赢家再过一次完整 Stage-2 校验（上采样可能冲淡/放大 Stage-1 纹理差异）。
- **数据划分**：在一组 prompt 上搜索/训练，在**留出 prompt**（photo / art / text-render / flat-design / faces / dense-texture 多桶）上报告；绝不在被优化过的 prompt 上报。固定 4–8 seeds，报均值 ± CI + 同种子配对比较。
- **抗 hacking 的第二指标（必报）**：优化 IR 就必须同时报未优化的 CLIP / HPSv2 / aesthetic；真赢会带动留出指标，hacking 只动 IR。
- **过饱和体检**：留出输出的逐通道饱和直方图、裁剪像素比例、对比度、colorfulness；IR 涨而这些同涨 = hacking 信号。
- **调度可视化**：每个学到的策略 dump 每步 `a_t` 曲线，叠加手调调度对比（≤50 个数，可完全可视化）。
- **停止规则**：采纳“**第一个**留出 IR 增益 (a) 超种子 CI、(b) 第二指标不降、(c) 无饱和度升高”的最低阶梯。最可能是 A2 或 C，可能 D；full-latent 仅在通过 shuffle/残差/分层判别实验后才采纳。

---

## 11. 待决策项（OPEN）

- **[OPEN-1] 控制器条件输入**：从 `D+C` 起步（建议）vs 直接 full-latent。→ 影响 Phase 2 网络结构。
- **[OPEN-2] Phase 顺序**：先 Phase 1（建议，低风险 + 直接出图）vs 直接 Phase 2 训练 harness。
- **[OPEN-3] conda 环境**：默认 `diff_attn`（torch 2.5.1 + diffusers 0.32.1）。注意 `pyproject` 锁 diffusers~=0.21.4，跑 pipeline 可能有 API 差异需适配。

---

## 12. 环境与运行

- 工作 env：`conda run -n diff_attn ...`（torch 2.5.1 / diffusers 0.32.1）。`AttnGuidance` 纯 torch，不受 diffusers 版本影响。
- 模型走本地缓存（`local_files_only=True`），cache_dir 相对路径，需预置 SDXL / ControlNet / DPT 权重。
- 入口示例：`InferCases/RepLDM/SDXL/`（t2i 与 controlnet）。

---

## 13. 评审补充（对抗性、代码-grounded review 产出）

> 来源：一轮 3×代码 grounding + 4×对抗性视角（优化/RL 理论、reward-hacking/评估、文献/novelty、Occam/方法设计）。**原 §0–§12 不改**，本节为增量修正。标注「✅已验证」= 读真实代码确认；「建议」= 待验证提案。

### 13.0 已验证的代码事实（可直接采信）
- ✅ **H1 成立**：`attention_guidance.py:266/269` 两条路径都是 `latents = latents + scale·(vanilla_attn_guidance(filter(latents),alpha_t) − latents)`；`δ=TFSA(filter(z))−z` 与 `scale` 无关 ⇒ `dz/dscale = δ` 精确，DRaFT 前提结构上成立。
- ✅ **Phase 0 正确**：`scale=None` 走 `no_grad`（与会议版逐字节等价）；`scale` 为张量时全程记录梯度，且梯度可同时回流到标量与 TFSA 的 softmax/matmul。
- ✅ **ImageReward 不在 `diff_attn`**：仅安装在 env `sana_cby`（py3.11，含 `clip/open_clip/hpsv2`）。DRaFT/搜索训练必须在 `sana_cby` 跑，或把 `ImageReward` 装进 `diff_attn`。**§12 的"工作 env=diff_attn"对 reward 路径不适用。**
- ✅ **可微 reward 入口是 `score_gard`（非 `score`）**：`ImageReward.py:84-100` 纯 torch、无 detach；`score()`（:103-137）走 PIL + `.detach().cpu().numpy()`，不可用于回传。**`ReFL.py:744-756` 是仓库内现成的 DRaFT/ReFL 配方**（in-graph VAE decode → torchvision Resize(224)+CenterCrop+Normalize → `score_gard` → `F.relu(2−reward)` → backward），直接复用。
- ✅ **一次性迭代器不可重入**：`guidance_step_scale`（L111）与 `filter_range`（L203）是 `iter(...)`。`scale` 路径不消费 `guidance_step_scale`，但启用 `guidance_filter` 时 `filter()` 仍每步 `next(filter_range)` ⇒ **单个 `AttnGuidance` 实例只能跑一次去噪环**。这会阻塞**每一个**多-rollout 循环（CMA-ES 搜索、DRaFT），原 RISK-7「中/later」严重低估。
- ✅ **pipeline L1152 尚未接 `scale` kwarg**：控制器 hook 在 `AttnGuidance` 已就绪，但 `pipeline_repldm_sdxl.py:1152` 还没把控制器输出喂进去。
- ✅ Stage-2 不调用 guidance（L1337 注释掉），`init_rate=0.8` 重启，确认控制器只作用于 Stage-1。

### 13.1 关于 RL vs DRaFT：不是推翻你的决策，是**限定作用域 + 补三处**
1. **Stage-1 上 DRaFT-K 仍是对的**（前提 H1 已验证；RL 在 ~1e-3 标量、近平坦最优上做策略梯度，方差吞噬信号，你 §1 的判断成立）。
2. **但"linear ⇒ 低方差梯度"只对末步的直接路径成立**。`δ_t` 依赖 `z_t`，`z_t` 又由所有 `a_{<t}` 经 K 个**冻结 UNet** Jacobian 复合而来；早步 `dR/da_t = (dR/dz_final)·∏(frozen-UNet Jacobian)·δ_t`，**无权重更新去吸收条件数**，可像 BPTT 一样消失/爆炸。DRaFT-LV 只是降方差的**后缀**梯度，不完全修截断偏置。⇒ **新增 Exp-1.5（SPSA 梯度核对，~10 次 no-grad rollout）**：整条 `a` 调度做 ±ε·Rademacher 扰动两次前向，`(R₊−R₋)/2ε` 投影出真·全轨迹方向导数，与 DRaFT-K 解析梯度对比 —— 同时验证 autograd 路径（含 fp16 FFT 往返）、量化早步截断偏置。这是**信息/算力比最高**的实验，应在搭控制器前先做。
3. **RL 有一个你的计划"截肢"掉的合法生态位 = Stage-2 端到端**。Stage-1-only 训练优化的是**proxy**；你实际交付的是 Stage-2 之后的 3K/8K 图（穿 tiled VAE 重编码 + restart，回传不可行，但前向 rollout + 终局 hi-res reward 廉价）—— 这正是零阶/RL **结构上唯一可用**的地方，把 RISK-4 从"祈祷"变成"可优化目标"。**建议 hybrid**：DRaFT-K 做 Stage-1 内层 `a_t`，外层用 CMA-ES/SPSA（Exp-1.2 已要建）裹住 Stage-2 的 1–2 个 `init_rate`，用终局 hi-res reward。外层维度 1–2，几十次评估收敛。
4. **reviewer 的 RL 对手不是 DDPO/DPOK（2022），是 GRPO-for-diffusion / Diffusion-DPO / D3PO（2024-25）**。一句话区分写进论文：*它们微调 base UNet 权重以移动采样分布；我们在**冻结**模型上学一个加在固定修正方向 `δ_t` 上的微小标量增益 —— gradient 精确（H1），GRPO 的组相对方差缩减在此无关，而其动作采样会破坏已知偏低的 action-SNR（RISK-1）。*

### 13.2 ⚠️ 最重要的新发现：ImageReward 在 224px 下**看不见 detail**（可能动摇主论点）
- ✅ **代码确认 H2**：`ImageReward` 的 BLIP ViT-L 硬编码 224×224（`_transform(224)=Resize(224,BICUBIC)+CenterCrop+Normalize`，pos-embed 也是为 224 写死）。1024→224 ≈ **21× 面积**双三次下采样 = 低通滤波：**guidance 注入的高频纹理几乎被抹掉，而 color/saturation 低频幸存**。可微 `score_gard` 走**同一** 224 瓶颈。
- ✅ **且这是系统性的**：env 内**所有**可学 reward（HPSv2 open_clip ViT-H、aesthetic/CLIPScore ViT-L/14）都卡在 224px，aesthetic 是 CLIP 特征上的小 MLP、**更**盲于细节。
- **双重后果**：(a) "自适应控制 detail"的卖点对 reward **基本不可见**；(b) reward 能看见的、且 `δ_t=TFSA−z` 主要在动的轴恰是 **saturation** ⇒ **reward-hacking 阻力最小路径**就是顶饱和度。这也解释了为什么 §8 要堆那么多护栏。
- **缓解（建议）**：(1) **patch/multi-crop IR** —— 对 1024² 解码图取若干 224 随机/中心裁剪过 `score_gard` 再平均，把原生分辨率细节留在画面内、仍可微，恢复 detail 敏感的梯度方向；(2) 加一个**全分辨率高频/锐度可微辅助 reward**（Laplacian 方差 / 高通能量 / FFT 频带），给梯度一个**非饱和**的攀爬方向；(3) **reward 前对解码图做 color/histogram 归一化**（gray-world），使 reward 对全局色移**构造性不变** —— 唯一能涨分的途径只剩真实结构。验证：patch-IR 是否在 global-IR 单调的区间出现内部最优。

### 13.3 主方法重新洗牌：把 SEARCH-THEN-DISTILL 升为**一等方法**，让 DRaFT **挣**它的复杂度
- **SEARCH-THEN-DISTILL（建议升为主候选，原计划只当基线）**：per-prompt（或 per-bucket）CMA-ES 在 **guard+holdout 校验过的** reward 上搜最优低维调度 → 收集 `(prompt_embedding → 最优调度)` → 拟合一个 tiny regressor（ridge / 2 层 MLP，≪10k 参数）。部署即 regressor，**零 backprop-through-diffusion**。因为回归**目标是搜索-校验过的最优**，它**结构上免疫"穿过计算图的 reward-hacking"** —— §8 整章防的东西在这里**根本不存在**；同时删掉 in-graph VAE、grad checkpoint、DRaFT-LV、6 项 reward 的全部工程。**做出与论文同款主张（"learned, content-adaptive > best static"），代价是零头。**
- **DRaFT-K 的真实优势不是 reward 更高（近平坦最优处大概率打平），而是 amortization**：一次训练泛化到**未见 prompt**、无需 per-prompt 搜索。**让 DRaFT 挣复杂度**：在 holdout IR **和** "覆盖 N 个新 prompt 的成本"两条轴上打败 search-then-distill 才上位；明确写出 crossover N（N 小则 distill 主导，N 大/要广泛泛化则 DRaFT 的摊销是真故事）。
- **三方法同源对照**（隔离"自适应"与"梯度优化器更强"）：用**同一** DRaFT-K harness 把**静态基线也做成梯度优化的 50-向量**（vs 黑盒 CMA-ES）。若梯度-静态≈条件控制器，自适应主张坍塌；若条件控制器 > 梯度-静态，赢点**明确**来自 content-adaptivity 而非"梯度比 CMA-ES 强"。

### 13.4 Motivation 重排（叙事 delta 升级）—— 三视角独立收敛
- **把 per-prompt ORACLE GAP 升为头号 motivation 图**（原 Exp-1.4 从"加分"提为主）：每个 prompt 搜其**最优调度**（oracle，无训练），对比 Exp-1.2 的**最优全局静态**。这个 gap = **content-adaptivity 的全部可得收益**，且**任何静态调度按构造都关不掉**，也是任何控制器的**上界**。贡献改报为 **"recovered X% of the oracle gap"**（reviewer 接受"关掉 60% 静态够不着的 headroom"，不接受"IR +0.03"）。**若 gap 落在 seed-CI 内 ⇒ 整个扩展在训练前就该停**（最便宜的 go/no-go）。
- **Exp-1.1 换判据**：不是"均值波动 > 种子带"，而是 **IR(scale) 非单调、有内部 argmax `a*>0` 且跨 prompt 异质**。**单调** ⇒ 最优策略退化为"顶到 `a_max`/clamp"，项目变成"调 clamp 阈值"而非"学 guidance"。复用同一 ~750 次生成 + 在手调上限之上加几点（0.004/0.007…）看拐头。报告"内部最优占比"与 `a*` 跨 prompt 离散度。
- **跨 ≥2 个 reward 验 content-optimum**（ImageReward + HPSv2/aesthetic）并**预注册** `gap > seed-CI` 为硬 go/no-go：TFSA 主要动饱和、IR 也均匀偏好饱和 ⇒ 各桶最优可能**内容不变**（"大家都想更 pop 一点"），gap≈0 会**静默作废**自适应论点（RISK-14）。

### 13.5 反 hacking 升级（§8 的修正）
- **第二指标不是独立见证**：HPSv2/aesthetic/CLIP 全是 CLIP/ViT@224，与 saturation/colorfulness **正相关** ⇒ "IR 涨而第二指标平"可能**永不触发**，全体随 hack 同涨、读成"干净的赢"。**预注册相关性 disqualification**：在 Exp-1.1 输出上把每个 guard 回归到 per-image colorfulness，`|r|>0.5` 者取消见证资格；**至少换一个去相关 witness**（全分辨率 sharpness/NIQE、color-normalized CLIP、photo 桶 FID），**早停/选型用去相关指标**而非 IR 家族。
- **构造性去饱和 > 单边惩罚**：§8 的 `saturation_excess` 只把优化器**停在惩罚边界**（=最大允许饱和，正是最坏的"调 guard"结局）。改用 (a) 把 IR 梯度对 colorfulness 梯度做**正交化投影**，或廉价 (b) **reward 前 color 归一化**（见 13.2）。
- **power_calibrate / λ 混淆**：`vanilla_attn_guidance` 的 `power_calibrate`(0/1/2, L236-241) 与 `attn_scaling`(λ) 都直接调饱和轴。**CMA-ES 静态基线必须把这两维纳入搜索空间**，否则静态基线被削弱、控制器 margin 虚高（reviewer 会说"没给静态基线同样的旋钮"）。每次 sweep 都固定并汇报 `power_calibrate`。
- **de-scope §8**：先只上 **hard-clamp + trust-region-to-expert**（你自己说 clamp 是最强最便宜的护栏），其余 4 项（ensemble 方差、x0-shaping…）**按 holdout 指标下降时反应式地加**，别在 Exp-1.1 确认信号前就全建。

### 13.6 over-engineering 判据（修正一个数学错误）
- ✅ **schedule→reward 非可分**：`δ_t` 依赖 `z_t`、`z_t` 含全部历史 `a_{<t}` ⇒ 各步贡献**不可加**。**Exp-1.3 leave-one-out 是必要非充分**（它只测"其余保持 expert"的一阶敏感度，是耦合曲面上的一个点）；**不能**由"几步显著"反推"per-step 结构存在"。
- **决定性的 over-engineering 测试**：`CMA-ES{2–3 段调度}` vs `CMA-ES{自由 50-向量}`。若 50-向量没 > seed-CI 地打败 2–3 段，**控制器就不该输出 per-step `a_t`**（退化为 2–3 段），直接杀掉 RISK-6。
- **控制器输入从 C / D-only 起步**（非 D+C）：C（`alpha_t`+通道均值/方差/‖z‖+attn 熵）≈零成本、负责 step/state 自适应；**D-only**（纯 prompt、无 latent 反馈）是**最干净的 content-adaptivity 测试**，也直接喂 latent-shuffle 判别实验。full-latent conv encoder 推到 shuffle/残差/分层判别**通过后**再建。

### 13.7 Phase 0.5 —— must-fix（阻塞一切多-rollout 实验，先于 Phase 1）
1. **迭代器可重入**：把 `guidance_step_scale`(L111)/`filter_range`(L203) 的 `iter()` 换成**按 `t_index` 索引预计算张量** + 加 `reset()`；否则 CMA-ES（需数百次 replay）与 DRaFT rollout 第二次就 `StopIteration`，或被"循环 cycle"悄悄喂错每步 filter 窗（伪装成训练噪声的 reward 腐蚀 bug）。
2. **接线**：`pipeline_repldm_sdxl.py:1152` 把控制器输出作为 `scale` kwarg 传入；核对去噪环 UNet 调用的 grad/no_grad 上下文与 DRaFT-K 兼容。
3. **reward 环境**：在 `sana_cby` 或将 `ImageReward` 装进 `diff_attn`；复用 `ReFL.py:744-756` 配方；用 `madebyollin/sdxl-vae-fp16-fix` 保解码在图内（避开 upcast 到 fp32 后无法回传）；reward-grad pass 对 `(b,c,c)` softmax + FFT 往返用 fp32/autocast 防数值噪声。

### 13.8 related work 必补（reviewer "not novel" kill 防御）
- **自适应/学习式 guidance-scale 线**（计划完全没提）：CFG scheduling / Guidance-Interval（Kynkäänniemi 2024）、Autoguidance（Karras 2024）、CADS、dynamic/rescaled CFG（Lin 2024）。差异化：*它们调的是 CFG 项（文本条件外推），与采样纠缠；我们调的是**冻结**模型上一次 post-step 的 TFSA 修正，正交且可与任意 CFG 调度复合。*
- **现代 RL 对手**：见 13.1.4（GRPO-for-diffusion / Diffusion-DPO / D3PO）。

### 13.9 修订后的 Occam kill-ladder（把并行雄心改成串行漏斗）
```
Gate-0.5 wiring  : 迭代器可重入 + 接 scale kwarg + reward 环境就绪   ← 不通过寸步难行
Gate-1 SIGNAL    : Exp-1.1 → IR(scale) 有内部最优？  否⇒停（无方法可救）
Gate-1.5 GRAD    : Exp-1.5 SPSA 核对 → DRaFT 梯度可信、早步偏置可控？
Gate-2 GRANULAR  : CMA-ES{2-3 段} vs {50-向量} → per-step 值得吗？  否⇒退化为段调度
Gate-3 ADAPTIVE  : per-bucket / per-prompt oracle gap 跨 ≥2 reward > seed-CI？  否⇒自适应论点死
Gate-4 METHOD    : SEARCH-THEN-DISTILL 设为主方法跑 holdout；
                   DRaFT-K 仅当在 holdout IR + 去相关第二指标 + 摊销成本上打败它才上位
```

### 13.10 风险登记新增
| ID | 严重度 | 风险 | 最便宜的确认/否决 |
|---|---|---|---|
| RISK-9 | 高(可能致命) | reward 在 224px 看不见 detail，只见 saturation ⇒ 卖点不可见 + 强化 hacking | ✅已确认(H2)；缓解 13.2 patch-IR/HF-reward/color-norm |
| RISK-10 | 高 | 早步 `dR/da_t` 经 K 个冻结 UNet Jacobian 连乘，消失/爆炸；DRaFT-LV 不完全修截断偏置 | **Exp-1.5** SPSA 梯度核对(~10 rollout) |
| RISK-11 | 中-高 | Stage-1 proxy ≠ shipped hi-res；DRaFT 决策只对 Stage-1 成立 | Stage-2 `init_rate` 零阶外环(hybrid 13.1.3) |
| RISK-12 | 中(阻塞) | 一次性迭代器不可重入，阻塞所有多-rollout 循环 | ✅已确认；Phase 0.5 索引化 |
| RISK-13 | 中 | 第二指标与 hack 相关，非独立见证 | colorfulness 相关性 disqualification + 去相关 witness |
| RISK-14 | 中-高 | content-optimum 可能内容不变(saturation 驱动)，oracle gap≈0 ⇒ 否决自适应 | oracle gap 跨 ≥2 reward，预注册 go/no-go |
| RISK-15 | 中(定位) | related work 缺失(adaptive-guidance line + GRPO) ⇒ "not novel" | §13.8 补文献 + 差异化 |

## 14. S7 实验漏斗：scheduler-consistent correction

S7 是与 S5/LR-1/S6 不同的固定动作候选。它用 scheduler 的 `sigma` 转移计算
ancestral drift，并以 `mix` 和可选 trust cap 注入 Euler transition。实现和公式见
`_experiments_expansion/rethinking/trajectory_correction.md`；相关 predictor-corrector
和 trajectory-correction 工作只作为动机与竞争基线，不能被写成新 RL 算法。

执行顺序固定如下：

1. 在 development manifest 上只比较 `no_correction`、deterministic drift
   (`none`, `.25/.50`) 和 stochastic ancestral (`sqrt`, `.25/.50/.75`)，核对 exact
   `mix=0` parity、sidecar diagnostics 和 finite pixel guards。
2. 若 development 有信号，重新冻结更大的 prompt-disjoint validation split，先
   选择一个固定 mix；不得在 validation 上追加 mix、noise mode 或 cap。
3. 只有固定动作在 validation 同时通过 TOPIQ、HPSv2/CLIP 非劣、clipping/saturation
   guards、crossed bootstrap、prompt sign-flip 和 Holm gate，才允许设计小型
   state-conditioned latent renderer。renderer 先 search-then-distill；RL 只有在
   同一 held-out 协议下胜过固定搜索和蒸馏后才进入。
4. 任一 gate 失败就关闭 S7，不用更复杂 controller 或写作手段掩盖固定动作没有
   headroom。

本轮开发 split 使用的 PartiPrompts commit `5a657978134374ce28973948331b319adef164bd`
和源/prompt hash 已在本地 clone 中核验；它仍只能标为 development，不能当 TPAMI 的
最终 test。正式 validation/test 必须重新冻结更大的 prompt-disjoint split，并记录
完整 source/prompt hash。
