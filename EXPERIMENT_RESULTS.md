# 实验结果与阶段性结论

> 分支：`rl-version` ｜ 配套文档：[`EXPERIMENT_PLAN.md`](./EXPERIMENT_PLAN.md)（计划/风险登记）
> 定位：RepLDM (NeurIPS 2025) 期刊扩展——把手调的 Attention Guidance 超参 `(scale, density, decay)` 换成**动态、可学习、内容自适应**的控制器。
> 说明：本文件是活文档，记录**已经跑出来的**结果与由数据得出的判断；假设/待办仍以 `EXPERIMENT_PLAN.md` 为准。数值均可由 §7 命令复现。

---

## 0. TL;DR（一句话结论）

**在 12-prompt × 3-seed 的 pilot 上，Attention Guidance 的 scale 是一个对像素空间（饱和度/锐度/对比度/裁剪）单调且显著的旋钮，但对 ImageReward 基本不可见——最优 scale 的"内容自适应红利"（oracle gap = +0.047 ± 0.100）落在种子噪声内，无法与 0 区分。** 因此**在扩大样本 + 补足 ≥2 个去相关 reward 之前，不进入 Phase 2 训练**。这既是核心 go/no-go 门（`EXPERIMENT_PLAN.md` §13.9 Gate-1/Gate-3）的**当前判定**，也直接印证了 RISK-1（动作 SNR 太低）与 RISK-9（224px reward 看不见 detail）。

---

## 1. 当前进度总览

| 阶段 | 内容 | 状态 |
|---|---|---|
| **Phase 0** | `AttnGuidance` 解耦 per-step `scale`，向后兼容 | ✅ 已完成并验证（见 §2） |
| **测量仪** | `eval-pipeline/`：解耦的两阶段生成→打分→聚合 harness | ✅ 已搭好并跑通（见 §3） |
| **Exp-1.1** | 恒定 scale 敏感度扫描（生死关，Gate-1/Gate-3） | ✅ pilot 已跑完并分析（见 §4，**主结果**） |
| Exp-1.2 | CMA-ES 静态调度基线（必须打败的标杆） | ⬜ 未做 |
| Exp-1.3 | 逐步 leave-one-out（per-step 结构） | ⬜ 未做 |
| Exp-1.4 | 最优 scale 的内容异质性（per-bucket/per-prompt oracle） | 🟡 pilot 数据已有初步信号（见 §4.5），未做专门实验 |
| Exp-1.5 | SPSA 梯度核对（DRaFT 梯度可信度） | ⬜ 未做 |
| Phase 2 | 控制器 + DRaFT-K 训练 | ⛔ 被 Gate-1/Gate-3 阻塞，暂缓 |

---

## 2. Phase 0 —— 地基重构（✅ 已完成并验证）

`AttnGuidance.__call__` 新增可选 `scale` 参数，**逐字节向后兼容**：

- `scale=None` 走原 `no_grad` 路径，与会议版训练-free 调度逐元素相等。
- `scale` 为可微张量时，梯度可回流到 `scale`（以及在 `latents` 需要梯度时回流到 TFSA 的 softmax/matmul）——DRaFT 的结构前提 `dz/dscale = TFSA(z)−z` 成立。
- 已通过 4 项单元验证：①原路径 `requires_grad=False`；②可学习路径梯度回流到 `scale`；③`scale=0` ≡ 恒等；④新路径在同一标量下与原调度路径逐元素相等。

（详见 `EXPERIMENT_PLAN.md` §5 / §13.0。）

---

## 3. 测量仪 —— `eval-pipeline/`（✅ 已搭好并跑通）

解耦的三段式 harness，生成与打分分处不同 conda 环境，靠磁盘上的 PNG + JSON manifest 通信，打分可重跑、可扩展：

```
generate.py  [生成环境]   prompt×seed×scale → images/*.png + manifest.jsonl
score.py     [打分环境]   manifest + PNG    → scores.jsonl
aggregate.py [分析环境]   manifest + scores → eval_results.csv + analysis.png + go/no-go 诊断
```

**已实现的打分指标**（`scorers/`，每个是自包含模块；**精确定义与参考文献见 §8**）：

| 指标 | 说明 | 本轮是否已算 |
|---|---|---|
| `imagereward` | 全图 IR（224px 下采样 → 偏色彩/布局，**非 detail**） | ✅ |
| `patch_ir_mean/std/n` | 原生分辨率 224 crop 上的 IR（detail 敏感，§13.2） | ✅ |
| `colorfulness / laplacian_sharpness / mean_saturation / clipped_fraction / contrast_std` | 无权重、去相关的 reward-hacking 见证指标（§13.5） | ✅ |
| `clip_cosine / clipscore` | CLIP-Score 对齐见证 | ⬜ **本轮未算** |
| `hpsv2` | HPSv2 人类偏好 | ⬜ **本轮未算** |
| `aesthetic` | LAION 美学分 | ⬜ **本轮未算** |

> ⚠️ **重要缺口**：CLIP/HPSv2/aesthetic 三个见证指标本轮**没有跑**。`EXPERIMENT_PLAN.md` §13.4 预注册的 go/no-go 要求 oracle gap **跨 ≥2 个 reward** 验证；当前只有 ImageReward 家族（global-IR + patch-IR），**该门尚未真正满足**。

---

## 4. Exp-1.1 —— 恒定 scale 敏感度扫描（主结果）

### 4.1 实验设置

| 项 | 值 |
|---|---|
| 输出目录 | `outputs/exp1.1_scale_sweep/pilot_12prompt_3seed_7scale/` |
| 生成 | **仅 Stage-1，1024²**（Stage-2 跳过），恒定 per-step guidance scale |
| Prompts | 12 条，6 桶各 2 条：`dense-texture / flat-design / photo / face / art / text-render`（`eval-pipeline/prompts/eval_v1.csv`） |
| Seeds | `{0, 42, 123}`（3 个，**同种子跨 scale 复用 → 配对设计**） |
| Scales | `{0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.007}`（0.0 = 无 guidance 基线） |
| 固定混淆量 | CFG=7.5，`power_calibrate=0`，50 步，负 prompt 固定 |
| 总生成数 | 12×3×7 = **252 张 Stage-1 图**，全部完成并打分 |
| 生成时 commit | `c744783`（记录于 `config.json`） |

![scale sweep montage](outputs/exp1.1_scale_sweep/pilot_12prompt_3seed_7scale/figs/scale_sweep_montage.png)

*图 1：每桶取 1 条 prompt（seed 0），7 个 scale 从左到右。像素随 scale 明显变化（饱和度/锐度上升，0.007 处 Persian 地毯被过度处理、text-render 文字劣化），而绿框（该行 IR 最优）散落在不同列、且 3/6 落在红色（>10% 像素裁剪/过曝）的 tile 上——**IR 的"最优 scale"既不稳定也常落在画质已破坏处**。*

### 4.2 结果 A：均值 IR vs scale —— 无群体信号（RISK-1）

| scale | mean IR | std |
|---|---|---|
| 0.0000 | +0.853 | 0.696 |
| 0.0010 | +0.874 | 0.692 |
| 0.0020 | +0.843 | 0.699 |
| 0.0030 | +0.797 | 0.764 |
| 0.0040 | +0.842 | 0.704 |
| 0.0050 | +0.818 | 0.733 |
| 0.0070 | +0.908 | 0.736 |

- **均值曲线跨 scale 的极差 = 0.111**，而**典型的 per-prompt 种子标准差 = 0.241**。
- **→ 极差落在种子噪声带内：群体层面看不到 scale 的作用（RISK-1 触发）。**

### 4.3 结果 B：配对 Δ IR(scale)−IR(0) —— 无一显著

配对设计（同 prompt 同 seed，只有 scale 不同，消除了 prompt 难度方差）：

| scale | 配对 Δ mean | ±2·SE | 显著? |
|---|---|---|---|
| 0.0010 | +0.022 | 0.034 | 否 |
| 0.0020 | −0.010 | 0.044 | 否 |
| 0.0030 | −0.056 | 0.075 | 否 |
| 0.0040 | −0.011 | 0.065 | 否 |
| 0.0050 | −0.034 | 0.067 | 否 |
| 0.0070 | +0.055 | 0.075 | 否 |

- **即便在配对设计下，没有任何 scale 的 Δ 超过 2 个标准误。** 这是 RISK-1 最干净的陈述：**guidance 在 ImageReward 上移动不出配对噪声。**

### 4.4 结果 C（决策关键）：seed-CV oracle gap —— 自适应红利 ≈ 0

留一种子交叉验证（train seed 上选 scale，held-out seed 上评估相对无-guidance 的增益）：

| 策略 | Δ IR（held-out） |
|---|---|
| 最优**全局静态** scale | **−0.028 ± 0.060** |
| **per-prompt 自适应** scale | **+0.019 ± 0.135** |
| **ORACLE GAP（自适应 − 静态）** | **+0.047 ± 0.100** |

- **oracle gap = +0.047 ± 0.100 → 与 0 无法区分**（std 是 mean 的 2 倍）。这个 gap 是"内容自适应"能拿到的**全部红利上界**；它在噪声内 ⇒ **自适应论点在当前样本上不成立（RISK-14 触发）**。
- 更糟：**最优全局静态 scale 在 held-out 种子上是负增益（−0.028）**——train 上选出的"最优静态"泛化到新种子反而略微掉分，说明当前信号大部分是种子噪声的过拟合。

### 4.5 结果 D：结构性观察（弱信号，不足以翻案）

- **per-prompt argmax**：12 条 prompt 中 **0/12 单调递增**（排除了"纯粹顶 clamp"的退化情形），**5/12 有内部最优**，argmax 取到 5 个不同值 `{0.0, 0.003, 0.004, 0.005, 0.007}`。**但**——鉴于 §4.3 配对 Δ 全不显著，这些 argmax 基本是在**噪声里挑峰**，异质性大概率是伪信号。
- **per-bucket argmax**（内容异质性的初步观察，仅供 Exp-1.4 参考）：

  | 桶 | argmax scale |
  |---|---|
  | dense-texture | 0.002 |
  | flat-design | 0.007 |
  | photo | 0.007 |
  | face | 0.007 |
  | art | 0.005 |
  | text-render | 0.004 |

  桶间 argmax 确有差异，但每条曲线本身的波动都在种子噪声量级，**不能作为 content-adaptive 的证据**（需 Exp-1.4 专门实验 + 更多种子）。

![aggregate analysis](outputs/exp1.1_scale_sweep/pilot_12prompt_3seed_7scale/analysis.png)

*图 2（`aggregate.py` 自动产出）：左＝均值 IR vs scale（误差带＝种子 std，细线＝各 prompt）；右＝per-prompt argmax scale 直方图。左图各 prompt 曲线互相缠绕、无一致趋势；右图 argmax 分散在多个 scale 上——但结合 §4.3 配对不显著，这种"异质性"大概率是噪声。*

### 4.6 结果 E：动作在像素空间高度可见（RISK-9 佐证）

![action visibility](outputs/exp1.1_scale_sweep/pilot_12prompt_3seed_7scale/figs/action_visibility.png)

*图 3：左＝把各见证指标与 IR 都 min-max 归一化后叠加——5 个像素见证指标随 scale **平滑单调上升**，而 ImageReward（黑）在窄带内**锯齿抖动**、不跟随（归一化把它极小的波动拉满到 [0,1]，绝对幅度见右图）；右＝原始均值 IR ±SE **全部落在 ±种子噪声带（橙）内**，极差 0.111 < 种子噪声 0.241。*

各"见证指标"随 scale **单调上升**（与 IR 的平坦形成鲜明对比）：

| scale | colorfulness | sharpness | saturation | clipped_frac | contrast |
|---|---|---|---|---|---|
| 0.0000 | 45.1 | 1077 | 0.321 | 0.0031 | 51.4 |
| 0.0010 | 46.7 | 1215 | 0.334 | 0.0044 | 52.4 |
| 0.0020 | 48.2 | 1313 | 0.351 | 0.0070 | 53.6 |
| 0.0030 | 50.0 | 1411 | 0.370 | 0.0118 | 54.7 |
| 0.0040 | 52.4 | 1438 | 0.392 | 0.0206 | 56.0 |
| 0.0050 | 54.9 | 1497 | 0.417 | 0.0487 | 57.9 |
| 0.0070 | 61.7 | 1782 | 0.483 | **0.175** | 64.2 |

- guidance scale 是**饱和度/色彩/锐度/对比度的干净单调旋钮**：0→0.007 时 colorfulness +37%、sharpness +65%、saturation +50%，**clipped_fraction 暴涨 56×（0.003→0.175，高 scale 严重过曝/裁剪）**。
- **对照**：ImageReward 在**同一根轴**上却是平的/噪声的（§4.2–4.3）。**动作在像素空间又大又单调，却对 224px 的 ImageReward 不可见**——这正是 RISK-1 + RISK-9 的实证形态。
- **global-IR 与 patch-IR 的 argmax 不一致**：global-IR 峰在 scale=0.007，patch-IR（detail 敏感）峰在 scale=0.003。两者分歧 + 都很平，弱支持"224px reward 与 detail 脱钩"的判断。
- **within-prompt 去均值相关**：`corr(IR, saturation)=−0.045`、`corr(IR, colorfulness)=−0.142`、`corr(IR, sharpness)=−0.151`——**均为小负值**。即在单条 prompt 内，把 scale 顶高（饱和度↑）并**不会**带着 IR 上升，反而略降（因高 scale 进入过曝破坏区）。这说明"IR 天然偏好饱和 ⇒ 最省力的 hacking 就是顶饱和"这一担忧在**当前 scale 区间**并未直接兑现，但高 scale 的破坏性是明确的。

---

## 5. 由数据得出的判断

1. **Gate-1（SIGNAL）当前判定：未通过。** 配对 Δ 全不显著、均值极差 < 种子噪声 ⇒ ImageReward 在此样本上看不见 guidance 动作。
2. **Gate-3（ADAPTIVE）当前判定：未通过。** oracle gap +0.047 ± 0.100 落在噪声内，且尚未跨 ≥2 reward 验证（CLIP/HPS/aesthetic 未算）。
3. **两种活的解释，尚不能证伪任一：**
   - **(a) 真无红利**：最优 scale 的内容自适应收益本就 ≈0，扩展应在训练前止损（Occam kill-ladder）。
   - **(b) reward 盲 + 欠功率**：global-IR@224 看不见 guidance 主要在动的 detail/纹理轴（§4.6 + RISK-9），且 3 seeds 太吵；真信号被掩盖，需换 detail 敏感 reward + 加样本再判。
4. **RISK-1 / RISK-9 / RISK-14 三条高危风险，均由本轮数据从"假设"升级为"已观测"。**

---

## 6. 下一步（在进 Phase 2 之前必须做的）

按信息/算力比排序：

1. **扩样本收紧 CI**：12→~50 prompts、3→≥5 seeds，重算 oracle gap（当前 CI ±0.10 太宽，是"无信号"还是"欠功率"分不开）。
2. **补 ≥2 个去相关 reward**：先把 `score.py` 的 CLIP/HPSv2/aesthetic 跑起来（权重预置见 `prestage_weights.py`），并把 **patch-IR / 全分辨率锐度** 作为 detail 见证——直接检验解释 (b)。**预注册**：oracle gap 需在 ≥2 reward 上同时 > seed-CI 才算通过 Gate-3。
3. **Exp-1.2 CMA-ES 静态基线**：把 `power_calibrate / λ(attn_scaling)` 一并纳入搜索空间（§13.5），设定"必须打败"的标杆。
4. **Exp-1.5 SPSA 梯度核对**（若最终仍要 DRaFT）：~10 次 no-grad rollout，验证解析梯度可信、早步截断偏置可控。
5. **若扩样本后 oracle gap 仍在噪声内且跨 reward 一致为 0** ⇒ 按 `EXPERIMENT_PLAN.md` §13.9 **止损**：要么重构问题（换 detail reward / 换动作定义），要么放弃"动态自适应"主张、退回"自动化静态调度"这一更弱但更稳的贡献。

---

## 7. 复现命令

```bash
# 生成（Stage-1，12×3×7=252 张）——生成环境
python eval-pipeline/generate.py \
  --prompts eval-pipeline/prompts/eval_v1.csv \
  --out_dir outputs/exp1.1_scale_sweep/pilot_12prompt_3seed_7scale \
  --scales 0,0.001,0.002,0.003,0.004,0.005,0.007 --seeds 0,42,123

# 打分（ImageReward + patch-IR + 像素见证）——打分环境
<score_env>/bin/python eval-pipeline/score.py \
  --run_dir outputs/exp1.1_scale_sweep/pilot_12prompt_3seed_7scale --device cuda:0

# 聚合 + go/no-go 诊断（需 pandas/numpy/matplotlib）
python eval-pipeline/aggregate.py \
  --run_dir outputs/exp1.1_scale_sweep/pilot_12prompt_3seed_7scale --reward imagereward

# 可视化图 1/图 3（需 numpy + Pillow + matplotlib，如 promoe 环境）
python eval-pipeline/visualize.py \
  --run_dir outputs/exp1.1_scale_sweep/pilot_12prompt_3seed_7scale --seed 0
```

产物：`eval_results.csv`（252 行合并表）、`scores.jsonl`、`analysis.png`（图 2：均值曲线 + argmax 直方图）、`figs/scale_sweep_montage.png`（图 1）、`figs/action_visibility.png`（图 3）。

> 注：本仓库当前可用的 conda 环境为 `base1 / core_diff / promoe`（`repldm`/`sana_cby` 等文档旧名已不存在）；`aggregate.py` 依赖 pandas，本轮的诊断数值由等价的 numpy-only 脚本复算，结论一致。

---

## 8. 附录：新增评测指标与参考文献

`eval-pipeline/` 相对会议版 RepLDM（无评测管线）新增的指标分两类：**(8.1) 本工作新构造的指标**（`[RepLDM-unique]`，是本扩展的贡献之一）与 **(8.2) 集成的现成 reward 模型**（标准指标，本工作把它们正确接入并解耦）。表内 `[n]` 对应文末[参考文献](#参考文献)。

### 8.1 本工作新构造的指标 `[RepLDM-unique]`

| 指标（列名） | 定义（`eval-pipeline` 实现） | 为何新增 | 参考文献（作者，出处，年份） |
|---|---|---|---|
| **patch-IR**（`patch_ir_mean/std/n`） | 在**原生分辨率**取 中心+4角 共 **5 个 224² crop**，各过 ImageReward 后取均值/标准差（`imagereward_scorer.py:_native_crops`） | ImageReward 的 BLIP-ViT 主干硬编码 224px，1024→224 双三次下采样把 guidance 注入的**高频细节抹掉**（§13.2 / RISK-9）；patch-IR 让细节留在 reward 视野内，是 detail 敏感的 IR 变体 | ImageReward：**Xu et al., NeurIPS 2023** (arXiv:2304.05977) [1]；multi-crop 思想：**Krizhevsky et al., NeurIPS 2012** [6]、**Szegedy et al., CVPR 2015** [7] |
| **colorfulness** | Hasler–Süsstrunk M3：`√(σ_rg²+σ_yb²) + 0.3·√(μ_rg²+μ_yb²)`，`rg=R−G, yb=½(R+G)−B` | reward-hacking 见证：判定 IR 增益是否只是**色彩推高** | **Hasler & Süsstrunk, SPIE 2003**（Measuring Colorfulness in Natural Images）[8] |
| **laplacian_sharpness** | 亮度通道 3×3 拉普拉斯的**方差**（variance-of-Laplacian） | 原生分辨率**高频能量**代理；对细节敏感、恰被 224 下采样破坏（§13.2） | **Pech-Pacheco et al., ICPR 2000**（variance-of-Laplacian 对焦度量）[9] |
| **mean_saturation** | HSV **S 通道均值** ∈ [0,1] | 饱和轴见证（`δ_t=TFSA−z` 主要动的轴） | **Smith, SIGGRAPH 1978**（HSV 颜色空间）[10] |
| **clipped_fraction** | 任一通道 ≤2 或 ≥253 的像素比例 | **过曝/欠曝裁剪**见证（过饱和体检） | 标准动态范围裁剪诊断（highlight/shadow clipping，无单一经典文献） |
| **contrast_std** | 亮度标准差（RMS contrast） | 全局对比见证 | **Peli, JOSA A 1990**（Contrast in Complex Images，RMS contrast）[11] |

> **新意所在**：后 5 个像素见证指标本身是**经典底层图像统计量**；本工作的贡献是把它们组织成一组 **去相关的 reward-hacking 见证**（§13.5）——用于判定 IR/CLIP 家族 reward 的增益是**真实结构改善**还是仅仅**全局色彩/对比/锐度推高**（§4.6 正是靠这组见证坐实"动作在像素空间可见、对 IR 不可见"）。patch-IR 则把"多裁剪评测"这一旧思想用于绕开固定分辨率 reward 的 detail 盲区。

### 8.2 集成的现成 reward 模型

| 指标（列名） | 主干 | 说明 | 参考文献（作者，出处，年份） |
|---|---|---|---|
| `imagereward`（global） | BLIP ViT-L @224 | 人类偏好 reward；全图下采样到 224 ⇒ 偏色彩/布局、**非 detail** | **Xu et al., NeurIPS 2023** (arXiv:2304.05977) [1] |
| `clipscore` / `clip_cosine` | CLIP ViT-B/32 @224 | 规范 CLIP-Score `= 2.5·max(cos,0)`（另报原始 cos）；对齐见证，与饱和度相关，**非独立 detail 见证** | CLIP-Score：**Hessel et al., EMNLP 2021** [2]；CLIP：**Radford et al., ICML 2021** [3] |
| `hpsv2` | OpenCLIP ViT-H-14 @224（HPS v2.1） | 人类偏好分；与 CLIP 家族相关 | **Wu et al., 2023** (arXiv:2306.09341) [4]；OpenCLIP：**Radford et al., ICML 2021** [3] |
| `aesthetic` | CLIP ViT-L/14 @224 → MLP | LAION improved-aesthetic-predictor（SAC+LOGOS+AVA 训练），~[1,10] | LAION-Aesthetics：**Schuhmann et al., NeurIPS 2022** (arXiv:2210.08402) [5]；AVA 数据：**Murray et al., CVPR 2012** [12] |

> **§13.5 警示**：`clip / hpsv2 / aesthetic` 全是 **CLIP 家族 @224**，彼此高度相关，**不是**独立的 detail 见证；做 go/no-go 时必须配 **patch-IR + 全分辨率锐度**（§8.1）这类去相关见证，否则"IR 涨、第二指标也涨"会被误读成"干净的赢"。（本轮这三项**尚未打分**，见 §3。）

### 参考文献

1. J. Xu, X. Liu, Y. Wu, Y. Tong, Q. Li, M. Ding, J. Tang, Y. Dong. **ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation.** NeurIPS 2023. arXiv:2304.05977.
2. J. Hessel, A. Holtzman, M. Forbes, R. Le Bras, Y. Choi. **CLIPScore: A Reference-free Evaluation Metric for Image Captioning.** EMNLP 2021. arXiv:2104.08718.
3. A. Radford, J. W. Kim, C. Hallacy, et al. **Learning Transferable Visual Models From Natural Language Supervision (CLIP).** ICML 2021. arXiv:2103.00020.
4. X. Wu, Y. Hao, K. Sun, Y. Chen, F. Zhu, R. Zhao, H. Li. **Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis.** 2023. arXiv:2306.09341.
5. C. Schuhmann, R. Beaumont, R. Vencu, et al. **LAION-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models.** NeurIPS 2022 Datasets & Benchmarks. arXiv:2210.08402.（LAION-Aesthetics；预测器权重 `sac+logos+ava1-l14-linearMSE.pth`，见 github.com/christophschuhmann/improved-aesthetic-predictor）
6. A. Krizhevsky, I. Sutskever, G. E. Hinton. **ImageNet Classification with Deep Convolutional Neural Networks.** NeurIPS 2012.（multi-crop 测试评测）
7. C. Szegedy, W. Liu, Y. Jia, et al. **Going Deeper with Convolutions (GoogLeNet).** CVPR 2015. arXiv:1409.4842.（multi-crop / multi-scale 评测）
8. D. Hasler, S. Süsstrunk. **Measuring Colorfulness in Natural Images.** Proc. SPIE 5007 (Human Vision and Electronic Imaging VIII), 2003, pp. 87–95.
9. J. L. Pech-Pacheco, G. Cristóbal, J. Chamorro-Martínez, J. Fernández-Valdivia. **Diatom Autofocusing in Brightfield Microscopy: A Comparative Study.** ICPR 2000, vol. 3, pp. 314–317.（variance-of-Laplacian 清晰度度量的经典出处）
10. A. R. Smith. **Color Gamut Transform Pairs.** ACM SIGGRAPH Computer Graphics 12(3), 1978, pp. 12–19.（HSV 颜色空间）
11. E. Peli. **Contrast in Complex Images.** J. Opt. Soc. Am. A 7(10), 1990, pp. 2032–2040.（RMS contrast 定义）
12. N. Murray, L. Marchesotti, F. Perronnin. **AVA: A Large-Scale Database for Aesthetic Visual Analysis.** CVPR 2012.（LAION aesthetic 预测器训练数据之一）
