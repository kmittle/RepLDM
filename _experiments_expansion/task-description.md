# 最终目的：把NeurIPS 2025的RepLDM扩展为TPAMI期刊版本

# 核心任务：使用 OPD、renderer-DPO 和 RL 训练小型 latent renderer，直到取得较为显著的性能改进

## 当前唯一研究主线（2026-09-01）

- OPD、renderer-DPO 和 RL 必须使用同一个 renderer、数据、prompt、seed、NFE、
  action space、reward 查询预算和评测协议，确保差异只来自训练方法。
- Attention Guidance 只作为会议版基线、teacher 候选和“latent 结构渲染”的思想联系，
  不再围绕动态 scale 或静态 guidance sweep 继续做增量改进。
- 首轮只验证一个参数量小于 1M、可审计、可严格 no-op 的 renderer；确认有效后再扩容。
- 每个固定 checkpoint 必须完整评测 HPSv2（3,200 张）和 GenEval（2,212 张），
  训练 reward 不能作为唯一主指标，也不能用 benchmark 子集替代正式结果。

1. 认真通读项目，理解会议版的 latent 结构引导思想；期刊版的核心方法改为使用
   OPD、renderer-DPO 和 RL 训练小型 latent renderer，而不是继续调 Attention Guidance。
2. 认真阅读'/mnt/miah204/bycao/RepLDM/doc/research/EXPERIMENT_PLAN.md'。
3. 查阅当前实验进展。
4. 调研最新Diffusion RL的进展，不断思考、设计新的模型，不能照搬别人的方法，要有创新性，贡献要足够会议扩展为期刊。
5. 调研并迭代 OPD、renderer-DPO 和 RL 训练 latent renderer 的方案，直到生成性能
   在完整定量评测和定性检查中得到显著提升。在完成任务前，不能停止。
6. 合理设计项目代码架构，使得项目结构清晰、明了
7. 每次完成代码、配置或实验协议改动后，都必须先做认真仔细的审查，至少通过 '$check 1'；不确定的实现应使用更重的检查。审查通过后先 push，并确认远端已包含被审查的提交，之后才能运行实验。测试、CPU 审计和环境探针可以作为审查或授权流程的一部分，但不得借此提前生成、训练、评分或读取实验结果。
8. 合理使用git和md进行版本管理，每版模型要在md中记录关键信息
9. 需要用到的数据集、模型权重自行准备；当前正式队列使用 4-7 号 GPU，启动前必须
   重新确认 GPU 空闲并遵守已注册的设备配置。
10. 本项目目标是做高分辨率生成，但是Diffusion RL强化生成性能和图像视觉体验可以先在base resolution上验证
11. 你需要从一个严苛审稿人的角度来设计算法和实验
12. 在'_experiments_expansion/rethinking'目录中：对所做过的实验进行分组和整理，每组实验写一个md报告进行总结反思。报告中应该提及这组实验室的motivation、参考文献、结果分析、反思以及下一组实验的规划。报告应该使用直白易懂的语言。

## 实验工程与版本管理硬约束

- 实验必须采用干净、清晰、可扩展的项目结构。数据读取、模型、训练目标、评测和
  启动入口应分离；公共逻辑只能有一个实现，实验差异通过版本化配置表达，禁止为
  每个设置复制整份脚本。
- 每个正式运行都必须保存解析后的完整配置、数据 manifest 及其哈希、代码 commit、
  环境信息、随机种子和输出清单，使结果能够从一个已 push 的 commit 重现。
- 每个可以独立验证的代码、配置或协议阶段都要及时提交。提交前必须测试并通过
  `$check 1`；提交后必须 push 并核对远端 commit，之后才允许启动对应实验。
- 正式实验不得从 dirty worktree 或仅存在于本地的 commit 启动。checkpoint 和大量
  生成结果不进入 Git，但其路径、哈希、状态和对应 commit 必须写入实验记录。
- 新设计先补最小单元测试、配置校验和 smoke test，再扩展到完整训练与全量 benchmark；
  不接受难以复用的单体脚本、隐藏参数或依赖人工记忆的运行步骤。

# Reminder

- 在你正式进入RL research前，我有一点想提醒你的：做RL设计的时候其实并不一定要完全follow住现有attention guidance的范式，attention guidance本质上是想做latent渲染，完全可以用RL训练一个比Diffusion基础模型小的latent神经网络来做这个事
- 当然，如果训练这个小渲染神经网络能与attention guidance具有同样的原理是最好的。
  最近的一些工作在强调生成过程中的结构性质的重要性，例如，我们会议版本中引用过的FreeU，利用UNet的天然结构对生成进行增强
  最近一些工作表明，在latent space中增强结构信息也是有益处的，例如：

  1. 'https://arxiv.org/pdf/2502.14831'：作者对 FluxAE、CogVideoX-AE 等 autoencoder 的 latent 做 DCT 频谱分析后发现：

        - RGB 图像通常以低频、整体结构信息为主；
        - 现有 AE latent 中却存在异常强的高频成分；
        - diffusion 从白噪声开始，天然倾向于先生成低频结构、再补充高频细节；
        - latent 过于高频，会破坏这种 coarse-to-fine 过程，使 diffusion 更难学习。
        作者通过微调 VAE/AE，抑制 latent 中异常强的高频成分，使 latent 的频谱更接近自然图像，从而让 diffusion 更容易按照 coarse-to-fine 的方式生成。

  2. 'https://arxiv.org/pdf/2502.09509'：通过对VAE encoder编码的图像做不变性正则来实现更鲁棒的latent space

> 我说这些的意思是，我们应该广泛调研，从不同工作中获得启发

> 另一方面，由于我们做的是会议版本的扩展，因此改进需要有跟会议版本的联系，但是这种联系不一定是框架、技术上的硬性联系，也可以是思想、底层原理方面的软联系

> 但是请你注意：很多时候论文的写作，其实带有主观成分，这在客观上本来就是难以避免的，有时候我们可以通过写作的手段来体现会议和期刊版本的关联。当然，我们期刊的目标是投稿TPAMI这样的顶级会议，因此最好要有够好的效果、足够的验证、严密的逻辑

> 可以在别的方案训练、生成、评估的时候分一个sub-agent接管实验进程，然后继续进行文献调研，多做几个假设，等上一个实验跑完无缝衔接下一个实验，确保实验队列中总是有实验可以跑，而不是让显卡空着

> 所有描述、报告务必使用直白易懂的语言

希望我说这些能给你带来启发

可以在'/mnt/miah204/bycao/RepLDM/_experiments_expansion/task-description.md'回顾本项目的目标

确保理解了所有要求，有不理解向我提问，完全理解就开始执行任务
