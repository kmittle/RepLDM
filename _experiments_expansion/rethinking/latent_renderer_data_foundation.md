# Latent Renderer 数据基础审计

## 为什么先做这个工作

OPD、DPO 和 RL 都会反复读取 prompt、参考图或 reward 数据。如果数据来源不清楚、
混入 benchmark prompt，或者训练中途图片被替换，三种方法的比较就不可信。因此在
训练 renderer 前，先把用户指定的五个目录整理成同一份可检查的候选目录。

本阶段不是性能实验，也没有读取旧实验分数。目标是回答三个简单问题：哪些数据能
训练，哪些只能登记，哪些评测内容必须隔离。

## 参考资料

- 4KLSDB、Aesthetic-4K、Style30K 和 PixVerve 的本地 README 与数据卡。
- Unsplash Full Dataset 的本地 `TERMS.md`。
- Sana 中 PickScore、OCR、GenEval 和 DrawBench 的本地划分文件。
- HPSv2、GenEval、MJHQ、PixVerve-Bench、Aesthetic-4K eval 和 RepLDM 的冻结评测
  prompt。

## 观察

1. Pixel-Render 的总 manifest 有 5,058,099 行，但只有 3,287,110 个文件存在，
   1,770,989 个文件缺失。这个 manifest 没有图片内容哈希。
2. PickScore 和 OCR 只有当前文本文件，没有能覆盖数据内容的专属许可证据。因此
   “文件在 Sana 仓库里”不能推出“允许训练”。
3. Unsplash 条款只允许带严格限制的非商业机器学习用途，也要求派生模型保持非商业。
   它不能作为普通的无条件训练数据。
4. Style30K 的五个压缩分卷有 Git-LFS SHA-256 OID，但当前解压后的 25,868 张图片
   没有逐文件哈希。OID 不能直接证明解压目录没有被改动。
5. Sana 的 8 对 feature/mask shard 共约 179 GB。尺寸和 129,484 个 ID 的顺序能
   对上，但没有 shard 内容哈希，等长替换无法被发现。
6. Aesthetic-4K 的两个 eval sidecar 过去只被登记，没有进入全局 prompt 防火墙。
7. 第一版保护索引还漏掉本地 4KLSDB validation/test。补入 metadata 和 x4 HR
   分片后，3,984 个 metadata ID 与 3,984 张图片一一对应；`caption` 和所有非空
   `cogvlm_caption` 都进入文本防火墙。

## 结果

候选集固定为 313,094 行：129,484 个 4KLSDB 图文对、95,735 个 PixVerve 源行中有
95,733 行通过训练资格检查（另外两行与 benchmark prompt 精确重合）、
50,000 个 GenEval 训练 prompt、12,009 张 Aesthetic-4K 训练图和 25,868 张
Style30K 图片。其中 275,217 行含 prompt，263,094 行引用图片。

PickScore、OCR、Unsplash 和 Pixel-Render 大 manifest 全部不能进入候选训练集。
49,393 条 benchmark 记录进入统一 holdout，其中有 46,619 条规范化唯一文本、
39,712 条带图记录和 37,160 张唯一图片。Aesthetic-4K 的两个 eval split 与本地
4KLSDB validation/test 都加入精确文本和图片防火墙。

目录验证改为 fail-closed：配置固定每个来源的条数和哈希、每个 artifact 的统计与
顺序；验证时重新生成来源记录，并逐字节检查训练候选集。当前图片 payload 没有完整
checksum，所以该目录明确标记为 `training_ready: false`。

扩展保护集后的旧开发 release 保留在
`DATA/dev-catalog-protected-20260901/`，但它早于当前 source-containment 检查，不能
再当作最新证据。每次 catalog builder 或配置改变后，都必须重新生成一个带日期的新
开发目录并重新做完整复验。`--require-training-ready` 会在 selected-payload manifest
尚未建立时拒绝任何这类 metadata-only release。

## 结论与反思

之前的 358,173 行数字把许可待审的 PickScore/OCR prompt 算进了训练数据，这是不
合格的实验基础。只检查路径存在也不能保证训练可复现。更严格后，样本少了 45,077
行，但 OPD、DPO、RL 的比较边界更清楚，也不会因为数据许可或 benchmark 泄漏导致
整组结果作废。

## 下一步

先从 313,094 行候选目录中冻结首轮共享数据视图，对实际选中的每张图片生成 SHA-256
manifest。随后冻结同一个 sub-1M renderer、同一批 prompt、同一 NFE 和 reward 查询
预算，再依次实现 OPD、DPO 和 RL。任何训练入口都必须先验证 selected-payload
manifest；未通过时直接退出。
