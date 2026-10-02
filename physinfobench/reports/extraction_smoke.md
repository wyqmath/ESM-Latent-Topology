# 表示提取冒烟验证（P2.05；extraction_smoke）

时间：2026-09-24 04:0x Asia/Shanghai。模型=ESM-2 650M（P2.04 验证版；snapshot 08e4846e…，加载参数 add_pooling_layer=False）。

## 样本（全部 development 集）

| 名称 | 长度 | 覆盖点 |
|---|---|---|
| porter_87 A（2k0qA） | 74 | 最短 strict 端点 |
| porter_77 A（1jfkA） | 134 | 最短段 |
| porter_72 A（4rmbA） | 141 | 结构域片段 |
| porter_20 A（5c1vA） | 317 | 常规 |
| porter_62 D（4o01D） | 482 | 常规 |
| porter_51 D（1h38D） | 857 | 最长 |
| SYNTH_trunc1500（合成） | 1500→1022 | 截断路径（非科研样本，仅代码验证） |

## 检验结果

1. **缓存键与命中**：key=sha24(model_revision+seq_sha256+layer_set+prep_hash)；首跑 0 命中/7 miss → 二跑 7/7 hit。**元数据失配拒绝已实测**（审核后补做受控实验）：篡改一个 npz 的 meta.seq_sha256 → 运行打印 "[cache REJECT meta mismatch]"，该样本转 miss 重提取（manifest 出现 reject+miss 双行，其余 6 样本 hit）——拒绝语义=不复用失配缓存并重生成。
2. **token↔残基对齐**：BOS/EOS 剥离后 token 数=残基数（P2.04 已证 token_equivalence；提取器 resid_last 形状断言=L_used）。
3. **截断不静默**：>1022 → 取前 1022，trunc_coverage=0.6813 写入 meta（真实 FS 端点均 ≤1022，合成样本仅验证代码路径；正式提取遇到截断样本将在 manifest 报告覆盖率）。
4. **多层导出**：mean_layers=34×1280（逐层去 BOS/EOS 后均值）；resid_last=末层逐残基 L×1280。
5. **发现与修复（本轮 P2.05 的实质产出）**：AutoModel 加载时 checkpoint 无 pooler 头 → transformers 随机初始化 pooler.dense（警告 MISSING）——验证其**不影响 hidden_states**（两次实例化 hidden 逐元素相等、池化指纹前后一致 9285fbc33037a0d8），但按锁定权重纪律改为 `add_pooling_layer=False` 消除随机头，P2.04 脚本同步修复并复跑（指纹不变）。
6. **资源**：74aa 单样本前向 ~0.4s / 857aa ~2.9s（M 系列本地 CPU）；缓存 npz 大小 0.4–6MB/样本。

## 残留

- 正式全量提取（P3 前执行）：按本脚本与缓存键；截断样本清单届时报告。
- SaProt 结构 token 通道未验证（models.yaml pending）。
