# 折叠转换三层设计可行性（strict_positive_feasibility）

- 记录时间：2026-09-23 00:24 Asia/Shanghai；数据源：strict_positive_audit.tsv（10 对逐对审计）+
  strict_positive_qc.json + fold_switch_global/regions。

## 1. 严格阳性证据分层（positive_evidence_tier=strict，10 对）

| 指标 | 数值 | 说明 |
|---|---|---|
| same UniProt（SIFTS） | **10/10** | 全部同蛋白 |
| SIFTS mapping_identity 最小值 | 见 audit 表 | 全对 0/0 突变（P1.02） |
| 证据可追溯（primary_full_text* 审计） | **10/10** | 其中 8 对依赖旧项目原文审计（无 round2 扫描行） |
| 观测序列逐位一致 | 1/10 | **非突变**：差异为构建体成熟边界/编号错位（从第 1 位起全不同+长度差 1–70） |
| 双端为 UniProt canonical 子串 | **3/10** | 另 2 对仅 A 端、5 对无子串关系；非子串端疑为标签/isoform/边界，需局部对齐甄别（G1 前补做） |
| 有 fine 区段 | 9/10 | porter_68 仅 coarse |
| 有 usable 文献区段标签 | 3/10 | 残基任务受双口径约束 |

porter_8 冲突：corrections 建议 extension_condition_or_assembly，资格表仍 pending_evidence——
该对非 strict，不影响本表；冲突留痕待 G1 裁决。

## 2. 独立计数

strict 阳性=**10 个 UniProt**（每个 UniProt 恰 1 对）；全池 96 对=103 UniProt。
家族候选数待 P2.01（旧 foldseek/SCOPe/CATH 注释可复用，先核验）。

## 3. 三层可行性初判

- **L1（配对机制诊断）**：可行——10 对、残基级 fine 9 对/usable 3 对；结论限于阳性内部。
- **L2（PU 排序）**：可行——阳 10 + 背景（P1.10 建）；独立组=10+背景组。
- **L3（匹配病例-对照）**：**待建**——PN 候选池（P1.11）与匹配（P2.01）完成后重估；
  10 个病例对 1:1/1:3/1:5 匹配的家族/长度/结构覆盖约束可行性在 P2.01 输出。
- **历史接触（原文有误，2026-09-24 P1.25 修正）**：本报告原写"96 对全部在旧 v5 development/test
  中有分配"——实测失实（P1.17 发现）。修正为 P1.19 四口径：S1 旧 v5 划分收录=35/96（26 dev-only/
  7 test-only/2 both）；S2 v1–v5 并集=35；S3 前项目 legacy 登记=96/96 yes（G1 2026-09-24 裁决：
  legacy 登记+旧项目数据构建使用一并视为历史接触记录）；S5 正式评价使用=0。96 对不可重新包装为
  未见；P2.02 以历史接触记录为输入；53 pending 对不构成可靠未曝光备用确认池。

## 4. 待 G1 决策清单（由本审计升级）

1. 7 对（非双端子串）的序列甄别（细分：5 无子串+2 仅 A 端；标签/isoform/边界）——建议 G1 前完成局部对齐复核。
2. porter_8 分层改判采纳与否。
3. 18 对无 round2 扫描行的补扫描（其中 8 个 strict 依赖旧审计）。
4. usable(3 对)/fine(9 对) 双口径联合定义残基任务样本。
5. 备用确认池来源方案（split_protocol 3b 候选 A/B/C）。
6. 家族层隔离可行性（能否支撑"未见家族"）。

## 5. 局限

- 本审计未查看任何模型结果（P1.09 验收：审计仅用于数据有效性）。
- 观测序列对齐为子串级初判；完整对齐（标签剥离+isoform 核对）留 G1 前补做。
