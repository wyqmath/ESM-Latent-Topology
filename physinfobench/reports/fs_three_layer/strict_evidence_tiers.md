# strict 证据强度分列（P1.20；G1 材料修订）

时间：2026-09-23 21:38–21:49 Asia/Shanghai（含 21:46–21:48 审核整改重跑）。指令来源：用户 2026-09-23 21:05 第 2 项（序列身份复核通过与双态原文证据强度分列；不得用"10/10 双态原文均已独立确认"概括）。产物：strict_evidence_tiers.tsv（10×13）+ qc；脚本 scripts/build_strict_evidence_tiers.py（断言：同序列双态=仅 porter_87、10/10 序列身份确认、分层枚举完备、isoform 三态分布 7/1/2）。

## 1. 两个维度分列（结果）

**维度一：序列身份（P1.13，20 端点）**——10/10 `same_protein_confirmed`（对齐区间零实质错配；porter_61 两端各 1 列 M1L 工艺错配）。**两端观测序列完全相同：仅 porter_87**（两端=canonical 同一 74 aa 区段，Q58AD3 N 侧翼 20）；其余 9 对观测序列不同，差异全部为构建体级（成熟链/结构域边界、内部缺段、覆盖差）。

**维度二：双态原文证据（P1.14 16 端点+round2 通道 2 对）**——分层：

| 层 | 对数 | 对 | 旧审计依赖 |
|---|---|---|---|
| fulltext_both（两端全文自核） | 2 | porter_20、porter_62（62 的 4o0p 端 kw 命中但全文未见 PDB 码） | 无 |
| fulltext_one_abstract_one | 1 | porter_68（4tsy 端全文；3zwg 端摘要） | 部分（1 端） |
| abstract_both（两端摘要级） | 5 | porter_51、61、72、77、87 | 部分（状态语义佐证仍依赖旧审计指针+结构标题；摘要探针为本项目自核） |
| historical_fulltext_old_audit | 2 | porter_9、porter_80（不在 P1.14 表内；证据走 P1.02 导入的 round2 行=旧项目 trial10 时代全文审计，A+B 双端覆盖） | 完全（本项目未自核原文） |

- 摘要级且无状态关键词（kw=0）端点：strict 范围 **5 个**（porter_51 的 1qln 端、porter_68 的 3zwg 端、porter_72 两端、porter_77 的 1jfk 端）；P1.14 报告的"6 个"为 18 行全口径（另 1 个=porter_8 的 3m1b 端，extension 不在本表范围）。
- isoform：porter_9 canonical 严格更优；porter_20/80 与最佳 isoform identity 并列（归属沿用 SIFTS，歧义登记）；其余 7 对无 isoform 注记。

## 2. 统一定位

10 对全部=**evidence_tiered_candidate**（有证据分层的候选）：
- "确认"只用于维度一（序列身份 10/10）；维度二只能按层引用（2/1/5/2 分布）。
- 禁止合并表述："10/10 双态原文均已独立确认"不成立（全文自核仅 2 对；2 对完全依赖旧审计）。

## 3. L1 混杂检查与结论限制（逐对，见 TSV 列 l1_confound_items/conclusion_limit）

- porter_87（唯一）：同序列双态——状态差异不可归因序列差异；其余结论对不受此保护。
- 其余 9 对：观测序列/覆盖不同（构建体差异明细见 construct_difference_detail：N/C 侧翼最大 493/668 aa、内部缺段 1-5 块（canonical 侧 3-25 aa；TSV 中缺段 aa 按观测侧渲染常为 0，缺残基在 canonical 侧，见 P1.13 internal_gap_canon_res）、覆盖最低 0.9484）→ L1 主分析必须把构建体差异登记为混杂协变量；状态间差异的解释限定为"同蛋白双态、构建体差异已解释但未被实验平衡"。
- porter_20/80（isoform 并列）+porter_9（canonical 更优）：isoform 竞争随行登记，不改变归属。

## 4. G1 修订差异

- 重点建议 1 与第 1 项改写：两维度分列、分层计数、porter_87 唯一同序列、9 对构建体差异入 L1 混杂清单；"不允许"新增两条合并表述禁令。
- 附录 A1/A2 措辞同步（A1 只承担维度一；A2 补分层计数与 porter_9/80 旧审计层）。
