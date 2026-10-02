# G1 补证附录（P1.13–P1.17 产物的索引与关键数字汇总）

生成：2026-09-23 20:41 Asia/Shanghai（P1.18 第二轮修正）；2026-09-23 21:22 P1.19 修订（A5 改写+新增 A6 行）；2026-09-23 21:44 P1.20 修订（A1/A2 两维度分列）；2026-09-23 21:58 P1.21 修订（新增 A7 行）。各附录的完整方法/证据/审核链见对应任务报告与 logs/reviews/P1.1x.md。

## 附录索引

| 附录 | 主题 | 任务报告 | 关键产物 |
|---|---|---|---|
| A1 | strict 序列身份复核 | reports/tasks/P1.13.md | sequence_identity_review.tsv（20×21） |
| A2 | strict 双态原文证据 | reports/tasks/P1.14.md | dual_state_evidence_review.tsv（18×14） |
| A3 | 残基标签联合审计 | reports/tasks/P1.15.md | region_label_joint_audit.tsv |
| A4 | L3 对照证据+检索快照 | reports/tasks/P1.16.md | pna_evidence_review.tsv + epmc_snapshot_lock.md |
| A5 | 划分可行性只读预演 | reports/tasks/P1.17.md | split_rehearsal.md/_groups.tsv |
| A6 | 历史曝光范围审计（G1 修订轮） | reports/tasks/P1.19.md | exposure_scope_audit.tsv/_qc.json |
| A7 | 背景版本跨源连接核查（G1 修订轮） | reports/tasks/P1.21.md | background_version_join_audit.tsv/_qc.json |

## 关键数字速览（全部可由上列产物复算）

1. **序列身份（A1，维度一）**：10/10 same_protein_confirmed；对齐区间零实质错配（porter_61 仅 M→L 工艺）；**同序列双态仅 porter_87**（其余 9 对构建体级差异→L1 混杂清单，见 strict_evidence_tiers.tsv）；3/10 accession 有 isoform（porter_9 canonical 严格更优；porter_20/80 并列→归属沿用 SIFTS）。
2. **双态原文（A2，维度二）**：分层=fulltext_both 2（porter_20/62）+半 1（porter_68）+abstract_both 5（porter_51/61/72/77/87）+**historical_fulltext_old_audit 2（porter_9/80：round2 旧项目全文审计导入，本项目未自核）**；kw=0 端点 strict 范围 5 个（P1.14 报告"6 个"为 18 行全口径含 porter_8）；porter_8（extension）建议 extension_condition_or_assembly 不升格。
3. **残基标签（A3）**：双满足=3 对 6 行（porter_20/61/62）→主分析仅案例级；fine_only 6 对=敏感性；不得宣称 9 对可靠标签。
4. **L3 对照（A4）**：6 个有家族候选的阳性核查后各保有 ≥2 可用 PN-A 主选（Q12931=名义 1:3 零冗余）；2 个标记唯一候选（P02829/P0DP29）待人工全文复核；回补 5 complete+2 partial 后冻结；快照锁定方案二选一（推荐 A=冻结当前）。
5. **预演（A5）+曝光范围审计（P1.19）**：96 对→93 组（最大 2）；**v5 划分收录=35/96（非 96/96）**，v1–v5 全版本并集亦=35（61 对从未进任何划分表）；**前项目 legacy 登记=96/96 yes**（91_ESM-Latent-Topology_On-Hold SaProt 控制清单；89 对端点级可回指、7 对旧包内部不一致 porter_6/16/37/71/78/86/89）；正式评价使用=0（formal_model_evaluation=false）；10 strict 全部 v5 收录且全部 legacy=yes，references 证据深挖恰好覆盖 10 strict；**暂无可靠备用确认池**（A 路径=53 pending 对不在任何旧划分表但需先补证+legacy 口径用户裁决）。逐对清单=exposure_scope_audit.tsv。

## 补证揭示的既有产物缺陷（G1 后处置：缺陷 1 已由 P1.24 落实、缺陷 2 已由 P1.25 处置（均 2026-09-24））

- ~~P1.09 strict_positive_audit.tsv 的 construct_difference_class 字段 10/10 exactly_identical 与 9 对非同一观测序列矛盾~~ **已处置（P1.24，2026-09-24）**：fold_switch_global 第 19 列更名 construct_global_identity_class（旧值保留），新增 observed_sequence_identity_class（strict 10 按 P1.20 填值）；strict_positive_audit.tsv 同步；porter_8 表内分层随同落实（G1 限定 2）。详见 reports/tasks/P1.24.md。
- ~~P1.09/旧 G1 材料的"96 对全部在旧 v5 曝光"与 v5 实测（35 对）不符~~ **已处置（P1.25，2026-09-24）**：strict_positive_feasibility.md 与 reports/tasks/P1.09.md 两处原文已改为 P1.19 四口径并注记；feasibility_matrix.tsv 两行"历史曝光"同步改为"历史接触（S1–S5 分记）"（G1 限定 4 采纳）。详见 reports/tasks/P1.25.md。
