# 治理变更报告：折叠转换三层任务设计登记

- 记录时间：2026-09-22 22:00（Asia/Shanghai）；性质：**计划治理变更**，非科研任务，不计科研完成数。
- 指令来源：用户三层设计文档（原文归档 reports/plan/three_layer_design_user_directive_20260922.md）。
- 提交计划：独立审核通过后单独提交，标题 `PLAN register three-layer fold-switch workflow`。

## 变更内容（已落实到文件）

| 文件 | 变更 |
|---|---|
| logs/decisions.md | 新增 2026-09-22 条目：依据（阴性缺失与收录偏差风险）、影响范围、确认状态（用户已确认）、执行纪律 8 条 |
| configs/claims.yaml | T-FS-GLOBAL 保留为伞任务并指向三层；新增 T-FS-L1-PAIRED / T-FS-L2-PU-RANK / T-FS-L3-MATCHED；T-FS-REGION 登记为 L1 残基级任务；新增 three_layer_design 节（E1–E4 证据问题+四条纪律）；OD-NEGATIVE-FS → direction_approved__implementation_pending（P1.12+G1 后 resolved）；C-IN2 证据任务映射更新 |
| configs/sample_schema.yaml | samples 表新增 11 列：label_epistemic_status、biological_target(nullable)、pu_observed_label、positive_evidence_tier、negative_evidence_tier(PN-A/B/C)、background_source_version、literature_search_query/date、structure_count、independent_study_count、matched_set_id；conventions 新增 unlabeled_semantics（unlabeled 的 biological_target 恒空、pu_observed_label≠target） |
| configs/evaluation_protocol.yaml | 新增 4b three_layer_metrics：L1（层级可读性/条件贡献/区域定位，限阳性内部）；L2（recall@k、enrichment@k、rank percentile+family-stratified/leave-family-out enrichment；观测标签语义声明）；L3（AUROC/AUPRC/balanced accuracy/MCC/per-class PR；校准 disabled；PN-B/C 仅敏感性）；全 YAML 实测解析通过 |
| configs/split_protocol.yaml | 新增 3c three_layer_split_rules：L2 背景按组+家族整分+历史曝光不可洗白；L3 matched_set_id 不可跨集合+每集合报告病例/对照/家族/PN 构成；样本不足降级不拆组 |
| TODO.md | 头部快照注记；进度行更新；Phase 1 树新增 P1.10/P1.11/P1.12（未开始，不标完成）；G1 里程碑条件扩为含三层合同冻结确认；P1.09 详情扩充 strict positive 专项审计（逐对审计/porter_8 处置/8 个旧审计抽查/evidence_tier/独立计数/可行性数字+3 份交付物）；新增 P1.10/P1.11/P1.12 完整任务详情（各含 5 步执行方法与验收条件）；P2.01（匹配工具链+7 步+4 交付物）、P2.02（三层划分清单）、P2.06（偏差基线三组+关键判据）、P3.05（按层分列输出）、P5.06（E1–E4 证据矩阵）各增"三层补充"块 |
| README.md | 任务数 46→49 并注明治理变更 |

## 未改动项（纪律）

- P1.02/P1.03 完成状态与历史记录保留，不重开、不改写。
- 不把 P1.10–P1.12 标为已开始/已完成；其执行在 P1.09 之后串行。
- 不下载 PLM 权重（P2.04 前）；不生成正式划分（G1/P2.01/P2.02 前）；$B 保持只读。
- 背景蛋白 biological_target 恒空；无文献命中≠生物学阴性；病例-对照比例≠发生率；不因样本不足拆同源/匹配/配对组。

## 后续顺序

PLAN 提交后回到科研线：P1.05（DisProt）→P1.06–P1.08 可先行；P1.09（扩充版审计）→P1.10→P1.11→P1.12 串行；P1.12+G1 用户确认后进入 Phase 2 三层匹配与划分。
