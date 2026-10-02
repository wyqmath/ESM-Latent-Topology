# P1.09 两处旧产物缺陷的处置方案（P1.23；G1 材料修订收口）

时间：2026-09-23 23:16–23:24 Asia/Shanghai。指令来源：用户 2026-09-23 21:05 第 5 项（复查两处缺陷原始字节证据；给出修订任务与重跑影响清单；不静默改写原表）。**本文件只做方案，不改任何原表**；修订任务（P1.24/P1.25）已登记 TODO，触发条件=G1 决议后、P2.01 前。

## 缺陷 1：construct_difference_class 语义失实

**字节证据（2026-09-23 复查）**
- 载体 A：reports/fs_three_layer/strict_positive_audit.tsv 第 19 列，10/10 行=`exactly_identical`。
- 载体 B：data/curated/fold_switch_global.tsv 第 19 列（第 18 列为 same_uniprot），strict 10 行=`exactly_identical`；**全池 31 对同标**（另 explainable 34/local_fragment 18/still_uncertain 13）——缺陷覆盖面大于 G1_supplement 原登记（原只记 audit 表）。
- 反证：P1.13/P1.20——两端观测序列完全相同仅 porter_87（两端=canonical 同一 74 aa 区段）；其余 9 对 obs 长度/侧翼/缺段均不同（如 porter_20 317 vs 311 aa、porter_72 141 vs 155 aa）。

**根因**：字段值承自旧项目资格表（import_fold_switch_v1.py 透传），旧语义=「构建体对齐区间全局 identity=1.0」（旧 trial10 即以 317/311 aa 标 exactly_identical），与「观测序列相同」混同——不是计算错误而是语义标签错位，但在 strict 语境下按字面读即失实。

**修订任务 P1.24（G1 后执行）**
- 动作：import_fold_switch_v1.py 源头修正——字段更名 `construct_global_identity_class`（保留旧值，语义注明），新增 `observed_sequence_identity_class`（10 strict 按 P1.20 tiers 填：porter_87=identical_observed；其余 9=different_observed_construct_explained；非 strict 21 个旧标 exactly_identical 对留 pending_not_reverified，不静默改判）。
- 逐项差异：fold_switch_global.tsv（列更名+新列+96 行值）；strict_positive_audit.tsv 同步列更名与 10 行新值；两表 provenance 注记。
- 重跑清单：import_fold_switch_v1.py（fold_switch_global 重建+断言）、audit_p1_v1.py（strict 审计重建）、P1.15 audit_region_labels_joint.py 复跑一致性检查（其输入含 global 表）、G1_supplement 缺陷登记条目更新为已处置。
- 消费方核查（本轮已扫）：P1.12 feasibility_matrix/P1.15 结论/G1 正文未按字面消费该字段（已改用 P1.13/P1.20 表述）；无需重跑 P1.12。
- 审核点：新列与 sequence_identity_review.tsv/strict_evidence_tiers.tsv 逐行一致断言；旧值保留可追溯。

## 缺陷 2："96 对全部在旧 v5 曝光"失实

**字节证据（2026-09-23 复查）**
- 载体 A：reports/fs_three_layer/strict_positive_feasibility.md L32–33「历史曝光：96 对全部在旧 v5 development/test 中有分配（旧表在册）」。
- 载体 B：reports/tasks/P1.09.md L32「历史曝光=96 对全部在旧 v5 有分配」。
- 反证：P1.17 实测 v5=35 对（26 dev-only/7 test-only/2 both）；P1.19 四口径（v1–v5 并集亦=35；legacy 登记=96/96 yes 系另一层；正式评价使用=0）。
- 无其他载体：data_qc.md/strict_positive_qc.json/feasibility_matrix.tsv 无该表述（本轮 grep 证实）。

**根因**：P1.09 把"96 对都在旧项目在册"直接写成"全部在 v5 有分配"，未实测 v5 成员表（P1.09 时点未读旧划分表）。

**修订任务 P1.25（G1 后执行）**
- 动作：两处原文替换为 P1.19 四口径表述（35 v5 收录/并集 35/legacy 96 yes·89 端点可回指/评价使用 0），并加"2026-09-23 P1.19 修正"注记；strict_positive_feasibility.md 如由脚本生成则改脚本重生成（P1.24 执行时一并溯源）。
- 重跑清单：无数据路径重跑（纯文档修正+注记）；G1_supplement 缺陷登记条目更新为已处置。
- 消费方核查：G1 第 5/9 项、附录 A5、P1.17/P1.19 报告均已按修正口径书写（本轮确认）；无残留失实消费点（reports/ 内"96 对全部"仅剩缺陷登记与修正说明语境）。

## 两任务的共同纪律
- 均按"差异清单→独立审核→单独提交"执行；旧版本经 git 历史保留；不在本文件内提前改表。
- 若 G1 对 legacy 曝光口径的裁决改变表述（如判定 legacy 层计为曝光），P1.25 的替换文本随之采用裁决后口径。
