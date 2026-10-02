# P1.09 数据质量与可行性审计（data_qc）

- 记录时间：2026-09-23 00:34 Asia/Shanghai；执行：scripts/audit_p1_v1.py（最终运行 00:38:22，PROBLEMS=0）
- 范围：P1.02–P1.08 全部 curated 表（fold_switch_global 96 / fold_switch_regions 203 / knots 1401 /
  disorder 13983+4981 / pyp_pairs 6 / rnase_a_pairs 9 / bpti_pairs 9）+ 旧项目 constructs（只读）+ UniProt REST。

## 机器检查（13 项全过）

主键唯一性（各表）；(entry,region) 唯一性；disorder 区间边界 1≤s≤e≤len 全量；pdf 区间序；
target⟺mask 语义（fold_switch、knots 候选行、disorder NA&mask 互斥）；DisProt 字母全量校验完成（发现 13 条目含非标准字母：11×X、2×U、1×Z，qc.json 全列表；DP02886 的 Z 残基落入 mask=1 区段为唯一反例，P2.05 处理）；
同 UniProt 对重复 0。

开发期修正 2 处检查口径（regions 真键=entry+region；knots 阴性行 target=0/mask=1 合法）——
初版 FAIL 均为口径错误，非数据问题。

## 任务流失表（原始→证据合格→可映射→可评价）

| 任务 | 原始 | 证据合格 | 可映射 | 可评价 | 说明 |
|---|---|---|---|---|---|
| T-FS-GLOBAL/L1（strict 阳性） | 96 | 10 | 10 | **10** | pending 84 不入分母 |
| T-FS-REGION | 203 | 203 | 167 | **16 行（9 对，fine+full）** | usable 文献标签仅 3 对 |
| T-FS-L2-PU | 10 阳 | 10 | 10 | 10（+背景待 P1.10） | 三层治理变更 |
| T-FS-L3-MATCHED | 10 | 10 | 10 | **0**（待 P1.11/P2.01） | — |
| T-KNOT-PRESENCE | 1401 | 1401 | 1020 | **1020**（188 阳+832 阴） | 候选 380+review 层隔离 |
| T-KNOT-TYPE | 1401 | 112 | 112 | **112**（全 K 整链） | 多标签规则 OD 待决 |
| T-DISORDER-RES | 3337 | 3337 | 3319 | **5（双类蛋白）** | 有序 436 残基→降级请求 |
| T-PYP-STATE | 2 | 2 | 2 | **1 对**（案例级） | 延迟 pending_fulltext |
| T-RNASE-ASSEMBLY | 4 | 4 | 4 | **2 对**（案例级） | C-swap 残基 pending |
| T-BPTI-REDOX | 15 | 5 | 5 | **pending_fulltext**（ox/red 数值） | 功能终点、0 变体结构 |

## 独立单位（严格口径）

fold_switch：96 对覆盖 **103 个 UniProt**（同蛋白多对存在→必须组绑定）；strict 阳性=10 个 UniProt。
knots：1401 链（188 阳/832 阴/rejected 1）。DisProt：3337 条目（双类 5）。
受控系统：PYP/RNase A/BPTI 各=**1 个独立实验体系**（案例级结论边界，不得跨体系外推）。

## 排除与隔离清单（data/manifests/exclusions.tsv，7 行）

fold_switch excluded 2（historical_not_usable：porter_26/66）；knots rejected_knotted 1（5ljqA，
关系清理归 P2.01）+ excluded_artifact 1；rnase_a 假阳性排除 3（9R6Q/R/P）。
隔离（不入排除表、仅计数）：DisProt conditional_transition 区段 1007、other_structural_state 154、
disorder/order 重叠冲突条目 92；knots 各 review 层 380+。

## 混杂与不支持对照检查

- 阴性匹配只覆盖 K 链（S 滑结 349 无匹配）→ presence 任务若混 K+S 需分层（P2.01）。
- 717 vs 旧 v5 710 阴性差集（6 verified+1 rejected）→ P2.01 裁定。
- 受控系统链数与组装状态同义（RNase A）→ 输入充分性对照边界已标注。
- BPTI 功能终点+0 变体结构 → 表示来源决策请求（P4.05/P4.08）。
- PYP 晶体 vs 溶液 pH 6.5/8 差异已随行登记。

## 与 P0 协议对照（G1 输入）

- 三集合划分可行性：**总体独立组数不足以支持原 70:15:15 直拆**（strict 阳性仅 10 个 UniProt、
  残基任务 9 对、受控系统各 1 体系）——与用户 2026-09-21 决策一致（不预设比例；feasibility_gate 输入
  本报告 + split_protocol 3b）。
- DisProt 二分类不可评价（有序 436 残基）→ 降级请求（P1.05 提交，G1 裁决）。
- 统计参数/模型名单未冻结不影响本审计。
