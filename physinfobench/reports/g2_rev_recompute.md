# G2-REV 复算报告（G2 有条件确认核对修订）

时间：2026-09-24 16:30。输入=冻结划分 split_manifest.tsv（4,858 行）+ split_qc.json + knots.tsv + fold_switch_global.tsv + fs_l2_background_manifest.tsv.gz + p201_step1_structural_edges.tsv。复算脚本=scripts/g2_rev_recompute.py（断言全过才产出）。结论：G2_milestone.md §3 的 knots/FS/disorder/受控三系统行原数字沿用 P2.02 初版划分（commit cd768cd 版 split_qc.json），未随 P2.03 审计驱动重划分更新；L2 背景行原值即现行值（并非沿自初版，见 §4）。现行修正值如下。

## 1. 各任务三集合数量（行/样本；修正对照）

| 任务 | G2 材料原值（初版划分） | 现行复算值（dev/conf/hold） | 总量对账 |
|---|---|---|---|
| knots presence | 1,015/170/216 | 970/247/184 | 1,401=P1.09 ✔ |
| FS（D6） | 78/4/14 | 81/7/8 | 96 ✔ |
| disorder | 2,366/498/473 | 2380/447/510 | 3,337 ✔ |
| bpti/pyp/rnase_a | 9/-/- 各全 dev | 9/6/9（全 dev） | 9/6/9 ✔ |

## 2. 打结类型 × 三集合（112 行；修正对照）

| 集合 | 类型构成 | 合计 |
|---|---|---|
| development | 3_1=75、4_1=7、5_2=2、5_1=1 | 85 |
| confirmation | 3_1=14（**仅 3_1**，其余类型在确认集不可评价） | 14 |
| final_holdout | 3_1=11、4_1=1、**5_2=1** | 13 |

修正要点：原值 80/14/13 不构成任一单一划分版本的一致快照（初版划分=80/14/18，现行=85/14/13；dev/conf 与初版一致而 holdout 与现行一致），且『5_1 仅 dev』注记遗漏现行 final_holdout 中的 5_2×1（用户指出处）。现行口径：5_1 仅 dev 1 行（conf/hold NA 如实）；5_2 在 dev 2 行+hold 1 行、confirmation 无（确认集该类 NA）；4_1 三集合皆有；confirmation 打结类型任务实际只可评价 3_1。

## 3. strict 10 对归属（复算）

10/10 全部 development（porter_9_3j7wB__3j7vG, porter_20_5c1vA__5c1vB, porter_51_1h38D__1qlnA, porter_61_4gqcC__4gqcB, porter_62_4o0pA__4o01D, porter_68_3zwgN__4tsyD, porter_72_4rmbA__4rmbB, porter_77_2nxqB__1jfkA, porter_80_5f3kA__5f5rB, porter_87_2k0qA__2lelA）。G2 材料『strict 10 全 dev』在现行划分下成立。

## 4. L2 背景三集合（蛋白数口径，与 P2.02 报告更正一致）

- 非 EP_ 蛋白：dev 343,682 / conf 68,654 / hold 71,850。G2 材料此行原值（343,682 dev）即现行值，复算维持——该行并非沿自初版划分（初版背景分布=335,662/73,349/75,175，EP_ 行 156/8/28，git show cd768cd 可复算）。
- EP_ 端点行：dev 162 / conf 14 / hold 16（不入背景蛋白计数）。

## 5. max 口径跨集合边全表（31 条；用户 G2 裁决第 1 项）

全表=reports/g2_rev_maxrule_edges.tsv（31 行，按 tm_sym_max 降序）。跨集合类别：confirmation-development=8、development-final_holdout=23。涉及唯一样本（对）22 个：

| pair | tier | 集合 | 组 |
|---|---|---|---|
| porter_10_2lqwA__2bzyB | pending_evidence | final_holdout | G4301 |
| porter_19_2naoF__1iytA | pending_evidence | confirmation | G095 |
| porter_1_1g2cF__5c6bF | fragment_only | development | G068 |
| porter_21_4zt0C__4cmqB | pending_evidence | final_holdout | G4306 |
| porter_22_5jytA__2qkeE | pending_evidence | final_holdout | G4307 |
| porter_25_5k5gA__2kb8A | pending_evidence | development | G127 |
| porter_33_2ougC__2lclA | pending_evidence | development | G139 |
| porter_35_4rr2D__3l9qB | fragment_only | final_holdout | G4313 |
| porter_38_4y0mJ__4xwsD | extension_construct_difference | development | G4315 |
| porter_52_5b3zA__5bmyA | pending_evidence | confirmation | G225 |
| porter_56_4twaA__4ydqB | pending_evidence | final_holdout | G4325 |
| porter_59_1xntA__3lqcA | extension_construct_difference | development | G4327 |
| porter_61_4gqcC__4gqcB | strict_state_candidate | development | G4330 |
| porter_62_4o0pA__4o01D | strict_state_candidate | development | G058 |
| porter_63_4dxtA__4dxrA | pending_evidence | final_holdout | G4331 |
| porter_73_2ce7C__3kdsG | pending_evidence | confirmation | G4338 |
| porter_78_5l35D__5l35G | pending_evidence | final_holdout | G126 |
| porter_7_3gmhL__2vfxL | pending_evidence | final_holdout | G172 |
| porter_81_4qdsA__2qqjA | fragment_only | development | G4344 |
| porter_86_2a73B__3l5nB | extension_construct_difference | development | G4349 |
| porter_8_3m1bF__3lowA | extension_condition_or_assembly | confirmation | G4353 |
| porter_9_3j7wB__3j7vG | strict_state_candidate | development | G4357 |

涉及 strict 阳性对：3 个（porter_61_4gqcC__4gqcB、porter_62_4o0pA__4o01D、porter_9_3j7wB__3j7vG）。

### 影响面（按任务）

- **L1 配对机制诊断**：10 对 strict 全 dev、分析宇宙在 dev 内——dev 内部结论不受影响；涉及 strict 的 max 边意味着对这些相似跨集合亲属不得做跨集合外推主张。

- **L2 PU 排序**：排序宇宙=L2 development（343,682 背景+10 strict）——跨集合边另一端在宇宙外，对 dev 排序点估计无影响；将来若把 L2 扩到 conf/hold 需先做敏感性。

- **L3 匹配病例-对照**：matched_set 17 全部与阳性同集（dev）；31 边为未构成绑定的单侧相似——『通过结构近邻绑定检查』只能按 min 口径声称，不得写成所有单侧结构相似已跨集合隔离（用户裁决原文）。

### 预登记敏感性方案（模型结果出现时执行）

1. 主分析照常（min 口径冻结划分）；2. 敏感性变体=按 max 口径合并涉及组重算（或剔除涉边样本），报告主结论方向/显著性是否改变；3. 若结论依赖这 31 条边或涉边后样本量不足以支持原定主张，按协议降级结论并在报告标注——不改口径掩盖风险。全程不读取 conf/holdout 标签选择规则。

## 6. 与 split_qc.json / split_manifest_report.md 的一致性

本报告全部数字与 data/splits/split_qc.json（run_ts 2026-09-24 03:27）及 reports/split_manifest_report.md 逐项一致（两者为 P2.03 重划分后的现行版本）；需要修正的只有 reports/G2_milestone.md §3（已同步修订）。
