# G2-REV 独立审核记录（G2 有条件确认核对修订）

- 审核时间：2026-09-24 16:10–16:26 Asia/Shanghai（真实时钟）
- 审核人：独立审核（ZCode 代理 agent_3a0a79bb，未参与被审改动）
- 对象：工作区未提交改动 vs HEAD 57fefa1，9 文件：新增 scripts/g2_rev_recompute.py、reports/g2_rev_recompute.md、reports/g2_rev_recompute_qc.json、reports/g2_rev_maxrule_edges.tsv；修订 reports/G2_milestone.md、reports/fs_three_layer/process_deviation_log.md（追加 Phase 2 节）、TODO.md、configs/split_protocol.yaml（g2_confirmation 节）、logs/decisions.md（15:47 条目）。与授权清单（用户 G2 有条件确认核对修订任务四项+两裁决落地）一致。

## 结论：通过（0 BLOCKER / 0 MAJOR / 2 MINOR）——2 MINOR 均为文字/溯源精度，修复后通过（修复未再开轮，处置见文末）

## 独立复测结果（审核人自写代码，未 import 被审脚本）

1. **复算全对**：manifest 4,858 行 5 列无标签列；task_area×split=knot 970/247/184、FS 81/7/8、disorder 2,380/447/510、受控三系统全 dev（总量对账 1,401/96/3,337/9/6/9）；打结类型（eligible，c2_primary，112 行）=dev 3_1 75/4_1 7/5_2 2/5_1 1（85）、conf 仅 3_1×14、hold 3_1 11/4_1 1/**5_2 1**；strict 10 对（target=1 且 valid_mask=1）全 development；L2 背景 343,682/68,654/71,850+EP 162/14/16——与三方产物逐项一致。
2. **max 边独立复算=31**：21,528 行边表中 max≥0.6>min 共 58 条，27 条一端为 PN_（正确排除），FS-FS 跨集合=31（conf-dev 8/dev-hold 23）、22 个唯一样本对、涉 strict 3 个（porter_9/61/62）——与 leakage_audit_qc L4=31 及 g2_rev_maxrule_edges.tsv 全部 12 字段×31 行一致（排序无关比对 True）；唯一样本/所属集合/相似度/影响面四要素齐备（recompute §5 按 L1/L2/L3 逐层+预登记敏感性）。
3. **初版划分溯源**：git show cd768cd 版 split_qc.json=knots 1,015/170/216、FS 78/4/14、disorder 2,366/498/473、type 80/14/18——G2 原值确为初版（且 80/14/13 属混合快照的说法成立）；现行 split_qc.json（03:27）与全部复算一致。
4. **脚本复跑**：备份后 exit=0；maxrule TSV cmp 逐字节一致；md/qc 除 run_ts 外全同；已从备份恢复（三文件 md5 与运行前相同）。
5. **Git 事实逐条核实**：reflog 链 88b5015→c011428（仅 TODO 5+/4-）→835203f（7 文件 233+/3-）；三提交父均=cd768cd；c011428 仅 reflog 可达；P2.03 关键文件 blob 哈希三次提交间不变；`git log -L44` 证实 license 运行时实测行入史于 835203f、提交版 json 末字节 7d 0a（有换行）——**"MINOR-1/2 修复未经复审随 amend 入库"缺口登记属实**；闭合证据（logs/reviews/P2.05.md 第 4 条复跑+逐字段比对仅 load/forward 漂移）有原文支撑；P2.04 审核对象=6 文件、时间窗 03:45–03:49 与提交时刻 03:48:53 的先后关系登记一致；P2-2 并行事实（两树行均 03:56、cae49d6→7cbf8ff、41f2618 时刻、8/10 文件互不污染、metrics/baselines 与 extractor 无交叉引用）全部核实。
6. **裁决落地保真**：split_protocol.yaml 可解析；g2_confirmation/G2_milestone §4/§5/§7/decisions.md 条目与用户原文逐条对照无加码无弱化无走样（min 主张边界、31 边登记、敏感性预案、不改口径掩盖风险、不读 conf/holdout 标签选规则、knots 补算前表述禁令、三条停止条件、放行范围）；§5 第 2/6 项括注系 G1 既有限定如实沿用（decisions.md L97/L98）。
7. **TODO 与越权**：进度行勘误正确；全仓 grep 无"P2.03=c011428"/"六项均单独提交"的当前性陈述残留（仅存于指令引述、勘误说明与历史审核记录原文）；G2-REV 节审核时为诚实"待审核"状态；改动恰为 9 文件，data/、logs/reviews/、reports/tasks/、P2.02 历史报告正文未动；P3.01 经查确不依赖打结隔离（放行范围成立）。

## 发现与处置

- MINOR-1：process_deviation_log.md P2-1"实际=五个可达提交覆盖六任务"但冒号后列 6 个哈希（P2.01 两步两提交）——计数错误。**处置：改"六个可达提交覆盖六任务（P2.01 两步两提交；P2.03/P2.04 共享一提交）"并重读核对。**
- MINOR-2：G2_milestone §3 L2 行"与初版一致"不实——初版划分（cd768cd 版 gz）背景蛋白=335,662/73,349/75,175（EP 156/8/28，审核人实测、执行人复核复现），343,682 为现行复算值且恰为 G2 原值。**处置：G2_milestone 行改"G2 原值即现行值（复算维持）；此行并非沿自初版（初版=335,662/73,349/75,175）"；g2_rev_recompute.py 引言与 §4 同步限定并重跑（run_ts 16:30 版），maxrule TSV 复跑逐字节一致。**
- 两个 MINOR 修复均为文字/溯源精度，不动摇数字、裁决落地或偏差登记实质；修复后未再开审核轮（与 P1.17/P1.19"MINOR 处置后收口"先例一致）。
