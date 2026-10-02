# KNOT-RESPLIT 步骤 2：标签曝光核查（打结确认/保留标签读取盘点）

- 时间：2026-09-25 17:20–17:30 Asia/Shanghai
- 依据：用户裁决步骤 2 原文——"查明 confirmation/final holdout 标签何时、由谁、经哪个脚本读取，
  是否影响模型选择或本次划分决策；按 split_protocol 的规则补记 exposure_log。若曾影响方法选择，
  执行相应降级，不把这些样本重新称为未见样本。后续结构图和可行性计算只使用所需的 ID、结构
  相似度及已登记的分组约束。"

## 读取清单（exposure_log.tsv 已补记 5+3=8 行，登记 2026-09-25 17:25 与 19:00）

**修订（步骤 5 审核一审 MAJOR-1）**：原表未穷举——另 3 次确认/保留标签读取已补登记：
(i) 09-25 01:11 p303_fetch_knot_sequences.py 读全部 1,020 usable 行（含确认 202/保留 130 链）
写 knots_sequences.tsv 并做 same-sha×标签跨 split 冲突检查（冲突=0，无方法影响）；
(ii) 09-25 16:59 步骤 3 预演的分配后构成统计读 knots.tsv 确认/保留行；
(iii) 09-25 17:07 步骤 4 finalize 的 split_qc 构成统计同上。三行 influenced=false 如实。
以下原表保留为初版记录（其自身范围穷举性以此修订为准）。

| # | 时间 | 谁 | 脚本 | 范围 | 内容 | influenced_method_selection |
|---|---|---|---|---|---|---|
| 1 | 09-24 03:14 | ZCode(P2.02/03) | build_p202_split_manifest.py | 全任务含 knots conf/hold 对应行 | 划分生成本身（split 尚不存在） | No（数据构建动作） |
| 2 | 09-24 15:55 | ZCode(G2-REV) | g2_rev_recompute.py | knot type×split 聚合 | 三集合类型构成修正 | No（聚合报告） |
| 3 | 09-24 16:46 | ZCode(P3.01) | check_input_identifiability.py | knots 全体聚合 | presence 阳阴总数/type eligible/类型×集合 | No（聚合报告） |
| 4 | **09-25 16:10** | **ZCode(G2 决策点)** | 内联 python→knot_binding_check.py | **usable 1,006 链逐链（含确认 ~247/保留 ~184 子集）** | split 归属+presence_target 逐链；跨集合对清点、涉边阳性链 39 条计数 | **No**：边与停止判定完全由 US-align 相似度×集合归属决定；标签仅用于影响面报告（涉边阳性 39 条） |
| 5 | 09-25 17:10 | ZCode(步骤 1) | knot_binding_check.py+validate_knot_binding_freeze.py | 同 #4 | 冻结校验复现（F4：2,998/679） | No（复算） |

## 判定

1. **influenced_method_selection=No（全部 5 条）**：结构绑定边由 US-align 相似度单独决定；
   停止条件触发由"结构相似×集合归属"判定；标签仅出现在影响面统计（涉边阳性 39 条）。
   不存在"读了确认/保留标签来挑规则/挑样本/调方法"的通道。
2. **降级：无**（无方法选择消费确认/保留标签；全部探针/选择仅 development——
   run_probes_p303.py knots 分支与 build_p303_inputs.py knots fasta 均显式过滤 split==development，
   已代码级复核）。
3. **"未见"措辞纪律维持**：确认/保留样本从未被称为未见方法样本；本次补记后性质=结构隔离
   验证未通过（决策点报告），不因曝光核查改变。
4. **前置约束（用户原文落地）**：步骤 3 预演与后续结构图/可行性计算只使用（a）链/样本 ID、
   （b）US-align 结构相似度、（c）已登记分组约束（S1/S2 曝光强制 dev 组、matched_set、
   同 UniProt/同序列绑定）；**presence/type 标签不进入任何分配计算**，仅用于分配完成后的
   构成报告（与 split_qc 先例同口径）。

## 产物

- logs/exposure_log.tsv +8 行（5 行登记 17:25；3 行补登记 19:00）
- 本文件
