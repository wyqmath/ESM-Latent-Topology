# 决策与异常记录

记录时间：2026-09-21 08:16 Asia/Shanghai
关联任务：P0.01
事实与证据：本机从未配置 Git 身份——无 `/Users/yuan/.gitconfig`、无系统级配置、无 Git 身份环境变量；`git var` 仅能自动识别 `圆 <yuan@yuandeMacBook-Air.local>`（证据：logs/reviews/P0.01.md 第一轮审核第 8 条独立复验）。首个提交的作者身份将永久写入提交历史。
影响：P0.01 验收条件"完成首个经审核的提交"在身份确定前无法执行；不影响其他可验证验收条件（均已通过审核）。
可选处理与建议：a) 接受系统识别身份写入本仓库 local 配置；b) 用户提供姓名/邮箱，仅写入本仓库 local 配置；c) 用户自行配置后由代理重试；d) 暂不提交保持待提交状态。按用户启动提示词规则：不编造身份、不修改全局配置，故须用户决策。
讨论确认：用户 2026-09-21 08:16 决策选择 b，提供身份 `Jiayuan Ma <m01030709@163.com>`（仅写入 `/Users/yuan/Documents/ChatGPT/PLM` 的 local 配置）。
落实记录：`git config --local user.name 'Jiayuan Ma'`、`git config --local user.email 'm01030709@163.com'` 已写入并验证；`/Users/yuan/.gitconfig` 确认仍不存在（全局未动）；随后完成 P0.01 首个提交。

---

记录时间：2026-09-21 08:42 Asia/Shanghai
关联任务：P0.03
事实与证据：P0.03 验收条件含"模型名单和试跑范围经讨论确认"。资源实测（本机 M5/16GB/磁盘余 35GB）与试跑规则草案经独立审核通过（agent_e7c27c22，2026-09-21 08:38，logs/reviews/P0.03.md），三项决策点提交用户裁决。
影响：决定远程算力可用性、模型名单时点与排期阻塞范围。
可选处理与建议：见 reports/scope.md §5（集群三选项、名单三选项、工期两选项）。
讨论确认：用户 2026-09-21 08:42 裁决——① sblab 集群可用于本项目，保守共用（P2.05 实测前默认 ≤1 节点）；② 模型短名单讨论推迟至 Phase 1 数据审计后、P2.04 准入核验前（TODO 第 5 节允许模型版本在 P0.03、P2.04 两处解决）；③ 截止日期/每周时间/预算登记"待确认"，全量运行排期保持阻塞。
落实记录：configs/resources.yaml status=confirmed_partial、sblab_cluster status=confirmed_conservative_sharing；reports/scope.md 状态段与 §5 改为确认记录；模型候选表在其被确认前不得用于下载或运行。

---

记录时间：2026-09-21 09:16 Asia/Shanghai
关联任务：P0.05（关联 P0.06 提前询问）
事实与证据：P0.05 四个统计参数（bootstrap B=2000、置信水平 0.95、种子 [13,42,2026]、Holm–Bonferroni）与 P0.06 划分阈值（同源 ≥30% 同一性且覆盖短序列 ≥70%；结构近邻 USalign TM≥0.6；比例目标 70:15:15）为推荐默认值，2026-09-21 09:12 向用户发起确认，未收到答复。独立审核已通过（agent_534b73a2，2026-09-21 09:08，logs/reviews/P0.05.md）。
影响：参数不能视为已冻结；任何科研运行在冻结前不得引用这些具体数值。
可选处理与建议：a) 用户随时答复即冻结回写；b) P0.05 参数冻结死线=P3.02 配置冻结前；c) P0.06 阈值冻结死线=P2.01 分组实现/P2.02 划分生成前（TODO 第 5 节允许 P0.06、P2.01–02 两处解决）。
讨论确认：未确认（2026-09-21 09:16 记录为待确认，保留推荐值并显式标注未冻结）。
落实记录：evaluation_protocol.yaml status=formulas_frozen_params_pending、参数标注 [待确认·未冻结] 并写入 freeze_rule；P0.06 的 split_protocol.yaml 将以同样方式登记推荐阈值与冻结死线。

---

记录时间：2026-09-21 09:25 Asia/Shanghai
关联任务：P0.06（对 09:16 条目的澄清，独立审核 agent_87237dbc 2026-09-21 09:23 指出的表述偏差）
事实与证据：P0.06 待冻结项共 6 处；09:12 询问的书面逐项列举仅含 3 个数值阈值（同源 ≥30%/覆盖 ≥70%、结构 TM≥0.6、比例 70:15:15）。
影响：无（保护性纪律对全部 6 处生效）；仅澄清记录口径。
可选处理与建议：补记本澄清条目（已选）。
讨论确认：未确认项状态不变；逐项处置——同源/结构/比例三阈值随 09:16 条目待确认；划分种子绑定 evaluation_protocol 种子参数一并确认；缺失结构规则随同一冻结死线（P2.01/P2.02 前）确认；最低可评价组数为过程性延后项，P1.09 数据审计后逐任务确定。
落实记录：P0.06 执行报告对应句子已修正为本澄清口径；split_protocol.yaml 的 [待冻结] 标注与死线不变。

---

记录时间：2026-09-21 09:35 Asia/Shanghai
关联任务：P0.06（关联 P1.09、P2.01–02、P2.02 备用确认池）
事实与证据：用户审阅 G0 材料后就划分参数提出五点理由——①折叠转换严格候选有限，分完可能某集合只剩极少数独立组；②序列/结构/配对约束合并后组数可能进一步减少；③原方案设"备用确认池"但 70:15:15 已分满 100%，备用数据来源未说明；④旧数据有历史使用记录，重新分配不自动使其成为"未见数据"；⑤序列与结构近邻隔离不自动等于满足"未见家族"测试。
影响：划分比例/阈值/种子的冻结流程改变——由"死线前确认"改为"P1.09 后先交审计+可行性建议、用户确认后才冻结"；备用确认池来源成为必答设计项；family 级声明约束进一步收紧。
可选处理与建议：立即采纳（维持协议纪律、强化前置 Gate）；不采纳则无法回应用户指出的可行性风险。
讨论确认：**用户已确认（2026-09-21 09:35）**：划分参数现在不最终冻结；P1.09 后先提交每任务的有效样本、独立组、类别与历史曝光审计以及划分可行性建议，再由用户确认；不得为凑三集合偷偷拆组或放宽门槛。
落实记录：split_protocol.yaml——status 改为 protocol_frozen_thresholds_deferred_to_post_P1_09；allocation.target_proportions 改为"无预设比例，候选仅作讨论输入"；新增 3b feasibility_gate 硬前置节（必交材料六项：每任务有效样本/独立组/类别、合并约束后组数量与巨型组分析、历史曝光审计、家族层隔离可行性评估、备用池来源方案、冻结建议与不可行清单）；backup_confirmation_pool 增加 sourcing_open_question（候选 A 发现集内预留 / B 后续新增数据 / C 分层多折确认，由可行性建议评估）；allocation.guardrail 明确禁止拆组与放宽门槛。G0 材料风险项与 TODO P0.06 详情同步更新。

---

记录时间：2026-09-22 21:58 Asia/Shanghai
关联任务：治理变更（PLAN register three-layer fold-switch workflow）；关联 P1.09–P1.12、P2.01/02/06、P3.05、P5.06
事实与证据：用户 2026-09-22 提供三层任务设计文档（原文归档 reports/plan/three_layer_design_user_directive_20260922.md；变更摘要 reports/plan/three_layer_governance_20260922.md）。科学依据：折叠转换全局二分类缺乏可靠生物学阴性（P1.02 疑点 3，OD-NEGATIVE-FS），单层 AUROC/AUPRC 结论易被数据收录偏差解释——阳性蛋白因"研究得多、结构多"而入选。三层设计将结论分解为：L1 阳性内部机制诊断（不受阴性缺失影响）；L2 阳性在固定未标注背景中的排序/富集（PU 语义，不做发生率解释）；L3 与操作性定义的匹配推定阴性（PN-A/B/C 证据层）的病例-对照比较（类别比例由设计产生，校准禁用）。
影响：claims 任务注册（新增 3 子任务，T-FS-GLOBAL 保留为总任务、T-FS-REGION 归 L1 残基级）；sample_schema 增 11 列（label_epistemic_status/biological_target nullable/pu_observed_label/positive_evidence_tier/negative_evidence_tier/background_source_version/literature_search_query/literature_search_date/structure_count/independent_study_count/matched_set_id）；evaluation_protocol 增三层指标（L2 recall@k/enrichment@k/rank percentile/family-stratified 与 leave-family-out enrichment；L3 AUROC/AUPRC/balanced accuracy/MCC/per-class PR，校准 disabled）；split_protocol 增历史曝光/matched_set_id/未见家族独立审计规则；TODO 增 P1.10–P1.12（Phase 1 任务数 9→12，全项目 46→49）、扩充 P1.09；P1.02/P1.03 完成状态保留不重开。
可选处理与建议：按用户文档立即登记；未实现部分（背景构建、PN 筛选、匹配、三层协议冻结）归 P1.10–P1.12 与 P2.01。
讨论确认：**用户已确认（2026-09-22，提供三层设计文档并指示直接实施）**；OD-NEGATIVE-FS 状态改为 direction_approved/implementation_pending，P1.12+G1 确认后方可 resolved。
落实记录：本次治理变更单独提交（标题 PLAN register three-layer fold-switch workflow）；独立 subagent 审核计划修改后才提交；不把任何新增科研任务标为完成；完成计划提交后继续 P1.05。执行纪律：无文献命中≠生物学阴性；unlabeled 不写 target=0；病例-对照比例≠总体发生率；不因样本不足拆同源/匹配/配对组；$B 只读；不下载 PLM 权重（P2.04 前）；不生成正式划分。

---

记录时间：2026-09-23 08:39 Asia/Shanghai
关联任务：P1.10（折叠转换未标注背景）
事实与证据：两个冻结背景宇宙建成并通过三轮独立审核（logs/reviews/P1.10.md）。宇宙 A=UniProtKB/Swiss-Prot 2026_03（575,748 官方条数锚定；484,179 行：排除 FS tier 39+长度 844+非标字母 2,667+精确重复 88,019）；宇宙 B=SIFTS 2026-09-13（PDB 37.26/UniProt 2026.04）+wwPDB derived 2026-09-12（75,593 蛋白/990,140 有效键）。版本偏差 2026_03 vs 2026.04 已登记（覆盖 cap 1.0）。pending 53 对按 PU 语义保留为未标注；56 个 pending accession 中 6 个由同序列代表承载（qc 逐一映射），G1 裁决阳性时须按 sequence_sha256 连带剔除代表行而非仅按 accession。
影响：T-FS-L2-PU-RANK 的排序宇宙与 P1.11 PN 候选池的结构基础就绪；SIFTS 原始数据质量边界成文（倒置行 72/出表键 36/重叠键 227+252/并集后 observed>mapped 键 59——后者连同全倒置键列入下游禁用规则，不得当覆盖证据）；覆盖口径=区间并集（重叠去重、相邻不并、缺口不计）+主表与子表重聚合一致性断言 A10。
可选处理与建议：全部采纳（独立审核三轮 PASS，全量重算零差异）。
讨论确认：用户未逐项确认（本任务为 G1 前数据建设）；SIFTS 异常清单与 pending 代表规则将随 G1 材料一并提交。
落实记录：configs/fold_switch_background.yaml 冻结（含 A1–A10）；大于 5MB 产物 gz 打包（GitHub 100MB 单文件限制；总量约 44MB）；manifest 登记 7 源 SHA256；fetch/build 脚本入库；data/raw 不入 git。

---

记录时间：2026-09-23 09:50 Asia/Shanghai
关联任务：P1.11（折叠转换推定阴性候选池）
事实与证据：PN 候选池建成：139 行/138 唯一候选（每阳性 top-30，家族/长度/结构数预注册排序）；Europe PMC 系统检索 138 候选（216 文件归档+SHA256 清单），命中分类 842 条（退出类 fold_switching 14+domain_swapping 52）；PN 层 PN-A 59/PN-B 38/PN-C 26/EXITED 15。两轮独立审核（FAIL→PASS）：第一轮拦下 BLOCKER=长度窗因阳性长度源错误全程失效（改从 P1.09 audit fasta 取）与 MAJOR=Europe PMC nextPageUrl 未编码导致分页全败（cursorMark 重编码修复）；第二轮全量重算 0/139 差异。
影响：L3 匹配对照的候选基础就绪；关键可行性事实——3/10 strict positives（B9W5G6/Q58AD3/Q8E473）在 SIFTS 家族宇宙零同家族者、3 个阳性候选池极薄（3/2 个）；Europe PMC 快照不稳定（同查询一小时内 EXITED 19→16→15）须以归档快照+全文复核语义消化；4 个复用截断候选登记 P2.01 前补抓。
可选处理与建议：全部采纳（含 n5 登记）；全文级复核 pending 为 G1 决策项。
讨论确认：用户未逐项确认（G1 前数据建设）；PN 池与退出队列将随 G1 材料提交。
落实记录：configs/fold_switch_putative_negative.yaml 冻结；B1–B8 断言全过；live==cached 三表逐字节一致；报告 reports/tasks/P1.11.md；审核 logs/reviews/P1.11.md 两轮。

---

记录时间：2026-09-23 10:09 Asia/Shanghai
关联任务：P1.12（三层任务合同冻结与 G1 可行性材料）；关联 G1
事实与证据：configs/fold_switch_three_layer_protocol.yaml 冻结（L1/L2/L3 八要素、进入条件、D1–D6 降级规则、k_rules 冻结 k=10/100/1000/10000 与家族主口径、背景版本锁定）；validate_three_layer_protocol.py 六项校验全过（jsonschema+claims/evaluation/split 交叉核对）；feasibility_matrix.tsv 26 行全部复算（usable 限定 strict 人群 3 对 6 行的口径修正经两轮审核）；G1_fold_switch_decision.md 六项冻结+14 项待确认。
影响：折叠转换三层合同进入 frozen_pending_G1_confirmation 状态；G1 前置的 P2.01 匹配/划分/模型实验全部锁定；OD-NEGATIVE-FS 维持 direction_approved__implementation_pending（P1.12+G1 后方 resolved——现 P1.12 已过审，resolved 待 G1 确认）。
可选处理与建议：两轮审核（FAIL→PASS）全部修复；r2-2（矩阵脚本 docstring 措辞收窄）留待下次触碰。
讨论确认：G1 需用户确认（14 项清单见 G1 文档）；统计参数与模型名单两项独立冻结死线不变。
落实记录：本条随 P1.12 单独提交（标题含 P1.12）。

---

记录时间：2026-09-24 00:15 Asia/Shanghai（用户消息处理开始时刻；消息送达与本记录同处一会话）
关联任务：G1 Gate 决议记录（关联 P1.12/P1.19–P1.23/P2.01）
事实与证据：用户 2026-09-24 00:14（处理时刻）明确回复：**有条件确认 G1，采纳 G1_resolution_options.md 选项一，附六项限定**。原文要点与对原选项一的修改逐条如下。
1. strict=10 对按证据强度分层候选；序列身份与双态证据分别报告；porter_9/80 保留旧审计依赖；L1 九对构建体不等性须入混杂检查并设长度/覆盖/边界等简单对照，通过前不得把状态判别表现归因为模型识别双态机制。【修改：新增 L1 混杂对照门槛（原选项一无此执行项）】
2. 采纳残基 fine∧usable 3 对 6 行案例级口径、porter_8 extension 分层、DisProt 残基二分类降级；L2 仅报告已知阳性在固定未标注背景中的排序与富集；L3 确证性主分析仅用完成逐例人工全文复核的实际入选 PN-A 对照——**适用于所有入选对照，不限于两个关键词标记候选**；最终病例数以复核+匹配完成后为准。【修改：D4 从"标记候选复核"强化为"全部入选对照逐例复核"；porter_8/DisProt 由建议升为已采纳】
3. 当前注记下无同家族候选的三例暂不进 L3 主结果，可探索结构相似候选；报告须说明家族注记缺口；不得把"当前未找到同家族候选"写成生物学上不存在同家族蛋白；PN-C 不用于凑 L3 主结果。【修改：新增措辞禁令（防 D1 过度解读）】
4. 历史使用按 P1.19 S1–S5 口径分记；**用户裁决：96/96 前项目 legacy 登记+旧项目数据构建使用=历史接触记录**；旧项目正式评价使用=0；v5 收录=35/96；53 pending 对现不构成可靠未曝光备用池；未来新建独立确认池须另交样本来源、补证结果与方法选择曝光审计后决定。【修改：解决了原选项一挂起的"legacy 口径裁决"（C 类项收口为已裁决）】
5. 有条件接受当前背景版本组合；P1.21 三项例外（TrEMBL 锚定×2、家族注记缺口、PubMed 缺失）随样本进下游报告；接受范围以已审计连接与用途为准；Europe PMC 选快照锁定方案 A；发现新关键证据另立版本并记录影响，不静默改写已锁定结果；BPTI 表示来源、统计参数、模型名单维持原决策节点。【修改：方案 A 由"推荐"升为"已选"；新增"不静默改写已锁定结果"约束】
6. G1 后顺序：先 P1.24、P1.25（各自独立审核、单独提交）；随后 P2.01 拆为可分别验收步骤——先用候选阈值补结构近邻关系+合并分组可行性报告（独立组数/巨型组/各任务可评价量/历史接触/匹配可行性）交用户确认；**确认前不生成正式 group_map、最终 matched_set、split manifest，不开始模型实验**；阈值确认后完成正式分组与匹配，再进 P2.02。【修改：P2.01 拆两步+新增用户阈值确认 Gate（原选项一无此中间确认）】
另外：任何未获证据支持的"未见家族""独立确认"或 L3 阴性结论继续保留为未成立。
影响：G1 状态 待讨论→已确认（有条件·选项一，2026-09-24 用户）；三层协议状态更新为 g1_confirmed 并落入六项限定；P2.01 步骤一解锁（仅可行性报告，无 group_map/matched_set/manifest）；P1.24 范围随第 2 条限定扩入 porter_8 表内分层（extension_condition_or_assembly，随 P1.24 同一重建落实并单列差异）；porter_8/DisProt 的表级与分析级处置在 P1.24 与协议中落实，不改 DisProt 数据表。
可选处理与建议：全部按原文采纳；无降级。
讨论确认：用户已确认（本条即决议原文要点+修改对照；完整原文见会话记录，与 G1_resolution_options.md 选项一对照阅读）。
落实记录：本条随 G1 记录单独提交；同步改动=TODO（里程碑/进度/P1.24 范围注记/P2.01 两步验收）、configs/fold_switch_three_layer_protocol.yaml（status+g1_confirmation+D 规则修订）、G1_fold_switch_decision.md 与 G1_resolution_options.md 决议标注；validate_three_layer_protocol.py 回归通过后提交。

---

记录时间：2026-09-24 02:34 Asia/Shanghai
关联任务：P2.01 步骤二（阈值冻结）起的运行授权
事实与证据：用户 2026-09-24 02:30 前后消息："你现在先存档，之后自己定一个参数开始一直运行，能运行多久就多久，能运行多少步就多少步，不要自己停下来。后面的你替我抉择"——构成对 P2.01 步骤一报告（p201_step1_grouping_feasibility.md）所列决定项的**授权委托**：参数由我抉择冻结，持续运行，后续决策代行。G1 限定 6 的"阈值确认 Gate"由此视为用户以委托方式放行。
影响与抉择（我代用户做出，理由如下）：
1. 结构近邻口径=**US-align 单链 TM≥0.60 + min 对称化**（双侧均≥0.6 才连边）。理由：max@0.6 产生 35 对巨组并把 6/10 strict 锁进同组，破坏组级分配空间与可评价独立性；min@0.6=89 组接近序列层粒度（93），绑定保守，残余单侧相似泄露由 P2.03 审计与划分报告显式标注。max 口径数据保留在边表可随时改用。
2. 序列同源=mmseqs2 18-8cc5c easy-cluster min-seq-id 0.30、-c 0.70、cov-mode 0（P1.17 敏感性 92–95 组、阈值不敏感，取候选值）。
3. 全局组级比例 70:15:15；划分种子=2026。
4. 最低可评价量：按任务在 manifest 如实报告，不设人为下限，不可行走 D 规则降级、不拆组凑数。
5. L3 确证子集=完成逐例人工全文复核的入选对照（复核状态随 matched_set 携带；复核列为后续任务，未复核前相应对照只入敏感性不入确证主分析——G1 限定 2 原文维持）。
可选处理与建议：无（授权范围内执行）。
讨论确认：用户以 2026-09-24 消息委托；本条为委托下首次行使，逐项可追溯可撤销（改口径=差异清单+重跑清单另立任务）。
落实记录：随 P2.01 步骤二单独提交；split_protocol.yaml 相应 [待冻结] 项转 frozen 并注授权来源。

---

记录时间：2026-09-24 03:28 Asia/Shanghai
关联任务：P2.03（泄露与预训练曝光审计）及其对 P2.02 的驱动修正
事实与证据：P2.03 审计（scripts/audit_leakage.py）发现两类划分绑定缺陷：①1,470 个有标注↔背景同 UniProt 无绑定跨集合（E0 边未联动背景簇）；②3 个 FS 端点序列（porter_10/64/95，pending/extension）与背景精确同序列未绑定。属 P2.02 绑定不完整，非规则缺陷。
影响：按铁律 2 修根因并重建——build_p202_split_manifest.py 增两类绑定+forced_dev 根传播；划分重生成（69,084 组；dev/conf/hold=48,372/10,359/10,353=70.0/15.0/15.0%）；重审 leakage 全零（L4 max 口径跨集残余 31 条如实披露，为已冻结口径的已知特性）。
可选处理与建议：绑定修复为唯一正解（排除背景蛋白会破坏 PU 宇宙）；max 残余交由用户已冻结口径决定，不改。
讨论确认：用户委托授权范围内（decisions.md 2026-09-24 02:34）；本条随 P2.03 提交。
落实记录：P2.02 产物与报告修订（含修订段）；P2.03 报告与 qc；同一次提交。

---

记录时间：2026-09-24 15:47 Asia/Shanghai（消息收到即记；核对修订执行 15:50 起）
关联任务：G2 里程碑确认；核对修订任务 G2-REV；Phase 3 放行
事实与证据：用户 2026-09-24 15:47 消息，对 G2 作**有条件确认**，条件：
1. 先完成 G2 材料的事实核对、修订和独立审核（核对修订任务 G2-REV）：①用最终 split manifest/split_qc.json/split_manifest_report.md 复算 G2 各任务三集合数量与打结类型分布，修正沿用旧划分的数字（含 final holdout 中的 5_2 样本）；②核对 Git 历史（835203f 同时含 P2.03/P2.04 文件、c011428 已被 amend 替代），修正"六项均单独提交"表述、任务提交号与 TODO 状态，如实登记流程偏差，不静默改写提交制造整齐历史；③核查 P2.05/P2.06 相同开始时间是否并行并记录实际经过与影响；④G2 材料写清后续适用范围（strict 10 对全 dev；17 个 L3 对照逐例全文复核通过前不进确证主分析；P2.06 逻辑回归仅描述性诊断；统计参数 P3.02 前冻结）。按项目纪律执行：时间入账、产物可复算、独立审核通过后提交、更新 TODO 与运行记录。
2. 科学裁决两项（用户裁定，不必重复征求确认）：
   裁决一：折叠转换结构近邻维持冻结的 US-align TM≥0.6 min 对称化规则为本轮分组与主分析操作口径；论证强度与证据相称——只能声称 min 规则下通过结构近邻绑定检查，不得声称所有单侧结构相似关系均已跨集合隔离；保留并核对 max 口径 31 条跨集合边（唯一样本/所属集合/相似度/可能影响的分析），模型结果出现时按事先写明方法做敏感性检查并报告结论是否受影响；结果依赖这些边或样本量不足即按协议降级，不改口径掩盖风险；不读取 confirmation/final_holdout 标签选择规则。
   裁决二：打结 Foldseek 结构近邻补算未完成，但允许推进不依赖"打结样本已通过完整结构隔离验证"结论的工作（输入准备/表示提取/分析代码编写测试/development 探索）；补算及必要 US-align 确认完成前，不得把打结 confirmation/final_holdout 结果表述为已完成结构隔离的独立验证；集群恢复后补算；若发现应绑定的跨集合近邻，保留旧划分与差异记录，按协议修订划分并重跑受影响分析。
3. 放行：上述条件经审核满足后，可依据本有条件确认继续执行，不必为 min 规则或打结非依赖性工作再次等待回复；若补算发现需重划分、敏感性检查推翻主结论，或下一任务必须依赖尚未完成的确证证据，停在决策点提交具体差异和处理方案。
影响：G2 状态 待讨论→已确认（有条件，2026-09-24 15:47）；Phase 2 收口并放行 Phase 3 非依赖任务（P3.01 起）；G2 材料修订为 G2-REV 版（§3 数字复算修正、§4 残余与敏感性、§5 适用范围、§6 偏差登记）；31 条 max 边全表落盘 reports/g2_rev_maxrule_edges.tsv。
可选处理与建议：全部按原文采纳；无降级。
讨论确认：用户已确认（本条即决议原文要点；完整原文见会话记录）。
落实记录：本条随 G2-REV 单独提交；同步改动=TODO（进度行/G2 块/G2-REV 任务）、reports/G2_milestone.md（§1/§3/§5/§6/§7）、configs/split_protocol.yaml（g2_confirmation 节）、process_deviation_log.md（Phase 2 节）、复算三产物。

---

记录时间：2026-09-25 00:25 Asia/Shanghai
关联任务：P3.02（冻结开发期探针对比配置）+ 统计参数冻结（G2 适用范围第 4 项死线）
事实与证据：用户委托（decisions.md 2026-09-24 02:34"后面的你替我抉择"）+ 用户 G2 有条件确认适用范围第 4 项（"统计参数须在 P3.02 前冻结"）。P2.06 以来无任何科研运行消费过统计参数（bootstrap 未运行），本冻结发生在首个科研消费方（P3.03）之前，无回溯暴露。
抉择内容（我代用户做出，逐项可追溯可否决）：
1. 统计参数四项=原推荐值原样冻结：bootstrap B=2000、CI=0.95 percentile、种子 [13,42,2026]、多重比较=Holm–Bonferroni（仅预注册主对比族：层可读性/容量增益/L2 vs 基线，定义见 probes.yaml stats.primary_families）。理由：推荐值即原候选清单中的保守中位选项，且种子 2026 与划分种子族一致；无需引入新数值。
2. 探针配置冻结（configs/probes.yaml）：层集合 global=[5,11,17,23,29,33]、residue=[11,23,33]（残基 3 层为磁盘预算事前约束，非事后选择）；读取器=线性（逻辑回归 C 网格）为主，非线性 MLP 四配置归 P3.04 容量对照；选择=dev_fold 5 折 CV 仅 development；L2 两阶段（stage-A 簇代表抽样 10,000 选择、stage-B 全宇宙 343,682 官方指标）；执行序 A–D；资源：df 实测可用 13Gi（2026-09-25 00:37，/dev/disk3s1s1），峰值 ≈4.6Gi（残基任务只存任务分母残基域——disorder dev 223,660 残基 fp16×3 层 ≈1.60Gi；存储域定义冻结于 probes.yaml layers.residue_storage_domain）。
影响：evaluation_protocol.yaml 四项 [待确认·未冻结]→[已冻结]；P3.03 可开跑（preflight=knots/disorder 序列获取）；确认/保留集继续零接触。
可选处理与建议：全部按委托执行；如用户否决任一项，按差异清单流程重跑受影响部分（当前无已消费结果，返工成本=0）。
讨论确认：委托范围内执行；本条随 P3.02 提交。
落实记录：configs/probes.yaml（冻结）；configs/evaluation_protocol.yaml（回写）；scripts/validate_probes_config.py（V1–V7 全过）；reports/tasks/P3.02.md。
（00:53 整改修订：本条目资源数字按一审更正——磁盘可用 18Gi→实测 13Gi、残基存储改任务分母域口径后峰值 8.6Gi→4.6Gi。）

---

记录时间：2026-09-25 16:50 Asia/Shanghai
关联任务：P3.03（打结 Foldseek 结构近邻补算，G2 裁决二兑现）
事实与证据：集群恢复后完成补算（SLURM 812980/813024 系列；US-align 与 FS 冻结规则同口径 min≥0.6）：
1,006 usable 链中绑定级对 2,998，其中**跨集合 679**（conf↔dev 301/dev↔hold 230/conf↔hold 148），
涉 339 链（阳性 39），minTM 中位 0.853/最高 0.998；另 132 条跨集合单侧相似未构成绑定（信息性）。
打结任务未通过 min 口径结构近邻绑定检查。
影响：**触发用户 G2 停止条件**——停在该决策点。已保留旧划分（未动任何划分文件）；P3.03 其余部分
（L2 stage-A/B、FS/disorder/knots dev 探针）已完成不受影响；knots conf/hold 表述维持"结构隔离
验证未通过"。方案 A（全量重划分+重跑）/B（打结局部重划分）/C（降级冻结）已写入
reports/knot_binding_decision_point.md 供用户裁决。
可选处理与建议：建议 A 或 B；C 兜底。等待用户裁决，未获裁决前不动划分。
讨论确认：待用户裁决（本条为停点登记，非决议）。
落实记录：reports/knot_binding_decision_point.md；results/probes/knot_cross_set_binding.tsv（679 行）+
knot_binding_check_qc.json；TODO P3.03 阻塞标注。

---

记录时间：2026-09-25 19:15 Asia/Shanghai
关联任务：KNOT-RESPLIT（方案 A）执行完毕；P3.03 收口
事实与证据：用户 2026-09-25 17:00 裁决方案 A 五步顺序。执行：步骤 1 冻结校验（9 项输入钉版、3,854/2,998/679 口径重现、14 条无结构链保持未验证）；步骤 2 曝光核查（8 行 exposure_log 补记、全部 influenced=false、无降级）；步骤 3 预演可行（mode-0 逐行复现等价性证明；68,875 组；knots 确认 24 阳/126 阴、保留 26/83；S1/S2 35 对全 dev）；步骤 4 新版划分生成（留档旧版+校验值；P2.03 审计重跑 L1–L5 全零+新增 L6=0/2,998；逐样本差异 4,110 行、S1/S2 迁出 dev=0；增量嵌入集群 GPU 813037；六任务探针+L2 stage-B+基线全量重跑 813047–813054）；步骤 5 独立审核两轮（一审 1 BLOCKER+2 MAJOR+7 MINOR→整改→复审通过 0/0/4 并入批次执行完毕）。
新版划分关键数字：组 69,084→68,875；knots 确认 24 阳/126 阴、保留 26 阳/83 阴（结构隔离验证通过后的可评价构成）；L2 官方宇宙 342,014、percentile 0.00031、recall@1000=1.0（元数据基线 percentile 0.353–0.613、recall@10000≤0.1）。
影响：data/splits 三件套+checksums+knot_structural_edges.tsv 替换为新版（旧版留档 reports/knot_resplit/old_version/ 与 git 3c5a596）；P3.03 六任务探针与 L2/基线结果以新版为准（旧划分数字留 git 历史）；leakage 审计新增 L6 检验入协议常备。
可选处理与建议：用户裁决 A 已执行完毕；B 未采用、C 未触发（预演可行）。
讨论确认：用户裁决在案（17:00 消息）；本条为执行完毕登记。
落实记录：KNOT-RESPLIT 详情节完成记录（TODO）；logs/reviews/KNOT-RESPLIT.md；reports/knot_resplit/step1–4 系列；reports/tasks/P3.03.md 终版；split_protocol.yaml 注记；knot_binding_decision_point.md 闭环注。

---

记录时间：2026-09-25 23:55 Asia/Shanghai
关联任务：P3.04（容量对照）一审 BLOCKER 处置
事实与证据：P3.04 一审（agent_3dda8fe9）判不通过：**BLOCKER=Δ 用单位级分数差（概率差）而非 Δmetric**——审核人以 sklearn 独立对照证明 DISORDER500"+0.037 显著正增益"为校准偏移伪像（同 OOF 上真实 ΔAUPRC≈−0.0016，符号翻转）；MAJOR=集群多版中途脚本产物不可再生、summary 破坏性覆写、FS_REGION 16 链违反冻结宇宙（双满足 6 行且敏感性层不跑非线性）、DISORDER500 无 dev 过滤（本地数据下会把 final_holdout 蛋白 DP01351 保底入样）。
处置：统计核心重写为**两臂 pooled metric 之差的配对 bootstrap**（按独立单位组重采样 B=2000 种子 2026；NaN 退化抽样弃除；多重性=族内 99% CI Bonferroni 水平——bootstrap-CI 无 p 值不适用 Holm）；FS_REGION 收敛到冻结 6 链宇宙；DISORDER500 加 dev 过滤断言；全宇宙集群重跑（813061/813066/813097）。
影响：P3.04 头条结论翻转为"五宇宙均未建立可靠非线性正增益"（DISORDER500 新版为不稳定/不结论；FS_REGION bootstrap 病理不结论；knots presence 95% CI 同负=MLP 显著更差）——容量诊断结论方向未变（线性读取器已足够），但显著性表述全部按 Δmetric 口径重述。
可选处理与建议：无（统计口径修正为唯一正解）。
讨论确认：用户委托授权范围（decisions.md 02:34）+ 一审审核意见；本条随 P3.04 提交。
落实记录：scripts/run_p304_capacity.py 重写版；results/probes/capacity/*（新统计）；reports/tasks/P3.04.md（重写版）；TODO 树行/详情节更新。


---

记录时间：2026-09-26 01:45 Asia/Shanghai（补登记 09-26 02:0x）
关联任务：P3.04 二审与三审处置（KNOT-RESPLIT 后续）
事实与证据：P3.04 一审后统计核心重写（分数差→Δmetric 配对 bootstrap）并集群重跑（813061/813066/813097/813102/813103）。二审（agent_3dda8fe9）仍判不通过：新 BLOCKER=pooled() 硬编码 roc_auc_score 致 AUPRC 宇宙 Δ 为跨量纲差（FS_REGION/DISORDER500 数字无效）；FS_L2A TSV 系本地手工重写无脚本/日志出处（流程违规，如实登记）；DISORDER500 无 dev 过滤断言；qc 覆盖销毁 per-seed 记录。三审前 v4 全量重写：pooled 按 kind 分派、DISORDER500 显式 dev 过滤+2,279 断言+500 名单落盘（disorder500_sample.tsv）、FS_L2A 改 2-半交叉拟合 OOF、summary 重建函数（防覆写）、集群重跑 813102→813103 完成。三审（agent_99889feb）：**计算全部合格、五宇宙 Δmetric 通过**，报告/账目层 1 BLOCKER（旧结论块残留）+5 MAJOR→已全部整改（摘要重建脚本化、作业账目更正为 15 份日志、"development"表述对 FS_L2A 更正、非分层措辞、truncation 披露）。
影响：P3.04 终版结论（v4）——**容量增益仅 FS_REGION 池化残基 AUPRC 小而显著（+0.008，sig99）；knots presence 负向趋势（95% 同负、99% 未全同号→不结论）；type/disorder/FS_L2A 无可靠增益**。早期"DISORDER500 +0.037 显著正增益"确认为校准偏移伪像并撤回。capacity_summary.tsv 由脚本 rebuild_summary 从五份 TSV 重建（不再手拼）。
可选处理与建议：无。P3.04 三审通过后收口提交。
讨论确认：用户委托授权范围内。
落实记录：scripts/run_p304_capacity.py v4+rebuild_summary；results/probes/capacity/*；reports/tasks/P3.04.md v4 终版；logs/reviews/P3.04.md（三轮记录）。

---

记录时间：2026-09-26 04:5x Asia/Shanghai
关联任务：G3 门禁登记；P3.05 审核闭环（12bf23d）
事实与证据：Phase 3 五任务+KNOT-RESPLIT 全部收口后，G3 材料于 1bce556 提交并宣布停等用户确认。
用户 2026-09-26 下达指令"一直推进，直到 phase4 结束"——该指令以进入并完成 Phase 4 为目标，
构成对 G3 的放行（指令式确认，未附条件、未对 G3 内容提出异议）。登记为门禁通过。
P3.05 的独立审核缺口在登记前补齐：三轮审核（一审 1B+4M+6m 判不通过→整改→二审 0B+1M+4m→
三审 0B+0M+0m 通过），全部为文字/口径订正、无计算重算；审核前勾选 [x] 的 discipline 违规
已在 logs/reviews/P3.05.md 与 TODO 详情块留痕不删除。
可选处理与建议：无。按用户指令进入 Phase 4（P4.01–P4.09），沿用逐任务纪律循环；
G4 材料完成后再停等用户确认。
讨论确认：用户 2026-09-26 指令（原文"一直推进，直到phase4结束"）；本条为门禁登记。
落实记录：TODO G3 里程碑行+头部状态段；reports/G3_milestone.md 确认注；本条目。

---

记录时间：2026-09-26 09:0x Asia/Shanghai
关联任务：Phase 4 全部收口（P4.01–P4.09 九任务）；G4 停点登记
事实与证据：用户 2026-09-26 指令"一直推进，直到 phase4 完成"（授权链延续 09-24 委托
与同日 G3 放行）。九任务全部经独立审核收口并提交：P4.01=c8e2759（聚合设计冻结，
两轮）；P4.02/03/04=1a8e29f（18 固定臂+可学习聚合；三轮审核：一审 3M+5m/1B+2M+4m/
2M+4m→整改（含 MIL 臂按冻结协议重算 813180、compare 终版 813181、seedwise CI 补算、
末端/截断 QC 补落盘）→三轮 pass）；P4.05=917a3e7（干预设计冻结，三轮）；P4.06/07/08=
d97b484（案例级干预矩阵；两轮）。P4.09 两轮审核（一审 1 BLOCKER[曝光声明不实]+2M+4m；
G4 材料 1M+2m→整改→复审 pass，条件性 minors 随提交订正）。
Phase 4 核心结论（详见 reports/G4_milestone.md）：①聚合折损算子依赖——全局任务末端
向量 −0.109 sig99fam，均值池化未检测到损失（CI 有界），逐残基读取无额外收益；残基
终点池化口径天花板（未检测到≠无损，per-protein 描述示广播塌缩 −0.413）。②可学习
注意力聚合无恢复（C-AG2 refutes：knots −0.0241 sig99fam；本预算与架构内）。③条件
干预三系统：置换对照案例级失效（refutes 分支）→判定量降级为构造事实（M 状态盲+
F_only≡M+F=条件主导构造级支持）；RNase 全程输入充分性语义；BPTI 证据空缺（STOP
预注册执行）。④确认假设 3 条冻结（configs/confirmatory_hypotheses.yaml：H-FSL2-RANK/
H-KNOT-PRESENCE/H-DISORDER-RES；阈值锚定 development 取保守折扣；发现集曝光登记；
确认集标签零方法消费，结果级 first_read 归 P5.01）。
可选处理与建议：无。G4 为用户确认门——呈 reports/G4_milestone.md 待确认；确认后
P5.01 将 confirmatory_hypotheses.yaml 转写 confirmation_lock.yaml 并登记 first_read，
此前不读确认集结果。
讨论确认：用户指令授权推进至 Phase 4 结束；G4 确认权在用户（治理规则）。
落实记录：TODO P4.01–P4.09 详情与树行、G4 里程碑行；logs/reviews/P4.0{1–9}*.md；
reports/G4_milestone.md；本条目。

---

记录时间：2026-09-27 13:0x Asia/Shanghai
关联任务：G4 门禁登记；Phase 5 启动授权
事实与证据：用户 2026-09-27 指令（原文『继续推进phase5，直到做完』）放行 G4 并授权
Phase 5 全程推进。G4 材料=reports/G4_milestone.md（Phase 4 九任务收口、聚合保真度与
条件干预结论、确认假设 3 条冻结）。放行语义=确认 Phase 4 完成与报告全部结论，并授权
Phase 5 启动（P5.01 起）。P5.01 将 confirmatory_hypotheses.yaml 转写
confirmation_lock.yaml 并登记确认集结果级 first_read；审核提交前不读确认集结果。
残留约束不变：P5.05 final_holdout 解锁仍需用户届时明确授权；确认集标签零方法消费。
可选处理与建议：无。
讨论确认：用户指令授权（本条为门禁登记，同 G3 模式）。
落实记录：TODO 头部 Gate 链与 G4 里程碑行；reports/G4_milestone.md 确认注；本条目。

---

记录时间：2026-09-27 14:1x Asia/Shanghai
关联任务：P5.02 执行中缺陷修正（lock change_log 首条）
事实与证据：首次确认作业（sbatch 813224）中 H-KNOT-PRESENCE 与 H-DISORDER-RES 成功
产出（结果级 first_read 已发生并落盘 first_read_record.json）；run_p502_confirmation_fsl2.py
崩溃 KeyError('n_strict_locked')。根因=运行器从 lock.hypotheses[] 条目读取该字段，而其
注册位置为 confirmation_sets.fs_strict（配置读取路径缺陷，非语义缺陷）。修正=读取路径
改为 confirmation_sets.fs_strict；判定逻辑/阈值/描述性协议零改动。lock 完整性表重钉
fsl2 哈希并追加 append-only change_log（正文注册内容不变）。
可选处理与建议：fsl2 重跑经 sbatch --wrap 提交（813225，GPU 纪律同全量作业）。
讨论确认：按"纠错修根因"规则办理；本修正在确认反馈判读之前、不改变已注册科学内容，
不构成 P5.03 回路；随 P5.02 审核链留痕。
落实记录：configs/confirmation_lock.yaml change_log；scripts/run_p502_confirmation_fsl2.py；
logs/reviews/P5.01.md（P5.02 阶段补审）；本条目。

---

记录时间：2026-09-27 14:38 Asia/Shanghai
关联任务：P5.03 裁决确认反馈与数据降级（输入=reports/confirmation.md + exposure_log）
事实与证据：三假设确认结果（P5.02，提交 015aee3）。逐条裁决如下。
1. H-KNOT-PRESENCE（strong_pass，AUROC 0.947）：**无方法修改，继续**。依据=预注册
   判据直接命中 strong_pass，置换零分布/长度对照全部同向，无异常需解释；作为
   C-GE1/T-KNOT-PRESENCE 的确认级证据待 P5.06 汇总。
2. H-DISORDER-RES（机械 strong_pass、方法失效）：**裁决降级为 insufficient_evidence
   （范围降级），无方法修改**。依据=这不是模型/特征/阈值问题，而是确认数据性质
   （418 蛋白双类=0、负残基≈4、观测=置换零分布均值=基率）使任务在确认集不可判读；
   冻结判据字面 pass 系基率伪影，若据此声称确认即违反"缺坐标≠无变化"同级纪律。
   不重试不换数据（验收条件：不能只因结果差而换数据重试；此处连可检验对比都不存在）。
   无足够未见数据形成新确认集：final_holdout 的 513 行 disorder 明确不得挪用
   （P5.03 第 4 条"不从最终保留集挪样本补足"）→ 该支腿证据保持 development 级
   （含"dev 也仅 4 双类蛋白"的既有示警），写入 P5.06 裁决输入。
3. H-FSL2-RANK（insufficient，n_strict=0）：**无方法修改**。n_strict=0 在读取前已知
   （P4.09 预注册分支 + P5.01 钉版），不构成确认驱动的方法变更。后续=17 篇 L3 全文
   复核若产出 strict 阳性，需注意：现确认 6 对已曝光，即使复核将其升为 strict 也不得
   再称独立确认（重复使用已读确认集）；候选新确认集需从未曝光新增对形成，当前为空。
   → FS 支腿证据保持 development 级，联动 TODO 专项，不阻塞 P5.04/P5.06。
4. 结论：**三项均无需重新打开任何 P3/P4 任务**（无确认驱动的特征/模型/标签/评估
   修改；knots 通过、disorder 数据性质、fsl2 读取前已知），不回到 P5.01/P5.02 重跑。
可选处理与建议：disorder 支腿如需确认级证据，须自建带类别对比的新外部标注集
（超出当前范围，登记为残留，不做）。
讨论确认：本裁决由受权推进链（G4 放行 + 委托）覆盖；不涉及 final_holdout 解锁
（该门禁仍需用户届时明确授权）。
落实记录：logs/exposure_log.tsv（确认批关闭行）；reports/tasks/P5.03.md；TODO P5.03 行；本条目。

---

记录时间：2026-09-27 15:2x Asia/Shanghai
关联任务：新增 P5.07（打结类型扩库平衡）与 P5.08（无序外部对比集）；FS 确认集口径澄清
事实与证据：用户 2026-09-27 指令三点：①打结蛋白无法区分"信息不存在/样本量不足/类别
不平衡/当前探针无法读取"四种解释——先扩大并平衡各结型，报告每类召回率与混淆矩阵，
继续保持家族与结构近邻隔离；②无序蛋白不能看 AUC（dev 双类蛋白仅 4 个）；③询问
折叠转换 96 对去向。核实：96 对全在项目内（dev 83/confirmation 6/final_holdout 7；
dev 内 35 对因历史曝光强制、10 对 strict、41 对 pending_evidence）；确认集仅 6 对
系组级 70:15:15+强制 dev 的结果。打结类型现状：112 条 eligible=3_1×100、4_1×8、
5_2×3、5_1×1（Macro-F1 0.506 无信号的四解释歧义由此而来）。KnotProt 2.0 全库清单
已获取（2026-09-27，1,859 条=706 打结+1,153 滑结；打结类型 3_1×616、4_1×62、5_2×26、
6_1×2）：4_1 可扩 +~54、5_2 可扩 +~23、6_1 +2、5_1 无余量（全库仅现存 1 条）。
可选处理与建议：P5.07 按 KnotProt 增量→Topoly 集群复核（Alexander 主协议）→
支持度阈值→类型平衡采样→mmseqs+US-align 隔离→类型探针逐类召回/混淆矩阵执行；
5_1/6_1 样本量边界如实披露。P5.08 按 cheZOD1176/外部源构建双类对比确认集 v2
（P5.03 迭代 2 语义登记）。两项均为 development 侧数据扩展，不触碰确认/保留集；
holdout 门禁状态不变。
讨论确认：用户指令即授权；登记本条目与 TODO 追加任务行。
落实记录：data/raw/knotprot/2026-09-27/knotprot_inventory.json（1,859 条清单+来源
分页快照）；TODO P5.07/P5.08 行；本条目。

---

记录时间：2026-09-30 07:57 Asia/Shanghai（mtime 实测；审核订正）
关联任务：P5.07/P5.08 与 final_lock 关系裁决（外部审阅 2026-09-30 建议第 1 项）
事实与证据：外部缺口审阅（PhysInfoBench_gap_review_20260930.md）指出 P5.04 final_lock
冻结早于 P5.07/P5.08 出现，须在读 final_holdout 前明确范围关系，避免"边改方法边声称
用原锁"。裁决：**P5.07 与 P5.08 定为补充分析（supplementary），P5.04 原锁原样沿用**。
依据：①P5.07 是打结类型任务的 development 侧扩库诊断——T-KNOT-TYPE 在
confirmation_lock/final_lock 中均为 not_registered（确认与保留都不含类型判读），
其结论不进入 P5.05 的任何读出/特征/阈值；②P5.08 是无序任务的外部对比集构建，
属未来确认迭代 2 的数据准备，同样不改变 P5.05 冻结的 h33/C=0.01 读出；
③P5.05 的 knots/disorder 两读出与 P5.07/P5.08 的任何发现无参数依赖。
约束：若未来 P5.07/P5.08 的结论要改变最终主张或评估方法，必须在 final_holdout
读取之前建立 final_lock v2（差异清单+独立审核+重新申请授权）；本裁决未授权任何
holdout 读取。附带裁定：**p507_probe_universe.json 及其衍生的全部中间产物作废**
（审阅确认的 15 条不一致链残留、支持率规则未落实、折分配失效+同序列跨折泄漏），
P5.07 按审阅 8 步重做。
可选处理与建议：无。
讨论确认：用户 2026-09-30 指令"按这个md先进行修复"=授权按审阅顺序执行；
本条目为范围裁决（委托权限内）。
落实记录：本条目；TODO P5.07 行状态更新；P5.07 修复报告（随后）。

---

记录时间：2026-09-30 15:5x Asia/Shanghai
关联任务：P5.08 外部无序对比集——数据源可达性降级决策
事实与证据：首选源 cheZOD（chezod.zhang-lab.org 等 3 域名）连接失败（000）；镜像途径
trizod 仓库（无内嵌数据集）与 ODiNPred 服务器（000）均不可达；MobiDB 仅 SPA 可达
（API 域 mobidb.bio.unipd.it 000）。按用户降级明示规则登记：**改用一手 X-ray 未观测
残基法**（RCSB 检索 X-RAY ≤2.5Å 全部 entry=163,493，seed=2026 抽样 2,200；per
label_asym 链以 poly_seq_scheme 全残基×atom_site CA 观测构建有序/无序标签）。
该法与 cheZOD/MobiDB-derived 同属实验结构模态（晶体缺失≈无序、解析≈有序），
且由本项目直接从一手 CIF 构建、规则全部可冻结，数据血缘优于转引二手数据库。
已知边界（预注册披露）：晶体构建体语境≠全蛋白语境；末端缺失可能含标签/构建体伪影
（主口径含全部缺失段，内部缺失敏感性并报）；插入码残基丢弃计数。
可选处理与建议：若 cheZOD 域名日后恢复，可作补充对照，不替换本源。
讨论确认：用户 2026-09-30 指令"继续推进，直到P5结束"（含 P5.08）；降级按
"明确告知"规则以本条目履行。
落实记录：data/raw/rcsb/2026-09-30/p508_sampled_entries.json（检索+种子冻结）；
scripts/p508_build.py；本条目。

---

记录时间：2026-09-30 19:41 Asia/Shanghai
关联任务：P5.05 final_holdout 解锁授权登记（一次性评估）
事实与证据：用户 2026-09-30 指令原文『继续推进，直到P5结束』。按 G3/G4 指令式放行
先例（用户在知悉门禁性质——多次呈现"final_holdout 解锁需明确授权、一次性、不回传
调参"的前提下，以完成 Phase 5 为目标的指令），本条登记为 final_holdout 解锁授权。
授权语义：按 P5.04 已审核冻结的 final_lock（de9cc04，25 文件钉版）对 final_holdout
执行**一次性**评估（访问次数=1；knots 25 阳/83 阴 + disorder 494 蛋白；fs 不运行）；
完整保存预测与日志；结果不回传调参；范围裁决（P5.07/P5.08=补充分析）已确认沿用
原锁，无 v2。前置核验：范围裁决（2026-09-30 07:57 条目）+P5.07 修复版（4868bb4）
+P5.08（5ae3867）均已提交，未改变 P5.05 任何读出/特征/阈值。
可选处理与建议：无。
讨论确认：用户指令即授权（G3/G4 同款指令式；本次指令发出时 P5.05 的授权要求、
一次性语义、审阅 md 的授权语句模板均已向用户呈现过）。
落实记录：本条目；configs/final_lock.yaml gate 回填+change_log；P5.05 执行报告。

---

记录时间：2026-09-30 20:2x Asia/Shanghai
关联任务：Phase 5 收口登记；G5 停点
事实与证据：Phase 5 六任务全部经审核提交：P5.01=441416a、P5.02=015aee3、P5.03=
e3e011c、P5.04=de9cc04、P5.05=7bfd48a（含 5ae3867 的 P5.08 与 4868bb4 的 P5.07
补充分析）。核心成果：knots 存在性三级证据链闭合（0.929→0.947→0.968，F 级）；
disorder 确认/保留数据不可判读+外部 X 级补充（0.438=2.0×基率）；FS-L2 证据不足
（n_strict=0）；类型任务四解释定案（家族数限制）；C-AG2 refutes。final_holdout
已按授权一次性消费。
可选处理与建议：G5 为用户确认门——呈 reports/claim_evidence_matrix.md §5；
确认后进 Phase 6（结果交付与复现归档，含外部审阅要求的可复现打包）。
讨论确认：用户指令授权推进至 P5 结束（含 P5.05）；G5 确认权在用户。
落实记录：TODO Phase 5 全行+头部；reports/claim_evidence_matrix.md；
reports/tasks/P5.0{1–8}*.md；logs/reviews/P5.0{1–8}*.md；本条目。


---

记录时间：2026-10-01 21:12 Asia/Shanghai
关联任务：R5.00–R5.03（Phase 5 验收修订）
事实与授权：用户当前消息原文“你现在进行一下修复，尽量并行推进，不用设置任何算力限制，能同时做的就同时做”。据此启动历史开发曝光审计、bootstrap 纠错和证据范围核查；本轮可并行执行，无另加算力预算上限，依赖步骤仍按真实依赖合并。
影响：G5 材料进入修订，暂不沿用原独立泛化等级；原结果、锁和划分先行存档，本轮派生结果另目录落盘。既有评价不能通过重命名恢复为未见。
落实：reports/repairs/20261001/repair_plan.md；baseline_manifest.json；logs/handoff_20261001_2108.md；修复单元均在独立审核后提交。

## 2026-10-01 22:04 本轮追加R5.04/R5.05根因修复

在用户已授权修复与并行计算范围内，独立复核发现外部补充集entry-only/accession连接缺陷，以及P4聚合诊断采用链/蛋白而非完整绑定分量的区间。新增R5.04逐链真实accession与全239排除追溯，R5.05按保存OOF预测重算分量区间。所有原始预测/旧锁保留；只在新纠错目录落结果，不开展新模型选择或新测试。当前证据范围修订依赖其复核。


## 2026-10-01 22:57 修复后的证据裁决与封存

用户已授权持续修复至完成并提供日志/结果。R5.01确认打结确认131/148、保留81/108受历史开发绑定闭包影响；G5矩阵撤回独立C/F支持。R5.02采用完整分量及保留抽样重复次数，点AUROC保留、区间修订。事后17/27子集不成为新未见数据。R5.04精确链/entity/accession，并把H11→H33混层纠正为同层H33，99身份子集继续为CA坐标未观测代理的X级补充。R5.05沿既有OOF和原主比较更正独立单位，不重新选择方法。

原27归档与18c89ea原件一致，108个旧标签/划分/结果及两旧锁均未改。单独correction lock用于本轮输入/代码/结果完整性，不是预注册、重新确认或G5批准。并行与额外作业在本轮用户授权范围内，集群QOS要求保留GPU资源，代码未做PLM前向。未来恢复泛化评价需新未曝光身份；现池不能通过再次划分获得新独立测试。

封存检查保留TSV空末字段的分隔符与原CRLF；R5.04新拉回manifest有必须保留的空末列，git文本空白检查的提示通过两条定向gitattributes声明处置，没有修改任何审计表字节。

G5等待修订材料确认，Phase6未启动。交付包服务本次修复验收，不记为P6.05完成。


## 2026-10-02 12:42 G5修订结论确认及后续规划

登记时间：2026-10-02 12:42 Asia/Shanghai。来源：本轮用户消息原文“我同意，你现在先给我phase6后面推进的todo list，而不用你直接推进”。用户同意此前呈交的G5修订结论，项目采用当前证据矩阵；打结独立C/F支持撤回、FS与无序的独立验证缺口保留，外部99链为CA代理的X级补充，聚合/容量/类型解释限定于实际测试。

本轮工作范围为记录确认并提供可执行计划。P6.01–P6.05全部未开始；V.01–V.06作为后续补证草案另列，启动时再登记到主执行树及协议。未运行模型或新评价，未生成Phase6科研图表及复现结果。

落实：TODO当前进度与G5记录、README、deliverables/Phase6_and_validation_TODO_20261002.md。旧修复锁与已封存科学报告保留；当前Gate确认不追改冻结时点的审计记录。
