# 历史曝光范围审计（P1.19；G1 材料修订）

时间：2026-09-23 21:15–21:22 Asia/Shanghai。指令来源：用户 2026-09-23 21:05 第 1 项（把"旧 v5 划分表收录 35/96 对"与"项目历史曝光"分开登记）。性质：**只读审计**——不写任何旧项目文件，不改任何冻结配置/资格表。

## 1. 口径定义（预先注册，四层分开）

| 口径 | 定义 | 载体 |
|---|---|---|
| S1 v5 划分收录 | 在 task_sample_splits_v5_2026-09-15.tsv 有 dev/test 分配 | 旧 manifests/ |
| S2 全版本划分收录 | 在 v1–v5 任一版本有分配 | task_sample_splits{,_v2..v5} |
| S3 前项目 legacy 登记 | benchmark_v0.1/samples.tsv legacy_exposure 列；端点级=legacy_exposure.tsv 的 fold_switching 行 | v0.1 发布包 |
| S4 旧项目产物使用 | 出现在登记/复核计算/规则记录/试算/纠错/报告/交付文档/参考文献/日志（"出现"≠评价使用，也≠影响方法选择） | 918 个文本文件 |
| S5 评价使用 | 任何正式模型评价读取 | acceptance_report.md（核验） |

## 2. 方法

- 名单：新项目 data/curated/fold_switch_global.tsv 全 96 对（tier 分布 strict 10/pending 53/extension 27/fragment 4/excluded 2）。
- 结构化解析：v1–v5 划分表（porter 样本行）；samples.tsv（96 行 fold_switching_pair）；legacy_exposure.tsv（category=fold_switching 的端点 (pdb,chain) 行）；trial10 表；data_corrections json；v0.1 splits.tsv。
- 内容扫描：预登记载体清单（见脚本 CARRIERS，918 个文本文件、约 33 MB），键匹配=精确 pair_id / `porter_N` 词边界 / 端点 PDB 码词边界（大小写不敏感）；named 命中（对名/对号）与 pdb-only 命中分列。二进制/结构文件不扫内容；DSSP 输出仅按文件名（dssp_local* 仅 2 文件：1fzp、2a01，覆盖 2 对各 1/2 端点；批量大头在 batches/ 未按名展开，由 computation_review 汇总表代表）。
- 脚本：scripts/audit_exposure_scope.py（退出码 0；断言：96 对名单完整、v5=35、trial10 键⊆名单、acceptance_report 含 formal_model_evaluation=false 与 locked_split=false；corrections json 键为过滤登记而非断言——不在名单的键会被忽略，实测两键均在名单内）。

## 3. 结果

### 3.1 划分收录（S1/S2）
- 各版本收录数：v1=32、v2=32、v3=34、v4=34、v5=35；**五版本并集=35**（只增不减，无"早期收录后被移出"的对）。
- 61 对**从未进入任何划分表**（never_in_split_tables）：53 pending_evidence + 4 fragment_only + 2 excluded + 2 extension（1 construct_difference + 1 condition_or_assembly）。与 P1.17 的 61 对口径一致。
- v5 35 对构成：10 strict + 21 extension_construct_difference + 4 extension_condition_or_assembly；dev/test 三分口径（26 dev-only/7 test-only/2 both）与 P1.17 复算一致。
- v0.1 发布包 splits.tsv：porter 96 行**全部 unassigned_not_frozen**（"benchmark freeze conditions not met"）——发布包自身从未做集合分配。

### 3.2 前项目 legacy 登记（S3）
- samples.tsv：**96/96 对 legacy_exposure=yes**，唯一来源 `91_ESM-Latent-Topology_On-Hold/scripts/saprot_control/data/manifest.csv`（前项目 SaProt 控制数据清单；legacy_exposure.tsv 共 1275 行=anchor 856+fold_switching 133+knotted 286）。
- 端点级回指：**89/96 对**至少 1 个端点在 legacy_exposure.tsv 有 fold_switching 行；**7 对 0/2**：porter_6/16/37/71/78/86/89（其中 porter_71 的 2pbk 以 anchor 类别在册）。
- **不一致登记**（不裁决哪张表为准）：samples.tsv 对这 7 对仍标 yes，但端点登记表查无其 fold_switching 行（除 2pbk anchor 行）。旧包未给出 samples 层 yes 的判定规则文件。

### 3.3 旧项目产物使用（S4，named 命中对数/96）
| 载体类别 | named 命中 | 说明 |
|---|---|---|
| registration（资格/登记表 5 文件） | 96 | fold_switching_pair_audit、release_eligibility×4 版本 |
| rule_records | 96 | eligibility_rule_reconciliation_2026-09-13.tsv（全 96 对逐对调和） |
| v01_release（发布包） | 96 | samples/fold_switching_pairs/protein_groups/region_mapping 等 16 文件 |
| exec_round（执行轮快照/复核） | 96 | baseline/snapshot/manifests、structure/recomputed_batch02 等 24 文件 |
| logs_old | 68 | 映射/并行轮调和 json |
| reports_root（RUN_LOG 等 10 md） | 38 | RUN_LOG 全程记载（porter_1/5/32/35/36/38/54/81/92 等读文轮次） |
| references（证据深挖） | **10=恰好全部 strict** | references_case（porter_62 逐案）/references_batch（usalign 对照等） |
| pilot（trial10） | 10 | rank1–10：porter_20/30/26/9/43/62/66/80/83/87；现行 tier=5 strict+3 extension（porter_30 construct+porter_43/83 condition）+2 excluded（porter_26/66） |
| deliverables（治理文档） | 7 | porter_8/20/32/35/36/54/55（裁决/纠错记录） |
| computation_review | 2 named / 96 pdb-only | 汇总表按端点键而非对名 |
| sweep（calcineurin） | 端点 0 命中 | 见 3.4 |

- trial10=10 对流程试算（US-align/DSSP/证据审计/决策表）；data_corrections=porter_8、porter_55 两对纠错。

### 3.4 calcineurin sweep 归属（更正性发现）
manifests/calcineurin_proline_state_sweep_2026-09-15.tsv **不含 porter_20 的端点 5c1v（0 命中）**；按 RUN_LOG.md L222，该 sweep（其余 Q08209 相关条目共 27 个链行，含 PI4KA 复合物链；RUN_LOG 记 "27 chains"）是 **porter_20（5c1v A/B cis/trans）状态核证**的一部分：A 端据 9b9g 独立结构判 verified_same_state，B 端 no_independent_reference_exists。即：这是**以同蛋白其他 PDB 为载体的一对一方法验证使用**，未含该对端点坐标本身。

### 3.5 评价使用（S5）
acceptance_report.md L176–177：`locked_split=false`、`formal_model_evaluation=false`。**96 对中没有任何一对被正式模型评价读取过。**

## 4. 可陈述与不可陈述（结论边界）

可陈述：
1. 35 对在旧 v5 划分表有分配（S1）；61 对不在 v5、也不在 v1–v4（S2）。
2. 96 对全部有 legacy_exposure=yes 登记（S3；89 对端点级可回指，7 对为旧包内部不一致）。
3. 96 对全部出现在旧项目数据构建/复核产物中（S4）；其中 trial10 10 对、纠错 2 对、porter_20 状态核证 sweep、读文轮次若干对。
4. 从未发生正式模型评价（S5）。

不可陈述：
1. **61 对（53 pending）是"可靠未曝光备用确认池"**——它们有 legacy=yes 登记（96/96 层）且全部在数据构建产物中出现过；legacy 层（前项目 SaProt 控制清单收录）是否计为损害评价完整性的曝光，属 G1 裁决事项，本审计不代裁。
2. "96 对全部在旧 v5 划分表曝光"（P1.09 原表述，失实，维持 P1.17 修正）。
3. 任何"旧项目已评价/已冻结划分"的暗示。

## 5. 本项目 exposure_log 补登记（3 行，真实时间戳+标注）

读取发生时间与登记时间分离，notes 列注明"补登记"；均 influenced_method_selection=false（本项目至今无任何模型/方法选择实验）：
1. 2026-09-21 12:39（P1.02）：导入旧资格表+round2 证据（96 对全量，数据血缘见 P1.02 完成记录）。
2. 2026-09-23 18:38（P1.17）：读取 task_sample_splits_v5（35 对分配，预演曝光口径核查）。
3. 2026-09-23 21:15（P1.19）：旧项目 918 文件使用范围审计（本任务）。

## 6. G1 材料修订差异（本任务落实部分）

- G1 第 5 项（备用确认池）：事实证据改为"53 个 pending 对不在任何旧划分表（v1–v5 并集=35）且从未被评价；但 96 对全部有 legacy=yes 登记（89 端点可回指+7 对旧包不一致）且全部见于旧项目数据构建产物"；建议维持"暂无可靠备用池"，并新增"legacy 层是否计为曝光=需用户裁决"。
- G1 第 9 项（D6）：事实证据并入四口径数字（S1=35/96、S3=96/96、S5=0）；"10 strict 全曝光"改为"10 strict 全部 v5 收录且全部 legacy=yes"。
- 重点建议 5 与附录 A5 第 5 条同步改写；全部措辞见 G1_fold_switch_decision.md 本日修订。

## 7. 残留与移交
- 7 对 legacy 0/2 不一致：登记不裁决；若 G1 需要 legacy 层精确口径，需回旧包构建脚本查 samples 层 yes 的判定规则（超出本任务）。
- references 命中=10 strict 恰好重合：旧项目证据深挖与 strict 名单同源（trial10+读文轮次），不构成新的曝光层，但写入 G1 第 1 项证据语境（strict 层在旧项目被反复人工处理）。
