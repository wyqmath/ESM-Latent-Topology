基于三层设计，现有 TODO 需要增加一个明确的“折叠转换数据重构工作包”。它应当放在 Phase 1 数据阶段完成，并把结果传递给 Phase 2 的分组、匹配与数据划分。

当前已经完成的 P1.02 和 P1.03 保持完成状态。三层设计属于对任务合同的细化，不能通过重开 P1.02、修改历史完成记录来实现。正确做法是记录一次治理变更，新增 P1.10–P1.12，并更新 Phase 2–5 中相关任务的输入和验收条件。

现有主文件是 [TODO.md](/Users/yuan/Documents/ChatGPT/PLM/TODO.md)，关键上游数据是：

- [fold_switch_global.tsv](/Users/yuan/Documents/ChatGPT/PLM/data/curated/fold_switch_global.tsv)
- [fold_switch_regions.tsv](/Users/yuan/Documents/ChatGPT/PLM/data/curated/fold_switch_regions.tsv)
- [claims.yaml](/Users/yuan/Documents/ChatGPT/PLM/configs/claims.yaml)
- [evaluation_protocol.yaml](/Users/yuan/Documents/ChatGPT/PLM/configs/evaluation_protocol.yaml)
- [split_protocol.yaml](/Users/yuan/Documents/ChatGPT/PLM/configs/split_protocol.yaml)
- [sample_schema.yaml](/Users/yuan/Documents/ChatGPT/PLM/configs/sample_schema.yaml)

## 一、要加入 TODO 的工作包

### 治理变更：登记折叠转换三层设计

这一步先于新增科研任务执行，只修改计划、协议和决策记录。

具体操作：

1. 在 `logs/decisions.md` 记录三层设计的依据、影响范围和结论边界。
2. 在 `configs/claims.yaml` 中保留 `T-FS-GLOBAL` 作为总任务，并在其下新增三个子任务：
   - `T-FS-L1-PAIRED`：严格双态阳性内部的机制诊断；
   - `T-FS-L2-PU-RANK`：严格阳性在未标注背景中的排序；
   - `T-FS-L3-MATCHED`：严格阳性与推定阴性的匹配比较。
3. 保留现有 `T-FS-REGION`，并声明它属于第一层残基级任务。
4. 将 `OD-NEGATIVE-FS` 状态更新为：
   - `direction_approved`
   - `implementation_pending`
   - 在 P1.10–P1.12 完成并经 G1 确认后才标记 resolved。
5. 更新 `configs/evaluation_protocol.yaml`，为三层分别定义指标。
6. 更新 `configs/sample_schema.yaml`，加入证据状态字段。
7. 更新 `configs/split_protocol.yaml`，加入未标注背景、匹配集合和历史曝光规则。
8. 在 `TODO.md` 的 Phase 1 中新增 P1.10–P1.12。
9. 使用独立 subagent 审核计划修改。
10. 单独提交一次治理变更，例如：

```text
PLAN register three-layer fold-switch workflow
```

本次计划提交不将任何新增科研任务标为完成。

### P1.09 完成数据质量与可行性审计

现有 P1.09 保留，同时增加折叠转换专项审计。

具体工具：

- Python；
- `pandas` 或 `polars`：读取资格表、区域表和证据表；
- `pyarrow`：大表持久化；
- `Biopython`：序列校验与构建体对齐；
- `gemmi`：PDB/mmCIF 链、残基编号和缺失位置核查；
- YAML 解析器：检查协议可执行性；
- Git 和 SHA-256：版本留痕。

具体执行：

1. 对10个 strict positive 建立逐对审计表。
2. 核对：
   - 两端 UniProt；
   - 完整序列；
   - 构建体差异；
   - 突变；
   - PDB chain；
   - 状态名称；
   - 环境或组装条件；
   - 原始论文证据位置；
   - 作者标注区域；
   - 可评价残基掩码。
3. 抽查依赖旧审计的8个 strict positive，确认其证据指针仍可回到原始材料。
4. 处理 porter_8 状态冲突。
5. 给每个阳性分配 `positive_evidence_tier`。
6. 按 UniProt、序列 hash、家族候选和结构簇统计独立蛋白与独立组数。
7. 输出三层设计的初步可行性：
   - 严格阳性数；
   - extension 数；
   - 独立家族候选数；
   - 可用于残基级任务的对数；
   - 可用于条件状态分析的对数；
   - 历史曝光情况。

交付物：

```text
reports/fs_three_layer/strict_positive_audit.tsv
reports/fs_three_layer/strict_positive_qc.json
reports/fs_three_layer/strict_positive_feasibility.md
```

验收条件：

- 每个 strict positive 都有可解析的证据指针；
- 同序列、构建体差异和条件差异分列；
- 证据缺口不会被空值静默隐藏；
- 独立蛋白数与结构记录数分别统计；
- 无法支持严格阳性的样本降级至 extension 或 pending，并保留改判记录。

### P1.10 建立未标注背景

这一任务生成第二层所需的背景集合。建议建立两个背景宇宙。

#### 背景 A：序列背景

用于纯序列模型的 PU／排序分析。

推荐来源：

- UniProtKB/Swiss-Prot reviewed release；
- 或在资源受限时，从固定版本的 Swiss-Prot 中按预注册规则抽样。

推荐工具：

- UniProt REST API 或官方 release 文件；
- `requests` 或 `httpx`；
- `Biopython.SeqIO`；
- `duckdb` 或 `polars`；
- `sha256sum`；
- `MMseqs2` 在 Phase 2 进行聚类。

必须保存：

- UniProt accession；
- 序列；
- sequence hash；
- 蛋白长度；
- 物种；
- reviewed 状态；
- annotation score；
- PDB cross-reference；
- 数据库 release；
- 下载日期；
- strict positive 排除状态；
- 历史曝光状态。

#### 背景 B：结构背景

用于结构感知模型和推定阴性候选筛选。

推荐来源：

- RCSB PDB 实验结构；
- PDBe SIFTS 的 PDB-chain → UniProt 映射；
- CATH、SCOPe 和 Foldseek 聚类作为结构／家族信息。

推荐工具：

- RCSB Search API；
- RCSB Data API；
- PDBe/SIFTS；
- `gemmi`；
- `Foldseek`；
- `US-align`；
- 现有旧项目的 family、CATH、SCOPe 和 Foldseek 产物。

数据规则：

1. 从固定 release 构建背景。
2. 对蛋白身份去重，结构记录保留为子表。
3. strict positive 和 extension 样本不能混入背景。
4. 背景标签使用：
   - `label_epistemic_status=unlabeled`
   - `biological_target` 为空；
   - `pu_observed_label=0` 只表示“当前未被确认为阳性”，不能存入真实 target 字段。
5. 保存背景纳入和排除原因。
6. 保存背景中每个蛋白的结构数量、实验方法和结构覆盖信息。

脚本与交付物：

```text
configs/fold_switch_background.yaml
scripts/build_fs_unlabeled_background.py
data/curated/fold_switch_unlabeled_sequence.tsv
data/curated/fold_switch_unlabeled_structure.tsv
data/manifests/fold_switch_background_sources.tsv
reports/tasks/P1.10.md
logs/reviews/P1.10.md
```

脚本至少支持：

```text
--config
--source-manifest
--strict-positive-table
--output-dir
--run-id
```

质量检查：

- strict positive 与背景交集为0；
- extension 与背景交集为0或按协议明确标记；
- exact sequence duplicate 有统一处理；
- 非法氨基酸与异常长度有排除记录；
- 每行都能追溯至数据库 release；
- `biological_target=0` 的行数必须为0；
- 所有 unlabeled 行的 `pu_observed_label=0`；
- 输出稳定、可复跑且带校验值。

### P1.11 建立推定阴性候选池

这一任务筛选第三层的候选对照，暂不生成最终匹配关系。

候选来源为 P1.10 的结构背景。

推荐工具：

- RCSB Search/Data API：获取结构、实验方法、分辨率、配体和组装信息；
- PDBe/SIFTS：确认 UniProt 与构建体范围；
- Europe PMC API：检索原始论文；
- Crossref：补 DOI 和书目信息；
- CATH、SCOPe、Foldseek：建立结构相似性与家族候选；
- `gemmi`：核查结构链；
- `pandas`／`polars`：候选表；
- `requests-cache`：保存 API 查询结果；
- `rapidfuzz`：论文题名与蛋白名称去重；
- 原始查询 JSON 和响应 SHA-256：可追溯性。

为每个候选记录：

- UniProt accession；
- sequence hash；
- PDB 数量；
- 独立研究数量；
- 独立 DOI 数量；
- 结构实验方法；
- 结构覆盖率；
- 配体条件数量；
- 组装状态数量；
- 是否存在不同构象描述；
- 是否命中 metamorphic、fold switching、alternative fold、conformational switch、domain swapping 等检索词；
- 文献检索式；
- 检索日期；
- 人工审核状态；
- 与哪个 strict positive 构成候选匹配；
- 家族、长度和结构相似性信息。

建议定义三个证据层：

```text
PN-A：多项独立结构或多条件覆盖，系统文献检索未发现替代折叠证据
PN-B：有多个结构记录，实验和条件覆盖有限
PN-C：仅完成长度、家族或结构类型匹配，研究覆盖较弱
```

PN-A、PN-B、PN-C 表示证据覆盖程度，不表示生物学上已经证明不会转换。

输出：

```text
configs/fold_switch_putative_negative.yaml
scripts/build_fs_putative_negative_candidates.py
data/curated/fold_switch_putative_negative_candidates.tsv
data/evidence/fold_switch_negative_literature_queries.tsv
data/evidence/fold_switch_negative_literature_hits.tsv
reports/tasks/P1.11.md
logs/reviews/P1.11.md
```

质量检查：

- 每个候选都有明确来源和查询日期；
- “无检索命中”与“完成全文审查”分开；
- 同一研究中的多个 PDB 不计作多个独立实验；
- domain swapping、局部开合和真正的替代折叠分开；
- 研究覆盖不足的候选进入 PN-C；
- 候选阶段不写 `target=0`；
- 与 strict positive 同一蛋白或同一序列的候选自动排除；
- 文献命中包含折叠转换证据的候选退出推定阴性池并进入复核队列。

### P1.12 冻结三层任务合同与 G1 可行性报告

这一任务把前三项产物合并成正式任务设计。

工具：

- Python／pandas；
- YAML；
- Jinja2 或普通 Markdown 模板；
- `jsonschema`；
- Git diff；
- 独立 subagent 审核。

创建：

```text
configs/fold_switch_three_layer_protocol.yaml
reports/fs_three_layer/feasibility_matrix.tsv
reports/fs_three_layer/G1_fold_switch_decision.md
reports/tasks/P1.12.md
logs/reviews/P1.12.md
```

`fold_switch_three_layer_protocol.yaml` 至少包含：

```yaml
layer_1:
  task_id: T-FS-L1-PAIRED
  population: strict positive and registered extension strata
  unit: protein or paired state group
  inputs:
    - sequence representation
    - state condition where applicable
    - residue mapping and valid mask
  outputs:
    - layer readability
    - condition contribution
    - region localization
  primary_claim_boundary: within-confirmed-positive mechanism analysis

layer_2:
  task_id: T-FS-L2-PU-RANK
  positives: strict positive only
  background: unlabeled
  biological_target_for_background: null
  observed_label:
    positive: 1
    unlabeled: 0
  metrics:
    - recall_at_k
    - enrichment_at_k
    - positive_rank_percentile
    - family_stratified_enrichment
    - leave_family_out_enrichment
  claim_boundary: enrichment of known positives in a fixed unlabeled universe

layer_3:
  task_id: T-FS-L3-MATCHED
  cases: strict positive
  controls:
    - PN-A
    - PN-B
    - PN-C sensitivity only
  matching:
    status: pending_P2.01
    variables:
      - family_or_structure_group
      - sequence_length
      - experimental_structure_count
      - independent_study_count
      - structure_coverage
      - method_and_resolution
      - oligomeric_state
  metrics:
    - AUROC
    - AUPRC
    - balanced_accuracy
    - MCC
    - per_class_precision_recall
  calibration:
    status: disabled_unless_population_prevalence_correction_is_registered
  claim_boundary: discrimination between confirmed cases and operationally defined matched controls
```

可行性报告应明确列出：

- strict positive 独立蛋白数；
- strict positive 独立家族数；
- extension 各层数量；
-可用于残基级分析的对数；
-序列背景规模；
-结构背景规模；
-PN-A／PN-B／PN-C候选数；
-每个 strict positive 可以匹配到多少候选；
-历史曝光情况；
-哪些任务可以进入三集合划分；
-哪些任务只能做描述性分析；
-哪些任务需要缩小结论范围。

G1 冻结事项：

1. 三层任务定义；
2. 未标注背景来源；
3. 推定阴性证据层；
4. 匹配变量；
5. 主结果与敏感性分析的分工；
6. 样本不足时的降级规则。

### P2.01 增加分组和匹配步骤

更新现有 P2.01，使用以下工具：

- `MMseqs2`：序列相似关系；
- `GraphPart`：带标签和分组约束的序列划分候选；
- `Foldseek`：结构相似关系；
- `US-align`：重点样本结构复核；
- `networkx`：连通分量和不可拆分组；
- `scipy.optimize.linear_sum_assignment`：一对多匹配的最小成本分配；
- `pandas`／`numpy`：协变量标准化和匹配平衡；
- `statsmodels`：标准化均值差等平衡统计。

P2.01 新增步骤：

1. 将同一蛋白、相同 sequence hash、状态配对、近同源和结构近邻合并为不可拆分组。
2. 将 strict positive 与候选对照建立匹配成本。
3. 匹配变量只使用预先登记的样本属性，不使用模型性能。
4. 比较1:1、1:3和1:5候选匹配的可行性。
5. 输出每个匹配版本的：
   - 匹配成功率；
   - 独立组数；
   - 长度平衡；
   - 结构数量平衡；
   - 独立研究数量平衡；
   - 家族覆盖；
   - PN等级分布；
   - 无法匹配的 strict positive。
6. 在查看模型结果前冻结主匹配版本。
7. `matched_set_id` 必须成为不可跨 split 的绑定关系。

新增交付物：

```text
data/splits/fs_group_edges.tsv
data/curated/fold_switch_matched_controls.tsv
reports/fs_three_layer/matching_balance.md
reports/fs_three_layer/matching_balance.tsv
```

### P2.02 增加三层划分规则

第二层和第三层分别生成划分。

第二层：

- strict positive 及其同源／结构近邻必须在同一集合；
- unlabeled 背景按组分配；
- 历史用于方法选择的蛋白不能重新包装成未见确认或保留样本；
- 留出家族外测试时，整个家族进入同一集合。

第三层：

- 一个 strict positive 和其匹配的 controls 共享 `matched_set_id`；
- 整个 matched set 进入同一集合；
- 每个集合报告 strict positive 数、控制数、独立家族数和PN等级；
- 某集合无法同时包含病例和对照时，任务降级，不通过拆分同源组凑数量。

### P2.06 增加数据偏差基线

在提取 PLM 表示之前运行轻量基线，检查数据是否可以被简单混杂解释。

工具：

- scikit-learn；
- statsmodels；
- pandas；
- SciPy bootstrap。

输入特征：

- 序列长度；
- 氨基酸组成；
- 低复杂度比例；
- 结构记录数量；
- 独立论文数量；
- 实验方法；
- 结构分辨率；
- 配体和组装记录数量；
- 家族或结构簇。

第二层基线：

- 随机排序；
- 长度排序；
- 研究覆盖度排序；
- 逻辑回归 observed-positive vs unlabeled；
- bagging PU 基线；
- family-stratified ranking。

第三层基线：

- 匹配变量逻辑回归；
- 条件逻辑回归或 matched-set 分层分析；
- 与随机背景和PN-A／PN-B分别比较。

关键判据：

- 简单元数据基线性能很高时，优先解释为数据收录偏差；
- PLM 相对轻量基线的增益在共同样本上计算；
- case-control 数据的类别比例由设计产生，ECE和发生概率解释保持关闭；
- 所有指标按独立蛋白或组进行 bootstrap。

### P3–P5 的模型与结论更新

P3 的探针实验要分别生成：

- 第一层：层级可读性、条件贡献和区域定位；
- 第二层：strict positive 排序及家族外富集；
- 第三层：匹配病例—对照区分能力。

模型结果必须与以下基线比较：

- 长度与组成；
- 家族或结构簇；
- 研究覆盖；
- F-only；
- 匹配协变量。

P5 最终证据矩阵应分开裁决：

```text
E1：在确认阳性内部是否存在可读状态／区域信号
E2：这些信号是否能在未标注背景中富集已知阳性
E3：这些信号是否能在匹配对照中保持区分能力
E4：结果是否在未见蛋白和未见家族中复现
```

## 二、可直接发给执行 agent 的提示词

下面整段可以直接粘贴给当前执行 agent：

```text
请在现有 PhysInfoBench 项目中落实“折叠转换三层任务设计”，更新 TODO、协议、配置和决策记录，然后继续当前 Phase 1。项目根目录为：

/Users/yuan/Documents/ChatGPT/PLM

本次指令属于研究设计细化。请保留现有已完成任务和 Git 历史，不重开 P1.02/P1.03，不覆盖其完成记录。先核对 TODO、最新 handoff、git status、适用的 AGENTS.md、当前任务和最近提交。现有下一科研任务应为 P1.05；如启动时状态已经变化，以真实状态为准。

一、需要登记的设计决策

折叠转换主线采用三层设计：

1. T-FS-L1-PAIRED：
   严格双态阳性内部的机制诊断，包括多状态潜能、条件状态预测和残基区域定位。

2. T-FS-L2-PU-RANK：
   严格阳性在固定未标注背景中的 positive-unlabeled／排序分析。

3. T-FS-L3-MATCHED：
   严格阳性与操作性定义的匹配推定阴性之间的病例—对照比较。

T-FS-GLOBAL 保留为总任务 ID，不删除，以维持已有配置和报告引用兼容。T-FS-REGION 保留，并登记为第一层的残基级任务。

背景蛋白的 biological_target 必须保持空值。可以另设 pu_observed_label=0 表示“当前未被确认为阳性”，但禁止将该字段混入真实 target。推定阴性使用 putative_negative 和 PN-A/PN-B/PN-C 证据层名称。

二、先完成一次治理变更

1. 在 logs/decisions.md 记录三层设计的科学依据、范围、影响文件、待实现部分和确认节点。
2. 更新 configs/claims.yaml：
   - 保留 T-FS-GLOBAL；
   - 增加 T-FS-L1-PAIRED、T-FS-L2-PU-RANK、T-FS-L3-MATCHED；
   - 更新 claim→task 映射；
   - OD-NEGATIVE-FS 改为 direction_approved / implementation_pending；
   - P1.12 和 G1 后才能标记 resolved。
3. 更新 configs/sample_schema.yaml，加入：
   - label_epistemic_status；
   - biological_target nullable；
   - pu_observed_label；
   - positive_evidence_tier；
   - negative_evidence_tier；
   - background_source_version；
   - literature_search_query/date；
   - structure_count；
   - independent_study_count；
   - matched_set_id。
4. 更新 configs/evaluation_protocol.yaml：
   - 第一层：配对状态、条件贡献和残基区域指标；
   - 第二层：recall@k、enrichment@k、positive rank percentile、family-stratified 和 leave-family-out enrichment；
   - 第三层：AUROC、AUPRC、balanced accuracy、MCC、per-class precision/recall；
   - 第三层概率校准保持 disabled，除非以后预注册总体发生率校正；
   - observed-positive vs unlabeled 的 AP/AUROC 必须明确其观测标签语义。
5. 更新 configs/split_protocol.yaml：
   - 历史曝光不能通过重新命名恢复未见状态；
   - matched_set_id 不可跨集合；
   - strict positive 的同源和结构近邻不可跨集合；
   - 未见家族需要独立家族审计。
6. 更新 TODO.md：
   - 扩充 P1.09 的 strict positive 审计；
   - 在 Phase 1 的 P1.09 后增加 P1.10、P1.11、P1.12；
   - 更新 G1 进入条件；
   - 更新 P2.01、P2.02、P2.06、P3、P5 对三层数据的输入和验收标准。
7. 不把 P1.10–P1.12 标记为已开始或已完成。
8. 派独立 subagent 审核此次协议和 TODO 修改。
9. 修正后以独立计划提交完成治理变更，建议标题：
   PLAN register three-layer fold-switch workflow

三、P1.09 新增要求

P1.09 对现有 strict positive 做逐对审计，使用：

- Python + pandas/polars；
- Biopython；
- gemmi；
- 现有 fold_switch_global.tsv、fold_switch_regions.tsv；
- 原始证据表和旧项目只读来源。

输出：

reports/fs_three_layer/strict_positive_audit.tsv
reports/fs_three_layer/strict_positive_qc.json
reports/fs_three_layer/strict_positive_feasibility.md

逐对核对 UniProt、sequence hash、构建体、突变、PDB chain、两种状态、条件、论文证据、区域标签和 valid mask。处理 porter_8 冲突，抽查依赖旧审计的8个 strict positive。按独立蛋白、配对组和家族候选分别计数。

四、新增 P1.10

标题：建立折叠转换未标注背景。

建立两个冻结背景：

A. sequence universe：
优先使用固定 release 的 UniProtKB/Swiss-Prot reviewed，服务于纯序列 PU／排序。

B. structure-observed universe：
使用 RCSB PDB 实验结构 + PDBe/SIFTS 映射，服务于结构感知模型和推定阴性候选。

具体工具：

- UniProt REST API 或官方 release；
- RCSB Search/Data API；
- PDBe/SIFTS；
- requests/httpx；
- requests-cache；
- Biopython；
- gemmi；
- duckdb/polars/pyarrow；
- SHA-256。

实现：

scripts/build_fs_unlabeled_background.py
configs/fold_switch_background.yaml

输出：

data/curated/fold_switch_unlabeled_sequence.tsv
data/curated/fold_switch_unlabeled_structure.tsv
data/manifests/fold_switch_background_sources.tsv
reports/tasks/P1.10.md
logs/reviews/P1.10.md

强制断言：

- strict positive 与背景交集=0；
- extension 与背景交集按协议处理；
- biological_target=0 的行数=0；
- unlabeled 的 biological_target 为空；
- pu_observed_label=0；
- exact sequence duplicate 可追踪；
- 每行有 source release 和校验值；
- 非法序列、异常长度和映射失败有排除记录。

完成独立审核后单独提交 P1.10。

五、新增 P1.11

标题：建立折叠转换推定阴性候选池。

从 structure-observed universe 中筛选候选。使用：

- RCSB Search/Data API；
- PDBe/SIFTS；
- Europe PMC API；
- Crossref；
- CATH；
- SCOPe；
- Foldseek；
- gemmi；
- pandas/polars；
- requests-cache。

为每个候选保存：

- UniProt、sequence hash；
- PDB/独立 DOI/独立研究数量；
- 实验方法和结构覆盖；
- 配体、组装和实验条件覆盖；
- 家族和结构簇；
- 检索词、检索式、检索日期；
- 文献命中和人工审核状态；
- 对应 strict positive；
- 排除原因。

定义：

PN-A：多项独立实验/条件覆盖，系统检索未发现转换证据；
PN-B：多个结构记录，实验覆盖有限；
PN-C：只完成长度、家族或结构类型匹配，研究覆盖较弱。

PN等级表示证据覆盖程度。候选阶段 biological_target 继续为空。

实现：

scripts/build_fs_putative_negative_candidates.py
configs/fold_switch_putative_negative.yaml

输出：

data/curated/fold_switch_putative_negative_candidates.tsv
data/evidence/fold_switch_negative_literature_queries.tsv
data/evidence/fold_switch_negative_literature_hits.tsv
reports/tasks/P1.11.md
logs/reviews/P1.11.md

检查：

- 同一研究的多个 PDB 不增加独立研究数；
- domain swapping、局部开合、配体诱导变化和替代折叠分列；
- 命中明确转换证据的候选退出阴性池；
- 文献无命中和完成全文审核分列；
- 每个候选保留查询与来源；
- strict positive 同蛋白/同序列候选自动排除。

完成独立审核后单独提交 P1.11。

六、新增 P1.12

标题：冻结折叠转换三层任务合同与 G1 可行性。

生成：

configs/fold_switch_three_layer_protocol.yaml
reports/fs_three_layer/feasibility_matrix.tsv
reports/fs_three_layer/G1_fold_switch_decision.md
reports/tasks/P1.12.md
logs/reviews/P1.12.md

协议至少定义：

- 每层 population、unit、input、target/observed label、mask；
- 每层主/辅指标；
- strict/extension/unlabeled/PN-A/PN-B/PN-C 的进入条件；
- 可支持的结论；
- 不足时的降级规则；
- P2.01 需要执行的分组与匹配；
- P2.02 的划分约束；
- P2.06 的轻量混杂基线。

可行性表必须报告：

- strict positive 独立蛋白/组/家族候选数；
- extension 各层数量；
- fine/usable 区域覆盖；
- sequence/structure background 数量；
- PN-A/B/C 数量；
- 每个 strict positive 的候选对照数；
- 历史曝光；
- 可以支持三集合划分的任务；
- 只能做描述性分析的任务。

此任务审核通过并提交后，准备 G1 材料；没有用户明确确认不得进入 Phase 2。

七、更新 P2.01

P2.01 使用：

- MMseqs2/GraphPart：序列相似关系；
- Foldseek/US-align：结构近邻；
- networkx：不可拆分连通组；
- scipy.optimize.linear_sum_assignment 或等价最小成本匹配；
- pandas/numpy/statsmodels：匹配平衡。

执行：

1. 建立 protein/pair/sequence/homology/structure/matched-set 边表。
2. 对 strict positive 和 PN 候选生成1:1、1:3、1:5匹配可行性。
3. 匹配变量包括：
   - family/structure group；
   - sequence length；
   - structure count；
   - independent study count；
   - structure coverage；
   - method/resolution；
   - oligomeric state。
4. 匹配不使用任何模型输出。
5. 输出匹配成功率、无法匹配病例、标准化均值差、家族覆盖和PN等级。
6. 在看到模型结果前冻结主匹配版本。
7. matched_set_id 不可跨 split。

输出：

data/splits/fs_group_edges.tsv
data/curated/fold_switch_matched_controls.tsv
reports/fs_three_layer/matching_balance.tsv
reports/fs_three_layer/matching_balance.md

八、更新 P2.02

为三层分别生成数据清单：

- 第一层：strict/extension 配对分析；
- 第二层：strict positive + unlabeled background；
- 第三层：strict positive + matched putative controls。

组级分配；同蛋白、同源组、结构近邻组和 matched set 不可拆。历史用于方法选择的数据保留曝光状态。若某任务无法在 confirmation/final holdout 中同时保持足够病例和独立组，报告不可行并提交讨论。

九、更新 P2.06

在PLM表示之前实现轻量偏差基线：

- sequence length；
- amino-acid composition；
- low complexity；
- structure count；
- DOI/study count；
- experimental method；
- resolution；
- ligand/assembly count；
- family/structure cluster。

使用 scikit-learn、statsmodels、SciPy bootstrap。

第二层运行随机、长度、研究覆盖排序和 bagging-PU 基线。
第三层运行匹配变量逻辑回归和 matched-set 分层分析。
任何 PLM 增益都在共同样本上与这些基线比较。

十、执行纪律

- 先完成治理变更，再回到真实的下一任务；当前预计为 P1.05。
- P1.05–P1.08 可继续，不等待 P1.10。
- P1.09 之后按 P1.10→P1.11→P1.12 串行闭环。
- 每项必须：真实时间戳→执行报告→独立 subagent 审核→修正复审→更新 TODO→单独 Git commit。
- 不下载或运行 PLM 权重；模型准入仍归 P2.04。
- 不生成正式数据划分；阈值冻结与划分仍归 G1/P2.01/P2.02。
- 不覆盖旧项目文件，$B 保持只读。
- 不把无文献命中解释成生物学阴性。
- 不把未标注背景写入 target=0。
- 不把病例—对照类别比例解释为总体发生率。
- 不因样本不足拆开同源组、匹配组或状态配对组。
- 所有来源、查询、API响应、配置和输出保存版本与校验值。

请先汇报你核对到的真实状态、计划修改文件和治理变更范围，然后直接实施 TODO/协议更新、独立审核并提交。完成计划提交后，继续当前应执行的最小科研任务。
```

这套 TODO 把三层设计落到了四类具体工具链上：

- 数据来源：UniProt、RCSB、SIFTS、Europe PMC、Crossref；
- 生物信息处理：Biopython、gemmi、MMseqs2、GraphPart、Foldseek、US-align；
- 数据与匹配：pandas/polars、DuckDB、SciPy、networkx、statsmodels；
- 可复现与治理：YAML、JSON Schema、SHA-256、独立审核和逐项 Git 提交。

最关键的执行顺序是：**先登记设计变化，继续 P1.05–P1.08，P1.09 审计严格阳性，再建立未标注背景与推定阴性候选，G1 确认后才进行匹配、划分和模型实验。**