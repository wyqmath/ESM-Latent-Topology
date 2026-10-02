# R5.04 P5.08 真实 accession 连接修复

记录时间：2026-10-01 21:44 Asia/Shanghai。修复脚本首次落盘：2026-10-01 21:42（文件创建时间）；执行和回归检查在随后完成。前面的发现过程见 claims_scope_audit.md。原审计过程中没有单独记录本任务开始分钟，不使用当前时刻回填开始时间。

更新：2026-10-01 22:17 Asia/Shanghai。独立审核发现额外H11/H33错配后，完成同层H33纠错、全缓存核验与精确分数归档。

## 根因与实现

旧脚本把多链候选表压为entry唯一键，随后用 `_struct_ref.db_code` 下划线前缀猜accession。45/105条保留链因此取到其他asym的候选行；3条的db_code也不同。56条的db_code前缀并非accession。此轮修改 `scripts/p508_fix_exclusions.py`，通过 `(entry,label_asym)` 核对原候选行，再通过本地CIF的 `_struct_asym.entity_id` 找到 `_struct_ref.pdbx_db_accession`。链ID保持原大小写；同一entity的全部UniProt映射都登记，任一与DisProt重叠即排除。未知、无映射、无效或多accession情况明确进入隔离名单。

旧脚本此前重复原地运行后，`data/interim/p508_final_set.json` 的三类dropped记录已为空。本轮后来从集群取回历史 `p508_mmseqs_cluster.tsv`，按全部239条已选链重建同源排除与序列去重：239→排除108→131→去重26→105，重建105名单与历史评价105名单差集为空。239条逐链真实accession审计发现36条与DisProt重叠，全部已经被同源步骤排除；19条身份未解决。此处恢复的是从保存来源派生的排除表，没有回填当时的时间线。评价范围始终限定在历史105条之内。

修复脚本改为correction-only CLI，默认写入 `results/repairs/20261001/p508/`，拒绝覆盖非空目录；原输入集、原结果和原运行锁保持原样。原脚本已由主代理保存在baseline，可用于历史代码查验。集群历史P5.08 runner没有重跑。

独立审核还发现原runner把dev缓存的`resid_layers[0]`作为H33。实际dev元数据是`resid_11_23_33`，slot0为H11；外部元数据是`resid_33`，slot0为H33。原105结果是H11开发拟合→H33外部输入的跨层错配。第一次99链复现作业814881与原105对应逐蛋白行全部一致，已经单列为历史混层诊断，不能支撑原定H33读取器主张。

修改原 `scripts/p508_cluster_eval.py`：按元数据和manifest严格定位H33，开发使用slot2、外部使用slot0；目标层缺失即失败，无降级默认索引。新增固定模型/checkpoint校验、dev缓存残基索引与冻结ridx表顺序核对，新增输出目录非空守卫；默认产物写新correction路径，原105目录不再作为默认输出。整个开发2279条和外部99条缓存已独立通过上述检查，证据为`h33_cache_preflight.json`。当前代码新增的额外守卫在作业814885之后进行了独立全量核验，未把该后置核验写成当时作业的执行过程。

## 身份核验结果

- 105条中100条取得真实UniProt accession；其中99条为唯一accession，1条多accession。
- 与项目DisProt accession交集为0；本轮未发现实际accession重叠污染。该结果不替代预训练曝光或其他隔离审计。
- 5条无UniProt映射：2Q2C_A、2Q86_A、5ELJ_A、5ELU_A、5FEU_A；全部隔离。
- 5EP9_C存在多accession，未任意选择一个，进入隔离名单。
- 99条身份已核验的链组成事后补充子集；原105条历史结果完整保留。

逐链表在 `accession_audit.tsv`，包含entry、label_asym、entity_id、真实accession、旧错用字段和处置。`correction_set.json` 记录99条保留和6条隔离。

## 历史混层复现与H33同层纠错

先从原四位小数逐蛋白表取得99条身份核验子集；随后从已有缓存按固定dev2279、C=.01、balanced/liblinear、seed2026复现历史拟合，并执行预定H33→H33纠错。冻结参数没有重新选择，没有提取新表示，没有新增测试链或调整0.5阈值。原拟合系数未归档，因此这里明确称固定参数拟合复现，而非“零重拟合”。

| 指标 | 原105保存表（混层历史） | 99混层复现（814881） | 99 H33同层纠错（814885） |
|---|---:|---:|---:|
| 残基数 | 23,977 | 22,646 | 22,646 |
| 逐蛋白AUPRC均值 | 0.4383105 | 0.4368547 | 0.4760740 |
| 逐蛋白基率均值 | 0.2139238 | 0.2153465 | 0.2153465 |
| AUPRC减基率均值 | 0.2243867 | 0.2215083 | 0.2607275 |
| balanced accuracy均值 | 0.5243705 | 0.5229278 | 0.5033038 |
| MCC均值 | 0.0726971 | 0.0693423 | 0.0154488 |
| 池化AUPRC（次要） | 历史0.3489 | 0.3472567 | 0.3600609 |
| 池化基率 | 历史0.2112 | 0.2129294 | 0.2129294 |
| 100次固定预测蛋内置换mean/max | 历史0.2116/0.2171 | 0.2120260/0.2201027 | 0.2108549/0.2159368 |

原105条未发生accession重叠剔除；99条变化来自5条缺映射和1条多映射的机械身份规则。子集规则只使用CIF身份，不参考预测分数。子集仍属于事后补充分析。

原105结果未保存逐残基预测。本轮分别在`exact_metrics/`（历史混层诊断）和`h33_exact_metrics/`（同层纠错）保存22646行17位有效数字预测、0基序列索引、原CIF label/auth坐标与标签；固定读取器系数也已保存。各目录包含精确逐蛋白指标、池化及100次同语义置换结果。99结果使用99分母计算，未照搬105池化数字。实际执行源码副本、缓存校验值、软件和层审计一并归档；最新防混层守卫与实际执行源码通过单列全量核验连接，不混写版本。

建议进入当前报告的文本：

> 原105条结果存在H11拟合→H33评分的层错配，保留为历史诊断。逐链身份修复后，5条无映射、1条多映射隔离，99条核验链无DisProt accession重叠。按原定H33/C=.01执行同层纠错后，逐蛋白AUPRC均值0.476，高于基率0.215；0.5阈值下balanced accuracy为0.503、MCC为0.015。当前读取器提供了排序信号，固定阈值分类表现较弱。该结果是CA坐标未观测操作标签上的事后补充分析，与内在无序的对应关系尚未逐段验证。

## 校验与运行

环境：本地 `.venv_local/bin/python`，gemmi0.7.5。运行：

```sh
.venv_local/bin/python scripts/p508_fix_exclusions.py --output results/repairs/20261001/p508
.venv_local/bin/python -m unittest discover -s tests -p test_p508_accessions.py -v
```

3项回归测试通过：多链entry映射归属、db_code前缀拒绝与isoform归一、多accession/未知身份显式处置。脚本拒绝再次写入非空输出目录。结果目录另存 `historical_evaluated_set.json`、`historical_per_protein_105.tsv`，`summary.json` 记录原文件及105份CIF SHA256和软件。旧结果文件未修改。

新增4项层读取测试通过：H33在不同缓存布局中的显式选取、目标缺失、metadata/shape冲突，以及错误模型拒绝。全239来源哈希与逐链排除重建由 `scripts/audit_p508_candidate_cif.py` 复算；`scripts/check_p508_cache_layers.py` 全量核查2279+99缓存层身份与索引。

SLURM记录：CPU-only申请因QOSMinGRES拒绝；814879首次wrapper将索引误认为0基，验证阶段2秒失败，未拟合；索引按原extractor确认1基后，814881历史混层复现61秒完成；814885同层H33纠错完成。均为CPU执行，GPU资源仅为调度QOS要求预留。失败日志和成功日志全部保存，未删除半成品伪造首轮成功。

## R5.02 独立审核（claims_audit）

审核时间：2026-10-01 21:44 Asia/Shanghai。结论PASS（0 blocker/0 major）。代码保留抽中次数，完整绑定分量内每条链得到相同分量抽中权重；点估计仍为链权AUROC；两方法差值使用相同抽样。六项原回归测试通过。

独立数值核查采用显式重复行展开，再用正负样本成对比较计算AUROC（同分计半），没有复用weighted sklearn实现。覆盖full_set与事后子集、确认与保留、分量与链抽样、三种子，共24套输出。每套前20次抽样逐次复现，480次独立展开比较全部一致；每套全部2000条抽样的有效/退化计数和95%分位数与JSON一致。证据文件 `results/repairs/20261001/claims/bootstrap_second_review.json`。

正确重算的区间只反映既有读取器/数据下的抽样不确定性，无法消除历史开发使用。事后子集仍按描述性敏感性报告。该PASS审核的是统计实现和数值，不升级科学证据等级。


## 最终版本、身份与归档核查

核查时间：2026-10-01 22:36 Asia/Shanghai。`results/repairs/20261001/p508/final_provenance_consistency_review.json` 为本节逐项证据。原105身份审计的109个输入校验、全239候选的244个输入校验全部一致；原105逐链表、池化JSON和final_set均保持原字节。`configs/source_manifest.yaml` 指定proposal的现存文件SHA256与登记一致。

814885实际执行源码与当前防错源码分别归档，版本对应如下：

| 作用 | p508_cluster_eval.py SHA256 | repair_p508_cached_metrics.py SHA256 |
|---|---|---|
| 814885实际执行副本，位于h33_exact_metrics/ | a5d575958707b98482611e3775c2e3a64dbb44f4f101029cb3a9bb66ce0110d7 | 7671599d5a0bd25b1b7d40acaea19b8d89ef20911c68545c2f2c2ac207f50bcf |
| 当前scripts/源码 | 13c5db036237663d2046e1ae91f7de5d1a9a95b68cbe16f31316c784bccd6610 | d7ff88226e88ffb932a9e150df96aa77012d7abe9126a44bac6e1b80f59d53ea |

实际执行版本已显式指定dev H33槽2及external H33槽0。当前版本在此基础上增加冻结模型/revision断言、dev残基索引顺序断言、CIF序列0基索引顺序断言，并强制使用99链纠错身份集、缓存存在与输出目录守卫；不再通过旧105 final_set绕过身份修复或重新提取外部缓存。这些新增守卫在作业结束后全量核验2279个dev缓存及99个external缓存，结果PASS。该核验时间与作业运行时间分列，不声称新增守卫曾在814885运行期间执行。

239与105两张逐链审计表分别提供真实entry/asym/entity/accession及冻结CIF/序列哈希证据；239候选依历史MMseqs表可重建108条同源排除、26条精确序列排除，最终105集合差为0。历史final_set的原始剔除时间线未恢复，重建表明确标为事后推导。`summary.json`保留第一阶段由四位小数历史表形成的身份审计快照，其当时“原始预测本地不可用/239未重建”限制已由后续归档补齐；当前数字来源以`h33_exact_metrics/metrics_exact.json`为准。

读取器头`h33_exact_metrics/frozen_reader_coefficients.npz`现可本地直接读取，大小11022字节，SHA256为`3d010c83e2115c2f9e7a2d7c72ec88f6094ff4349dbf6273c99783dcd236a0a8`，保存coef(1,1280)、intercept(1)、classes(2)。逐残基分数22646行本地压缩保存；集群表示缓存仍位于`/lenovofs1/home/jyma/PLM_benchmark/p303_resplit/proj/data/interim/p303/emb_disorder_dev/`及`data/interim/p508_emb/`，每个实际使用NPZ的SHA256在replay_provenance中登记。开发/外部manifest及两份残基索引表补存于`replay_input_tables/`，均为既有输入元数据的读取副本，未新增评价样本或模型提取。维护时须同时保留此处系数、预测、源代码副本、缓存哈希和远程缓存；缓存缺失应显式失败，不能静默重算并冒用原运行身份。

独立审核`reports/repairs/20261001/p508_final_independent_review.json`为PASS：由已存系数在全部99个真实H33缓存直接计算sigmoid，与已存预测最大差4.44e-16；逐链指标、池化指标及100次固定预测置换全部一致。`reports/tasks/P5.08.md`与`reports/claim_evidence_matrix.md`使用同一99/22646、同层H33数字，并均保留固定参数重建、事后补充、CA未观测操作标签的结论范围。本核查只确认来源和报告一致性，G5/G6状态由主治理记录单独裁决。


归档补充核查时间：2026-10-01 22:43 Asia/Shanghai。历史混层与H33同层两份`frozen_reader_coefficients.json`现分别保存在`exact_metrics/`和`h33_exact_metrics/`，作为可纳入Git的数字数组副本。JSON登记的source_npz_sha256均与对应NPZ一致，coef、intercept、classes的shape、dtype与逐元素数值精确相同，未重新拟合。JSON复核SHA256：

- `exact_metrics/frozen_reader_coefficients.json`：`47c0b3a2bed7e4c22467572d31ba45821cd6d53fbf97c2ab6ff2eb84fd80986c`。
- `h33_exact_metrics/frozen_reader_coefficients.json`：`2113e8a76f0982ceed42b83e1bce8653e92b98ef98593facc56a8ca881e3cf40`。
